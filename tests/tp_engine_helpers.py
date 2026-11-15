"""Tiny real checkpoints and isolated serving-engine execution for TP tests."""
import json
import os
import signal
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM

import myvllm.engine.llm_engine as engine_module
import myvllm.layers.attention as attention
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import get_context
from tp_helpers import PROMPT, TOLERANCES, checkpoint_names, create_checkpoint, measure_collectives


def create_engine_checkpoint(path, dtype_name="float32"):
    reference = create_checkpoint(path, "qwen3", False, False)
    vocab = {"[UNK]": 0, **{f"t{i}": i for i in range(1, 66)}, "[EOS]": 66}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(path)
    hf = reference.config
    config = {
        "model_name_or_path": str(path), "model_type": "Qwen3-0.6B",
        "world_size": 1, "block_size": 2, "enforce_eager": True,
        "vocab_size": hf.vocab_size, "hidden_size": hf.hidden_size,
        "num_heads": hf.num_attention_heads, "num_kv_heads": hf.num_key_value_heads,
        "head_dim": hf.head_dim,
        # Attention applies head_dim**-0.5 itself.
        "scale": 1.0, "rms_norm_epsilon": hf.rms_norm_eps,
        "qkv_bias": hf.attention_bias, "base": hf.rope_theta,
        "max_position": hf.max_position_embeddings,
        "intermediate_size": hf.intermediate_size, "ffn_bias": False,
        "num_layers": hf.num_hidden_layers, "tie_word_embeddings": hf.tie_word_embeddings,
        "max_num_sequences": 1, "max_num_batched_tokens": 16,
        "max_model_length": 16, "max_cached_blocks": 16,
        "gpu_memory_utilization": 0.1, "dtype": dtype_name, "eos": [66],
    }
    return reference, config


def engine_reference_run(checkpoint, dtype_name):
    # Native HF dtype loading retains FP32 RoPE inv_freq. Casting an already
    # initialized FP32 model to BF16 would also quantize this nonpersistent buffer.
    model = Qwen3ForCausalLM.from_pretrained(checkpoint, dtype=getattr(torch, dtype_name)).eval()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    tokens, logits = list(PROMPT), []
    with torch.inference_mode():
        for _ in range(3):
            output = model(torch.tensor([tokens]), use_cache=False).logits[:, -1, :]
            logits.append(output.float())
            tokens.append(output.argmax(dim=-1).item())
    torch.save({
        "tokens": tokens, "logits": logits, "dtype": dtype_name, "device": "cpu",
        "tolerances": TOLERANCES[dtype_name],
        "buffer_dtypes": {name: str(value.dtype) for name, value in model.named_buffers()},
    }, checkpoint.parent / "hf-reference.pt")
    return tokens, logits


@contextmanager
def observe_runner(runner, output_dir):
    """Observe real serving calls after warmup; never replace kernel math."""
    device = torch.device("cuda", runner.rank)
    torch.cuda.reset_peak_memory_stats(device)
    data = {
        "rank": runner.rank, "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded": runner.loaded_params,
        "model_names": set(dict(runner.model.named_parameters())),
        "checkpoint_names": checkpoint_names(runner.config["model_name_or_path"]),
        "steps": [], "logits": [],
        "attention_calls": {"paged_prefill": 0, "paged_decode": 0, "cache_free_prefill": 0},
    }

    def observe_inputs(module, args):
        context = get_context()
        assert runner.kv_cache.numel() > 0 and context.block_tables is not None
        data["steps"].append({
            "is_prefill": context.is_prefill,
            "input_ids": args[0].tolist(),
            "positions": context.positions.tolist(),
            "slot_mapping": context.slot_mapping.tolist(),
            "block_tables": context.block_tables.tolist(),
            "context_lens": context.context_lens.tolist() if context.context_lens is not None else None,
        })

    def observe_logits(module, args, output):
        if runner.rank == 0:
            data["logits"].append(output.detach().float().cpu())

    input_hook = runner.model.register_forward_pre_hook(observe_inputs)
    logits_hook = runner.model.lm_head.register_forward_hook(observe_logits)
    functions = {
        "paged_prefill": "paged_attention_prefill",
        "paged_decode": "paged_attention_decode",
        "cache_free_prefill": "flash_attention_prefill",
    }
    try:
        with ExitStack() as stack:
            for name, attribute in functions.items():
                original = getattr(attention, attribute)

                def observed(*args, _name=name, _original=original, **kwargs):
                    data["attention_calls"][_name] += 1
                    return _original(*args, **kwargs)

                stack.enter_context(patch.object(attention, attribute, observed))
            timings = stack.enter_context(measure_collectives(device))
            yield data
    finally:
        input_hook.remove()
        logits_hook.remove()
    torch.cuda.synchronize(device)
    elapsed = {name: sum(start.elapsed_time(end) for start, end in values)
               for name, values in timings.items()}
    data["measurement"] = {
        "rank": runner.rank, "world_size": runner.world_size, "device": str(device),
        "timing_kind": "cuda_event_stream_elapsed",
        "all_reduce_ms": elapsed["all_reduce"], "all_reduce_calls": len(timings["all_reduce"]),
        "gather_ms": elapsed["gather"], "gather_calls": len(timings["gather"]),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_scope": "engine_ready_through_request_and_exit",
    }
    print("TP measurement: " + json.dumps(data["measurement"], sort_keys=True), flush=True)
    torch.save(data, Path(output_dir) / f"rank-{runner.rank}.pt")


def observed_worker(config, rank, event):
    runner = ModelRunner(config, rank, event)
    try:
        with observe_runner(runner, config["_tp_test_output"]):
            runner.loop()
    finally:
        runner.exit()


def engine_process(config, output_dir):
    torch.set_num_threads(1)
    engine = None
    try:
        # The rank worker still constructs a real ModelRunner and executes loop(),
        # including production shared-memory/pickle dispatch for every run.
        with patch.object(engine_module, "worker_process", observed_worker):
            engine = LLMEngine(config)
        with observe_runner(engine.model_runner, output_dir):
            seq = engine.add_prompt(
                " ".join(f"t{i}" for i in PROMPT),
                SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
            )
            assert seq.prompt_token_ids == PROMPT
            prefixes = []
            while not seq.is_finished:
                prefixes.append(list(seq.token_ids))
                engine.step()
            assert engine.scheduler.is_finished()
            assert not engine.scheduler.block_manager.used_block_ids
            result = {"tokens": list(seq.token_ids), "prefixes": prefixes}
            engine.exit()
            result["worker_exitcodes"] = [process.exitcode for process in engine.processes]
            assert not dist.is_initialized()
        torch.save(result, Path(output_dir) / "engine.pt")
    finally:
        if engine is not None:
            engine.exit()


def run_engine(path, config, world_size):
    output_dir = path / f"engine-tp{world_size}"
    output_dir.mkdir()
    settings = {**config, "world_size": world_size, "_tp_test_output": str(output_dir)}
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    log_path = output_dir / "engine.log"
    # A separate interpreter contains runner's process-global default dtype/device.
    # A private process group also permits cleanup of its TP children on timeout.
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()),
             "--config", str(config_path), "--output", str(output_dir)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            process.wait(timeout=180)
        finally:
            if process.poll() is None or process.returncode != 0:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            print(log_path.read_text(), flush=True)
    assert process.returncode == 0, f"engine exited {process.returncode}; see {log_path}"
    result = torch.load(output_dir / "engine.pt", map_location="cpu", weights_only=False)
    result["ranks"] = [torch.load(output_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
                       for rank in range(world_size)]
    return result


def assert_engine_matches(result, ref_tokens, ref_logits, dtype_name):
    assert result["tokens"] == ref_tokens
    assert result["prefixes"] == [ref_tokens[:len(PROMPT) + i] for i in range(3)]
    world_size = len(result["ranks"])
    assert result["worker_exitcodes"] == [0] * (world_size - 1)
    assert [rank["device"] for rank in result["ranks"]] == [f"cuda:{i}" for i in range(world_size)]
    for rank in result["ranks"]:
        assert rank["model_names"] <= rank["loaded"]
        assert rank["checkpoint_names"] <= rank["loaded"]
        steps = rank["steps"]
        assert [step["is_prefill"] for step in steps] == [True, False, False]
        assert [step["input_ids"] for step in steps] == [PROMPT, [ref_tokens[5]], [ref_tokens[6]]]
        assert [step["positions"] for step in steps] == [list(range(5)), [5], [6]]
        assert [step["slot_mapping"] for step in steps] == [list(range(5)), [5], [6]]
        assert [step["context_lens"] for step in steps] == [None, [6], [7]]
        assert [len(step["block_tables"][0]) for step in steps] == [3, 3, 4]
        assert rank["attention_calls"] == {
            "paged_prefill": 2, "paged_decode": 4, "cache_free_prefill": 0,
        }
        measurement = rank["measurement"]
        assert measurement["timing_kind"] == "cuda_event_stream_elapsed"
        assert measurement["peak_allocated_bytes"] > 0
        for name in ("all_reduce", "gather"):
            assert measurement[f"{name}_ms"] >= 0
            if world_size > 1:
                assert measurement[f"{name}_calls"] > 0
            else:
                assert measurement[f"{name}_calls"] == 0
    actual_logits = result["ranks"][0]["logits"]
    assert len(actual_logits) == len(ref_logits) == 3
    for actual, expected in zip(actual_logits, ref_logits):
        torch.testing.assert_close(actual, expected, **TOLERANCES[dtype_name])
    max_error = max((actual - expected).abs().max().item()
                    for actual, expected in zip(actual_logits, ref_logits))
    print(f"Actual engine TP{world_size} {dtype_name}: paged prefill + 2 decode, "
          f"HF max_abs={max_error:.8g}, greedy={result['tokens'][len(PROMPT):]}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    engine_process(json.loads(Path(args.config).read_text()), args.output)
