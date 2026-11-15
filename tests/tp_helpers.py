"""Small real checkpoints and distributed model execution for TP tests."""
import os
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from safetensors import safe_open
from transformers import LlamaConfig, Qwen3Config
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from myvllm.models.llama import LlamaForCausalLM
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.utils import reset_context, set_context
from myvllm.utils.loader import load_weights_from_checkpoint


# Fixed before executing the comparisons; these are tiny-model CPU/GPU guards,
# not the pending acceptance threshold for Qwen3-14B on the target server.
TOLERANCES = {
    "float32": {"rtol": 2e-4, "atol": 2e-5},
    "bfloat16": {"rtol": 5e-2, "atol": 2e-2},
}
PROMPT = [1, 9, 4, 6, 12]


def create_checkpoint(path, family, tied, bias):
    torch.manual_seed(12345)
    common = dict(vocab_size=67, hidden_size=64, head_dim=16,
                  num_attention_heads=4, num_key_value_heads=2,
                  intermediate_size=128, num_hidden_layers=2,
                  max_position_embeddings=64, rms_norm_eps=1e-5,
                  tie_word_embeddings=tied, attention_bias=bias,
                  attention_dropout=0.0)
    if family == "qwen3":
        config = Qwen3Config(**common, rope_theta=10000)
        model = HFQwen3ForCausalLM(config)
    else:
        config = LlamaConfig(
            **common, rope_theta=500000, mlp_bias=bias,
            rope_scaling={"rope_type": "llama3", "factor": 32.0,
                          "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                          "original_max_position_embeddings": 8192},
        )
        model = HFLlamaForCausalLM(config)
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            torch.nn.init.normal_(param, std=0.05)
    model.eval().save_pretrained(path, safe_serialization=True, max_shard_size="40KB")
    return model


def checkpoint_names(path):
    names = set()
    for shard in sorted(Path(path).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            names.update(weights.keys())
    return names


def cpu_flash(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, max_seq_len):
    assert q.shape[1:] == (num_heads, head_dim)
    assert k.shape[1:] == (num_kv_heads, head_dim)
    outputs = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        output = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1).unsqueeze(0),
            k[start:end].transpose(0, 1).unsqueeze(0),
            v[start:end].transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=scale, enable_gqa=True,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def make_model(family, tied, bias):
    common = dict(vocab_size=67, hidden_size=64, head_dim=16, num_kv_heads=2,
                  intermediate_size=128, num_layers=2, tie_word_embeddings=tied,
                  rms_norm_epsilon=1e-5, block_size=16)
    if family == "qwen3":
        return Qwen3ForCausalLM(**common, num_heads=4, max_position=64,
                               qkv_bias=bias, ffn_bias=False)
    return LlamaForCausalLM(**common, num_qo_heads=4, max_position_embeddings=64,
                            has_attn_bias=bias, ffn_bias=bias)


@contextmanager
def measure_collectives(device):
    records = {"all_reduce": [], "gather": []}
    originals = {name: getattr(dist, name) for name in records}
    for name, original in originals.items():
        def timed(*args, _name=name, _original=original, **kwargs):
            if device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                result = _original(*args, **kwargs)
                end.record()
                records[_name].append((start, end))
            else:
                start = time.perf_counter()
                result = _original(*args, **kwargs)
                records[_name].append((time.perf_counter() - start) * 1000)
            return result
        setattr(dist, name, timed)
    try:
        yield records
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


def model_worker(rank, world_size, rendezvous, checkpoint, out_dir, family, tied, bias, dtype_name, device_type):
    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        import myvllm.layers.attention as attention
        attention.flash_attention_prefill = cpu_flash
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                            init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        dtype = getattr(torch, dtype_name)
        model = make_model(family, tied, bias).to(device=device, dtype=dtype).eval()
        loaded = load_weights_from_checkpoint(model, checkpoint)
        missing = set(dict(model.named_parameters())) - loaded
        missing_checkpoint = checkpoint_names(checkpoint) - loaded
        assert not missing, f"unloaded model parameters: {sorted(missing)}"
        assert not missing_checkpoint, f"unloaded checkpoint tensors: {sorted(missing_checkpoint)}"
        tokens = list(PROMPT)
        logits = []
        with measure_collectives(device) as timings, torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            for _ in range(3):
                ids = torch.tensor(tokens, device=device)
                set_context(is_prefill=True,
                            cu_seqlens_q=torch.tensor([0, len(tokens)], dtype=torch.int32, device=device),
                            max_seqlen_q=len(tokens), positions=torch.arange(len(tokens), device=device))
                output = model.compute_logits(model(ids))
                next_token = torch.zeros(1, dtype=torch.long, device=device)
                if rank == 0:
                    logits.append(output.float().cpu())
                    next_token[0] = output.argmax(dim=-1)[0]
                dist.broadcast(next_token, src=0)
                tokens.append(next_token.item())
                reset_context()
        if device_type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            elapsed = {name: sum(start.elapsed_time(end) for start, end in values)
                       for name, values in timings.items()}
        else:
            peak_allocated_bytes = None
            elapsed = {name: sum(values) for name, values in timings.items()}
        measurement = {
            "rank": rank, "world_size": world_size, "device": str(device),
            "timing_kind": "cuda_event_stream_elapsed" if device_type == "cuda" else "gloo_wall_diagnostic",
            "all_reduce_ms": elapsed["all_reduce"], "all_reduce_calls": len(timings["all_reduce"]),
            "gather_ms": elapsed["gather"], "gather_calls": len(timings["gather"]),
            "peak_allocated_bytes": peak_allocated_bytes,
        }
        print("TP measurement: " + json.dumps(measurement, sort_keys=True), flush=True)
        torch.save({"tokens": tokens, "logits": logits, "loaded": loaded,
                    "model_names": set(dict(model.named_parameters())),
                    "checkpoint_names": checkpoint_names(checkpoint),
                    "rank": rank, "device": str(device),
                    "measurement": measurement,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                   Path(out_dir) / f"result-{world_size}-{rank}.pt")
    finally:
        reset_context()
        dist.destroy_process_group()


def run_model(path, family, tied, bias, dtype_name, world_size, device_type="cpu"):
    rendezvous = path / f"rendezvous-{world_size}"
    mp.spawn(model_worker, nprocs=world_size,
             args=(world_size, str(rendezvous), str(path / "checkpoint"), str(path),
                   family, tied, bias, dtype_name, device_type), join=True)
    return [torch.load(path / f"result-{world_size}-{rank}.pt", weights_only=False)
            for rank in range(world_size)]


def reference_run(model, dtype_name):
    model = model.to(dtype=getattr(torch, dtype_name)).eval()
    tokens = list(PROMPT)
    logits = []
    with torch.inference_mode():
        for _ in range(3):
            output = model(torch.tensor([tokens]), use_cache=False).logits[:, -1, :]
            logits.append(output.float())
            tokens.append(output.argmax(dim=-1).item())
    return tokens, logits
