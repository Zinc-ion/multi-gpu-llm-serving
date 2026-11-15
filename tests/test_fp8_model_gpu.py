import gc

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from fp8_helpers import tolerances
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.sequence import Sequence
from myvllm.layers.attention import Attention
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import reset_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def tiny_checkpoint(path):
    path.mkdir()
    model_config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=32, tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20261116)
        Qwen3ForCausalLM(model_config).save_pretrained(path)
    vocab = {"[UNK]": 0, "hello": 1, "[EOS]": 63}
    vocab.update({f"token{i}": i for i in range(2, 63)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(path)
    return {
        "model_name_or_path": str(path), "model_type": "Qwen3-0.6B",
        "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
        "world_size": 1, "block_size": 4,
        "vocab_size": 64, "hidden_size": 32, "num_heads": 2, "head_dim": 16,
        # Attention.forward applies the sole 1/sqrt(head_dim) factor.
        "scale": 1.0, "num_kv_heads": 1, "rms_norm_epsilon": model_config.rms_norm_eps,
        "qkv_bias": False, "base": model_config.rope_theta, "max_position": 32,
        "intermediate_size": 64, "ffn_bias": False, "num_layers": 2,
        "tie_word_embeddings": False, "max_num_sequences": 7,
        "max_num_batched_tokens": 32, "max_model_length": 32,
        "max_cached_blocks": 32, "gpu_memory_utilization": .05,
        "eos": [63], "seed": 0, "kv_cache_dtype": "float8_e4m3fn",
    }


def exercise_engine(config):
    previous_dtype = torch.get_default_dtype()
    engine = None
    try:
        engine = LLMEngine(config)
        runner = engine.model_runner
        cache, scales = runner.kv_cache, runner.kv_cache_scales
        assert cache.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.float32
        assert cache.shape == (2, 2, 32, 4, 1, 16)
        assert scales.shape == (2, 2, 32, 4, 1)
        assert next(runner.model.parameters()).dtype == getattr(torch, config["dtype"])
        pointers = (cache.data_ptr(), scales.data_ptr())
        for index, module in enumerate(m for m in runner.model.modules() if isinstance(m, Attention)):
            assert module.k_cache.data_ptr() == cache[0, index].data_ptr()
            assert module.v_cache.data_ptr() == cache[1, index].data_ptr()
            assert module.k_scale_cache.data_ptr() == scales[0, index].data_ptr()
            assert module.v_scale_cache.data_ptr() == scales[1, index].data_ptr()
        assert torch.count_nonzero(cache.view(torch.uint8)) == 0
        assert torch.all(scales == 1)
        cache[:, :, -1].view(torch.uint8).fill_(0x55)
        scales[:, :, -1].fill_(3.25)

        seqs = [
            Sequence(list(range(1, length + 1)), 4, SamplingParams(
                temperature=0, max_tokens=output, ignore_eos=True,
            )) for length, output in ((5, 2), (7, 4), (9, 6))
        ]
        for seq in seqs:
            engine.scheduler.add_sequence(seq)
        logits, callbacks, decode_batches = [], [], []
        hook = runner.model.lm_head.register_forward_hook(
            lambda module, args, output: logits.extend(output.detach().float().cpu().clone())
        )
        try:
            while not engine.scheduler.is_finished():
                _, _, is_prefill = engine.step(lambda seq, token: callbacks.append((seq.seq_id, token)))
                if not is_prefill and not config["enforce_eager"]:
                    variables = runner.graph_vars
                    active = int((variables["context_lens"] > 0).sum().item())
                    decode_batches.append(active)
                    assert torch.all(variables["slot_mapping"][active:] == -1)
                    assert torch.all(variables["context_lens"][active:] == 0)
        finally:
            hook.remove()
        assert len(logits) == len(callbacks) == 12
        result = {
            "tokens": [seq.completion_token_ids for seq in seqs],
            "logits": [torch.stack([logit for logit, (seq_id, _) in zip(logits, callbacks)
                                     if seq_id == seq.seq_id]) for seq in seqs],
        }
        assert all(torch.isfinite(logit).all() for logit in result["logits"])
        assert not engine.scheduler.block_manager.used_block_ids
        assert torch.count_nonzero(cache[:, :, :-1].view(torch.uint8)) > 0
        assert torch.any(scales[:, :, :-1] != 1)
        assert torch.all(cache[:, :, -1].view(torch.uint8) == 0x55)
        assert torch.all(scales[:, :, -1] == 3.25)
        assert pointers == (cache.data_ptr(), scales.data_ptr())
        if not config["enforce_eager"] and not config["enable_chunked_prefill"]:
            assert {1, 2, 3}.issubset(decode_batches)

        # Exercise a first-token terminal request and block reuse.
        terminal = Sequence([1, 2, 3], 4, SamplingParams(temperature=0, max_tokens=1))
        engine.scheduler.add_sequence(terminal)
        while not engine.scheduler.is_finished():
            engine.step()
        assert terminal.is_finished and terminal.num_completion_tokens == 1
        assert not engine.scheduler.block_manager.used_block_ids
        if config["enable_chunked_prefill"]:
            cancelled = Sequence(list(range(1, 13)), 4, SamplingParams(temperature=0, max_tokens=2))
            engine.scheduler.add_sequence(cancelled)
            engine.step()
            assert cancelled.num_cached_tokens == 5
            assert engine.cancel(cancelled.seq_id)
            assert not engine.scheduler.block_manager.used_block_ids
        return result
    finally:
        if engine is not None:
            engine.exit()
        reset_context()
        torch.set_default_device("cpu")
        torch.set_default_dtype(previous_dtype)
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_fp8_model_whole_chunked_eager_and_graph(tmp_path, dtype):
    base = tiny_checkpoint(tmp_path / "checkpoint")
    baseline = None
    for chunked, eager in ((False, True), (False, False), (True, True), (True, False)):
        config = dict(base, dtype=dtype, enforce_eager=eager, enable_chunked_prefill=chunked,
                      max_num_batched_tokens=5 if chunked else 32)
        result = exercise_engine(config)
        if baseline is None:
            baseline = result
        else:
            assert result["tokens"] == baseline["tokens"]
            for actual, expected in zip(result["logits"], baseline["logits"]):
                print(f"dtype={dtype} chunked={chunked} eager={eager} "
                      f"max_logit_abs_error={(actual - expected).abs().max().item():.8g}")
                torch.testing.assert_close(actual, expected, **tolerances(getattr(torch, dtype)))
