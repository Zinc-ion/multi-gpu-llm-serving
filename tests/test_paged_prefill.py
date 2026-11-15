import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import myvllm.layers.attention as attention
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.layers.sampler import SamplerLayer
from myvllm.models.llama import LlamaForCausalLM
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import reset_context


def reference_prefill(q, k_cache, v_cache, cu_q, cu_k, tables,
                      max_q, scale, heads, kv_heads, dim, block_size):
    outputs = []
    for i in range(len(cu_q) - 1):
        start, end = int(cu_q[i]), int(cu_q[i + 1])
        length = int(cu_k[i + 1] - cu_k[i])
        pos = torch.arange(length, device=q.device)
        slots = tables[i, pos // block_size].long() * block_size + pos % block_size
        k = k_cache.flatten(0, 1)[slots]
        v = v_cache.flatten(0, 1)[slots]
        mask = pos[None, :] <= (length - (end - start) + torch.arange(end - start, device=q.device))[:, None]
        outputs.append(F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1)[None],
            k.transpose(0, 1)[None], v.transpose(0, 1)[None],
            attn_mask=mask, scale=scale, enable_gqa=True,
        )[0].transpose(0, 1))
    return torch.cat(outputs)


@pytest.fixture
def cpu_model_backend(monkeypatch, tmp_path):
    """Replace device transport and Triton kernels, retaining runner/model math."""
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)

    def store(k, v, kc, vc, slots, block_size):
        kc.flatten(0, 1)[slots] = k
        vc.flatten(0, 1)[slots] = v

    def decode(q, kc, vc, tables, lengths, scale, heads, kv_heads, dim, block_size):
        cu_q = torch.arange(len(q) + 1)
        cu_k = torch.cat([torch.zeros(1, dtype=lengths.dtype), lengths.cumsum(0)])
        return reference_prefill(q, kc, vc, cu_q, cu_k, tables,
                                 1, scale, heads, kv_heads, dim, block_size)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(attention, "store_kvcache", store)
    monkeypatch.setattr(attention, "paged_attention_prefill", reference_prefill)
    monkeypatch.setattr(attention, "paged_attention_decode", decode)
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        with torch.compiler.set_stance("force_eager"):
            yield
    finally:
        reset_context()
        dist.destroy_process_group()


@pytest.mark.parametrize("family", ["qwen3", "llama"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_model_and_step_match_whole_prefill_on_cpu(cpu_model_backend, family, dtype):
    torch.manual_seed(8)
    common = dict(vocab_size=64, hidden_size=32, head_dim=16, num_kv_heads=1,
                  intermediate_size=64, num_layers=2, block_size=4)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if family == "qwen3":
            model = Qwen3ForCausalLM(**common, num_heads=2, max_position=32)
        else:
            model = LlamaForCausalLM(**common, num_qo_heads=2, max_position_embeddings=32)
    finally:
        torch.set_default_dtype(previous_dtype)
    # Match runner construction: BF16 weights retain FP32 RoPE cache buffers.
    model = model.eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=.1)
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank, runner.block_size = 0, 4
    runner.model, runner.sampler = model, SamplerLayer()
    runner.enforce_eager, runner.timings, runner.world_size = True, None, 1
    for module in model.modules():
        if isinstance(module, attention.Attention):
            module.k_cache = torch.full((16, 4, 1, 16), float("nan"), dtype=dtype)
            module.v_cache = torch.full_like(module.k_cache, float("nan"))
    tokens_by_mode, logits_by_mode = [], []
    for chunked in (False, True):
        engine = LLMEngine.__new__(LLMEngine)
        engine.model_runner = runner
        engine.scheduler = Scheduler(4, 3 if chunked else 32, 16, 4, [63], 32,
                                     enable_chunked_prefill=chunked)
        seqs = [Sequence(list(range(1, length + 1)), 4,
                         SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
                for length in (7, 10)]
        for seq in seqs:
            engine.scheduler.add_sequence(seq)
        logits, callback_tokens = [], []
        hook = model.lm_head.register_forward_hook(
            lambda module, args, output: logits.extend(output.detach().float().clone())
        )
        try:
            with torch.inference_mode():
                while not engine.scheduler.is_finished():
                    engine.step(lambda seq, token: callback_tokens.append((seq.seq_id, token)))
        finally:
            hook.remove()
        assert len(callback_tokens) == 8
        assert not engine.scheduler.block_manager.used_block_ids
        tokens_by_mode.append([seq.completion_token_ids for seq in seqs])
        # Different batch orders are expected; index logits by request callback.
        logits_by_mode.append([
            torch.stack([logit for logit, (seq_id, _) in zip(logits, callback_tokens)
                         if seq_id == seq.seq_id]) for seq in seqs
        ])
    assert tokens_by_mode[0] == tokens_by_mode[1]
    for baseline, chunked in zip(*logits_by_mode):
        assert torch.isfinite(baseline).all() and torch.isfinite(chunked).all()
        if dtype == torch.float32:
            torch.testing.assert_close(baseline, chunked)
        # BF16 GEMM/SDPA rounding depends on batch shape. Token equality and
        # finiteness are unit guards; measured BF16 logits need the project gate.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_paged_prefill_kernel_matches_sdpa_with_suffix_queries(dtype, block_size):
    torch.manual_seed(31)
    heads, kv_heads, dim = 4, 2, 32
    q_lengths, k_lengths = [5, 1, 17], [11, 3, 26]
    blocks_per_seq = [(length + block_size - 1) // block_size for length in k_lengths]
    total_blocks = sum(blocks_per_seq)
    physical = torch.randperm(total_blocks, device="cuda")
    tables = torch.full((3, max(blocks_per_seq)), -1, dtype=torch.int32, device="cuda")
    offset = 0
    for i, n in enumerate(blocks_per_seq):
        tables[i, :n] = physical[offset:offset + n]
        offset += n
    q = torch.randn(sum(q_lengths), heads, dim, dtype=dtype, device="cuda")
    k = torch.randn(total_blocks, block_size, kv_heads, dim, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 5, 6, 23], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 11, 14, 40], dtype=torch.int32, device="cuda")
    args = (q, k, v, cu_q, cu_k, tables, 17, dim ** -.5, heads, kv_heads, dim, block_size)
    actual = attention.paged_attention_prefill(*args)
    expected = reference_prefill(*args)
    # Kernel-level guards fixed before GPU execution; project alignment gates
    # still require the user's approval on measured full-model comparisons.
    torch.testing.assert_close(actual, expected, rtol=1e-4 if dtype == torch.float32 else .02,
                               atol=1e-5 if dtype == torch.float32 else .02)
