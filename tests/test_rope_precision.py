import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from myvllm.layers.rotary_embedding import RotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_matches_hf_at_short_and_long_positions(dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        config = Qwen3Config(head_dim=128, rope_theta=1000000, max_position_embeddings=32768)
        actual = RotaryEmbedding(1000000, 128, 32768)
        reference = Qwen3RotaryEmbedding(config)
    finally:
        torch.set_default_dtype(previous)
    positions = torch.tensor([0, 1, 1023, 4095, 16383, 32767])
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(6, 4, 128, generator=generator).to(dtype)
    k = torch.randn(6, 2, 128, generator=generator).to(dtype)
    cos, sin = reference(q, positions[None])
    expected_q, expected_k = apply_rotary_pos_emb(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin
    )
    # Isolate arithmetic from compiler fusion; compiled execution has a GPU test.
    with torch.compiler.set_stance("force_eager"):
        actual_q, actual_k = actual(positions, q, k)
    assert actual_q.dtype == actual_k.dtype == dtype
    torch.testing.assert_close(actual.inv_freq, reference.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(actual_q, expected_q[0].transpose(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k[0].transpose(0, 1), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_bf16_rope_can_feed_bf16_projection():
    rope = RotaryEmbedding(1000000, 128, 32).cuda()
    q = torch.randn(8, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    actual, _ = rope(torch.arange(8, device="cuda"), q, k)
    projection = torch.nn.Linear(256, 64, bias=False, device="cuda", dtype=q.dtype)
    assert projection(actual.reshape(8, 256)).dtype == q.dtype
