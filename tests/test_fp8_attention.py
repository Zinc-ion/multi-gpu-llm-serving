import pytest
import torch

from fp8_helpers import attention_reference, quantize_reference, tolerances
from myvllm.layers.attention import paged_attention_decode, paged_attention_prefill, store_kvcache


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads,kv_heads,block_size,dim", [(4, 2, 3, 32), (2, 2, 16, 32), (4, 2, 16, 128)])
def test_fp8_paged_prefill_and_decode_match_quantized_torch_reference(dtype, heads, kv_heads, block_size, dim):
    torch.manual_seed(20261116)
    q_lengths = [5, 1, 17]
    k_lengths = [11, 3, 26]
    sizes = [(length + block_size - 1) // block_size for length in k_lengths]
    block_count = sum(sizes) + 3
    physical = torch.randperm(block_count, device="cuda")[:sum(sizes)]
    tables = torch.full((3, max(sizes)), -1, dtype=torch.int32, device="cuda")
    slots = []
    offset = 0
    for i, (size, length) in enumerate(zip(sizes, k_lengths)):
        table = physical[offset:offset + size]
        tables[i, :size] = table
        positions = torch.arange(length, device="cuda")
        slots.append(table[positions // block_size] * block_size + positions % block_size)
        offset += size
    slots = torch.cat(slots)
    assert (physical[1:] - physical[:-1]).abs().ne(1).any().item()
    assert any(length % block_size for length in k_lengths)
    k = torch.randn(sum(k_lengths), kv_heads, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    factors = torch.tensor([.003, .1, 1., 4.], device="cuda", dtype=dtype)
    k *= factors[torch.arange(len(k), device="cuda") % len(factors), None, None]
    v *= factors[torch.arange(len(v), device="cuda").flip(0) % len(factors), None, None]
    k[0, 0] = 0
    v[-1, -1] = 0
    kc = torch.zeros(block_count, block_size, kv_heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    ks = torch.ones(block_count, block_size, kv_heads, device="cuda", dtype=torch.float32)
    vs = torch.ones_like(ks)
    store_kvcache(k, v, kc, vc, slots, block_size, ks, vs)
    for original, cache, scales in ((k, kc, ks), (v, vc, vs)):
        expected_data, expected_scales = quantize_reference(original)
        assert torch.equal(cache.view(torch.uint8).flatten(0, 1)[slots], expected_data.view(torch.uint8))
        torch.testing.assert_close(scales.flatten(0, 1)[slots], expected_scales, rtol=1e-6, atol=0)
    scale = dim ** -.5

    q = torch.randn(sum(q_lengths), heads, dim, device="cuda", dtype=dtype)
    cu_q = torch.tensor([0, 5, 6, 23], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 11, 14, 40], dtype=torch.int32, device="cuda")
    actual = paged_attention_prefill(q, kc, vc, cu_q, cu_k, tables, 17, scale,
                                      heads, kv_heads, dim, block_size, ks, vs)
    expected = attention_reference(q, k, v, q_lengths, k_lengths, scale)
    torch.testing.assert_close(actual, expected, **tolerances(dtype))
    assert torch.isfinite(actual).all()

    # The last row is graph padding with no valid KV; it must return exact zero.
    decode_q = torch.randn(4, heads, dim, device="cuda", dtype=dtype)
    padded_tables = torch.cat([tables, torch.full_like(tables[:1], -1)], dim=0)
    lengths = torch.tensor(k_lengths + [0], dtype=torch.int32, device="cuda")
    actual_decode = paged_attention_decode(decode_q, kc, vc, padded_tables, lengths, scale,
                                            heads, kv_heads, dim, block_size, ks, vs)
    expected_decode = attention_reference(decode_q[:3], k, v, [1, 1, 1], k_lengths, scale)
    torch.testing.assert_close(actual_decode[:3], expected_decode, **tolerances(dtype))
    assert torch.equal(actual_decode[3], torch.zeros_like(actual_decode[3]))
    assert torch.isfinite(actual_decode).all()
