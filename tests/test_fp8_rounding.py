"""FP8 conversion boundaries and non-finite values against the original reference."""

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from fp8_helpers import quantize_reference
from myvllm.layers.attention import store_kvcache, store_kvcache_kernel


def rounding_boundaries():
    fp8 = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (fp8[:-1] + fp8[1:]) / 2
    negative = torch.full_like(midpoints, float("-inf"))
    positive = torch.full_like(midpoints, float("inf"))
    values = torch.cat((midpoints, torch.nextafter(midpoints, negative),
                        torch.nextafter(midpoints, positive), fp8))
    return torch.cat((values, -values))


def test_round_to_odd_half_model_matches_fp8_boundaries():
    values = rounding_boundaries()
    nearest = values.to(torch.float16)
    bits = nearest.view(torch.int16).to(torch.int32)
    bits -= (nearest.float().abs() > values.abs()).to(torch.int32)
    truncated = bits.to(torch.int16).view(torch.float16)
    bits |= (truncated.float() != values).to(torch.int32)
    odd = bits.to(torch.int16).view(torch.float16)
    assert torch.equal(odd.to(torch.float8_e4m3fn).view(torch.uint8),
                       values.to(torch.float8_e4m3fn).view(torch.uint8))


@pytest.mark.parametrize("input_type", ["bf16", "fp32"])
def test_sm89_store_compiles_with_sticky_rounding_and_nan_propagation(tmp_path, input_type):
    signature = {
        "key_ptr": "*" + input_type, "value_ptr": "*" + input_type, "k_cache_ptr": "*fp8e4nv",
        "v_cache_ptr": "*fp8e4nv", "slot_mapping_ptr": "*i64",
        "k_scale_ptr": "*fp32", "v_scale_ptr": "*fp32",
    }
    constants = {"num_kv_heads": 8, "head_dim": 128, "block_size": 256, "FP8_CACHE": True}
    ptx = triton.compile(ASTSource(store_kvcache_kernel, signature, constants),
                         target=GPUTarget("cuda", 89, 32)).asm["ptx"]
    (tmp_path / f"store-sm89-{input_type}-d128.ptx").write_text(ptx)
    assert "cvt.rz.f16.f32" in ptx
    assert "cvt.rn.satfinite.e4m3x2.f16x2" in ptx
    assert "max.NaN.f32" in ptx and "min.NaN.f32" in ptx
    assert "setp.neu.f32" in ptx and "or.b16" in ptx


def run_store(key, value):
    tokens, heads, dim = key.shape
    data = torch.zeros(2, 1, tokens, heads, dim, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 1, tokens, heads, device="cuda", dtype=torch.float32)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    store_kvcache(key, value, data[0], data[1], slots, tokens, scales[0], scales[1])
    return data[:, 0], scales[:, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_store_preserves_sm89_rounding_counterexamples(dtype):
    key = torch.zeros(2, 1, 128, device="cuda", dtype=dtype)
    if dtype == torch.float32:
        amax, value = 448.0, 1.0625001192092896
    else:
        amax, value = 1.015625, 0.0012054443359375
    key[:, 0, 0] = amax
    key[0, 0, 1] = value
    key[1, 0, 1] = -value
    data, scales = run_store(key, key)
    expected, expected_scales = quantize_reference(key)
    for kv in range(2):
        assert torch.equal(data[kv].view(torch.uint8), expected.view(torch.uint8))
        torch.testing.assert_close(scales[kv], expected_scales, rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_store_rounds_all_fp8_midpoints_and_adjacent_fp32_values():
    values = rounding_boundaries().to("cuda")
    key = torch.zeros(values.numel(), 1, 16, device="cuda")
    key[:, 0, 0] = 448
    key[:, 0, 1] = values
    data, scales = run_store(key, key.flip(0).contiguous())
    for kv, source in enumerate((key, key.flip(0).contiguous())):
        expected, expected_scales = quantize_reference(source)
        assert torch.equal(data[kv].view(torch.uint8), expected.view(torch.uint8))
        torch.testing.assert_close(scales[kv], expected_scales, rtol=1e-6, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_store_propagates_nan_and_infinity_like_torch():
    key = torch.zeros(5, 1, 16, device="cuda")
    key[:, 0, 1] = 1
    key[:, 0, 0] = torch.tensor([float("nan"), float("inf"), float("-inf"),
                                 float("nan"), 0.0], device="cuda")
    key[3, 0, 2] = float("inf")
    value = key.flip(0).contiguous()
    data, scales = run_store(key, value)
    for kv, source in enumerate((key, value)):
        expected, expected_scales = quantize_reference(source)
        torch.testing.assert_close(scales[kv], expected_scales, rtol=1e-6, atol=0, equal_nan=True)
        assert torch.equal(data[kv].view(torch.uint8), expected.view(torch.uint8))
