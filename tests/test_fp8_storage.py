import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fp8_helpers import quantize_reference
from myvllm.config import resolve_kv_cache_dtype
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.layers.attention import store_kvcache
from myvllm.pd.protocol import model_spec


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cache_dtype_resolution_is_independent_of_model_dtype(model_dtype):
    assert resolve_kv_cache_dtype("auto", model_dtype) == model_dtype
    assert resolve_kv_cache_dtype("float8_e4m3fn", model_dtype) == torch.float8_e4m3fn
    for value in ("fp8", "float8_e5m2", "FLOAT8_E4M3FN", None, 1):
        with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
            resolve_kv_cache_dtype(value, model_dtype)


def test_pd_rejects_fp8_storage_before_transport():
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = {"world_size": 1, "model_revision": "a" * 40, "dtype": "bfloat16"}
    engine.model_runner = ModelRunner.__new__(ModelRunner)
    engine.model_runner.kv_cache = torch.zeros(2, 1, 4, 4, 1, 16, dtype=torch.float8_e4m3fn)
    engine.scheduler = Scheduler(1, 8, 4, 4, [0], 8)
    with pytest.raises(ValueError, match="KV cache dtype"):
        model_spec(engine)


def test_capacity_report_includes_scale_bytes(tmp_path):
    log = tmp_path / "startup.log"
    log.write_text(json.dumps({
        "event": "capacity", "block_count": 100, "block_bytes": 4 * 2 * 2 * 2 * (16 + 4),
        "model_dtype": "bfloat16", "kv_cache_dtype": "float8_e4m3fn",
        "data_block_bytes": 4 * 2 * 2 * 2 * 16, "scale_block_bytes": 4 * 2 * 2 * 2 * 4,
    }) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"block_size": 4, "max_model_length": 17408}))
    output = tmp_path / "capacity.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools/capacity_report.py"),
                    "--log", str(log), "--config", str(config), "--output", str(output)],
                   cwd=root, check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["measured"]["model_dtype"] == "bfloat16"
    assert report["measured"]["kv_cache_dtype"] == "float8_e4m3fn"
    assert report["bytes_per_token"] == 2 * 2 * 2 * (16 + 4)
    assert report["kv_bytes"] == 100 * 4 * 2 * 2 * 2 * (16 + 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_store_matches_independent_fp8_bits_and_per_head_scales(dtype):
    torch.manual_seed(503)
    # The input layout is deliberately noncontiguous; slots skip physical blocks
    # and -1 is graph padding. Each head has its own dynamic range.
    raw = torch.randn(2, 8, 2, 32, device="cuda", dtype=dtype)
    multipliers = torch.tensor([0, 1e-5, .1, 1, 10, 100, .003, 3], device="cuda", dtype=dtype)
    key = (raw[0] * multipliers[:, None, None]).transpose(0, 1).contiguous().transpose(0, 1)
    value = (raw[1] * multipliers.flip(0)[:, None, None]).transpose(0, 1).contiguous().transpose(0, 1)
    key[3, 1] = 0
    value[5, 0] = 0
    slots = torch.tensor([7, 0, -1, 14, 3, 9, 5, 12], device="cuda", dtype=torch.int64)
    kc = torch.zeros(4, 4, 2, 32, device="cuda", dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    ks = torch.ones(4, 4, 2, device="cuda", dtype=torch.float32)
    vs = torch.ones_like(ks)
    store_kvcache(key, value, kc, vc, slots, 4, ks, vs)

    expected_k, expected_ks = quantize_reference(key)
    expected_v, expected_vs = quantize_reference(value)
    active = slots >= 0
    destination = slots[active]
    for actual, expected in ((kc, expected_k), (vc, expected_v)):
        assert torch.equal(actual.view(torch.uint8).flatten(0, 1)[destination],
                           expected.view(torch.uint8)[active])
    for actual, expected in ((ks, expected_ks), (vs, expected_vs)):
        torch.testing.assert_close(actual.flatten(0, 1)[destination], expected[active], rtol=1e-6, atol=0)
    assert ks.flatten(0, 1)[14, 1].item() == 1.0
    assert vs.flatten(0, 1)[9, 0].item() == 1.0
    unused = torch.tensor([1, 2, 4, 6, 8, 10, 11, 13, 15], device="cuda")
    assert torch.count_nonzero(kc.view(torch.uint8).flatten(0, 1)[unused]) == 0
    assert torch.all(ks.flatten(0, 1)[unused] == 1)
    assert torch.all(vs.flatten(0, 1)[unused] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_store_scale_clamps_to_fp32_tiny():
    tiny = torch.finfo(torch.float32).tiny
    key = torch.zeros(5, 1, 16, device="cuda", dtype=torch.float32)
    key[0, 0, :3] = torch.tensor([tiny, tiny * 2, -tiny], device="cuda")
    key[1, 0, :2] = torch.tensor([tiny / 2, -tiny / 4], device="cuda")
    key[2, 0, 0] = tiny * 2**-23
    key[3, 0, :2] = torch.tensor([torch.finfo(torch.float32).max,
                                   -torch.finfo(torch.float32).max / 2], device="cuda")
    value = key.clone()
    cache = torch.zeros(2, 1, 5, 1, 16, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones(2, 1, 5, 1, device="cuda", dtype=torch.float32)
    slots = torch.tensor([0, 1, 2, 3, 4], device="cuda")
    store_kvcache(key, value, cache[0], cache[1], slots, 5, scales[0], scales[1])
    expected, expected_scales = quantize_reference(key)
    assert torch.equal(cache[0].view(torch.uint8).reshape(5, 1, 16), expected.view(torch.uint8))
    torch.testing.assert_close(scales[0].reshape(5, 1), expected_scales, rtol=1e-6, atol=0)
    assert scales[0, 0, 0, 0].item() == tiny
    assert scales[0, 0, 1, 0].item() == tiny
    assert scales[0, 0, 2, 0].item() == tiny
    assert scales[0, 0, 4, 0].item() == 1.0
