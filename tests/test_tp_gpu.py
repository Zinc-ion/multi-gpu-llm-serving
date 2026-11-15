import pytest
import torch

from tp_helpers import TOLERANCES, create_checkpoint, reference_run, run_model


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA GPUs")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_two_visible_gpus_hf_checkpoint_tp1_tp2(tmp_path, dtype_name):
    reference = create_checkpoint(tmp_path / "checkpoint", "qwen3", False, False)
    ref_tokens, ref_logits = reference_run(reference, dtype_name)
    single = run_model(tmp_path, "qwen3", False, False, dtype_name, 1, "cuda")[0]
    shards = run_model(tmp_path, "qwen3", False, False, dtype_name, 2, "cuda")
    assert [result["device"] for result in shards] == ["cuda:0", "cuda:1"]
    assert all(result["tokens"] == ref_tokens for result in [single, *shards])
    for result in shards:
        measurement = result["measurement"]
        assert measurement["timing_kind"] == "cuda_event_stream_elapsed"
        assert measurement["all_reduce_calls"] > 0 and measurement["gather_calls"] > 0
        assert measurement["all_reduce_ms"] >= 0 and measurement["gather_ms"] >= 0
        assert measurement["peak_allocated_bytes"] > 0
    for expected, one, two in zip(ref_logits, single["logits"], shards[0]["logits"]):
        torch.testing.assert_close(one, expected, **TOLERANCES[dtype_name])
        torch.testing.assert_close(two, expected, **TOLERANCES[dtype_name])
        torch.testing.assert_close(two, one, **TOLERANCES[dtype_name])
