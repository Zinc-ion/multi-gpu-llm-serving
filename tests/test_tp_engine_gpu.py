import pytest
import torch

from tp_engine_helpers import assert_engine_matches, create_engine_checkpoint, engine_reference_run, run_engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_actual_engine_paged_prefill_then_decode_matches_hf(tmp_path, dtype_name):
    _, config = create_engine_checkpoint(tmp_path / "checkpoint", dtype_name)
    ref_tokens, ref_logits = engine_reference_run(tmp_path / "checkpoint", dtype_name)
    result = run_engine(tmp_path, config, 1)
    assert_engine_matches(result, ref_tokens, ref_logits, dtype_name)
