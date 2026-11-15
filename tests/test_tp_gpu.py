import pytest
import torch

from tp_engine_helpers import assert_engine_matches, create_engine_checkpoint, engine_reference_run, run_engine
from tp_helpers import TOLERANCES


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA GPUs")
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_two_visible_gpus_hf_checkpoint_tp1_tp2(tmp_path, dtype_name):
    _, config = create_engine_checkpoint(tmp_path / "checkpoint", dtype_name)
    ref_tokens, ref_logits = engine_reference_run(tmp_path / "checkpoint", dtype_name)
    single = run_engine(tmp_path, config, 1)
    shards = run_engine(tmp_path, config, 2)
    for result in (single, shards):
        assert_engine_matches(result, ref_tokens, ref_logits, dtype_name)
    for one, two in zip(single["ranks"][0]["logits"], shards["ranks"][0]["logits"]):
        torch.testing.assert_close(two, one, **TOLERANCES[dtype_name])
