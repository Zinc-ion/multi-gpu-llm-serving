import json
from pathlib import Path

import pytest

from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


@pytest.mark.parametrize("profiled_blocks", [6, 2048])
def test_14b_budgets_advance_every_resident_request(profiled_blocks):
    config = json.loads((Path(__file__).parents[1] / "configs/qwen3_14b.json").read_text())
    scheduler = Scheduler(
        max_num_sequences=config["max_num_sequences"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        max_cached_blocks=min(profiled_blocks, config["max_cached_blocks"]),
        block_size=config["block_size"],
        eos=config["eos"],
        max_model_length=min(1024, profiled_blocks * config["block_size"]),
    )
    sequences = []
    # Separate arrivals reproduce running growing beyond a single prefill batch.
    for _ in range(3):
        wave = [
            Sequence([1], config["block_size"], SamplingParams(max_tokens=8))
            for _ in range(2)
        ]
        sequences.extend(wave)
        for seq in wave:
            scheduler.add_sequence(seq)
        scheduled, prefill = scheduler.schedule()
        assert prefill
        assert scheduled == wave
        scheduler.postprocess(scheduled, [2] * len(scheduled))
    assert len(scheduler.running) == 6
    for _ in range(6):
        scheduled, prefill = scheduler.schedule()
        assert not prefill
        scheduler.postprocess(scheduled, [2] * len(scheduled))
        assert len({seq.num_completion_tokens for seq in sequences}) == 1
    for seq in sequences:
        assert scheduler.cancel(seq.seq_id)
    manager = scheduler.block_manager
    assert len(manager.free_block_ids) == len(manager.blocks)
    assert not manager.used_block_ids
    assert all(block.ref_count == 0 and block.hash == -1 for block in manager.blocks)


def test_14b_limits_leave_capacity_profiling_in_control():
    config = json.loads((Path(__file__).parents[1] / "configs/qwen3_14b.json").read_text())
    block_bytes = (2 * config["num_layers"] * config["num_kv_heads"]
                   * config["head_dim"] * 2 * config["block_size"])
    # Hardware snapshot: even the entire device cannot reach the configured cap.
    assert config["max_cached_blocks"] * block_bytes > 49140 * 1024 ** 2
    assert min(config["max_num_sequences"], config["max_num_batched_tokens"]) >= config["max_cached_blocks"]
