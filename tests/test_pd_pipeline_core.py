from copy import deepcopy

import pytest
import torch

from myvllm.pd import CapacityError, DecodeCore, PrefillCore, StateError
from myvllm.sampling_parameters import SamplingParams
from test_pd_core import make_engine, reference_tokens


def make_cores(**kwargs):
    p_engine = make_engine(**kwargs)
    p_engine.config["enforce_eager"] = True
    d_engine = make_engine(**kwargs)
    return PrefillCore(p_engine, 16, 1_000_000), DecodeCore(d_engine, 16, 1_000_000)


def prepare(p, d, prompt, params=None, deadline=None):
    params = params or SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    dh = d.open(deadline=deadline)
    ph = p.prepare(dh.session_id, "request", prompt, params, deadline=deadline)
    return ph, dh


def payload_for(p, ph, layer, start, end):
    seq = ph.sequence
    cache = p.engine.model_runner.kv_cache
    positions = [
        cache[:, layer, seq.block_table[pos // seq.block_size], pos % seq.block_size]
        for pos in range(start, end)
    ]
    return torch.stack(positions, dim=1).contiguous().view(torch.uint8).numpy().tobytes()


def assert_blocks(core):
    manager = core.manager
    free = set(manager.free_block_ids)
    used = manager.used_block_ids
    assert len(free) == len(manager.free_block_ids)
    assert free.isdisjoint(used)
    assert free | used == set(range(len(manager.blocks)))
    assert all(block.ref_count == 0 and block.hash == -1 for block in manager.blocks
               if block.block_id in free)
    assert all(block.ref_count > 0 for block in manager.blocks if block.block_id in used)


def first_token(p, handle):
    events = p.step()
    assert len(events) == 1 and events[0].handle is handle
    assert_blocks(p)
    return events[0]


@pytest.mark.parametrize("commit_before_frames", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_early_reservation_commit_and_ordered_regions(commit_before_frames, dtype):
    p, d = make_cores(blocks=12, dtype=dtype)
    other_p, other_d = prepare(p, d, [7])
    assert d.reserve_early(other_d, p.describe_early(other_p))
    prompt = [1, 2, 3, 4, 5]
    ph, dh = prepare(p, d, prompt)
    try:
        early = p.describe_early(ph)
        assert early["version"] == 3
        assert early["token_ids"] == prompt
        assert early["completion_count"] == early["cached_tokens"] == early["next_position"] == 0
        assert not p.scheduler.waiting and not p.scheduler.running
        assert p.step() == [] and p.engine.model_runner.calls == []
        assert p.stats()["reserved_prompt_blocks"] == 3
        assert p.prepare(dh.session_id, "request", prompt,
                         SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)) is ph
        assert d.reserve_early(dh, early)
        assert not d.reserve_early(dh, deepcopy(early))
        assert dh.sequence.prompt_token_ids == prompt
        assert len(dh.sequence.block_table) == 2
        assert d.stats()["reserved_blocks"] == 5
        assert_blocks(p)
        assert_blocks(d)
        with pytest.raises(StateError):
            d.activate(dh)
        with pytest.raises(StateError, match="ready"):
            p.start(ph)
        assert p.ready(ph)
        assert not p.ready(ph)
        assert p.start(ph)
        assert not p.start(ph)
        event = first_token(p, ph)
        assert not event.finished
        assert ph.sequence.block_table != dh.sequence.block_table
        metadata = p.describe(ph)
        if commit_before_frames:
            assert d.commit_early(dh, metadata)
            assert not d.commit_early(dh, deepcopy(metadata))
        with pytest.raises(StateError):
            d.activate(dh)

        for layer in range(2):
            for start, end in ((0, 3), (3, 5)):
                payload = payload_for(p, ph, layer, start, end)
                assert d.restore_region(dh, layer, start, end, payload)
                assert_blocks(d)
        if not commit_before_frames:
            with pytest.raises(StateError):
                d.activate(dh)
            assert d.commit_early(dh, metadata)
        assert dh.state == "restored"
        for pos in range(len(prompt)):
            source = ph.sequence.block_table[pos // 4]
            target = dh.sequence.block_table[pos // 4]
            assert torch.equal(p.engine.model_runner.kv_cache[:, :, source, pos % 4],
                               d.engine.model_runner.kv_cache[:, :, target, pos % 4])
        assert p.acknowledge_early(ph)
        assert not p.acknowledge_early(ph)
        assert d.activate(dh)
        assert not d.activate(dh)
        output = [event.token]
        while dh.state == "active":
            output.extend(item.token for item in d.step())
            assert_blocks(d)
        assert output == reference_tokens(prompt, dtype)
    finally:
        p.shutdown()
        d.shutdown()
    assert_blocks(p)
    assert_blocks(d)
    assert p.stats()["used_blocks"] == d.stats()["used_blocks"] == 0
    assert d.stats()["reserved_blocks"] == d.stats()["resident_requests"] == 0


def test_early_reservation_rejects_capacity_before_prefill():
    p_engine = make_engine()
    p_engine.config["enforce_eager"] = True
    p = PrefillCore(p_engine, 8, 1_000_000)
    d = DecodeCore(make_engine(sequences=1), 8, 1_000_000)
    first_p, first_d = prepare(p, d, [1, 2, 3])
    second_p, second_d = prepare(p, d, [4, 5])
    try:
        assert d.reserve_early(first_d, p.describe_early(first_p))
        with pytest.raises(CapacityError):
            d.reserve_early(second_d, p.describe_early(second_p))
        assert p.engine.model_runner.calls == []
        assert not p.scheduler.waiting and not p.scheduler.running
        assert d.stats()["used_blocks"] == 1
        assert d.stats()["resident_requests"] == 1
        assert_blocks(p)
        assert_blocks(d)
    finally:
        p.shutdown()
        d.shutdown()
    assert p.stats()["reserved_transfer_bytes"] == d.stats()["reserved_transfer_bytes"] == 0
    assert d.stats()["reserved_blocks"] == d.stats()["used_blocks"] == 0


@pytest.mark.parametrize("field", ["version", "session", "progress", "model", "length"])
def test_bad_early_metadata_rejected_without_d_allocation(field):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3])
    metadata = p.describe_early(ph)
    if field == "version":
        metadata["version"] = 2
    elif field == "session":
        metadata["session_id"] = "another"
    elif field == "progress":
        metadata["cached_tokens"] = 1
    elif field == "model":
        metadata["model"]["revision"] = "another"
    else:
        metadata["kv"]["nbytes"] += 1
    try:
        with pytest.raises(ValueError):
            d.reserve_early(dh, metadata)
        assert dh.state == "failed"
        assert d.stats()["reserved_blocks"] == d.stats()["used_blocks"] == 0
        assert d.stats()["reserved_transfer_bytes"] == d.stats()["resident_requests"] == 0
        assert p.engine.model_runner.calls == []
    finally:
        p.shutdown()
        d.shutdown()


@pytest.mark.parametrize("fault", ["duplicate", "wrong_layer", "gap", "bad_length", "bad_commit"])
def test_invalid_region_or_commit_reclaims_and_survivor_runs(fault):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5])
    try:
        d.reserve_early(dh, p.describe_early(ph))
        p.ready(ph)
        p.start(ph)
        first_token(p, ph)
        if fault == "bad_commit":
            metadata = p.describe(ph)
            metadata["request_id"] = "another"
            with pytest.raises(ValueError, match="commit"):
                d.commit_early(dh, metadata)
        else:
            payload = payload_for(p, ph, 0, 0, 2)
            if fault == "duplicate":
                d.restore_region(dh, 0, 0, 2, payload)
                args = (0, 0, 2, payload)
            elif fault == "wrong_layer":
                args = (1, 0, 2, payload)
            elif fault == "gap":
                args = (0, 1, 3, payload)
            else:
                args = (0, 0, 2, payload[:-1])
            with pytest.raises(ValueError, match="region"):
                d.restore_region(dh, *args)
        assert dh.state == "failed"
        assert d.stats()["used_blocks"] == d.stats()["reserved_blocks"] == 0
        assert d.stats()["reserved_transfer_bytes"] == d.stats()["resident_requests"] == 0
        with pytest.raises(StateError):
            d.restore_region(dh, 0, 0, 2, payload_for(p, ph, 0, 0, 2))
    finally:
        p.close(ph)
        d.close(dh)
    next_p, next_d = prepare(p, d, [5, 6])
    try:
        assert d.reserve_early(next_d, p.describe_early(next_p))
        assert p.ready(next_p)
        assert p.start(next_p)
        assert not first_token(p, next_p).finished
    finally:
        p.shutdown()
        d.shutdown()
    assert_blocks(p)
    assert_blocks(d)
    assert p.stats()["used_blocks"] == d.stats()["used_blocks"] == 0


def test_first_token_eos_and_expiry_release_early_reservations():
    p, d = make_cores(forced_token=62)
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=False)
    ph, dh = prepare(p, d, [1, 2, 3], params)
    assert d.reserve_early(dh, p.describe_early(ph))
    assert p.ready(ph)
    assert p.start(ph)
    event = first_token(p, ph)
    assert event.finished and event.finish_reason == "stop"
    assert d.cancel(dh)
    p.close(ph)
    d.close(dh)
    ph, dh = prepare(p, d, [1, 2, 3], deadline=10)
    assert d.reserve_early(dh, p.describe_early(ph))
    assert p.expire(now=10) == [ph]
    assert d.expire(now=10) == [dh]
    assert not p.cancel(ph) and not d.cancel(dh)
    p.close(ph)
    d.close(dh)
    assert_blocks(p)
    assert_blocks(d)
    assert p.stats()["used_blocks"] == d.stats()["used_blocks"] == 0
    assert d.stats()["reserved_blocks"] == d.stats()["resident_requests"] == 0


def test_known_terminal_first_uses_bulk_path():
    p, d = make_cores()
    dh = d.open()
    try:
        with pytest.raises(ValueError, match="second token"):
            p.prepare(dh.session_id, "one", [1, 2], SamplingParams(max_tokens=1))
        ph = p.open(dh.session_id, "one", [1, 2], SamplingParams(max_tokens=1))
        assert first_token(p, ph).finished
    finally:
        p.shutdown()
        d.shutdown()


@pytest.mark.parametrize("chunked", [False, True])
def test_early_prepare_requires_eager_prefill(chunked):
    engine = make_engine(chunked=chunked)
    engine.config["enforce_eager"] = False
    p = PrefillCore(engine, 8, 1_000_000)
    try:
        with pytest.raises(ValueError, match="eager prefill"):
            p.prepare("session", "request", [1, 2], SamplingParams(max_tokens=3))
        assert p.stats()["open_handles"] == p.stats()["reserved_transfer_bytes"] == 0
    finally:
        p.shutdown()


def test_prefill_prompt_reservation_accounts_for_bulk_requests():
    engine = make_engine(blocks=4)
    engine.config["enforce_eager"] = True
    p = PrefillCore(engine, 8, 1_000_000)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    try:
        early = p.prepare("early", "request", [1, 2, 3, 4], params)
        bulk = p.open("bulk", "request", [1] * 8, params)
        assert p.stats()["reserved_prompt_blocks"] == 3
        with pytest.raises(CapacityError, match="P prompt"):
            p.prepare("early-full", "request", [1] * 5, params)
        with pytest.raises(CapacityError, match="P prompt"):
            p.open("bulk-full", "request", [1] * 8, params)
        assert p.cancel(early)
        assert p.stats()["reserved_prompt_blocks"] == 0
        assert p.open("bulk-full", "request", [1] * 8, params)
        assert bulk.state == "queued"
    finally:
        p.shutdown()
    assert p.stats()["used_blocks"] == p.stats()["reserved_transfer_bytes"] == 0


def test_late_duplicate_region_cancels_restored_request():
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3])
    try:
        d.reserve_early(dh, p.describe_early(ph))
        p.ready(ph)
        p.start(ph)
        first_token(p, ph)
        d.commit_early(dh, p.describe(ph))
        first_payload = None
        for layer in range(2):
            payload = payload_for(p, ph, layer, 0, 3)
            if layer == 0:
                first_payload = payload
            d.restore_region(dh, layer, 0, 3, payload)
        assert dh.state == "restored"
        with pytest.raises(StateError):
            d.restore_region(dh, 0, 0, 3, first_payload)
        assert dh.state == "failed"
        assert d.stats()["reserved_blocks"] == d.stats()["used_blocks"] == 0
    finally:
        p.shutdown()
        d.shutdown()
