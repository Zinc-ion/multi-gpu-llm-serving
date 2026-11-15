from copy import deepcopy

import pytest
import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.pd import CapacityError, DecodeCore, PrefillCore, StateError
from myvllm.sampling_parameters import SamplingParams


class CPUExecution:
    """Replace model execution while retaining the engine and scheduler."""

    timings = None

    def __init__(self, config, forced_token=None):
        self.kv_cache = torch.full(
            (2, config["num_layers"], config["max_cached_blocks"], config["block_size"],
             config["num_kv_heads"], config["head_dim"]),
            -9, dtype=getattr(torch, config["dtype"]), device="cpu",
        )
        self.forced_token = forced_token
        self.calls = []

    def call(self, method, sequences, is_prefill):
        assert method == "run"
        sampled = []
        for seq in sequences:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            self.calls.append((seq.seq_id, start, end, tuple(seq.block_table), is_prefill))
            for pos in range(start, end):
                block_id = seq.block_table[pos // seq.block_size]
                for kv in range(2):
                    for layer in range(self.kv_cache.shape[1]):
                        self.kv_cache[kv, layer, block_id, pos % seq.block_size] = (
                            seq.token_ids[pos] + pos * 0.25 + layer * 0.5 + kv
                        )
            if end == len(seq):
                sampled.append(self.forced_token if self.forced_token is not None
                               else sum(seq.token_ids) % 60 + 1)
        return torch.tensor(sampled, device="cpu")


def make_engine(*, blocks=8, block_size=4, budget=16, sequences=4,
                model_length=16, chunked=False, dtype="float32", forced_token=None):
    config = {
        "model_name_or_path": "tiny-model", "model_type": "Qwen3-0.6B",
        "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
        "world_size": 1, "block_size": block_size, "vocab_size": 64,
        "hidden_size": 8, "num_heads": 2, "num_kv_heads": 1, "head_dim": 4,
        "num_layers": 2, "intermediate_size": 16, "base": 10000,
        "max_cached_blocks": blocks, "dtype": dtype,
    }
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = config
    engine.scheduler = Scheduler(sequences, budget, blocks, block_size, [62, 63],
                                 model_length, enable_chunked_prefill=chunked)
    engine.model_runner = CPUExecution(config, forced_token)
    engine.request_ids = {}
    return engine


def make_cores(**engine_kwargs):
    p_engine = make_engine(**engine_kwargs)
    d_engine = make_engine(**engine_kwargs)
    return (PrefillCore(p_engine, 16, 1_000_000),
            DecodeCore(d_engine, 16, 1_000_000))


def start(p, d, prompt, *, params=None, deadline=None):
    params = params or SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    dh = d.open(deadline=deadline)
    ph = p.open(dh.session_id, "same-external-id", prompt, params, deadline=deadline)
    assert_conserved(p)
    assert_conserved(d)
    return ph, dh


def first_token(p, handle):
    for _ in range(32):
        events = p.step()
        assert_conserved(p)
        for event in events:
            if event.handle is handle:
                return event
    pytest.fail("P did not produce the first token")


def handoff(p, d, ph, dh, *, activate=True):
    metadata = p.describe(ph)
    assert d.reserve(dh, metadata)
    assert_conserved(p)
    assert_conserved(d)
    payload = p.export(ph)
    assert_conserved(p)
    assert d.restore(dh, payload)
    assert_conserved(d)
    assert p.acknowledge(ph)
    assert_conserved(p)
    if activate:
        assert d.activate(dh)
        assert_conserved(d)
    return metadata, payload


def assert_conserved(core):
    manager = core.manager
    free = set(manager.free_block_ids)
    used = manager.used_block_ids
    assert len(free) == len(manager.free_block_ids)
    assert free.isdisjoint(used)
    assert free | used == set(range(len(manager.blocks)))
    for block in manager.blocks:
        if block.block_id in free:
            assert block.ref_count == 0 and block.hash == -1
        else:
            assert block.ref_count > 0

    waiting = list(core.scheduler.waiting)
    running = list(core.scheduler.running)
    assert len(set(waiting)) == len(waiting)
    assert len(set(running)) == len(running)
    assert set(waiting).isdisjoint(running)
    scheduled = set()
    owned_blocks = []
    for handle in core._handles.values():
        seq = handle.sequence
        state = handle.state
        if seq is not None:
            assert seq not in scheduled
            scheduled.add(seq)
            owned_blocks.extend(seq.block_table)

        in_waiting = seq is not None and seq in waiting
        in_running = seq is not None and seq in running
        if isinstance(core, PrefillCore) and state in ("queued", "prefill"):
            assert seq is not None and in_waiting != in_running
        elif isinstance(core, PrefillCore) and state in ("held", "exported"):
            assert seq is not None and seq.block_table and not in_waiting and not in_running
        elif isinstance(core, DecodeCore) and state == "restored":
            assert seq is not None and seq.block_table and not in_waiting and not in_running
        elif isinstance(core, DecodeCore) and state == "active":
            assert seq is not None and seq.block_table and in_running and not in_waiting
        elif state in ("open", "reserved"):
            assert seq is None
        else:
            assert state in ("finished", "handed_off", "cancelled", "expired", "failed")
            assert seq is None or not seq.block_table
            assert not in_waiting and not in_running

    assert set(waiting) | set(running) <= scheduled
    assert len(owned_blocks) == len(set(owned_blocks))
    assert set(owned_blocks) == used
    stats = core.stats()
    assert stats["free_blocks"] + stats["used_blocks"] == len(manager.blocks)
    assert stats["reserved_transfer_bytes"] <= core.max_transfer_bytes
    if isinstance(core, DecodeCore):
        assert stats["used_blocks"] <= stats["reserved_blocks"] <= len(manager.blocks)
        assert stats["resident_requests"] <= stats["resident_limit"]


def reference_tokens(prompt, dtype):
    engine = make_engine(dtype=dtype)
    seq = Sequence(prompt, 4, SamplingParams(temperature=0, max_tokens=6, ignore_eos=True))
    engine.scheduler.add_sequence(seq)
    while not engine.scheduler.is_finished():
        engine.step()
    return seq.completion_token_ids


@pytest.mark.parametrize("length", [3, 4, 5])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cached_blocks_tail_local_mapping_and_future_decode(length, dtype):
    p, d = make_cores(dtype=dtype)
    prompt = list(range(1, length + 1))
    ph, dh = start(p, d, prompt)
    event = first_token(p, ph)
    assert not event.finished and event.completion_tokens == 1
    assert ph.sequence.num_cached_tokens == length
    assert ph.sequence.num_blocks == (length + 4) // 4
    assert len(ph.sequence.block_table) == (length + 3) // 4

    # Occupy D's first block with a distinct request while P's target is held.
    other_p, other_d = start(p, d, [7])
    first_token(p, other_p)
    handoff(p, d, other_p, other_d, activate=False)
    metadata = p.describe(ph)
    assert p.stats()["buffered_bytes"] == 0
    assert metadata["cached_tokens"] == metadata["next_position"] == length
    assert metadata["completion_count"] == 1
    assert metadata["kv"]["shape"] == [2, 2, (length + 3) // 4, 4, 1, 4]
    assert d.reserve(dh, metadata)
    assert dh.sequence is None
    source_ids = list(ph.sequence.block_table)
    source = p.engine.model_runner.kv_cache
    payload = p.export(ph)
    assert type(payload) is bytes and len(payload) == metadata["kv"]["nbytes"]
    wire = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(
        getattr(torch, dtype)).reshape(metadata["kv"]["shape"])
    for i, block_id in enumerate(source_ids):
        valid = min(4, length - 4 * i)
        assert torch.equal(wire[:, :, i, :valid], source[:, :, block_id, :valid])
        assert torch.count_nonzero(wire[:, :, i, valid:]) == 0
    assert d.restore(dh, payload)
    seq = dh.sequence
    assert seq.seq_id != ph.sequence.seq_id
    assert seq.block_table != source_ids
    assert seq.prompt_token_ids == prompt
    assert seq.completion_token_ids == [event.token]
    assert seq.num_cached_tokens == length
    for i, block_id in enumerate(seq.block_table[:len(source_ids)]):
        assert torch.equal(d.engine.model_runner.kv_cache[:, :, block_id], wire[:, :, i])
    assert p.acknowledge(ph)
    assert d.cancel(other_d)
    d.close(other_d)
    p.close(other_p)
    assert d.activate(dh)

    emitted = [event.token]
    while dh.state == "active":
        emitted.extend(item.token for item in d.step())
        assert_conserved(d)
    assert emitted == reference_tokens(prompt, dtype)
    calls = [call for call in d.engine.model_runner.calls if call[0] == seq.seq_id]
    assert calls[0][1:3] == (length, length + 1)
    assert all(end == start + 1 and not prefill for _, start, end, _, prefill in calls)
    assert len(calls[-1][3]) > len(source_ids)
    assert not d.manager.used_block_ids and not p.manager.used_block_ids
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)


@pytest.mark.parametrize(
    "forced,ignore,max_tokens,limit,finished,reason",
    [(62, False, 6, 16, True, "stop"), (63, False, 6, 16, True, "stop"),
     (63, True, 6, 16, False, None), (9, True, 1, 16, True, "length"),
     (9, True, 6, 4, True, "length")],
)
def test_first_token_stopping_conditions(forced, ignore, max_tokens, limit, finished, reason):
    p, d = make_cores(forced_token=forced)
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore, max_model_length=limit)
    ph, dh = start(p, d, [1, 2, 3], params=params)
    event = first_token(p, ph)
    assert (event.token, event.finished, event.finish_reason) == (forced, finished, reason)
    if finished:
        assert ph.state == "finished"
        assert not p.manager.used_block_ids
        with pytest.raises(StateError):
            p.describe(ph)
    else:
        handoff(p, d, ph, dh)
        while dh.state == "active":
            d.step()
        assert dh.sequence.num_completion_tokens == max_tokens
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)


@pytest.mark.parametrize("field", [
    "revision", "dtype", "eos", "architecture", "shape", "nbytes", "cached",
    "next", "completion", "tokens", "sampling", "session", "terminal_eos", "byte_order",
    "huge_temperature",
])
def test_invalid_metadata_rejected_before_allocation(field):
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3],
                   params=SamplingParams(max_tokens=6, ignore_eos=False))
    first_token(p, ph)
    metadata = p.describe(ph)
    bad = deepcopy(metadata)
    if field == "revision":
        bad["model"]["revision"] = "different"
    elif field == "dtype":
        bad["model"]["dtype"] = "float16"
    elif field == "eos":
        bad["model"]["eos"] = [63]
    elif field == "architecture":
        bad["model"]["architecture"]["head_dim"] += 1
    elif field == "shape":
        bad["kv"]["shape"][2] += 1
    elif field == "nbytes":
        bad["kv"]["nbytes"] -= 1
    elif field == "cached":
        bad["cached_tokens"] -= 1
    elif field == "next":
        bad["next_position"] += 1
    elif field == "completion":
        bad["completion_count"] = 2
    elif field == "tokens":
        bad["token_ids"].pop()
    elif field == "sampling":
        bad["sampling"]["max_model_length"] = 100
    elif field == "session":
        bad["session_id"] = "stale"
    elif field == "terminal_eos":
        bad["token_ids"][-1] = 63
    elif field == "byte_order":
        bad["kv"]["byte_order"] = "big" if bad["kv"]["byte_order"] == "little" else "little"
    elif field == "huge_temperature":
        bad["sampling"]["temperature"] = 10**400
    with pytest.raises(ValueError):
        d.reserve(dh, bad)
    assert dh.state == "failed"
    assert d.stats()["reserved_blocks"] == d.stats()["used_blocks"] == 0
    assert p.stats()["buffered_bytes"] == 0
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)


def test_short_payload_failure_releases_reservation_and_survivor_runs():
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3])
    first_token(p, ph)
    metadata = p.describe(ph)
    d.reserve(dh, metadata)
    payload = p.export(ph)
    with pytest.raises(ValueError, match="payload"):
        d.restore(dh, payload[:-1])
    assert dh.state == "failed"
    assert d.stats()["reserved_blocks"] == d.stats()["reserved_transfer_bytes"] == 0
    with pytest.raises(StateError):
        d.restore(dh, payload)
    assert p.cancel(ph)
    p.close(ph)
    d.close(dh)

    ph, dh = start(p, d, [4, 5])
    first_token(p, ph)
    handoff(p, d, ph, dh)
    while dh.state == "active":
        d.step()
    assert dh.state == "finished"
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)


@pytest.mark.parametrize("capacity", ["blocks", "resident", "transfer"])
def test_d_admission_rejects_full_reserved_budget(capacity):
    p = PrefillCore(make_engine(), 8, 1_000_000)
    if capacity == "blocks":
        d = DecodeCore(make_engine(blocks=4), 8, 1_000_000)
    elif capacity == "resident":
        d = DecodeCore(make_engine(sequences=1), 8, 1_000_000)
    else:
        d = DecodeCore(make_engine(), 8, 256)
    first_p, first_d = start(p, d, [1, 2, 3])
    first_token(p, first_p)
    d.reserve(first_d, p.describe(first_p))
    assert d.stats()["used_blocks"] == 0
    if capacity == "blocks":
        assert d.stats()["reserved_blocks"] == 3
    second_p, second_d = start(p, d, [4, 5, 6])
    first_token(p, second_p)
    with pytest.raises(CapacityError, match="capacity"):
        d.reserve(second_d, p.describe(second_p))
    assert second_d.state == "failed"
    assert_conserved(p)
    assert_conserved(d)
    p.shutdown()
    d.shutdown()
    assert not p.manager.used_block_ids and not d.manager.used_block_ids


def test_held_blocks_wait_without_scheduler_no_progress_error():
    p = PrefillCore(make_engine(blocks=2, budget=8, model_length=8), 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=4, budget=8, model_length=8), 8, 1_000_000)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    first_p, first_d = start(p, d, [1, 2, 3, 4], params=params)
    second_p, second_d = start(p, d, [5, 6, 7, 8], params=params)
    events = p.step()
    assert {event.handle for event in events} == {first_p, second_p}
    waiting_p, waiting_d = start(p, d, [1, 1, 1, 1], params=params)
    assert p.step() == []
    assert waiting_p.state == "queued"
    assert p.stats()["held_requests"] == 2
    handoff(p, d, first_p, first_d, activate=False)
    assert d.cancel(first_d)
    assert first_token(p, waiting_p).handle is waiting_p
    assert_conserved(p)
    assert_conserved(d)
    p.shutdown()
    d.shutdown()
    assert not p.manager.used_block_ids and not d.manager.used_block_ids


def test_duplicates_cancel_close_and_expiry_are_connection_scoped():
    p, d = make_cores(chunked=True, budget=2)
    ph, dh = start(p, d, [1, 2, 3, 4, 5], deadline=100)
    assert p.open(dh.session_id, "same-external-id", [1, 2, 3, 4, 5],
                  SamplingParams(temperature=0, max_tokens=6, ignore_eos=True),
                  deadline=100) is ph
    assert p.step() == []
    assert ph.state == "prefill" and ph.sequence.num_cached_tokens == 2
    assert ph.sequence in p.scheduler.running
    assert p.expire(now=99) == []
    assert p.expire(now=100) == [ph]
    assert ph.sequence.status == SequenceStatus.CANCELLED
    assert not p.cancel(ph)
    assert d.expire(now=100) == [dh]
    assert not d.cancel(dh)
    assert p.close(ph) and d.close(dh)
    assert not p.close(ph) and not d.close(dh)
    with pytest.raises(StateError):
        p.describe(ph)
    with pytest.raises(StateError):
        d.reserve(dh, {})
    assert_conserved(p)
    assert_conserved(d)

    ph, dh = start(p, d, [7, 8, 9])
    first_token(p, ph)
    metadata = p.describe(ph)
    original_metadata = deepcopy(metadata)
    metadata["model"]["revision"] = "changed"
    metadata["token_ids"].clear()
    assert p.describe(ph) == original_metadata
    metadata = p.describe(ph)
    assert d.reserve(dh, metadata)
    metadata["model"]["revision"] = "changed"
    assert not d.reserve(dh, deepcopy(original_metadata))
    changed = deepcopy(original_metadata)
    changed["request_id"] = "different"
    with pytest.raises(StateError, match="metadata"):
        d.reserve(dh, changed)
    payload = p.export(ph)
    assert p.export(ph) is payload
    assert d.restore(dh, payload)
    assert not d.restore(dh, payload)
    changed_payload = bytes([payload[0] ^ 1]) + payload[1:]
    first_result = d.engine.model_runner.kv_cache.clone()
    assert not d.restore(dh, changed_payload)
    assert torch.equal(d.engine.model_runner.kv_cache, first_result)
    with pytest.raises(ValueError, match="payload"):
        d.restore(dh, payload[:-1])
    assert p.acknowledge(ph)
    assert not p.acknowledge(ph)
    assert d.activate(dh)
    assert not d.activate(dh)
    assert d.cancel(dh)
    assert not d.cancel(dh)
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)
    new_d = d.open()
    assert new_d.session_id != dh.session_id
    with pytest.raises(ValueError, match="session"):
        d.reserve(new_d, original_metadata)
    d.close(new_d)
    assert p.stats()["open_handles"] == d.stats()["open_handles"] == 0


@pytest.mark.parametrize("phase", ["queued", "held", "exported", "reserved", "restored", "active"])
def test_cancel_at_each_public_handoff_phase(phase):
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3, 4])
    if phase != "queued":
        first_token(p, ph)
    if phase in ("reserved", "restored", "active"):
        d.reserve(dh, p.describe(ph))
    if phase in ("exported", "restored", "active"):
        payload = p.export(ph)
    if phase in ("restored", "active"):
        d.restore(dh, payload)
    if phase == "active":
        p.acknowledge(ph)
        d.activate(dh)
        assert d.step()
        assert dh.state == "active"
    assert_conserved(p)
    assert_conserved(d)
    assert p.cancel(ph) == (phase != "active")
    assert d.cancel(dh)
    assert not p.cancel(ph) and not d.cancel(dh)
    assert_conserved(p)
    assert_conserved(d)
    assert not p.manager.used_block_ids and not d.manager.used_block_ids
    assert d.stats()["reserved_blocks"] == d.stats()["resident_requests"] == 0
    p.close(ph)
    d.close(dh)
    assert ph.sequence is None and dh.sequence is None
    with pytest.raises(StateError):
        p.export(ph)
    with pytest.raises(StateError):
        d.activate(dh)


def test_two_decode_requests_keep_their_full_budgets_and_never_recompute():
    p, d = make_cores(blocks=6, model_length=16, budget=16, sequences=2)
    handles = [
        start(p, d, [1, 2, 3], params=SamplingParams(max_tokens=6, ignore_eos=True)),
        start(p, d, [4, 5, 6, 7], params=SamplingParams(max_tokens=5, ignore_eos=True)),
    ]
    first = p.step()
    assert len(first) == 2
    for ph, dh in handles:
        handoff(p, d, ph, dh)
    assert d.stats()["reserved_blocks"] == 6
    calls_before = len(d.engine.model_runner.calls)
    while any(dh.state == "active" for _, dh in handles):
        d.step()
        assert_conserved(d)
    for ph, dh in handles:
        seq = dh.sequence
        calls = [call for call in d.engine.model_runner.calls[calls_before:] if call[0] == seq.seq_id]
        assert calls and calls[0][1] == seq.num_prompt_tokens
        assert all(end == start + 1 and not prefill for _, start, end, _, prefill in calls)
        assert seq.status == SequenceStatus.FINISHED
        p.close(ph)
        d.close(dh)
    assert d.stats()["reserved_blocks"] == d.stats()["resident_requests"] == 0
    assert_conserved(p)
    assert_conserved(d)


def test_expiry_releases_exported_and_restored_resources():
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3], deadline=20)
    first_token(p, ph)
    d.reserve(dh, p.describe(ph))
    d.restore(dh, p.export(ph))
    assert p.expire(now=20) == [ph]
    assert d.expire(now=20) == [dh]
    assert ph.state == dh.state == "expired"
    assert p.stats()["reserved_transfer_bytes"] == 0
    assert d.stats()["reserved_blocks"] == d.stats()["reserved_transfer_bytes"] == 0
    assert_conserved(p)
    assert_conserved(d)
    with pytest.raises(StateError):
        p.acknowledge(ph)
    with pytest.raises(StateError):
        d.activate(dh)
    p.close(ph)
    d.close(dh)


def test_capacity_and_terminal_handles_are_bounded_until_close():
    p = PrefillCore(make_engine(), 1, 256)
    d = DecodeCore(make_engine(), 1, 256)
    ph, dh = start(p, d, [1, 2, 3])
    assert p.stats()["reserved_transfer_bytes"] == 256
    with pytest.raises(CapacityError, match="connection"):
        d.open()
    with pytest.raises(CapacityError, match="transfer"):
        p.open("different-session", "id", [4, 5, 6], SamplingParams(max_tokens=2))
    first_token(p, ph)
    handoff(p, d, ph, dh)
    while dh.state == "active":
        d.step()
    assert d.stats()["resident_requests"] == 0
    with pytest.raises(CapacityError, match="connection"):
        d.open()
    p.close(ph)
    d.close(dh)
    assert p.stats()["open_handles"] == d.stats()["open_handles"] == 0

    # Repeated completed connections leave no retained transaction registry.
    for _ in range(20):
        ph, dh = start(p, d, [1], params=SamplingParams(max_tokens=1))
        assert first_token(p, ph).finished
        p.close(ph)
        d.close(dh)
        assert p.stats()["open_handles"] == d.stats()["open_handles"] == 0
    assert_conserved(p)
    assert_conserved(d)


def test_block_size_one_transfers_only_the_cached_prompt_blocks():
    p, d = make_cores(block_size=1, blocks=16)
    ph, dh = start(p, d, [1, 2, 3], params=SamplingParams(max_tokens=4, ignore_eos=True))
    first_token(p, ph)
    assert ph.sequence.num_blocks == 4
    assert len(ph.sequence.block_table) == 3
    metadata, payload = handoff(p, d, ph, dh)
    assert metadata["kv"]["shape"] == [2, 2, 3, 1, 1, 4]
    assert len(payload) == metadata["kv"]["nbytes"]
    while dh.state == "active":
        d.step()
    assert dh.sequence.num_completion_tokens == 4
    assert_conserved(p)
    assert_conserved(d)
    p.close(ph)
    d.close(dh)
