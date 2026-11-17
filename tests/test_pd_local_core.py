import pytest

from myvllm.engine.sequence import SequenceStatus
from myvllm.pd import CapacityError, DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from test_pd_core import first_token, make_engine, reference_tokens, start


def conserved(core):
    manager = core.manager
    free, used = set(manager.free_block_ids), manager.used_block_ids
    assert len(free) == len(manager.free_block_ids)
    assert free.isdisjoint(used) and free | used == set(range(len(manager.blocks)))
    for block in manager.blocks:
        if block.block_id in free:
            assert block.ref_count == 0 and block.hash == -1
        else:
            assert block.ref_count > 0
    waiting, running = set(core.scheduler.waiting), set(core.scheduler.running)
    assert waiting.isdisjoint(running)
    handles = list(core._handles.values())
    owned = [block for h in handles if h.sequence for block in h.sequence.block_table]
    assert len(owned) == len(set(owned)) and set(owned) == used
    assert waiting | running == {
        h.sequence for h in handles if h.state in ('queued', 'prefill', 'active')
    }
    stats = core.stats()
    assert stats['reserved_blocks'] == sum(h._reserved_blocks for h in handles)
    assert stats['resident_requests'] == sum(bool(h._reserved_blocks) for h in handles)
    assert stats['used_blocks'] <= stats['reserved_blocks'] <= stats['total_blocks']
    assert stats['resident_requests'] == stats['local_resident_requests'] + stats['remote_resident_requests']


def test_shared_remote_reservation_local_chunk_and_real_mixed_batch():
    p = PrefillCore(make_engine(), 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=4, budget=3, sequences=3, model_length=16, chunked=True), 8, 1_000_000)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    ph, dh = start(p, d, [1, 2, 3], params=params)
    first = first_token(p, ph)
    assert d.reserve(dh, p.describe(ph))
    local = d.open_local('local', [4, 5, 6, 7, 8], params)
    conserved(d)
    assert (d.stats()['reserved_blocks'], d.stats()['used_blocks']) == (4, 0)
    assert local.sequence in d.scheduler.waiting
    with pytest.raises(CapacityError, match='block'):
        d.open_local('over-budget', [9], params)
    conserved(d)
    assert d.step() == []
    assert local.state == 'prefill' and local.sequence.num_cached_tokens == 3
    conserved(d)
    payload = p.export(ph)
    assert d.restore(dh, payload)
    assert p.acknowledge(ph)
    assert d.activate(dh)
    conserved(d)
    before = len(d.engine.model_runner.calls)
    events = d.step()
    calls = d.engine.model_runner.calls[before:]
    assert len(calls) == 2 and all(call[-1] for call in calls)
    assert {(call[0], call[1], call[2]) for call in calls} == {
        (dh.sequence.seq_id, 3, 4), (local.sequence.seq_id, 3, 5)}
    assert {(e.handle, e.completion_tokens) for e in events} == {(dh, 2), (local, 1)}
    tokens = {dh: [first.token], local: []}
    for event in events:
        tokens[event.handle].append(event.token)
    conserved(d)
    while not d.scheduler.is_finished():
        for event in d.step():
            tokens[event.handle].append(event.token)
        conserved(d)
    assert tokens[dh] == reference_tokens([1, 2, 3], 'float32')[:3]
    assert tokens[local] == reference_tokens([4, 5, 6, 7, 8], 'float32')[:3]
    remote_calls = [c for c in d.engine.model_runner.calls if c[0] == dh.sequence.seq_id]
    assert all(start >= 3 and end == start + 1 for _, start, end, _, _ in remote_calls)
    p.shutdown()
    d.shutdown()
    conserved(d)


@pytest.mark.parametrize('action', ['cancel-queued', 'expire-queued', 'cancel-partial', 'expire-partial', 'shutdown-partial'])
def test_local_lifetimes_release_once(action):
    d = DecodeCore(make_engine(budget=2, sequences=2, chunked=True), 8, 1_000_000)
    h = d.open_local('id', [1, 2, 3, 4, 5], SamplingParams(max_tokens=4), deadline=100)
    conserved(d)
    if action.endswith('partial'):
        assert d.step() == [] and h.sequence.num_cached_tokens == 2
        conserved(d)
    if action.startswith('cancel'):
        assert d.cancel(h) and not d.cancel(h)
    elif action.startswith('expire'):
        assert d.expire(now=99) == []
        assert d.expire(now=100) == [h] and d.expire(now=100) == []
    else:
        d.shutdown()
    assert h.state in ('cancelled', 'expired', 'closed')
    conserved(d)
    assert d.stats()['reserved_blocks'] == d.stats()['used_blocks'] == 0
    d.shutdown()


@pytest.mark.parametrize('forced,ignore,max_tokens,limit,reason,count', [
    (62, False, 4, 16, 'stop', 1),
    (63, False, 4, 16, 'stop', 1),
    (9, True, 1, 16, 'length', 1),
    (9, True, 8, 6, 'length', 3),
    (63, True, 3, 16, 'length', 3),
])
def test_local_samples_every_token_and_stops(forced, ignore, max_tokens, limit, reason, count):
    d = DecodeCore(make_engine(budget=2, chunked=True, forced_token=forced), 8, 1_000_000)
    h = d.open_local('id', [1, 2, 3], SamplingParams(
        temperature=0, max_tokens=max_tokens, ignore_eos=ignore, max_model_length=limit))
    events = []
    while not d.scheduler.is_finished():
        events.extend(d.step())
        conserved(d)
    assert [e.token for e in events] == [forced] * count
    assert [e.completion_tokens for e in events] == list(range(1, count + 1))
    assert events[-1].finished and events[-1].finish_reason == reason
    assert h.sequence.status == SequenceStatus.FINISHED
    assert d.stats()['reserved_blocks'] == d.stats()['used_blocks'] == 0
    d.shutdown()


def test_local_requires_chunking_and_validates_same_boundary_as_p():
    d = DecodeCore(make_engine(), 8, 1_000_000)
    with pytest.raises(ValueError, match='chunked'):
        d.open_local('id', [1], SamplingParams())
    d.shutdown()
    d = DecodeCore(make_engine(chunked=True), 8, 1_000_000)
    for request_id, tokens, params in [('', [1], SamplingParams()),
                                      ('id', [], SamplingParams()),
                                      ('id', [64], SamplingParams()),
                                      ('id', [1], SamplingParams(max_tokens=0)),
                                      ('id', [1], SamplingParams(max_model_length=1))]:
        with pytest.raises(ValueError):
            d.open_local(request_id, tokens, params)
        assert d.stats()['open_handles'] == d.stats()['reserved_blocks'] == 0
    d.shutdown()


def test_prefill_pending_tokens_track_actual_progress():
    p = PrefillCore(make_engine(budget=2, chunked=True), 8, 1_000_000)
    h = p.open('session', 'request', [1, 2, 3, 4, 5], SamplingParams(max_tokens=3))
    assert p.stats()['pending_prefill_tokens'] == 5
    assert p.stats()['admitted_requests'] == 1
    assert p.step() == [] and p.stats()['pending_prefill_tokens'] == 3
    assert p.step() == [] and p.stats()['pending_prefill_tokens'] == 1
    assert len(p.step()) == 1 and h.state == 'held'
    assert p.stats()['pending_prefill_tokens'] == 0 and p.stats()['held_requests'] == 1
    p.cancel(h)
    assert p.stats()['admitted_requests'] == 0
    p.shutdown()


def test_early_remote_reservation_and_local_share_budget():
    pe = make_engine()
    pe.config['enforce_eager'] = True
    p = PrefillCore(pe, 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=4, budget=3, sequences=3, model_length=16, chunked=True), 8, 1_000_000)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    remote = d.open()
    prepared = p.prepare(remote.session_id, 'remote', [1, 2, 3], params)
    assert d.reserve_early(remote, p.describe_early(prepared))
    local = d.open_local('local', [4, 5, 6, 7, 8], params)
    conserved(d)
    assert (d.stats()['used_blocks'], d.stats()['reserved_blocks']) == (1, 4)
    assert (d.stats()['remote_resident_requests'], d.stats()['local_resident_requests']) == (1, 1)
    with pytest.raises(CapacityError, match='block'):
        d.open_local('too-many', [9], params)
    conserved(d)
    assert d.cancel(remote)
    assert d.step() == [] and local.sequence.num_cached_tokens == 3
    conserved(d)
    d.shutdown()
    p.shutdown()
    conserved(d)
