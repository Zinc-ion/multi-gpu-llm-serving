import asyncio
import sys
import threading

import pytest

from myvllm.pd import CapacityError, DecodeCore, PrefillCore
from myvllm.pd.client import PDClient
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_network_helpers import cpu_engine
from test_pd_core import first_token, handoff, make_engine, start
from test_pd_local_core import conserved


@pytest.mark.parametrize("action", ["finish", "cancel-queued", "cancel-partial", "cancel-active",
                                    "expire", "close", "shutdown"])
def test_local_resident_count_tracks_each_release_and_rejected_admission(action):
    p = PrefillCore(make_engine(), 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=5, budget=3, sequences=3, chunked=True), 8, 1_000_000)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    ph, remote = start(p, d, [1, 2, 3], params=params)
    try:
        first_token(p, ph)
        assert d.reserve(remote, p.describe(ph))
        local = d.open_local("local", [4, 5, 6, 7, 8], params, deadline=100)
        stats = d.stats()
        assert (stats["local_resident_requests"], stats["remote_resident_requests"]) == (1, 1)
        with pytest.raises(CapacityError):
            d.open_local("over-budget", [9, 10, 11, 12], params)
        assert (d.stats()["local_resident_requests"], d.stats()["remote_resident_requests"]) == (1, 1)
        conserved(d)
        if action in ("finish", "cancel-partial", "cancel-active", "expire", "close", "shutdown"):
            assert d.step() == []
            assert local.state == "prefill"
            conserved(d)
        if action == "cancel-active":
            assert d.step()[0].completion_tokens == 1
            assert local.state == "active"
            conserved(d)
        if action == "finish":
            events = []
            while not d.scheduler.is_finished():
                events.extend(d.step())
                conserved(d)
            assert [event.completion_tokens for event in events] == [1, 2, 3]
            assert events[-1].finished
        elif action.startswith("cancel"):
            assert d.cancel(local) and not d.cancel(local)
        elif action == "expire":
            assert d.expire(now=100) == [local]
            assert d.expire(now=100) == []
        elif action == "close":
            assert d.close(local) and not d.close(local)
        else:
            d.shutdown()
        stats = d.stats()
        assert stats["local_resident_requests"] == 0
        assert stats["remote_resident_requests"] == (0 if action == "shutdown" else 1)
        conserved(d)
        if action != "shutdown":
            d.close(local)
            assert d.cancel(remote)
            assert d.stats()["resident_requests"] == 0
    finally:
        d.shutdown()
        p.shutdown()
    conserved(d)


def test_mixed_residents_complete_independently_after_rejected_admission():
    p = PrefillCore(make_engine(), 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=12, budget=3, sequences=3, chunked=True), 8, 1_000_000)
    remote_params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    ph, remote = start(p, d, [1, 2], params=remote_params)
    try:
        first_token(p, ph)
        handoff(p, d, ph, remote)
        short = d.open_local("short", [3], SamplingParams(max_tokens=1, ignore_eos=True))
        long = d.open_local("long", [4, 5, 6, 7, 8], SamplingParams(max_tokens=2, ignore_eos=True))
        stats = d.stats()
        assert (stats["local_resident_requests"], stats["remote_resident_requests"]) == (2, 1)
        with pytest.raises(CapacityError, match="resident"):
            d.open_local("over-limit", [9], SamplingParams(max_tokens=1))
        assert d.stats()["open_handles"] == 3
        conserved(d)

        events = d.step()
        assert {(e.handle, e.completion_tokens, e.finished) for e in events} == {
            (remote, 2, False), (short, 1, True)}
        assert long.state == "prefill"
        stats = d.stats()
        assert (stats["local_resident_requests"], stats["remote_resident_requests"]) == (1, 1)
        conserved(d)
        events = []
        while not d.scheduler.is_finished():
            events.extend(d.step())
            conserved(d)
        assert [(e.completion_tokens, e.finished) for e in events if e.handle is remote] == [
            (3, False), (4, True)]
        assert [(e.completion_tokens, e.finished) for e in events if e.handle is long] == [
            (1, False), (2, True)]
        stats = d.stats()
        assert (stats["local_resident_requests"], stats["remote_resident_requests"]) == (0, 0)
    finally:
        d.shutdown()
        p.shutdown()
    conserved(d)


@pytest.mark.parametrize("chunked", [False, True])
def test_pure_remote_step_work_does_not_grow_with_running_sequences(chunked):
    p = PrefillCore(make_engine(blocks=16, sequences=8), 8, 1_000_000)
    d = DecodeCore(make_engine(blocks=16, sequences=8, chunked=chunked), 8, 1_000_000)
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    remotes = []
    try:
        for token in range(1, 9):
            ph, remote = start(p, d, [token], params=params)
            first_token(p, ph)
            handoff(p, d, ph, remote)
            remotes.append(remote)

        def traced_step():
            lines = 0

            def trace(frame, event, arg):
                nonlocal lines
                if frame.f_code is DecodeCore.step.__code__:
                    if event == "line":
                        lines += 1
                    return trace
                return None

            previous = sys.gettrace()
            sys.settrace(trace)
            try:
                events = d.step()
            finally:
                sys.settrace(previous)
            return lines, events

        many_lines, events = traced_step()
        assert len(events) == len(d.scheduler.running) == 8
        assert all(e.completion_tokens == 2 and not e.finished for e in events)
        for remote in remotes[1:]:
            assert d.cancel(remote)
        one_lines, events = traced_step()
        assert len(events) == len(d.scheduler.running) == 1
        assert events[0].completion_tokens == 3 and not events[0].finished
        assert many_lines == one_lines
        conserved(d)
    finally:
        d.shutdown()
        p.shutdown()


def test_terminal_worker_stats_work_does_not_grow_with_open_handles():
    records = []
    thread_state = threading.local()
    stats_code = DecodeCore.stats.__code__

    def trace(frame, event, arg):
        if event == "call":
            if frame.f_code is stats_code:
                thread_state.record = [frame.f_back.f_code.co_name, 0]
                return trace
            if getattr(thread_state, "record", None) is not None and frame.f_code.co_filename == stats_code.co_filename:
                return trace
            return None
        record = getattr(thread_state, "record", None)
        if record is not None and event == "line":
            record[1] += 1
        elif record is not None and event == "return" and frame.f_code is stats_code:
            records.append(tuple(record))
            thread_state.record = None
        return trace

    async def run():
        config = {"world_size": 1, "blocks": 16, "budget": 16, "sequences": 8, "model_length": 16}
        d = PDWorker("decode", config, max_connections=8, max_transfer_bytes=1_000_000,
                     request_timeout=30, engine_factory=cpu_engine)
        de = await d.start("127.0.0.1", 0)
        try:
            assert (await PDClient.stats(de))["open_handles"] == 0
            p = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                         request_timeout=30, decode_endpoint=de, engine_factory=cpu_engine)
            pe = await p.start("127.0.0.1", 0)
            connections = []
            try:
                params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True)
                for i in range(8):
                    remote = await PDClient.open_d(de, lifetime=25)
                    connections.append(remote)
                    prefill = await PDClient.open_p(pe, remote.session_id, str(i), [i + 1], params, lifetime=25)
                    connections.append(prefill)
                    first = await prefill.next_token()
                    assert first["completion_tokens"] == 1 and not first["finished"]
                    assert await remote.reserve(await prefill.describe())
                    assert await remote.restore(await prefill.export())
                    assert await prefill.acknowledge()
                assert (await PDClient.stats(de))["open_handles"] == 8
                remotes = connections[::2]
                await asyncio.gather(*(remote.activate() for remote in remotes))
                events = await asyncio.gather(*(remote.next_token() for remote in remotes))
                assert all(e["completion_tokens"] == 2 and e["finished"] for e in events)
                baseline = [lines for caller, lines in records if caller == "_stats"][0]
                terminal = [lines for caller, lines in records if caller == "_step"]
                assert len(terminal) == 8
                assert terminal == [baseline] * 8
            finally:
                try:
                    await asyncio.gather(*(connection.close() for connection in connections))
                finally:
                    await p.close()
        finally:
            await d.close()

    previous = threading.gettrace()
    threading.settrace(trace)
    try:
        asyncio.run(run())
    finally:
        threading.settrace(previous)
