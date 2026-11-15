import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import json
import multiprocessing as mp
import socket
import struct
import threading
import time

import pytest
import torch

from myvllm.pd.client import PDClient, PDConnection, PDRemoteError
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_network_helpers import cpu_engine, cpu_worker_process, make_worker
from test_pd_core import CPUExecution, first_token, make_cores, make_engine, reference_tokens, start


@asynccontextmanager
async def workers(**config):
    d = make_worker("decode", **config)
    de = await d.start("127.0.0.1", 0)
    p = make_worker("prefill", decode_endpoint=de, **config)
    pe = await p.start("127.0.0.1", 0)
    try:
        yield pe, de
    finally:
        await p.close()
        await d.close()


async def empty(endpoint):
    for _ in range(200):
        stats = await PDClient.stats(endpoint)
        if stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0:
            assert stats["queued_events"] == 0
            if "reserved_blocks" in stats:
                assert stats["reserved_blocks"] == stats["resident_requests"] == 0
            return stats
        await asyncio.sleep(.01)
    pytest.fail(f"resources did not drain: {stats}")


async def pair(pe, de, prompt, params=None, lifetime=3):
    params = params or SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    d = await PDClient.open_d(de, lifetime=lifetime)
    try:
        p = await PDClient.open_p(pe, d.session_id, "same-request-id", prompt, params,
                                  lifetime=lifetime)
    except BaseException:
        await d.close()
        raise
    return p, d


async def finish_pair(pe, de, prompt, params=None):
    p, d = await pair(pe, de, prompt, params)
    try:
        first = await p.next_token()
        tokens = [first["token"]]
        if not first["finished"]:
            await PDClient.handoff(p, d, await p.describe())
            while True:
                event = await d.next_token()
                tokens.append(event["token"])
                if event["finished"]:
                    break
        return tokens
    finally:
        await p.close()
        await d.close()


def test_info_concurrent_requests_and_large_max_tokens():
    async def run():
        async with workers() as (pe, de):
            pi, di = await PDClient.info(pe), await PDClient.info(de)
            assert pi["role"] == "prefill" and di["role"] == "decode"
            assert pi["model"] == di["model"]
            assert pi["sampling_seed"] != di["sampling_seed"]
            assert pi["max_completion_tokens"] == pi["max_model_length"] - 1
            params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
            prompts = [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
            results = await asyncio.gather(*(finish_pair(pe, de, prompt, params) for prompt in prompts))
            assert results == [reference_tokens(prompt, "float32") for prompt in prompts]
            big_budget = SamplingParams(temperature=0, max_tokens=1000, ignore_eos=True,
                                        max_model_length=8)
            result = await finish_pair(pe, de, [3, 4, 5], big_budget)
            assert len(result) == 5
            await empty(pe)
            await empty(de)
    asyncio.run(run())


def test_duplicates_and_cancel_before_draining_first_event():
    async def run():
        async with workers() as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3])
            try:
                await p.next_token()
                metadata = await p.describe()
                assert await d.reserve(metadata)
                assert not await d.reserve(metadata)
                payload = await p.export()
                assert await p.export() == payload
                assert await d.restore(payload)
                assert not await d.restore(payload)
                assert await p.acknowledge()
                assert not await p.acknowledge()
                assert await d.activate()
                assert not await d.activate()
                while not (await d.next_token())["finished"]:
                    pass
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            await empty(de)
            p, d = await pair(pe, de, [1, 2], SamplingParams(temperature=0, max_tokens=1))
            try:
                for _ in range(200):
                    if (await PDClient.stats(pe))["queued_events"]:
                        break
                    await asyncio.sleep(.005)
                else:
                    pytest.fail("first token never reached network queue")
                assert await p.cancel()
                with pytest.raises(PDRemoteError, match="cancelled"):
                    await p.next_token()
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            await empty(de)
    asyncio.run(run())


def test_duplicate_malformed_metadata_preserves_core_transaction():
    p, d = make_cores()
    ph, dh = start(p, d, [1, 2, 3])
    try:
        first_token(p, ph)
        metadata = p.describe(ph)
        assert d.reserve(dh, metadata)
        bad = deepcopy(metadata)
        bad["sampling_seed"] = False
        with pytest.raises(ValueError, match="sampling_seed"):
            d.reserve(dh, bad)
        assert dh.state == "reserved"
        assert not d.reserve(dh, metadata)
    finally:
        p.shutdown()
        d.shutdown()


def test_malformed_duplicate_and_model_type_reject_and_release():
    async def run():
        async with workers() as (pe, de):
            for duplicate in (False, True):
                p, d = await pair(pe, de, [1, 2, 3])
                try:
                    await p.next_token()
                    metadata = await p.describe()
                    if duplicate:
                        assert await d.reserve(metadata)
                        bad = deepcopy(metadata)
                        bad["sampling_seed"] = False
                    else:
                        bad = deepcopy(metadata)
                        bad["model"]["architecture"]["num_kv_heads"] = True
                    with pytest.raises(PDRemoteError):
                        await d.reserve(bad)
                finally:
                    await p.close()
                    await d.close()
                await empty(pe)
                await empty(de)
            assert len(await finish_pair(pe, de, [3, 4])) == 6
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["held", "reserved", "restored", "active"])
def test_cancel_phases_and_survivor(phase):
    async def run():
        async with workers(delay=.01) as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3, 4, 5])
            try:
                await p.next_token()
                if phase in ("reserved", "restored", "active"):
                    await d.reserve(await p.describe())
                if phase in ("restored", "active"):
                    await d.restore(await p.export())
                if phase == "active":
                    await p.acknowledge()
                    await d.activate()
                await p.cancel()
                await d.cancel()
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [2])) == 6
    asyncio.run(run())


def test_cancel_request_queued_behind_held_blocks():
    async def run():
        async with workers(blocks=2, budget=8, sequences=2, model_length=8) as (pe, de):
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            pairs = []
            try:
                for prompt in ([1, 2, 3, 4], [5, 6, 7, 8]):
                    p, d = await pair(pe, de, prompt, params)
                    pairs.append((p, d))
                    assert not (await p.next_token())["finished"]
                queued_p, queued_d = await pair(pe, de, [9], params)
                pairs.append((queued_p, queued_d))
                stats = await PDClient.stats(pe)
                assert stats["open_handles"] == 3
                assert stats["held_requests"] == stats["used_blocks"] == 2
                assert await queued_p.cancel()
                assert await queued_d.cancel()
                stats = await PDClient.stats(pe)
                assert stats["held_requests"] == stats["used_blocks"] == 2
                p, d = pairs[0]
                await PDClient.handoff(p, d, await p.describe())
                assert not (await d.next_token())["finished"]
                assert (await d.next_token())["finished"]
            finally:
                for p, d in pairs:
                    await p.close()
                    await d.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [2], params)) == 3
    asyncio.run(run())


def test_timeout_held_and_reserved_reclaims_idle_resources():
    async def run():
        async with workers() as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3], lifetime=.3)
            await p.next_token()
            await d.reserve(await p.describe())
            await asyncio.sleep(.45)
            await empty(pe)
            await empty(de)
            await p.close()
            await d.close()
            assert len(await finish_pair(pe, de, [2])) == 6
    asyncio.run(run())


def test_short_payload_real_eof_and_oversized_header_recover():
    async def run():
        async with workers() as (pe, de):
            reader, writer = await asyncio.open_connection(*de)
            writer.write(struct.pack("!IQ", DEFAULT_HEADER_LIMIT + 1, 0))
            await writer.drain()
            header, n = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            assert not header["ok"] and n == 0
            writer.close()
            await writer.wait_closed()

            reader, writer = await asyncio.open_connection(*de)
            await write_frame(writer, {"op": "open_d", "lifetime": 3})
            response, n = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            assert n == 0 and response["ok"]
            session = response["value"]["session_id"]
            p = await PDClient.open_p(pe, session, "truncated", [1, 2, 3],
                                      SamplingParams(temperature=0, max_tokens=4), lifetime=3)
            try:
                await p.next_token()
                metadata = await p.describe()
                await write_frame(writer, {"op": "reserve", "metadata": metadata})
                response, n = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert n == 0 and response["ok"]
                body = await p.export()
                header = json.dumps({"op": "restore"}).encode()
                writer.write(struct.pack("!IQ", len(header), len(body)) + header + body[:3])
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            finally:
                await p.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [5])) == 6
    asyncio.run(run())


def test_closed_during_queued_open_releases_capacity():
    async def run():
        async with workers(delay=.6) as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3])
            queued_d = await PDClient.open_d(de, lifetime=3)
            try:
                first = asyncio.create_task(p.next_token())
                reader, writer = await asyncio.open_connection(*pe)
                await write_frame(writer, {"op": "open_p", "lifetime": 3,
                                          "session_id": queued_d.session_id, "request_id": "abandoned",
                                          "token_ids": [2], "sampling": {"temperature": 0,
                                          "max_tokens": 3, "ignore_eos": True,
                                          "max_model_length": None}})
                await asyncio.sleep(.05)
                # A rejected second claim proves the first open reached D while P is busy.
                with pytest.raises(PDRemoteError, match="claim"):
                    await PDClient.open_p(pe, queued_d.session_id, "duplicate", [2],
                                          SamplingParams(temperature=0, max_tokens=3), lifetime=3)
                assert not first.done()
                writer.close()
                await writer.wait_closed()
                await first
            finally:
                await p.close()
                await d.close()
                await queued_d.close()
            await empty(pe)
            await empty(de)
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            assert len(await finish_pair(pe, de, [2], params)) == 3
    asyncio.run(run())


def _start_process(ctx, role, decode_endpoint=None, port=0):
    ready, child = ctx.Pipe(duplex=False)
    stop = ctx.Event()
    process = ctx.Process(target=cpu_worker_process, args=(role, {}, child, stop, decode_endpoint, port))
    process.start()
    child.close()
    assert ready.poll(30)
    endpoint = ready.recv()
    ready.close()
    return process, stop, endpoint


def _stop_process(process, stop):
    if process.is_alive():
        stop.set()
        process.join(15)
    if process.is_alive():
        process.terminate()
        process.join(10)
    assert not process.is_alive()


def test_killed_prefill_process_decode_survives_and_prefill_restarts():
    ctx = mp.get_context("spawn")
    dproc, dstop, de = _start_process(ctx, "decode")
    pproc, pstop, pe = _start_process(ctx, "prefill", de)
    replacement = None
    try:
        async def disconnect():
            p, d = await pair(pe, de, [1, 2, 3])
            await p.next_token()
            await d.reserve(await p.describe())
            pproc.kill()
            pproc.join(10)
            with pytest.raises((ConnectionError, OSError, asyncio.IncompleteReadError)):
                await p.export()
            await p.close()
            await d.close()
            await empty(de)
        asyncio.run(disconnect())
        replacement = _start_process(ctx, "prefill", de)
        async def survivor():
            assert len(await finish_pair(replacement[2], de, [1, 2])) == 6
            await empty(replacement[2])
            await empty(de)
        asyncio.run(survivor())
    finally:
        _stop_process(pproc, pstop)
        _stop_process(dproc, dstop)
        if replacement:
            _stop_process(replacement[0], replacement[1])


def test_connection_and_decode_reservation_capacity_recover():
    async def run():
        async with workers(sequences=1) as (pe, de):
            held = [await PDClient.open_d(de, lifetime=3) for _ in range(4)]
            try:
                assert (await PDClient.stats(de))["open_handles"] == 4
                with pytest.raises(PDRemoteError, match="capacity"):
                    await PDClient.open_d(de, lifetime=3)
            finally:
                for connection in held:
                    await connection.close()
            await empty(de)

            p1, d1 = await pair(pe, de, [1, 2, 3])
            p2, d2 = await pair(pe, de, [4, 5, 6])
            try:
                await p1.next_token()
                await p2.next_token()
                assert await d1.reserve(await p1.describe())
                with pytest.raises(PDRemoteError, match="capacity"):
                    await d2.reserve(await p2.describe())
                stats = await PDClient.stats(de)
                assert stats["resident_requests"] == 1 and stats["reserved_blocks"] > 0
            finally:
                for p, d in ((p1, d1), (p2, d2)):
                    await p.close()
                    await d.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [3])) == 6
    asyncio.run(run())


def test_cancel_client_waiting_for_owner_does_not_strand_admission():
    async def run():
        async with workers(delay=.2) as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3])
            wait = asyncio.create_task(p.next_token())
            await asyncio.sleep(.02)
            wait.cancel()
            with pytest.raises(asyncio.CancelledError):
                await wait
            await p.close()
            await d.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [1])) == 6
    asyncio.run(run())


def test_socket_closed_after_complete_restore_body_reclaims_reservation():
    async def run():
        async with workers(delay=.05) as (pe, de):
            reader, writer = await asyncio.open_connection(*de)
            await write_frame(writer, {"op": "open_d", "lifetime": 3})
            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            p = await PDClient.open_p(pe, response["value"]["session_id"], "restore-eof", [1, 2, 3],
                                      SamplingParams(temperature=0, max_tokens=6), lifetime=3)
            try:
                await p.next_token()
                metadata = await p.describe()
                await write_frame(writer, {"op": "reserve", "metadata": metadata})
                response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert response["ok"]
                body = await p.export()
                await write_frame(writer, {"op": "restore"}, body)
                writer.close()
                await writer.wait_closed()
            finally:
                await p.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [3])) == 6
    asyncio.run(run())


def test_timing_records_transfer_completion_and_releases(tmp_path):
    async def run():
        worker_log = tmp_path / "workers.jsonl"
        client_log = tmp_path / "client.jsonl"
        async with workers(timing_path=str(worker_log)) as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3])
            try:
                await p.next_token()
                metadata = await p.describe()
                await PDClient.handoff(p, d, metadata, timing_path=str(client_log))
                while not (await d.next_token())["finished"]:
                    pass
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            await empty(de)
        records = [json.loads(line) for line in worker_log.read_text().splitlines()]
        names = {record["event"] for record in records}
        assert {"pd_admitted", "pd_first", "pd_export_done", "pd_restore_done",
                "pd_p_released", "pd_second", "pd_finished", "pd_released"} <= names
        transaction = [record for record in records if record["request_id"] == "same-request-id"]
        assert {record["transaction_id"] for record in transaction} == {metadata["transaction_id"]}
        transfers = [record for record in transaction if record["event"] in ("pd_export_done", "pd_restore_done")]
        assert all(record["payload_bytes"] == metadata["kv"]["nbytes"] for record in transfers)
        assert all(record.get("export_ms", record.get("restore_ms")) >= 0 for record in transfers)
        transfer = json.loads(client_log.read_text())
        assert transfer["payload_bytes"] == metadata["kv"]["nbytes"] and transfer["transfer_ms"] >= 0
        assert transfer["session_id"] == metadata["session_id"]
    asyncio.run(run())


def test_session_cannot_be_claimed_by_second_prefill_connection():
    async def run():
        async with workers() as (pe, de):
            p, d = await pair(pe, de, [1, 2, 3])
            try:
                await p.next_token()
                await p.close()
                with pytest.raises(PDRemoteError, match="claim"):
                    await PDClient.open_p(pe, d.session_id, "replay", [1, 2, 3],
                                          SamplingParams(temperature=0, max_tokens=3), lifetime=3)
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            await empty(de)
    asyncio.run(run())


def test_concurrent_claims_while_decode_owner_is_busy():
    async def run():
        async with workers(delay=.3) as (pe, de):
            assert (await PDClient.info(de))["max_socket_connections"] == 9
            ds = [await PDClient.open_d(de, lifetime=5) for _ in range(4)]
            ps = []
            try:
                params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
                first = await PDClient.open_p(pe, ds[0].session_id, "active", [1, 2, 3],
                                              params, lifetime=5)
                ps.append(first)
                await first.next_token()
                await PDClient.handoff(first, ds[0], await first.describe())
                claims = [asyncio.create_task(PDClient.open_p(pe, d.session_id, f"claim-{i}",
                                                              [i + 1], params, lifetime=5))
                          for i, d in enumerate(ds[1:])]
                status = asyncio.create_task(PDClient.stats(de))
                await asyncio.sleep(.05)
                assert not status.done()
                assert all(not task.done() for task in claims)
                ps.extend(await asyncio.gather(*claims))
                assert (await status)["open_handles"] == 4
                for p in ps[1:]:
                    assert not (await p.next_token())["finished"]
            finally:
                for p in ps:
                    await p.close()
                for d in ds:
                    await d.close()
            await empty(pe)
            await empty(de)
    asyncio.run(run())


def test_decode_restart_rejects_stale_session_and_prefill_starts_offline():
    ctx = mp.get_context("spawn")
    dproc, dstop, de = _start_process(ctx, "decode")
    pproc, pstop, pe = _start_process(ctx, "prefill", de)
    offline_p = None
    replacement_d = None
    try:
        async def kill_decode():
            p, d = await pair(pe, de, [1, 2, 3])
            old_session = d.session_id
            try:
                assert not (await p.next_token())["finished"]
                dproc.kill()
                dproc.join(10)
                assert dproc.exitcode is not None
            finally:
                await p.close()
                await d.close()
            await empty(pe)
            return old_session

        old_session = asyncio.run(kill_decode())
        offline_p = _start_process(ctx, "prefill", de)

        async def reject_unavailable():
            with pytest.raises(PDRemoteError, match="claim"):
                await PDClient.open_p(offline_p[2], old_session, "unavailable", [1],
                                      SamplingParams(temperature=0, max_tokens=3), lifetime=3)
            await empty(offline_p[2])

        asyncio.run(reject_unavailable())
        replacement_d = _start_process(ctx, "decode", port=de[1])
        assert replacement_d[2] == de

        async def restart():
            with pytest.raises(PDRemoteError, match="claim"):
                await PDClient.open_p(pe, old_session, "stale", [1],
                                      SamplingParams(temperature=0, max_tokens=3), lifetime=3)
            assert len(await finish_pair(pe, de, [1, 2])) == 6
            assert len(await finish_pair(offline_p[2], de, [3, 4])) == 6
            await empty(pe)
            await empty(offline_p[2])
            await empty(de)

        asyncio.run(restart())
    finally:
        _stop_process(pproc, pstop)
        _stop_process(dproc, dstop)
        if offline_p:
            _stop_process(offline_p[0], offline_p[1])
        if replacement_d:
            _stop_process(replacement_d[0], replacement_d[1])


def test_cancelled_start_releases_owner_thread():
    def slow_start(config):
        time.sleep(.4)
        return cpu_engine(config)

    async def run():
        before = set(threading.enumerate())
        worker = PDWorker("prefill", {"world_size": 1}, max_connections=2,
                          max_transfer_bytes=1_000_000, request_timeout=5,
                          decode_endpoint=("127.0.0.1", 1), engine_factory=slow_start)
        try:
            starting = asyncio.create_task(worker.start("127.0.0.1", 0))
            await asyncio.sleep(.05)
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
            await asyncio.sleep(.5)
            assert not [thread for thread in threading.enumerate()
                        if thread not in before and thread.name == "pd-model-owner"]
        finally:
            await worker.close()
    asyncio.run(run())


def test_shutdown_during_disconnected_admission_releases_capacity():
    async def run():
        dworker = make_worker("decode")
        de = await dworker.start("127.0.0.1", 0)
        pworker = PDWorker("prefill", {"world_size": 1, "delay": .8}, max_connections=2,
                           max_transfer_bytes=1_000_000, request_timeout=5,
                           decode_endpoint=de, engine_factory=cpu_engine)
        pe = await pworker.start("127.0.0.1", 0)
        connections = []
        try:
            d1, d2 = await asyncio.gather(*(PDClient.open_d(de, lifetime=5) for _ in range(2)))
            connections.extend([d1, d2])
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            p1 = await PDClient.open_p(pe, d1.session_id, "busy", [1, 2, 3], params, lifetime=5)
            connections.append(p1)
            first = asyncio.create_task(p1.next_token())
            reader, writer = await asyncio.open_connection(*pe)
            await write_frame(writer, {"op": "open_p", "lifetime": 5,
                                      "session_id": d2.session_id, "request_id": "abandoned",
                                      "token_ids": [2], "sampling": {"temperature": 0,
                                      "max_tokens": 3, "ignore_eos": True,
                                      "max_model_length": None}})
            await asyncio.sleep(.05)
            assert not first.done()
            writer.write_eof()
            await writer.drain()
            await asyncio.sleep(.05)
            await pworker.close()
            assert await asyncio.wait_for(reader.read(1), .3) == b""
            writer.close()
            await writer.wait_closed()
            await asyncio.gather(first, return_exceptions=True)
            for connection in connections:
                await connection.close()
            connections.clear()
            await empty(de)

            pe = await pworker.start("127.0.0.1", 0)
            d1, d2 = await asyncio.gather(*(PDClient.open_d(de, lifetime=5) for _ in range(2)))
            connections.extend([d1, d2])
            p1 = await PDClient.open_p(pe, d1.session_id, "after-1", [1], params, lifetime=5)
            connections.append(p1)
            p2 = await PDClient.open_p(pe, d2.session_id, "after-2", [2], params, lifetime=5)
            connections.append(p2)
            assert (await PDClient.stats(pe))["open_handles"] == 2
        finally:
            for connection in connections:
                await connection.close()
            await pworker.close()
            await dworker.close()
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["close", "cancel"])
def test_client_control_while_next_waits_for_capacity(operation):
    async def run():
        async with workers(blocks=2, budget=8, sequences=2, model_length=8) as (pe, de):
            params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            pairs = []
            waiting = closing = None
            try:
                for prompt in ([1, 2, 3, 4], [5, 6, 7, 8]):
                    p, d = await pair(pe, de, prompt, params)
                    pairs.append((p, d))
                    assert not (await p.next_token())["finished"]
                p, d = await pair(pe, de, [9], params)
                pairs.append((p, d))
                waiting = asyncio.create_task(p.next_token())
                await asyncio.sleep(.05)
                assert not waiting.done()
                stats = await PDClient.stats(pe)
                assert stats["held_requests"] == stats["used_blocks"] == 2
                if operation == "close":
                    closing = asyncio.create_task(p.close())
                    await asyncio.wait_for(asyncio.shield(closing), .3)
                    with pytest.raises((ConnectionError, OSError, asyncio.IncompleteReadError)):
                        await waiting
                else:
                    with pytest.raises(RuntimeError, match="in.flight"):
                        await asyncio.wait_for(p.cancel(), .3)
                    assert not waiting.done()
            finally:
                for task in (waiting, closing):
                    if task and not task.done():
                        task.cancel()
                await asyncio.gather(*(task for task in (waiting, closing) if task),
                                     return_exceptions=True)
                for p, d in pairs:
                    await p.close()
                    await d.close()
            await empty(pe)
            await empty(de)
            assert len(await finish_pair(pe, de, [2], params)) == 3
    asyncio.run(run())


def test_timed_out_slow_export_aborts_buffered_body():
    def big_engine(config):
        engine = make_engine(blocks=8, budget=32, sequences=2, model_length=32)
        engine.config["head_dim"] = 32768
        engine.model_runner = CPUExecution(engine.config)
        engine.exit = lambda: None
        return engine

    async def run():
        dworker = PDWorker("decode", {"world_size": 1}, max_connections=1,
                           max_transfer_bytes=16 * 1024 * 1024, request_timeout=5,
                           engine_factory=big_engine)
        de = await dworker.start("127.0.0.1", 0)
        pworker = PDWorker("prefill", {"world_size": 1}, max_connections=1,
                           max_transfer_bytes=16 * 1024 * 1024, request_timeout=2,
                           decode_endpoint=de, engine_factory=big_engine)
        pe = await pworker.start("127.0.0.1", 0)
        d = await PDClient.open_d(de, lifetime=4)
        before = set(asyncio.all_tasks())
        reader, writer = await asyncio.open_connection(*pe)
        writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        try:
            await write_frame(writer, {"op": "open_p", "lifetime": 2,
                                      "session_id": d.session_id, "request_id": "slow-reader",
                                      "token_ids": [1] * 16,
                                      "sampling": {"temperature": 0, "max_tokens": 3,
                                                   "ignore_eos": True, "max_model_length": None}})
            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            assert response["ok"]
            await write_frame(writer, {"op": "next"})
            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            assert response["ok"] and not response["value"]["finished"]
            accepts = [task for task in asyncio.all_tasks() - before
                       if task.get_coro().__qualname__ == "PDWorker._accept"]
            assert len(accepts) == 1
            accept = accepts[0]
            await write_frame(writer, {"op": "describe"})
            response, _ = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
            metadata = response["value"]
            assert await d.reserve(metadata)
            await write_frame(writer, {"op": "export"})
            await asyncio.sleep(2.3)
            await empty(pe)
            await asyncio.wait_for(asyncio.shield(accept), 1)
            await d.close()
            await empty(de)
            p, next_d = await pair(pe, de, [1],
                                   SamplingParams(temperature=0, max_tokens=1), lifetime=1.5)
            try:
                assert (await p.next_token())["finished"]
            finally:
                await p.close()
                await next_d.close()
        finally:
            writer.transport.abort()
            await writer.wait_closed()
            await d.close()
            await pworker.close()
            await dworker.close()

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        asyncio.run(run())
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("operation", ["cancel_task", "close", "timeout"])
def test_client_aborts_buffered_restore_when_operation_ends(operation):
    async def run():
        received = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def slow_peer(reader, writer):
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            try:
                header, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 16 * 1024 * 1024)
                assert header == {"op": "restore"} and size == 16 * 1024 * 1024
                received.set()
                await release.wait()
            finally:
                writer.transport.abort()
                await writer.wait_closed()
                finished.set()

        server = await asyncio.start_server(slow_peer, "127.0.0.1", 0)
        reader, writer = await asyncio.open_connection(*server.sockets[0].getsockname()[:2])
        deadline = asyncio.get_running_loop().time() + (1 if operation == "timeout" else 10)
        connection = PDConnection(reader, writer, deadline)
        restoring = asyncio.create_task(connection.restore(b"x" * (16 * 1024 * 1024)))
        try:
            await asyncio.wait_for(received.wait(), 2)
            await asyncio.sleep(.05)
            assert writer.transport.get_write_buffer_size() > 0
            if operation == "cancel_task":
                restoring.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await restoring
                await connection.close()
            elif operation == "close":
                await asyncio.wait_for(connection.close(), .5)
                with pytest.raises((ConnectionError, OSError, asyncio.IncompleteReadError)):
                    await restoring
            else:
                with pytest.raises(TimeoutError):
                    await restoring
                await connection.close()
            assert writer.transport.get_write_buffer_size() == 0
        finally:
            writer.transport.abort()
            await writer.wait_closed()
            if not restoring.done():
                restoring.cancel()
            await asyncio.gather(restoring, return_exceptions=True)
            release.set()
            server.close()
            await server.wait_closed()
            await asyncio.wait_for(finished.wait(), 2)
    asyncio.run(run())
