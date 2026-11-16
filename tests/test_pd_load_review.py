import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json

import pytest

from myvllm.pd.client import PDClient
from myvllm.pd.http import GatewayError
from myvllm.pd.ready import check
from myvllm.pd.wire import read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from tools.analyze_pd import analyze
from test_pd_load_bookings import pending
from test_pd_load_routing import elastic, http_gateway, layered_engine, load_engine, nodes, records, settings, wait_sample
from test_pd_server import body, drained, make_config


@asynccontextmanager
async def hold_first_response(target, operation):
    tasks = set()
    entered, release = asyncio.Event(), asyncio.Event()
    held = False

    async def relay(reader, writer):
        nonlocal held
        task = asyncio.current_task()
        tasks.add(task)
        upstream = None
        try:
            upstream_reader, upstream = await asyncio.open_connection(*target)
            while True:
                header, size = await read_header(reader, 1_000_000, 1_000_000)
                await write_frame(upstream, header, await read_body(reader, size))
                response, size = await read_header(upstream_reader, 1_000_000, 1_000_000)
                payload = await read_body(upstream_reader, size)
                if header["op"] == operation and not held:
                    held = True
                    entered.set()
                    await release.wait()
                await write_frame(writer, response, payload)
                if header["op"] in ("info", "stats", "close"):
                    break
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            if upstream is not None:
                upstream.close()
            writer.close()
            await asyncio.gather(*(stream.wait_closed() for stream in (upstream, writer)
                                   if stream is not None), return_exceptions=True)
            tasks.discard(task)

    server = await asyncio.start_server(relay, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[:2], entered, release
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("pipeline", [False, True])
def test_reserve_ack_clears_booking_after_a_sample_during_handoff(tmp_path, pipeline):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        factory = layered_engine if pipeline else load_engine
        frame = 64 if pipeline else 0
        decode = PDWorker("decode", {**config, "sequences": 2}, max_connections=8,
                          max_transfer_bytes=1_000_000, request_timeout=6,
                          pipeline_frame_bytes=frame, engine_factory=factory)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                           request_timeout=6, decode_endpoint=de,
                           pipeline_frame_bytes=frame, engine_factory=factory)
        pe = await prefill.start("127.0.0.1", 0)
        try:
            async with hold_first_response(pe, "start" if pipeline else "export") as (proxy, entered, release):
                cfg = settings([(proxy, de)], poll_interval=.01, pipeline=2 if pipeline else 0)
                async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                    first = asyncio.create_task(client.post(
                        "/v1/completions", json=body(length=4), headers={"x-request-id": "first"}))
                    try:
                        await asyncio.wait_for(entered.wait(), 3)
                        _, sampled_at = await wait_sample(gateway, de, lambda row: row["resident_requests"] == 1)
                        async with asyncio.timeout(2):
                            while gateway.samples[de][1] <= sampled_at:
                                await asyncio.sleep(.005)
                        active = next(entry for entry in gateway.requests.values()
                                      if entry.pending.request_id == "first")
                        assert active.d_admitted
                        assert gateway.d_bookings[de]["requests"] == 0

                        assert gateway.claim()
                        booked = gateway.submit(pending("new-booking"))
                        try:
                            assert gateway.d_bookings[de]["requests"] == 1
                            assert gateway.claim()
                            try:
                                with pytest.raises(GatewayError, match="capacity"):
                                    gateway.submit(pending("exceeds-capacity"))
                            finally:
                                gateway.unclaim()
                        finally:
                            gateway.finish(booked.seq_id, False)
                        await asyncio.sleep(0)
                        await asyncio.sleep(0)

                        second = await client.post(
                            "/v1/completions", json=body(length=4),
                            headers={"x-request-id": "second"})
                        assert second.status_code == 200, second.text
                        assert not first.done()
                    finally:
                        release.set()
                        await asyncio.gather(first, return_exceptions=True)
                    assert first.result().status_code == 200
                    await drained(gateway, [(pe, de)])
        finally:
            await prefill.close()
            await decode.close()

    asyncio.run(run())


def test_unassigned_http_rejections_are_in_gateway_analysis(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 1}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                decode = await PDClient.open_d(ds[0], lifetime=5)
                prefill = await PDClient.open_p(
                    ps[0], decode.session_id, "external", [1, 2],
                    pending("external").params, lifetime=5)
                try:
                    assert not (await prefill.next_token())["finished"]
                    await decode.reserve(await prefill.describe())
                    await gateway._poll_once()
                    responses = await asyncio.gather(*(
                        client.post("/v1/completions", json=body(length=4),
                                    headers={"x-request-id": f"rejected-{index}"})
                        for index in range(3)
                    ))
                    assert [response.status_code for response in responses] == [503, 503, 503]
                    result = analyze([tmp_path / "gateway.jsonl"])
                    rejected = [row for row in result["transactions"]
                                if row["request_id"].startswith("rejected-")]
                    assert len(rejected) == 3
                    assert len({row["gateway_id"] for row in rejected}) == 3
                    assert all(row["route"] == "unassigned" and row["outcome"] == "failed"
                               and row["payload_bytes"] is None and row["payload_unknown"]
                               and row["client_outcome"] is None for row in rejected)
                    assert all(any(event["event"] == "pd_gateway_failed"
                                   and event["status"] == 503 and event["reason"]
                                   for event in row["events"]) for row in rejected)
                    assert result["routes"]["unassigned"]["unknown_payload_requests"] == 3
                finally:
                    await prefill.close()
                    await decode.close()
                await drained(gateway, pairs)

    asyncio.run(run())


@pytest.mark.parametrize("small_role", ["prefill", "decode"])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("unicode_header", [False, True])
def test_request_control_header_uses_an_edge_with_room(tmp_path, small_role, pipeline, unicode_header):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        frame = 64 if pipeline else 0
        factory = layered_engine if pipeline else load_engine
        workers, ps, ds = [], [], []
        try:
            for index in range(2):
                worker = PDWorker("decode", {**config, "seed": index}, max_connections=8,
                                  max_transfer_bytes=1_000_000, request_timeout=6,
                                  max_header_bytes=1024 if small_role == "decode" and index == 0 else 1_048_576,
                                  pipeline_frame_bytes=frame, engine_factory=factory)
                ds.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            for index in range(2):
                worker = PDWorker("prefill", {**config, "seed": index}, max_connections=8,
                                  max_transfer_bytes=1_000_000, request_timeout=6,
                                  max_header_bytes=1024 if small_role == "prefill" and index == 0 else 1_048_576,
                                  decode_endpoint=ds[index], allowed_decode_endpoints=ds,
                                  pipeline_frame_bytes=frame, engine_factory=factory)
                ps.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.01, pipeline=2 if pipeline else 0)
            readiness = await check(cfg, None)
            assert len(readiness["control_header_limits"]) == 4
            for edge in readiness["control_header_limits"]:
                assert edge["prefill_max_bytes"] == (1024 if small_role == "prefill"
                                                     and tuple(edge["prefill"]) == ps[0] else 1_048_576)
                assert edge["decode_max_bytes"] == (1024 if small_role == "decode"
                                                    and tuple(edge["decode"]) == ds[0] else 1_048_576)
                assert edge["client_max_bytes"] == 1_048_576
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                wire_id = ("汉" * 180).encode() if unicode_header else (b'a"\\z' * 350)
                request_id = wire_id.decode("latin-1")
                response = await client.post("/v1/completions", json=body(length=4),
                                             headers={b"x-request-id": wire_id})
                assert response.status_code == 200, response.text
                row = next(item for item in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                           if item["request_id"] == request_id)
                assert tuple(row[small_role]) == (ps[1] if small_role == "prefill" else ds[1])
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_local_header_uses_decode_with_room(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers, ds = [], []
        try:
            for index in range(2):
                worker = PDWorker("decode", {**config, "seed": index, "chunked": True, "budget": 3},
                                  max_connections=8, max_transfer_bytes=1_000_000, request_timeout=6,
                                  max_header_bytes=1024 if index == 0 else 1_048_576,
                                  engine_factory=load_engine)
                ds.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, decode_endpoint=ds[0], allowed_decode_endpoints=ds,
                               engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, de) for de in ds]
            cfg = settings(pairs, elastic=elastic(backlog=0, idle=True))
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                response = await client.post("/v1/completions", json=body(length=4),
                                             headers={"x-request-id": "local" * 300})
                assert response.status_code == 200, response.text
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                assert assigned["route"] == "local" and tuple(assigned["decode"]) == ds[1]
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_actual_control_header_is_checked_at_the_peer_limit(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers = []
        try:
            decode = PDWorker("decode", config, max_connections=8, max_transfer_bytes=1_000_000,
                              request_timeout=6, engine_factory=load_engine)
            de = await decode.start("127.0.0.1", 0)
            workers.append(decode)
            request_id = 'a"\\z' * 300
            sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, max_model_length=32)
            lower_bound_header = {
                "op": "open_p", "lifetime": .1, "session_id": "0" * 32, "request_id": request_id,
                "token_ids": [1, 2], "sampling": asdict(sampling), "decode_endpoint": de,
            }
            limit = len(json.dumps(lower_bound_header, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, max_header_bytes=limit, decode_endpoint=de,
                               engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, de)]
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                response = await client.post("/v1/completions", json=body(length=1),
                                             headers={"x-request-id": request_id})
                assert response.status_code == 503 and "header" in response.text.lower()
                failure = records(tmp_path / "gateway.jsonl", "pd_gateway_failed")[-1]
                assert failure["phase"] == "open_p" and failure["status"] == 503
                await drained(gateway, pairs)
                assert (await client.post("/v1/completions", json=body(length=1))).status_code == 200
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_pipeline_routes_around_full_prefill_prompt_blocks(tmp_path):
    async def run():
        opened = []
        async with nodes(tmp_path, frame=64, engine_factory=layered_engine) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, poll_interval=.01, pipeline=2), tmp_path) as (client, gateway, _):
                try:
                    for index in range(2):
                        decode = await PDClient.open_d(ds[index], lifetime=5)
                        opened.append(decode)
                        prefill = await PDClient.open_p(
                            ps[0], decode.session_id, f"hold-{index}", [1] * 30,
                            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                            lifetime=5, decode_endpoint=ds[index])
                        opened.append(prefill)
                        assert not (await prefill.next_token())["finished"]
                    decode = await PDClient.open_d(ds[1], lifetime=5)
                    opened.append(decode)
                    prefill = await PDClient.prepare_p(
                        ps[1], decode.session_id, "queued", [1, 2, 3],
                        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                        lifetime=5, decode_endpoint=ds[1])
                    opened.append(prefill)
                    await wait_sample(gateway, ps[0], lambda row: row["used_blocks"] == 16
                                      and row["held_requests"] == 2)
                    await wait_sample(gateway, ps[1], lambda row: row["pending_prefill_tokens"] == 3)
                    response = await client.post("/v1/completions", json=body(length=4),
                                                 headers={"x-request-id": "p-capacity"})
                    assert response.status_code == 200, response.text
                    assigned = next(row for row in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                                    if row["request_id"] == "p-capacity")
                    assert tuple(assigned["prefill"]) == ps[1]
                finally:
                    for connection in reversed(opened):
                        await connection.close()
                await drained(gateway, pairs)

    asyncio.run(run())


def test_pipeline_prompt_block_booking_spreads_simultaneous_requests(tmp_path):
    async def run():
        opened = []
        async with nodes(tmp_path, frame=64, engine_factory=layered_engine) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, pipeline=2), tmp_path) as (client, gateway, _):
                try:
                    for index in range(2):
                        decode = await PDClient.open_d(ds[index], lifetime=5)
                        opened.append(decode)
                        prefill = await PDClient.open_p(
                            ps[0], decode.session_id, f"hold-{index}", [1] * 28,
                            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                            lifetime=5, decode_endpoint=ds[index])
                        opened.append(prefill)
                        assert not (await prefill.next_token())["finished"]
                    decode = await PDClient.open_d(ds[1], lifetime=5)
                    opened.append(decode)
                    prefill = await PDClient.prepare_p(
                        ps[1], decode.session_id, "queued", [1] * 24,
                        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                        lifetime=5, decode_endpoint=ds[1])
                    opened.append(prefill)
                    await gateway._poll_once()
                    assert gateway.samples[ps[0]][0]["used_blocks"] == 14
                    assert gateway.samples[ps[1]][0]["pending_prefill_tokens"] == 24
                    responses = await asyncio.gather(*(
                        client.post("/v1/completions", json=body(length=4),
                                    headers={"x-request-id": f"p-booking-{index}"})
                        for index in range(3)
                    ))
                    assert [response.status_code for response in responses] == [200, 200, 200]
                    assigned = [row for row in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                                if row["request_id"].startswith("p-booking-")]
                    assert [tuple(row["prefill"]) for row in assigned] == [ps[0], ps[0], ps[1]]
                finally:
                    for connection in reversed(opened):
                        await connection.close()
                await drained(gateway, pairs)

    asyncio.run(run())
