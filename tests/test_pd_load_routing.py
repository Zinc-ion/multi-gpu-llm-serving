import asyncio
from contextlib import asynccontextmanager
import json
import socket
import time

import httpx
import pytest
import uvicorn

from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway
from myvllm.pd.ready import check
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from myvllm.server import PendingRequest, create_app
from pd_network_helpers import SlowCPUExecution
from pd_pipeline_helpers import pipeline_engine
from test_pd_server import body, cpu_gateway_engine, drained, make_config


def load_engine(config):
    engine = cpu_gateway_engine(config)
    engine.config["enforce_eager"] = config.get("enforce_eager", True)
    return engine


class BatchRecordingExecution(SlowCPUExecution):
    def __init__(self, config, delay, forced_token):
        super().__init__(config, delay, forced_token)
        self.batches = []

    def call(self, method, sequences, is_prefill):
        self.batches.append([(seq.seq_id, seq.num_cached_tokens, seq.num_prompt_tokens,
                              seq.num_completion_tokens) for seq in sequences])
        return super().call(method, sequences, is_prefill)


def batch_engine(config):
    engine = load_engine(config)
    engine.model_runner = BatchRecordingExecution(engine.config, config.get("delay", 0),
                                                   config.get("forced_token"))
    return engine


def layered_engine(config):
    engine = pipeline_engine(config)
    engine.config["model_name_or_path"] = config["model_name_or_path"]
    engine.config["seed"] = config.get("seed", 0)
    return engine


def settings(pairs, *, poll_interval=30, sample_age=60, elastic=None, pipeline=0):
    result = {
        "pairs": [{"prefill": p, "decode": d} for p, d in pairs],
        "routing_mode": "load", "max_active_requests": 8,
        "max_payload_bytes": 1_000_000, "max_staging_bytes": 8_000_000,
        "max_request_bytes": 4096, "request_timeout": 5,
        "poll_interval_s": poll_interval, "poll_timeout_s": 1,
        "max_sample_age_s": sample_age,
    }
    if elastic is not None:
        result["elastic"] = elastic
    if pipeline:
        result["pipeline_min_tokens"] = pipeline
    return result


def elastic(*, backlog=2, idle=True):
    return {
        "enabled": True, "prefill_backlog_tokens": backlog,
        "tpot_budget_ms": 1000, "tpot_margin_ms": 1,
        "step_sample_max_age_ms": 10000, "allow_idle_without_sample": idle,
    }


@asynccontextmanager
async def nodes(path, *, count=2, p_options=None, d_options=None, frame=0, engine_factory=load_engine):
    config, tokenizer = make_config(path)
    config["enforce_eager"] = True
    workers, ps, ds = [], [], []
    p_options = p_options or [{} for _ in range(count)]
    d_options = d_options or [{} for _ in range(count)]
    try:
        for index in range(count):
            options = {"seed": index, "timing_path": str(path / f"d{index}.jsonl"), **d_options[index]}
            worker = PDWorker("decode", {**config, **options}, max_connections=8,
                              max_transfer_bytes=1_000_000, request_timeout=6,
                              pipeline_frame_bytes=frame, engine_factory=engine_factory)
            endpoint = await worker.start("127.0.0.1", 0)
            workers.append(worker)
            ds.append(endpoint)
        for index in range(count):
            options = {"seed": index, "timing_path": str(path / f"p{index}.jsonl"), **p_options[index]}
            worker = PDWorker("prefill", {**config, **options}, max_connections=8,
                              max_transfer_bytes=1_000_000, request_timeout=6,
                              decode_endpoint=ds[index], allowed_decode_endpoints=ds,
                              pipeline_frame_bytes=frame, engine_factory=engine_factory)
            endpoint = await worker.start("127.0.0.1", 0)
            workers.append(worker)
            ps.append(endpoint)
        yield config, tokenizer, ps, ds, workers
    finally:
        for worker in reversed(workers):
            await worker.close()


@asynccontextmanager
async def http_gateway(config, gateway_settings, path):
    config = {**config, "timing_path": str(path / "gateway.jsonl")}
    gateway = PDGateway(config, gateway_settings)
    app = create_app(config, "tiny", pd_worker=gateway)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    raise RuntimeError("HTTP gateway failed to start")
                await asyncio.sleep(.01)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10,
                                     trust_env=False) as client:
            yield client, gateway, port
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 20)


def records(path, event):
    if not path.exists():
        return []
    return [row for row in map(json.loads, path.read_text().splitlines()) if row["event"] == event]


async def wait_sample(gateway, endpoint, predicate):
    async with asyncio.timeout(5):
        while True:
            sample = gateway.samples.get(endpoint)
            if sample and predicate(sample[0]):
                return sample
            await asyncio.sleep(.005)


async def reserved_remote(pe, de, prompt=None):
    prompt = prompt or [1, 2, 3]
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    decode = await PDClient.open_d(de, lifetime=5)
    prefill = await PDClient.open_p(pe, decode.session_id, "external", prompt, params,
                                    lifetime=5, decode_endpoint=de)
    assert not (await prefill.next_token())["finished"]
    await decode.reserve(await prefill.describe())
    return prefill, decode


def test_bookings_distribute_simultaneous_real_http_requests(tmp_path):
    async def run():
        async with nodes(tmp_path, p_options=[{"delay": .1}, {"delay": .1}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                responses = await asyncio.gather(*(
                    client.post("/v1/completions", json=body(length=4), headers={"x-request-id": str(i)})
                    for i in range(4)
                ))
                assert all(response.status_code == 200 for response in responses)
                assert all(response.json()["usage"]["completion_tokens"] == 4 for response in responses)
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                assert [tuple(row["prefill"]) for row in assigned[:2]] == ps
                assert [tuple(row["decode"]) for row in assigned[:2]] == ds
                assert assigned[1]["decision"]["prefill_pending_tokens"] == 0
                assert assigned[1]["decision"]["decode_reserved_blocks"] == 0
                await drained(gateway, pairs)
                assert all(value == {"tokens": 0, "requests": 0, "blocks": 0, "early_requests": 0,
                                     "transfer_bytes": 0}
                           for value in gateway.p_bookings.values())
                assert all(value == {"blocks": 0, "requests": 0, "transfer_bytes": 0}
                           for value in gateway.d_bookings.values())
                assert all(value == 0 for value in gateway.connection_bookings.values())
    asyncio.run(run())


def test_real_prefill_pending_tokens_change_selection(tmp_path):
    async def run():
        options = [{"chunked": True, "budget": 2, "delay": .06}, {}]
        async with nodes(tmp_path, p_options=options) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, poll_interval=.01, sample_age=3), tmp_path) as (client, gateway, _):
                dc = await PDClient.open_d(ds[0], lifetime=5)
                pc = await PDClient.open_p(ps[0], dc.session_id, "backlog", list(range(1, 27)),
                                           SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
                                           lifetime=5)
                try:
                    await wait_sample(gateway, ps[0], lambda row: row["pending_prefill_tokens"] >= 8)
                    response = await client.post("/v1/completions", json=body(length=3))
                    assert response.status_code == 200
                    assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                    assert tuple(assigned["prefill"]) == ps[1]
                finally:
                    await pc.close()
                    await dc.close()
                await drained(gateway, pairs)
    asyncio.run(run())


@pytest.mark.parametrize("pipeline", [False, True])
def test_reserved_decode_selects_nondefault_target_over_real_tcp(tmp_path, pipeline):
    async def run():
        factory = layered_engine if pipeline else load_engine
        frame = 64 if pipeline else 0
        async with nodes(tmp_path, frame=frame, engine_factory=factory) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.01, sample_age=3, pipeline=2 if pipeline else 0)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                prefill, decode = await reserved_remote(ps[1], ds[0])
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["reserved_blocks"] > 0)
                    response = await client.post("/v1/completions", json=body(length=4),
                                                 headers={"x-request-id": "routed"})
                    assert response.status_code == 200
                    assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                    opened = records(tmp_path / "gateway.jsonl", "pd_gateway_open")[-1]
                    assert tuple(assigned["prefill"]) == ps[0]
                    assert tuple(assigned["decode"]) == ds[1] != ds[0]
                    assert opened["transfer_mode"] == ("pipeline" if pipeline else "bulk")
                    assert any(row["request_id"] == "routed" for row in
                               records(tmp_path / "d1.jsonl", "pd_reserved"))
                    if pipeline:
                        assert records(tmp_path / "gateway.jsonl", "pd_pipeline_done")
                finally:
                    await prefill.close()
                    await decode.close()
                await drained(gateway, pairs)
    asyncio.run(run())


def test_elastic_local_and_remote_with_real_http_and_chunked_decode(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, p_options=[{"chunked": True, "budget": 2, "delay": .05}],
                         d_options=[{"chunked": True, "budget": 3, "delay": .04}],
                         engine_factory=batch_engine) as (config, _, ps, ds, workers):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.01, sample_age=5, elastic=elastic(idle=False))
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                warm = await client.post("/v1/completions", json=body(length=3))
                assert warm.status_code == 200
                await wait_sample(gateway, ds[0], lambda row: row["decode_step_sample_count"] > 0
                                  and row["resident_requests"] == 0)
                remote = asyncio.create_task(
                    client.post("/v1/completions", json=body(length=25), headers={"x-request-id": "remote"}))
                backlog_d = backlog_p = None
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["remote_resident_requests"] == 1)
                    backlog_d = await PDClient.open_d(ds[0], lifetime=5)
                    backlog_p = await PDClient.open_p(
                        ps[0], backlog_d.session_id, "backlog", list(range(1, 25)),
                        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True), lifetime=5)
                    await wait_sample(gateway, ps[0], lambda row: row["pending_prefill_tokens"] >= 2)
                    local = await client.post(
                        "/v1/completions", json={**body(length=4), "prompt": " ".join(["hello"] * 16)},
                        headers={"x-request-id": "local"})
                    responses = [await remote, local]
                finally:
                    if backlog_p is not None:
                        await backlog_p.close()
                    if backlog_d is not None:
                        await backlog_d.close()
                    if not remote.done():
                        remote.cancel()
                    await asyncio.gather(remote, return_exceptions=True)
                assert [response.status_code for response in responses] == [200, 200]
                assert [response.json()["usage"]["completion_tokens"] for response in responses] == [25, 4]
                assigned = {row["request_id"]: row for row in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")}
                assert assigned["remote"]["route"] == "remote"
                assert assigned["local"]["route"] == "local"
                assert assigned["local"]["decision"]["decode_step_wall_ms"] > 0
                local_records = [row for row in records(tmp_path / "d0.jsonl", "pd_admitted")
                                 if row["request_id"] == "local"]
                assert len(local_records) == 1 and local_records[0]["route"] == "local"
                assert local_records[0]["payload_bytes"] == 0
                batches = workers[0].owner.engine.model_runner.batches
                assert any(any(prompt == 16 and cached < prompt for _, cached, prompt, _ in batch)
                           and any(prompt == 2 and cached >= prompt and completion >= 1
                                   for _, cached, prompt, completion in batch)
                           for batch in batches)
                await drained(gateway, pairs)
    asyncio.run(run())


def test_seed_collision_on_cross_edge_and_load_ready_allowlist(tmp_path):
    async def run():
        async with nodes(tmp_path, p_options=[{"seed": 0}, {"seed": 0x80000000}],
                         d_options=[{"seed": 0}, {"seed": 0x80000000}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            fixed = settings(pairs)
            fixed.pop("routing_mode")
            fixed_gateway = PDGateway(config, fixed)
            await fixed_gateway.start()
            await fixed_gateway.close()
            with pytest.raises(ValueError, match="seed collision"):
                await PDGateway(config, settings(pairs)).start()
            with pytest.raises(ValueError, match="allowed edge"):
                await check(settings(pairs), None)
    asyncio.run(run())


def test_startup_checks_actual_length_format_and_pipeline_features(tmp_path):
    async def run():
        async with nodes(tmp_path, d_options=[{}, {"model_length": 31}]) as (config, _, ps, ds, _):
            with pytest.raises(ValueError, match="length"):
                await PDGateway(config, settings(list(zip(ps, ds)))).start()
        other = tmp_path / "other"
        other.mkdir()
        async with nodes(other) as (config, _, ps, ds, _):
            with pytest.raises(ValueError, match="model"):
                await PDGateway({**config, "head_dim": 8}, settings(list(zip(ps, ds)))).start()
            with pytest.raises(ValueError, match="Pipeline"):
                await PDGateway(config, settings(list(zip(ps, ds)), pipeline=2)).start()
            with pytest.raises(ValueError, match="timeout"):
                await PDGateway(config, {**settings(list(zip(ps, ds))), "request_timeout": 7}).start()
            with pytest.raises(ValueError, match="body limit"):
                await PDGateway(config, {**settings(list(zip(ps, ds))), "max_payload_bytes": 100}).start()
    asyncio.run(run())


def test_repeated_prefill_inventory_and_ready_reports_all_allowed_edges(tmp_path):
    async def run():
        async with nodes(tmp_path) as (config, _, ps, ds, _):
            cfg = settings([(ps[0], ds[0]), (ps[0], ds[1])])
            gateway = PDGateway(config, cfg)
            await gateway.start()
            try:
                assert gateway.prefills == [ps[0]] and gateway.decodes == ds
                result = await check(cfg, None)
                assert result["allowed_edges"] == [
                    {"prefill": list(ps[0]), "decode": list(ds[0])},
                    {"prefill": list(ps[0]), "decode": list(ds[1])},
                ]
            finally:
                await gateway.close()
    asyncio.run(run())


def test_timing_write_failure_and_prestart_cancellation_release_bookings(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                timing = tmp_path / "gateway.jsonl"
                timing.symlink_to("/dev/full")
                response = await client.post("/v1/completions", json=body(length=4))
                assert response.status_code == 500 and "timing" in response.text.lower()
                assert gateway.p_bookings[ps[0]] == {
                    "tokens": 0, "requests": 0, "blocks": 0, "early_requests": 0, "transfer_bytes": 0}
                assert gateway.d_bookings[ds[0]] == {"blocks": 0, "requests": 0, "transfer_bytes": 0}
                assert all(value == 0 for value in gateway.connection_bookings.values())
                assert gateway.claimed == 0 and not gateway.requests
                timing.unlink()
                loop = asyncio.get_running_loop()
                pending = PendingRequest("prestart", [1, 2],
                                         SamplingParams(temperature=0, max_tokens=4, ignore_eos=True,
                                                        max_model_length=32),
                                         time.perf_counter(), loop, loop.create_future(), asyncio.Queue())
                assert gateway.claim()
                seq = gateway.submit(pending)
                assert gateway.p_bookings[ps[0]]["requests"] == 1
                gateway.finish(seq.seq_id, False)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                await drained(gateway, pairs)
                assert gateway.p_bookings[ps[0]] == {
                    "tokens": 0, "requests": 0, "blocks": 0, "early_requests": 0, "transfer_bytes": 0}
                assert gateway.d_bookings[ds[0]] == {"blocks": 0, "requests": 0, "transfer_bytes": 0}
                assert all(value == 0 for value in gateway.connection_bookings.values())
                assert any(row.get("phase") == "before_start" for row in
                           records(timing, "pd_gateway_cancelled"))
                assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                await drained(gateway, pairs)
    asyncio.run(run())


def test_stale_and_unavailable_samples_recover_without_retries(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, workers):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, poll_interval=.5, sample_age=.08), tmp_path) as (client, gateway, _):
                initial = gateway.samples[ps[0]][1]
                async with asyncio.timeout(2):
                    while time.monotonic() - initial <= .09:
                        await asyncio.sleep(.005)
                response = await client.post("/v1/completions", json=body(length=1))
                assert response.status_code == 503
                async with asyncio.timeout(2):
                    while gateway.samples[ps[0]][1] == initial:
                        await asyncio.sleep(.005)
                assert (await client.post("/v1/completions", json=body(length=1))).status_code == 200
                await drained(gateway, pairs)
            second = tmp_path / "second"
            second.mkdir()
            async with http_gateway(config, settings(pairs, poll_interval=.02, sample_age=2), second) as (client, gateway, _):
                await workers[-1].close()
                async with asyncio.timeout(2):
                    while ps[0] in gateway.samples:
                        await asyncio.sleep(.005)
                assert (await client.post("/v1/completions", json=body(length=1))).status_code == 503
                assert await workers[-1].start(*ps[0]) == ps[0]
                await wait_sample(gateway, ps[0], lambda row: True)
                assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                await drained(gateway, pairs)
    asyncio.run(run())


def test_owner_capacity_rejects_stale_estimate_and_later_request_succeeds(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 1}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                prefill, decode = await reserved_remote(ps[0], ds[0])
                try:
                    response = await client.post("/v1/completions", json=body(length=4))
                    assert response.status_code == 503
                finally:
                    await prefill.close()
                    await decode.close()
                await drained(gateway, pairs)
                assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                await drained(gateway, pairs)
    asyncio.run(run())


def test_local_first_token_eos_and_disconnect_cleanup(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"chunked": True, "budget": 2,
                                                       "delay": .04, "forced_token": 62}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, elastic=elastic(backlog=0, idle=True))
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, port):
                for request, reason in ((body(length=4, ignore_eos=False), "stop"),
                                        (body(length=1), "length")):
                    response = await client.post("/v1/completions", json=request)
                    assert response.status_code == 200
                    assert response.json()["usage"]["completion_tokens"] == 1
                    assert response.json()["choices"][0]["finish_reason"] == reason
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                data = json.dumps({**body(length=8), "prompt": " ".join(["hello"] * 15)}).encode()
                writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                             b"Content-Type: application/json\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data)
                await writer.drain()
                async with asyncio.timeout(2):
                    while not gateway.requests:
                        await asyncio.sleep(.005)
                writer.transport.abort()
                await writer.wait_closed()
                await drained(gateway, pairs)
                assert all(row["route"] == "local" for row in
                           records(tmp_path / "gateway.jsonl", "pd_gateway_assigned"))
                assert (await client.post("/v1/completions", json=body(length=1))).status_code == 200
                await drained(gateway, pairs)
    asyncio.run(run())
