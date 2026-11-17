import asyncio
import json

import httpx
import pytest

from myvllm.server import create_app
from test_pd_load_routing import http_gateway, nodes, records, settings
from test_pd_server import body, drained
from tools.analyze_pd import analyze


def test_capacity_and_validation_failures_keep_request_identity(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, p_options=[{"delay": .15}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs)
            cfg["max_active_requests"] = 1
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                first = asyncio.create_task(client.post(
                    "/v1/completions", json=body(length=4), headers={"x-request-id": "same"}))
                try:
                    async with asyncio.timeout(4):
                        while not records(tmp_path / "gateway.jsonl", "pd_gateway_open"):
                            await asyncio.sleep(.005)
                    rejected = await client.post(
                        "/v1/completions", json=body(length=4), headers={"x-request-id": "same"})
                    assert rejected.status_code == 503
                    assert rejected.json() == {"detail": "PD gateway request capacity exhausted"}
                    assert (await first).status_code == 200
                    invalid = await client.post(
                        "/v1/completions", json={**body(), "model": "wrong"},
                        headers={"x-request-id": "bad-model"})
                    assert invalid.status_code == 400
                    too_long = await client.post(
                        "/v1/completions", json={**body(), "prompt": " ".join(["hello"] * 32)},
                        headers={"x-request-id": "too-long"})
                    assert too_long.status_code == 400
                    await drained(gateway, pairs)
                    result = analyze([tmp_path / "gateway.jsonl"])
                    assert result["outcomes"] == {"completed": 1, "failed": 3}
                    transactions = result["transactions"]
                    assert len(transactions) == len({row["gateway_id"] for row in transactions}) == 4
                    assert [row["request_id"] for row in transactions].count("same") == 2
                    failures = [row for row in transactions if row["outcome"] == "failed"]
                    assert {row["request_id"] for row in failures} == {"same", "bad-model", "too-long"}
                    assert all(row["route"] == "unassigned" and row["client_outcome"] is None for row in failures)
                    assert all(next(event for event in row["events"] if event["event"] == "pd_gateway_failed")
                               ["phase"] == "admission" for row in failures)
                finally:
                    if not first.done():
                        first.cancel()
                    await asyncio.gather(first, return_exceptions=True)

    asyncio.run(run())


def test_disconnect_and_task_cancellation_before_body_retain_permits_and_identity(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (_, gateway, port):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                             b"Content-Type: application/json\r\nContent-Length: 100\r\n"
                             b"X-Request-ID: disconnected\r\n\r\n{")
                await writer.drain()
                async with asyncio.timeout(4):
                    while gateway.claimed != 1:
                        await asyncio.sleep(.001)
                writer.transport.abort()
                await writer.wait_closed()
                async with asyncio.timeout(4):
                    while gateway.claimed:
                        await asyncio.sleep(.001)

                started, release = asyncio.Event(), asyncio.Event()

                async def upload():
                    yield b"{"
                    started.set()
                    await release.wait()

                app = create_app(config, "tiny", pd_worker=gateway)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                    request = asyncio.create_task(client.post(
                        "/v1/completions", content=upload(), headers={"x-request-id": "cancelled"}))
                    await asyncio.wait_for(started.wait(), 4)
                    assert gateway.claimed == 1
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                assert gateway.claimed == 0
                await drained(gateway, pairs)
                result = analyze([tmp_path / "gateway.jsonl"])
                assert result["outcomes"] == {"cancelled": 2}
                assert {row["request_id"] for row in result["transactions"]} == {"disconnected", "cancelled"}
                assert all(row["route"] == "unassigned" and row["client_outcome"] is None
                           for row in result["transactions"])

    asyncio.run(run())


def test_gateway_shutdown_during_upload_is_recorded_once(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (_, gateway, port):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                data = json.dumps(body(length=2)).encode()
                writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                             b"Content-Type: application/json\r\nConnection: close\r\n"
                             b"X-Request-ID: shutdown-upload\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data[:1])
                await writer.drain()
                async with asyncio.timeout(4):
                    while gateway.claimed != 1:
                        await asyncio.sleep(.001)
                await gateway.close()
                writer.write(data[1:])
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 5)
                writer.close()
                await writer.wait_closed()
                assert response.startswith(b"HTTP/1.1 503")
                await drained(gateway, pairs)
                result = analyze([tmp_path / "gateway.jsonl"])
                assert result["outcomes"] == {"failed": 1}
                assert result["transactions"][0]["request_id"] == "shutdown-upload"
                assert result["transactions"][0]["route"] == "unassigned"

    asyncio.run(run())


def test_routing_failure_is_recorded_once(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, workers):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.02, sample_age=2)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                await workers[-1].close()
                async with asyncio.timeout(4):
                    while ps[0] in gateway.samples:
                        await asyncio.sleep(.005)
                response = await client.post(
                    "/v1/completions", json=body(), headers={"x-request-id": "unroutable"})
                assert response.status_code == 503
                assert await workers[-1].start(*ps[0]) == ps[0]
                await drained(gateway, pairs)
                failed = records(tmp_path / "gateway.jsonl", "pd_gateway_failed")
                assert len(failed) == 1 and failed[0]["phase"] == "routing"
                result = analyze([tmp_path / "gateway.jsonl"])
                assert result["outcomes"] == {"failed": 1}
                assert result["transactions"][0]["request_id"] == "unroutable"

    asyncio.run(run())


def test_admission_timing_failure_is_visible_and_releases_claim(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                path = tmp_path / "gateway.jsonl"
                path.symlink_to("/dev/full")
                try:
                    response = await client.post(
                        "/v1/completions", json={**body(), "model": "wrong"},
                        headers={"x-request-id": "timing-failure"})
                    assert response.status_code == 500
                    assert "PD timing write failed" in response.json()["detail"]
                    assert gateway.claimed == 0 and not gateway.requests
                finally:
                    path.unlink()
                assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                await drained(gateway, pairs)

    asyncio.run(run())
