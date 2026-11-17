import asyncio
import json

import pytest

from myvllm.pd.client import PDClient
from test_pd_load_routing import http_gateway, nodes, records, settings
from test_pd_server import body, drained


async def no_resources(gateway, pairs):
    await drained(gateway, pairs)
    assert not gateway.requests
    assert all(not any(value.values()) for value in gateway.p_bookings.values())
    assert all(not any(value.values()) for value in gateway.d_bookings.values())
    assert all(value == 0 for value in gateway.connection_bookings.values())
    for endpoint in {endpoint for pair in pairs for endpoint in pair}:
        stats = await PDClient.stats(endpoint)
        assert stats["open_connections"] == stats["open_handles"] == stats["queued_events"] == 0
        assert stats["pipeline_staging_bytes"] == stats["pipeline_busy_slots"] == stats["pipeline_slots"] == 0
        if "reserved_blocks" in stats:
            assert stats["reserved_blocks"] == stats["resident_requests"] == 0


async def wait_open(path):
    async with asyncio.timeout(4):
        while not records(path, "pd_gateway_open"):
            await asyncio.sleep(.005)


@pytest.mark.parametrize("fault", ["partial-open", "timeout", "disconnect", "close", "timing"])
def test_load_faults_release_workers_and_all_bookings(tmp_path, fault):
    async def run():
        async with nodes(tmp_path, count=1, p_options=[{"delay": .2}]) as (config, _, ps, ds, workers):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs)
            if fault == "timeout":
                cfg["request_timeout"] = .08
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, port):
                request = None
                path = tmp_path / "gateway.jsonl"
                if fault == "partial-open":
                    await workers[-1].close()
                    response = await client.post("/v1/completions", json=body(length=4))
                    assert response.status_code == 503
                    failed = records(path, "pd_gateway_failed")[-1]
                    assert failed["phase"] == "open_p" and failed["session_id"]
                    assert await workers[-1].start(*ps[0]) == ps[0]
                elif fault == "timeout":
                    response = await client.post("/v1/completions", json=body(length=4))
                    assert response.status_code == 504
                elif fault == "disconnect":
                    _, writer = await asyncio.open_connection("127.0.0.1", port)
                    data = json.dumps(body(length=8)).encode()
                    writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                 b"Content-Type: application/json\r\nContent-Length: "
                                 + str(len(data)).encode() + b"\r\n\r\n" + data)
                    await writer.drain()
                    await wait_open(path)
                    writer.transport.abort()
                    await writer.wait_closed()
                else:
                    request = asyncio.create_task(client.post("/v1/completions", json=body(length=8)))
                    await wait_open(path)
                    if fault == "close":
                        await gateway.close()
                        assert (await request).status_code == 503
                    else:
                        saved = path.with_name("before-failure.jsonl")
                        path.rename(saved)
                        path.symlink_to("/dev/full")
                        try:
                            response = await request
                            assert response.status_code == 500 and "timing" in response.text.lower()
                        finally:
                            path.unlink()
                            saved.rename(path)
                await no_resources(gateway, pairs)
                if fault not in ("timeout", "close"):
                    assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                    await no_resources(gateway, pairs)
            if fault in ("timeout", "close"):
                recovery = tmp_path / "recovery"
                recovery.mkdir()
                async with http_gateway(config, settings(pairs), recovery) as (client, gateway, _):
                    assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                    await no_resources(gateway, pairs)

    asyncio.run(run())
