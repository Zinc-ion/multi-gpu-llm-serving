import asyncio

import pytest

from test_pd_load_routing import http_gateway, nodes, records, reserved_remote, settings, wait_sample
from test_pd_server import body, drained


async def wait_queued(path, request, count=1):
    async with asyncio.timeout(3):
        while len(records(path, "pd_gateway_queued")) < count:
            if request.done():
                pytest.fail(f"Request returned before queueing: {request.result().text}")
            await asyncio.sleep(.005)


def test_gateway_waits_for_decode_capacity_and_bounds_queue(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 1}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = {**settings(pairs, poll_interval=.02), "route_queue_size": 1, "route_wait_timeout_s": 2}
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                p, d = await reserved_remote(ps[0], ds[0])
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                    first = asyncio.create_task(client.post("/v1/completions", json=body(length=4)))
                    log = tmp_path / "gateway.jsonl"
                    await wait_queued(log, first)
                    assert not records(log, "pd_gateway_assigned")
                    overflow = await client.post("/v1/completions", json=body(length=4))
                    assert overflow.status_code == 503 and "queue is full" in overflow.text
                    assert not first.done()
                finally:
                    await d.close()
                    await p.close()
                response = await first
                assert response.status_code == 200, response.text
                assigned, = records(log, "pd_gateway_assigned")
                assert assigned["route_wait_ms"] > 0
                await drained(gateway, pairs)

    asyncio.run(run())


def test_route_timeout_and_client_disconnect_release_gateway_claim(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 1}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            cfg = {**settings(pairs, poll_interval=.02), "route_queue_size": 1, "route_wait_timeout_s": .3}
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                p, d = await reserved_remote(ps[0], ds[0])
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                    expired = await client.post("/v1/completions", json=body(length=4))
                    assert expired.status_code == 503 and "route wait timed out" in expired.text
                    assert gateway.claimed == 0
                    waiting = asyncio.create_task(client.post("/v1/completions", json=body(length=4)))
                    log = tmp_path / "gateway.jsonl"
                    await wait_queued(log, waiting, count=2)
                    waiting.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiting
                    async with asyncio.timeout(2):
                        while gateway.claimed:
                            await asyncio.sleep(.005)
                    cancelled, = records(log, "pd_gateway_cancelled")
                    assert cancelled["phase"] == "route_wait"
                    assert not records(log, "pd_gateway_assigned")
                finally:
                    await d.close()
                    await p.close()
                response = await client.post("/v1/completions", json=body(length=4))
                assert response.status_code == 200, response.text
                await drained(gateway, pairs)

    asyncio.run(run())
