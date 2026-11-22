import asyncio
import json
import time

import pytest
from starlette.requests import ClientDisconnect, Request

from test_pd_load_routing import http_gateway, nodes, records, settings
from test_pd_server import body


def test_http_admission_fifo_and_overflow(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            cfg = dict(settings(list(zip(ps, ds))), max_active_requests=4,
                       admission_queue_size=2, admission_wait_timeout_s=3)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                for _ in range(4):
                    assert gateway.claim()
                async def post(name):
                    return await client.post("/v1/completions", headers={"x-request-id": name}, json=body(length=4))
                async def enqueue(name, n):
                    task = asyncio.create_task(post(name))
                    async with asyncio.timeout(2):
                        while len(gateway.admission_waiters) != n:
                            await asyncio.sleep(.005)
                    return task
                first = await enqueue("first", 1)
                second = await enqueue("second", 2)
                overflow = await post("over")
                assert overflow.status_code == 503 and "admission queue is full" in overflow.text
                assert gateway.claimed == 4
                gateway.unclaim()
                assert (await first).status_code == (await second).status_code == 200
                assert [r["request_id"] for r in records(tmp_path/"gateway.jsonl", "pd_admission_granted")] == ["first", "second"]
                for _ in range(3):
                    gateway.unclaim()
                assert gateway.claimed == 0 and not gateway.admission_waiters
    asyncio.run(run())


def test_admission_timeout_and_asgi_disconnect(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1) as (config, _, ps, ds, _):
            cfg = dict(settings(list(zip(ps, ds))), max_active_requests=1,
                       admission_queue_size=1, admission_wait_timeout_s=.2)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                assert gateway.claim()
                expired = await client.post("/v1/completions", json=body(length=4))
                assert expired.status_code == 503 and "admission wait timed out" in expired.text
                assert gateway.claimed == 1 and not gateway.admission_waiters
                events = asyncio.Queue()
                events.put_nowait({"type": "http.request", "body": json.dumps(body(length=4)).encode()})
                events.put_nowait({"type": "http.disconnect"})
                with pytest.raises(ClientDisconnect):
                    await gateway.admit_body(Request({"type": "http"}, events.get), "gone", time.perf_counter())
                assert gateway.claimed == 1 and not gateway.admission_waiters
                gateway.unclaim()
                assert (await client.post("/v1/completions", json=body(length=4))).status_code == 200
    asyncio.run(run())
