import asyncio
import time

import pytest

from myvllm.pd.gateway import PDGateway
from myvllm.pd.http import GatewayError
from myvllm.sampling_parameters import SamplingParams
from myvllm.server import PendingRequest
from test_pd_load_routing import http_gateway, nodes, records, reserved_remote, settings, wait_sample
from test_pd_server import body, drained


def pending(name):
    loop = asyncio.get_running_loop()
    return PendingRequest(
        name, [1, 2], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, max_model_length=32),
        time.perf_counter(), loop, loop.create_future(), asyncio.Queue(),
    )


def test_booking_adds_to_external_resident_in_same_sample(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 2}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            gateway = PDGateway(config, settings(pairs, poll_interval=.01))
            await gateway.start()
            prefill, decode = await reserved_remote(ps[0], ds[0])
            first = None
            try:
                await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                assert gateway.claim()
                first = gateway.submit(pending("first"))
                assert gateway.claim()
                second = None
                try:
                    with pytest.raises(GatewayError, match="capacity"):
                        second = gateway.submit(pending("second"))
                finally:
                    if second is None:
                        gateway.unclaim()
                    else:
                        gateway.finish(second.seq_id, False)
                gateway.finish(first.seq_id, False)
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            finally:
                if first is not None:
                    gateway.finish(first.seq_id, False)
                await prefill.close()
                await decode.close()
                await drained(gateway, pairs)
                await gateway.close()

    asyncio.run(run())


def test_concurrent_http_booking_accounts_for_external_resident(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, p_options=[{"delay": .1}],
                         d_options=[{"sequences": 2}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, poll_interval=.01), tmp_path) as (client, gateway, _):
                prefill, decode = await reserved_remote(ps[0], ds[0])
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                    responses = await asyncio.gather(*(
                        client.post("/v1/completions", json=body(length=4),
                                    headers={"x-request-id": f"concurrent-{index}"})
                        for index in range(4)
                    ))
                    assert sorted(response.status_code for response in responses) == [200, 503, 503, 503]
                    assigned = [row for row in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                                if row["request_id"].startswith("concurrent-")]
                    assert len(assigned) == 1
                finally:
                    await prefill.close()
                    await decode.close()
                await drained(gateway, pairs)
                await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 0)
                assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
                await drained(gateway, pairs)

    asyncio.run(run())


def test_sampled_own_resident_is_not_counted_twice(tmp_path):
    async def run():
        async with nodes(tmp_path, count=1, d_options=[{"sequences": 2, "delay": .04}]) as (config, _, ps, ds, _):
            pairs = list(zip(ps, ds))
            async with http_gateway(config, settings(pairs, poll_interval=.01), tmp_path) as (client, gateway, _):
                first = asyncio.create_task(client.post("/v1/completions", json=body(length=20)))
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                    assert gateway.d_bookings[ds[0]]["requests"] == 0
                    assert gateway.connection_bookings[ds[0]] == 0
                    second = await client.post("/v1/completions", json=body(length=2))
                    assert second.status_code == 200
                    assert (await first).status_code == 200
                finally:
                    if not first.done():
                        first.cancel()
                    await asyncio.gather(first, return_exceptions=True)
                await drained(gateway, pairs)

    asyncio.run(run())
