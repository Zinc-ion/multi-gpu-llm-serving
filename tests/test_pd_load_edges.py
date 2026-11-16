import asyncio

import pytest

from myvllm.pd.ready import check
from myvllm.pd.gateway import PDGateway
from myvllm.pd.worker import PDWorker
from test_pd_load_routing import elastic, http_gateway, load_engine, records, reserved_remote, settings, wait_sample
from test_pd_server import body, drained, make_config


def test_load_chooses_edge_that_can_transfer_actual_payload(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers = []
        try:
            decodes = []
            for index, limit in enumerate((256, 1_000_000)):
                worker = PDWorker("decode", {**config, "seed": index}, max_connections=8,
                                  max_transfer_bytes=limit, request_timeout=6, engine_factory=load_engine)
                decodes.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, decode_endpoint=decodes[0],
                               allowed_decode_endpoints=decodes, engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, decode) for decode in decodes]
            async with http_gateway(config, settings(pairs), tmp_path) as (client, gateway, _):
                response = await client.post(
                    "/v1/completions", json={**body(length=4), "prompt": "hello " * 5})
                assert response.status_code == 200, response.text
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                assert tuple(assigned["decode"]) == decodes[1]
                await drained(gateway, pairs)
                response = await client.post(
                    "/v1/completions", json={**body(length=1), "prompt": "hello " * 5})
                assert response.status_code == 200, response.text
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                assert tuple(assigned["decode"]) == decodes[0]
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_load_uses_other_decode_when_connection_slots_are_booked(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers, external = [], []
        try:
            decodes = []
            for index, connections in enumerate((1, 8)):
                worker = PDWorker("decode", {**config, "seed": index, "sequences": 4},
                                  max_connections=connections, max_transfer_bytes=1_000_000,
                                  request_timeout=6, engine_factory=load_engine)
                decodes.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            prefill = PDWorker("prefill", {**config, "delay": .1}, max_connections=8,
                               max_transfer_bytes=1_000_000, request_timeout=6,
                               decode_endpoint=decodes[0], allowed_decode_endpoints=decodes,
                               engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, decode) for decode in decodes]
            async with http_gateway(config, settings(pairs, poll_interval=.01), tmp_path) as (client, gateway, _):
                try:
                    external.append(await reserved_remote(pe, decodes[1]))
                    external.append(await reserved_remote(pe, decodes[1], prompt=[4, 5, 6]))
                    await wait_sample(gateway, decodes[1], lambda row: row["resident_requests"] == 2)
                    responses = await asyncio.gather(*(
                        client.post("/v1/completions", json=body(length=4),
                                    headers={"x-request-id": f"connections-{index}"})
                        for index in range(2)
                    ))
                    assert [response.status_code for response in responses] == [200, 200]
                    assigned = [row for row in records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")
                                if row["request_id"].startswith("connections-")]
                    assert [tuple(row["decode"]) for row in assigned] == decodes
                finally:
                    for pc, dc in external:
                        await pc.close()
                        await dc.close()
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_load_accounts_for_existing_transfer_reservations(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers, external = [], []
        try:
            decodes = []
            for index, limit in enumerate((512, 1_000_000)):
                worker = PDWorker("decode", {**config, "seed": index, "sequences": 4},
                                  max_connections=8, max_transfer_bytes=limit,
                                  request_timeout=6, engine_factory=load_engine)
                decodes.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, decode_endpoint=decodes[0],
                               allowed_decode_endpoints=decodes, engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, decode) for decode in decodes]
            async with http_gateway(config, settings(pairs, poll_interval=.01), tmp_path) as (client, gateway, _):
                try:
                    external.append(await reserved_remote(pe, decodes[0]))
                    external.append(await reserved_remote(pe, decodes[1]))
                    external.append(await reserved_remote(pe, decodes[1], prompt=[4, 5, 6]))
                    await wait_sample(gateway, decodes[0], lambda row: row["reserved_transfer_bytes"] == 256)
                    await wait_sample(gateway, decodes[1], lambda row: row["resident_requests"] == 2)
                    response = await client.post(
                        "/v1/completions", json={**body(length=4), "prompt": "hello " * 5})
                    assert response.status_code == 200, response.text
                    assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                    assert tuple(assigned["decode"]) == decodes[1]
                finally:
                    for pc, dc in external:
                        await pc.close()
                        await dc.close()
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_ready_checks_timeout_and_limits_of_every_inventory_worker(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers = []
        try:
            decode = PDWorker("decode", config, max_connections=8, max_transfer_bytes=1_000_000,
                              request_timeout=6, engine_factory=load_engine)
            de = await decode.start("127.0.0.1", 0)
            workers.append(decode)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, decode_endpoint=de, engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            cfg = settings([(pe, de)])
            with pytest.raises(ValueError, match="timeout"):
                await check({**cfg, "request_timeout": 7}, None)
            with pytest.raises(ValueError, match="payload"):
                await check({**cfg, "max_payload_bytes": 100}, None)
            with pytest.raises(ValueError, match="chunked"):
                await check({**cfg, "elastic": elastic()}, None)
            with pytest.raises(ValueError, match="routing_mode"):
                await check({**cfg, "routing_mode": "invalid"}, None)
            result = await check(cfg, None)
            assert result["allowed_edges"] == [{"prefill": list(pe), "decode": list(de)}]
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_elastic_requires_a_reachable_chunked_decode(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers, decodes = [], []
        try:
            for chunked in (False, True):
                worker = PDWorker("decode", {**config, "chunked": chunked, "budget": 3 if chunked else 32},
                                  max_connections=8, max_transfer_bytes=1_000_000,
                                  request_timeout=6, engine_factory=load_engine)
                decodes.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                               request_timeout=6, decode_endpoint=decodes[0], engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            cfg = settings([(pe, de) for de in decodes], elastic=elastic())
            gateway = PDGateway(config, cfg)
            try:
                with pytest.raises(ValueError, match="chunked"):
                    await gateway.start()
            finally:
                await gateway.close()
            with pytest.raises(ValueError, match="chunked"):
                await check(cfg, None)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_local_route_does_not_require_prefill_transfer_capacity(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        workers = []
        try:
            decode = PDWorker("decode", {**config, "chunked": True, "budget": 3},
                              max_connections=8, max_transfer_bytes=1_000_000,
                              request_timeout=6, engine_factory=load_engine)
            de = await decode.start("127.0.0.1", 0)
            workers.append(decode)
            prefill = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=256,
                               request_timeout=6, decode_endpoint=de, engine_factory=load_engine)
            pe = await prefill.start("127.0.0.1", 0)
            workers.append(prefill)
            pairs = [(pe, de)]
            cfg = settings(pairs, elastic=elastic(backlog=0, idle=True))
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                response = await client.post(
                    "/v1/completions", json={**body(length=4), "prompt": "hello " * 5})
                assert response.status_code == 200, response.text
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                assert assigned["route"] == "local"
                await drained(gateway, pairs)
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())
