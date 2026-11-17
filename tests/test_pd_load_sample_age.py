import asyncio
from contextlib import asynccontextmanager

import pytest

from myvllm.pd.client import PDClient
from myvllm.pd.wire import read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from test_pd_load_routing import elastic, http_gateway, load_engine, records, settings
from test_pd_server import body, drained, make_config


@asynccontextmanager
async def delay_stats(target, delay):
    tasks = set()

    async def relay(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        upstream = None
        try:
            upstream_reader, upstream = await asyncio.open_connection(*target)
            header, size = await read_header(reader, 1_000_000, 1_000_000)
            await write_frame(upstream, header, await read_body(reader, size))
            if header["op"] == "stats":
                response, size = await read_header(upstream_reader, 1_000_000, 1_000_000)
                payload = await read_body(upstream_reader, size)
                await asyncio.sleep(delay)
                await write_frame(writer, response, payload)
            else:
                async def copy(source, destination):
                    while data := await source.read(65536):
                        destination.write(data)
                        await destination.drain()

                pumps = [asyncio.create_task(copy(reader, upstream)),
                         asyncio.create_task(copy(upstream_reader, writer))]
                try:
                    await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for pump in pumps:
                        pump.cancel()
                    await asyncio.gather(*pumps, return_exceptions=True)
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
        yield server.sockets[0].getsockname()[:2]
    finally:
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("max_age,expected", [(.25, 503), (2, 200)])
def test_delayed_stats_age_includes_the_real_tcp_response_time(tmp_path, max_age, expected):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        d = PDWorker("decode", {**config, "chunked": True, "budget": 3},
                     max_connections=8, max_transfer_bytes=1_000_000,
                     request_timeout=6, engine_factory=load_engine)
        de = await d.start("127.0.0.1", 0)
        try:
            warm = await PDClient.open_local(
                de, "warm", [1, 2], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True), lifetime=5)
            try:
                while not (await warm.next_token())["finished"]:
                    pass
            finally:
                await warm.close()
            assert (await PDClient.stats(de))["decode_step_sample_count"] > 0
            async with delay_stats(de, .6) as decode_proxy:
                p = PDWorker("prefill", config, max_connections=8, max_transfer_bytes=1_000_000,
                             request_timeout=6, decode_endpoint=decode_proxy, engine_factory=load_engine)
                pe = await p.start("127.0.0.1", 0)
                try:
                    async with delay_stats(pe, .6) as prefill_proxy:
                        pairs = [(prefill_proxy, decode_proxy)]
                        policy = {**elastic(backlog=0, idle=False), "step_sample_max_age_ms": 250}
                        cfg = {**settings(pairs, poll_interval=30, sample_age=max_age, elastic=policy),
                               "poll_timeout_s": 2}
                        async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                            response = await client.post("/v1/completions", json=body(length=2))
                            assert response.status_code == expected, response.text
                            if expected == 200:
                                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                                assert assigned["route"] == "remote"
                                assert assigned["decision"]["decode_step_sample_age_ms"] >= 600
                                assert assigned["decision"]["step_sample_fresh"] is False
                            await drained(gateway, [(pe, de)])
                finally:
                    await p.close()
        finally:
            await d.close()

    asyncio.run(run())
