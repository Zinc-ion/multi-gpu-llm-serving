import asyncio
from contextlib import asynccontextmanager
import errno
import gc
import json
import socket

import httpx
import pytest
import uvicorn

from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway
from myvllm.pd.wire import read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from myvllm.server import create_app
from pd_network_helpers import cpu_engine
from test_pd_server import body, cpu_gateway_engine, drained, make_config


@asynccontextmanager
async def gateway_service(path, *, on_body=None, restore_gate=None, gate_operation="restore"):
    config, _ = make_config(path)
    config["delay"] = .3
    timing = path / "gateway.jsonl"
    gateway_config = {**config, "timing_path": str(timing)}
    decode = PDWorker("decode", config, max_connections=4, max_transfer_bytes=1_000_000,
                      request_timeout=5, engine_factory=cpu_gateway_engine)
    de = await decode.start("127.0.0.1", 0)
    proxy = None
    proxy_tasks = set()
    if restore_gate is not None:
        decode_endpoint = de

        async def relay(reader, writer):
            task = asyncio.current_task()
            proxy_tasks.add(task)
            upstream_reader, upstream_writer = await asyncio.open_connection(*decode_endpoint)

            async def send_requests():
                while True:
                    header, size = await read_header(reader, 1_000_000, 1_000_000)
                    payload = await read_body(reader, size)
                    if header["op"] == gate_operation:
                        restore_gate[0].set()
                        await restore_gate[1].wait()
                    await write_frame(upstream_writer, header, payload)

            async def send_responses():
                while data := await upstream_reader.read(65536):
                    writer.write(data)
                    await writer.drain()

            tasks = [asyncio.create_task(send_requests()), asyncio.create_task(send_responses())]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for pending in tasks:
                    pending.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                upstream_writer.close()
                writer.close()
                await asyncio.gather(upstream_writer.wait_closed(), writer.wait_closed(),
                                     return_exceptions=True)
                proxy_tasks.discard(task)

        proxy = await asyncio.start_server(relay, "127.0.0.1", 0)
        de = proxy.sockets[0].getsockname()[:2]
    prefill = PDWorker("prefill", config, max_connections=4, max_transfer_bytes=1_000_000,
                       request_timeout=5, decode_endpoint=de, engine_factory=cpu_gateway_engine)
    pe = await prefill.start("127.0.0.1", 0)
    settings = {"pairs": [{"prefill": pe, "decode": de}], "max_active_requests": 1,
                "max_payload_bytes": 1_000_000, "max_staging_bytes": 1_000_000,
                "max_request_bytes": 4096, "request_timeout": 5}
    gateway = PDGateway(gateway_config, settings)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = create_app(gateway_config, "tiny", pd_worker=gateway)
    if on_body is not None:
        inner_app = app

        async def app(scope, receive, send):
            async def send_with_hook(message):
                await send(message)
                if message["type"] == "http.response.body":
                    on_body(timing, message.get("body", b""))
            await inner_app(scope, receive, send_with_hook)

    server = uvicorn.Server(uvicorn.Config(app,
                                           log_level="error", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(1000):
            if server.started:
                break
            if task.done():
                raise RuntimeError("HTTP server did not start")
            await asyncio.sleep(.01)
        else:
            raise RuntimeError("HTTP server did not start")
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10,
                                     trust_env=False) as client:
            yield client, gateway, [(pe, de)], timing
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 20)
        await prefill.close()
        await decode.close()
        if proxy is not None:
            proxy.close()
            await proxy.wait_closed()
            if proxy_tasks:
                await asyncio.gather(*proxy_tasks)


def break_sink(path):
    saved = path.with_name("before-failure.jsonl")
    path.rename(saved)
    path.symlink_to("/dev/full")
    return saved


def restore_sink(path, saved):
    path.unlink()
    saved.rename(path)


async def wait_for_gateway_open(path):
    for _ in range(1000):
        if path.exists() and any(json.loads(line)["event"] == "pd_gateway_open"
                                 for line in path.read_text().splitlines()):
            return
        await asyncio.sleep(.005)
    raise AssertionError("request did not open")


def test_gateway_timing_failure_returns_500_and_releases(tmp_path):
    async def run():
        async with gateway_service(tmp_path) as (client, gateway, endpoints, timing):
            request = asyncio.create_task(client.post("/v1/completions", json=body(length=4)))
            saved = None
            try:
                await wait_for_gateway_open(timing)
                saved = break_sink(timing)
                response = await asyncio.wait_for(asyncio.shield(request), 3)
                assert response.status_code == 500
                assert "timing" in response.text.lower()
                await drained(gateway, endpoints)
            finally:
                if saved is not None:
                    restore_sink(timing, saved)
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
            await drained(gateway, endpoints)
    asyncio.run(run())


def test_stream_timing_failure_sends_error_without_success(tmp_path):
    async def run():
        async with gateway_service(tmp_path) as (client, gateway, endpoints, timing):
            saved = None
            try:
                async with client.stream("POST", "/v1/completions",
                                         json=body(stream=True, length=6)) as response:
                    assert response.status_code == 200
                    lines = response.aiter_lines()
                    first = await asyncio.wait_for(anext(lines), 3)
                    assert first.startswith("data: ") and "choices" in json.loads(first[6:])
                    saved = break_sink(timing)
                    async with asyncio.timeout(3):
                        rest = [line async for line in lines]
                errors = [json.loads(line[6:])["error"] for line in rest
                          if line.startswith("data: {") and "error" in json.loads(line[6:])]
                assert errors and errors[-1]["code"] == 500
                assert "[DONE]" not in "\n".join(rest)
                await drained(gateway, endpoints)
            finally:
                if saved is not None:
                    restore_sink(timing, saved)
    asyncio.run(run())


def test_stream_chunk_sent_timing_failure_sends_error(tmp_path):
    async def run():
        saved = []

        def fail_after_send(timing, data):
            if b'"choices"' in data and not saved:
                saved.append(break_sink(timing))

        async with gateway_service(tmp_path, on_body=fail_after_send) as (client, gateway, endpoints, timing):
            try:
                response = await asyncio.wait_for(client.post(
                    "/v1/completions", json=body(stream=True, length=6,
                                                 stream_options={"include_usage": True})), 5)
                assert saved and response.status_code == 200
                rows = [json.loads(line[6:]) for line in response.text.splitlines()
                        if line.startswith("data: {")]
                assert rows[-1]["error"]["code"] == 500
                assert "timing" in rows[-1]["error"]["message"].lower()
                assert all(choice["finish_reason"] is None
                           for row in rows for choice in row.get("choices", []))
                assert not any("usage" in row for row in rows)
                assert "[DONE]" not in response.text
                await drained(gateway, endpoints)
            finally:
                if saved:
                    restore_sink(timing, saved[0])
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
            await drained(gateway, endpoints)
    asyncio.run(run())


def test_terminal_chunk_sent_timing_failure_keeps_sent_choice_and_sends_error(tmp_path):
    async def run():
        saved = []

        def fail_after_send(timing, data):
            if b'"choices"' in data and not saved:
                saved.append(break_sink(timing))

        async with gateway_service(tmp_path, on_body=fail_after_send) as (client, gateway, endpoints, timing):
            try:
                response = await asyncio.wait_for(client.post(
                    "/v1/completions", json=body(stream=True, length=1,
                                                 stream_options={"include_usage": True})), 5)
                rows = [json.loads(line[6:]) for line in response.text.splitlines()
                        if line.startswith("data: {")]
                assert saved and response.status_code == 200
                assert len(rows) == 2 and len(rows[0]["choices"]) == 1
                assert rows[0]["choices"][0]["finish_reason"] == "length"
                assert rows[1]["error"]["code"] == 500
                assert "timing" in rows[1]["error"]["message"].lower()
                assert not any("usage" in row for row in rows)
                assert "[DONE]" not in response.text
                await drained(gateway, endpoints)
            finally:
                if saved:
                    restore_sink(timing, saved[0])
    asyncio.run(run())


def test_handoff_transfer_timing_failure_is_500_and_releases(tmp_path):
    async def run():
        reached, resume = asyncio.Event(), asyncio.Event()
        async with gateway_service(tmp_path, restore_gate=(reached, resume)) as (client, gateway, endpoints, timing):
            request = asyncio.create_task(client.post("/v1/completions", json=body(length=4)))
            saved = None
            try:
                await asyncio.wait_for(reached.wait(), 3)
                saved = break_sink(timing)
                resume.set()
                response = await asyncio.wait_for(request, 3)
                assert response.status_code == 500
                assert "timing" in response.text.lower()
                assert "peer unavailable" not in response.text.lower()
                await drained(gateway, endpoints)
            finally:
                resume.set()
                if saved is not None:
                    restore_sink(timing, saved)
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
            await drained(gateway, endpoints)
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_release_timing_failure_prevents_success_response(tmp_path, stream):
    async def run():
        reached, resume = asyncio.Event(), asyncio.Event()
        async with gateway_service(tmp_path, restore_gate=(reached, resume),
                                   gate_operation="close") as (client, gateway, endpoints, timing):
            request = asyncio.create_task(client.post(
                "/v1/completions", json=body(stream=stream, length=1)))
            saved = None
            try:
                await asyncio.wait_for(reached.wait(), 3)
                assert any(json.loads(line)["event"] == "pd_gateway_completed"
                           for line in timing.read_text().splitlines())
                saved = break_sink(timing)
                resume.set()
                response = await asyncio.wait_for(request, 3)
                assert response.status_code == 500
                assert "timing" in response.text.lower()
                assert "choices" not in response.text and "[DONE]" not in response.text
                await drained(gateway, endpoints)
            finally:
                resume.set()
                if saved is not None:
                    restore_sink(timing, saved)
                if not request.done():
                    request.cancel()
                await asyncio.gather(request, return_exceptions=True)
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
            await drained(gateway, endpoints)
    asyncio.run(run())


def test_disconnect_with_broken_timing_releases_gateway_and_workers(tmp_path):
    async def run():
        async with gateway_service(tmp_path) as (client, gateway, endpoints, timing):
            port = client.base_url.port
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            data = json.dumps(body(length=8)).encode()
            writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data)
            await writer.drain()
            saved = None
            try:
                await wait_for_gateway_open(timing)
                saved = break_sink(timing)
                writer.transport.abort()
                await writer.wait_closed()
                await drained(gateway, endpoints)
                assert not gateway.requests
            finally:
                writer.close()
                if saved is not None:
                    restore_sink(timing, saved)
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
            await drained(gateway, endpoints)
    asyncio.run(run())


def test_worker_timing_failure_cleans_core_and_preserves_error(tmp_path):
    async def run():
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop_errors = []
        loop.set_exception_handler(lambda _, context: loop_errors.append(context))
        timing = tmp_path / "prefill.jsonl"
        decode = PDWorker("decode", {"world_size": 1}, max_connections=4,
                          max_transfer_bytes=1_000_000, request_timeout=5, engine_factory=cpu_engine)
        de = await decode.start("127.0.0.1", 0)
        prefill = PDWorker("prefill", {"world_size": 1, "timing_path": str(timing), "delay": .3},
                           max_connections=4, max_transfer_bytes=1_000_000, request_timeout=5,
                           decode_endpoint=de, engine_factory=cpu_engine)
        pe = await prefill.start("127.0.0.1", 0)
        dc = pc = None
        try:
            dc = await PDClient.open_d(de, lifetime=4)
            pc = await PDClient.open_p(pe, dc.session_id, "enospc", [1, 2, 3],
                                       SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
                                       lifetime=4)
            break_sink(timing)
            with pytest.raises((asyncio.IncompleteReadError, ConnectionError, OSError)):
                await asyncio.wait_for(pc.next_token(), 3)
            await asyncio.wait_for(prefill.failed.wait(), 3)
            await asyncio.to_thread(prefill.owner.thread.join, 3)
            assert not prefill.owner.thread.is_alive()
            stats = prefill.owner.core.stats()
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert not prefill.owner.contexts
            with pytest.raises(OSError) as error:
                await prefill.close()
            assert error.value is prefill.owner.error and error.value.errno == errno.ENOSPC
        finally:
            if pc is not None:
                await pc.close()
            if dc is not None:
                await dc.close()
            try:
                await prefill.close()
            except OSError:
                pass
            await decode.close()
            gc.collect()
            await asyncio.sleep(.05)
            loop.set_exception_handler(previous_handler)
        assert not [item for item in loop_errors if "never retrieved" in item.get("message", "").lower()]
    asyncio.run(run())
