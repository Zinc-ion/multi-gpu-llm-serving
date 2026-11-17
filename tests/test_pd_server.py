import asyncio
from contextlib import asynccontextmanager
import json
import socket
import time

import httpx
from openai import OpenAI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import uvicorn

from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from myvllm.server import create_app
from pd_network_helpers import cpu_engine
from test_pd_core import make_engine


def cpu_gateway_engine(config):
    engine = cpu_engine(config)
    engine.config["model_name_or_path"] = config["model_name_or_path"]
    engine.config["seed"] = config.get("seed", 0)
    return engine


def make_config(path):
    vocab = {"[UNK]": 0, "hello": 1, **{f"token{i}": i for i in range(2, 62)},
             "[EOS1]": 62, "[EOS2]": 63}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS1]",
        additional_special_tokens=["[EOS2]"],
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}",
    )
    tokenizer.save_pretrained(path)
    config = dict(make_engine().config)
    config.update(model_name_or_path=str(path), max_model_length=32, eos=[62, 63])
    return config, tokenizer


@asynccontextmanager
async def service(tmp_path, *, pairs=1, delay=0, forced_token=None, sequences=4,
                  timeout=5, active=4, staging=1_000_000):
    config, tokenizer = make_config(tmp_path)
    logs = tmp_path / "gateway.jsonl"
    config["timing_path"] = str(logs)
    workers = []
    endpoints = []
    server = task = None
    try:
        for index in range(pairs):
            worker_config = {**config, "delay": delay, "forced_token": forced_token,
                             "sequences": sequences, "seed": index}
            d = PDWorker("decode", worker_config, max_connections=4, max_transfer_bytes=1_000_000,
                         request_timeout=5, engine_factory=cpu_gateway_engine)
            de = await d.start("127.0.0.1", 0)
            workers.append(d)
            p = PDWorker("prefill", worker_config, max_connections=4, max_transfer_bytes=1_000_000,
                         request_timeout=5, decode_endpoint=de, engine_factory=cpu_gateway_engine)
            pe = await p.start("127.0.0.1", 0)
            workers.append(p)
            endpoints.append((pe, de))
        settings = {"pairs": [{"prefill": pe, "decode": de} for pe, de in endpoints],
                    "max_active_requests": active, "max_payload_bytes": 1_000_000,
                    "max_staging_bytes": staging, "max_request_bytes": 4096,
                    "request_timeout": timeout}
        gateway = PDGateway(config, settings)
        app = create_app(config, "tiny", pd_worker=gateway)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
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
            yield client, gateway, endpoints, tokenizer, logs, workers, port
    finally:
        if server:
            server.should_exit = True
        if task:
            await asyncio.wait_for(task, 20)
        for worker in reversed(workers):
            await worker.close()


def body(*, chat=False, stream=False, length=4, **extra):
    value = {"model": "tiny", "temperature": 0, "max_tokens": length,
             "ignore_eos": True, "stream": stream, **extra}
    if chat:
        value["messages"] = [{"role": "user", "content": "hello token2"}]
    else:
        value["prompt"] = "hello token2"
    return value


def parse_sse(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


async def drained(gateway, endpoints):
    for _ in range(300):
        stats = [await PDClient.stats(endpoint) for pair in endpoints for endpoint in pair]
        if gateway.claimed == 0 and all(s["open_handles"] == s["used_blocks"] == 0 for s in stats):
            assert all(s["reserved_transfer_bytes"] == s["queued_events"] == 0 for s in stats)
            assert gateway.staging_bytes == 0
            return
        await asyncio.sleep(.01)
    raise AssertionError(f"resources did not drain: {gateway.claimed}, {stats}")


def test_http_stream_and_nonstream_reuse_decoder_and_counts(tmp_path):
    async def run():
        async with service(tmp_path) as (client, gateway, endpoints, _, logs, _, _):
            for chat in (False, True):
                url = "/v1/chat/completions" if chat else "/v1/completions"
                ordinary = await client.post(url, json=body(chat=chat), headers={"x-request-id": "reused"})
                stream = await client.post(url, json=body(chat=chat, stream=True,
                                           stream_options={"include_usage": True}),
                                           headers={"x-request-id": "reused"})
                assert ordinary.status_code == stream.status_code == 200
                chunks = parse_sse(stream.text)
                choices = [row["choices"][0] for row in chunks if row["choices"]]
                assert len(choices) == chunks[-1]["usage"]["completion_tokens"] == 4
                assert chunks[-1]["choices"] == [] and stream.text.endswith("data: [DONE]\n\n")
                result = ordinary.json()
                text = "".join(c["delta"]["content"] if chat else c["text"] for c in choices)
                assert text == (result["choices"][0]["message"]["content"] if chat else result["choices"][0]["text"])
                assert result["usage"] == chunks[-1]["usage"]
                assert choices[-1]["finish_reason"] == result["choices"][0]["finish_reason"] == "length"
            await drained(gateway, endpoints)
            records = [json.loads(line) for line in logs.read_text().splitlines()]
            opened = [r for r in records if r["event"] == "pd_gateway_open"]
            assert len(opened) == len({r["transaction_id"] for r in opened}) == 4
            assert all(r["request_id"] == "reused" for r in opened)
    asyncio.run(run())


def test_terminal_first_token_eos_and_length_skip_transfer(tmp_path):
    async def run():
        async with service(tmp_path, forced_token=62) as (client, gateway, endpoints, _, logs, _, _):
            for request, reason in ((body(length=4, ignore_eos=False), "stop"),
                                    (body(length=1), "length")):
                response = await client.post("/v1/completions", json=request)
                assert response.status_code == 200
                assert response.json()["usage"]["completion_tokens"] == 1
                assert response.json()["choices"][0]["finish_reason"] == reason
            await drained(gateway, endpoints)
            assert not any(json.loads(line)["event"] in ("pd_gateway_handoff_start", "pd_transfer_done")
                           for line in logs.read_text().splitlines())
    asyncio.run(run())


def test_second_eos_ignore_eos_and_total_length_limit(tmp_path):
    async def run():
        async with service(tmp_path, forced_token=63) as (client, gateway, endpoints, _, _, _, _):
            stopped = await client.post("/v1/completions", json=body(length=4, ignore_eos=False))
            assert stopped.json()["usage"]["completion_tokens"] == 1
            assert stopped.json()["choices"][0]["finish_reason"] == "stop"
            ignored = await client.post("/v1/completions", json=body(length=2))
            assert ignored.json()["usage"]["completion_tokens"] == 2
            assert ignored.json()["choices"][0]["finish_reason"] == "length"
            limited = await client.post("/v1/completions", json={**body(length=5), "prompt": " ".join(["hello"] * 31)})
            assert limited.json()["usage"]["prompt_tokens"] == 31
            assert limited.json()["usage"]["completion_tokens"] == 1
            assert limited.json()["choices"][0]["finish_reason"] == "length"
            await drained(gateway, endpoints)
    asyncio.run(run())


def test_static_pairs_and_overlapping_different_budgets(tmp_path):
    async def run():
        async with service(tmp_path, pairs=2, delay=.02, staging=4_000_000) as (client, gateway, endpoints, _, logs, _, _):
            responses = await asyncio.gather(*(
                client.post("/v1/completions", json=body(length=length), headers={"x-request-id": "same"})
                for length in (2, 4, 6, 3)
            ))
            assert [r.json()["usage"]["completion_tokens"] for r in responses] == [2, 4, 6, 3]
            await drained(gateway, endpoints)
            opened = [json.loads(line) for line in logs.read_text().splitlines()
                      if json.loads(line)["event"] == "pd_gateway_assigned"]
            assert [r["prefill"] for r in opened] == [list(endpoints[i % 2][0]) for i in range(4)]
    asyncio.run(run())


def test_errors_before_headers_and_after_first_choice(tmp_path):
    async def run():
        async with service(tmp_path, sequences=1) as (client, gateway, endpoints, _, _, _, _):
            pe, de = endpoints[0]
            held = [await PDClient.open_d(de, lifetime=4) for _ in range(4)]
            try:
                for stream in (False, True):
                    r = await client.post("/v1/completions", json=body(stream=stream))
                    assert r.status_code == 503 and "error" in r.json()
            finally:
                await asyncio.gather(*(d.close() for d in held))
            await drained(gateway, endpoints)
            d = await PDClient.open_d(de, lifetime=4)
            p = await PDClient.open_p(pe, d.session_id, "held", [1],
                                      SamplingParams(temperature=0, max_tokens=4, ignore_eos=True), lifetime=4)
            try:
                await p.next_token()
                await d.reserve(await p.describe())
                r = await client.post("/v1/completions", json=body(stream=True,
                                      stream_options={"include_usage": True}))
                assert r.status_code == 200
                chunks = parse_sse(r.text)
                assert len([row for row in chunks if row.get("choices")]) == 1
                assert chunks[-1]["error"]["code"] == 503
                assert "[DONE]" not in r.text and not any("usage" in row for row in chunks)
                r = await client.post("/v1/completions", json=body())
                assert r.status_code == 503 and "error" in r.json()
            finally:
                await p.close()
                await d.close()
            await drained(gateway, endpoints)
            assert (await client.post("/v1/completions", json=body())).status_code == 200
    asyncio.run(run())


def test_timeout_is_per_request_and_gateway_recovers(tmp_path):
    async def run():
        async with service(tmp_path, delay=.3, timeout=.1) as (client, gateway, endpoints, _, _, _, _):
            r = await client.post("/v1/completions", json=body(stream=True))
            assert r.status_code == 504 and "error" in r.json()
            await drained(gateway, endpoints)
            r = await client.post("/v1/completions", json={**body(), "prompt": ""})
            assert r.status_code == 400
    asyncio.run(run())


def test_disconnect_before_headers_and_during_stream_reclaims_both_peers(tmp_path):
    async def run():
        async with service(tmp_path, delay=.08) as (client, gateway, endpoints, _, _, _, port):
            for stream in (False, True):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                data = json.dumps(body(stream=stream, length=20)).encode()
                writer.write(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: "
                             + str(len(data)).encode() + b"\r\n\r\n" + data)
                await writer.drain()
                for _ in range(100):
                    if gateway.requests:
                        break
                    await asyncio.sleep(.005)
                assert gateway.requests
                writer.transport.abort()
                await writer.wait_closed()
                await drained(gateway, endpoints)
            async with client.stream("POST", "/v1/completions", json=body(stream=True, length=20)) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        break
            await drained(gateway, endpoints)
            assert (await client.post("/v1/completions", json=body(length=2))).status_code == 200
    asyncio.run(run())


def test_gateway_rejects_full_request_capacity_and_unavailable_peer(tmp_path):
    async def run():
        async with service(tmp_path, delay=.15, active=1) as (client, gateway, endpoints, _, _, workers, _):
            first = asyncio.create_task(client.post("/v1/completions", json=body(length=3)))
            for _ in range(100):
                if gateway.requests:
                    break
                await asyncio.sleep(.005)
            response = await client.post("/v1/completions", json=body(length=1))
            assert response.status_code == 503
            assert (await first).status_code == 200
            await drained(gateway, endpoints)
            await workers[-1].close()
            r = await client.post("/v1/completions", json=body(length=1))
            assert r.status_code == 503 and "error" in r.json()
    asyncio.run(run())


def test_gateway_rejects_startup_model_and_fixed_pair_mismatch(tmp_path):
    import pytest

    async def run():
        async with service(tmp_path, pairs=2) as (_, gateway, endpoints, _, _, _, _):
            settings = {"pairs": [{"prefill": endpoints[0][0], "decode": endpoints[0][1]}],
                        "max_active_requests": 2, "max_payload_bytes": 1_000_000,
                        "max_staging_bytes": 1_000_000, "max_request_bytes": 4096,
                        "request_timeout": 5}
            for field, value in (("eos", [63]), ("model_revision", "c" * 40),
                                 ("dtype", "bfloat16"), ("head_dim", 8),
                                 ("max_model_length", 31)):
                bad = PDGateway({**gateway.config, field: value}, settings)
                with pytest.raises(ValueError, match="Incompatible"):
                    await bad.start()
            mismatched_pair = {**settings, "pairs": [{"prefill": endpoints[0][0], "decode": endpoints[1][1]}]}
            with pytest.raises(ValueError, match="fixed decode endpoint"):
                await PDGateway(gateway.config, mismatched_pair).start()
    asyncio.run(run())


def test_request_body_limit_releases_admission(tmp_path):
    async def run():
        async with service(tmp_path, active=1) as (client, gateway, endpoints, _, _, _, _):
            response = await client.post("/v1/completions", content=b"x" * 4097,
                                         headers={"content-type": "application/json"})
            assert response.status_code == 413
            await drained(gateway, endpoints)
            assert (await client.post("/v1/completions", json=body(length=1))).status_code == 200
    asyncio.run(run())


def test_gateway_close_during_active_request_completes_and_releases_resources(tmp_path):
    async def run():
        async with service(tmp_path, delay=.25) as (client, gateway, endpoints, _, _, _, _):
            request = asyncio.create_task(client.post("/v1/completions", json=body(length=10)))
            for _ in range(100):
                if (gateway.requests and
                        (await PDClient.stats(endpoints[0][0]))["open_handles"] and
                        (await PDClient.stats(endpoints[0][1]))["open_handles"]):
                    break
                await asyncio.sleep(.005)
            else:
                raise AssertionError("P and D did not both admit the request")
            await gateway.close()
            response = await asyncio.wait_for(request, 2)
            assert response.status_code == 503 and "error" in response.json()
            await drained(gateway, endpoints)
            assert (await client.post("/v1/completions", json=body(length=1))).status_code == 503
    asyncio.run(run())
