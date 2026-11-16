"""Real tiny GPU workers, gateway CLI and OpenAI-compatible clients."""

import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import aiohttp
import httpx
from openai import OpenAI
import pytest
import torch
from transformers import AutoTokenizer

from myvllm.pd.client import PDClient
from myvllm.sampling_parameters import SamplingParams
from pd_helpers import create_tiny_qwen_checkpoint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_tiny_gpu_real_http_pd_sdk_and_locked_bench(tmp_path):
    from vllm.benchmarks.lib.endpoint_request_func import (
        RequestFuncInput, async_request_openai_completions,
    )

    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(max_num_sequences=2, max_num_batched_tokens=16, max_model_length=16,
                  max_cached_blocks=8, gpu_memory_utilization=.1, seed=0,
                  timing_path=str(tmp_path / "worker-timing.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tokenizer.save_pretrained(config["model_name_or_path"])
    config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps(config, indent=2))
    pe, de = ("127.0.0.1", free_port()), ("127.0.0.1", free_port())
    http_port = free_port()
    settings = {"pairs": [{"prefill": pe, "decode": de}], "max_active_requests": 4,
                "max_payload_bytes": 1_000_000, "max_staging_bytes": 2_000_000,
                "max_request_bytes": 4096, "request_timeout": 90}
    gateway_path = tmp_path / "gateway.json"
    gateway_path.write_text(json.dumps(settings, indent=2))
    base_url = f"http://127.0.0.1:{http_port}"
    processes = []
    logs = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OPENAI_API_KEY"] = "test"
    env["TOKENIZERS_PARALLELISM"] = "false"
    common = ["--config", str(config_path), "--host", "127.0.0.1", "--max-connections", "4",
              "--max-transfer-bytes", "1000000", "--request-timeout", "120"]

    def start(name, args, worker_env):
        log = (tmp_path / f"{name}.log").open("w", encoding="utf-8")
        logs.append(log)
        process = subprocess.Popen([sys.executable, "-m", *args], stdout=log, stderr=subprocess.STDOUT,
                                   env=worker_env)
        processes.append((name, process))
        return process

    def wait_worker(endpoint, process):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            assert process.poll() is None, f"worker exited: {process.returncode}"
            try:
                return asyncio.run(PDClient.info(endpoint, timeout=.5))
            except (OSError, ConnectionError, TimeoutError, asyncio.IncompleteReadError):
                time.sleep(.2)
        raise AssertionError("worker did not become ready")

    try:
        dproc = start("decode", ["myvllm.pd.worker", "--role", "decode", *common, "--port", str(de[1])], env)
        assert wait_worker(de, dproc)["role"] == "decode"
        pproc = start("prefill", ["myvllm.pd.worker", "--role", "prefill", *common, "--port", str(pe[1]),
                                  "--decode-host", de[0], "--decode-port", str(de[1])], env)
        assert wait_worker(pe, pproc)["role"] == "prefill"
        gateway_env = dict(env, CUDA_VISIBLE_DEVICES="")
        gateway = start("gateway", ["myvllm.pd.gateway", "--config", str(config_path),
                                    "--gateway-config", str(gateway_path), "--served-model-name", "tiny",
                                    "--host", "127.0.0.1", "--port", str(http_port)], gateway_env)
        deadline = time.monotonic() + 120
        with httpx.Client(base_url=base_url, timeout=30, trust_env=False) as client:
            while time.monotonic() < deadline:
                assert gateway.poll() is None, f"gateway exited: {gateway.returncode}"
                try:
                    if client.get("/openapi.json").status_code == 200:
                        break
                except httpx.ConnectError:
                    time.sleep(.2)
            else:
                raise AssertionError("gateway did not become ready")
            request = {"model": "tiny", "prompt": "hello token2", "temperature": 0,
                       "max_tokens": 4, "ignore_eos": True}
            ordinary = client.post("/v1/completions", json=request)
            stream = client.post("/v1/completions", json={**request, "stream": True,
                                 "stream_options": {"include_usage": True}})
            assert ordinary.status_code == stream.status_code == 200
            (tmp_path / "ordinary.json").write_text(json.dumps(ordinary.json(), indent=2))
            (tmp_path / "stream.sse").write_text(stream.text)
            chunks = [json.loads(line[6:]) for line in stream.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            choices = [row["choices"][0] for row in chunks if row["choices"]]
            assert len(choices) == chunks[-1]["usage"]["completion_tokens"] == 4
            assert stream.text.endswith("data: [DONE]\n\n")
            assert "".join(choice["text"] for choice in choices) == ordinary.json()["choices"][0]["text"]
            with OpenAI(base_url=base_url + "/v1", api_key="test",
                        http_client=httpx.Client(trust_env=False, timeout=30)) as sdk:
                chat = sdk.chat.completions.create(
                    model="tiny", messages=[{"role": "user", "content": "hello"}],
                    temperature=0, max_tokens=3, extra_body={"ignore_eos": True},
                )
                assert chat.usage.completion_tokens == 3 and chat.choices[0].finish_reason == "length"
                (tmp_path / "sdk-chat.json").write_text(chat.model_dump_json(indent=2))

        async def client_checks():
            async with aiohttp.ClientSession() as session:
                result = await async_request_openai_completions(RequestFuncInput(
                    prompt="hello token2", api_url=base_url + "/v1/completions",
                    prompt_len=len(tokenizer.encode("hello token2")), output_len=4,
                    model="tiny", ignore_eos=True, extra_body={"temperature": 0},
                    request_id="http-gpu-bench",
                ), session)
            assert result.success, result.error
            assert result.output_tokens == 4 and len(result.itl) == 3
            (tmp_path / "locked-client.json").write_text(json.dumps({
                "success": result.success, "output_tokens": result.output_tokens,
                "itl": result.itl, "generated_text": result.generated_text,
            }, indent=2))
            d = await PDClient.open_d(de, lifetime=90)
            p = None
            try:
                p = await PDClient.open_p(pe, d.session_id, "direct-reference", tokenizer.encode("hello token2"),
                                          SamplingParams(temperature=0, max_tokens=4, ignore_eos=True,
                                                         max_model_length=16), lifetime=90)
                event = await p.next_token()
                tokens = [event["token"]]
                if not event["finished"]:
                    await PDClient.handoff(p, d, await p.describe())
                    while True:
                        event = await d.next_token()
                        tokens.append(event["token"])
                        if event["finished"]:
                            break
            finally:
                if p:
                    await p.close()
                await d.close()
            assert len(tokens) == 4
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            assert text == ordinary.json()["choices"][0]["text"]
            (tmp_path / "direct-tokens.json").write_text(json.dumps({"token_ids": tokens, "text": text}, indent=2))
            final_stats = {}
            for role, endpoint in (("prefill", pe), ("decode", de)):
                stats = await PDClient.stats(endpoint)
                assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
                final_stats[role] = stats
            (tmp_path / "worker-stats.json").write_text(json.dumps(final_stats, indent=2))

        asyncio.run(client_checks())
    finally:
        exits = []
        for name, process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            exits.append({"name": name, "pid": process.pid, "returncode": process.returncode,
                          "exited": process.poll() is not None})
        for log in logs:
            log.close()
        (tmp_path / "process-exits.json").write_text(json.dumps(exits, indent=2))
        failures = [item for item in exits if not item["exited"] or (
            item["returncode"] != 0 and not
            (item["name"] == "gateway" and item["returncode"] == -signal.SIGTERM)
        )]
        gateway_exit = next((item for item in exits if item["name"] == "gateway"), None)
        if gateway_exit and gateway_exit["returncode"] == -signal.SIGTERM:
            text = (tmp_path / "gateway.log").read_text()
            if "Application shutdown complete." not in text or "Finished server process" not in text:
                failures.append(gateway_exit)
        assert not failures, f"children failed to shut down: {failures}; see {tmp_path}"
