"""TCP and mixed-batch boundaries for FP8 prefill computation."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from fp8_helpers import quantize_reference, tolerances
from myvllm.engine.input_batch import build_prefill_batch
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.layers.attention import Attention
from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.gateway import _header_bytes, _metadata_header_bounds
from myvllm.pd.ready import check as check_ready
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, close_writer, read_body, read_header, write_frame
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import reset_context, set_context
from pd_pipeline_helpers import empty_pipeline
from test_fp8_prefill_compute import cpu_fp8_engine, reference
from test_pd_load_routing import http_gateway, load_engine, records, settings
from test_pd_server import body, drained, make_config


def test_worker_large_info_and_oversized_inbound(tmp_path):
    async def run():
        config, _ = make_config(tmp_path / ("long-model-path-" + "x" * 100))
        config["enforce_eager"] = True
        options = dict(max_connections=8, max_transfer_bytes=1_000_000,
                       request_timeout=6, engine_factory=load_engine)
        decodes = []
        workers = []
        try:
            for seed in range(2):
                worker = PDWorker("decode", {**config, "seed": seed}, **options)
                decodes.append(await worker.start("127.0.0.1", 0))
                workers.append(worker)
            worker = PDWorker("prefill", config, max_header_bytes=1024,
                              decode_endpoint=decodes[0], allowed_decode_endpoints=decodes,
                              **options)
            endpoint = await worker.start("127.0.0.1", 0)
            workers.append(worker)
            info = await PDClient.info(endpoint)
            encoded = json.dumps({"ok": True, "value": info}, separators=(",", ":")).encode()
            assert 1024 < len(encoded) <= DEFAULT_HEADER_LIMIT
            assert info["max_header_bytes"] == 1024
            assert info["fp8_prefill_compute_dtype"] == "float32"

            reader, writer = await asyncio.open_connection(*endpoint)
            try:
                await write_frame(writer, {"op": "info", "oversized": "x" * 1024})
                response, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert not size and response["ok"] is False
                assert response["error"] == "ProtocolError"
                assert "control frame size" in response["message"].lower()
            finally:
                await close_writer(writer, asyncio.get_running_loop().time() + 2)
            assert (await PDClient.info(endpoint))["max_header_bytes"] == 1024
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())


def test_worker_error_response_uses_public_limit(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        worker = PDWorker("decode", config, max_connections=4, max_transfer_bytes=1_000_000,
                          max_header_bytes=16, request_timeout=6, engine_factory=load_engine)
        endpoint = await worker.start("127.0.0.1", 0)
        try:
            reader, writer = await asyncio.open_connection(*endpoint)
            try:
                assert _header_bytes({"op": "invalid"}) <= 16
                await write_frame(writer, {"op": "invalid"})
                response, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 0)
                assert not size and response["ok"] is False
                assert response["error"] == "ProtocolError"
                assert _header_bytes(response) > 16
            finally:
                await close_writer(writer, asyncio.get_running_loop().time() + 2)
            assert (await PDClient.info(endpoint))["max_header_bytes"] == 16
        finally:
            await worker.close()

    asyncio.run(run())


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("mode", ["float32", "query"])
def test_load_routes_response_larger_than_prefill_inbound_limit(tmp_path, pipeline, mode):
    async def run():
        config, _ = make_config(tmp_path / ("long-model-path-" + "x" * 100))
        config.update(head_dim=3, kv_cache_dtype="float8_e4m3fn", enforce_eager=True,
                      fp8_prefill_compute_dtype=mode)
        options = dict(max_connections=4, max_transfer_bytes=1_000_000,
                       request_timeout=6, pipeline_frame_bytes=28,
                       engine_factory=cpu_fp8_engine)
        d = PDWorker("decode", config, **options)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", config, max_header_bytes=1024, decode_endpoint=de, **options)
        pe = await p.start("127.0.0.1", 0)
        pairs = [(pe, de)]
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True,
                                max_model_length=32)
        prompt = [1, 2, 3, 4, 5]
        request_id = "response-larger-than-inbound"
        try:
            cfg = settings(pairs, poll_interval=.01, pipeline=2 if pipeline else 0)
            readiness = await check_ready(cfg, None)
            pinfo = readiness["pairs"][0]["prefill"]
            assert pinfo["max_header_bytes"] == 1024
            assert _header_bytes({"ok": True, "value": pinfo}) > 1024
            open_header = {"op": "prepare_p" if pipeline else "open_p", "lifetime": 5,
                           "session_id": "0" * 32, "request_id": request_id, "token_ids": prompt,
                           "sampling": asdict(params), "decode_endpoint": de}
            assert _header_bytes(open_header) < 1024

            dc = await PDClient.open_d(de, lifetime=5)
            pc = await (PDClient.prepare_p if pipeline else PDClient.open_p)(
                pe, dc.session_id, request_id, prompt, params, lifetime=5,
                decode_endpoint=de, max_header_bytes=1024)
            try:
                if pipeline:
                    metadata = await pc.describe_early()
                else:
                    assert not (await pc.next_token())["finished"]
                    metadata = await pc.describe()
                response_bytes = _header_bytes({"ok": True, "value": metadata})
                request_bytes = _header_bytes({
                    "op": "reserve_early" if pipeline else "reserve", "metadata": metadata})
                assert 1024 < response_bytes <= DEFAULT_HEADER_LIMIT
                assert request_bytes <= DEFAULT_HEADER_LIMIT
                print(f"mode={mode} pipeline={pipeline} open={_header_bytes(open_header)} "
                      f"response={response_bytes} reserve={request_bytes}", flush=True)
            finally:
                await pc.close()
                await dc.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                response = await client.post(
                    "/v1/completions",
                    json={**body(length=4), "prompt": "hello token2 token3 token4 token5"},
                    headers={"x-request-id": request_id})
                assert response.status_code == 200, response.text
                assert response.json()["usage"]["completion_tokens"] == 4
                assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                assert assigned["route"] == "remote"
                assert tuple(assigned["prefill"]) == pe and tuple(assigned["decode"]) == de
                assert records(tmp_path / "gateway.jsonl", "pd_gateway_handoff_done")
                await drained(gateway, pairs)
        finally:
            await p.close()
            await d.close()

    asyncio.run(run())


@asynccontextmanager
async def old_response_limit_proxy(endpoint, limit):
    tasks = set()

    async def relay(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        upstream = None
        try:
            remote, upstream = await asyncio.open_connection(*endpoint)
            while True:
                request, size = await read_header(reader, DEFAULT_HEADER_LIMIT, 1_000_000)
                await write_frame(upstream, request, await read_body(reader, size))
                response, size = await read_header(remote, DEFAULT_HEADER_LIMIT, 1_000_000)
                payload = await read_body(remote, size)
                if request["op"] == "info" and response["ok"]:
                    response["value"].pop("fp8_prefill_compute_dtype")
                if _header_bytes(response) > limit:
                    await write_frame(writer, {"ok": False, "error": "ProtocolError",
                                               "message": "Control frame exceeds limit"})
                    break
                await write_frame(writer, response, payload)
                if request["op"] in ("info", "stats", "close"):
                    break
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


def test_load_preserves_legacy_prefill_response_bound(tmp_path):
    async def run():
        config, _ = make_config(tmp_path)
        config["enforce_eager"] = True
        options = dict(max_connections=4, max_transfer_bytes=1_000_000,
                       request_timeout=6, engine_factory=load_engine)
        d = PDWorker("decode", config, **options)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", config, max_header_bytes=1024, decode_endpoint=de, **options)
        pe = await p.start("127.0.0.1", 0)
        try:
            async with old_response_limit_proxy(pe, 1024) as legacy_pe:
                info = await PDClient.info(legacy_pe)
                assert "fp8_prefill_compute_dtype" not in info
                assert _header_bytes({"ok": True, "value": info}) <= 1024
                params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True,
                                        max_model_length=32)
                prompt = [1, 2]
                for count in range(1, 800):
                    request_id = "legacy-" + "x" * count
                    pending = SimpleNamespace(token_ids=prompt, params=params, request_id=request_id)
                    pmeta, dmeta = _metadata_header_bounds(pending, info, 32, False)
                    popen = _header_bytes({
                        "op": "open_p", "lifetime": 5, "session_id": "0" * 32,
                        "request_id": request_id, "token_ids": prompt, "sampling": asdict(params),
                        "decode_endpoint": de,
                    })
                    if popen < 1024 < pmeta and dmeta < DEFAULT_HEADER_LIMIT:
                        break
                else:
                    raise AssertionError("No legacy boundary request found")
                pairs = [(legacy_pe, de)]
                async with http_gateway(config, settings(pairs, poll_interval=.01), tmp_path) as (
                        client, gateway, _):
                    response = await client.post("/v1/completions", json=body(length=4),
                                                 headers={"x-request-id": request_id})
                    assert response.status_code == 503
                    assert not records(tmp_path / "gateway.jsonl", "pd_gateway_open")
                    response = await client.post("/v1/completions", json=body(length=1))
                    assert response.status_code == 200, response.text
                    await drained(gateway, pairs)
        finally:
            await p.close()
            await d.close()

    asyncio.run(run())


@pytest.mark.parametrize("d_mode", ["float32", "query"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_direct_tcp_mode_rejection_cleanup_and_matching_requests(tmp_path, d_mode, pipeline):
    async def run():
        config, _ = make_config(tmp_path / "tokenizer")
        config.update(head_dim=3, kv_cache_dtype="float8_e4m3fn", enforce_eager=True)
        options = dict(max_connections=4, max_transfer_bytes=1_000_000,
                       request_timeout=6, pipeline_frame_bytes=28,
                       engine_factory=cpu_fp8_engine)
        # Omission exercises the original float32 identity and metadata shape.
        d_config = {**config, "enable_chunked_prefill": True}
        if d_mode == "query":
            d_config["fp8_prefill_compute_dtype"] = d_mode
        d = PDWorker("decode", d_config, **options)
        de = await d.start("127.0.0.1", 0)
        other_mode = "float32" if d_mode == "query" else "query"
        wrong = PDWorker("prefill", {**config, "fp8_prefill_compute_dtype": other_mode},
                         decode_endpoint=de, **options)
        we = await wrong.start("127.0.0.1", 0)
        matching_config = dict(config)
        if d_mode == "query":
            matching_config["fp8_prefill_compute_dtype"] = d_mode
        matching = PDWorker("prefill", matching_config, decode_endpoint=de, **options)
        pe = await matching.start("127.0.0.1", 0)
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        prompt = [1, 2, 3, 4, 5]
        try:
            pinfo, dinfo = await asyncio.gather(PDClient.info(we), PDClient.info(de))
            dc = await PDClient.open_d(de, lifetime=5)
            pc = await (PDClient.prepare_p if pipeline else PDClient.open_p)(
                we, dc.session_id, "mismatch", prompt, params, lifetime=5)
            try:
                if pipeline:
                    metadata = await pc.describe_early()
                else:
                    assert not (await pc.next_token())["finished"]
                    metadata = await pc.describe()
                with pytest.raises(PDRemoteError, match="Incompatible model"):
                    await (dc.reserve_early(metadata) if pipeline else dc.reserve(metadata))
                assert pinfo["model"] != dinfo["model"]
                for info, mode in ((pinfo, other_mode), (dinfo, d_mode)):
                    cache = info["model"]["kv_cache"]
                    assert cache.get("fp8_prefill_compute_dtype", "float32") == mode
                stats = await PDClient.stats(de)
                assert stats["used_blocks"] == stats["reserved_blocks"] == 0
                assert stats["reserved_transfer_bytes"] == stats["resident_requests"] == 0
            finally:
                await pc.close()
                await dc.close()
            await empty_pipeline(we)
            await empty_pipeline(de)

            dc = await PDClient.open_d(de, lifetime=5)
            pc = await (PDClient.prepare_p if pipeline else PDClient.open_p)(
                pe, dc.session_id, "matching", prompt, params, lifetime=5)
            try:
                if pipeline:
                    await PDClient.start_pipeline(pc, dc)
                event = await pc.next_token()
                assert not event["finished"]
                metadata = await pc.describe()
                if d_mode == "float32":
                    assert metadata["model"]["kv_cache"] == {
                        "dtype": "float8_e4m3fn", "scale_dtype": "float32",
                        "scale_granularity": "token_head"}
                    assert json.dumps(metadata["model"]["kv_cache"], separators=(",", ":")).encode() == (
                        b'{"dtype":"float8_e4m3fn","scale_dtype":"float32",'
                        b'"scale_granularity":"token_head"}')
                if pipeline:
                    await PDClient.handoff_pipeline(pc, dc, metadata)
                else:
                    await PDClient.handoff(pc, dc, metadata)
                count = 1
                while not event["finished"]:
                    event = await dc.next_token()
                    count += 1
                assert count == 4
            finally:
                await pc.close()
                await dc.close()
            await empty_pipeline(pe)
            await empty_pipeline(de)

            local = await PDClient.open_local(de, "local", prompt, params, lifetime=5)
            try:
                count = 0
                while True:
                    event = await local.next_token()
                    count += 1
                    if event["finished"]:
                        break
                assert count == 4
            finally:
                await local.close()
            await empty_pipeline(de)
        finally:
            await matching.close()
            await wrong.close()
            await d.close()

    asyncio.run(run())


def mixed_interpreter_checks():
    torch.set_num_threads(1)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    for dtype in (torch.float32, torch.float16):
        scheduler = Scheduler(2, 4, 8, 4, [], 16, enable_chunked_prefill=True)
        running = Sequence([1, 2, 3], 4, params)
        scheduler.add_sequence(running)
        initial, is_prefill = scheduler.schedule()
        assert initial == [running] and is_prefill
        scheduler.postprocess(initial, [10])
        incoming = Sequence([4, 5, 6], 4, params)
        scheduler.add_sequence(incoming)
        batch, is_prefill = scheduler.schedule()
        assert batch == [running, incoming] and is_prefill
        assert [seq.num_scheduled_tokens for seq in batch] == [1, 3]
        assert [seq.num_completion_tokens for seq in batch] == [1, 0]
        meta = build_prefill_batch(batch, 4)

        generator = torch.Generator().manual_seed(20261117)
        shape = (8, 4, 1, 16)
        kc = torch.zeros(shape, dtype=torch.float8_e4m3fn)
        vc = torch.zeros_like(kc)
        ks, vs = torch.ones(shape[:-1]), torch.ones(shape[:-1])
        for seq in batch:
            length = seq.num_cached_tokens + seq.num_scheduled_tokens
            for cache, scales in ((kc, ks), (vc, vs)):
                source = (torch.randn(length, 1, 16, generator=generator) * 3).to(dtype)
                data, scale = quantize_reference(source)
                slots = torch.tensor([seq.block_table[pos // 4] * 4 + pos % 4
                                      for pos in range(length)])
                cache.view(torch.uint8).flatten(0, 1)[slots] = data.view(torch.uint8)
                scales.flatten(0, 1)[slots] = scale
        q = torch.randn(4, 2, 16, generator=generator, dtype=dtype)
        layer = Attention(2, 16, num_kv_heads=1, block_size=4)
        layer.k_cache, layer.v_cache, layer.k_scale_cache, layer.v_scale_cache = kc, vc, ks, vs
        tables = torch.tensor(meta["block_tables"], dtype=torch.int32)
        cuq = torch.tensor(meta["cu_seqlens_q"], dtype=torch.int32)
        cuk = torch.tensor(meta["cu_seqlens_k"], dtype=torch.int32)
        results = {}
        for mode in ("float32", "query"):
            layer.fp8_prefill_compute_dtype = mode
            set_context(True, cu_seqlens_q=cuq, cu_seqlens_k=cuk,
                        max_seqlen_q=meta["max_seqlen_q"], block_tables=tables)
            try:
                mixed = layer(q, torch.empty(0), torch.empty(0)).reshape_as(q)
            finally:
                reset_context()
            expected = reference((q, kc, vc, ks, vs, cuq, cuk, tables),
                                 (1, 3), (4, 3), 4, mode)
            torch.testing.assert_close(mixed, expected, **tolerances(dtype))
            set_context(False, block_tables=tables[:1],
                        context_lens=torch.tensor([len(running)], dtype=torch.int32))
            try:
                isolated = layer(q[:1], torch.empty(0), torch.empty(0)).reshape_as(q[:1])
            finally:
                reset_context()
            results[mode] = mixed, isolated
        assert torch.equal(results["float32"][1], results["query"][1])
        if dtype == torch.float16:
            assert not torch.equal(results["float32"][0][0], results["query"][0][0])
        else:
            assert torch.equal(results["float32"][0], results["query"][0])
        print(f"mixed scheduler dtype={dtype}: passed", flush=True)

    tail_scheduler = Scheduler(1, 2, 4, 4, [], 8, enable_chunked_prefill=True)
    tail = Sequence([1, 2, 3], 4, params)
    tail_scheduler.add_sequence(tail)
    first, first_prefill = tail_scheduler.schedule()
    assert first_prefill and first[0].num_scheduled_tokens == 2
    tail_scheduler.postprocess(first, [])
    second, tail_prefill = tail_scheduler.schedule()
    assert second == [tail] and tail.num_scheduled_tokens == 1 and not tail_prefill
    assert tail.num_completion_tokens == 0
    tail_table = torch.tensor([tail.block_table], dtype=torch.int32)
    tail_outputs = []
    for mode in ("float32", "query"):
        layer.fp8_prefill_compute_dtype = mode
        set_context(tail_prefill, block_tables=tail_table,
                    context_lens=torch.tensor([len(tail)], dtype=torch.int32))
        try:
            tail_outputs.append(layer(q[:1], torch.empty(0), torch.empty(0)).reshape_as(q[:1]))
        finally:
            reset_context()
    assert torch.equal(*tail_outputs)
    cu_tail_q = torch.tensor([0, 1], dtype=torch.int32)
    cu_tail_k = torch.tensor([0, len(tail)], dtype=torch.int32)
    expected = reference((q[:1], kc, vc, ks, vs, cu_tail_q, cu_tail_k, tail_table),
                         (1,), (len(tail),), 4, "float32")
    torch.testing.assert_close(tail_outputs[0], expected, **tolerances(q.dtype))
    print("single-token prompt tail uses decode dispatch: passed", flush=True)


def test_real_scheduler_mixed_batch_in_cpu_interpreter():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--mixed-interpreter"],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "mixed scheduler dtype=torch.float16: passed" in result.stdout
    print(result.stdout)


if __name__ == "__main__" and sys.argv[1:] == ["--mixed-interpreter"]:
    mixed_interpreter_checks()
