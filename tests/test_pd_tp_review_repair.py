"""Regressions for legacy fixed workers, failed TP2 owners and capture limits."""

import asyncio
import gc
import hashlib
import io
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback
import weakref
from unittest.mock import patch

import pytest
import torch.distributed as dist

from myvllm.engine.llm_engine import LLMEngine
from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway
from myvllm.pd.ready import check
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from test_compare_pd import ROOT, capture_args
from test_pd_server import body
from tools import compare_pd


M5_SHA = "4b16c312159d0dfdc8ebf3bc1d7b70c078e73ffb"


@pytest.fixture(scope="module")
def legacy_source(tmp_path_factory):
    provided = os.environ.get("M8_LEGACY_SOURCE")
    if provided:
        source = Path(provided).resolve()
    else:
        source = tmp_path_factory.mktemp("m5-source")
        archive = subprocess.check_output(["git", "-C", str(ROOT), "archive", M5_SHA])
        with tarfile.open(fileobj=io.BytesIO(archive)) as files:
            files.extractall(source, filter="data")
    tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "-rz", M5_SHA, "--", "src/myvllm"])
    expected = set()
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, kind, digest = header.decode().split()
        assert mode == "100644" and kind == "blob"
        name = name.decode()
        expected.add(name)
        data = (source / name).read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert actual == digest, name
    actual = {str(path.relative_to(source)) for path in (source / "src/myvllm").rglob("*.py")}
    assert actual == {name for name in expected if name.endswith(".py")}
    return source


def test_legacy_fixed_gateway_and_ready_over_real_tcp(tmp_path, legacy_source):
    from test_pd_load_routing import http_gateway, settings

    async def run():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        child = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "tests/pd_tp_review_repair_legacy.py"),
            str(legacy_source), str(tmp_path), env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        try:
            async with asyncio.timeout(30):
                while True:
                    line = await child.stdout.readline()
                    if not line:
                        raise RuntimeError("Legacy worker exited before ready")
                    if line.startswith(b"READY="):
                        data = json.loads(line.removeprefix(b"READY="))
                        break
            pe, de, bad_pe = map(tuple, (data["p"], data["d"], data["bad_p"]))
            pi, di = await asyncio.gather(PDClient.info(pe), PDClient.info(de))
            assert "world_size" not in pi and "world_size" not in di
            assert "pipeline_frame_bytes" not in pi
            cfg = {"pairs": [{"prefill": list(pe), "decode": list(de)}],
                   "max_active_requests": 4, "max_payload_bytes": 1000000,
                   "max_staging_bytes": 1000000, "max_request_bytes": 4096,
                   "request_timeout": 5}
            ready = await check(cfg, None)
            assert ready["pairs"][0]["prefill"] == pi
            async with http_gateway(data["config"], cfg, tmp_path) as (client, _, _):
                response = await client.post("/v1/completions", json=body(length=3))
                assert response.status_code == 200, response.text
                assert response.json()["usage"]["completion_tokens"] == 3

            invalid = [
                ({"model_revision": "f" * 40}, cfg, "model or length"),
                ({}, {**cfg, "request_timeout": 11}, "timeout"),
                ({}, {**cfg, "max_payload_bytes": 999999}, "payload"),
                ({}, {**cfg, "pipeline_min_tokens": 1}, "Pipeline"),
                ({}, {**cfg, "pairs": [{"prefill": list(bad_pe), "decode": list(de)}]}, "seed"),
            ]
            for config_change, settings_change, message in invalid:
                gateway = PDGateway({**data["config"], **config_change}, settings_change)
                with pytest.raises(ValueError, match=message):
                    await gateway.start()
                await gateway.close()
            load = settings([(pe, de)])
            with pytest.raises((KeyError, ValueError)):
                await PDGateway(data["config"], load).start()
            with pytest.raises((KeyError, ValueError)):
                await check(load, None)
        finally:
            if child.returncode is None:
                child.stdin.write(b"\n")
                await child.stdin.drain()
                try:
                    await asyncio.wait_for(child.wait(), 20)
                except TimeoutError:
                    child.kill()
                    await child.wait()
                    raise
            assert child.returncode == 0

    asyncio.run(run())


def test_failed_real_tp2_worker_releases_retained_traceback_objects_and_restarts(tmp_path):
    from pd_tp_bulk_helpers import cpu_tp_backend, decode_engine
    from test_pd_tp_bulk import config_for

    config = config_for(tmp_path, True)
    config.pop("_pd_tp_bulk")
    config["max_cached_blocks"] = 8
    context = mp.get_context("spawn")
    entered = context.Event()
    gate, release = context.Pipe(duplex=False)
    config.update(_export_entered=entered, _export_gate=gate, tp_command_timeout_s=10)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    staging_refs = []
    gather = dist.gather

    def observe_gather(tensor, gather_list=None, *args, **kwargs):
        if gather_list is not None:
            staging_refs.extend(weakref.ref(item) for item in (tensor, *gather_list))
        return gather(tensor, gather_list=gather_list, *args, **kwargs)

    async def run():
        workers = []
        prefill = decode = None
        p = None
        try:
            d = PDWorker("decode", {**config, "world_size": 1}, max_connections=4,
                         max_transfer_bytes=1000000, request_timeout=30, engine_factory=decode_engine)
            de = await d.start("127.0.0.1", 0)
            workers.append(d)
            p = PDWorker("prefill", config, max_connections=4, max_transfer_bytes=1000000,
                         request_timeout=30, decode_endpoint=de, engine_factory=LLMEngine)
            pe = await p.start("127.0.0.1", 0)
            workers.append(p)
            runner_ref = weakref.ref(p.owner.engine.model_runner)
            cache_ref = weakref.ref(p.owner.engine.model_runner.kv_cache)
            scales_ref = weakref.ref(p.owner.engine.model_runner.kv_cache_scales)
            decode = await PDClient.open_d(de, lifetime=30)
            prefill = await PDClient.open_p(pe, decode.session_id, "rank-loss", [1, 2, 3, 4, 5],
                                            params, lifetime=30)
            assert not (await prefill.next_token())["finished"]
            await decode.reserve(await prefill.describe())
            export = asyncio.create_task(prefill.export())
            assert await asyncio.to_thread(entered.wait, 6), "rank 1 did not consume export"
            process = p.owner.engine.processes[0]
            process.kill()
            await asyncio.to_thread(process.join, 5)
            with pytest.raises((OSError, RuntimeError, asyncio.IncompleteReadError, ConnectionError)):
                await asyncio.wait_for(export, 15)
            await asyncio.wait_for(p.failed.wait(), 10)
            failure = p.owner.error
            assert isinstance(failure, RuntimeError)
            failure.add_note("rank-loss regression note")
            original = (type(failure), str(failure), failure.__cause__, failure.__context__,
                        tuple(failure.__notes__))
            locations = [(item.filename, item.lineno, item.name)
                         for item in traceback.extract_tb(failure.__traceback__)]
            assert {"call", "export_pd_tp_kv", "export_tp_kv"} <= {item[2] for item in locations}
            failed_staging_refs = tuple(staging_refs)
            assert len(failed_staging_refs) == 3
            assert all(ref() is not None for ref in failed_staging_refs)
            await prefill.close()
            prefill = None
            await decode.close()
            decode = None
            with pytest.raises(RuntimeError) as raised:
                await p.close()
            assert raised.value is failure is p.owner.error
            workers.remove(p)
            assert not p.owner.thread.is_alive()
            assert not p.connections and p.sessions == 0
            assert p.owner.core.stats()["used_blocks"] == 0
            assert p.owner.engine.model_runner is None and not p.owner.engine.processes
            assert (type(failure), str(failure), failure.__cause__, failure.__context__,
                    tuple(failure.__notes__)) == original
            after = [(item.filename, item.lineno, item.name)
                     for item in traceback.extract_tb(failure.__traceback__)]
            assert all(location in after for location in locations)
            gc.collect()
            assert all(ref() is None for ref in (runner_ref, cache_ref, scales_ref, *failed_staging_refs))

            clean = {key: value for key, value in config.items()
                     if key not in ("_export_entered", "_export_gate")}
            restarted = PDWorker("prefill", clean, max_connections=4, max_transfer_bytes=1000000,
                                 request_timeout=30, decode_endpoint=de, engine_factory=LLMEngine)
            endpoint = await restarted.start("127.0.0.1", 0)
            workers.append(restarted)
            decode = await PDClient.open_d(de, lifetime=30)
            prefill = await PDClient.open_p(endpoint, decode.session_id, "restart", [1, 2, 3, 4, 5],
                                            params, lifetime=30)
            assert not (await prefill.next_token())["finished"]
            await PDClient.handoff(prefill, decode, await prefill.describe())
            while not (await decode.next_token())["finished"]:
                pass
            await prefill.close()
            prefill = None
            await decode.close()
            decode = None
            assert p.owner.error is failure
            gc.collect()
            assert all(ref() is None for ref in (runner_ref, cache_ref, scales_ref, *failed_staging_refs))
        finally:
            for connection in (prefill, decode):
                if connection is not None:
                    await connection.close()
            for worker in reversed(workers):
                try:
                    await worker.close()
                except RuntimeError:
                    if worker is not p:
                        raise

    logger = logging.getLogger("myvllm.pd.worker")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    try:
        # pytest's log capture retains the separate exit exception and its runner.
        with patch.object(logger, "propagate", False), cpu_tp_backend(config), patch.object(dist, "gather", observe_gather):
            asyncio.run(run())
    finally:
        logger.removeHandler(handler)
        handler.close()
        gate.close()
        release.close()


def test_capture_records_actual_case_bounded_worker_limits(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    compare_pd.capture(args)
    root = Path(args.output_dir)
    cases = json.loads((root / "cases.json").read_text())
    evidence = json.loads((root / "worker-info.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(cases) == 2
    assert evidence["case_count"] == len(cases)
    assert evidence["connection_budget"] == max(1, len(cases))
    assert "legacy" in evidence["rationale"].lower()
    for role in ("prefill", "decode"):
        assert evidence[role]["role"] == role
        assert evidence[role]["max_connections"] == len(cases)
    assert manifest["status"] == "complete"
    assert len(manifest["completed_cases"]) == len(cases)
    assert "worker-info.json" in manifest["visible_artifacts"]
