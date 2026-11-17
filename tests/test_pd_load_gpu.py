"""Opt-in four-GPU PD CLI smoke; all artifacts must be under /Data2/zincion/ai-infra."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from myvllm.pd.client import PDClient
from myvllm.sampling_parameters import SamplingParams
from pd_helpers import create_tiny_qwen_checkpoint


pytestmark = pytest.mark.skipif(
    os.environ.get("M8_FOUR_GPU_SMOKE") != "1", reason="four-GPU smoke requires explicit opt-in",
)
ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = Path("/Data2/zincion/ai-infra")
CLI = ROOT / "tests/pd_load_gpu_cli.py"


def nvidia_query(fields):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-" + fields[0] + "=" + ",".join(fields[1:]),
         "--format=csv,noheader,nounits"], text=True, timeout=15,
    )
    return [[item.strip() for item in line.split(",")] for line in output.splitlines() if line.strip()]


def ports(count):
    sockets = []
    try:
        for _ in range(count):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def records(path, request_id, event=None):
    if not path.exists():
        return []
    return [row for row in map(json.loads, path.read_text().splitlines())
            if row.get("request_id") == request_id and (event is None or row["event"] == event)]


def verify_candidate(expected):
    assert re.fullmatch(r"[0-9a-f]{40}", expected)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert head == expected
    for path in (Path(__file__).resolve(), CLI, ROOT / "tests/pd_helpers.py"):
        relative = path.relative_to(ROOT)
        content = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{head}:{relative}"])
        assert path.read_bytes() == content, f"{path} differs from the candidate"


async def wait_worker(endpoint, process, deadline):
    while time.monotonic() < deadline:
        assert process.poll() is None, f"worker exited with {process.returncode}"
        try:
            return await PDClient.info(endpoint, timeout=2)
        except (OSError, ConnectionError, TimeoutError, asyncio.IncompleteReadError):
            await asyncio.sleep(.2)
    raise TimeoutError(f"worker {endpoint} did not start")


async def wait_http(port, process):
    deadline = time.monotonic() + 120
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            assert process.poll() is None, f"gateway exited with {process.returncode}"
            try:
                if (await client.get("/openapi.json")).status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            await asyncio.sleep(.2)
    raise TimeoutError("gateway did not start")


async def empty_owners(endpoints):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        stats = {name: await PDClient.stats(endpoint, timeout=3) for name, endpoint in endpoints.items()}
        keys = ("open_connections", "open_handles", "used_blocks", "reserved_transfer_bytes",
                "queued_events", "pipeline_staging_bytes", "pipeline_pinned_bytes",
                "pipeline_busy_slots", "pipeline_slots")
        if all(all(row[key] == 0 for key in keys) and row["free_blocks"] == row["total_blocks"]
               and row.get("reserved_blocks", 0) == row.get("resident_requests", 0) == 0
               and row.get("local_resident_requests", 0) == row.get("remote_resident_requests", 0) == 0
               for row in stats.values()):
            return stats
        await asyncio.sleep(.1)
    raise AssertionError(f"owner resources did not drain: {stats}")


async def stop_processes(processes):
    errors = []
    for item in reversed(processes):
        process = item["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(f'{item["name"]} SIGTERM: {exc}')
    deadline = time.monotonic() + 25
    for item in reversed(processes):
        process = item["process"]
        if process.poll() is None:
            try:
                await asyncio.to_thread(process.wait, max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
            except OSError as exc:
                errors.append(f'{item["name"]} wait: {exc}')
    for item in processes:
        process = item["process"]
        if process.poll() is None:
            item["forced"] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(f'{item["name"]} SIGKILL: {exc}')
    for item in processes:
        process = item["process"]
        if process.poll() is None:
            try:
                await asyncio.to_thread(process.wait, 5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f'{item["name"]} final wait: {exc}')
        try:
            item["log"].close()
        except OSError as exc:
            errors.append(f'{item["name"]} log close: {exc}')
        item["returncode"] = process.returncode
    return ([{key: value for key, value in item.items() if key not in ("process", "log")}
             for item in processes], errors)


def test_four_gpu_real_cli_load_bulk_pipeline_and_local(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    root = SERVER_ROOT.resolve()
    assert tmp_path.resolve().is_relative_to(root), "pass --basetemp below /Data2/zincion/ai-infra"
    assert Path(os.environ["TMPDIR"]).resolve().is_relative_to(root)
    expected = os.environ["M8_GPU_EXPECTED_SHA"]
    verify_candidate(expected)
    inventory = nvidia_query(("gpu", "index", "uuid", "name", "pci.bus_id"))
    gpus = {int(index): {"uuid": uuid, "name": name, "pci_bus_id": bus}
            for index, uuid, name, bus in inventory}
    assert all(index in gpus for index in (0, 1, 3, 4))
    assert len({gpus[index]["uuid"] for index in (0, 1, 3, 4)}) == 4

    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(max_num_sequences=4, max_num_batched_tokens=24, max_model_length=24,
                  max_cached_blocks=24, gpu_memory_utilization=.1, eos=[63], enforce_eager=True,
                  dtype="bfloat16", scale=1.0)
    checkpoint_hashes = {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((tmp_path / "checkpoint").iterdir()) if path.is_file()}
    selected_ports = ports(7)
    endpoints = {name: ("127.0.0.1", port) for name, port in zip(
        ("p0", "p1", "d0", "d1"), selected_ports)}
    settings = {
        "pairs": [{"prefill": endpoints["p0"], "decode": endpoints["d0"]},
                  {"prefill": endpoints["p1"], "decode": endpoints["d1"]}],
        "routing_mode": "load", "max_active_requests": 8, "max_payload_bytes": 1_000_000,
        "max_staging_bytes": 2_000_000, "max_request_bytes": 4096, "request_timeout": 180,
        "poll_interval_s": .1, "poll_timeout_s": 5, "max_sample_age_s": 30,
    }
    processes = []
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(tmp_path / "cache/hf"), XDG_CACHE_HOME=str(tmp_path / "cache/xdg"),
        TRITON_CACHE_DIR=str(tmp_path / "cache/triton"), CUDA_CACHE_PATH=str(tmp_path / "cache/cuda"),
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "cache/inductor"), TMPDIR=str(tmp_path),
        M8_GPU_EXPECTED_SHA=expected,
    )

    def start(name, module, args, visible):
        log = (tmp_path / f"{name}.log").open("w", encoding="utf-8")
        provenance = tmp_path / f"{name}.source.json"
        env = dict(environment, CUDA_VISIBLE_DEVICES=visible, M8_GPU_PROVENANCE=str(provenance))
        command = [sys.executable, str(CLI), module, *map(str, args)]
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except BaseException:
            log.close()
            raise
        processes.append({"name": name, "pid": process.pid, "process": process, "log": log,
                          "command": command, "cuda_visible_devices": visible, "forced": False})
        return process

    async def run():
        external = []
        responses = {}
        final_stats = None
        cleanup_errors = []
        try:
            worker_processes = {}
            for name, role, index, seed in (("d0", "decode", 3, 0), ("d1", "decode", 4, 1),
                                            ("p0", "prefill", 0, 0), ("p1", "prefill", 1, 1)):
                role_config = {**config, "seed": seed, "timing_path": str(tmp_path / f"{name}.jsonl"),
                               "enable_chunked_prefill": role == "decode"}
                if role == "decode":
                    role_config["max_num_batched_tokens"] = 4
                config_path = tmp_path / f"{name}.json"
                config_path.write_text(json.dumps(role_config, indent=2) + "\n")
                args = ["--role", role, "--config", config_path, "--host", "127.0.0.1",
                        "--port", endpoints[name][1], "--max-connections", 8,
                        "--max-transfer-bytes", 1_000_000, "--request-timeout", 240,
                        "--pipeline-frame-bytes", 256]
                if role == "prefill":
                    target = endpoints["d0" if name == "p0" else "d1"]
                    args += ["--decode-host", target[0], "--decode-port", target[1],
                             "--allowed-decode-endpoints", json.dumps([endpoints["d0"], endpoints["d1"]])]
                worker_processes[name] = start(name, "myvllm.pd.worker", args, gpus[index]["uuid"])

            deadline = time.monotonic() + 240
            infos = {name: await wait_worker(endpoints[name], process, deadline)
                     for name, process in worker_processes.items()}
            assert all(infos[name]["role"] == ("prefill" if name.startswith("p") else "decode")
                       for name in infos)
            observed = {}
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                observed = {}
                for pid, gpu in nvidia_query(("compute-apps", "pid", "gpu_uuid")):
                    observed.setdefault(int(pid), set()).add(gpu)
                if all(process.pid in observed for process in worker_processes.values()):
                    break
                await asyncio.sleep(.5)
            expected_uuids = {worker_processes[name].pid: gpus[index]["uuid"]
                              for name, index in (("p0", 0), ("p1", 1), ("d0", 3), ("d1", 4))}
            assert {pid: observed.get(pid) for pid in expected_uuids} == {
                pid: {gpu} for pid, gpu in expected_uuids.items()}
            (tmp_path / "gpu-assignments.json").write_text(json.dumps({
                "inventory": gpus, "expected_process_uuids": expected_uuids,
                "observed_process_uuids": {pid: sorted(devices) for pid, devices in observed.items()},
            }, indent=2) + "\n")
            for name, process in worker_processes.items():
                source = json.loads((tmp_path / f"{name}.source.json").read_text())
                assert source["pid"] == process.pid and source["head"] == expected
                assert source["cuda_visible_devices"] == expected_uuids[process.pid]

            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, max_model_length=24)
            decode = await PDClient.open_d(endpoints["d0"], lifetime=220)
            external.append(decode)
            prefill = await PDClient.open_p(
                endpoints["p1"], decode.session_id, "external-resident", [1, 2, 3, 4, 5],
                params, lifetime=220, decode_endpoint=endpoints["d0"],
            )
            external.append(prefill)
            assert not (await prefill.next_token())["finished"]
            assert await decode.reserve(await prefill.describe())
            assert (await PDClient.stats(endpoints["d0"]))["resident_requests"] == 1

            for offset, mode in enumerate(("bulk", "pipeline", "local")):
                timing = tmp_path / f"gateway-{mode}.jsonl"
                gateway_config = {**config, "timing_path": str(timing)}
                gateway_settings = dict(settings)
                if mode == "pipeline":
                    gateway_settings["pipeline_min_tokens"] = 1
                if mode == "local":
                    gateway_settings["elastic"] = {
                        "enabled": True, "prefill_backlog_tokens": 0, "tpot_budget_ms": 1_000_000,
                        "tpot_margin_ms": 0, "step_sample_max_age_ms": 60_000,
                        "allow_idle_without_sample": True,
                    }
                model_path = tmp_path / f"gateway-{mode}-model.json"
                settings_path = tmp_path / f"gateway-{mode}.json"
                model_path.write_text(json.dumps(gateway_config, indent=2) + "\n")
                settings_path.write_text(json.dumps(gateway_settings, indent=2) + "\n")
                port = selected_ports[4 + offset]
                gateway = start(f"gateway-{mode}", "myvllm.pd.gateway", [
                    "--config", model_path, "--gateway-config", settings_path, "--served-model-name", "tiny",
                    "--host", "127.0.0.1", "--port", port,
                ], "")
                await wait_http(port, gateway)
                source = json.loads((tmp_path / f"gateway-{mode}.source.json").read_text())
                assert source["pid"] == gateway.pid and source["head"] == expected
                assert source["cuda_visible_devices"] == ""
                request_id = f"smoke-{mode}"
                request = {"model": "tiny", "prompt": "hello token2", "temperature": 0,
                           "max_tokens": 4, "ignore_eos": True}
                if mode == "local":
                    request.update(stream=True, stream_options={"include_usage": True})
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                             timeout=180, trust_env=False) as client:
                    response = await client.post("/v1/completions", json=request,
                                                 headers={"x-request-id": request_id})
                assert response.status_code == 200, response.text
                (tmp_path / f"client-{mode}.txt").write_text(response.text)
                if mode == "local":
                    assert response.text.endswith("data: [DONE]\n\n")
                    chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    choices = [row["choices"][0] for row in chunks if row["choices"]]
                    assert choices[-1]["finish_reason"] == "length"
                    assert chunks[-1]["usage"]["completion_tokens"] == 4
                    text = "".join(choice["text"] for choice in choices)
                else:
                    result = response.json()
                    assert result["usage"]["completion_tokens"] == 4
                    assert result["choices"][0]["finish_reason"] == "length"
                    text = result["choices"][0]["text"]
                responses[mode] = text
                assigned = records(timing, request_id, "pd_gateway_assigned")
                opened = records(timing, request_id, "pd_gateway_open")
                assert len(assigned) == len(opened) == 1
                assert tuple(assigned[0]["decode"]) == endpoints["d1"]
                if mode == "local":
                    assert assigned[0]["route"] == "local" and opened[0]["payload_bytes"] == 0
                    assert not any(records(tmp_path / f"{name}.jsonl", request_id) for name in ("p0", "p1"))
                    admitted = records(tmp_path / "d1.jsonl", request_id, "pd_admitted")
                    assert len(admitted) == 1 and admitted[0]["route"] == "local"
                    assert admitted[0]["payload_bytes"] == 0
                    assert not any(records(timing, request_id, event) for event in (
                        "pd_gateway_handoff_start", "pd_transfer_done", "pd_pipeline_done"))
                else:
                    assert assigned[0]["route"] == "remote"
                    assert tuple(assigned[0]["prefill"]) == endpoints["p0"]
                    assert opened[0]["transfer_mode"] == mode
                    assert records(tmp_path / "d1.jsonl", request_id, "pd_reserved")
                    event = "pd_transfer_done" if mode == "bulk" else "pd_pipeline_done"
                    assert records(timing, request_id, event)[0]["payload_bytes"] > 0
                    if mode == "pipeline":
                        assert records(tmp_path / "p0.jsonl", request_id, "pd_region_sent")
                        assert records(tmp_path / "d1.jsonl", request_id, "pd_h2d_done")
            assert responses["bulk"] == responses["pipeline"] == responses["local"]
        finally:
            active_error = sys.exception()
            for connection in reversed(external):
                try:
                    await connection.close()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if len(processes) >= 4 and all(item["process"].poll() is None for item in processes[:4]):
                try:
                    final_stats = await empty_owners(endpoints)
                    (tmp_path / "final-owner-stats.json").write_text(json.dumps(final_stats, indent=2) + "\n")
                except Exception as exc:
                    cleanup_errors.append(exc)
            try:
                exits, stop_errors = await stop_processes(processes)
                (tmp_path / "process-exits.json").write_text(json.dumps(exits, indent=2) + "\n")
                cleanup_errors.extend(RuntimeError(message) for message in stop_errors)
                bad = [item for item in exits if item["forced"] or item["returncode"] not in (0, -signal.SIGTERM)]
                if bad:
                    cleanup_errors.append(RuntimeError(f"CLI process cleanup failed: {bad}"))
            except Exception as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                if active_error is not None:
                    raise BaseExceptionGroup("smoke and cleanup failures", [active_error, *cleanup_errors])
                raise ExceptionGroup("smoke cleanup failures", cleanup_errors)
        (tmp_path / "smoke-summary.json").write_text(json.dumps({
            "candidate": expected, "checkpoint_hashes": checkpoint_hashes,
            "responses": responses, "final_owner_stats": final_stats,
        }, indent=2) + "\n")

    asyncio.run(run())
