"""Sequential, bounded L40S checks using existing service and measurement CLIs."""

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(os.environ["PROJECT_ROOT"])
ASSETS = ROOT / "acceptance-ready"
RUNS = ROOT / "runs/acceptance-20261118"
PYTHON = sys.executable
COMMANDS = (json.loads((RUNS / "measurement-commands.json").read_text())
            if (RUNS / "measurement-commands.json").exists() else [])


def save(name, value):
    (RUNS / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def command(args, **kwargs):
    COMMANDS.append({"argv": args, "cuda_visible_devices": kwargs.get("env", os.environ).get("CUDA_VISIBLE_DEVICES")})
    save("measurement-commands.json", COMMANDS)
    return subprocess.run(args, check=True, **kwargs)


def launch(args, gpu, logfile):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    stream = (RUNS / logfile).open("w")
    COMMANDS.append({"argv": args, "cuda_visible_devices": gpu, "log": logfile})
    save("measurement-commands.json", COMMANDS)
    proc = subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return proc, stream


def stop(items):
    failures = []
    for proc, stream in reversed(items):
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                failures.append(f"pid {proc.pid} required SIGKILL")
        if proc.returncode not in (0, -signal.SIGTERM):
            failures.append(f"pid {proc.pid} exited {proc.returncode}")
        stream.close()
    if failures:
        raise RuntimeError("; ".join(failures))


def ready(items):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        for proc, _ in items:
            if proc.poll() is not None:
                raise RuntimeError(f"service exited during startup: {proc.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18000/openapi.json", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(.5)
    raise TimeoutError("service startup exceeded 240 seconds")


async def wait_worker(endpoint, items):
    from myvllm.pd.client import PDClient
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if any(proc.poll() is not None for proc, _ in items):
            raise RuntimeError("PD worker exited during startup")
        try:
            return await PDClient.info(endpoint, timeout=2)
        except (ConnectionError, OSError, TimeoutError):
            await asyncio.sleep(.5)
    raise TimeoutError(f"worker {endpoint} startup exceeded 240 seconds")


async def lifecycle(pe, des, label, tp1_reference):
    from myvllm.pd.client import PDClient
    from myvllm.sampling_parameters import SamplingParams
    cases = json.loads((ASSETS / "workloads/correctness.json").read_text())["cases"]
    expected = {row["name"]: row["token_ids"] for row in tp1_reference["cases"]}

    async def request(case, de, suffix, count=16):
        d = await PDClient.open_d(de, lifetime=180)
        p = None
        try:
            p = await PDClient.open_p(
                pe, d.session_id, case["name"] + suffix, case["input_ids"],
                SamplingParams(temperature=0, max_tokens=count, ignore_eos=True),
                lifetime=180, decode_endpoint=de,
            )
            first = await p.next_token()
            tokens = [first["token"]]
            metadata = None
            if not first["finished"]:
                metadata = await p.describe()
                assert metadata["prompt_length"] == len(case["input_ids"])
                await PDClient.handoff(p, d, metadata)
                while True:
                    event = await d.next_token()
                    tokens.append(event["token"])
                    assert event["completion_tokens"] == len(tokens)
                    if event["finished"]:
                        break
            return {"name": case["name"], "decode": de, "tokens": tokens,
                    "matches_tp1": tokens == expected[case["name"]][:count],
                    "metadata": metadata}
        finally:
            if p is not None:
                await p.close()
            await d.close()

    rows = [await request(case, des[i % len(des)], f"-serial-{i}")
            for i, case in enumerate(cases)]
    concurrent = await asyncio.gather(*(
        request(case, des[i % len(des)], f"-concurrent-{i}")
        for i, case in enumerate(cases[2:5])
    ))
    early = await request(cases[0], des[0], "-one-token", count=1)
    assert early["metadata"] is None and len(early["tokens"]) == 1
    d = await PDClient.open_d(des[0], lifetime=180)
    p = None
    try:
        p = await PDClient.open_p(
            pe, d.session_id, label + "-cancel", cases[3]["input_ids"],
            SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
            lifetime=180, decode_endpoint=des[0],
        )
        assert not (await p.next_token())["finished"]
        await PDClient.handoff(p, d, await p.describe())
        await d.cancel()
    finally:
        if p is not None:
            await p.close()
        await d.close()
    recovery = await request(cases[0], des[0], "-after-cancel", count=4)
    endpoints = [pe, *des]
    stats = [await PDClient.stats(endpoint) for endpoint in endpoints]
    for row in stats:
        assert row["open_handles"] == row["used_blocks"] == row["reserved_transfer_bytes"] == 0
    report = {"label": label, "serial": rows, "concurrent": concurrent, "one_token": early,
              "cancel_then_recovery": recovery, "final_stats": stats,
              "all_match_tp1": all(row["matches_tp1"] for row in [*rows, *concurrent, early, recovery])}
    save(label + "-lifecycle.json", report)
    print(label, "lifecycle", report["all_match_tp1"], flush=True)


def measure(label, gpu_ids, config, items, *, concurrencies=(1, 4), requests=8,
            dataset="4k-128.jsonl"):
    state = {"phase": "ready"}
    finished = threading.Event()
    def sample():
        with (RUNS / (label + "-memory.jsonl")).open("w") as output:
            while not finished.is_set():
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, check=True,
                )
                output.write(json.dumps({"monotonic_s": time.monotonic(), "phase": state["phase"],
                                         "gpus": result.stdout.strip().splitlines()}) + "\n")
                output.flush()
                finished.wait(.2)
    thread = threading.Thread(target=sample)
    thread.start()
    try:
        for concurrency in concurrencies:
            state["phase"] = f"c{concurrency}-including-warmups"
            output = RUNS / f"{label}-c{concurrency}.json"
            command([PYTHON, str(ASSETS / "client/measure_small_batch.py"),
                     "--config", str(config), "--dataset", str(ASSETS / "workloads" / dataset),
                     "--label", f"{label}-c{concurrency}", "--gpu-ids", gpu_ids,
                     "--concurrency", str(concurrency), "--requests", str(requests), "--warmups", "2",
                     "--output", str(output)])
            if any(proc.poll() is not None for proc, _ in items):
                raise RuntimeError("service died during measurement")
    finally:
        finished.set()
        thread.join(timeout=10)


def colocated(label, config_name, gpu_ids, **measurement):
    config = ASSETS / f"configs/{config_name}.json"
    items = []
    try:
        items.append(launch([PYTHON, "-m", "myvllm.server", "--config", str(config),
                             "--served-model-name", "Qwen3-14B", "--host", "127.0.0.1",
                             "--port", "18000"], gpu_ids, label + "-server.log"))
        ready(items)
        measure(label, gpu_ids, config, items, **measurement)
    finally:
        stop(items)


def pd(label, *, four=False):
    import torch
    reference = torch.load(RUNS / "tp1-bf16.pt", map_location="cpu", weights_only=False)
    items = []
    pe = ("127.0.0.1", 19100)
    des = [("127.0.0.1", 19200), ("127.0.0.1", 19201)] if four else [("127.0.0.1", 19200)]
    common = ["--max-connections", "8", "--max-transfer-bytes", "3221225472",
              "--max-payload-bytes", "1073741824", "--request-timeout", "600"]
    gateway = ASSETS / "gateway.json"
    try:
        for i, de in enumerate(des):
            config = ASSETS / "configs/pd-decode-bf16.json"
            items.append(launch([PYTHON, "-m", "myvllm.pd.worker", "--role", "decode",
                                 "--config", str(config), "--host", de[0], "--port", str(de[1]), *common],
                                str(i+2 if four else 1), f"{label}-decode-{i}.log"))
            asyncio.run(wait_worker(de, items))
        config = ASSETS / ("configs/tp2-bf16.json" if four else "configs/pd-prefill-bf16.json")
        route = ["--decode-host", des[0][0], "--decode-port", str(des[0][1])]
        if four:
            route += ["--allowed-decode-endpoints", json.dumps(des)]
        items.append(launch([PYTHON, "-m", "myvllm.pd.worker", "--role", "prefill",
                             "--config", str(config), "--host", pe[0], "--port", str(pe[1]),
                             *common, *route], "0,1" if four else "0", label + "-prefill.log"))
        asyncio.run(wait_worker(pe, items))
        asyncio.run(lifecycle(pe, des, label, reference))
        if four:
            settings = json.loads(gateway.read_text())
            settings.update(pairs=[{"prefill": pe, "decode": de} for de in des],
                            routing_mode="load", poll_interval_s=.05, poll_timeout_s=2,
                            max_sample_age_s=5, elastic={"enabled": False})
            gateway = ASSETS / "gateway-four.json"
            gateway.write_text(json.dumps(settings, indent=2) + "\n")
        config = ASSETS / "configs/tp1-bf16.json"
        items.append(launch([PYTHON, "-m", "myvllm.pd.gateway", "--config", str(config),
                             "--gateway-config", str(gateway), "--served-model-name", "Qwen3-14B",
                             "--host", "127.0.0.1", "--port", "18000"], "", label + "-gateway.log"))
        ready(items)
        measure(label, "0,1,2,3" if four else "0,1", config, items)
        save(label + "-after-http-stats.json", asyncio.run(_stats([pe, *des])))
    finally:
        stop(items)


async def _stats(endpoints):
    from myvllm.pd.client import PDClient
    values = [await PDClient.stats(endpoint) for endpoint in endpoints]
    for row in values:
        assert row["open_handles"] == row["used_blocks"] == row["reserved_transfer_bytes"] == 0
    return values


if __name__ == "__main__":
    os.chdir(ROOT / "repo")
    paths = {
        "bf16-tp1": lambda: colocated("bf16-tp1", "tp1-bf16", "0"),
        "fp8-kv-tp1": lambda: colocated("fp8-kv-tp1", "tp1-fp8-kv", "0"),
        "bf16-pd": lambda: pd("bf16-pd"),
        "bf16-tp2": lambda: colocated("bf16-tp2", "tp2-bf16", "0,1"),
        "bf16-four-pd": lambda: pd("bf16-four-pd", four=True),
        "bf16-tp2-fixed": lambda: colocated("bf16-tp2-fixed", "tp2-bf16", "0,1"),
        "bf16-four-pd-fixed": lambda: pd("bf16-four-pd-fixed", four=True),
        "bf16-tp1-final32": lambda: colocated(
            "bf16-tp1-final32", "tp1-bf16", "0", concurrencies=(4,), requests=32,
            dataset="4k-128-32.jsonl"),
        "bf16-tp2-final32": lambda: colocated(
            "bf16-tp2-final32", "tp2-bf16", "0,1", concurrencies=(4,), requests=32,
            dataset="4k-128-32.jsonl"),
    }
    statuses = (json.loads((RUNS / "measurement-progress.json").read_text())
                if (RUNS / "measurement-progress.json").exists() else {})
    for name in sys.argv[1:] or paths:
        try:
            paths[name]()
            statuses[name] = "completed"
        except Exception as exc:
            statuses[name] = f"{type(exc).__name__}: {exc}"
            print(name, statuses[name], flush=True)
        save("measurement-progress.json", statuses)
