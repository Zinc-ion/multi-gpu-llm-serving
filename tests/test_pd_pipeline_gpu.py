import asyncio
import json
import multiprocessing as mp
import signal
import sys
import time

import pytest
import torch

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.sampling_parameters import SamplingParams
from pd_gpu_helpers import reference_process
from pd_helpers import create_tiny_qwen_checkpoint
from pd_pipeline_gpu_helpers import PausedDataRelay, pipeline_gpu_worker_process
from pd_pipeline_helpers import empty_pipeline
from test_pd_gpu import _cases, _request, _stop_worker


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _stop_workers(items):
    active_error = sys.exception()
    errors = []
    for item in items:
        try:
            _stop_worker(item)
        except BaseException as exc:
            errors.append(exc)
            process, stop, _ = item
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join(10)
                if process.is_alive():
                    errors.append(RuntimeError(f"Worker {process.pid} did not exit after kill"))
            except BaseException as cleanup_error:
                errors.append(cleanup_error)
            finally:
                try:
                    stop.close()
                except BaseException as cleanup_error:
                    errors.append(cleanup_error)
    if active_error is not None and errors:
        raise BaseExceptionGroup("Test and worker cleanup failures", [active_error, *errors])
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Worker cleanup failures", errors)


def start_worker(ctx, role, config, de=None, progress_lock=None):
    parent, child = ctx.Pipe(duplex=False)
    stop_child, stop = ctx.Pipe(duplex=False)
    process = ctx.Process(target=pipeline_gpu_worker_process,
                          args=(role, config, child, stop_child, de, progress_lock))
    process.start()
    child.close()
    stop_child.close()
    try:
        assert parent.poll(180), f"{role} failed to start: {process.exitcode}"
        response = parent.recv()
        assert response[0] == "ready", response
        print(f"M6 {role} pid={process.pid} endpoint={response[1]}", flush=True)
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(10)
        stop.close()
        raise
    finally:
        parent.close()
    return process, stop, response[1]


async def pipeline_case(pe, de, case):
    params = SamplingParams(**case["sampling"])
    if params.max_tokens == 1 or len(case["prompt"]) + 1 >= (params.max_model_length or 24):
        return await _request(pe, de, case)
    d = await PDClient.open_d(de, lifetime=90)
    p = None
    try:
        p = await PDClient.prepare_p(pe, d.session_id, case["id"], case["prompt"], params, lifetime=90)
        await PDClient.start_pipeline(p, d)
        event = await p.next_token()
        tokens = [event["token"]]
        if not event["finished"]:
            await PDClient.handoff_pipeline(p, d, await p.describe())
            while True:
                event = await d.next_token()
                tokens.append(event["token"])
                if event["finished"]:
                    break
        return tokens, event["finish_reason"]
    finally:
        if p:
            await p.close()
        await d.close()


@pytest.mark.parametrize("dtype,decode_eager", [
    ("float32", True), ("float32", False), ("bfloat16", True), ("bfloat16", False),
])
def test_tiny_actual_layer_pipeline_vs_bulk_and_colocated(dtype, decode_eager, tmp_path):
    # Fixed before execution, shared with the existing B2 tiny checks.
    rtol, atol = (1e-4, 1e-5) if dtype == "float32" else (.02, .02)
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(dtype=dtype, max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=6, eos=[56, 60], seed=0)
    cases = _cases()
    ctx = mp.get_context("spawn")
    reference_path = tmp_path / "reference.pt"
    parent, child = ctx.Pipe(duplex=False)
    ref = ctx.Process(target=reference_process, args=(
        {**config, "capture_role": "reference", "capture_path": str(reference_path)}, cases, child))
    ref.start()
    child.close()
    try:
        assert parent.poll(180), f"reference failed: {ref.exitcode}"
        assert parent.recv()[0] == "ok"
        ref.join(30)
        assert ref.exitcode == 0
    finally:
        parent.close()
        if ref.is_alive():
            ref.terminate()
            ref.join(10)
    p_path, d_path = tmp_path / "p.pt", tmp_path / "d.pt"
    p_timing, d_timing = tmp_path / "p.jsonl", tmp_path / "d.jsonl"
    d = start_worker(ctx, "decode", {
        **config, "capture_role": "decode", "capture_path": str(d_path),
        "enforce_eager": decode_eager, "timing_path": str(d_timing)})
    p = None
    actual, bulk = {}, {}

    async def exercise():
        for case in cases:
            value = {**case, "id": "bulk-" + case["id"]}
            bulk[case["id"]] = await _request(p[2], d[2], value)
        for case in cases[:-2]:
            actual[case["id"]] = await pipeline_case(p[2], d[2], case)
        values = await asyncio.gather(*(pipeline_case(p[2], d[2], case) for case in cases[-2:]))
        actual.update({case["id"]: value for case, value in zip(cases[-2:], values)})
        for endpoint in (p[2], d[2]):
            stats = await PDClient.stats(endpoint)
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
            assert stats["pipeline_slots"] == stats["pipeline_pinned_bytes"] == 0

    try:
        p = start_worker(ctx, "prefill", {
            **config, "capture_role": "prefill", "capture_path": str(p_path),
            "timing_path": str(p_timing)}, d[2])
        asyncio.run(exercise())
    finally:
        _stop_workers([item for item in (p, d) if item is not None])
    reference = torch.load(reference_path, weights_only=False)
    pc, dc = torch.load(p_path, weights_only=False), torch.load(d_path, weights_only=False)
    assert all(capture["device"] == 0 and not capture["missing"] for capture in (reference, pc, dc))
    checks = torch.load(str(p_path) + ".copies.pt", weights_only=False)
    assert checks and all(c["d2h_drained"] and c["pinned"] and c["pinned_slots"] == 2
                          and c["slot_bytes"] == 512 for c in checks)
    used_blocks = [block for check in checks for block in check["blocks"]]
    assert len(used_blocks) > len(set(used_blocks)), "Source block reuse was not exercised"
    assert dc["step_kinds"] and not any(dc["step_kinds"])
    report = {"dtype": dtype, "decode_eager": decode_eager, "rtol": rtol, "atol": atol, "cases": {}}
    for case in cases:
        name = case["id"]
        expected = reference["requests"][name]["events"]
        events = pc["requests"][name]["events"] + dc["requests"].get(name, {}).get("events", [])
        tokens = [event["token"] for event in expected]
        assert actual[name] == bulk[name]
        assert actual[name][0] == [e["token"] for e in events] == tokens
        got, want = torch.stack([e["logits"] for e in events]), torch.stack([e["logits"] for e in expected])
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
        if len(tokens) > 1:
            source, target = pc["requests"][name]["kv"], dc["requests"][name]["kv"]
            baseline = dc["requests"]["bulk-" + name]["kv"]
            assert len(source) == len(target) == len(baseline)
            assert all(torch.equal(a, b) and torch.equal(b, c) for a, b, c in zip(source, target, baseline))
        report["cases"][name] = {
            "tokens": tokens, "max_abs_error": (got - want).abs().max().item(),
            "same_prefix_positions": len(tokens), "valid_kv_exact": len(tokens) > 1}
    pr = [json.loads(line) for line in p_timing.read_text().splitlines()]
    dr = [json.loads(line) for line in d_timing.read_text().splitlines()]
    sent = [r for r in pr if r["event"] == "pd_region_send_start" and r["request_id"] == "length-b-plus-1"]
    first = next(r for r in pr if r["event"] == "pd_first" and r["request_id"] == "length-b-plus-1")
    assert sent and sent[0]["layer"] == 0 and sent[0]["monotonic_ns"] < first["monotonic_ns"]
    d2h = [r for r in pr if r["event"] == "pd_d2h_done"]
    h2d = [r for r in dr if r["event"] == "pd_h2d_done"]
    assert d2h and h2d
    assert all(0 <= r["compute_ready_gpu_ms"] <= r["d2h_start_gpu_ms"] <= r["d2h_end_gpu_ms"] for r in d2h)
    assert all(0 <= r["h2d_start_gpu_ms"] <= r["h2d_end_gpu_ms"] for r in h2d)
    closed = [r for r in pr if r["event"] == "pd_pipeline_sender_closed"]
    assert closed and all(r["slot_count"] == 2 and r["staging_bytes"] == r["pinned_bytes"] == 512
                          and r["peak_slots"] <= 2 for r in closed)
    report.update(d2h_events=len(d2h), h2d_events=len(h2d),
                  acked_payload_bytes=sum(r["payload_bytes"] for r in closed),
                  first_layer_send_before_first_token_ns=first["monotonic_ns"] - sent[0]["monotonic_ns"],
                  source_copy_checks=checks)
    (tmp_path / "numeric-summary.json").write_text(json.dumps(report, indent=2) + "\n")


def test_tiny_gpu_backpressure_faults_and_owned_prefill_restart(tmp_path):
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(max_num_sequences=2, max_num_batched_tokens=24, max_model_length=24,
                  max_cached_blocks=6, seed=0)
    ctx = mp.get_context("spawn")
    progress_lock = ctx.RLock()
    d = start_worker(ctx, "decode", {
        **config, "capture_role": "decode", "capture_path": str(tmp_path / "d.pt"),
        "timing_path": str(tmp_path / "d.jsonl")}, progress_lock=progress_lock)
    workers = []
    report = []
    case = {"id": "warmup", "prompt": [1, 3, 5, 7, 9],
            "sampling": {"temperature": 0, "max_tokens": 4, "ignore_eos": True}}

    async def exercise():
        relay = PausedDataRelay(d[2])
        endpoint = await relay.start()

        def spawn(name):
            item = start_worker(ctx, "prefill", {
                **config, "capture_role": "prefill", "capture_path": str(tmp_path / f"{name}.pt"),
                "timing_path": str(tmp_path / f"{name}.jsonl")}, endpoint, progress_lock)
            workers.append(item)
            return item

        p = spawn("p")
        try:
            expected = await pipeline_case(p[2], d[2], case)
            for fault in ("close", "cancel", "data_eof", "decode_close", "timeout", "kill_prefill"):
                relay.pause = True
                relay.received.clear()
                relay.release.clear()
                dc = await PDClient.open_d(d[2], lifetime=20)
                pc = None
                try:
                    pc = await PDClient.prepare_p(
                        p[2], dc.session_id, fault, case["prompt"], SamplingParams(**case["sampling"]),
                        lifetime=1.5 if fault == "timeout" else 20)
                    await PDClient.start_pipeline(pc, dc)
                    await asyncio.wait_for(relay.received.wait(), 3)
                    assert relay.acks[-1] == {"copies_done": False, "payload_bytes": 256}
                    stats = await PDClient.stats(d[2])
                    assert stats["pipeline_pinned_bytes"] == stats["pipeline_staging_bytes"] == 512
                    assert stats["pipeline_slots"] == 2 and stats["used_blocks"] == 2
                    started = time.monotonic()
                    if fault == "close":
                        await asyncio.wait_for(pc.close(), 2)
                    elif fault == "cancel":
                        try:
                            await asyncio.wait_for(pc.cancel(), 2)
                        except (PDRemoteError, ConnectionError, OSError, asyncio.IncompleteReadError):
                            pass
                    elif fault == "data_eof":
                        for writer in relay.data_writers:
                            writer.transport.abort()
                    elif fault == "decode_close":
                        await asyncio.wait_for(dc.close(), 2)
                    elif fault == "timeout":
                        with pytest.raises((PDRemoteError, TimeoutError, ConnectionError,
                                            OSError, asyncio.IncompleteReadError)):
                            await pc.next_token()
                    else:
                        p[0].kill()
                        p[0].join(10)
                        assert p[0].exitcode == -signal.SIGKILL
                        print(f"M6 owned prefill killed pid={p[0].pid}, exit={p[0].exitcode}", flush=True)
                        p[1].close()
                        workers.remove(p)
                    report.append({"fault": fault, "seconds_to_interrupt": time.monotonic() - started})
                finally:
                    relay.release.set()
                    if pc:
                        await pc.close()
                    await dc.close()
                if fault != "kill_prefill":
                    await empty_pipeline(p[2])
                await empty_pipeline(d[2])
                relay.pause = False
                if fault == "kill_prefill":
                    p = spawn("replacement")
                recovered = await pipeline_case(p[2], d[2], {**case, "id": "after-" + fault})
                assert recovered == expected
                await empty_pipeline(p[2])
                await empty_pipeline(d[2])
        finally:
            await relay.close()

    try:
        asyncio.run(exercise())
    finally:
        _stop_workers([*workers, d])
    copies = torch.load(str(tmp_path / "replacement.pt") + ".copies.pt", weights_only=False)
    assert copies and all(c["d2h_drained"] and c["pinned"] for c in copies)
    records = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    closed = [r for r in records if r["event"] == "pd_pipeline_sender_closed"]
    faults = {r["request_id"]: r for r in closed}
    for fault in ("close", "cancel", "data_eof", "decode_close", "timeout"):
        assert faults[fault]["peak_slots"] == 2
        assert faults[fault]["payload_bytes"] < 2 * 2 * len(case["prompt"]) * 16 * 4
    (tmp_path / "fault-summary.json").write_text(json.dumps(report, indent=2) + "\n")
