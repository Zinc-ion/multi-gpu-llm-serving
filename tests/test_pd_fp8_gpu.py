import asyncio
import json
import multiprocessing as mp

import pytest
import torch

from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.sampling_parameters import SamplingParams
from pd_fp8_gpu_helpers import fp8_gpu_worker_process
from pd_gpu_helpers import reference_process
from pd_helpers import create_tiny_qwen_checkpoint
from pd_pipeline_gpu_helpers import PausedDataRelay
from pd_pipeline_helpers import empty_pipeline
from test_pd_gpu import _cases, _request
from test_pd_pipeline_gpu import _stop_workers, pipeline_case


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def start_worker(ctx, role, config, de=None, lock=None):
    parent, child = ctx.Pipe(duplex=False)
    stop_child, stop = ctx.Pipe(duplex=False)
    process = ctx.Process(target=fp8_gpu_worker_process,
                          args=(role, config, child, stop_child, de, lock))
    process.start()
    child.close()
    stop_child.close()
    try:
        assert parent.poll(180), f"{role} failed to start: {process.exitcode}"
        response = parent.recv()
        assert response[0] == "ready", response
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(10)
        stop.close()
        raise
    finally:
        parent.close()
    return process, stop, response[1]


def reference(ctx, config, cases, path):
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=reference_process, args=(
        {**config, "capture_role": "reference", "capture_path": str(path)}, cases, child))
    process.start()
    child.close()
    try:
        assert parent.poll(180), f"reference failed: {process.exitcode}"
        response = parent.recv()
        assert response[0] == "ok", response
        process.join(30)
        assert process.exitcode == 0
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
    return torch.load(path, weights_only=False)


def same_bits(source, target):
    assert len(source) == len(target)
    return all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
               for a, b in zip(source, target))


@pytest.mark.parametrize("dtype,decode_eager", [
    ("float32", True), ("float32", False), ("bfloat16", True), ("bfloat16", False),
])
def test_actual_fp8_colocated_bulk_and_pipeline(dtype, decode_eager, tmp_path):
    rtol, atol = (1e-4, 1e-5) if dtype == "float32" else (.02, .02)
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(dtype=dtype, kv_cache_dtype="float8_e4m3fn", scale=1.0,
                  max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=6, seed=0, eos=[63])
    ctx = mp.get_context("spawn")
    probe = {"id": "probe", "prompt": [5, 6],
             "sampling": {"temperature": 0, "max_tokens": 1, "ignore_eos": True}}
    captured = reference(ctx, config, [probe], tmp_path / "probe.pt")
    config["eos"] = [captured["requests"]["probe"]["events"][0]["token"]]
    cases = _cases()
    cases = [case for case in cases if case["id"] != "eos-other"]
    expected = reference(ctx, config, cases, tmp_path / "reference.pt")
    d_path, p_path = tmp_path / "d.pt", tmp_path / "p.pt"
    d_timing, p_timing = tmp_path / "d.jsonl", tmp_path / "p.jsonl"
    d = start_worker(ctx, "decode", {
        **config, "capture_role": "decode", "capture_path": str(d_path),
        "enforce_eager": decode_eager, "timing_path": str(d_timing)})
    p = None
    bulk, pipeline = {}, {}

    async def exercise():
        for case in cases:
            bulk[case["id"]] = await _request(p[2], d[2], {**case, "id": "bulk-" + case["id"]})
        for case in cases[:-2]:
            pipeline[case["id"]] = await pipeline_case(p[2], d[2], case)
        values = await asyncio.gather(*(pipeline_case(p[2], d[2], case) for case in cases[-2:]))
        pipeline.update({case["id"]: value for case, value in zip(cases[-2:], values)})
        await empty_pipeline(p[2])
        await empty_pipeline(d[2])

    try:
        p = start_worker(ctx, "prefill", {
            **config, "capture_role": "prefill", "capture_path": str(p_path),
            "timing_path": str(p_timing)}, d[2])
        asyncio.run(exercise())
    finally:
        _stop_workers([item for item in (p, d) if item is not None])
    pc, dc = torch.load(p_path, weights_only=False), torch.load(d_path, weights_only=False)
    assert all(c["device"] == 0 and not c["missing"] for c in (expected, pc, dc))
    assert dc["step_kinds"] and not any(dc["step_kinds"])
    assert pipeline["eos"][1] == "stop" and len(pipeline["eos"][0]) == 1
    checks = torch.load(str(p_path) + ".copies.pt", weights_only=False)
    assert checks and all(c["d2h_drained"] and c["pinned"] and c["pinned_slots"] == 2
                          and c["slot_bytes"] == 160 for c in checks)
    blocks = [block for check in checks for block in check["blocks"]]
    assert len(blocks) > len(set(blocks))
    report = {"dtype": dtype, "decode_eager": decode_eager,
              "rtol": rtol, "atol": atol, "cases": {}}
    for case in cases:
        name = case["id"]
        ref = expected["requests"][name]["events"]
        tokens = [event["token"] for event in ref]
        assert pipeline[name] == bulk[name]
        assert pipeline[name][0] == tokens
        errors = {}
        for label, request in (("pipeline", name), ("bulk", "bulk-" + name)):
            events = pc["requests"][request]["events"] + dc["requests"].get(request, {}).get("events", [])
            assert [event["token"] for event in events] == tokens
            actual_logits = torch.stack([event["logits"] for event in events])
            reference_logits = torch.stack([event["logits"] for event in ref])
            torch.testing.assert_close(actual_logits, reference_logits, rtol=rtol, atol=atol)
            errors[label] = (actual_logits - reference_logits).abs().max().item()
            if len(tokens) > 1:
                assert same_bits(pc["requests"][request]["kv"], dc["requests"][request]["kv"])
        report["cases"][name] = {"tokens": tokens, "max_abs_error": errors,
                                  "valid_data_and_scales_exact": len(tokens) > 1}
    records = [json.loads(line) for line in p_timing.read_text().splitlines()]
    closed = [r for r in records if r["event"] == "pd_pipeline_sender_closed"]
    assert closed and all(r["payload_bytes"] == r["data_bytes"] + r["scale_bytes"] for r in closed)
    assert any(r["scale_bytes"] > 0 for r in closed)
    sent = [r for r in records if r["event"] == "pd_d2h_done"]
    assert sent and all(r["scale_bytes"] > 0 and r["payload_bytes"] == r["data_bytes"] + r["scale_bytes"]
                        for r in sent)
    report["copy_checks"] = checks
    (tmp_path / "numeric-summary.json").write_text(json.dumps(report, indent=2) + "\n")


def test_fp8_gpu_backpressure_cancel_eof_and_survivor(tmp_path):
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(kv_cache_dtype="float8_e4m3fn", scale=1.0,
                  max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=6, seed=0)
    ctx = mp.get_context("spawn")
    lock = ctx.RLock()
    d = start_worker(ctx, "decode", {
        **config, "capture_role": "decode", "capture_path": str(tmp_path / "d.pt"),
        "timing_path": str(tmp_path / "d.jsonl")}, lock=lock)
    workers = []
    case = {"id": "warmup", "prompt": [1, 3, 5, 7, 9],
            "sampling": {"temperature": 0, "max_tokens": 4, "ignore_eos": True}}

    async def exercise():
        relay = PausedDataRelay(d[2])
        endpoint = await relay.start()
        p = start_worker(ctx, "prefill", {
            **config, "capture_role": "prefill", "capture_path": str(tmp_path / "p.pt"),
            "timing_path": str(tmp_path / "p.jsonl")}, endpoint, lock)
        workers.append(p)
        try:
            expected = await pipeline_case(p[2], d[2], case)
            for fault in ("cancel", "data_eof", "decode_close", "timeout"):
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
                    assert relay.acks[-1] == {"copies_done": False, "payload_bytes": 80}
                    stats = await PDClient.stats(d[2])
                    assert stats["pipeline_pinned_bytes"] == 160 and stats["pipeline_slots"] == 2
                    if fault == "cancel":
                        try:
                            await asyncio.wait_for(pc.cancel(), 2)
                        except (PDRemoteError, ConnectionError, OSError, asyncio.IncompleteReadError):
                            pass
                    elif fault == "data_eof":
                        for writer in relay.data_writers:
                            writer.transport.abort()
                    elif fault == "decode_close":
                        await asyncio.wait_for(dc.close(), 2)
                    else:
                        with pytest.raises((PDRemoteError, TimeoutError, ConnectionError,
                                            OSError, asyncio.IncompleteReadError)):
                            await pc.next_token()
                finally:
                    relay.release.set()
                    if pc:
                        await pc.close()
                    await dc.close()
                await empty_pipeline(p[2])
                await empty_pipeline(d[2])
                relay.pause = False
                assert await pipeline_case(p[2], d[2], {**case, "id": "after-" + fault}) == expected
                await empty_pipeline(p[2])
                await empty_pipeline(d[2])
        finally:
            await relay.close()

    try:
        asyncio.run(exercise())
    finally:
        _stop_workers([*workers, d])
    checks = torch.load(str(tmp_path / "p.pt") + ".copies.pt", weights_only=False)
    assert checks and all(c["d2h_drained"] and c["pinned"] for c in checks)
    records = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    faults = {r["request_id"]: r for r in records if r["event"] == "pd_pipeline_sender_closed"}
    for fault in ("cancel", "data_eof", "decode_close", "timeout"):
        assert faults[fault]["peak_slots"] == 2
        assert faults[fault]["payload_bytes"] < 2 * len(case["prompt"]) * 40
