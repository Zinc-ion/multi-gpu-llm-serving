import asyncio
import gzip
import json
import os
from pathlib import Path
import sys

import pytest

from test_trace_formal import formal_inputs
from tools.analyze_trace import analyze
from tools.prepare_trace import file_sha256, json_bytes
from tools.replay_trace import replay
from trace_helpers import choice, event, fixture_freeze, make_plan, replay_args, send_response, tcp_server, usage


ROOT = Path(__file__).parents[1]


def rows(output, section):
    return [json.loads(line) for line in (output / section / "results.jsonl").read_bytes().split(b"\n") if line]


def save_rows(output, section, values):
    result_path = output / section / "results.jsonl"
    result_path.write_bytes(b"".join(json_bytes(value) for value in values))
    run_path = output / "run.json"
    run = json.loads(run_path.read_bytes())
    run[section]["results_sha256"] = file_sha256(result_path)
    run_path.write_bytes(json_bytes(run))


async def collected_formal(path):
    async def respond(body, headers, writer):
        identity = "c-" + headers["x-request-id"]
        if body["prompt"] == "hello":
            await asyncio.sleep(.03)
        wire = (event(choice(identity, "x", "length")) + event(usage(identity, 1, 1))
                + b"data: [DONE]\n\n")
        await send_response(writer, [wire])

    async with tcp_server(respond) as (endpoint, counter):
        args, _, _, binding = formal_inputs(path, endpoint)
        args.freeze = str(fixture_freeze(path / "freeze", scope="formal", binding=binding))
        run = await replay(args)
        assert len(counter["requests"]) == 2
    return args, run, Path(args.output_dir)


def test_capture_binding_and_actual_measurement_tail(tmp_path):
    async def run():
        args, metadata, output = await collected_formal(tmp_path)
        for section in ("warmup", "formal"):
            row = rows(output, section)[0]
            assert row["raw_sha256"] == file_sha256(output / section / row["raw_path"])
            assert row["content_type"] == "text/event-stream"
        report = analyze(args.plan, output)
        assert report["formal_contract_checks_passed"] and report["batch_complete"]
        assert report["requests"][0]["completion_tokens"] == 1
        assert report["requests"][0]["raw_capture_binding"] == "capture_bound"
        assert report["overall"]["successful_completion_tokens"] == 1
        assert report["formal_end_offset_s"] == metadata["formal"]["end_offset_s"] >= .03
        assert report["successful_output_tokens_per_s"] == 1 / report["formal_end_offset_s"]
        assert report["overall"]["completion_tail_s"] > 0

    asyncio.run(run())


@pytest.mark.parametrize("section", ["formal", "warmup"])
@pytest.mark.parametrize("damage", ["changed", "same_size", "missing", "results_missing", "results_hash"])
def test_damaged_raw_or_results_cannot_pass_formal(tmp_path, section, damage):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        result_path = output / section / "results.jsonl"
        row = rows(output, section)[0]
        raw_path = output / section / row["raw_path"]
        if damage == "missing":
            raw_path.unlink()
        elif damage == "results_missing":
            result_path.unlink()
        elif damage == "results_hash":
            result_path.write_bytes(result_path.read_bytes() + b" ")
        elif damage == "same_size":
            raw = raw_path.read_bytes()
            changed = raw.replace(b'"text": "x"', b'"text": "y"')
            assert changed != raw and len(changed) == len(raw)
            raw_path.write_bytes(changed)
        else:
            raw_path.write_bytes(b"broken raw\n")
        report = analyze(args.plan, output)
        assert report["integrity_problems"] and not report["batch_complete"]
        assert not report["formal_contract_checks_passed"]
        assert report["overall"]["planned"] == 1
        assert report["overall"]["success"] == report["overall"]["slo_pass"] == 0
        assert report["overall"]["successful_completion_tokens"] in (None, 0)
        if damage == "same_size" and section == "warmup":
            output_file = tmp_path / "analysis.json"
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-B", str(ROOT / "tools/analyze_trace.py"), "--plan", args.plan,
                "--run-dir", str(output), "--output", str(output_file), cwd=ROOT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr = await asyncio.wait_for(process.communicate(), 20)
            assert process.returncode == 1, stderr.decode()
            assert json.loads(output_file.read_bytes())["integrity_problems"]

    asyncio.run(run())


@pytest.mark.parametrize("section", ["formal", "warmup"])
@pytest.mark.parametrize("damage", ["duplicate", "unknown", "ordinal", "ordinal_boolean", "phase",
                                     "offset", "offset_boolean", "endpoint"])
def test_result_binding_covers_both_plans(tmp_path, section, damage):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        values = rows(output, section)
        if damage == "duplicate":
            values.append(dict(values[0]))
        elif damage == "unknown":
            values[0]["request_id"] = "unknown"
        else:
            key, value = {"ordinal": ("ordinal", 9), "ordinal_boolean": ("ordinal", False),
                          "phase": ("phase", "other"), "offset": ("planned_offset_s", 9),
                          "offset_boolean": ("planned_offset_s", False),
                          "endpoint": ("endpoint", "http://other")}[damage]
            values[0][key] = value
        save_rows(output, section, values)
        report = analyze(args.plan, output)
        assert report["integrity_problems"] and not report["formal_contract_checks_passed"]
        assert report["overall"]["planned"] == 1 and report["overall"]["success"] == 0
        assert report["overall"]["slo_pass"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("damage", ["model", "identity", "usage", "finish", "done", "choices",
                                     "path", "length", "truncated", "encoding"])
def test_reparse_success_even_if_result_digest_is_rewritten(tmp_path, damage):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        values = rows(output, "formal")
        row = values[0]
        raw_path = output / "formal" / row["raw_path"]
        wire = raw_path.read_bytes()
        if damage == "model":
            wire = wire.replace(b'"model": "tiny"', b'"model": "fake"')
        elif damage == "identity":
            wire = wire.replace(b'"id": "c-', b'"id": "d-', 1)
        elif damage == "usage":
            wire = wire.replace(b'"completion_tokens": 1', b'"completion_tokens": 2')
        elif damage == "finish":
            wire = wire.replace(b'"finish_reason": "length"', b'"finish_reason": null')
        elif damage == "done":
            wire = wire.replace(b"data: [DONE]\n\n", b"")
        elif damage == "choices":
            wire = event(choice(row["response_id"], "first")) + wire
        elif damage == "path":
            row["raw_path"] = "raw/../raw/00000000.sse"
        elif damage == "length":
            row["raw_bytes"] += 1
        elif damage == "truncated":
            row["raw_truncated"] = True
        else:
            row["content_encoding"] = "gzip"
        if damage in ("model", "identity", "usage", "finish", "done", "choices"):
            raw_path.write_bytes(wire)
            row["raw_sha256"] = file_sha256(raw_path)
            row["raw_bytes"] = row["received_body_bytes"] = len(wire)
        save_rows(output, "formal", values)
        report = analyze(args.plan, output)
        assert report["integrity_problems"] and not report["formal_contract_checks_passed"]
        assert report["overall"]["success"] == report["overall"]["slo_pass"] == 0
        assert report["overall"]["successful_completion_tokens"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("case", ["partial", "bad_utf8", "limit", "limit_cr", "gzip", "http_error", "empty"])
def test_failed_raw_is_bound_and_partial_evidence_remains_valid(tmp_path, case):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}])
        first = event(choice("c", "first"))
        if case == "limit_cr":
            first = first.replace(b"\n", b"\r")
        wire = first + b"data: \xff\n\n" if case == "bad_utf8" else first + b": " + b"x" * 1000 + b"\n\n"
        if case == "gzip":
            wire = gzip.compress(wire, mtime=0)

        async def respond(body, headers, writer):
            if case == "partial":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                             b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                writer.write(f"{len(first):x}\r\n".encode() + first + b"\r\n")
                await writer.drain()
                await asyncio.sleep(.15)
            elif case == "gzip":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Encoding: gzip\r\n"
                             b"Content-Length: " + str(len(wire)).encode() + b"\r\nConnection: close\r\n\r\n" + wire)
                await writer.drain()
            elif case == "empty":
                await send_response(writer, [], status=503)
            else:
                await send_response(writer, [wire], status=503 if case == "http_error" else 200)

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], request_timeout_s=.05,
                                     max_raw_bytes=len(first) if case in ("limit", "limit_cr") else 100000))
        row = rows(output, "formal")[0]
        assert row["status"] == "failed"
        assert row["raw_sha256"] == file_sha256(output / "formal" / row["raw_path"])
        assert row["raw_truncated"] == (case in ("limit", "limit_cr"))
        report = analyze(plan, output)
        assert report["batch_complete"] and not report["integrity_problems"]
        assert report["overall"]["planned"] == 1 and report["overall"]["success"] == 0
        assert report["overall"]["successful_completion_tokens"] == 0
        if case == "empty":
            assert row["raw_bytes"] == 0
        assert report["successful_output_tokens_per_s"] == 0

    asyncio.run(run())


def test_legacy_is_revalidated_but_never_capture_bound_or_formal_accepted(tmp_path):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        for section in ("warmup", "formal"):
            values = rows(output, section)
            values[0].pop("raw_sha256")
            values[0].pop("content_type")
            save_rows(output, section, values)
        report = analyze(args.plan, output)
        assert not report["formal_contract_checks_passed"]
        assert "legacy_raw_unbound" in report["integrity_problems"]
        assert report["requests"][0]["raw_capture_binding"] == "legacy_unbound"
        assert report["requests"][0]["status"] == "success"
        assert report["overall"]["success"] == report["overall"]["slo_pass"] == 1
        assert report["overall"]["successful_completion_tokens"] is None
        assert report["successful_output_tokens_per_s"] is None
        raw_path = output / "formal" / rows(output, "formal")[0]["raw_path"]
        raw_path.write_bytes(raw_path.read_bytes().replace(b'"model": "tiny"', b'"model": "fake"'))
        report = analyze(args.plan, output)
        assert report["overall"]["success"] == 0 and "invalid_results" in report["integrity_problems"]

    asyncio.run(run())


def test_mixed_legacy_and_capture_bound_successes_do_not_produce_partial_throughput(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}] * 2)

        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "x", "length"))
                                         + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint]))
            assert len(counter["requests"]) == 2
        values = rows(output, "formal")
        values[0].pop("raw_sha256")
        values[0].pop("content_type")
        save_rows(output, "formal", values)
        report = analyze(plan, output)
        assert report["overall"]["success"] == 2
        assert [row["raw_capture_binding"] for row in report["requests"]] == ["legacy_unbound", "capture_bound"]
        assert [row["completion_tokens"] for row in report["requests"]] == [None, 1]
        assert report["overall"]["successful_completion_tokens"] is None
        assert report["phases"]["l2"]["successful_completion_tokens"] is None
        assert report["successful_output_tokens_per_s"] is None

    asyncio.run(run())


def test_formal_end_offset_must_cover_the_actual_completion(tmp_path):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        path = output / "run.json"
        document = json.loads(path.read_bytes())
        document["formal"]["end_offset_s"] = .000001
        path.write_bytes(json_bytes(document))
        report = analyze(args.plan, output)
        assert "formal_timing_invalid" in report["integrity_problems"]
        assert report["formal_end_offset_s"] is report["successful_output_tokens_per_s"] is None
        assert not report["formal_contract_checks_passed"]

    asyncio.run(run())


def test_legacy_failed_done_only_preserves_partial_protocol_state(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])

        async def respond(body, headers, writer):
            await send_response(writer, [b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint]))
        values = rows(output, "formal")
        assert values[0]["status"] == "failed" and values[0]["done"]
        values[0].pop("raw_sha256")
        values[0].pop("content_type")
        save_rows(output, "formal", values)
        report = analyze(plan, output)
        assert report["requests"][0]["status"] == "failed"
        assert report["requests"][0]["raw_capture_binding"] == "legacy_unbound"
        assert report["integrity_problems"] == ["legacy_raw_unbound"]

    asyncio.run(run())


def test_warmup_plan_reference_cannot_be_removed_from_a_completed_run(tmp_path):
    async def run():
        args, _, output = await collected_formal(tmp_path)
        path = output / "run.json"
        document = json.loads(path.read_bytes())
        document["warmup_plan_path"] = None
        path.write_bytes(json_bytes(document))
        report = analyze(args.plan, output)
        assert "warmup_plan_hash_mismatch" in report["integrity_problems"]
        assert report["overall"]["planned"] == 1 and report["overall"]["success"] == 0
        assert report["overall"]["successful_completion_tokens"] == 0
        assert not report["formal_contract_checks_passed"]

    asyncio.run(run())


def test_no_dispatch_is_not_a_zero_throughput_measurement(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        warm, _, _ = make_plan(tmp_path / "warm", [{"prompt": "hello", "output_tokens": 1}], kind="warmup")

        async def respond(body, headers, writer):
            await send_response(writer, [b"busy"], status=503)

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], warmup_plan=str(warm), no_warmup=False))
            assert len(counter["requests"]) == 1
        row = rows(output, "formal")[0]
        assert row["status"] == "not_executed" and row["raw_path"] is None and row["raw_sha256"] is None
        report = analyze(plan, output)
        assert report["overall"]["planned"] == 1 and report["overall"]["successful_completion_tokens"] is None
        assert report["formal_end_offset_s"] is report["successful_output_tokens_per_s"] is None

    asyncio.run(run())


@pytest.mark.parametrize(("remove_summary", "clear_budget"), [(False, False), (True, False), (True, True)])
def test_warmup_references_cannot_all_be_removed_from_capture_only_run(tmp_path, remove_summary, clear_budget):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        warm, _, _ = make_plan(tmp_path / "warm", [{"prompt": "hello", "output_tokens": 1}], kind="warmup")

        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "x", "length"))
                                         + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], warmup_plan=str(warm), no_warmup=False))
            assert len(counter["requests"]) == 2
        path = output / "run.json"
        document = json.loads(path.read_bytes())
        for key in ("warmup_plan_path", "warmup_plan_sha256", "warmup_manifest_sha256"):
            document[key] = None
        if remove_summary:
            del document["warmup"]
        if clear_budget:
            document["budget"]["warmup_requests"] = 0
        path.write_bytes(json_bytes(document))
        report = analyze(plan, output)
        assert "warmup_plan_hash_mismatch" in report["integrity_problems"]
        assert not report["batch_complete"]
        assert report["overall"]["planned"] == 1 and report["overall"]["success"] == 0
        assert report["overall"]["successful_completion_tokens"] == 0

    asyncio.run(run())


async def assert_incomplete_cli(plan, output, destination):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-B", str(ROOT / "tools/analyze_trace.py"), "--plan", str(plan),
        "--run-dir", str(output), "--output", str(destination), cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await asyncio.wait_for(process.communicate(), 20)
    assert process.returncode == 1, stderr.decode()
    assert not json.loads(destination.read_bytes())["batch_complete"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root may bypass directory write permissions")
@pytest.mark.parametrize("prior_dispatch", [False, True])
def test_raw_permission_error_before_dispatch_is_incomplete(tmp_path, prior_dispatch):
    async def run():
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "x", "length"))
                                         + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            plan, manifest, _ = make_plan(
                tmp_path / "slow-formal", [{"prompt": "hello", "output_tokens": 1}] * (1 + prior_dispatch),
                rate=2, provenance_scope="formal")
            args.plan = str(plan)
            binding["plans"][0]["plan_sha256"] = manifest["plan_sha256"]
            binding["plans"][0]["plan_manifest_sha256"] = file_sha256(str(plan) + ".manifest.json")
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            output = Path(args.output_dir)
            raw_dir = output / "formal" / "raw"

            async def prevent_raw_creation():
                while not raw_dir.exists() or len(counter["requests"]) < 1 + prior_dispatch:
                    await asyncio.sleep(.001)
                os.chmod(raw_dir, 0o500)

            worker = asyncio.create_task(prevent_raw_creation())
            try:
                metadata = await replay(args)
                await worker
            finally:
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
                if raw_dir.exists():
                    os.chmod(raw_dir, 0o700)
            assert len(counter["requests"]) == 1 + prior_dispatch

        values = sorted(rows(output, "formal"), key=lambda row: row["ordinal"])
        failed = values[-1]
        assert failed["status"] == "failed" and failed["error_class"] == "PermissionError"
        assert failed["dispatch_offset_s"] is None and failed["raw_path"] is failed["raw_sha256"] is None
        assert metadata["state"] == "incomplete"
        document = json.loads((output / "run.json").read_bytes())
        document["state"] = "complete"
        (output / "run.json").write_bytes(json_bytes(document))
        report = analyze(args.plan, output)
        assert not report["batch_complete"] and not report["formal_contract_checks_passed"]
        assert "batch_incomplete" in report["integrity_problems"]
        assert report["overall"]["planned"] == 1 + prior_dispatch
        assert report["overall"]["success"] == report["overall"]["slo_pass"] == prior_dispatch
        assert report["requests"][-1]["failure_reason"] == "PermissionError"
        if not prior_dispatch:
            assert report["overall"]["successful_completion_tokens"] is None
            assert report["formal_end_offset_s"] is report["successful_output_tokens_per_s"] is None
            await assert_incomplete_cli(args.plan, output, tmp_path / "analysis.json")

    asyncio.run(run())


def test_deadline_evidence_rejects_complete_state(tmp_path):
    async def run():
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            if body["prompt"] != "hello":
                await send_response(writer, [event(choice(identity, "x", "length"))
                                             + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])
                return
            partial = event(choice(identity, "x"))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            writer.write(f"{len(partial):x}\r\n".encode() + partial + b"\r\n")
            await writer.drain()
            await asyncio.sleep(.5)

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            args.batch_timeout_s = binding["batch_timeout_s"] = .15
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            metadata = await replay(args)
            assert len(counter["requests"]) == 2
        output = Path(args.output_dir)
        row = rows(output, "formal")[0]
        assert metadata["state"] == "incomplete" and metadata["formal"]["stop_reason"] == "batch_deadline"
        assert row["status"] == "failed" and row["error_class"] == "BatchDeadline"
        path = output / "run.json"
        document = json.loads(path.read_bytes())
        document["state"] = "complete"
        for stop_reason in ("batch_deadline", None):
            document["formal"]["stop_reason"] = stop_reason
            path.write_bytes(json_bytes(document))
            report = analyze(args.plan, output)
            assert not report["batch_complete"] and not report["formal_contract_checks_passed"]
            assert "batch_incomplete" in report["integrity_problems"]
            assert report["overall"]["planned"] == 1 and report["overall"]["success"] == 0
            assert report["requests"][0]["failure_reason"] == "BatchDeadline"
        await assert_incomplete_cli(args.plan, output, tmp_path / "analysis.json")

    asyncio.run(run())


def test_complete_raw_with_timeout_cannot_be_relabelled_success(tmp_path):
    async def run():
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            wire = (event(choice(identity, "x", "length")) + event(usage(identity, 1, 1))
                    + b"data: [DONE]\n\n")
            if body["prompt"] != "hello":
                await send_response(writer, [wire])
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
            writer.write(f"{len(wire):x}\r\n".encode() + wire + b"\r\n")
            await writer.drain()
            await asyncio.sleep(.3)

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            args.request_timeout_s = binding["request_timeout_s"] = .08
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            await replay(args)
            assert len(counter["requests"]) == 2
        output = Path(args.output_dir)
        original = rows(output, "formal")[0]
        assert original["status"] == "failed" and original["error_class"] == "TimeoutError"
        assert original["error"] is not None and original["done"] and original["usage"]["completion_tokens"] == 1
        assert analyze(args.plan, output)["overall"]["success"] == 0
        for error_class, error in ((original["error_class"], None), (None, original["error"]),
                                   (original["error_class"], original["error"])):
            row = dict(original, status="success", error_class=error_class, error=error,
                       first_output_offset_s=original["choice_times_s"][0],
                       last_output_offset_s=original["choice_times_s"][-1])
            save_rows(output, "formal", [row])
            report = analyze(args.plan, output)
            assert "invalid_results" in report["integrity_problems"]
            assert not report["formal_contract_checks_passed"]
            assert report["overall"]["planned"] == 1
            assert report["overall"]["success"] == report["overall"]["slo_pass"] == 0
            assert report["overall"]["successful_completion_tokens"] in (None, 0)

    asyncio.run(run())
