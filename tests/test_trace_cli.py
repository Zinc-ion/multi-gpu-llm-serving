import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from tools.prepare_trace import file_sha256, json_bytes
from tools.replay_trace import replay
from trace_helpers import (choice, event, fixture_contract, make_plan, replay_args,
                           send_response, tcp_server)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("scenario", ["deadline", "warmup_failure", "completed_failures"])
def test_cli_exit_status_distinguishes_incomplete_batch_from_executed_failures(tmp_path, scenario):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}] * 3)
        warmup = None
        if scenario == "warmup_failure":
            warmup, _, _ = make_plan(tmp_path / "warm-input", [{"prompt": "hello", "output_tokens": 1}],
                                     kind="warmup")
        partial = event(choice("c", "first"))

        async def respond(body, headers, writer):
            if scenario == "deadline":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                             b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                writer.write(f"{len(partial):x}\r\n".encode() + partial + b"\r\n")
                await writer.drain()
                await asyncio.sleep(.3)
            else:
                await send_response(writer, [b"busy"], status=503)

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            command = [sys.executable, "-B", str(ROOT / "tools/replay_trace.py"), "--plan", str(plan),
                       "--endpoint", endpoint, "--model", "tiny", "--ignore-eos", "true", "--concurrency", "1",
                       "--request-timeout-s", "1", "--batch-timeout-s", ".08" if scenario == "deadline" else "2",
                       "--max-raw-bytes", "100000", "--gpu-count", "1", "--output-dir", str(output)]
            command += ["--warmup-plan", str(warmup)] if warmup else ["--no-warmup"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
                                                           stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
            assert process.returncode == (0 if scenario == "completed_failures" else 1), stderr.decode()
            run = json.loads((output / "run.json").read_text())
            result = [json.loads(line) for line in (output / "formal" / "results.jsonl").read_bytes().split(b"\n") if line]
            assert len(result) == 3 and json.loads(stdout)["state"] == run["state"]
            if scenario == "deadline":
                assert run["state"] == "incomplete" and run["formal"]["stop_reason"] == "batch_deadline"
                assert run["formal"]["counts"]["not_executed"] == 2
                assert (output / "formal" / result[0]["raw_path"]).read_bytes() == partial
            elif scenario == "warmup_failure":
                assert run["state"] == "incomplete" and run["formal"]["stop_reason"] == "warmup_failed"
                assert run["formal"]["counts"]["not_executed"] == 3 and len(counter["requests"]) == 1
            else:
                assert run["state"] == "complete" and run["formal"]["counts"]["failed"] == 3
                assert run["formal"]["counts"]["not_executed"] == 0 and len(counter["requests"]) == 3
        analysis_output = tmp_path / "analysis.json"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", str(ROOT / "tools/analyze_trace.py"), "--plan", str(plan),
            "--run-dir", str(output), "--output", str(analysis_output), cwd=ROOT, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        report = json.loads(analysis_output.read_bytes())
        assert len(report["requests"]) == report["overall"]["planned"] == 3
        assert report["batch_complete"] == (scenario == "completed_failures")
        assert json.loads(stdout)["batch_complete"] == report["batch_complete"]
        assert process.returncode == (0 if scenario == "completed_failures" else 1), stderr.decode()

    asyncio.run(run())


@pytest.mark.parametrize("problem", ["results_missing", "duplicate_ids", "evidence_invalid"])
def test_analysis_cli_writes_report_and_exits_nonzero_for_invalid_batch(tmp_path, problem):
    async def run():
        plan, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 2}] * 2)

        async def respond(body, headers, writer):
            await send_response(writer, [b"busy"], status=503)

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], deployment=str(deployment),
                                     protocol_contract=str(contract)))
        results_path = output / "formal" / "results.jsonl"
        if problem == "results_missing":
            results_path.unlink()
        elif problem == "duplicate_ids":
            first = results_path.read_bytes().split(b"\n")[0] + b"\n"
            results_path.write_bytes(first * 2)
            run_path = output / "run.json"
            metadata = json.loads(run_path.read_bytes())
            metadata["formal"]["results_sha256"] = file_sha256(results_path)
            run_path.write_bytes(json_bytes(metadata))
        else:
            config = tmp_path / "fixture" / "deployment-config.json"
            config.write_bytes(config.read_bytes() + b" ")
        analysis_output = tmp_path / "analysis.json"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-B", str(ROOT / "tools/analyze_trace.py"), "--plan", str(plan),
            "--run-dir", str(output), "--output", str(analysis_output), cwd=ROOT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        report = json.loads(analysis_output.read_bytes())
        assert len(report["requests"]) == report["overall"]["planned"] == 2
        assert report["integrity_problems"] and json.loads(stdout)["integrity_problems"]
        assert process.returncode == 1, stderr.decode()

    asyncio.run(run())
