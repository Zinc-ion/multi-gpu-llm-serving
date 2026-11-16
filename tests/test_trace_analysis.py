import asyncio
import argparse
import hashlib
import json
import shutil

import pytest

from tools.analyze_trace import analyze
from tools.prepare_trace import file_sha256, json_bytes, load_plan, prepare
from tools.replay_trace import replay, validate_freeze_binding
from trace_helpers import choice, event, fixture_contract, fixture_freeze, make_plan, replay_args, send_response, tcp_server, usage


def test_frozen_fixture_policy_uses_full_plan_and_single_token_ttft(tmp_path):
    async def run():
        prompts = ["hello", "hello world", "world", "world hello"]
        budgets = [1, 2, 2, 2]
        plan, manifest, tokenizer = make_plan(
            tmp_path / "inputs", [{"prompt": p, "output_tokens": n} for p, n in zip(prompts, budgets)])

        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            if body["prompt"] == "world hello":
                await send_response(writer, [b"error"], status=503)
                return
            tokens = len(tokenizer.encode(body["prompt"]))
            if body["prompt"] == "hello":
                events = event(choice(identity, "", "length")) + event(usage(identity, tokens, 1))
            elif body["prompt"] == "world":
                events = event(choice(identity, "merged", "length")) + event(usage(identity, tokens, 2))
            else:
                events = (event(choice(identity, "a")) + event(choice(identity, "b", "length"))
                          + event(usage(identity, tokens, 2)))
            await send_response(writer, [events + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            freeze = fixture_freeze(tmp_path / "freeze")
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], deployment=str(deployment),
                                     protocol_contract=str(contract), freeze=str(freeze), gpu_count=4))
            report = analyze(plan, output)
            assert report["mode"] == "fixture" and not report["formal_contract_checks_passed"] and report["batch_complete"]
            overall = report["overall"]
            assert (overall["planned"], overall["success"], overall["failed_or_missing"]) == (4, 3, 1)
            assert (overall["slo_pass"], overall["slo_indeterminate"], overall["slo_failed"]) == (2, 1, 1)
            assert overall["slo_pass_fraction"] == .5
            assert overall["offered_requests_per_s"] == 4 / manifest["window_s"]
            assert overall["observed_passing_requests_per_s_per_gpu"] == 2 / manifest["window_s"] / 4
            single, multi, merged, failed = report["requests"]
            assert single["slo"] == "pass" and single["client_tpot_est_ms"] is None
            assert multi["slo"] == "pass" and multi["client_tpot_est_ms"] is not None
            assert merged["slo"] == "indeterminate" and merged["client_tpot_est_ms"] is None
            assert failed["slo"] == "failed"
            assert overall["distributions_ms"]["planned_ttft_ms"]["n"] == 3
            assert overall["distributions_ms"]["client_tpot_est_ms"]["n"] == 1
            assert overall["distributions_ms"]["chunk_itl_ms"]["n"] == 1
            assert all("p99" in item for item in overall["distributions_ms"].values())

    asyncio.run(run())


def test_planned_ttft_includes_global_queue_delay(tmp_path):
    async def run():
        prompts = ["hello", "world"]
        plan, manifest, tokenizer = make_plan(
            tmp_path / "inputs", [{"prompt": p, "output_tokens": 1} for p in prompts])

        async def respond(body, headers, writer):
            if body["prompt"] == "hello":
                await asyncio.sleep(.08)
            identity = "c-" + headers["x-request-id"]
            stream = (event(choice(identity, "", "length"))
                      + event(usage(identity, len(tokenizer.encode(body["prompt"])), 1))
                      + b"data: [DONE]\n\n")
            await send_response(writer, [stream])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            freeze = fixture_freeze(tmp_path / "freeze", ttft_ms=50)
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], deployment=str(deployment),
                                     protocol_contract=str(contract), freeze=str(freeze)))
            report = analyze(plan, output)
            second = report["requests"][1]
            assert second["planned_ttft_ms"] > 50
            assert second["dispatch_ttft_ms"] < 50
            assert second["slo"] == "not_met"
            assert report["overall"]["slo_pass"] == 0

    asyncio.run(run())


def test_missing_duplicate_and_hash_mismatch_keep_full_denominator(tmp_path):
    async def run():
        plan, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}] * 2)

        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "x", "length"))
                                        + event(usage(identity, len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            original = tmp_path / "original"
            await replay(replay_args(plan, original, [endpoint]))
        source = original / "formal" / "results.jsonl"
        lines = source.read_bytes().split(b"\n")[:-1]
        altered = tmp_path / "altered"
        shutil.copytree(original, altered)
        result_path = altered / "formal" / "results.jsonl"
        result_path.write_bytes(lines[0] + b"\n" + lines[0] + b"\n")
        run_path = altered / "run.json"
        run = json.loads(run_path.read_text())
        run["formal"]["results_sha256"] = file_sha256(result_path)
        run_path.write_bytes(json_bytes(run))
        report = analyze(plan, altered)
        assert "duplicate_request_ids" in report["integrity_problems"]
        assert "missing_request_ids" in report["integrity_problems"]
        assert len(report["duplicate_ids"]) == len(report["missing_ids"]) == 1
        assert report["overall"]["planned"] == report["overall"]["failed_or_missing"] == 2
        assert not report["batch_complete"]
        run["plan_sha256"] = "0" * 64
        run_path.write_bytes(json_bytes(run))
        assert "plan_hash_mismatch" in analyze(plan, altered)["integrity_problems"]
        run["state"] = "running"
        run_path.write_bytes(json_bytes(run))
        assert "batch_incomplete" in analyze(plan, altered)["integrity_problems"]

    asyncio.run(run())


def test_protocol_requires_bound_source_probe_and_formal_approval(tmp_path):
    async def run():
        plan, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])

        async def respond(body, headers, writer):
            await send_response(writer, [])

        async with tcp_server(respond) as (endpoint, _):
            deployment, contract = fixture_contract(tmp_path / "fixture", manifest, [endpoint])
            args = replay_args(plan, tmp_path / "run", [endpoint], deployment=str(deployment),
                               protocol_contract=str(contract))
            document = json.loads(contract.read_text())
            document["scope"] = "formal"
            contract.write_bytes(json_bytes(document))
            with pytest.raises((ValueError, KeyError)):
                await replay(args)
            document["scope"] = "fixture"
            contract.write_bytes(json_bytes(document))
            (tmp_path / "fixture" / "deployment-config.json").write_text("changed")
            with pytest.raises(ValueError, match="evidence hash mismatch"):
                await replay(args)
            assert not (tmp_path / "run").exists()

    asyncio.run(run())


def test_freeze_approval_binds_threshold_values(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        freeze = fixture_freeze(tmp_path / "freeze")
        document = json.loads(freeze.read_text())
        document["tpot_ms"] *= 2
        freeze.write_bytes(json_bytes(document))

        async def respond(body, headers, writer):
            await send_response(writer, [])

        async with tcp_server(respond) as (endpoint, _):
            with pytest.raises(ValueError, match="approval does not attest"):
                await replay(replay_args(plan, tmp_path / "run", [endpoint], freeze=str(freeze)))
            assert not (tmp_path / "run").exists()

    asyncio.run(run())


def test_formal_freeze_binding_requires_the_exact_client_and_plan_pair(tmp_path):
    plan, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
    settings = {"deployment_sha256": "3" * 64, "analyze_tool_sha256": "4" * 64,
                "replay_tool_sha256": "1" * 64, "concurrency": 2, "request_timeout_s": 5,
                "batch_timeout_s": 30, "max_raw_bytes": 10000, "gpu_count": 4, "ignore_eos": True}
    plan_manifest_hash = file_sha256(str(plan) + ".manifest.json")
    binding = {"model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer"]["revision"],
               "tokenizer_files_sha256": hashlib.sha256(json_bytes(manifest["tokenizer"]["files"])).hexdigest(),
               **settings, "plans": [{"plan_sha256": manifest["plan_sha256"], "plan_manifest_sha256": plan_manifest_hash,
                                      "warmup_plan_sha256": None, "warmup_plan_manifest_sha256": None}]}
    freeze = {"scope": "formal", "binding": binding}
    validate_freeze_binding(freeze, manifest, None, settings, plan_manifest_hash=plan_manifest_hash)
    with pytest.raises(ValueError, match="client settings"):
        validate_freeze_binding(freeze, manifest, None, {**settings, "concurrency": 3},
                                plan_manifest_hash=plan_manifest_hash)
    with pytest.raises(ValueError, match="plan and warmup"):
        validate_freeze_binding(freeze, manifest, "2" * 64, settings,
                                plan_manifest_hash=plan_manifest_hash, warmup_manifest_hash="5" * 64)


def test_warmup_failure_aborts_formal_rows_and_preserves_budget(tmp_path):
    async def run():
        plan, _, _ = make_plan(tmp_path / "formal-input", [{"prompt": "hello", "output_tokens": 2}] * 2)
        warm, _, _ = make_plan(tmp_path / "warm-input", [{"prompt": "hello", "output_tokens": 1}], kind="warmup")

        async def respond(body, headers, writer):
            await send_response(writer, [b"busy"], status=503)

        async with tcp_server(respond) as (endpoint, counter):
            output = tmp_path / "run"
            result = await replay(replay_args(plan, output, [endpoint], warmup_plan=str(warm), no_warmup=False))
            assert result["state"] == "incomplete" and len(counter["requests"]) == 1
            assert result["budget"]["total_requests"] == 3
            assert result["budget"]["total_max_output_tokens"] == 5
            assert result["formal"]["stop_reason"] == "warmup_failed"
            report = analyze(plan, output)
            assert not report["batch_complete"]
            assert report["overall"]["planned"] == 2 and report["overall"]["failed_or_missing"] == 2
            assert all(row["status"] == "not_executed" for row in report["requests"])

    asyncio.run(run())


def test_l3_analysis_uses_planned_phase_windows_and_separate_completion_tail(tmp_path):
    async def run():
        _, _, tokenizer = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}])
        root = tmp_path / "inputs"
        phases = root / "phases.json"
        phases.write_bytes(json_bytes([
            {"name": "chat", "duration_s": .12, "rate": 40, "chat_probability": 1},
            {"name": "long", "duration_s": .12, "rate": 40, "chat_probability": 0},
        ]))
        plan = root / "l3.jsonl"
        options = argparse.Namespace(
            mode="l3", kind="formal", chat=str(root / "m2.jsonl"), long=str(root / "m2.jsonl"),
            tokenizer=str(root / "tokenizer"), tokenizer_provenance=str(root / "source-manifest.json"),
            model_revision="a" * 40, tokenizer_revision="a" * 40, seed=7, id_prefix="phase",
            rate=None, count=None, duration_s=None, chat_probability=None, phases=str(phases),
            max_requests=100, allow_recycle=True, output=str(plan),
        )
        manifest = prepare(options)
        _, planned = load_plan(plan)
        assert all(phase["request_count"] > 0 for phase in manifest["phases"])

        async def respond(body, headers, writer):
            await asyncio.sleep(.05)
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "", "length"))
                                        + event(usage(identity, len(tokenizer.encode(body["prompt"])), 1))
                                        + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            output = tmp_path / "run"
            await replay(replay_args(plan, output, [endpoint], batch_timeout_s=4))
            report = analyze(plan, output)
            assert report["mode"] == "capture_only" and report["batch_complete"]
            for phase in manifest["phases"]:
                result = report["phases"][phase["name"]]
                expected = [row for row in planned if row["phase"] == phase["name"]]
                assert result["planned"] == len(expected) == phase["request_count"]
                assert result["planned_window_s"] == .12
                assert result["offered_requests_per_s"] == len(expected) / .12
            assert report["overall"]["completion_tail_s"] > 0
            assert report["overall"]["completed_after_window"] > 0

    asyncio.run(run())
