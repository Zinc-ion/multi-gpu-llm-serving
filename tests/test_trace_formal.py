import asyncio
import json
from pathlib import Path

import pytest

from tools.analyze_trace import analyze
from tools.prepare_trace import file_sha256, json_bytes, sha256
from tools.replay_trace import load_contract, load_freeze, replay
from trace_helpers import (choice, event, fixture_contract, fixture_freeze, make_plan, ref, replay_args,
                           send_response, tcp_server, usage)


ROOT = Path(__file__).parents[1]


def save_approved_contract(path, document):
    approval = Path(document["approval"]["path"])
    approval_doc = json.loads(approval.read_bytes())
    approval_doc["subject_sha256"] = sha256(json_bytes({key: value for key, value in document.items()
                                                        if key != "approval"}))
    approval.write_bytes(json_bytes(approval_doc))
    document["approval"]["sha256"] = file_sha256(approval)
    path.write_bytes(json_bytes(document))


def formal_inputs(path, endpoint, *, output_tokens=1):
    plan, manifest, tokenizer = make_plan(path / "formal", [{"prompt": "hello", "output_tokens": output_tokens}],
                                          provenance_scope="formal")
    warmup, warm_manifest, _ = make_plan(path / "warmup", [{"prompt": "world", "output_tokens": 1}],
                                         kind="warmup", provenance_scope="formal")
    deployment, contract = fixture_contract(path / "deployment", manifest, [endpoint], scope="formal")
    args = replay_args(plan, path / "run", [endpoint], warmup_plan=str(warmup), no_warmup=False,
                       deployment=str(deployment), protocol_contract=str(contract), gpu_count=4)
    binding = {
        "model_revision": manifest["model_revision"], "tokenizer_revision": manifest["tokenizer"]["revision"],
        "tokenizer_files_sha256": sha256(json_bytes(manifest["tokenizer"]["files"])),
        "deployment_sha256": file_sha256(deployment), "replay_tool_sha256": file_sha256(ROOT / "tools/replay_trace.py"),
        "analyze_tool_sha256": file_sha256(ROOT / "tools/analyze_trace.py"), "concurrency": args.concurrency,
        "request_timeout_s": args.request_timeout_s, "batch_timeout_s": args.batch_timeout_s,
        "max_raw_bytes": args.max_raw_bytes, "gpu_count": args.gpu_count, "ignore_eos": True,
        "plans": [{"plan_sha256": manifest["plan_sha256"],
                   "plan_manifest_sha256": file_sha256(str(plan) + ".manifest.json"),
                   "warmup_plan_sha256": warm_manifest["plan_sha256"],
                   "warmup_plan_manifest_sha256": file_sha256(str(warmup) + ".manifest.json")}],
    }
    return args, manifest, tokenizer, binding


def test_structured_synthetic_formal_chain_runs_and_keeps_authenticity_separate(tmp_path):
    async def run():
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "", "length"))
                                        + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            approval = json.loads((tmp_path / "freeze" / "approval.json").read_bytes())
            assert approval["approver"] == "SYNTHETIC TEST APPROVAL"
            run_result = await replay(args)
            report = analyze(args.plan, args.output_dir)
            assert run_result["state"] == "complete" and len(counter["requests"]) == 2
            assert report["mode"] == "formal" and report["formal_contract_checks_passed"]
            assert report["overall"]["slo_pass"] == 1
            assert report["evidence_authenticity"] == "requires_primary_agent_verification"
            assert report["stage_acceptance"] is None
            assert "formal_evidence_complete" not in report
            source = tmp_path / "formal" / "m2.jsonl"
            source.write_bytes(source.read_bytes() + b" ")
            with pytest.raises(ValueError, match="provenance|source"):
                analyze(args.plan, args.output_dir)

    asyncio.run(run())


@pytest.mark.parametrize("change", ["warmup_source", "replay_tool", "concurrency", "probe_raw"])
def test_analysis_rechecks_formal_run_and_evidence(tmp_path, change):
    async def run():
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            await send_response(writer, [event(choice(identity, "", "length"))
                                        + event(usage(identity, 1, 1)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, _):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            await replay(args)
            if change == "warmup_source":
                path = tmp_path / "warmup" / "m2.jsonl"
                path.write_bytes(path.read_bytes() + b" ")
                with pytest.raises(ValueError, match="provenance|source"):
                    analyze(args.plan, args.output_dir)
                return
            if change == "probe_raw":
                path = tmp_path / "deployment" / "empty_first.sse"
                path.write_bytes(path.read_bytes() + b" ")
            else:
                path = Path(args.output_dir) / "run.json"
                run_document = json.loads(path.read_bytes())
                if change == "replay_tool":
                    run_document["tool_sha256"] = "0" * 64
                else:
                    run_document["concurrency"] += 1
                path.write_bytes(json_bytes(run_document))
            report = analyze(args.plan, args.output_dir)
            assert report["mode"] == "capture_only" and not report["formal_contract_checks_passed"]
            assert any(problem.startswith("evidence_invalid:") for problem in report["integrity_problems"])

    asyncio.run(run())


@pytest.mark.parametrize("field", [
    "deployment_sha256", "replay_tool_sha256", "analyze_tool_sha256", "concurrency", "concurrency_boolean",
    "request_timeout_s", "batch_timeout_s", "max_raw_bytes", "gpu_count", "ignore_eos",
    "plan_sha256", "plan_manifest_sha256", "warmup_plan_sha256", "warmup_plan_manifest_sha256",
])
def test_formal_freeze_rejects_wrong_actual_binding_before_sending(tmp_path, field):
    async def run():
        async def respond(body, headers, writer):
            await send_response(writer, [])

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint)
            key = "concurrency" if field == "concurrency_boolean" else field
            target = binding["plans"][0] if key in binding["plans"][0] else binding
            value = target[key]
            target[key] = (True if field == "concurrency_boolean" else
                           not value if isinstance(value, bool) else value + 1
                           if isinstance(value, (int, float)) else "0" * 64)
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            with pytest.raises(ValueError, match="formal freeze"):
                await replay(args)
            assert not counter["requests"] and not Path(args.output_dir).exists()

    asyncio.run(run())


@pytest.mark.parametrize("segment", [
    "source_reference", "config_reference", "config_record", "runtime_reference", "runtime_record",
    "source_review_reference", "source_review_record", "probe_reference", "probe_record",
    "case_record", "request_reference", "raw_reference", "observation_reference", "observation_record",
    "approval_reference", "approval_record",
])
def test_formal_contract_rejects_each_fixture_scope(tmp_path, segment):
    plan, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}],
                                   provenance_scope="formal")
    endpoint = "http://127.0.0.1:12345"
    deployment, contract = fixture_contract(tmp_path / "deployment", manifest, [endpoint], scope="formal")
    deployment_doc = json.loads(deployment.read_bytes())
    contract_doc = json.loads(contract.read_bytes())
    if segment in ("source_reference", "config_reference"):
        key = "source_files" if segment == "source_reference" else "config_files"
        deployment_doc[key][0]["scope"] = "fixture"
    elif segment == "config_record":
        path = Path(deployment_doc["config_files"][0]["path"])
        document = json.loads(path.read_bytes())
        document["scope"] = "fixture"
        path.write_bytes(json_bytes(document))
        deployment_doc["config_files"][0]["sha256"] = file_sha256(path)
    elif segment == "runtime_reference":
        deployment_doc["runtime_provenance"]["scope"] = "fixture"
    elif segment == "runtime_record":
        path = Path(deployment_doc["runtime_provenance"]["path"])
        document = json.loads(path.read_bytes())
        document["scope"] = "fixture"
        path.write_bytes(json_bytes(document))
        deployment_doc["runtime_provenance"]["sha256"] = file_sha256(path)
    elif segment.endswith("_reference") and segment in ("source_review_reference", "probe_reference", "approval_reference"):
        key = {"source_review_reference": "source_review", "probe_reference": "probe_manifest",
               "approval_reference": "approval"}[segment]
        contract_doc[key]["scope"] = "fixture"
    elif segment in ("source_review_record", "approval_record"):
        key = "source_review" if segment == "source_review_record" else "approval"
        path = Path(contract_doc[key]["path"])
        document = json.loads(path.read_bytes())
        document["scope"] = "fixture"
        path.write_bytes(json_bytes(document))
        contract_doc[key]["sha256"] = file_sha256(path)
    else:
        path = Path(contract_doc["probe_manifest"]["path"])
        probe = json.loads(path.read_bytes())
        case = probe["cases"]["empty_first"]
        sample = case["samples"][0]
        if segment == "probe_record":
            probe["scope"] = "fixture"
        elif segment == "case_record":
            case["scope"] = "fixture"
        elif segment == "observation_record":
            observation = Path(sample["independent_observation"]["path"])
            document = json.loads(observation.read_bytes())
            document["scope"] = "fixture"
            observation.write_bytes(json_bytes(document))
            sample["independent_observation"]["sha256"] = file_sha256(observation)
        else:
            key = {"request_reference": "request", "raw_reference": "raw_sse",
                   "observation_reference": "independent_observation"}[segment]
            sample[key]["scope"] = "fixture"
        path.write_bytes(json_bytes(probe))
        contract_doc["probe_manifest"]["sha256"] = file_sha256(path)
    deployment.write_bytes(json_bytes(deployment_doc))
    contract_doc["deployment_sha256"] = file_sha256(deployment)
    contract.write_bytes(json_bytes(contract_doc))
    with pytest.raises(ValueError):
        load_contract(contract, deployment, [endpoint], "tiny", True, manifest)


@pytest.mark.parametrize("segment", ["calibration_reference", "calibration_record", "approval_reference", "approval_record"])
def test_formal_freeze_rejects_fixture_scope(tmp_path, segment):
    freeze = fixture_freeze(tmp_path / "freeze", scope="formal", binding={})
    document = json.loads(freeze.read_bytes())
    key = "calibration_files" if segment.startswith("calibration") else "approval"
    reference = document[key][0] if key == "calibration_files" else document[key]
    if segment.endswith("_reference"):
        reference["scope"] = "fixture"
    else:
        path = Path(reference["path"])
        nested = json.loads(path.read_bytes())
        nested["scope"] = "fixture"
        path.write_bytes(json_bytes(nested))
        reference["sha256"] = file_sha256(path)
    freeze.write_bytes(json_bytes(document))
    with pytest.raises(ValueError):
        load_freeze(freeze)


@pytest.mark.parametrize("problem", ["raw_hash", "observation_raw_link", "observation_request_link"])
def test_formal_probe_rejects_bad_raw_and_observation_links(tmp_path, problem):
    _, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}],
                               provenance_scope="formal")
    endpoint = "http://127.0.0.1:12345"
    deployment, contract = fixture_contract(tmp_path / "deployment", manifest, [endpoint], scope="formal")
    document = json.loads(contract.read_bytes())
    probe_path = Path(document["probe_manifest"]["path"])
    probe = json.loads(probe_path.read_bytes())
    sample = probe["cases"]["empty_first"]["samples"][0]
    if problem == "raw_hash":
        raw = Path(sample["raw_sse"]["path"])
        raw.write_bytes(raw.read_bytes() + b" ")
    else:
        observation_path = Path(sample["independent_observation"]["path"])
        observation = json.loads(observation_path.read_bytes())
        key = "raw_sse_sha256" if problem == "observation_raw_link" else "request_sha256"
        observation[key] = "0" * 64
        observation_path.write_bytes(json_bytes(observation))
        sample["independent_observation"]["sha256"] = file_sha256(observation_path)
        probe_path.write_bytes(json_bytes(probe))
        document["probe_manifest"]["sha256"] = file_sha256(probe_path)
        contract.write_bytes(json_bytes(document))
    with pytest.raises(ValueError, match="hash|observation"):
        load_contract(contract, deployment, [endpoint], "tiny", True, manifest)


@pytest.mark.parametrize("condition", ["honest_gap", "corner_subset", "unreviewed_gap",
                                       "unexplained_gap", "missing_ordinary", "foreign_endpoint"])
def test_probe_requires_ordinary_per_endpoint_but_allows_reviewed_corner_gaps(tmp_path, condition):
    _, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}],
                               provenance_scope="formal")
    endpoints = ["http://127.0.0.1:12345", "http://127.0.0.1:12346"]
    deployment, contract = fixture_contract(tmp_path / "deployment", manifest, endpoints, scope="formal")
    document = json.loads(contract.read_bytes())
    probe_path = Path(document["probe_manifest"]["path"])
    probe = json.loads(probe_path.read_bytes())
    review_path = Path(document["source_review"]["path"])
    review = json.loads(review_path.read_bytes())
    if condition in ("honest_gap", "unreviewed_gap", "unexplained_gap"):
        probe["cases"]["pure_terminal"] = {
            "scope": "formal", "covered": False, "samples": [],
            "reason": "" if condition == "unexplained_gap" else "not constructible in this synthetic probe",
        }
        if condition != "unreviewed_gap":
            review["uncovered_cases"] = ["pure_terminal"]
    elif condition == "corner_subset":
        probe["cases"]["empty_first"]["samples"] = probe["cases"]["empty_first"]["samples"][:1]
    elif condition == "missing_ordinary":
        probe["cases"]["ordinary"]["samples"] = probe["cases"]["ordinary"]["samples"][:1]
    else:
        probe["cases"]["empty_first"]["samples"][0]["endpoint"] = "http://127.0.0.1:9999"
    probe_path.write_bytes(json_bytes(probe))
    review_path.write_bytes(json_bytes(review))
    document["probe_manifest"]["sha256"] = file_sha256(probe_path)
    document["source_review"]["sha256"] = file_sha256(review_path)
    save_approved_contract(contract, document)
    if condition in ("honest_gap", "corner_subset"):
        assert load_contract(contract, deployment, endpoints, "tiny", True, manifest)["scope"] == "formal"
    else:
        with pytest.raises(ValueError):
            load_contract(contract, deployment, endpoints, "tiny", True, manifest)


@pytest.mark.parametrize("problem", ["missing_done", "wrong_prompt_usage", "comment_only"])
def test_ordinary_probe_must_be_complete_sse_and_match_real_tokenizer(tmp_path, problem):
    _, manifest, _ = make_plan(tmp_path / "inputs", [{"prompt": "hello", "output_tokens": 1}],
                               provenance_scope="formal")
    endpoint = "http://127.0.0.1:12345"
    deployment, contract = fixture_contract(tmp_path / "deployment", manifest, [endpoint], scope="formal")
    document = json.loads(contract.read_bytes())
    probe_path = Path(document["probe_manifest"]["path"])
    probe = json.loads(probe_path.read_bytes())
    sample = probe["cases"]["ordinary"]["samples"][0]
    raw_path = Path(sample["raw_sse"]["path"])
    raw = raw_path.read_bytes()
    if problem == "missing_done":
        raw = raw.replace(b"data: [DONE]\n\n", b"")
    elif problem == "wrong_prompt_usage":
        raw = raw.replace(b'"prompt_tokens": 1', b'"prompt_tokens": 2').replace(
            b'"total_tokens": 3', b'"total_tokens": 4')
    else:
        raw = b": fixture\n\n"
    raw_path.write_bytes(raw)
    sample["raw_sse"]["sha256"] = file_sha256(raw_path)
    observation_path = Path(sample["independent_observation"]["path"])
    observation = json.loads(observation_path.read_bytes())
    observation["raw_sse_sha256"] = file_sha256(raw_path)
    observation_path.write_bytes(json_bytes(observation))
    sample["independent_observation"]["sha256"] = file_sha256(observation_path)
    probe_path.write_bytes(json_bytes(probe))
    document["probe_manifest"]["sha256"] = file_sha256(probe_path)
    save_approved_contract(contract, document)
    with pytest.raises(ValueError, match="ordinary|no completion"):
        load_contract(contract, deployment, [endpoint], "tiny", True, manifest)


@pytest.mark.parametrize("first,last,no_control,counts_special,count,events,expected_slo", [
    (False, False, False, True, 1, 2, "indeterminate"),
    (False, False, True, False, 1, 2, "indeterminate"),
    (True, False, False, True, 1, 2, "pass"),
    (False, True, False, True, 1, 2, "indeterminate"),
    (True, True, True, False, 1, 2, "indeterminate"),
    (True, True, True, False, 2, 3, "indeterminate"),
    (True, True, False, True, 2, 3, "pass"),
])
def test_synthetic_formal_protocol_success_is_separate_from_timing_observability(
        tmp_path, first, last, no_control, counts_special, count, events, expected_slo):
    async def run():
        # These are synthetic event/token statements, including control choices and
        # excluded special tokens; neither text nor finish establishes membership.
        async def respond(body, headers, writer):
            identity = "c-" + headers["x-request-id"]
            event_count = events if body["prompt"] == "hello" else 1
            token_count = count if body["prompt"] == "hello" else 1
            wire = b"".join(event(choice(identity, "", "length" if index == event_count - 1 else None))
                            for index in range(event_count))
            await send_response(writer, [wire + event(usage(identity, 1, token_count)) + b"data: [DONE]\n\n"])

        async with tcp_server(respond) as (endpoint, counter):
            args, _, _, binding = formal_inputs(tmp_path, endpoint, output_tokens=count)
            contract = Path(args.protocol_contract)
            document = json.loads(contract.read_bytes())
            assertions = {"first_choice": first, "last_choice": last, "no_control_choices": no_control,
                          "usage_counts_special_tokens": counts_special}
            document["assertions"] = assertions
            for key in ("source_review", "probe_manifest"):
                path = Path(document[key]["path"])
                nested = json.loads(path.read_bytes())
                nested["assertions"] = assertions
                path.write_bytes(json_bytes(nested))
                document[key]["sha256"] = file_sha256(path)
            save_approved_contract(contract, document)
            args.freeze = str(fixture_freeze(tmp_path / "freeze", scope="formal", binding=binding))
            run_result = await replay(args)
            row = json.loads((Path(args.output_dir) / "formal" / "results.jsonl").read_bytes())
            assert run_result["state"] == "complete" and len(counter["requests"]) == 2
            assert row["status"] == "success" and len(row["choice_times_s"]) == events
            assert row["first_output_offset_s"] == (row["choice_times_s"][0] if first else None)
            assert row["last_output_offset_s"] == (row["choice_times_s"][-1] if last else None)
            report = analyze(args.plan, args.output_dir)
            assert report["mode"] == "formal" and report["formal_contract_checks_passed"]
            assert not report["integrity_problems"] and report["batch_complete"]
            assert report["overall"]["planned"] == report["overall"]["success"] == 1
            assert report["overall"]["slo_indeterminate"] == (expected_slo == "indeterminate")
            detail = report["requests"][0]
            assert detail["slo"] == expected_slo
            assert (detail["planned_ttft_ms"] is not None) == first
            assert (detail["client_tpot_est_ms"] is not None) == (first and last and counts_special and count > 1)
            assert len(detail["chunk_itl_ms"]) == (events - 1 if first and last and no_control else 0)
            assert report["evidence_authenticity"] == "requires_primary_agent_verification"
            assert report["stage_acceptance"] is None

    asyncio.run(run())
