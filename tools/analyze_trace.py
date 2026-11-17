"""Analyze one replay against its complete planned arrival and frozen SLO."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .prepare_trace import file_sha256, load_plan, read_json
    from .replay_trace import (CompletionStream, ProtocolError, SSEDecoder, load_contract, load_freeze,
                               output_event_times, validate_freeze_binding)
else:
    from prepare_trace import file_sha256, load_plan, read_json
    from replay_trace import (CompletionStream, ProtocolError, SSEDecoder, load_contract, load_freeze,
                              output_event_times, validate_freeze_binding)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None}

    def percentile(percent):
        position = (len(values) - 1) * percent / 100
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": percentile(50), "p90": percentile(90), "p99": percentile(99)}


def read_results(path):
    if not path.exists():
        return [], ["results_missing"]
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    else:
        return [], ["results_truncated"]
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return rows, ["results_invalid_json"]
        if not isinstance(row, dict):
            return rows, ["results_invalid_row"]
        rows.append(row)
    return rows, []


def validate_evidence(run, manifest):
    contract = freeze = None
    problems = []
    saved_contract = run.get("protocol_contract")
    saved_freeze = run.get("freeze")
    try:
        if saved_contract is not None:
            contract = load_contract(saved_contract["path"], saved_contract["deployment_path"],
                                     run["endpoints"], run["model"], run["ignore_eos"], manifest)
            if (contract["sha256"] != saved_contract["sha256"]
                    or contract["deployment_sha256"] != saved_contract["deployment_sha256"]):
                raise ValueError("protocol evidence changed since replay")
        if saved_freeze is not None:
            freeze = load_freeze(saved_freeze["path"])
            if freeze["sha256"] != saved_freeze["sha256"]:
                raise ValueError("SLO freeze changed since replay")
            if freeze["scope"] == "formal" and (contract is None or contract["scope"] != "formal"):
                raise ValueError("formal SLO lacks formal protocol evidence")
            replay_tool_hash = file_sha256(Path(__file__).with_name("replay_trace.py"))
            if freeze["scope"] == "formal" and run["tool_sha256"] != replay_tool_hash:
                raise ValueError("replay tool has changed since the run")
            validate_freeze_binding(freeze, manifest, run.get("warmup_plan_sha256"), {
                "deployment_sha256": contract["deployment_sha256"] if contract else None,
                "replay_tool_sha256": replay_tool_hash, "analyze_tool_sha256": file_sha256(__file__),
                "concurrency": run["concurrency"],
                "request_timeout_s": run["request_timeout_s"], "batch_timeout_s": run["batch_timeout_s"],
                "max_raw_bytes": run["max_raw_bytes"], "gpu_count": run["gpu_count"], "ignore_eos": run["ignore_eos"],
            }, plan_manifest_hash=run["plan_manifest_sha256"],
               warmup_manifest_hash=run["warmup_manifest_sha256"])
        start = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        for evidence in (contract, freeze):
            if evidence is not None and evidence["approval"] is not None:
                approved = datetime.fromisoformat(evidence["approval"]["approved_at"].replace("Z", "+00:00"))
                if approved > start:
                    raise ValueError("approval occurred after replay started")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        problems.append(f"evidence_invalid: {exc}")
        contract = freeze = None
    return contract, freeze, problems


def valid_result(row, planned, manifest, run, contract):
    required = {"dispatch_offset_s", "completed_offset_s", "choice_times_s", "first_output_offset_s",
                "last_output_offset_s", "finish_offset_s", "error_class"}
    if (not required.issubset(row) or type(row.get("ordinal")) is not int
            or row["ordinal"] != planned["ordinal"] or row.get("phase") != planned["phase"]
            or row.get("plan_sha256") != manifest["plan_sha256"]
            or type(row.get("planned_offset_s")) not in (int, float)
            or row.get("planned_offset_s") != planned["offset_s"]
            or row.get("endpoint") != run["endpoints"][planned["ordinal"] % len(run["endpoints"])]
            or row.get("status") not in ("success", "failed", "not_executed")):
        return False
    dispatch = row.get("dispatch_offset_s")
    completed = row.get("completed_offset_s")
    if row["status"] == "not_executed":
        return dispatch is None and completed is None
    if completed is None or type(completed) not in (int, float) or not math.isfinite(completed):
        return False
    if dispatch is not None and (type(dispatch) not in (int, float) or not math.isfinite(dispatch)
                                 or dispatch < planned["offset_s"] or completed < dispatch):
        return False
    if row["status"] == "failed":
        return isinstance(row.get("error_class"), str)
    usage = row.get("usage")
    if (dispatch is None or row.get("error_class") is not None or row.get("error") is not None
            or row.get("http_status") != 200 or row.get("done") is not True
            or row.get("finish_reason") not in ("stop", "length")
            or not isinstance(row.get("response_id"), str) or not row["response_id"]
            or not isinstance(usage, dict)
            or any(type(usage.get(key)) is not int for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or usage["prompt_tokens"] != planned["prompt_tokens"]
            or not 0 < usage["completion_tokens"] <= planned["output_tokens"]
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        return False
    times = row.get("choice_times_s")
    if (not isinstance(times, list) or not times or
            any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
            or times != sorted(times) or times[0] < dispatch or times[-1] > completed
            or row.get("finish_offset_s") != times[-1]):
        return False
    if contract is not None:
        first, last = output_event_times(times, usage["completion_tokens"], contract["assertions"])
        if row.get("first_output_offset_s") != first or row.get("last_output_offset_s") != last:
            return False
    return True


def validate_raw(row, planned, run, directory):
    raw_path = directory / "raw" / f"{planned['ordinal']:08d}.sse"
    if row["dispatch_offset_s"] is None:
        if (row.get("raw_path") is not None or row.get("raw_sha256") is not None
                or row.get("raw_bytes") != 0 or row.get("received_body_bytes") != 0
                or row.get("raw_truncated") is not False or raw_path.exists()):
            return None
        return "no_dispatch"
    legacy = "raw_sha256" not in row
    if (row.get("raw_path") != f"raw/{planned['ordinal']:08d}.sse"
            or raw_path.resolve() != raw_path or not raw_path.is_file()
            or type(row.get("raw_bytes")) is not int or type(row.get("received_body_bytes")) is not int
            or type(row.get("raw_truncated")) is not bool or not 0 <= row["raw_bytes"] <= run["max_raw_bytes"]
            or row["received_body_bytes"] != row["raw_bytes"] + int(row["raw_truncated"])
            or (row["raw_truncated"] and row["raw_bytes"] != run["max_raw_bytes"])
            or (row["error_class"] == "RawLimitError" and not row["raw_truncated"])
            or (not legacy and ("content_type" not in row or not isinstance(row["raw_sha256"], str)))
            or not isinstance(row.get("choice_times_s"), list)):
        return None
    status = row.get("http_status")
    encoding = row.get("content_encoding")
    if (status is None and (encoding is not None or row.get("content_type") is not None)
            or status is not None and (type(status) is not int or not 100 <= status <= 999
                                       or not isinstance(encoding, str)
                                       or (not legacy and not isinstance(row["content_type"], str)))):
        return None
    if row["status"] == "success" and (row["raw_truncated"] or row["content_encoding"] != "identity"
                                        or (not legacy and row["content_type"] != "text/event-stream")):
        return None
    parse_sse = (row.get("http_status") == 200 and row.get("content_encoding") == "identity"
                 and (row.get("content_type") == "text/event-stream"
                      or (legacy and (row["status"] == "success" or row.get("response_id") is not None
                                      or row.get("done") is True or row["choice_times_s"]))))
    parser = SSEDecoder()
    stream = CompletionStream(planned["prompt_tokens"], planned["output_tokens"], run["model"])
    digest = hashlib.sha256()
    size = 0
    parse_error = False
    try:
        with raw_path.open("rb") as raw:
            for chunk in iter(lambda: raw.read(65536), b""):
                size += len(chunk)
                if size > row["raw_bytes"]:
                    return None
                digest.update(chunk)
                if parse_sse and not parse_error:
                    try:
                        for event, data, timestamp in parser.feed(chunk, 0):
                            stream.event(event, data, timestamp)
                    except (ProtocolError, UnicodeError):
                        parse_error = True
        if parse_sse and not parse_error and (row["status"] == "success" or row["raw_truncated"]
                                               or row["error_class"] in ("ProtocolError", "UnicodeDecodeError")):
            try:
                for event, data, timestamp in parser.finish(truncated=row["raw_truncated"]):
                    stream.event(event, data, timestamp)
                if row["status"] == "success":
                    stream.finish()
            except (ProtocolError, UnicodeError):
                parse_error = True
    except OSError:
        return None
    if size != row["raw_bytes"] or (not legacy and digest.hexdigest() != row["raw_sha256"]):
        return None
    if row["status"] == "success" and (not parse_sse or parse_error):
        return None
    if parse_sse and (stream.identity != row.get("response_id") or stream.usage != row.get("usage")
                      or stream.finish_reason != row.get("finish_reason") or stream.done != row.get("done")
                      or len(stream.choice_times) != len(row["choice_times_s"])):
        return None
    if not parse_sse and (row.get("response_id") is not None or row.get("usage") is not None
                          or row.get("finish_reason") is not None or row.get("done") is not False
                          or row["choice_times_s"]):
        return None
    return "legacy_unbound" if legacy else "capture_bound"


def inspect_results(manifest, planned_rows, run, directory, section, contract, problems):
    prefix = "" if section == "formal" else "warmup_"
    results_path = directory / "results.jsonl"
    rows, read_problems = read_results(results_path)
    problems.extend(prefix + problem for problem in read_problems)
    summary = run.get(section)
    globally_invalid = bool(read_problems)
    if summary is None:
        problems.append(prefix + "summary_missing")
        globally_invalid = True
    elif results_path.exists() and summary.get("results_sha256") != file_sha256(results_path):
        problems.append(prefix + "results_hash_mismatch")
        globally_invalid = True
    planned_ids = {row["request_id"] for row in planned_rows}
    by_id = defaultdict(list)
    unknown = []
    for row in rows:
        request_id = row.get("request_id")
        if not isinstance(request_id, str):
            unknown.append(str(request_id))
        else:
            by_id[request_id].append(row)
    missing = sorted(request_id for request_id in planned_ids if request_id not in by_id)
    duplicate = sorted(request_id for request_id, entries in by_id.items() if len(entries) > 1)
    unknown.extend(sorted(request_id for request_id in by_id if request_id not in planned_ids))
    for name, items in (("missing_request_ids", missing), ("duplicate_request_ids", duplicate),
                        ("unknown_request_ids", unknown)):
        if items:
            problems.append(prefix + name)
    invalid = []
    checked = []
    started = set()
    for planned in planned_rows:
        matches = by_id.get(planned["request_id"], [])
        result = matches[0] if len(matches) == 1 and not globally_invalid else None
        binding = None
        if result is not None:
            if valid_result(result, planned, manifest, run, contract):
                if result["dispatch_offset_s"] is not None:
                    started.add(planned["request_id"])
                binding = validate_raw(result, planned, run, directory)
            if binding is None:
                invalid.append(planned["request_id"])
                result = None
            elif binding == "legacy_unbound" and "legacy_raw_unbound" not in problems:
                problems.append("legacy_raw_unbound")
        checked.append((result, binding))
    if invalid:
        problems.append(prefix + "invalid_results")
    return checked, missing, duplicate, unknown, invalid, started


def metric_row(planned, result, contract, freeze, binding=None):
    detail = {"request_id": planned["request_id"], "ordinal": planned["ordinal"],
              "phase": planned["phase"], "prompt_tokens": planned["prompt_tokens"],
              "output_budget": planned["output_tokens"], "status": "missing" if result is None else result["status"],
              "failure_reason": None, "planned_ttft_ms": None, "dispatch_ttft_ms": None,
              "client_tpot_est_ms": None, "planned_e2e_ms": None, "dispatch_delay_ms": None,
              "chunk_itl_ms": [], "completion_offset_s": None, "slo": None,
              "raw_capture_binding": binding, "completion_tokens": None}
    if result is None:
        detail["failure_reason"] = "missing_or_invalid_result"
        if freeze is not None:
            detail["slo"] = "failed"
        return detail
    if result["dispatch_offset_s"] is not None:
        detail["dispatch_delay_ms"] = (result["dispatch_offset_s"] - planned["offset_s"]) * 1000
    detail["completion_offset_s"] = result["completed_offset_s"]
    if result["status"] != "success":
        detail["failure_reason"] = result["error_class"] or result["status"]
        if freeze is not None:
            detail["slo"] = "failed"
        return detail
    detail["planned_e2e_ms"] = (result["completed_offset_s"] - planned["offset_s"]) * 1000
    if binding == "capture_bound":
        detail["completion_tokens"] = result["usage"]["completion_tokens"]
    first = result["first_output_offset_s"]
    last = result["last_output_offset_s"]
    choice_times = result["choice_times_s"]
    count = result["usage"]["completion_tokens"]
    if contract is not None and first is not None:
        detail["planned_ttft_ms"] = (first - planned["offset_s"]) * 1000
        detail["dispatch_ttft_ms"] = (first - result["dispatch_offset_s"]) * 1000
    if contract is not None and first is not None and last is not None:
        if len(choice_times) > 1:
            if contract["assertions"]["no_control_choices"]:
                detail["chunk_itl_ms"] = [(right - left) * 1000 for left, right in zip(choice_times, choice_times[1:])]
            if count > 1 and contract["assertions"]["usage_counts_special_tokens"]:
                detail["client_tpot_est_ms"] = (last - first) * 1000 / (count - 1)
    if freeze is not None:
        ttft = detail["planned_ttft_ms"]
        limit = next(rule["ttft_ms"] for rule in freeze["ttft_ms_by_prompt_tokens"]
                     if rule["max_prompt_tokens"] is None or planned["prompt_tokens"] <= rule["max_prompt_tokens"])
        if ttft is None:
            detail["slo"] = "indeterminate"
        elif ttft > limit:
            detail["slo"] = "not_met"
        elif count == 1 and contract["assertions"]["usage_counts_special_tokens"]:
            detail["slo"] = "pass"
        elif detail["client_tpot_est_ms"] is None:
            detail["slo"] = "indeterminate"
        elif detail["client_tpot_est_ms"] > freeze["tpot_ms"]:
            detail["slo"] = "not_met"
        else:
            detail["slo"] = "pass"
    return detail


def summarize(details, duration, window_end, gpu_count, frozen, started=None):
    count = len(details)
    success = sum(row["status"] == "success" for row in details)
    slo = Counter(row["slo"] for row in details if row["slo"] is not None)
    passing = slo.get("pass", 0) if frozen else None
    metrics = ("dispatch_delay_ms", "planned_ttft_ms", "dispatch_ttft_ms", "client_tpot_est_ms", "planned_e2e_ms")
    distributions = {name: distribution([row[name] for row in details if row[name] is not None]) for name in metrics}
    distributions["chunk_itl_ms"] = distribution([value for row in details for value in row["chunk_itl_ms"]])
    completed = [row["completion_offset_s"] for row in details if row["completion_offset_s"] is not None]
    if started is None:
        started = any(row["dispatch_delay_ms"] is not None for row in details)
    successful_tokens = None
    if started and all(row["completion_tokens"] is not None for row in details if row["status"] == "success"):
        successful_tokens = sum(row["completion_tokens"] for row in details if row["completion_tokens"] is not None)
    return {
        "planned": count, "success": success, "failed_or_missing": count - success,
        "failure_reasons": dict(Counter(row["failure_reason"] for row in details if row["failure_reason"])),
        "slo_pass": passing, "slo_not_met": slo.get("not_met", 0) if frozen else None,
        "slo_indeterminate": slo.get("indeterminate", 0) if frozen else None,
        "slo_failed": slo.get("failed", 0) if frozen else None,
        "slo_pass_fraction": passing / count if frozen and count else None,
        "planned_window_s": duration, "offered_requests_per_s": count / duration if duration else None,
        "observed_passing_requests_per_s": passing / duration if frozen and duration else None,
        "observed_passing_requests_per_s_per_gpu": passing / duration / gpu_count if frozen and duration else None,
        "late_dispatches": sum(row["dispatch_delay_ms"] is not None and row["dispatch_delay_ms"] > 0 for row in details),
        "distributions_ms": distributions,
        "na_counts": {name: count - distributions[name]["n"] for name in metrics},
        "completed_after_window": sum(value > window_end for value in completed),
        "latest_completion_offset_s": max(completed) if completed else None,
        "completion_tail_s": max(0, max(completed) - window_end) if completed else None,
        "successful_completion_tokens": successful_tokens,
    }


def analyze(plan_path, run_dir):
    manifest, planned_rows = load_plan(plan_path)
    run_dir = Path(run_dir).resolve()
    run = read_json(run_dir / "run.json")
    if run["schema"] != "m8-replay-v1":
        raise ValueError("invalid replay schema")
    if type(run["gpu_count"]) is not int or run["gpu_count"] <= 0 or not run["endpoints"]:
        raise ValueError("invalid replay GPU count or endpoints")
    problems = []
    if run["plan_sha256"] != manifest["plan_sha256"]:
        problems.append("plan_hash_mismatch")
    if run["plan_manifest_sha256"] != file_sha256(str(plan_path) + ".manifest.json"):
        problems.append("plan_manifest_hash_mismatch")
    warmup = warmup_rows = None
    if run["warmup_plan_path"] is not None:
        warmup, warmup_rows = load_plan(run["warmup_plan_path"])
        if (warmup["kind"] != "warmup" or warmup["plan_sha256"] != run["warmup_plan_sha256"]
                or file_sha256(str(run["warmup_plan_path"]) + ".manifest.json") != run["warmup_manifest_sha256"]):
            problems.append("warmup_plan_hash_mismatch")
    elif (run["warmup_plan_sha256"] is not None or run["warmup_manifest_sha256"] is not None
          or run["budget"]["warmup_requests"] or run.get("warmup") is not None
          or (run_dir / "warmup").exists()):
        problems.append("warmup_plan_hash_mismatch")
    if run["state"] != "complete" or run.get("formal", {}).get("stop_reason") is not None:
        problems.append("batch_incomplete")
    contract, freeze, evidence_problems = validate_evidence(run, manifest)
    problems.extend(evidence_problems)
    if (warmup is not None and contract is not None and contract["scope"] == "formal"
            and read_json(warmup["tokenizer"]["provenance_path"]).get("scope") not in (None, "formal")):
        problems.append("warmup_evidence_scope_mismatch")
    warmup_valid = "warmup_plan_hash_mismatch" not in problems
    warmup_validation = None
    if warmup is not None:
        before = len(problems)
        checked, warm_missing, warm_duplicate, warm_unknown, warm_invalid, _ = inspect_results(
            warmup, warmup_rows, run, run_dir / "warmup", "warmup", contract, problems)
        warmup_valid = (not any(problem != "legacy_raw_unbound" for problem in problems[before:])
                        and "warmup_plan_hash_mismatch" not in problems
                        and all(result is not None and result["status"] == "success" for result, _ in checked))
        if not warmup_valid:
            problems.append("warmup_not_successful")
        warmup_validation = {
            "planned": len(warmup_rows),
            "success": sum(result is not None and result["status"] == "success" for result, _ in checked),
            "missing_ids": warm_missing, "duplicate_ids": warm_duplicate,
            "unknown_ids": warm_unknown, "invalid_ids": warm_invalid,
        }
    checked, missing, duplicate, unknown, invalid, started = inspect_results(
        manifest, planned_rows, run, run_dir / "formal", "formal", contract, problems)
    if ("batch_incomplete" not in problems and any(
            result is not None and (binding == "no_dispatch"
                                    or result["error_class"] in ("BatchDeadline", "CancelledError"))
            for result, binding in checked)):
        problems.append("batch_incomplete")
    globally_invalid = any(problem in problems for problem in (
        "plan_hash_mismatch", "plan_manifest_hash_mismatch", "results_hash_mismatch"))
    details = []
    for planned, (result, binding) in zip(planned_rows, checked):
        if globally_invalid or (not warmup_valid and result is not None and result["status"] == "success"):
            result = None
            binding = None
        details.append(metric_row(planned, result, contract, freeze if contract is not None else None, binding))
    end_offset = run.get("formal", {}).get("end_offset_s")
    if started and (type(end_offset) not in (int, float) or not math.isfinite(end_offset) or end_offset <= 0
                    or any(result is not None and result["completed_offset_s"] is not None
                           and result["completed_offset_s"] > end_offset for result, _ in checked)):
        problems.append("formal_timing_invalid")
        end_offset = None
    if not started:
        end_offset = None
    phases = {}
    for phase in manifest["phases"]:
        selected = [row for row in details if row["phase"] == phase["name"]]
        phases[phase["name"]] = summarize(
            selected, phase["duration_s"], phase["end_s"], run["gpu_count"],
            freeze is not None and contract is not None,
            any(row["request_id"] in started for row in selected))
    complete = (not any(not problem.startswith("evidence_invalid:") for problem in problems)
                and all(row["status"] != "not_executed" for row in details))
    if freeze is None or contract is None:
        mode = "capture_only"
    elif freeze["scope"] == "fixture" or contract["scope"] == "fixture":
        mode = "fixture"
    else:
        mode = "formal"
    overall = summarize(details, manifest["window_s"], manifest["window_s"], run["gpu_count"],
                        freeze is not None and contract is not None, bool(started))
    return {
        "schema": "m8-trace-analysis-v1", "plan_sha256": manifest["plan_sha256"],
        "analysis_tool_sha256": file_sha256(__file__),
        "mode": mode, "batch_complete": complete,
        "formal_contract_checks_passed": mode == "formal" and complete and not problems,
        "evidence_authenticity": "requires_primary_agent_verification", "stage_acceptance": None,
        "integrity_problems": problems, "missing_ids": missing, "duplicate_ids": duplicate,
        "unknown_ids": unknown, "invalid_ids": invalid,
        "overall": overall,
        "phases": phases, "requests": details,
        "formal_end_offset_s": end_offset,
        "successful_output_tokens_per_s": (overall["successful_completion_tokens"] / end_offset
                                            if end_offset is not None
                                            and overall["successful_completion_tokens"] is not None else None),
        "timing_note": "TPOT is a client first/last event estimate; chunk ITL is not token ITL; bias is unknown.",
        "budget": run["budget"], "warmup": run.get("warmup"), "warmup_validation": warmup_validation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.plan, args.run_dir)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "batch_complete": result["batch_complete"],
                      "integrity_problems": result["integrity_problems"], "overall": result["overall"]}, indent=2))
    if not result["batch_complete"] or result["integrity_problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
