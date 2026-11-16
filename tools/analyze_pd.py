"""Join PD timing records by gateway/session/transaction, retaining failures."""

import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(paths):
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))

    groups = {}
    by_session = {}
    by_transaction = {}
    for record in records:
        gateway_id = record.get("gateway_id")
        if gateway_id:
            group = groups.setdefault(gateway_id, {"gateway_id": gateway_id, "events": []})
            group["request_id"] = record["request_id"]
            if record.get("session_id"):
                group["session_id"] = record["session_id"]
                by_session[record["session_id"]] = gateway_id
            if record.get("transaction_id"):
                group["transaction_id"] = record["transaction_id"]
                by_transaction[record["transaction_id"]] = gateway_id

    orphans = []
    for record in records:
        gateway_id = (record.get("gateway_id") or by_transaction.get(record.get("transaction_id"))
                      or by_session.get(record.get("session_id")))
        if gateway_id in groups:
            groups[gateway_id]["events"].append(record)
        elif record.get("event", "").startswith("pd_"):
            orphans.append(record)

    for group in groups.values():
        events = group["events"]
        by_name = {record["event"]: record for record in events
                   if record["event"].startswith("pd_gateway_")}
        group["route"] = by_name.get("pd_gateway_assigned", {}).get(
            "route", by_name.get("pd_gateway_open", {}).get(
                "route", by_name.get("pd_gateway_failed", {}).get(
                    "route", by_name.get("pd_gateway_cancelled", {}).get("route", "remote"))))
        if "pd_gateway_failed" in by_name:
            group["outcome"] = "failed"
        elif "pd_gateway_cancelled" in by_name or ("pd_gateway_frontend_closed" in by_name
                                                     and "pd_gateway_completed" not in by_name):
            group["outcome"] = "cancelled"
        elif "pd_gateway_completed" in by_name:
            group["outcome"] = "completed"
        else:
            group["outcome"] = "incomplete"
        group["frontend_closed"] = "pd_gateway_frontend_closed" in by_name
        group["frontend_outcome"] = (
            "completed" if "pd_gateway_frontend_completed" in by_name else
            "closed" if group["frontend_closed"] else "unknown"
        )
        group["client_outcome"] = None
        group["worker_outcomes"] = {}
        for role in ("prefill", "decode"):
            names = {record["event"] for record in events if record.get("role") == role}
            group["worker_outcomes"][role] = (
                "failed" if "pd_failed" in names else
                "cancelled" if "pd_cancelled" in names else
                "completed" if "pd_finished" in names else
                "handed_off" if "pd_p_released" in names else
                "incomplete" if names else "unknown"
            )
        first = by_name.get("pd_gateway_first", {}).get("elapsed_ms")
        second = by_name.get("pd_gateway_second", {}).get("elapsed_ms")
        completed = by_name.get("pd_gateway_completed", {}).get("elapsed_ms")
        handoff = by_name.get("pd_gateway_handoff_done", {})
        transfer = next((r for r in events if r["event"] == "pd_transfer_done"), {})
        group.update(first_ms=first, second_ms=second, server_generation_ms=completed,
                     second_minus_first_ms=second - first if first is not None and second is not None else None,
                     payload_bytes=(0 if group["route"] == "local" else
                                    transfer.get("payload_bytes", handoff.get("payload_bytes"))),
                     handoff_ms=handoff.get("handoff_ms"))
        if (group["route"] == "remote" and "pd_gateway_completed" in by_name
                and by_name.get("pd_gateway_open", {}).get("transfer_mode") == "bulk"
                and "pd_gateway_handoff_start" not in by_name):
            group["payload_bytes"] = 0
        if "scale_bytes" in transfer or "scale_bytes" in handoff:
            group["data_bytes"] = transfer.get("data_bytes", handoff.get("data_bytes"))
            group["scale_bytes"] = transfer.get("scale_bytes", handoff.get("scale_bytes"))
        group["handoff_to_generation_ratio"] = (
            group["handoff_ms"] / completed if group["handoff_ms"] is not None and completed else None
        )
        group["transfer_ms"] = transfer.get("transfer_ms")
        pipeline = [r for r in events if r["event"] == "pd_pipeline_done"]
        sender = [r for r in events if r["event"] == "pd_pipeline_sender_closed"]
        frames = [r for r in events if r["event"] == "pd_region_sent"]
        submitted = [r for r in events if r["event"] == "pd_region_socket_done"]
        if pipeline or sender or frames or any(r["event"] == "pd_pipeline_connected" for r in events):
            observed = sender[-1] if sender else pipeline[-1] if pipeline else None
            acked_bytes = observed["payload_bytes"] if observed else sum(r["payload_bytes"] for r in frames)
            acked_frames = observed["frames"] if observed else len(frames)
            group["payload_bytes"] = acked_bytes
            if observed and "scale_bytes" in observed:
                group["data_bytes"] = observed["data_bytes"]
                group["scale_bytes"] = observed["scale_bytes"]
            elif frames and "scale_bytes" in frames[0]:
                group["data_bytes"] = sum(r["data_bytes"] for r in frames)
                group["scale_bytes"] = sum(r["scale_bytes"] for r in frames)
            group["pipeline"] = {
                "acked_frames": acked_frames, "acked_payload_bytes": acked_bytes,
                "socket_submitted_bytes_lower_bound": (
                    sender[-1]["socket_submitted_bytes"] if sender
                    else sum(r["payload_bytes"] for r in submitted)),
                "partial_send_unknown": sender[-1]["partial_send_unknown"] if sender else None,
                "residual_wait_ms": pipeline[-1]["residual_wait_ms"] if pipeline else None,
            }
        group["payload_unknown"] = group["payload_bytes"] is None
        if "pipeline" in group and not pipeline:
            detail = group["pipeline"]
            group["payload_unknown"] = (
                detail["partial_send_unknown"] is not False or
                detail["socket_submitted_bytes_lower_bound"] > detail["acked_payload_bytes"]
            )
    counts = Counter(group["outcome"] for group in groups.values())
    routes = {}
    for route in ("remote", "local", "unassigned"):
        selected = [group for group in groups.values() if group["route"] == route]
        known_bytes = [group["payload_bytes"] for group in selected if group["payload_bytes"] is not None]
        routes[route] = {
            "requests": len(selected),
            "gateway_outcomes": dict(Counter(group["outcome"] for group in selected)),
            "frontend_outcomes": dict(Counter(group["frontend_outcome"] for group in selected)),
            "observed_payload_bytes": sum(known_bytes) if known_bytes else None,
            "unknown_payload_requests": sum(group["payload_unknown"] for group in selected),
        }
    return {"transactions": list(groups.values()), "outcomes": dict(counts),
            "routes": routes, "measurement_scope": "gateway and supplied worker timing records",
            "orphan_pd_events": orphans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", nargs="+", required=True, help="Gateway and worker JSONL files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.timings)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outcomes": result["outcomes"], "orphan_pd_events": len(result["orphan_pd_events"])}))


if __name__ == "__main__":
    main()
