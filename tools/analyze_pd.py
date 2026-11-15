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
        first = by_name.get("pd_gateway_first", {}).get("elapsed_ms")
        second = by_name.get("pd_gateway_second", {}).get("elapsed_ms")
        completed = by_name.get("pd_gateway_completed", {}).get("elapsed_ms")
        handoff = by_name.get("pd_gateway_handoff_done", {})
        group.update(first_ms=first, second_ms=second, server_generation_ms=completed,
                     second_minus_first_ms=second - first if first is not None and second is not None else None,
                     payload_bytes=handoff.get("payload_bytes"), handoff_ms=handoff.get("handoff_ms"))
        group["handoff_to_generation_ratio"] = (
            group["handoff_ms"] / completed if group["handoff_ms"] is not None and completed else None
        )
        group["transfer_ms"] = next((r["transfer_ms"] for r in events
                                     if r["event"] == "pd_transfer_done"), None)
    counts = Counter(group["outcome"] for group in groups.values())
    return {"transactions": list(groups.values()), "outcomes": dict(counts),
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
