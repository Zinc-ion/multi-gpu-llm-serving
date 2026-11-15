import importlib.util
import json
from pathlib import Path


def test_analysis_joins_repeated_ids_and_keeps_failures_and_unmatched_records(tmp_path):
    path = Path(__file__).parents[1] / "tools/analyze_pd.py"
    spec = importlib.util.spec_from_file_location("analyze_pd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"event": "pd_gateway_assigned", "gateway_id": "a", "request_id": "repeated", "elapsed_ms": 1},
        {"event": "pd_gateway_open", "gateway_id": "a", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "elapsed_ms": 2},
        {"event": "pd_gateway_first", "gateway_id": "a", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "elapsed_ms": 5},
        {"event": "pd_gateway_handoff_done", "gateway_id": "a", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "elapsed_ms": 12, "handoff_ms": 6, "payload_bytes": 128},
        {"event": "pd_transfer_done", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "transfer_ms": 4, "payload_bytes": 128},
        {"event": "pd_export_done", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "export_ms": 2},
        {"event": "pd_gateway_second", "gateway_id": "a", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "elapsed_ms": 15},
        {"event": "pd_gateway_completed", "gateway_id": "a", "session_id": "s1", "transaction_id": "t1", "request_id": "repeated", "elapsed_ms": 30},
        {"event": "pd_gateway_assigned", "gateway_id": "b", "request_id": "repeated", "elapsed_ms": 1},
        {"event": "pd_gateway_failed", "gateway_id": "b", "request_id": "repeated", "elapsed_ms": 2, "status": 503},
        {"event": "pd_gateway_assigned", "gateway_id": "c", "request_id": "repeated", "elapsed_ms": 1},
        {"event": "pd_gateway_cancelled", "gateway_id": "c", "request_id": "repeated", "elapsed_ms": 3},
        {"event": "pd_released", "session_id": "unknown", "request_id": None},
    ]
    timing = tmp_path / "events.jsonl"
    timing.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = module.analyze([timing])
    assert result["outcomes"] == {"completed": 1, "failed": 1, "cancelled": 1}
    success = result["transactions"][0]
    assert success["payload_bytes"] == 128 and success["transfer_ms"] == 4
    assert success["second_minus_first_ms"] == 10
    assert success["handoff_to_generation_ratio"] == .2
    assert any(row["event"] == "pd_export_done" for row in success["events"])
    assert len(result["orphan_pd_events"]) == 1
