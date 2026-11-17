import json

from tools.analyze_pd import analyze


def test_partial_pipeline_send_is_unknown_and_worker_outcomes_are_separate(tmp_path):
    ids = {"request_id": "same", "session_id": "s", "transaction_id": "t"}
    rows = [
        {"event": "pd_gateway_open", "gateway_id": "g", **ids, "elapsed_ms": 1, "route": "remote"},
        {"event": "pd_reserved", "role": "decode", **ids},
        {"event": "pd_region_sent", "role": "prefill", **ids, "payload_bytes": 64},
        {"event": "pd_pipeline_sender_closed", "role": "prefill", **ids, "frames": 1,
         "payload_bytes": 64, "socket_submitted_bytes": 128, "partial_send_unknown": True},
        {"event": "pd_failed", "role": "prefill", **ids, "reason": "connection error"},
        {"event": "pd_gateway_failed", "gateway_id": "g", **ids, "elapsed_ms": 4},
        {"event": "pd_gateway_frontend_closed", "gateway_id": "g", **ids, "elapsed_ms": 5},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = analyze([path])
    group = result["transactions"][0]
    assert group["payload_bytes"] == 64
    assert group["pipeline"]["socket_submitted_bytes_lower_bound"] == 128
    assert group["pipeline"]["partial_send_unknown"] is True
    assert result["routes"]["remote"]["observed_payload_bytes"] == 64
    assert result["routes"]["remote"]["unknown_payload_requests"] == 1
    assert group["outcome"] == "failed" and group["frontend_outcome"] == "closed"
    assert group["worker_outcomes"] == {"prefill": "failed", "decode": "incomplete"}
    assert group["client_outcome"] is None


def test_completed_single_token_bulk_has_known_zero_payload(tmp_path):
    ids = {"request_id": "single", "gateway_id": "g"}
    rows = [
        {"event": "pd_gateway_open", **ids, "elapsed_ms": 1, "transfer_mode": "bulk"},
        {"event": "pd_gateway_first", **ids, "elapsed_ms": 2},
        {"event": "pd_gateway_completed", **ids, "elapsed_ms": 3},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = analyze([path])
    assert result["transactions"][0]["payload_bytes"] == 0
    assert result["routes"]["remote"]["unknown_payload_requests"] == 0
