import importlib.util
import json
from pathlib import Path


def test_failed_and_cancelled_transfers_keep_only_observed_bytes(tmp_path):
    path = Path(__file__).parents[1] / "tools/analyze_pd.py"
    spec = importlib.util.spec_from_file_location("analyze_pd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    for index, outcome in enumerate(("failed", "cancelled", "failed_before_transfer")):
        ids = {"gateway_id": str(index), "request_id": "same",
               "session_id": f"s{index}", "transaction_id": f"t{index}"}
        rows.extend([
            {"event": "pd_gateway_open", **ids, "elapsed_ms": 3.6619689999497496},
            {"event": "pd_gateway_handoff_start", **ids, "elapsed_ms": 4.521655000644387,
             "payload_bytes": 256},
        ])
        if outcome != "failed_before_transfer":
            rows.append({"event": "pd_transfer_done", "request_id": "same",
                         "session_id": f"s{index}", "transaction_id": f"t{index}",
                         "payload_bytes": 256, "transfer_ms": 0.6278500004555099})
        rows.append({"event": "pd_gateway_cancelled" if outcome == "cancelled" else "pd_gateway_failed",
                     **ids, "elapsed_ms": 5.621293001240701})

    timing = tmp_path / "events.jsonl"
    timing.write_text("".join(json.dumps(row) + "\n" for row in rows))
    groups = module.analyze([timing])["transactions"]
    for group, outcome in zip(groups, ("failed", "cancelled", "failed")):
        assert group["outcome"] == outcome
        assert group["handoff_ms"] is None
        assert group["handoff_to_generation_ratio"] is None
    for group in groups[:2]:
        assert group["payload_bytes"] == 256
        assert group["transfer_ms"] == 0.6278500004555099
    assert groups[2]["payload_bytes"] is None
    assert groups[2]["transfer_ms"] is None
