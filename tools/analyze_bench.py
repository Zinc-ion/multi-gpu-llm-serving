"""Check per-token timing semantics and recompute goodput from detailed results."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(args):
    data = json.loads(Path(args.result).read_text())
    errors = data["errors"]
    chunks_ok = []
    rows = []
    server = []
    if args.server_timings:
        server = [
            json.loads(line) for line in Path(args.server_timings).read_text().splitlines()
            if json.loads(line).get("event") == "chunk_sent" and json.loads(line).get("first")
        ]
    for index, (ttft, itl, count, error) in enumerate(zip(
        data["ttfts"], data["itls"], data["output_lens"], errors
    )):
        if error:
            rows.append({"index": index, "error": error})
            continue
        valid = len(itl) + 1 == count
        chunks_ok.append(valid)
        row = {
            "index": index, "output_tokens": count, "itl_samples": len(itl),
            "token_chunks_match": valid, "ttft_ms": ttft * 1000,
            "tpot_ms": sum(itl) / (count - 1) * 1000 if count > 1 else 0,
            "e2e_ms": (ttft + sum(itl)) * 1000,
            "start_time": data["start_times"][index],
        }
        request_id = args.request_id_prefix + str(index)
        candidates = [s for s in server if s["request_id"] == request_id]
        if candidates:
            # bench's initial validation request reuses the first ID; use the last record.
            match = candidates[-1]
            row["request_id"] = request_id
            row["server_first_chunk_ms"] = match["arrival_to_chunk_ms"]
            row["client_minus_server_ms"] = row["ttft_ms"] - match["arrival_to_chunk_ms"]
        rows.append(row)
    successful = [row for row in rows if "error" not in row]
    output = {
        "result": args.result, "requests": rows,
        "all_token_chunks_match": bool(chunks_ok) and all(chunks_ok),
        "relative_send_times_s": (np.array(data["start_times"]) - data["start_times"][0]).tolist(),
    }
    if args.compare_arrivals:
        other = json.loads(Path(args.compare_arrivals).read_text())
        other_relative = np.array(other["start_times"]) - other["start_times"][0]
        relative = np.array(output["relative_send_times_s"])
        output["arrival_comparison"] = {
            "other_result": args.compare_arrivals,
            "same_actual_relative_times": np.array_equal(relative, other_relative),
            "relative_time_differences_s": (relative - other_relative).tolist(),
        }
    if args.ttft_slo_ms is not None and args.tpot_slo_ms is not None:
        passing = sum(row["ttft_ms"] <= args.ttft_slo_ms and row["tpot_ms"] <= args.tpot_slo_ms for row in successful)
        output.update(
            slo={"ttft_ms": args.ttft_slo_ms, "tpot_ms": args.tpot_slo_ms},
            slo_pass_fraction=passing / len(rows),
            request_goodput=passing / data["duration"],
        )
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    if not output["all_token_chunks_match"]:
        raise SystemExit("Token/ITL count mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--server-timings")
    parser.add_argument("--request-id-prefix", default="")
    parser.add_argument("--compare-arrivals", help="Second detailed run using the same seed and request list")
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--output", required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
