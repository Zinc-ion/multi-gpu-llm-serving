"""Derive block-rounded request capacity from an actual startup log."""

import argparse
import json
import math
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    records = [
        json.loads(line) for line in Path(args.log).read_text().splitlines()
        if line.startswith('{"event": "capacity"')
    ]
    if len(records) != 1:
        raise ValueError("Expected one capacity record from a single startup")
    measured = records[0]
    bytes_per_token = measured["block_bytes"] // config["block_size"]
    rows = []
    for prompt in (1024, 4096, 16384):
        length = prompt + args.output_tokens
        blocks = math.ceil(length / config["block_size"])
        rows.append({
            "input_tokens": prompt, "output_tokens": args.output_tokens,
            "blocks_per_request": blocks,
            "kv_capacity_concurrency_upper_bound": measured["block_count"] // blocks,
            "within_model_length": length <= config["max_model_length"],
        })
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_log": args.log, "config": config,
        "measured": measured, "bytes_per_token": bytes_per_token,
        "kv_bytes": measured["block_count"] * measured["block_bytes"],
        "requests": rows,
        "interpretation": "Block-pool capacity only; latency, stable concurrency and goodput require benchmarking.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
