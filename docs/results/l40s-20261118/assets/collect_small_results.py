"""Derive tables from saved request results and the original device samples."""

import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import statistics


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path,
                    default=Path(os.environ.get("PROJECT_ROOT", ".")) / "runs/acceptance-20261118")
root = parser.parse_args().root


def distribution(values):
    values = sorted(values)
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    result = {"samples": len(values), "mean": statistics.mean(values)}
    for percentile in (50, 95, 99):
        position = (len(values) - 1) * percentile / 100
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        result[f"p{percentile}"] = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return result


gpu_map = {
    "bf16-tp1": [0], "fp8-kv-tp1": [0], "bf16-tp2": [0, 1],
    "bf16-pd": [0, 1], "bf16-four-pd": [0, 1, 2, 3],
    "bf16-tp2-fixed": [0, 1], "bf16-four-pd-fixed": [0, 1, 2, 3],
    "bf16-tp1-final32": [0], "bf16-tp2-final32": [0, 1],
}
report = []
fixed_comparison = root / "tp1-vs-tp2-fixed.json"
fixed_aligned = fixed_comparison.exists() and json.loads(fixed_comparison.read_text())["tokens_identical"]
four_lifecycle = root / "bf16-four-pd-fixed-lifecycle.json"
four_aligned = four_lifecycle.exists() and json.loads(four_lifecycle.read_text())["all_match_tp1"]
for path in sorted(root.glob("*-c?.json")):
    run = json.loads(path.read_text())
    label = path.stem.rsplit("-c", 1)[0]
    gpu_ids = gpu_map[label]
    memory_path = root / (label + "-memory.jsonl")
    if memory_path.exists():
        memory_text = memory_path.read_text()
    else:
        with gzip.open(memory_path.with_suffix(".jsonl.gz"), "rt") as stream:
            memory_text = stream.read()
    samples = [json.loads(line) for line in memory_text.splitlines()]
    selected = [row for row in samples if row["phase"] == f"c{run['concurrency']}-including-warmups"]
    per_gpu = {gpu: [] for gpu in gpu_ids}
    totals = []
    for sample in selected:
        usage = {int(values[0]): int(values[1])
                 for values in (line.replace(" ", "").split(",") for line in sample["gpus"])}
        for gpu in gpu_ids:
            per_gpu[gpu].append(usage[gpu])
        totals.append(sum(usage[gpu] for gpu in gpu_ids))
    gaps = [b["monotonic_s"]-a["monotonic_s"] for a,b in zip(selected,selected[1:])]
    first_two = [row["itl"][0]*1000 for row in run["results"] if row["success"] and row["itl"]]
    record = {
        key: run[key] for key in (
            "label", "concurrency", "requests", "successful", "failed", "warmups",
            "mean_ttft_ms", "mean_tpot_ms", "tpot_samples", "wall_seconds",
            "successful_input_tokens", "successful_output_tokens",
            "output_tokens_per_s", "input_plus_output_tokens_per_s",
        )
    }
    record.update(
        gpu_ids=gpu_ids, source=path.name,
        ttft_ms=distribution([row["ttft"] * 1000 for row in run["results"] if row["success"]]),
        tpot_ms=distribution([row["tpot_s"] * 1000 for row in run["results"]
                              if row["success"] and row["tpot_s"] is not None]),
        percentile_method="linear interpolation at (n-1)*q; empirical short-sample percentiles",
        request_lengths=[{"request_id": row["request_id"], "success": row["success"],
                          "input_tokens": row["prompt_len"], "output_tokens": row["output_tokens"]}
                         for row in run["results"]],
        length_source="input: tokenizer-verified materialized prompt; output: streamed server usage",
        failure_reasons=dict(Counter(row["error"] for row in run["results"] if not row["success"])),
        first_two_sse_choice_gap_ms=distribution(first_two),
        mean_first_two_sse_choice_gap_ms=statistics.mean(first_two) if first_two else None,
        first_two_gap_samples=len(first_two),
        device_memory_sampled_max_mib={gpu:max(values) for gpu,values in per_gpu.items()},
        simultaneous_selected_devices_max_mib=max(totals),
        memory_samples=len(selected),
        memory_sample_mean_interval_s=statistics.mean(gaps) if gaps else None,
        memory_scope="nvidia-smi device used memory; after startup, including this batch's warmups and requests; sampled maximum, not exact peak",
        correctness_status=(
            "strict greedy check failed; diagnostic timing only"
            if label in ("bf16-tp2", "bf16-four-pd")
            or label in ("bf16-tp2-fixed", "bf16-tp2-final32") and not fixed_aligned
            or label == "bf16-four-pd-fixed" and not (fixed_aligned and four_aligned)
            else "limited fixed-input checks; not a full quality evaluation"),
    )
    report.append(record)
(root / "performance-summary.json").write_text(json.dumps(report, indent=2) + "\n")
csv_rows = []
for row in report:
    flat = {key: row[key] for key in (
        "label", "concurrency", "requests", "successful", "failed", "warmups",
        "wall_seconds", "output_tokens_per_s", "input_plus_output_tokens_per_s",
        "mean_first_two_sse_choice_gap_ms", "correctness_status",
    )}
    for metric in ("ttft_ms", "tpot_ms"):
        flat.update({f"{metric}_{key}": value for key, value in row[metric].items()})
    for key in ("input_tokens", "output_tokens"):
        lengths = [request[key] for request in row["request_lengths"] if request["success"]]
        flat[f"{key}_min"] = min(lengths) if lengths else None
        flat[f"{key}_max"] = max(lengths) if lengths else None
    for gpu in range(4):
        flat[f"gpu{gpu}_sampled_max_mib"] = row["device_memory_sampled_max_mib"].get(gpu)
    csv_rows.append(flat)
with (root / "performance-summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
capacities = {}
for path in sorted(root.glob("*.log")):
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith('{"event": "capacity",'):
            capacities[path.name] = json.loads(line)
if capacities:
    (root / "capacity-summary.json").write_text(json.dumps(capacities, indent=2) + "\n")
for row in report:
    print("{label}: {successful}/{requests}, TTFT={mean_ttft_ms:.3f}ms, TPOT={mean_tpot_ms:.3f}ms, out={output_tokens_per_s:.3f}/s, total={input_plus_output_tokens_per_s:.3f}/s, mem={simultaneous_selected_devices_max_mib}MiB".format(**row))
for path in root.glob("*-memory.jsonl"):
    with gzip.open(path.with_suffix(".jsonl.gz"), "wt") as stream:
        stream.write(path.read_text())
