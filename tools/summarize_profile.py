"""Summarize each original Torch trace independently; overlapping times stay separate."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path


def union_us(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def category(name, event_category):
    name = name.lower()
    if "nccl" in name:
        return "communication"
    if event_category != "kernel" or any(key in name for key in ("memcpy", "memset")):
        return "memory"
    if any(key in name for key in ("gemm", "gemv", "matmul", "attention", "cutlass")):
        return "compute"
    if any(key in name for key in ("elementwise", "reduce", "triton_poi", "triton_red",
                                   "rope", "norm", "rotary", "silu", "vectorized")):
        return "elementwise"
    return "other"


def summarize(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu = [event for event in events if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and event.get("ph") == "X" and event.get("dur", 0) > 0]
    if not gpu:
        raise ValueError(f"No GPU activity in {path}")
    names = Counter()
    counts = Counter()
    intervals = {}
    all_intervals = []
    for event in gpu:
        kind = category(event["name"], event["cat"])
        interval = (event["ts"], event["ts"] + event["dur"])
        intervals.setdefault(kind, []).append(interval)
        counts[kind] += 1
        names[event["name"]] += event["dur"]
        all_intervals.append(interval)
    first = min(start for start, _ in all_intervals)
    last = max(stop for _, stop in all_intervals)
    busy = union_us(all_intervals)
    runtime = [event for event in events if event.get("cat") == "cuda_runtime"]
    phases = [event for event in events if event.get("name") in ("mini.prefill", "mini.decode")
              and event.get("ph") == "X" and event.get("cat") == "user_annotation"]
    scopes = {}
    for name in ("mini.prefill", "mini.decode"):
        selected = [event for event in phases if event["name"] == name]
        scopes[name] = {"calls": len(selected), "host_scope_sum_ms": sum(row["dur"] for row in selected) / 1000}
    return {
        "file": str(path), "gpu_event_count": len(gpu), "gpu_span_ms": (last-first) / 1000,
        "gpu_busy_union_ms": busy / 1000, "uncovered_gpu_span_ms": (last-first-busy) / 1000,
        "categories": {
            kind: {"events": counts[kind], "duration_sum_ms": sum(stop-start for start, stop in values) / 1000,
                   "busy_union_ms": union_us(values) / 1000}
            for kind, values in intervals.items()
        },
        "top_kernels_by_summed_duration": [
            {"name": name, "duration_sum_ms": duration / 1000}
            for name, duration in names.most_common(12)
        ],
        "host_scopes": scopes,
        "compile_inner_events": sum(
            "compile_inner" in row.get("name", "") or "compile_fx_inner" in row.get("name", "")
            for row in events
        ),
        "host_cuda_runtime_sum_ms": sum(row.get("dur", 0) for row in runtime) / 1000,
        "host_sync_calls_sum_ms": sum(
            row.get("dur", 0) for row in runtime
            if "Synchronize" in row["name"] or "Memcpy" in row["name"]
        ) / 1000,
        "limits": [
            "category inferred from kernel names; unresolved kernels remain other",
            "category duration sums can overlap; they are not additive wall-time fractions",
            "NCCL kernel time includes execution and possible peer waits",
            "uncovered GPU intervals can include host dispatch and synchronization; not pure removable wait",
            "host scopes include launches and blocking work; host and GPU times overlap",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"ranks": [summarize(path) for path in args.traces],
              "aggregation": "per trace; never add different rank times as elapsed request time"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
