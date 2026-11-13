"""Pinned-host PCIe copies and NCCL point-to-point measurements (remote only)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
import torch.distributed as dist


MIB = 1024 ** 2
SIZES_MIB = [1, 4, 16, 64, 160, 640, 1024, 2560]


def summarize(milliseconds, size, mode):
    values = np.asarray(milliseconds)
    return {
        "mode": mode, "bytes": size, "samples_ms": milliseconds,
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
        "effective_GB_s": size / np.median(values) / 1e6,
        "fraction_of_gen4_x16": size / np.median(values) / 1e6 / (16 * 16 * 128 / 130 / 8),
    }


def link_snapshot():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
        "--format=csv,noheader",
    ], text=True, timeout=15).strip()


def host_mapping(tensor):
    address = tensor.data_ptr()
    for line in Path("/proc/self/maps").read_text().splitlines():
        start, end = (int(x, 16) for x in line.split()[0].split("-"))
        if start <= address < end:
            prefix = f"{start:x} "
            return next((row for row in Path("/proc/self/numa_maps").read_text().splitlines() if row.startswith(prefix)), line)


def copies(args, size):
    a, b = args.gpus[0], args.gpus[-1]
    host_in = torch.empty(size, dtype=torch.uint8, pin_memory=True).fill_(1)
    host_out = torch.empty(size, dtype=torch.uint8, pin_memory=True).zero_()
    device_a = torch.empty(size, dtype=torch.uint8, device=f"cuda:{a}")
    device_a.fill_(3)
    if args.mode in ("bidirectional", "twohop", "dual-d2h"):
        device_b = torch.empty(size, dtype=torch.uint8, device=f"cuda:{b}")
        device_b.fill_(4)
    stream_a = torch.cuda.Stream(device=a)
    stream_b = torch.cuda.Stream(device=b)
    # Both streams on one device for simultaneous H2D and D2H.
    reverse = torch.cuda.Stream(device=a)
    torch.cuda.synchronize(a)
    torch.cuda.synchronize(b)
    if args.mode == "twohop" and a == b:
        raise ValueError("twohop requires two distinct GPUs")
    samples = []
    wall_samples = []
    links = []
    # Keep traffic active until NVML has sampled the link, including small sizes.
    with ThreadPoolExecutor(max_workers=1) as monitor:
        snapshot = monitor.submit(link_snapshot)
        while not snapshot.done():
            with torch.cuda.device(a), torch.cuda.stream(stream_a):
                device_a.copy_(host_in, non_blocking=True)
            stream_a.synchronize()
        links.append(snapshot.result())
    device_a.fill_(3)
    torch.cuda.synchronize(a)
    for iteration in range(args.warmup + args.repeats):
        wall_start = time.perf_counter()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with torch.cuda.device(a), torch.cuda.stream(stream_a):
            starts[0].record()
            if args.mode == "h2d":
                device_a.copy_(host_in, non_blocking=True)
            else:
                host_out.copy_(device_a, non_blocking=True)
            ends[0].record()
        if args.mode == "bidirectional":
            with torch.cuda.device(a), torch.cuda.stream(reverse):
                starts[1].record()
                device_b.copy_(host_in, non_blocking=True)
                ends[1].record()
        elif args.mode in ("twohop", "dual-d2h"):
            if args.mode == "twohop":
                ends[0].synchronize()  # Deliberately serial; pipelining belongs to M6.
            with torch.cuda.device(b), torch.cuda.stream(stream_b):
                starts[1].record()
                if args.mode == "twohop":
                    device_b.copy_(host_out, non_blocking=True)
                else:
                    host_in.copy_(device_b, non_blocking=True)
                ends[1].record()
        ends[0].synchronize()
        first = starts[0].elapsed_time(ends[0])
        if args.mode in ("bidirectional", "twohop", "dual-d2h"):
            ends[1].synchronize()
            second = starts[1].elapsed_time(ends[1])
            elapsed = first + second if args.mode == "twohop" else max(first, second)
        else:
            elapsed = first
        if iteration >= args.warmup:
            samples.append(elapsed)
            wall_samples.append((time.perf_counter() - wall_start) * 1000)
    row = summarize(samples, size, args.mode)
    if args.mode in ("bidirectional", "dual-d2h"):
        row["aggregate_GB_s"] = row["effective_GB_s"] * 2
    row["link_under_load"] = links
    row["host_buffer_numa"] = [host_mapping(host_in), host_mapping(host_out)]
    if args.mode in ("bidirectional", "dual-d2h"):
        row["event_interval"] = "maximum per-stream duration; host dispatch skew excluded"
    elif args.mode == "twohop":
        row["event_interval"] = "sum of serial per-hop durations; host handoff excluded"
    else:
        row["event_interval"] = "single-copy stream duration"
    row["samples_wall_ms"] = wall_samples
    row["median_wall_ms"] = float(np.median(wall_samples))
    row["effective_wall_GB_s"] = size / np.median(wall_samples) / 1e6
    return row


def nccl(args, size):
    rank = dist.get_rank()
    gpu = args.gpus[rank]
    torch.cuda.set_device(gpu)
    buffer = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")
    samples = []
    links = []
    with ThreadPoolExecutor(max_workers=1) as monitor:
        if rank == args.sender:
            snapshot = monitor.submit(link_snapshot)
        active = torch.ones((), dtype=torch.int, device=f"cuda:{gpu}")
        while True:
            if rank == args.sender:
                active.fill_(int(not snapshot.done()))
            dist.broadcast(active, src=args.sender)
            if not active.item():
                break
            work = dist.isend(buffer, dst=1-rank) if rank == args.sender else dist.irecv(buffer, src=1-rank)
            work.wait()
            torch.cuda.synchronize()
        if rank == args.sender:
            links.append(snapshot.result())
    for iteration in range(args.warmup + args.repeats):
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if rank == args.sender:
            work = dist.isend(buffer, dst=1-rank)
        else:
            work = dist.irecv(buffer, src=1-rank)
        work.wait()  # Establish NCCL-stream completion before the end event.
        end.record()
        end.synchronize()
        if iteration >= args.warmup:
            samples.append(start.elapsed_time(end))
    row = summarize(samples, size, f"nccl-rank-{rank}")
    row["link_under_load"] = links
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["h2d", "d2h", "bidirectional", "dual-d2h", "twohop", "nccl"], required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=SIZES_MIB)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sender", type=int, choices=[0, 1], default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode in ("dual-d2h", "twohop", "nccl") and len(args.gpus) != 2:
        parser.error("this mode requires exactly two GPU IDs")
    if args.mode in ("h2d", "d2h", "bidirectional") and len(args.gpus) != 1:
        parser.error("this mode requires exactly one GPU ID")
    if len(args.gpus) == 2 and args.gpus[0] == args.gpus[1]:
        parser.error("paired modes require distinct GPUs")
    if args.mode == "nccl":
        if os.environ.get("NCCL_DEBUG") != "INFO":
            parser.error("set NCCL_DEBUG=INFO to record the selected transport")
        torch.cuda.set_device(args.gpus[int(os.environ["LOCAL_RANK"])])
        dist.init_process_group("nccl")
    metadata = {
        "event": "environment", "command": sys.argv,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(), "gpus": args.gpus,
        "numa_maps": Path("/proc/self/numa_maps").read_text(),
        "numactl": subprocess.check_output(["numactl", "--show"], text=True),
        "time": time.time(),
    }
    rank = dist.get_rank() if args.mode == "nccl" else 0
    with open(args.output + (f".rank{rank}" if args.mode == "nccl" else ""), "w") as stream:
        stream.write(json.dumps(metadata) + "\n")
        for size_mib in args.sizes_mib:
            row = nccl(args, size_mib * MIB) if args.mode == "nccl" else copies(args, size_mib * MIB)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({key: value for key, value in row.items() if key != "samples_ms"}), flush=True)
    if args.mode == "nccl":
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
