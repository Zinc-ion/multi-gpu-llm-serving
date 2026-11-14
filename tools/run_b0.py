"""Run the unmodified, locked vLLM client against an already running instance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Materialized custom JSONL")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--tokenizer", required=True, help="Pinned local tokenizer snapshot")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["l1", "l2"], required=True)
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1])
    parser.add_argument("--request-rates", nargs="+", type=float, default=[1, 2, 4])
    parser.add_argument("--num-prompts", type=int, help="Default: all materialized requests; never oversample")
    parser.add_argument("--repetitions", type=int, help="Default: L1 once, L2 five arrival seeds")
    parser.add_argument("--num-warmups", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    content = Path(args.dataset).read_bytes()
    manifest = json.loads(Path(args.dataset + ".manifest.json").read_text())
    if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        parser.error("dataset SHA256 does not match its manifest; regenerate the workload")
    available = len(content.splitlines())
    num_prompts = args.num_prompts if args.num_prompts is not None else available
    if not 0 < num_prompts <= available:
        parser.error("num-prompts must fit the materialized list without oversampling")
    repetitions = args.repetitions if args.repetitions is not None else (5 if args.mode == "l2" else 1)
    if repetitions <= 0:
        parser.error("repetitions must be positive")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    values = args.concurrency if args.mode == "l1" else args.request_rates
    for value, repeat in ((value, repeat) for value in values for repeat in range(repetitions)):
        label = f"{Path(args.dataset).stem}-{args.mode}-{value}-rep{repeat}"
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "openai", "--endpoint", "/v1/completions",
            "--base-url", args.base_url, "--model", args.model,
            "--tokenizer", args.tokenizer,
            "--dataset-name", "custom", "--dataset-path", args.dataset,
            "--custom-output-len", "-1",
            "--skip-chat-template", "--disable-shuffle",
            "--num-prompts", str(num_prompts), "--no-oversample", "--temperature", "0", "--ignore-eos",
            "--num-warmups", str(args.num_warmups),
            "--seed", str(args.seed + repeat),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
            "--save-result", "--save-detailed", "--result-dir", str(directory),
            "--result-filename", label + ".json", "--request-id-prefix", label + "-",
            "--request-rate", "inf" if args.mode == "l1" else str(value),
        ]
        if args.mode == "l1":
            command += ["--max-concurrency", str(value)]
        metadata = {
            "command": command, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "client_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "client_numa_maps": Path("/proc/self/numa_maps").read_text(),
            "dataset_manifest": manifest, "repetition": repeat,
        }
        (directory / (label + ".command.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        with (directory / (label + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(directory / (label + ".json"), flush=True)


if __name__ == "__main__":
    main()
