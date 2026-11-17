"""Measure one small streaming batch with vLLM 0.15.0's HTTP request function."""

import argparse
import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import uuid

import aiohttp


async def run(args):
    manifest = json.loads(Path(args.dataset + ".manifest.json").read_text())
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()]
    if manifest["source"]["dataset"] != "random":
        raise ValueError("This small-batch runner requires the fixed-length synthetic workload")
    if not 0 < args.requests <= len(rows) or args.concurrency < 1 or args.warmups < 0:
        raise ValueError("Requests must fit the materialized list; concurrency > 0, warmups >= 0")
    module_path = Path(__file__).with_name("vllm_0_15_0_endpoint_request_func.py")
    spec = importlib.util.spec_from_file_location("locked_vllm_http", module_path)
    client = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = client
    spec.loader.exec_module(client)
    run_id = uuid.uuid4().hex[:12]
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async def request(index, row):
            async with semaphore:
                identifier = f"{run_id}-{index}"
                result = await client.async_request_openai_completions(
                    client.RequestFuncInput(
                        prompt=row["prompt"], api_url=args.url.rstrip("/") + "/v1/completions",
                        prompt_len=manifest["source"]["input_length"],
                        output_len=row["output_tokens"], model=args.model,
                        ignore_eos=True, extra_body={"temperature": 0}, request_id=identifier,
                    ), session,
                )
                if result.success and not result.output_tokens:
                    result.success = False
                    result.error = "Missing completion token count in streamed usage"
                record = asdict(result)
                record["request_id"] = identifier
                record["output_budget"] = row["output_tokens"]
                record["tpot_s"] = (
                    (result.latency - result.ttft) / (result.output_tokens - 1)
                    if result.success and result.output_tokens > 1 else None
                )
                return record

        for index in range(args.warmups):
            warmup = await request(f"warmup-{index}", rows[index % len(rows)])
            if not warmup["success"]:
                raise RuntimeError(f"Warmup failed: {warmup['error']}")
        started = time.perf_counter()
        results = await asyncio.gather(*(
            request(index, row) for index, row in enumerate(rows[:args.requests])
        ))
        wall_seconds = time.perf_counter() - started
    successes = [row for row in results if row["success"]]
    timed = [row for row in successes if row["tpot_s"] is not None]
    output_tokens = sum(row["output_tokens"] for row in successes)
    input_tokens = sum(row["prompt_len"] for row in successes)
    report = {
        "label": args.label, "command": sys.argv,
        "request_function": "unmodified vLLM 0.15.0 endpoint_request_func.py",
        "configuration": json.loads(Path(args.config).read_text()),
        "dataset_manifest": manifest, "gpu_ids": args.gpu_ids,
        "concurrency": args.concurrency, "warmups": args.warmups,
        "requests": len(results), "successful": len(successes),
        "failed": len(results) - len(successes), "wall_seconds": wall_seconds,
        "mean_ttft_ms": statistics.mean(row["ttft"] for row in successes) * 1000 if successes else None,
        "mean_tpot_ms": statistics.mean(row["tpot_s"] for row in timed) * 1000 if timed else None,
        "tpot_samples": len(timed), "successful_input_tokens": input_tokens,
        "successful_output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / wall_seconds,
        "input_plus_output_tokens_per_s": (input_tokens + output_tokens) / wall_seconds,
        "timing_scope": "warmups excluded; batch wall time includes completion/usage tail",
        "itl_scope": "client SSE-choice event gaps; network may coalesce multiple events",
        "memory_scope": "not collected here; use separate device sampling",
        "short_sample": True, "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"configuration", "dataset_manifest", "results"}}, indent=2))
    return 0 if len(successes) == len(results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="Qwen3-14B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu-ids", required=True, help="Physical GPUs used by this deployment")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", required=True)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
