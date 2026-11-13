"""Materialize pinned ShareGPT/LongBench/random requests for vLLM 0.15.0."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from myvllm.config import load_config


SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


def materialize(args):
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], revision=config["tokenizer_revision"],
        cache_dir=config["cache_dir"],
    )
    rng = random.Random(args.seed)
    source = {"dataset": args.dataset}
    if args.dataset == "sharegpt":
        path = hf_hub_download(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "ShareGPT_V3_unfiltered_cleaned_split.json",
            revision=SHAREGPT_REVISION, repo_type="dataset", cache_dir=args.cache_dir,
        )
        data = json.loads(Path(path).read_text())
        prompts = [
            row["conversations"][0]["value"] for row in data
            if row.get("conversations") and row["conversations"][0]["from"] == "human"
        ]
        source.update(revision=SHAREGPT_REVISION, file=path)
    elif args.dataset == "longbench":
        path = hf_hub_download(
            "THUDM/LongBench", "data.zip", revision=LONGBENCH_REVISION,
            repo_type="dataset", cache_dir=args.cache_dir,
        )
        with zipfile.ZipFile(path) as archive:
            rows = [json.loads(line) for line in archive.read(f"data/{args.subset}.jsonl").decode().splitlines()]
        prompts = [row["context"] + "\n\n" + row["input"] for row in rows]
        source.update(revision=LONGBENCH_REVISION, file=path, subset=args.subset)
    else:
        # Byte-level tokenization of this fixed motif gives an exact token-count check.
        ids = tokenizer.encode("Measurements of inference systems require reproducible inputs. " * (args.input_length // 6 + 1))
        if len(ids) < args.input_length:
            raise ValueError("Synthetic motif is shorter than requested input")
        prompt = tokenizer.decode(ids[:args.input_length], clean_up_tokenization_spaces=False)
        if len(tokenizer.encode(prompt)) != args.input_length:
            raise ValueError("Synthetic prompt failed exact token-length round trip")
        prompts = [prompt] * args.num_requests
        source["input_length"] = args.input_length
    rng.shuffle(prompts)
    selected = []
    rejected = 0
    for prompt in prompts:
        if args.dataset != "random":
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        length = len(tokenizer.encode(prompt))
        # Never truncate source examples: reserve the entire fixed completion budget.
        if length < 1 or length + args.output_tokens > config["max_model_length"]:
            rejected += 1
            continue
        selected.append({"prompt": prompt, "output_tokens": args.output_tokens})
        if len(selected) == args.num_requests:
            break
    if len(selected) != args.num_requests:
        raise ValueError(f"Only {len(selected)} valid prompts; requested {args.num_requests}")
    destination = Path(args.output)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    destination.write_bytes(content)
    manifest = {
        "source": source, "seed": args.seed, "requests": len(selected),
        "filtered_before_selection": rejected, "max_model_length": config["max_model_length"],
        "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
        "sha256": hashlib.sha256(content).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": subprocess.list2cmdline(sys.argv),
        "bench_parameters": {
            "dataset_name": "custom", "dataset_path": str(destination),
            "skip_chat_template": True, "disable_shuffle": True,
            "num_prompts": args.num_requests, "seed": args.seed,
            "temperature": 0, "ignore_eos": True,
            "percentile_metrics": "ttft,tpot,itl,e2el", "metric_percentiles": "50,90,99",
            "save_result": True, "save_detailed": True,
        },
    }
    Path(str(destination) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["sharegpt", "longbench", "random"], required=True)
    parser.add_argument("--subset", default="qasper")
    parser.add_argument("--input-length", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
