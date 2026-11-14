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
    if args.output_tokens is not None and args.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if args.num_requests is not None and args.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    num_requests = args.num_requests
    if num_requests is None and args.dataset != "longbench":
        num_requests = 1000 if args.dataset == "sharegpt" else 100
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
        examples = [
            (row["conversations"][0]["value"], row["conversations"][1]["value"])
            for row in data if len(row.get("conversations", [])) >= 2
            and row["conversations"][0]["from"] == "human"
            and row["conversations"][1]["from"] == "gpt"
        ]
        source.update(revision=SHAREGPT_REVISION, file=path)
    elif args.dataset == "longbench":
        path = hf_hub_download(
            "THUDM/LongBench", "data.zip", revision=LONGBENCH_REVISION,
            repo_type="dataset", cache_dir=args.cache_dir,
        )
        with zipfile.ZipFile(path) as archive:
            rows = [json.loads(line) for line in archive.read(f"data/{args.subset}.jsonl").decode().splitlines()]
        examples = [(row["context"] + "\n\n" + row["input"], None) for row in rows]
        source.update(revision=LONGBENCH_REVISION, file=path, subset=args.subset)
    else:
        # Byte-level tokenization of this fixed motif gives an exact token-count check.
        ids = tokenizer.encode("Measurements of inference systems require reproducible inputs. " * (args.input_length // 6 + 1))
        if len(ids) < args.input_length:
            raise ValueError("Synthetic motif is shorter than requested input")
        prompt = tokenizer.decode(ids[:args.input_length], clean_up_tokenization_spaces=False)
        if len(tokenizer.encode(prompt)) != args.input_length:
            raise ValueError("Synthetic prompt failed exact token-length round trip")
        examples = [(prompt, None)] * num_requests
        source["input_length"] = args.input_length
    rng.shuffle(examples)
    selected = []
    rejected = 0
    for prompt, reply in examples:
        output_tokens = args.output_tokens
        if args.dataset == "sharegpt":
            if output_tokens is None:
                output_tokens = len(tokenizer.encode(reply))
            # Match vLLM 0.15.0 ShareGPT's raw-text length pruning before template.
            raw_length = len(tokenizer.encode(prompt))
            if (raw_length < 4 or raw_length > 1024 or raw_length + output_tokens > 2048
                    or (args.output_tokens is None and output_tokens < 4)):
                rejected += 1
                continue
        elif output_tokens is None:
            output_tokens = 128
        if args.dataset != "random":
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        length = len(tokenizer.encode(prompt))
        # Never truncate source examples: reserve each request's completion budget.
        if length < 1 or length + output_tokens > config["max_model_length"]:
            rejected += 1
            continue
        selected.append({"prompt": prompt, "output_tokens": output_tokens})
        if len(selected) == num_requests:
            break
    if not selected or (num_requests is not None and len(selected) != num_requests):
        raise ValueError(f"Only {len(selected)} valid prompts; requested {num_requests}")
    destination = Path(args.output)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
    destination.write_bytes(content)
    manifest = {
        "source": source, "seed": args.seed, "requests": len(selected),
        "filtered_before_selection": rejected, "max_model_length": config["max_model_length"],
        "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
        "output_length": "first_reply_tokens" if args.dataset == "sharegpt" and args.output_tokens is None else "fixed",
        "length_filter": {
            "after_chat_template": True,
            "sharegpt_raw_limits": {"min_tokens": 4, "max_prompt_tokens": 1024, "max_total_tokens": 2048}
            if args.dataset == "sharegpt" else None,
        },
        "sha256": hashlib.sha256(content).hexdigest(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": subprocess.list2cmdline(sys.argv),
        "bench_parameters": {
            "dataset_name": "custom", "dataset_path": str(destination),
            "custom_output_len": -1,
            "skip_chat_template": True, "disable_shuffle": True,
            "num_prompts": len(selected), "seed": args.seed,
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
    parser.add_argument("--output-tokens", type=int, help="Override reply lengths; default: ShareGPT reply tokens, otherwise 128")
    parser.add_argument("--num-requests", type=int, help="Default: ShareGPT 1000, LongBench all valid examples, random 100")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
