"""Capture custom logits, then compare in a separate Transformers process."""

import argparse
import atexit
import json
import subprocess
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from myvllm.config import load_config, resolve_dtype, seed_everything
from myvllm.sampling_parameters import SamplingParams


def metadata(config):
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
    }


def prepare(args):
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], revision=config["tokenizer_revision"],
        cache_dir=config["cache_dir"],
    )
    cases = []
    for index, prompt in enumerate(config["prompts"]):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        cases.append({"name": f"chat-{index}", "prompt": text, "input_ids": tokenizer.encode(text)})
    for length in args.lengths:
        # Fixed text and token IDs are materialized, so no dataset sampling occurs later.
        text = "Explain the following observations about inference systems. " * (length // 7 + 1)
        ids = tokenizer.encode(text)[:length]
        assert len(ids) == length
        cases.append({"name": f"tokens-{length}", "prompt": tokenizer.decode(ids), "input_ids": ids})
    Path(args.output).write_text(json.dumps({
        "model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"],
        "cases": cases,
    }, ensure_ascii=False, indent=2) + "\n")


def custom(args):
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.engine.sequence import Sequence

    config = load_config(args.config)
    engine = LLMEngine(config)
    try:
        missing = set(dict(engine.model_runner.model.named_parameters())) - engine.model_runner.loaded_params
        if missing:
            raise RuntimeError(f"Unloaded parameters: {sorted(missing)}")
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        checkpoint = Path(config["model_name_or_path"])
        if not checkpoint.is_dir():
            checkpoint = Path(snapshot_download(
                config["model_name_or_path"], revision=config["model_revision"],
                cache_dir=config["cache_dir"], local_files_only=True,
            ))
        checkpoint_names = set()
        for shard in checkpoint.glob("*.safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as weights:
                checkpoint_names.update(weights.keys())
        skipped = checkpoint_names - engine.model_runner.loaded_params
        if skipped:
            raise RuntimeError(f"Checkpoint tensors not loaded: {sorted(skipped)}")
        prompts = json.loads(Path(args.prompts).read_text())
        for field in ("model_revision", "tokenizer_revision"):
            if prompts[field] != config[field]:
                raise ValueError(f"Prompt manifest has a different {field}")
        rows = []
        for case in prompts["cases"]:
            logits = []
            # A forward hook observes the actual LM head used by step(), including decode.
            handle = engine.model_runner.model.lm_head.register_forward_hook(
                lambda module, inputs, output: logits.append(output.detach().float().cpu())
            )
            try:
                seq = Sequence(case["input_ids"], config["block_size"], SamplingParams(
                    temperature=0, max_tokens=args.max_tokens,
                    max_model_length=config["max_model_length"],
                ))
                engine.scheduler.add_sequence(seq)
                while not seq.is_finished:
                    engine.step()
            finally:
                handle.remove()
            rows.append({
                **case, "token_ids": seq.completion_token_ids,
                "logits": torch.cat(logits), "num_prompt_tokens": seq.num_prompt_tokens,
            })
        torch.save({"metadata": metadata(config), "cases": rows}, args.output)
    finally:
        atexit.unregister(engine.exit)
        engine.exit()


def reference(args):
    config = load_config(args.config)
    seed_everything(config["seed"])
    custom_data = torch.load(args.custom, map_location="cpu", weights_only=False)
    for field in ("model_revision", "tokenizer_revision", "dtype"):
        if config[field] != custom_data["metadata"]["config"][field]:
            raise ValueError(f"Reference and custom {field} must match")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], revision=config["model_revision"],
        cache_dir=config["cache_dir"], dtype=resolve_dtype(config["dtype"]),
        attn_implementation=args.attention,
    ).cuda().eval()
    results = []
    output = Path(args.output)
    report = {
        "metadata": metadata(config), "attention": args.attention,
        "sdpa_backend": args.sdpa_backend, "expand_kv": args.expand_kv,
        "custom_metadata": custom_data["metadata"], "results": results, "status": "running",
        "acceptance_threshold": "pending user decision; measurements are not a pass/fail gate",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    backend = {
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
    }[args.sdpa_backend]
    # Select HF's existing repeat_kv branch explicitly for FP32 efficient SDPA:
    # PyTorch 2.9.1's efficient backend does not accept enable_gqa=True.
    expansion = patch(
        "transformers.integrations.sdpa_attention.use_gqa_in_sdpa", return_value=False
    ) if args.expand_kv else nullcontext()
    with torch.inference_mode(), sdpa_kernel(backend), expansion:
        for case in custom_data["cases"]:
            input_ids = torch.tensor([case["input_ids"]], device="cuda")
            generation = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=min(args.max_tokens, config["max_model_length"] - input_ids.shape[1]),
                eos_token_id=config["eos"], pad_token_id=config["eos"][-1],
                return_dict_in_generate=True, output_scores=True,
            )
            reference_tokens = generation.sequences[0, input_ids.shape[1]:].tolist()
            generated = case["token_ids"]
            first_difference = next(
                (i for i, (a, b) in enumerate(zip(generated, reference_tokens)) if a != b),
                None,
            )
            if first_difference is None and len(generated) != len(reference_tokens):
                first_difference = min(len(generated), len(reference_tokens))
            margin = None
            if first_difference is not None and first_difference < len(generation.scores):
                top2 = generation.scores[first_difference][0].float().topk(2).values
                margin = (top2[0] - top2[1]).item()
            # One same-prefix forward; only the scored suffix is materialized as logits.
            teacher_ids = torch.tensor([case["input_ids"] + generated[:-1]], device="cuda")
            hf_logits = model(
                teacher_ids, attention_mask=torch.ones_like(teacher_ids),
                logits_to_keep=len(generated), use_cache=False,
            ).logits[0].float().cpu()
            actual = case["logits"]
            errors = (actual - hf_logits).abs()
            row = {
                "name": case["name"], "prompt_tokens": len(case["input_ids"]),
                "completion_tokens": len(generated),
                "greedy_equal": generated == reference_tokens,
                "first_difference": first_difference, "hf_top1_top2_margin": margin,
                "max_abs_error": errors.max().item(), "mean_abs_error": errors.mean().item(),
                "top1_agreement": (actual.argmax(-1) == hf_logits.argmax(-1)).float().mean().item(),
            }
            print(json.dumps(row), flush=True)
            raw_path = str(output) + "." + case["name"] + ".pt"
            torch.save({
                "name": case["name"], "reference_tokens": reference_tokens, "logits": hf_logits,
            }, raw_path)
            row["raw_reference"] = raw_path
            results.append(row)
            output.write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "complete"
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "custom", "reference"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--max-tokens", type=int, default=32)
        if name == "prepare":
            child.add_argument("--lengths", nargs="*", type=int, default=[1024, 4096, 16384])
        elif name == "custom":
            child.add_argument("--prompts", required=True)
        else:
            child.add_argument("--custom", required=True)
            child.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
            child.add_argument("--sdpa-backend", choices=["efficient", "flash", "math"], default="efficient")
            child.add_argument("--expand-kv", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "custom": custom, "reference": reference}[args.mode](args)


if __name__ == "__main__":
    main()
