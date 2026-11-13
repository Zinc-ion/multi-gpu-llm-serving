"""Capture custom logits, then compare in a separate Transformers process."""

import argparse
import atexit
import json
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from myvllm.config import load_config, resolve_dtype, seed_everything
from myvllm.sampling_parameters import SamplingParams


def metadata(config):
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config": config,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
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
        text = "Explain the following observations about inference systems. " * length
        ids = tokenizer.encode(text)[:length]
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
        index_path = Path(config["model_name_or_path"]) / "model.safetensors.index.json"
        if index_path.exists():
            checkpoint_names = set(json.loads(index_path.read_text())["weight_map"])
            skipped = checkpoint_names - engine.model_runner.loaded_params
            if skipped:
                raise RuntimeError(f"Checkpoint tensors not loaded: {sorted(skipped)}")
        rows = []
        for case in json.loads(Path(args.prompts).read_text())["cases"]:
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
    if config["model_revision"] != custom_data["metadata"]["config"]["model_revision"]:
        raise ValueError("Reference and custom model revisions must match")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], revision=config["model_revision"],
        cache_dir=config["cache_dir"], dtype=resolve_dtype(config["dtype"]),
        attn_implementation=args.attention,
    ).cuda().eval()
    results = []
    raw = []
    with torch.inference_mode():
        for case in custom_data["cases"]:
            input_ids = torch.tensor([case["input_ids"]], device="cuda")
            generation = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False, max_new_tokens=args.max_tokens,
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
            results.append(row)
            raw.append({"name": case["name"], "reference_tokens": reference_tokens, "logits": hf_logits})
    output = Path(args.output)
    torch.save(raw, str(output) + ".pt")
    output.write_text(json.dumps({
        "metadata": metadata(config), "attention": args.attention,
        "custom_metadata": custom_data["metadata"], "results": results,
        "acceptance_threshold": "pending user decision; measurements are not a pass/fail gate",
    }, indent=2) + "\n")


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
    args = parser.parse_args()
    globals()[args.mode](args)


if __name__ == "__main__":
    main()
