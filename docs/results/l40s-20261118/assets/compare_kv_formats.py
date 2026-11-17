"""Report lossy KV-format differences without treating different prefixes as equal."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    for field in ("model_name_or_path", "model_revision", "tokenizer_revision", "dtype"):
        assert baseline["metadata"]["config"][field] == candidate["metadata"]["config"][field]
    rows = []
    assert len(baseline["cases"]) == len(candidate["cases"])
    for before, after in zip(baseline["cases"], candidate["cases"]):
        assert before["name"] == after["name"] and before["input_ids"] == after["input_ids"]
        old, new = before["token_ids"], after["token_ids"]
        common = min(len(old), len(new))
        difference = next((i for i in range(common) if old[i] != new[i]), None)
        if difference is None and len(old) != len(new):
            difference = common
        scored = min(common, difference+1) if difference is not None else common
        error = (before["logits"][:scored].float() - after["logits"][:scored].float()).abs()
        top = before["logits"][difference].topk(2).values if difference is not None and difference < common else None
        rows.append({
            "name": before["name"], "prompt_tokens": len(before["input_ids"]),
            "baseline_tokens": old, "candidate_tokens": new, "tokens_identical": old == new,
            "finite": bool(torch.isfinite(after["logits"]).all()),
            "same_prefix_positions": scored, "first_divergence": difference,
            "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
            "baseline_top1_top2_gap": (top[0]-top[1]).item() if top is not None else None,
        })
    report = {
        "comparison": "BF16 KV -> lossy runtime FP8 E4M3FN KV; weights/activations BF16; FP8 prefill FP32",
        "baseline_metadata": baseline["metadata"], "candidate_metadata": candidate["metadata"],
        "cases": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
