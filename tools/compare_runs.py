"""Compare saved custom runs on CPU without scoring logits after prefix divergence."""

import argparse
import json
from pathlib import Path

import torch


def compare_runs(baseline, candidate):
    for field in ("model_name_or_path", "model_type", "model_revision",
                  "tokenizer_revision", "dtype", "eos"):
        if baseline["metadata"]["config"][field] != candidate["metadata"]["config"][field]:
            raise ValueError(f"Different {field}")
    if [case["name"] for case in baseline["cases"]] != [case["name"] for case in candidate["cases"]]:
        raise ValueError("Runs must contain the same ordered prompt cases")
    rows = []
    for before, after in zip(baseline["cases"], candidate["cases"]):
        if before["input_ids"] != after["input_ids"]:
            raise ValueError(f"Different input tokens for {before['name']}")
        old_tokens, new_tokens = before["token_ids"], after["token_ids"]
        for case in (before, after):
            if case["logits"].ndim != 2 or case["logits"].shape[0] != len(case["token_ids"]):
                raise ValueError(f"Expected one logits row per output token: {case['name']}")
            if not torch.isfinite(case["logits"]).all():
                raise ValueError(f"Non-finite logits: {case['name']}")
        if before["logits"].shape[1] != after["logits"].shape[1]:
            raise ValueError(f"Different vocabulary width for {before['name']}")
        common = min(len(old_tokens), len(new_tokens))
        mismatch = next((i for i in range(common) if old_tokens[i] != new_tokens[i]), None)
        if mismatch is None and len(old_tokens) != len(new_tokens):
            mismatch = common
        # The row predicting the first mismatch still has an identical prefix.
        scored = min(common, mismatch + 1) if mismatch is not None else common
        if not scored:
            raise ValueError(f"No comparable output positions: {before['name']}")
        old_logits, new_logits = before["logits"][:scored].float(), after["logits"][:scored].float()
        error = (old_logits - new_logits).abs()
        row = {
            "name": before["name"], "tokens_identical": old_tokens == new_tokens,
            "baseline_output_tokens": len(old_tokens), "candidate_output_tokens": len(new_tokens),
            "first_divergence": mismatch, "same_prefix_logit_positions": scored,
            "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
            "top1_agreement": (old_logits.argmax(-1) == new_logits.argmax(-1)).float().mean().item(),
        }
        if mismatch is not None and mismatch < common:
            top_two = old_logits[mismatch].topk(2).values
            row.update(
                baseline_token=old_tokens[mismatch], candidate_token=new_tokens[mismatch],
                baseline_top1_top2_gap=(top_two[0] - top_two[1]).item(),
            )
        rows.append(row)
    return {
        "baseline_metadata": baseline["metadata"], "candidate_metadata": candidate["metadata"],
        "tokens_identical": bool(rows) and all(row["tokens_identical"] for row in rows),
        "logits_gate": None, "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="compare_hf.py custom output")
    parser.add_argument("--candidate", required=True, help="compare_hf.py custom output")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=False)
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    report = compare_runs(baseline, candidate)
    report.update(baseline=str(Path(args.baseline).resolve()), candidate=str(Path(args.candidate).resolve()))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["tokens_identical"]:
        raise SystemExit("Greedy token mismatch; inspect the saved comparison")


if __name__ == "__main__":
    main()
