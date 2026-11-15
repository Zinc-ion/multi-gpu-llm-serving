from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools.compare_runs import compare_runs


def run(tokens=(1, 2, 3)):
    return {
        "metadata": {"commit": "fixed-commit", "config": {
            "model_name_or_path": "Qwen/Qwen3-14B", "model_type": "Qwen3-14B",
            "model_revision": "a" * 40, "tokenizer_revision": "a" * 40,
            "dtype": "bfloat16", "eos": [10, 11],
        }},
        "cases": [{
            "name": "long-prompt", "input_ids": [4, 5, 6], "token_ids": list(tokens),
            "logits": torch.arange(len(tokens) * 4, dtype=torch.float32).reshape(len(tokens), 4),
        }],
    }


def test_comparison_reports_errors_without_inventing_an_acceptance_gate():
    baseline = run()
    candidate = deepcopy(baseline)
    candidate["cases"][0]["logits"][0, 0] += .25
    result = compare_runs(baseline, candidate)
    assert result["tokens_identical"]
    assert result["logits_gate"] is None
    assert result["cases"][0]["max_abs_error"] == .25
    assert result["cases"][0]["same_prefix_logit_positions"] == 3


def test_comparison_stops_scoring_after_the_first_divergence():
    candidate = run((1, 0, 3))
    candidate["cases"][0]["logits"][2] += 10000
    result = compare_runs(run(), candidate)
    row = result["cases"][0]
    assert not result["tokens_identical"]
    assert row["first_divergence"] == 1
    assert row["same_prefix_logit_positions"] == 2
    assert row["max_abs_error"] == 0
    assert row["baseline_top1_top2_gap"] == 1


def test_different_stop_length_is_a_token_difference():
    result = compare_runs(run(), run((1, 2)))
    assert not result["tokens_identical"]
    assert result["cases"][0]["first_divergence"] == 2
    assert result["cases"][0]["same_prefix_logit_positions"] == 2


@pytest.mark.parametrize("change,match", [
    ("revision", "model_revision"), ("prompt", "input tokens"),
    ("rows", "one logits row"), ("nan", "Non-finite"),
])
def test_mismatched_or_invalid_captures_are_rejected(change, match):
    candidate = run()
    if change == "revision":
        candidate["metadata"]["config"]["model_revision"] = "b" * 40
    elif change == "prompt":
        candidate["cases"][0]["input_ids"] = [9]
    elif change == "rows":
        candidate["cases"][0]["logits"] = torch.zeros(2, 4)
    else:
        candidate["cases"][0]["logits"][0, 0] = float("nan")
    with pytest.raises(ValueError, match=match):
        compare_runs(run(), candidate)


def test_cli_writes_the_report_before_reporting_a_token_mismatch(tmp_path):
    before, after, output = (tmp_path / name for name in ("baseline.pt", "candidate.pt", "comparison.json"))
    torch.save(run(), before)
    torch.save(run((1, 0, 3)), after)
    result = subprocess.run([
        sys.executable, str(Path(__file__).parents[1] / "tools/compare_runs.py"),
        "--baseline", str(before), "--candidate", str(after), "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert not json.loads(output.read_text())["tokens_identical"]
