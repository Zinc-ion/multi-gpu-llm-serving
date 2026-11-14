import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.prepare_workloads import materialize


ROOT = Path(__file__).parents[1]


def test_materialized_workload_filters_after_template_and_repeats_seed(tmp_path, monkeypatch):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]",
        chat_template="hello {{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    config = {
        "model_name_or_path": str(tmp_path / "tokenizer"), "model_revision": "0" * 40,
        "tokenizer_revision": "0" * 40, "cache_dir": str(tmp_path), "max_model_length": 8,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"conversations": [{"from": "human", "value": "world " * 8}, {"from": "gpt", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": "world " * 4}, {"from": "gpt", "value": "hello"}]},
        {"conversations": [{"from": "human", "value": "hello world " * 2}, {"from": "gpt", "value": "hello"}]},
    ]))
    # Substitute the network download only, preserving conversion and tokenization.
    monkeypatch.setattr("tools.prepare_workloads.hf_hub_download", lambda *a, **kw: str(source))
    for index in range(2):
        materialize(argparse.Namespace(
            config=str(config_path), dataset="sharegpt", seed=7, num_requests=2,
            output_tokens=2, cache_dir=str(tmp_path), output=str(tmp_path / f"requests{index}.jsonl"),
        ))
    first, second = (tmp_path / "requests0.jsonl").read_bytes(), (tmp_path / "requests1.jsonl").read_bytes()
    assert first == second
    manifest = json.loads((tmp_path / "requests0.jsonl.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(first).hexdigest()
    assert manifest["filtered_before_selection"] == 1
    for line in first.decode().splitlines():
        row = json.loads(line)
        assert row["prompt"].startswith("hello ")
        assert len(tokenizer.encode(row["prompt"])) + row["output_tokens"] <= 8


def test_capacity_report_rounds_each_request_to_whole_blocks(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(json.dumps({
        "event": "capacity", "block_count": 100,
        "block_bytes": 256 * 160 * 1024,
    }) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"block_size": 256, "max_model_length": 17408}))
    output = tmp_path / "capacity.json"
    subprocess.run([
        sys.executable, str(ROOT / "tools/capacity_report.py"),
        "--log", str(log), "--config", str(config), "--output", str(output),
    ], check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["bytes_per_token"] == 160 * 1024
    assert report["requests"][0]["blocks_per_request"] == 5
    assert report["requests"][0]["kv_capacity_concurrency_upper_bound"] == 20


def test_detailed_results_count_token_chunks_and_recompute_goodput(tmp_path):
    result = tmp_path / "bench.json"
    result.write_text(json.dumps({
        "ttfts": [0.01, 0.1], "itls": [[0.02, 0.02, 0.02], [0.03]],
        "output_lens": [4, 2], "errors": ["", ""], "start_times": [200, 201], "duration": 2,
    }))
    output = tmp_path / "analysis.json"
    comparison = tmp_path / "second.json"
    comparison.write_text(json.dumps({"start_times": [100, 101.01]}))
    subprocess.run([
        sys.executable, str(ROOT / "tools/analyze_bench.py"),
        "--result", str(result), "--ttft-slo-ms", "15", "--tpot-slo-ms", "25",
        "--output", str(output), "--compare-arrivals", str(comparison),
    ], check=True, capture_output=True)
    report = json.loads(output.read_text())
    assert report["all_token_chunks_match"]
    assert report["slo_pass_fraction"] == 0.5
    assert report["request_goodput"] == 0.5
    assert not report["arrival_comparison"]["same_actual_relative_times"]
