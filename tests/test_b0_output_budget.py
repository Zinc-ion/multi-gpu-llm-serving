import argparse
import hashlib
import json
import sys

import pytest
from vllm.benchmarks.datasets import add_dataset_parser, get_samples

from tools import run_b0
from test_server import tokenizer


@pytest.mark.parametrize("mode", ["l1", "l2"])
def test_b0_cli_passes_materialized_lengths_to_the_locked_dataset_parser(tmp_path, monkeypatch, tokenizer, mode):
    requests = [{"prompt": "hello", "output_tokens": 4}, {"prompt": "world", "output_tokens": 7}]
    content = "".join(json.dumps(row) + "\n" for row in requests).encode()
    dataset = tmp_path / "requests.jsonl"
    dataset.write_bytes(content)
    (tmp_path / "requests.jsonl.manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(content).hexdigest(), "requests": 2,
    }))
    monkeypatch.setattr(sys, "argv", [
        "run_b0", "--mode", mode, "--dataset", str(dataset), "--model", "tiny",
        "--tokenizer", str(tmp_path), "--output-dir", str(tmp_path / "results"),
        "--repetitions", "1", "--request-rates", "1",
    ])
    commands = []
    run_process = run_b0.subprocess.run

    def launch(command, **kwargs):
        if command[:1] == ["git"]:
            return run_process(command, **kwargs)
        commands.append(command)

    monkeypatch.setattr(run_b0.subprocess, "run", launch)
    run_b0.main()
    assert len(commands) == 1
    parser = argparse.ArgumentParser()
    add_dataset_parser(parser)
    args, _ = parser.parse_known_args(commands[0][5:])
    samples = get_samples(args, tokenizer)
    assert [sample.expected_output_len for sample in samples] == [4, 7]
