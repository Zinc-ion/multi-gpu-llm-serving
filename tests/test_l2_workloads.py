import argparse
import hashlib
import json
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools import run_b0
from tools.prepare_workloads import materialize


def test_sharegpt_uses_reply_lengths_and_prunes_before_and_after_template(tmp_path, monkeypatch):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template="word {{ messages[0]['content'] }}",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model_name_or_path": str(tmp_path / "tokenizer"), "model_revision": "0" * 40,
        "tokenizer_revision": "0" * 40, "cache_dir": str(tmp_path), "max_model_length": 16,
    }))
    source = tmp_path / "source.json"
    source.write_text(json.dumps([
        {"conversations": [{"from": "human", "value": "word " * prompt},
                           {"from": "gpt", "value": "word " * reply}]}
        for prompt, reply in [(4, 4), (4, 7), (4, 3), (1025, 4), (4, 2045), (8, 8)]
    ] + [{"conversations": [{"from": "human", "value": "word " * 4}]}]))
    monkeypatch.setattr("tools.prepare_workloads.hf_hub_download", lambda *a, **kw: str(source))
    for index in range(2):
        destination = tmp_path / f"requests{index}.jsonl"
        materialize(argparse.Namespace(
            config=str(config), dataset="sharegpt", seed=7, num_requests=2,
            output_tokens=None, cache_dir=str(tmp_path), output=str(destination),
        ))
    assert (tmp_path / "requests0.jsonl").read_bytes() == (tmp_path / "requests1.jsonl").read_bytes()
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert sorted(row["output_tokens"] for row in rows) == [4, 7]
    manifest = json.loads((tmp_path / "requests1.jsonl.manifest.json").read_text())
    assert manifest["output_length"] == "first_reply_tokens"
    assert manifest["length_filter"]["after_chat_template"]


def bench_args(tmp_path, monkeypatch):
    dataset = tmp_path / "requests.jsonl"
    content = b'{"prompt": "hello", "output_tokens": 4}\n' * 1200
    dataset.write_bytes(content)
    (tmp_path / "requests.jsonl.manifest.json").write_text(json.dumps({
        "sha256": hashlib.sha256(content).hexdigest(), "requests": 1200,
    }))
    monkeypatch.setattr(sys, "argv", [
        "run_b0", "--mode", "l2", "--dataset", str(dataset), "--model", "tiny",
        "--tokenizer", str(tmp_path), "--output-dir", str(tmp_path / "results"),
    ])
    return dataset


def test_l2_runs_the_full_list_with_repeated_arrival_seeds(tmp_path, monkeypatch):
    bench_args(tmp_path, monkeypatch)
    commands = []
    # Capture process launch only; no serving process or GPU is started.
    run_process = run_b0.subprocess.run

    def launch(command, **kwargs):
        if command[:1] == ["git"]:
            return run_process(command, **kwargs)
        commands.append(command)

    monkeypatch.setattr(run_b0.subprocess, "run", launch)
    run_b0.main()
    assert len(commands) == 15  # three rates, five arrival realizations
    assert len({command[command.index("--result-filename") + 1] for command in commands}) == 15
    for index, command in enumerate(commands):
        assert command[command.index("--num-prompts") + 1] == "1200"
        assert command[command.index("--seed") + 1] == str(7 + index % 5)
        assert "--no-oversample" in command
        assert "--save-detailed" in command


@pytest.mark.parametrize("problem", ["hash", "oversample"])
def test_b0_rejects_changed_or_insufficient_request_lists(tmp_path, monkeypatch, problem):
    dataset = bench_args(tmp_path, monkeypatch)
    if problem == "hash":
        dataset.write_text("{}\n")
    else:
        sys.argv += ["--num-prompts", "1201"]
    with pytest.raises(SystemExit):
        run_b0.main()
