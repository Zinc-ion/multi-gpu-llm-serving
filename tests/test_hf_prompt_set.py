import argparse
import json
import zipfile

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.compare_hf import prepare
from tools.prepare_workloads import LONGBENCH_REVISION


def test_prompt_manifest_includes_pinned_natural_long_context(tmp_path, monkeypatch):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template="{{messages[0]['content']}}",
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model_name_or_path": str(tmp_path / "tokenizer"), "model_revision": "0" * 40,
        "tokenizer_revision": "0" * 40, "cache_dir": str(tmp_path), "prompts": ["hello"],
    }))
    archive_path = tmp_path / "data.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/qasper.jsonl", json.dumps({
            "_id": "paper-id", "length": 100, "input": "Explain the observations",
            "context": "A natural research article discusses diverse observations and measurements. " * 10,
        }) + "\n")

    def download(repo, filename, **kwargs):
        assert repo == "THUDM/LongBench"
        assert filename == "data.zip"
        assert kwargs["revision"] == LONGBENCH_REVISION
        return str(archive_path)

    monkeypatch.setattr("tools.compare_hf.hf_hub_download", download)
    output = tmp_path / "prompts.json"
    prepare(argparse.Namespace(config=str(config), lengths=[16, 32], output=str(output)))
    first = output.read_bytes()
    prepare(argparse.Namespace(config=str(config), lengths=[16, 32], output=str(output)))
    assert output.read_bytes() == first
    cases = json.loads(first)["cases"]
    assert [case["name"] for case in cases] == ["chat-0", "tokens-16", "tokens-32", "qasper-32"]
    natural = cases[-1]
    assert len(natural["input_ids"]) == 32
    assert natural["source"]["id"] == "paper-id"
    assert natural["source"]["revision"] == LONGBENCH_REVISION
