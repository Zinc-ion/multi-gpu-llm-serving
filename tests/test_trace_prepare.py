import argparse
import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.prepare_trace import file_sha256, load_plan, prepare


REVISION = "a" * 40


@pytest.fixture
def inputs(tmp_path):
    directory = tmp_path / "tokenizer"
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(directory)
    provenance = tmp_path / "source-manifest.json"

    def source(name, rows):
        path = tmp_path / (name + ".jsonl")
        data = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        path.write_bytes(data)
        (tmp_path / (name + ".jsonl.manifest.json")).write_text(json.dumps({
            "sha256": hashlib.sha256(data).hexdigest(), "requests": len(rows),
            "model_revision": REVISION, "tokenizer_revision": REVISION,
            "source": {"dataset": name}, "max_model_length": 128,
        }))
        return path

    chat = source("chat", [{"prompt": "hello\u2028world\u0085hello", "output_tokens": 3},
                           {"prompt": "hello world", "output_tokens": 4}])
    long = source("long", [{"prompt": "world world", "output_tokens": 7}])
    files = {f"tokenizer/{path.name}": {"sha256": file_sha256(path)}
             for path in directory.iterdir() if path.is_file()}
    for path in (chat, long):
        source_manifest = tmp_path / (path.name + ".manifest.json")
        files[f"workloads/{path.name}"] = {"sha256": file_sha256(path)}
        files[f"workloads/{source_manifest.name}"] = {"sha256": file_sha256(source_manifest)}
    provenance.write_text(json.dumps({"scope": "fixture", "model_revision": REVISION, "files": files}))

    def args(**overrides):
        values = dict(mode="l2", kind="formal", chat=str(chat), long=None,
                      tokenizer=str(directory), tokenizer_provenance=str(provenance),
                      model_revision=REVISION, tokenizer_revision=REVISION,
                      seed=7, id_prefix="trace", rate=2.0, count=2, duration_s=None,
                      chat_probability=None, phases=None, max_requests=None,
                      allow_recycle=False, output=str(tmp_path / "plan.jsonl"))
        values.update(overrides)
        return argparse.Namespace(**values)

    return args, tokenizer, chat, long


def test_l2_uses_lf_rows_and_original_prompts_and_budgets(inputs):
    args, tokenizer, chat, _ = inputs
    options = args()
    manifest = prepare(options)
    loaded, rows = load_plan(options.output)
    assert loaded == manifest and len(rows) == 2
    original = [json.loads(line) for line in chat.read_bytes().split(b"\n") if line]
    assert [row["prompt"] for row in rows] == [row["prompt"] for row in original]
    assert [row["output_tokens"] for row in rows] == [3, 4]
    assert [row["prompt_tokens"] for row in rows] == [len(tokenizer.encode(row["prompt"])) for row in original]
    assert manifest["input_tokens"] == sum(row["prompt_tokens"] for row in rows)
    assert manifest["max_output_tokens"] == 7
    assert rows[0]["source_line"] == 1 and rows[1]["source_line"] == 2
    assert rows[0]["source_line_sha256"] == hashlib.sha256(chat.read_bytes().split(b"\n")[0]).hexdigest()
    assert rows[0]["source_sha256"] == file_sha256(chat)
    assert len({row["request_id"] for row in rows}) == 2
    assert all(row["phase"] == "l2" and row["source_cycle"] == 0 for row in rows)
    assert manifest["phases"][0]["realized_rate"] == 2 / manifest["window_s"]


def test_l3_phases_recycle_explicitly_and_deterministically(inputs, tmp_path):
    args, tokenizer, _, long = inputs
    phases = tmp_path / "phases.json"
    phases.write_text(json.dumps([
        {"name": "chat", "duration_s": .2, "rate": 30, "chat_probability": 1},
        {"name": "long", "duration_s": .3, "rate": 30, "chat_probability": 0},
    ]))
    common = dict(mode="l3", long=str(long), count=None, rate=None, phases=str(phases),
                  max_requests=100)
    with pytest.raises(ValueError, match="allow-recycle"):
        prepare(args(**common))
    first = args(**common, allow_recycle=True, output=str(tmp_path / "first.jsonl"))
    second = args(**common, allow_recycle=True, output=str(tmp_path / "second.jsonl"))
    meta = prepare(first)
    prepare(second)
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    _, rows = load_plan(first.output)
    assert meta["window_s"] == .5
    assert all(row["offset_s"] < .2 for row in rows if row["phase"] == "chat")
    assert all(.2 <= row["offset_s"] < .5 for row in rows if row["phase"] == "long")
    assert all(row["source"] == row["phase"] for row in rows)
    assert any(row["source_cycle"] > 0 for row in rows)
    assert meta["input_tokens"] == sum(len(tokenizer.encode(row["prompt"])) for row in rows)
    assert meta["max_output_tokens"] == sum(row["output_tokens"] for row in rows)
    assert all(phase["request_count"] > 0 for phase in meta["phases"])


def test_finite_l2_window_and_budget_are_not_shifted_to_last_arrival(inputs, tmp_path):
    args, _, _, _ = inputs
    options = args(count=None, duration_s=.5, rate=3, max_requests=10, allow_recycle=True,
                   output=str(tmp_path / "window.jsonl"))
    manifest = prepare(options)
    _, rows = load_plan(options.output)
    assert manifest["window_s"] == .5
    assert manifest["phases"][0]["rate"] == 3
    assert manifest["phases"][0]["realized_rate"] == len(rows) / .5
    assert all(row["offset_s"] < .5 for row in rows)


@pytest.mark.parametrize("alias", ["same_path", "same_bytes"])
def test_shared_source_across_roles_requires_and_records_recycle(inputs, tmp_path, alias):
    args, _, chat, long = inputs
    if alias == "same_bytes":
        long.write_bytes(chat.read_bytes())
        long_manifest = tmp_path / "long.jsonl.manifest.json"
        source_manifest = json.loads(long_manifest.read_text())
        source_manifest.update(sha256=file_sha256(long), requests=2)
        long_manifest.write_text(json.dumps(source_manifest))
        provenance_path = tmp_path / "source-manifest.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["files"]["workloads/long.jsonl"]["sha256"] = file_sha256(long)
        provenance["files"]["workloads/long.jsonl.manifest.json"]["sha256"] = file_sha256(long_manifest)
        provenance_path.write_text(json.dumps(provenance))
    options = dict(long=str(chat if alias == "same_path" else long), chat_probability=.2, count=4)
    with pytest.raises(ValueError, match="allow-recycle"):
        prepare(args(**options))
    reused = args(**options, allow_recycle=True, output=str(tmp_path / "reused.jsonl"))
    manifest = prepare(reused)
    _, rows = load_plan(reused.output)
    assert manifest["source_counts"] == {"chat": 2, "long": 2}
    assert [row["source_line"] for row in rows] == [1, 2, 1, 2]
    assert [row["source_cycle"] for row in rows] == [0, 0, 1, 1]
    assert [row["source_occurrence"] for row in rows] == [1, 2, 3, 4]


@pytest.mark.parametrize("broken", ["file", "count", "revision", "tokenizer"])
def test_invalid_source_or_tokenizer_provenance_is_rejected(inputs, broken, tmp_path):
    args, _, chat, _ = inputs
    if broken == "file":
        chat.write_bytes(chat.read_bytes() + b"{}\n")
    elif broken in ("count", "revision"):
        path = tmp_path / "chat.jsonl.manifest.json"
        manifest = json.loads(path.read_text())
        manifest["requests" if broken == "count" else "model_revision"] = 42 if broken == "count" else "b" * 40
        path.write_text(json.dumps(manifest))
    else:
        (tmp_path / "tokenizer" / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError):
        prepare(args())
    assert not (tmp_path / "plan.jsonl").exists()
