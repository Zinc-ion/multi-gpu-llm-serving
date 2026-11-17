import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from pd_helpers import create_tiny_qwen_checkpoint
from tools import compare_hf, compare_pd
from tools.compare_runs import compare_runs


ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = Path(os.environ.get("P4_ENGINE_SOURCE", str(ROOT))).resolve()
BASE = os.environ.get("P4_ENGINE_SHA") or subprocess.check_output(
    ["git", "-C", str(ENGINE_ROOT), "rev-parse", "HEAD"], text=True).strip()


def capture_args(tmp_path, max_tokens=3):
    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(max_num_batched_tokens=3, max_model_length=8, enable_chunked_prefill=True,
                  cache_dir=str(tmp_path / "model-cache"))
    configs = {}
    for role in ("reference", "prefill", "decode"):
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(config))
        configs[role + "_config"] = str(path)
        configs[role + "_gpu"] = ""
        configs[role + "_numa"] = None
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"model_revision": config["model_revision"],
        "tokenizer_revision": config["tokenizer_revision"], "cases": [
            {"name": "last-single-chunk", "prompt": "hello token2 token3 token4", "input_ids": [1, 2, 3, 4]},
            {"name": "short", "prompt": "hello", "input_ids": [1]},
        ]}))
    return argparse.Namespace(command="capture", engine_source=str(ENGINE_ROOT), expected_sha=BASE,
        prompts=str(prompts), output_dir=str(tmp_path / "output"), max_tokens=max_tokens,
        ignore_eos=True, transfer="bulk", raw_kv=True, max_transfer_bytes=1000000,
        pipeline_frame_bytes=None, startup_timeout=90.0, request_timeout=45.0,
        normal_exit_timeout=10.0, term_timeout=3.0, kill_timeout=3.0, **configs)


def test_real_cpu_model_independent_tcp_capture(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    compare_pd.capture(args)
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert all(role["exitcode"] == 0 for role in manifest["roles"].values())
    assert all(item["kv_handoff_equal"] is True for item in manifest["completed_cases"])
    assert all(info["source"] == str(ENGINE_ROOT) and info["commit"] == BASE for info in
               (json.loads((root / f"{role}.engine.json").read_text())["engine"]
                for role in ("reference", "prefill", "decode")))
    for index in range(2):
        reference = compare_pd.load_pt(root / f"reference-{index:05d}.pt")
        candidate = compare_pd.load_pt(root / f"candidate-{index:05d}.pt")
        result = compare_pd.analyze_data(reference, candidate, None)
        assert result["tokens_identical"] and result["all_finite"] and result["greedy_matches_logits"]
        assert compare_runs(reference, candidate)["tokens_identical"]
        torch.testing.assert_close(reference["cases"][0]["logits"], candidate["cases"][0]["logits"],
                                   rtol=1e-4, atol=1e-5)
        assert len(reference["cases"][0]["token_ids"]) == 3
        assert candidate["cases"][0]["input_ids"] == reference["cases"][0]["input_ids"]
        assert (root / f"partial/decode-{index:05d}-restored.pt").is_file()
        row = compare_pd.load_pt(root / f"partial/decode-{index:05d}-000003.pt")
        assert row["finished"] and row["kv"]["valid_tokens"] > len(reference["cases"][0]["input_ids"])
        assert row["blocks_after_postprocess"] == []


def prepare_qasper_cases(tmp_path, config, context, length, record_id):
    revision = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
    snapshot = Path(config["cache_dir"]) / "datasets--THUDM--LongBench" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    with zipfile.ZipFile(snapshot / "data.zip", "w") as archive:
        archive.writestr("data/qasper.jsonl", json.dumps({
            "_id": record_id, "length": length, "context": context, "input": "",
        }, ensure_ascii=False) + "\n")
    path = tmp_path / "prepare-config.json"
    path.write_text(json.dumps({**config, "prompts": []}))
    prompts = tmp_path / "prompts.json"
    compare_hf.prepare(argparse.Namespace(config=str(path), lengths=[length], output=str(prompts)))
    return json.loads(prompts.read_text())["cases"]


def test_actual_prepare_byte_prefix_capture_preserves_ids(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    config = json.loads(Path(args.reference_config).read_text())
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # Keep the genuine byte-level BPE small enough for the existing 64-token model.
    text = "Explain the following observations about inference systems. A é �\n\n"
    alphabet = sorted(set("".join(piece for piece, _ in byte_level.pre_tokenize_str(text))))
    assert len(alphabet) < 62
    vocab = {"[UNK]": 0, "[EOS]": 63}
    vocab.update({piece: i + 1 for i, piece in enumerate(alphabet)})
    vocab.update({f"<unused{i}>": i for i in range(len(alphabet) + 1, 63)})
    backend = Tokenizer(models.BPE(vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = byte_level
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.save_pretrained(config["model_name_or_path"])
    cases = prepare_qasper_cases(tmp_path, config, "A é " * 4, 3, "byte-prefix")
    lossy = cases[1]
    assert lossy["prompt"] == "A �"
    assert tokenizer.decode(lossy["input_ids"]) == lossy["prompt"]
    assert tokenizer.encode(lossy["prompt"]) != lossy["input_ids"]

    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    compare_pd.capture(args)
    root = Path(args.output_dir)
    info = compare_pd.compare_role_metadata(root)
    observation = info["reference"]["tokenizer"]
    assert all(role["tokenizer"] == observation for role in info.values())
    assert observation["model_input"] == "input_ids"
    assert observation["input_matches_prompt_encoding"] == {"0": True, "1": False}
    assert observation["prompt_matches_input_decoding"] == {"0": True, "1": True}
    assert observation["lossy_nonroundtrip"] == {"0": False, "1": True}
    assert observation["input_decodings"]["1"] == lossy["prompt"]
    assert observation["prompt_encodings"]["1"] == tokenizer.encode(lossy["prompt"])
    assert json.loads((root / "cases.json").read_text()) == cases
    for index, case in enumerate(cases):
        reference = compare_pd.load_pt(root / f"reference-{index:05d}.pt")
        candidate = compare_pd.load_pt(root / f"candidate-{index:05d}.pt")
        for run in (reference, candidate):
            stored = run["cases"][0]
            assert {field: stored[field] for field in case} == case
            assert stored["num_prompt_tokens"] == len(case["input_ids"]) == 3
            assert len(stored["token_ids"]) == args.max_tokens
        assert compare_pd.analyze_data(reference, candidate, None)["tokens_identical"]
        kv = compare_pd.load_pt(root / f"partial/prefill-{index:05d}-000001.pt")["kv"]
        assert kv["valid_tokens"] == len(case["input_ids"])
        assert json.loads((root / f"kv-{index:05d}.json").read_text())[
            "prefill_to_restored_decode"]["equal"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert all(role["exitcode"] == 0 and not role["remaining"] for role in manifest["roles"].values())


def test_real_qwen_prepared_6932_prefix_tokenizer(tmp_path):
    evidence = ROOT.parent / "evidence/m8-execution-20261116-01"
    tokenizer_path = Path(os.environ.get("P4_QWEN_TOKENIZER", str(
        evidence / "workload-tokenizer-inputs-01/tokenizer")))
    original_path = evidence / "p4-14b-bulk-same-01-audit/p4-14b-bulk-same-01-wrapper/prompts.json"
    if not tokenizer_path.is_dir() or not original_path.is_file():
        pytest.skip("Optional real Qwen tokenizer/original qasper evidence is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    original = next(case for case in json.loads(original_path.read_text())["cases"]
                    if case["name"] == "qasper-16384")
    compare_pd.tokenizer_identity(tokenizer, [original])
    config = {"model_name_or_path": str(tokenizer_path), "model_revision": "a" * 40,
              "tokenizer_revision": "b" * 40, "cache_dir": str(tmp_path / "cache")}
    cases = prepare_qasper_cases(tmp_path, config, original["prompt"], 6932, original["source"]["id"])
    case = cases[1]
    assert case["input_ids"] == original["input_ids"][:6932]
    assert case["input_ids"][-1] == 4891 and tokenizer.encode(case["prompt"])[-1] == 29333
    assert case["prompt"].endswith("either: �")
    identity = compare_pd.tokenizer_identity(tokenizer, cases)
    assert identity["model_input"] == "input_ids"
    assert identity["input_matches_prompt_encoding"]["1"] is False
    assert identity["prompt_matches_input_decoding"]["1"] is True
    assert identity["lossy_nonroundtrip"]["1"] is True
    assert identity["input_decodings"]["1"] == case["prompt"] == tokenizer.decode(case["input_ids"])


def capture_batch(tmp_path, temperature):
    args = capture_args(tmp_path)
    path = Path(args.reference_config)
    config = json.loads(path.read_text())
    config.update(enable_chunked_prefill=False, max_num_batched_tokens=8, max_num_sequences=2)
    path.write_text(json.dumps(config))
    cases = json.loads(Path(args.prompts).read_text())["cases"]
    spec = tmp_path / "batch.json"
    spec.write_text(json.dumps({"config": str(path), "output": str(tmp_path / "batch-output"),
                                "cases": cases, "source": str(ENGINE_ROOT), "sha": BASE,
                                "temperature": temperature}))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{ENGINE_ROOT / 'src'}:{ROOT / 'tests'}"}
    result = subprocess.run([sys.executable, "-P", str(ROOT / "tests/compare_pd_cpu_child.py"),
                             "--test-batch", str(spec)], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / "batch-output"


def test_multiple_head_rows_are_paired_before_callback(tmp_path):
    root = capture_batch(tmp_path, 0)
    callbacks = json.loads((root / "callbacks.json").read_text())
    assert [callback[0] for callback in callbacks] == ["case-00000", "case-00001"]
    assert callbacks[0][1] != callbacks[1][1]
    for index, (request_id, token, blocks) in enumerate(callbacks):
        row = compare_pd.load_pt(root / f"partial/reference-{index:05d}-000001.pt")
        assert row["request_id"] == request_id and row["token"] == token
        assert row["logits"].argmax().item() == token
        assert blocks == row["blocks_after_postprocess"] == []


def test_head_logits_copied_before_sampler_mutation(tmp_path):
    root = capture_batch(tmp_path, .5)
    head = compare_pd.load_pt(root / "head.pt")
    assert not torch.equal(head["before"], head["after"])
    torch.testing.assert_close(head["after"], head["before"] * 2, rtol=0, atol=0)
    for index in range(2):
        row = compare_pd.load_pt(root / f"partial/reference-{index:05d}-000001.pt")
        torch.testing.assert_close(row["logits"], head["before"][index], rtol=0, atol=0)


def test_provenance_uses_imported_source_from_different_cwd(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": f"{ENGINE_ROOT / 'src'}:{ROOT}"}
    code = ("import json,sys; from tools.compare_pd import engine_provenance; "
            "p=engine_provenance(sys.argv[1],sys.argv[2]); "
            "print(json.dumps({'source':p['source'],'commit':p['commit'],'module':p['module']}))")
    result = subprocess.run([sys.executable, "-P", "-c", code, str(ENGINE_ROOT), BASE],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual == {"source": str(ENGINE_ROOT), "commit": BASE,
                      "module": str(ENGINE_ROOT / "src/myvllm/__init__.py")}


def test_numa_arguments_and_actual_allowed_masks(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    args.reference_numa = args.prefill_numa = 0
    args.decode_numa = 1
    bindir = tmp_path / "bin"
    bindir.mkdir()
    wrapper = bindir / "numactl"
    wrapper.write_text('#!/bin/sh\nprintf "%s %s\\n" "$1" "$2" >> "$P4_NUMA_TEST_LOG"\nshift 2\nexec "$@"\n')
    wrapper.chmod(0o755)
    log = tmp_path / "numactl-args.log"
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setenv("P4_NUMA_TEST_LOG", str(log))
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    compare_pd.capture(args)
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    assert log.read_text().splitlines() == ["--cpunodebind=0 --membind=0",
                                            "--cpunodebind=1 --membind=1",
                                            "--cpunodebind=0 --membind=0"]
    for role, node in (("reference", 0), ("prefill", 0), ("decode", 1)):
        info = json.loads((root / f"{role}.engine.json").read_text())
        affinity = info["affinity"]
        assert affinity["requested_numa_node"] == node
        assert affinity["cpu_affinity"]
        assert set(affinity["allowed_masks"]) == {"Cpus_allowed", "Cpus_allowed_list",
                                                  "Mems_allowed", "Mems_allowed_list"}
        assert manifest["roles"][role]["command"][:3] == [
            "numactl", f"--cpunodebind={node}", f"--membind={node}"]


def test_missing_numactl_does_not_fall_back(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    args.reference_numa = 0
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    with pytest.raises(RuntimeError, match="Capture failed"):
        compare_pd.capture(args)
    manifest = json.loads((Path(args.output_dir) / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "numactl" in manifest["error"]
    assert manifest["roles"] == {}


def test_real_cpu_pipeline_or_legacy_refusal(tmp_path, monkeypatch):
    args = capture_args(tmp_path)
    args.transfer = "pipeline"
    args.pipeline_frame_bytes = 256
    path = Path(args.prefill_config)
    config = json.loads(path.read_text())
    config.update(enable_chunked_prefill=False, max_num_batched_tokens=8)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    if BASE.startswith("4b16c3"):
        with pytest.raises(RuntimeError, match="Capture failed"):
            compare_pd.capture(args)
        assert "does not support pipeline" in (Path(args.output_dir) / "decode.log").read_text()
    else:
        compare_pd.capture(args)
        root = Path(args.output_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert all(item["kv_handoff_equal"] is True for item in manifest["completed_cases"])
        for index in range(2):
            ref = compare_pd.load_pt(root / f"reference-{index:05d}.pt")
            cand = compare_pd.load_pt(root / f"candidate-{index:05d}.pt")
            assert cand["metadata"]["transfer"] == "pipeline"
            assert compare_pd.analyze_data(ref, cand, None)["tokens_identical"]


@pytest.mark.parametrize("eos_first", [False, True])
def test_terminal_first_token_needs_no_decode_handoff(tmp_path, monkeypatch, eos_first):
    args = capture_args(tmp_path, max_tokens=3 if eos_first else 1)
    args.ignore_eos = not eos_first
    if eos_first:
        for role in ("reference", "prefill", "decode"):
            path = Path(getattr(args, role + "_config"))
            config = json.loads(path.read_text())
            config["eos"] = [18]
            path.write_text(json.dumps(config))
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    compare_pd.capture(args)
    root = Path(args.output_dir)
    first = compare_pd.load_pt(root / "partial/prefill-00000-000001.pt")
    assert first["finished"] and first["blocks_after_postprocess"] == []
    assert first["kv"]["valid_tokens"] == 4
    if eos_first:
        assert first["token"] == 18
    assert not (root / "partial/decode-00000-restored.pt").exists()
    assert compare_pd.load_pt(root / "candidate-00000.pt")["metadata"]["kv_handoff_equal"] is None


@pytest.mark.parametrize("failure", ["scheduler", "kv_format", "prompt"])
def test_startup_failures_keep_original_error(tmp_path, monkeypatch, failure):
    args = capture_args(tmp_path)
    expected = {"scheduler": "max_num_sequences must be positive",
                "kv_format": "Requested FP8 KV cache was not allocated",
                "prompt": "Prompt text does not encode to input_ids"}[failure]
    if failure == "scheduler":
        path = Path(args.reference_config)
        config = json.loads(path.read_text())
        config["max_num_sequences"] = 0
        path.write_text(json.dumps(config))
    elif failure == "kv_format":
        for role in ("reference", "prefill", "decode"):
            path = Path(getattr(args, role + "_config"))
            config = json.loads(path.read_text())
            config["kv_cache_dtype"] = "float8" + "_e4m3fn"
            path.write_text(json.dumps(config))
    else:
        path = Path(args.prompts)
        prompts = json.loads(path.read_text())
        prompts["cases"][0]["input_ids"] = [2, 1, 3, 4]
        path.write_text(json.dumps(prompts))
    monkeypatch.setattr(compare_pd, "SCRIPT", ROOT / "tests/compare_pd_cpu_child.py")
    with pytest.raises(RuntimeError, match="Capture failed"):
        compare_pd.capture(args)
    root = Path(args.output_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert expected in (root / "reference.log").read_text()
    assert "AttributeError" not in (root / "reference.log").read_text()
    assert all(not any(item["state"] != "Z" for item in role["remaining"])
               for role in manifest["roles"].values())


def make_run(tokens, logits):
    config = dict(model_name_or_path="model", model_type="qwen", model_revision="a" * 40,
                  tokenizer_revision="b" * 40, dtype="float32", eos=[9], max_model_length=16)
    return {"metadata": {"config": config, "sampling": {"temperature": 0},
                         "cache": {"requested": "auto", "dtype": "torch.float32", "scale_dtype": None}}, "cases": [
        {"name": "case", "input_ids": [1, 2], "token_ids": tokens, "logits": torch.tensor(logits)}]}


def test_gate_scoring_prefix_and_nonfinite():
    reference = make_run([0, 1, 0], [[2., 1.], [0., 2.], [2., 1.]])
    candidate = make_run([0, 0, 1], [[2., 1.], [2., 0.], [0., 2.]])
    freeze = {"atol": .1, "rtol": 0.0}
    row = compare_pd.analyze_data(reference, candidate, freeze)
    assert row["cases"][0]["first_divergence"] == 1
    assert row["cases"][0]["same_prefix_logit_positions"] == 2
    assert row["cases"][0]["gate_failed_elements"] == 2
    assert not row["gate_passed"]
    identical = compare_pd.analyze_data(reference, reference, freeze)
    assert identical["gate_passed"] and identical["cases"][0]["same_prefix_logit_positions"] == 3
    nonfinite = make_run([0, 1, 0], [[2., 1.], [0., 2.], [float("nan"), 1.]])
    assert not compare_pd.analyze_data(reference, nonfinite, freeze)["gate_passed"]


@pytest.mark.parametrize("kind", ["pass", "fail", "nonfinite", "invalid-freeze", "capture-only"])
def test_gate_cli_preserves_report_and_exit_code(tmp_path, kind):
    reference = make_run([0, 1], [[2., 1.], [0., 2.]])
    candidate = make_run([0, 1], [[2.2 if kind == "fail" else 2., 1.], [0., 2.]])
    if kind == "nonfinite":
        candidate["cases"][0]["logits"][1, 0] = float("nan")
    baseline_path, candidate_path = tmp_path / "base.pt", tmp_path / "candidate.pt"
    compare_pd.save_pt(baseline_path, reference)
    compare_pd.save_pt(candidate_path, candidate)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"version": "v1", "approved": True, "approval_declaration": "test",
                                  "atol": .01, "rtol": float("inf") if kind == "invalid-freeze" else .01}))
    report_path = tmp_path / "report.json"
    command = [sys.executable, str(ROOT / "tools/compare_pd.py"), "analyze", "--baseline", str(baseline_path),
               "--candidate", str(candidate_path), "--output", str(report_path)]
    if kind != "capture-only":
        command.extend(["--gate-freeze", str(freeze)])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=20)
    report = json.loads(report_path.read_text())
    assert result.returncode == (0 if kind in ("pass", "capture-only") else 1)
    if kind == "pass":
        assert report["gate_passed"] is True and report["freeze"]["sha256"] == compare_pd.sha256(freeze)
    elif kind == "capture-only":
        assert report["status"] == "capture_only" and report["gate_passed"] is None
    elif kind == "invalid-freeze":
        assert report["status"] == "error" and "finite" in report["error"]
    else:
        assert report["gate_passed"] is False


def test_atomic_no_overwrite(tmp_path):
    path = tmp_path / "finished.pt"
    compare_pd.save_pt(path, {"token": 1})
    with pytest.raises(FileExistsError):
        compare_pd.save_pt(path, {"token": 2})
    assert compare_pd.load_pt(path) == {"token": 1}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["finished.pt"]


def test_bounded_cleanup_preserves_partial_and_stops_other_group(tmp_path):
    code = """
import os, signal, subprocess, sys, time
from tools.compare_pd import save_pt
if sys.argv[2] == 'stuck':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = ("import os,signal,sys,time; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "open(sys.argv[1] + '.tmp', 'w').write(str(os.getpid())); "
             "os.replace(sys.argv[1] + '.tmp', sys.argv[1]); time.sleep(30)")
    subprocess.Popen([sys.executable, '-c', child, sys.argv[1] + '.grandchild'])
save_pt(sys.argv[1], {'token': 7})
if sys.argv[2] == 'stuck':
    sys.stdin.readline()
    time.sleep(30)
else:
    sys.stdin.readline()
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""}
    roles = {}
    sentinel = None
    try:
        for name in ("good", "stuck"):
            target = tmp_path / f"{name}.pt"
            process = subprocess.Popen([sys.executable, "-c", code, str(target), name],
                                       cwd=ROOT, stdin=subprocess.PIPE, env=env, start_new_session=True)
            stat = Path(f"/proc/{process.pid}/stat").read_text()
            ticks = int(stat[stat.rfind(")") + 2:].split()[19])
            roles[name] = {"process": process, "pid": process.pid, "group": process.pid,
                           "start_ticks": ticks, "signals": [], "errors": [], "remaining": []}
            compare_pd.wait_for_file(target, roles[name], 20)
        grandchild = tmp_path / "stuck.pt.grandchild"
        compare_pd.wait_for_file(grandchild, roles["stuck"], 20)
        grandchild_pid = int(grandchild.read_text())
        assert grandchild_pid in {item["pid"] for item in compare_pd.process_group_members(
            roles["stuck"]["group"], roles["stuck"]["start_ticks"])}
        sentinel = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "sentinel.pt"), "good"],
                                    cwd=ROOT, stdin=subprocess.PIPE, env=env, start_new_session=True)
        deadline = time.monotonic() + 20
        while not (tmp_path / "sentinel.pt").exists() and time.monotonic() < deadline:
            assert sentinel.poll() is None
            time.sleep(.05)
        assert (tmp_path / "sentinel.pt").exists()
        limits = argparse.Namespace(normal_exit_timeout=.3, term_timeout=.3, kill_timeout=3)
        errors = compare_pd.stop_roles(roles, limits)
        assert roles["good"]["exitcode"] == 0
        assert roles["stuck"]["exitcode"] == -signal.SIGKILL
        assert roles["stuck"]["signals"] == ["SIGTERM", "SIGKILL"]
        assert errors and not roles["good"]["errors"]
        assert compare_pd.load_pt(tmp_path / "stuck.pt") == {"token": 7}
        assert all(not any(item["state"] != "Z" for item in role["remaining"]) for role in roles.values())
        assert sentinel.poll() is None
    finally:
        for role in roles.values():
            process = role["process"]
            if any(item["state"] != "Z" for item in
                   compare_pd.process_group_members(role["group"], role["start_ticks"])):
                os.killpg(role["group"], signal.SIGKILL)
            if process.poll() is None:
                process.wait(timeout=5)
        if sentinel is not None:
            if sentinel.poll() is None:
                sentinel.stdin.write(b"stop\n")
                sentinel.stdin.flush()
                sentinel.wait(timeout=5)
            sentinel.stdin.close()


@pytest.mark.parametrize("stage", ["reference", "pd"])
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["TERM", "INT"])
def test_controller_signal_stops_running_capture(tmp_path, sig, stage):
    args = capture_args(tmp_path, max_tokens=5)
    command = [sys.executable, "-P", str(ROOT / "tests/compare_pd_test_controller.py"), "capture",
               "--engine-source", args.engine_source, "--expected-sha", args.expected_sha,
               "--prompts", args.prompts, "--output-dir", args.output_dir, "--max-tokens", "5",
               "--ignore-eos", "--transfer", "bulk", "--raw-kv", "--max-transfer-bytes", "1000000",
               "--startup-timeout", "45", "--request-timeout", "45", "--normal-exit-timeout", "2",
               "--term-timeout", "2", "--kill-timeout", "2"]
    for role in ("reference", "prefill", "decode"):
        command.extend([f"--{role}-config", getattr(args, role + "_config"), f"--{role}-gpu", "0"])
    environment = os.environ.copy()
    environment["P4_TEST_STEP_DELAY"] = ".5"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONPATH"] = f"{ENGINE_ROOT / 'src'}:{ROOT / 'tests'}:{ROOT}"
    root = Path(args.output_dir)
    with (tmp_path / "controller.log").open("wb") as log:
        controller = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      env=environment, start_new_session=True)
    owned = {}
    try:
        deadline = time.monotonic() + 45
        partial = root / ("partial/reference-00000-000001.pt" if stage == "reference"
                          else "partial/decode-00000-000002.pt")
        while not partial.exists() and time.monotonic() < deadline and controller.poll() is None:
            time.sleep(.05)
        assert partial.exists(), (tmp_path / "controller.log").read_text()
        owned = json.loads((root / "manifest.json").read_text())["roles"]
        assert compare_pd.process_group_members(owned[stage if stage == "reference" else "decode"]["group"],
                                                owned[stage if stage == "reference" else "decode"]["start_ticks"])
        controller.send_signal(sig)
        time.sleep(.1)
        if controller.poll() is None:
            controller.send_signal(signal.SIGTERM)
            controller.send_signal(signal.SIGINT)
        controller.wait(timeout=15)
        manifest = json.loads((root / "manifest.json").read_text())
        assert controller.returncode != 0
        assert manifest["status"] == "failed"
        assert manifest["controller_signal"] == sig.name
        assert partial.exists()
        assert all(not any(item["state"] != "Z" for item in role["remaining"])
                   for role in manifest["roles"].values())
        for role in manifest["roles"].values():
            assert not any(item["state"] != "Z" for item in
                           compare_pd.process_group_members(role["group"], role["start_ticks"]))
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5)
        for role in owned.values():
            members = compare_pd.process_group_members(role["group"], role["start_ticks"])
            if any(item["state"] != "Z" for item in members):
                os.killpg(role["group"], signal.SIGKILL)
