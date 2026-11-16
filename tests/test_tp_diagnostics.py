import json
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from tools.diagnose_tp import (
    Capture, capture_issues, compare_collective_ranks, compare_events, compare_gather_ranks, historical_hashes,
    live_group_members, load_events, main, make_environment, original_gate, run_lane, scheduler_snapshot, verify_weights,
)


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tests/tp_diagnostic_helpers.py"
CONFIG = {"vocab_size": 13, "hidden_size": 8, "num_heads": 2, "num_kv_heads": 2,
          "head_dim": 4, "intermediate_size": 12, "num_layers": 1, "dtype": "float32"}


def run_cpu_model(world, checkpoint, output, log, config=None):
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with log.open("w") as stream:
        command = [sys.executable, str(HELPER), "--world", str(world), "--checkpoint", str(checkpoint),
                   "--output", str(output)]
        if config is not None:
            command.extend(["--config", str(config)])
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env=env,
        )
        try:
            code = process.wait(timeout=90)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    assert code == 0, log.read_text()


def test_real_qwen_gloo_capture_alignment_and_checkpoint(tmp_path):
    # This independent small fixture is not the preserved M4 checkpoint.
    torch.manual_seed(7)
    checkpoint = tmp_path / "independent-checkpoint"
    config = Qwen3Config(vocab_size=13, hidden_size=8, head_dim=4,
                         num_attention_heads=2, num_key_value_heads=2,
                         intermediate_size=12, num_hidden_layers=1,
                         max_position_embeddings=16, rms_norm_eps=1e-5,
                         tie_word_embeddings=False, attention_bias=False,
                         attention_dropout=0.0, rope_theta=10000)
    Qwen3ForCausalLM(config).eval().save_pretrained(checkpoint, safe_serialization=True)
    one, two = tmp_path / "tp1", tmp_path / "tp2"
    run_cpu_model(1, checkpoint, one, tmp_path / "tp1.log")
    run_cpu_model(2, checkpoint, two, tmp_path / "tp2.log")

    for world, directory in ((1, one), (2, two)):
        for rank in range(world):
            report = verify_weights(directory / f"rank-{rank}", checkpoint, CONFIG, rank, world)
            assert report["all_equal"], report
            assert all(row["dtype_matches_config"] for row in report["parameters"].values())
    one_logits = torch.load(one / "logits-0.pt", weights_only=True)
    two_logits = torch.load(two / "logits-0.pt", weights_only=True)
    torch.testing.assert_close(two_logits, one_logits, rtol=1e-5, atol=1e-6)

    prefixes = [[1, 2, 3, 4]] * 3
    report = compare_events(one / "rank-0", [two / "rank-0", two / "rank-1"], prefixes, prefixes, CONFIG)
    rows = report["timeline"]
    for layout in ("replicated", "qkv_packed", "gate_up_packed", "head_shard", "feature_shard", "vocab_shard"):
        matches = [row for row in rows if row.get("layout") == layout]
        assert matches and all(row["comparable"] for row in matches), (layout, matches)
    assert any(row.get("excluded_vocab_padding") == 1 for row in rows)
    mismatched = compare_events(one / "rank-0", [two / "rank-0", two / "rank-1"],
                                prefixes, [[1, 2, 3, 5]] * 3, CONFIG)
    assert all(not row["comparable"] and row["reason"] == "different or missing prefix"
               for row in mismatched["timeline"])

    rank_events = [load_events(two / f"rank-{rank}") for rank in range(2)]
    pre, post = [], []
    for events in rank_events:
        pre.append(next(tensor for entry, tensor in events if entry["kind"] == "collective"
                        and entry["module"].endswith(".self_attn.o_proj")
                        and entry["phase"] == "all_reduce_pre"))
        post.append(next(tensor for entry, tensor in events if entry["kind"] == "collective"
                         and entry["module"].endswith(".self_attn.o_proj")
                         and entry["phase"] == "all_reduce_post"))
    assert not torch.equal(pre[0], pre[1])
    torch.testing.assert_close(pre[0] + pre[1], post[0], rtol=1e-6, atol=1e-7)
    assert torch.equal(post[0], post[1])
    agreement = compare_collective_ranks([two / "rank-0", two / "rank-1"])
    assert agreement and all(row["equal"] for row in agreement)
    gathered = compare_gather_ranks([two / "rank-0", two / "rank-1"])
    assert len(gathered) == 6 and all(row["equal"] for row in gathered)
    report["all_reduce_post_rank_agreement"] = agreement
    report["gather_received_vs_sent"] = gathered
    finished = {"scheduler_finished": True, "used_blocks": []}
    assert capture_issues([one, two], CONFIG, [finished, finished], report) == []
    assert any(entry["phase"] == "gather_pre" for entry, _ in rank_events[1])
    assert any(entry["phase"] == "gather_post" for entry, _ in rank_events[0])

    altered = tmp_path / "different-checkpoint"
    altered.mkdir()
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    tensors["model.layers.0.self_attn.q_proj.weight"][0, 0] += 1
    save_file(tensors, altered / "model.safetensors")
    wrong = verify_weights(two / "rank-0", altered, CONFIG, 0, 2)
    assert not wrong["all_equal"]
    assert not wrong["parameters"]["model.layers.0.self_attn.qkv_projection.weight"]["equal"]


def test_capture_budget_leaves_readable_partial(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    with pytest.raises(RuntimeError, match="tensor budget exceeded"):
        with Capture(model, tmp_path / "rank", 0, 1, max_bytes=16):
            model(torch.ones(1, 2))
    events = load_events(tmp_path / "rank")
    assert len(events) == 1
    assert events[0][0]["kind"] == "weight"
    assert torch.equal(events[0][1], model[0].weight.detach())


def test_engine_environment_keeps_caches_under_output_and_offline(tmp_path):
    env = make_environment(tmp_path, ROOT / "src")
    for name in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "MPLCONFIGDIR",
                 "HF_HOME", "HF_HUB_CACHE", "PYTHONPYCACHEPREFIX", "TORCHINDUCTOR_CACHE_DIR",
                 "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        assert Path(env[name]).is_relative_to(tmp_path)
        assert Path(env[name]).is_dir()
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == env["DO_NOT_TRACK"] == "1"
    assert env["PYTHONPATH"] == str(ROOT / "src")


def test_scheduler_snapshot_follows_real_sequence_lifecycle():
    from myvllm.engine.scheduler import Scheduler
    from myvllm.engine.sequence import Sequence
    from myvllm.sampling_parameters import SamplingParams

    scheduler = Scheduler(max_num_sequences=1, max_num_batched_tokens=16, max_cached_blocks=16,
                          block_size=2, eos=[66], max_model_length=16)
    seq = Sequence([1, 9, 4, 6, 12], block_size=2,
                   sampling_params=SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    scheduler.add_sequence(seq)
    assert scheduler_snapshot(scheduler)["waiting"][0]["tokens"] == [1, 9, 4, 6, 12]
    for token in (32, 33, 41):
        scheduled, _ = scheduler.schedule()
        assert scheduled == [seq]
        scheduler.postprocess(scheduled, [token])
    final = scheduler_snapshot(scheduler)
    assert final == {"waiting": [], "running": [], "used_blocks": []}
    assert seq.token_ids[-3:] == [32, 33, 41]


HANG_SCRIPT = r'''
import json, pathlib, signal, subprocess, sys, time
root, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(120)"], stdout=subprocess.PIPE)
child.stdout.readline()
(root / "partial.txt").write_text(str(child.pid))
if mode != "startup":
    (root / "rank-0").mkdir()
    (root / "rank-0/ready.json").write_text("{}")
if mode == "exit":
    (root / "request-complete.json").write_text("{}")
while True:
    time.sleep(1)
'''


@pytest.mark.parametrize("stage", ["startup", "request", "exit"])
def test_process_tree_deadlines_keep_partial_and_kill_owned_group(tmp_path, stage):
    output = tmp_path / stage
    command = [sys.executable, "-c", HANG_SCRIPT, str(output), stage]
    status = run_lane(command, output, 1, os.environ.copy(), 1.5, 1.0, 1.0, 0.2, 2.0)
    assert status["stage"] == stage
    assert status["failure"] == f"{stage} deadline exceeded"
    assert status["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
    assert status["remaining_live_members"] == []
    assert not live_group_members(status["process_group"])
    assert int((output / "partial.txt").read_text()) > 0
    assert status["elapsed_seconds"] < 8


def test_process_normal_exit_needs_no_signal(tmp_path):
    output = tmp_path / "normal"
    script = r'''
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / "rank-0").mkdir()
(root / "rank-0/ready.json").write_text("{}")
(root / "request-complete.json").write_text("{}")
(root / "engine-exit.json").write_text(json.dumps({"worker_exitcodes": [], "process_group_initialized": False}))
'''
    status = run_lane([sys.executable, "-c", script, str(output)], output, 1,
                      os.environ.copy(), 2, 2, 2, 0.2, 1)
    assert status["failure"] is None
    assert status["exitcode"] == 0
    assert status["cleanup_signals"] == []
    assert status["remaining_live_members"] == []


def test_forced_exit_releases_only_the_recorded_shared_memory(tmp_path):
    output = tmp_path / "shared-memory"
    script = r'''
import pathlib, signal, sys, time
from multiprocessing.shared_memory import SharedMemory
from tools.diagnose_tp import tracked_shared_memory
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
owned = tracked_shared_memory(root, SharedMemory)(create=True, size=2**20)
for rank in (0, 1):
    (root / f"rank-{rank}").mkdir()
    (root / f"rank-{rank}/ready.json").write_text("{}")
time.sleep(120)
'''
    other = SharedMemory(create=True, size=128)
    try:
        other_path = Path("/dev/shm") / other.name
        other_stat = other_path.stat()
        status = run_lane([sys.executable, "-c", script, str(output)], output, 2,
                          os.environ.copy(), 20, 1, 1, 0.2, 2)
        assert status["failure"] == "request deadline exceeded"
        assert status["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
        assert status["shared_memory_cleanup"]["status"] == "removed_after_exit"
        identity = json.loads((output / "owned-shm.json").read_text())
        assert not (Path("/dev/shm") / identity["name"]).exists()
        assert os.path.samestat(other_stat, other_path.stat())
        assert json.loads((output / "owned-shm-cleanup.json").read_text()) == status["shared_memory_cleanup"]
    finally:
        other.close()
        other.unlink()


def test_tp2_shared_memory_can_be_attached_and_unlinked_normally(tmp_path):
    output = tmp_path / "normal-shared-memory"
    script = r'''
import json, pathlib, sys
from multiprocessing.shared_memory import SharedMemory
from tools.diagnose_tp import tracked_shared_memory
root = pathlib.Path(sys.argv[1])
engine = tracked_shared_memory(root, SharedMemory)(create=True, size=2**20)
worker = SharedMemory(name=engine.name)
engine.buf[:4] = b"test"
assert bytes(worker.buf[:4]) == b"test"
worker.close()
engine.close()
engine.unlink()
for rank in (0, 1):
    (root / f"rank-{rank}").mkdir()
    (root / f"rank-{rank}/ready.json").write_text("{}")
(root / "request-complete.json").write_text("{}")
(root / "engine-exit.json").write_text(json.dumps(
    {"worker_exitcodes": [0], "process_group_initialized": False}))
'''
    status = run_lane([sys.executable, "-c", script, str(output)], output, 2,
                      os.environ.copy(), 20, 2, 2, 0.2, 2)
    assert status["failure"] is None
    assert status["cleanup_signals"] == []
    assert status["shared_memory_cleanup"]["status"] == "already_released"
    identity = json.loads((output / "owned-shm.json").read_text())
    assert not (Path("/dev/shm") / identity["name"]).exists()


@pytest.mark.parametrize("option", ["startup", "run", "exit", "term", "kill"])
@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_cli_rejects_nonfinite_deadlines_before_creating_output(tmp_path, capsys, option, value):
    output = tmp_path / "output"
    paths = ["--engine-source", "--checkpoint", "--tp1-config", "--tp2-config", "--reference",
             "--original-tp1", "--original-tp2", "--historical-evidence", "--asset-manifest"]
    args = [item for flag in paths for item in (flag, str(tmp_path / "inputs" / "unused"))]
    args += ["--output", str(output), "--preflight-only", f"--{option}-timeout={value}"]
    with pytest.raises(SystemExit) as result:
        main(args)
    assert result.value.code == 2
    assert "finite" in capsys.readouterr().err
    assert not output.exists()


def test_tp2_shared_memory_owned_before_child_attaches(tmp_path):
    output = tmp_path / "before-attach"
    script = r'''
import json, os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
name = os.environ.get("MINIVLLM_P3A_SHM_NAME")
if name:
    stat = (pathlib.Path("/dev/shm") / name).stat()
    (root / "before-attach.json").write_text(json.dumps(
        {"name": name, "device": stat.st_dev, "inode": stat.st_ino}))
while True:
    time.sleep(1)
'''
    status = run_lane([sys.executable, "-c", script, str(output)], output, 2,
                      os.environ.copy(), 1.5, 1, 1, 0.2, 2)
    assert status["failure"] == "startup deadline exceeded"
    assert status["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
    identity = json.loads((output / "before-attach.json").read_text())
    assert identity == json.loads((output / "owned-shm.json").read_text())
    assert status["shared_memory_cleanup"]["status"] == "removed_after_exit"
    assert not (Path("/dev/shm") / identity["name"]).exists()
    assert status["remaining_live_members"] == []


def test_tp2_shared_memory_owned_while_factory_is_interrupted(tmp_path):
    output = tmp_path / "during-attach"
    script = r'''
import json, os, pathlib, signal, sys, time
from multiprocessing.shared_memory import SharedMemory
from tools.diagnose_tp import tracked_shared_memory
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
def paused_factory(*args, **kwargs):
    shared = SharedMemory(*args, **kwargs)
    stat = os.fstat(shared._fd)
    (root / "attached.json").write_text(json.dumps(
        {"name": shared.name, "device": stat.st_dev, "inode": stat.st_ino}))
    while True:
        time.sleep(1)
tracked_shared_memory(root, paused_factory)(create=True, size=2**20)
'''
    try:
        status = run_lane([sys.executable, "-c", script, str(output)], output, 2,
                          os.environ.copy(), 8, 1, 1, 0.2, 2)
        assert status["failure"] == "startup deadline exceeded"
        assert status["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
        identity = json.loads((output / "attached.json").read_text())
        assert identity == json.loads((output / "owned-shm.json").read_text())
        assert status["shared_memory_cleanup"]["status"] == "removed_after_exit"
        assert not (Path("/dev/shm") / identity["name"]).exists()
        assert status["remaining_live_members"] == []
    finally:
        if (output / "attached.json").exists():
            identity = json.loads((output / "attached.json").read_text())
            path = Path("/dev/shm") / identity["name"]
            if path.exists():
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == (identity["device"], identity["inode"]):
                    path.unlink()


def test_cleanup_leaves_a_replaced_shared_memory_inode(tmp_path):
    output = tmp_path / "replaced"
    child = r'''
import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
(root / "pid").write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
for rank in (0, 1):
    (root / f"rank-{rank}").mkdir()
    (root / f"rank-{rank}/ready.json").write_text("{}")
while True:
    time.sleep(1)
'''
    controller = r'''
import os, pathlib, sys
from tools.diagnose_tp import run_lane
root = pathlib.Path(sys.argv[1])
run_lane([sys.executable, "-c", sys.argv[2], str(root)], root, 2,
         os.environ.copy(), 20, 3, 1, 0.2, 2)
'''
    process = subprocess.Popen([sys.executable, "-c", controller, str(output), child],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    replacement = None
    try:
        deadline = time.monotonic() + 30
        while not (output / "rank-1/ready.json").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (output / "rank-1/ready.json").exists()
        identity = json.loads((output / "owned-shm.json").read_text())
        path = Path("/dev/shm") / identity["name"]
        original = path.stat()
        assert (original.st_dev, original.st_ino) == (identity["device"], identity["inode"])
        path.unlink()
        replacement = SharedMemory(name=identity["name"], create=True, size=128)
        replacement_stat = path.stat()
        assert not os.path.samestat(original, replacement_stat)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, (stdout, stderr)
        status = json.loads((output / "status.json").read_text())
        assert status["failure"] == "request deadline exceeded"
        assert status["shared_memory_cleanup"]["status"] == "identity_changed_not_removed"
        assert os.path.samestat(replacement_stat, path.stat())
        assert status["remaining_live_members"] == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if (output / "pid").exists():
            group = int((output / "pid").read_text())
            if live_group_members(group):
                os.killpg(group, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while live_group_members(group) and time.monotonic() < deadline:
                    time.sleep(0.02)
        if replacement is not None:
            replacement.close()
            replacement.unlink()
        elif (output / "owned-shm.json").exists():
            identity = json.loads((output / "owned-shm.json").read_text())
            path = Path("/dev/shm") / identity["name"]
            if path.exists():
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == (identity["device"], identity["inode"]):
                    path.unlink()


@pytest.mark.parametrize("interruption", [signal.SIGINT, signal.SIGTERM])
def test_first_interrupt_during_term_wait_still_cleans_and_records_status(tmp_path, interruption):
    output = tmp_path / "lane"
    child = r'''
import os, pathlib, signal, sys, time
from multiprocessing.shared_memory import SharedMemory
from tools.diagnose_tp import tracked_shared_memory
root = pathlib.Path(sys.argv[1])
(root / "pid").write_text(str(os.getpid()))
owned = tracked_shared_memory(root, SharedMemory)(create=True, size=2**20)
def term(signum, frame):
    (root / "term-seen").write_text(str(signum))
signal.signal(signal.SIGTERM, term)
for rank in (0, 1):
    (root / f"rank-{rank}").mkdir()
    (root / f"rank-{rank}/ready.json").write_text("{}")
while True:
    time.sleep(1)
'''
    controller = r'''
import os, pathlib, signal, sys
from tools.diagnose_tp import run_lane
root = pathlib.Path(sys.argv[1])
def interrupted(signum, frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise InterruptedError(f"received signal {signum}")
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
run_lane([sys.executable, "-c", sys.argv[2], str(root)], root, 2,
         os.environ.copy(), 30, 0.5, 1, 2, 2)
'''
    process = subprocess.Popen([sys.executable, "-c", controller, str(output), child],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 40
        while not (output / "term-seen").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (output / "term-seen").exists(), process.communicate(timeout=5) if process.poll() is not None else ""
        process.send_signal(interruption)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, (stdout, stderr)
        status = json.loads((output / "status.json").read_text())
        identity = json.loads((output / "owned-shm.json").read_text())
        assert status["interrupted_signal"] == interruption
        assert status["failure"] == "request deadline exceeded"
        assert status["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
        assert status["remaining_live_members"] == []
        assert status["shared_memory_cleanup"]["status"] in ("removed_after_exit", "already_released")
        assert not (Path("/dev/shm") / identity["name"]).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if (output / "pid").exists():
            group = int((output / "pid").read_text())
            if live_group_members(group):
                os.killpg(group, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while live_group_members(group) and time.monotonic() < deadline:
                    time.sleep(0.02)
        if (output / "owned-shm.json").exists():
            identity = json.loads((output / "owned-shm.json").read_text())
            path = Path("/dev/shm") / identity["name"]
            if path.exists():
                stat = path.stat()
                if (stat.st_dev, stat.st_ino) == (identity["device"], identity["inode"]):
                    path.unlink()


def test_original_assets_preflight_if_supplied(tmp_path):
    case = os.environ.get("P3_ORIGINAL_CASE")
    historical = os.environ.get("P3_HISTORICAL_EVIDENCE")
    manifest = os.environ.get("P3_ASSET_MANIFEST")
    source = os.environ.get("P3_M4_SOURCE")
    if not all((case, historical, manifest, source)):
        pytest.skip("set the four P3_* paths to run the preserved original asset check")
    case = Path(case)
    originals = [case / "engine-tp1/engine.pt", case / "engine-tp2/engine.pt"]
    configs = [case / "engine-tp1/config.json", case / "engine-tp2/config.json"]
    reference = torch.load(case / "hf-reference.pt", map_location="cpu", weights_only=False)
    runs = [torch.load(path, map_location="cpu", weights_only=False) for path in originals]
    ranks = [torch.load(path.parent / "rank-0.pt", map_location="cpu", weights_only=False) for path in originals]
    gate = original_gate(reference, *runs, *ranks)
    assert gate["strict_greedy_pass"] is False
    assert [gate[key][-3:] for key in ("reference_tokens", "tp1_tokens", "tp2_tokens")] == [
        [32, 33, 41], [32, 33, 41], [32, 33, 62]]
    assert all(len(rows) == 3 and all(row["same_prefix"] and row["within_original_tolerance"]
                                    for row in rows) for rows in gate["same_prefix_logits"].values())
    hashes = historical_hashes(historical, manifest, reference["dtype"], case / "checkpoint",
                               case / "hf-reference.pt", originals, configs)
    assert hashes["all_asset_files_match"] and hashes["checkpoint_and_reference_match"]
    output = tmp_path / "preflight"
    command = [sys.executable, str(ROOT / "tools/diagnose_tp.py"), "--preflight-only",
               "--engine-source", source, "--checkpoint", str(case / "checkpoint"),
               "--tp1-config", str(configs[0]), "--tp2-config", str(configs[1]),
               "--reference", str(case / "hf-reference.pt"), "--original-tp1", str(originals[0]),
               "--original-tp2", str(originals[1]), "--historical-evidence", historical,
               "--asset-manifest", manifest, "--output", str(output)]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, (result.stdout, result.stderr)
    preflight = json.loads((output / "preflight.json").read_text())
    assert not preflight["engine_launched"]
    assert preflight["engine_source"]["git_sha"] == "c7bf6d331b42b1a7e078f6f728126218dc774d51"
    assert (output / "manifest.json").exists()
    rejected_output = tmp_path / "nonfinite-preflight"
    rejected_command = command.copy()
    rejected_command[rejected_command.index("--output") + 1] = str(rejected_output)
    rejected_command.append("--startup-timeout=inf")
    rejected = subprocess.run(rejected_command, env=env, capture_output=True, text=True, timeout=30)
    assert rejected.returncode == 2 and "finite" in rejected.stderr
    assert not rejected_output.exists()
    altered = json.loads(Path(manifest).read_text())
    shard = next(name for name in altered["files"] if name.startswith(case.name + "/checkpoint/")
                 and name.endswith(".safetensors"))
    altered["files"][shard] = "0" * 64
    bad_manifest = tmp_path / "wrong-manifest.json"
    bad_manifest.write_text(json.dumps(altered))
    bad_output = tmp_path / "reject-before-start"
    command.remove("--preflight-only")
    command[command.index("--asset-manifest") + 1] = str(bad_manifest)
    command[command.index("--output") + 1] = str(bad_output)
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 3, (result.stdout, result.stderr)
    assert (bad_output / "input-validation-error.json").exists()
    assert not (bad_output / "tp1").exists()


def test_original_bf16_checkpoint_shards_on_cpu_if_supplied(tmp_path):
    original_case = os.environ.get("P3_ORIGINAL_CASE")
    if not original_case:
        pytest.skip("set P3_ORIGINAL_CASE to inspect the preserved checkpoint")
    case = Path(original_case)
    config_path = case / "engine-tp2/config.json"
    config = json.loads(config_path.read_text())
    output = tmp_path / "tp2"
    run_cpu_model(2, case / "checkpoint", output, tmp_path / "original-cpu.log", config_path)
    for rank in range(2):
        report = verify_weights(output / f"rank-{rank}", case / "checkpoint", config, rank, 2)
        assert report["all_equal"], report
    logits = torch.load(output / "logits-0.pt", weights_only=True)
    assert logits.shape == (1, config["vocab_size"])
    assert logits.dtype == getattr(torch, config["dtype"])
