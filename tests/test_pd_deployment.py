import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools/prepare_pd.py"


def arguments(topology="2p2d", output="/Data2/zincion/ai-infra/experiments/m5/run-01"):
    args = [sys.executable, str(TOOL), "--dry-run", "--output-dir", output,
            "--topology", topology, "--candidate-commit", "a" * 40,
            "--candidate-tag", "m5-candidate-01", "--repo-dir", "/Data2/zincion/ai-infra/MinivLLM",
            "--python", "/Data2/zincion/envs/m2/bin/python", "--model-path", "/readonly/pinned-model"]
    for name in ("l1-1k", "l1-4k", "l1-16k", "l2-sharegpt", "l2-longbench"):
        args += ["--" + name, f"/readonly/{name}.jsonl"]
    return args


@pytest.mark.parametrize("topology,physical", [
    ("same-numa", [(0, 1)]), ("cross-numa", [(0, 3)]), ("2p2d", [(0, 3), (1, 4)]),
])
def test_dry_run_generates_fixed_pairs_limits_and_syntax(topology, physical):
    result = subprocess.run(arguments(topology), cwd=ROOT, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    files = plan["files"]
    gateway = json.loads(files["gateway.json"])
    assert len(gateway["pairs"]) == len(physical)
    assert gateway["max_payload_bytes"] >= 16384 * 163840
    assert gateway["max_staging_bytes"] >= gateway["max_payload_bytes"]
    for i, (p, d) in enumerate(physical):
        assert f"CUDA_VISIBLE_DEVICES={p}" in files[f"prefill-{i}.sh"]
        assert f"CUDA_VISIBLE_DEVICES={d}" in files[f"decode-{i}.sh"]
        assert f"--cpunodebind={0 if p in (0, 1) else 1} --membind={0 if p in (0, 1) else 1}" in files[f"prefill-{i}.sh"]
        assert f"--cpunodebind={0 if d in (0, 1) else 1} --membind={0 if d in (0, 1) else 1}" in files[f"decode-{i}.sh"]
        assert json.loads(files[f"prefill-{i}.json"])["seed"] == i
        assert json.loads(files[f"decode-{i}.json"])["seed"] == i
        assert "--decode-host 127.0.0.1" in files[f"prefill-{i}.sh"]
    manifest = json.loads(files["manifest.json"])
    assert manifest["candidate_commit"] == "a" * 40
    assert manifest["candidate_tag"] == "m5-candidate-01"
    assert manifest["warmups"] == 8 and manifest["l2_repetitions"] == 5
    assert manifest["numa"]["gateway_node"] == manifest["numa"]["client_node"] == 0
    env = files["env.sh"]
    for name in ("PYTHONPYCACHEPREFIX", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "XDG_CACHE_HOME",
                 "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCHINDUCTOR_CACHE_DIR",
                 "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "VLLM_CACHE_ROOT", "NUMBA_CACHE_DIR", "TMPDIR"):
        assert f"export {name}=/Data2/zincion/ai-infra/experiments/m5/run-01/" in env
    assert "tools/run_b0.py" in files["bench-l2-sharegpt.sh"]
    assert "--seed 7 --num-warmups 8" in files["bench-l2-sharegpt.sh"]
    assert "--repetitions 5" in files["bench-l2-sharegpt.sh"]
    assert "--cpunodebind=0 --membind=0" in files["gateway.sh"]
    assert "--cpunodebind=0 --membind=0" in files["bench-l2-sharegpt.sh"]
    assert "myvllm.pd.ready" in files["ready.sh"]
    for name, text in files.items():
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n"], input=text, text=True, check=True)


@pytest.mark.parametrize("output", ["/tmp/unsafe", "/Data2/zincion/ai-infra-other/x",
                                    "/Data2/zincion/ai-infra/../outside"])
def test_output_root_rejects_escape(output):
    result = subprocess.run(arguments(output=output), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "output-dir must be a new directory below" in result.stderr


def test_cli_help_is_read_only():
    for argv in ([sys.executable, str(TOOL), "--help"],
                 [sys.executable, "-m", "myvllm.pd.gateway", "--help"],
                 [sys.executable, "-m", "myvllm.pd.ready", "--help"],
                 [sys.executable, str(ROOT / "tools/analyze_pd.py"), "--help"]):
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
