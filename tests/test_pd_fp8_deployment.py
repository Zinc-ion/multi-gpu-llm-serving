import json
import os
from pathlib import Path
import subprocess
import sys

from test_pd_deployment import arguments


def test_prepared_pipeline_frame_limit_includes_fp8_scales(tmp_path):
    root = Path(__file__).parents[1]
    config = json.loads((root / "configs/qwen3_14b.json").read_text())
    config["kv_cache_dtype"] = "float8_e4m3fn"
    path = tmp_path / "fp8.json"
    path.write_text(json.dumps(config))
    token_bytes = 2 * config["num_kv_heads"] * (config["head_dim"] + 4)
    base = arguments("same-numa") + [
        "--base-config", str(path), "--pipeline-min-tokens", "1024",
        "--pipeline-frame-bytes",
    ]
    rejected = subprocess.run(base + [str(token_bytes - 1)], cwd=root, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "pipeline frame must fit one KV token" in rejected.stderr
    accepted = subprocess.run(base + [str(token_bytes)], cwd=root, capture_output=True, text=True, check=True)
    files = json.loads(accepted.stdout)["files"]
    assert json.loads(files["prefill-0.json"])["kv_cache_dtype"] == "float8_e4m3fn"
    assert f"--pipeline-frame-bytes {token_bytes}" in files["prefill-0.sh"]


def test_documented_generator_uses_candidate_with_stale_pythonpath(tmp_path):
    root = Path(__file__).parents[1]
    legacy = tmp_path / "legacy"
    (legacy / "myvllm").mkdir(parents=True)
    (legacy / "myvllm/__init__.py").write_text("")
    (legacy / "myvllm/config.py").write_text("def resolve_dtype(value):\n    return value\n")
    config = json.loads((root / "configs/qwen3_14b.json").read_text())
    config["kv_cache_dtype"] = "float8_e4m3fn"
    config_path = tmp_path / "fp8.json"
    config_path.write_text(json.dumps(config))
    token_bytes = 2 * config["num_kv_heads"] * (config["head_dim"] + 4)

    lines = (root / "docs/m5_experiments_zh.md").read_text().splitlines()
    dry_run = next(line for line in lines if line.startswith('"$M5_PYTHON" tools/prepare_pd.py')
                   and "--dry-run" in line)
    bindings = [line for line in lines[:lines.index(dry_run)] if line.startswith("export PYTHONPATH=")]
    script = "\n".join([
        "set -euo pipefail",
        'cd "$M5_REPO"',
        *bindings,
        'M5_ARGS=("$@")',
        '''"$M5_PYTHON" -c 'import myvllm.config; print(myvllm.config.__file__)' ''',
        dry_run,
    ])
    args = arguments("same-numa")[3:]
    args[args.index("--repo-dir") + 1] = str(root)
    args.extend(["--base-config", str(config_path), "--pipeline-min-tokens", "1",
                 "--pipeline-frame-bytes", str(token_bytes)])
    env = {**os.environ, "PYTHONPATH": str(legacy), "M5_REPO": str(root),
           "M5_PYTHON": sys.executable, "M5_BOOT": str(tmp_path)}
    result = subprocess.run(["bash", "-c", script, "documented-generator", *args],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == root / "src/myvllm/config.py"
    files = json.loads((tmp_path / "plan.json").read_text())["files"]
    assert json.loads(files["prefill-0.json"])["kv_cache_dtype"] == "float8_e4m3fn"
    assert f"--pipeline-frame-bytes {token_bytes}" in files["prefill-0.sh"]
