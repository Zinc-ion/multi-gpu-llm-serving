import json
from pathlib import Path
import subprocess

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
