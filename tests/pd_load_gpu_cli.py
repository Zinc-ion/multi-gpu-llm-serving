"""Record the actual imports, then invoke an unmodified PD production CLI."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    module_name = sys.argv[1]
    if module_name not in ("myvllm.pd.worker", "myvllm.pd.gateway"):
        raise ValueError("Unsupported smoke CLI")
    root = Path(__file__).resolve().parents[1]
    expected = os.environ["M8_GPU_EXPECTED_SHA"]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != expected:
        raise ValueError(f"Checkout HEAD {head} does not match expected {expected}")
    names = (
        module_name, "myvllm.pd.core", "myvllm.pd.pipeline", "myvllm.pd.client",
        "myvllm.engine.llm_engine", "myvllm.engine.model_runner",
    )
    modules = {}
    for name in dict.fromkeys(names):
        module = importlib.import_module(name)
        source = Path(inspect.getsourcefile(module)).resolve()
        relative = source.relative_to(root)
        content = source.read_bytes()
        committed = subprocess.check_output(["git", "-C", str(root), "show", f"{head}:{relative}"])
        if content != committed:
            raise ValueError(f"Imported source differs from candidate: {source}")
        modules[name] = {"path": str(source), "sha256": hashlib.sha256(content).hexdigest()}
    manifest = {
        "pid": os.getpid(), "cwd": os.getcwd(), "python": sys.executable, "head": head,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "production_cli": module_name + ":main", "modules": modules,
    }
    Path(os.environ["M8_GPU_PROVENANCE"]).write_text(json.dumps(manifest, indent=2) + "\n")
    sys.argv = [module_name, *sys.argv[2:]]
    importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
