"""Generate foreground PD deployment and benchmark commands for a fixed candidate."""

import argparse
import json
from pathlib import Path
import re
import shlex

from myvllm.config import resolve_dtype, resolve_kv_cache_dtype
from myvllm.pd.protocol import cache_token_byte_counts


ROOT = Path("/Data2/zincion/ai-infra")
TOPOLOGIES = {
    "same-numa": [(0, 1)],
    "cross-numa": [(0, 3)],
    "2p2d": [(0, 3), (1, 4)],
}
PAYLOAD_LIMIT = 3 * 1024**3


def quote(value):
    return shlex.quote(str(value))


def command(args):
    return " ".join(quote(arg) for arg in args)


def generate(args):
    root = ROOT.resolve()
    output = Path(args.output_dir).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError(f"output-dir must be a new directory below {ROOT}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.candidate_commit):
        raise ValueError("candidate-commit must be a full 40-character commit SHA")
    if not 0 <= args.base_seed < 2**32:
        raise ValueError("base-seed must be a 32-bit unsigned integer")
    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    if (config["world_size"] != 1 or config["dtype"] != "bfloat16"
            or config["model_revision"] != "40c069824f4251a91eefaf281ebe4c544efd3e18"
            or config["tokenizer_revision"] != config["model_revision"]):
        raise ValueError("base config must be the pinned TP=1 Qwen3-14B BF16 configuration")
    if args.pipeline_min_tokens < 0 or args.pipeline_frame_bytes < 0:
        raise ValueError("pipeline options must be nonnegative")
    if bool(args.pipeline_min_tokens) != bool(args.pipeline_frame_bytes):
        raise ValueError("pipeline requires both min-tokens and frame-bytes")
    if args.pipeline_frame_bytes:
        cache_dtype = resolve_kv_cache_dtype(config.get("kv_cache_dtype", "auto"),
                                              resolve_dtype(config["dtype"]))
        token_bytes = sum(cache_token_byte_counts(
            config["num_kv_heads"], config["head_dim"], cache_dtype,
            config.get("kv_cache_dtype", "auto") != "auto"))
        if not token_bytes <= args.pipeline_frame_bytes <= PAYLOAD_LIMIT:
            raise ValueError("pipeline frame must fit one KV token and the payload limit")
        if not config.get("enforce_eager") or config.get("enable_chunked_prefill", False):
            raise ValueError("pipeline requires whole eager prefill")
    repo = Path(args.repo_dir)
    if not repo.is_absolute() or not Path(args.python).is_absolute() or not Path(args.model_path).is_absolute():
        raise ValueError("repo-dir, python and pinned model-path must be absolute")
    cache = output / "cache"
    config["model_name_or_path"] = args.model_path
    config["cache_dir"] = str(cache / "hf-hub")
    config.pop("timing_path", None)
    settings = {"pairs": [], "max_active_requests": 8, "max_payload_bytes": PAYLOAD_LIMIT,
                "max_staging_bytes": 2 * PAYLOAD_LIMIT, "max_request_bytes": 2 * 1024**2,
                "request_timeout": 600}
    if args.pipeline_min_tokens:
        settings["pipeline_min_tokens"] = args.pipeline_min_tokens
    files = {}
    prefix = "#!/usr/bin/env bash\nset -euo pipefail\n" + f"source {quote(output / 'env.sh')}\n"
    prefix += f"cd {quote(repo)}\n"
    prefix += f"test \"$(git rev-parse HEAD)\" = {quote(args.candidate_commit)}\n"
    prefix += f"test \"$(git rev-parse {quote(args.candidate_tag + '^{commit}')})\" = {quote(args.candidate_commit)}\n"
    exports = {
        "PYTHONPYCACHEPREFIX": cache / "pycache", "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": output / "python", "XDG_CACHE_HOME": cache / "xdg",
        "HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "hf-hub",
        "HF_DATASETS_CACHE": cache / "datasets", "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton", "CUDA_CACHE_PATH": cache / "cuda",
        "VLLM_CACHE_ROOT": cache / "vllm", "NUMBA_CACHE_DIR": cache / "numba",
        "TMPDIR": output / "tmp",
    }
    env = "#!/usr/bin/env bash\nset -euo pipefail\n"
    env += "".join(f"export {key}={quote(value)}\n" for key, value in exports.items())
    env += f"export PYTHONPATH={quote(repo / 'src')}\nexport CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
    env += "mkdir -p " + " ".join(quote(value) for value in exports.values()) + "\n"
    files["env.sh"] = env
    python = args.python
    for index, (p_gpu, d_gpu) in enumerate(TOPOLOGIES[args.topology]):
        seed = (args.base_seed + index) % 2**32
        pe, de = ["127.0.0.1", 9100 + index], ["127.0.0.1", 9200 + index]
        settings["pairs"].append({"prefill": pe, "decode": de})
        for role, gpu, endpoint in (("prefill", p_gpu, pe), ("decode", d_gpu, de)):
            name = f"{role}-{index}"
            node = 0 if gpu in (0, 1) else 1
            role_config = dict(config, seed=seed, timing_path=str(output / f"{name}.timing.jsonl"))
            files[f"{name}.json"] = json.dumps(role_config, indent=2) + "\n"
            cmd = [python, "-m", "myvllm.pd.worker", "--role", role,
                   "--config", output / f"{name}.json", "--host", endpoint[0], "--port", endpoint[1],
                   "--max-connections", 4, "--max-transfer-bytes", PAYLOAD_LIMIT, "--request-timeout", 600]
            if role == "prefill":
                cmd += ["--decode-host", de[0], "--decode-port", de[1]]
            if args.pipeline_frame_bytes:
                cmd += ["--pipeline-frame-bytes", args.pipeline_frame_bytes]
            files[f"{name}.sh"] = (prefix + f"export CUDA_VISIBLE_DEVICES={gpu}\nexec " +
                                   command(["numactl", f"--cpunodebind={node}", f"--membind={node}", *cmd]) + "\n")
    config["timing_path"] = str(output / "gateway.timing.jsonl")
    files["gateway-model.json"] = json.dumps(config, indent=2) + "\n"
    files["gateway.json"] = json.dumps(settings, indent=2) + "\n"
    gateway = [python, "-m", "myvllm.pd.gateway", "--config", output / "gateway-model.json",
               "--gateway-config", output / "gateway.json", "--served-model-name", "Qwen3-14B",
               "--host", "127.0.0.1", "--port", 8000]
    files["gateway.sh"] = (prefix + "export CUDA_VISIBLE_DEVICES=''\nexec " +
                           command(["numactl", "--cpunodebind=0", "--membind=0", *gateway]) + "\n")
    ready = [python, "-m", "myvllm.pd.ready", "--gateway-config", output / "gateway.json"]
    files["ready.sh"] = prefix + "export CUDA_VISIBLE_DEVICES=''\nexec " + command(ready) + "\n"
    files["workers-ready.sh"] = prefix + "export CUDA_VISIBLE_DEVICES=''\nexec " + command(ready + ["--workers-only"]) + "\n"
    workloads = [("l1-1k", "l1", args.l1_1k), ("l1-4k", "l1", args.l1_4k),
                 ("l1-16k", "l1", args.l1_16k), ("l2-sharegpt", "l2", args.l2_sharegpt),
                 ("l2-longbench", "l2", args.l2_longbench)]
    for label, mode, dataset in workloads:
        result = output / "results" / label
        cmd = [python, repo / "tools/run_b0.py", "--dataset", dataset, "--model", "Qwen3-14B",
               "--tokenizer", args.model_path, "--mode", mode, "--seed", 7, "--num-warmups", 8,
               "--output-dir", result]
        if mode == "l1":
            cmd += ["--concurrency", 1, "--repetitions", 1]
        else:
            cmd += ["--request-rates", 1, 2, 4, "--repetitions", 5]
        files[f"bench-{label}.sh"] = (prefix + "export CUDA_VISIBLE_DEVICES=''\n"
                                      + f"test ! -e {quote(result)}\nexec "
                                      + command(["numactl", "--cpunodebind=0", "--membind=0", *cmd]) + "\n")
    manifest = {"candidate_commit": args.candidate_commit, "candidate_tag": args.candidate_tag,
                "topology": args.topology, "physical_gpu_pairs": TOPOLOGIES[args.topology],
                "model_revision": config["model_revision"], "model_path": args.model_path,
                "numa": {"gpu_nodes": {str(gpu): 0 if gpu in (0, 1) else 1
                                       for pair in TOPOLOGIES[args.topology] for gpu in pair},
                         "gateway_node": 0, "client_node": 0},
                "payload_limit_bytes": PAYLOAD_LIMIT, "gateway_settings": settings,
                "pipeline_frame_bytes": args.pipeline_frame_bytes,
                "workloads": {name: str(dataset) for name, _, dataset in workloads},
                "bench_seed": 7, "warmups": 8, "l1_repetitions": 1, "l2_repetitions": 5}
    files["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    return output, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topology", required=True, choices=TOPOLOGIES)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model-path", required=True, help="Existing pinned local model snapshot")
    parser.add_argument("--base-config", default="configs/qwen3_14b.json")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--pipeline-min-tokens", type=int, default=0,
                        help="Opt-in prompt threshold; uncalibrated until measured on target hardware")
    parser.add_argument("--pipeline-frame-bytes", type=int, default=0)
    for name in ("l1-1k", "l1-4k", "l1-16k", "l2-sharegpt", "l2-longbench"):
        parser.add_argument("--" + name, required=True, help="Existing materialized JSONL and its manifest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        output, files = generate(args)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"output_dir": str(output), "files": files}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
