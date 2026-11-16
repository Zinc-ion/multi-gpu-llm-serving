"""Generate foreground PD deployment and benchmark commands for a fixed candidate."""

import argparse
import json
import math
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
    resources = {
        "worker_max_connections": args.worker_max_connections,
        "worker_max_transfer_bytes": args.worker_max_transfer_bytes,
        "worker_max_payload_bytes": args.worker_max_payload_bytes,
        "gateway_max_active_requests": args.gateway_max_active_requests,
        "gateway_max_payload_bytes": args.gateway_max_payload_bytes,
        "gateway_max_staging_bytes": args.gateway_max_staging_bytes,
    }
    if any(type(value) is not int or value <= 0 for value in resources.values()):
        raise ValueError("resource limits must be positive integers")
    if (resources["worker_max_payload_bytes"] > resources["worker_max_transfer_bytes"]
            or resources["worker_max_payload_bytes"] > resources["gateway_max_payload_bytes"]
            or resources["gateway_max_payload_bytes"] > resources["gateway_max_staging_bytes"]):
        raise ValueError("worker payload must fit its aggregate and gateway payload; staging must fit gateway payload")
    topology = TOPOLOGIES[args.topology]
    if args.m8_comparison:
        if args.topology != "2p2d" or args.physical_gpus is None:
            raise ValueError("M8 comparison requires 2p2d and four explicit physical GPUs")
        if len(set(args.physical_gpus)) != 4 or any(gpu not in range(5) for gpu in args.physical_gpus):
            raise ValueError("physical-gpus must be four distinct GPU indices from 0 to 4")
        topology = [(args.physical_gpus[0], args.physical_gpus[2]),
                    (args.physical_gpus[1], args.physical_gpus[3])]
        for name in ("poll_interval_s", "poll_timeout_s", "max_sample_age_s", "tpot_budget_ms",
                     "step_sample_max_age_ms"):
            value = getattr(args, name)
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be explicitly set to a positive finite number")
        if (args.tpot_margin_ms is None or not math.isfinite(args.tpot_margin_ms)
                or args.tpot_margin_ms < 0 or args.prefill_backlog_tokens is None
                or args.prefill_backlog_tokens < 0 or args.chunk_tokens is None or args.chunk_tokens <= 0
                or args.idle_without_sample is None):
            raise ValueError("M8 comparison requires explicit backlog, margin, chunk and idle policy")
        seeds = [(args.base_seed + index) % 2**32 for index in range(2)]
        if set(seeds) & {seed ^ 0x80000000 for seed in seeds}:
            raise ValueError("base-seed produces a P/D seed collision on a load edge")
    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    if (config["world_size"] != 1 or config["dtype"] != "bfloat16"
            or config["model_revision"] != "40c069824f4251a91eefaf281ebe4c544efd3e18"
            or config["tokenizer_revision"] != config["model_revision"]):
        raise ValueError("base config must be the pinned TP=1 Qwen3-14B BF16 configuration")
    if args.m8_comparison and (config.get("enable_chunked_prefill", False)
                               or config["max_num_batched_tokens"] < config["max_model_length"]):
        raise ValueError("M8 comparison base config must use whole prefill")
    if args.pipeline_min_tokens < 0 or args.pipeline_frame_bytes < 0:
        raise ValueError("pipeline options must be nonnegative")
    if bool(args.pipeline_min_tokens) != bool(args.pipeline_frame_bytes):
        raise ValueError("pipeline requires both min-tokens and frame-bytes")
    if args.m8_comparison and not args.pipeline_frame_bytes:
        raise ValueError("M8 comparison requires explicit pipeline min-tokens and frame-bytes")
    if args.pipeline_frame_bytes:
        cache_dtype = resolve_kv_cache_dtype(config.get("kv_cache_dtype", "auto"),
                                              resolve_dtype(config["dtype"]))
        token_bytes = sum(cache_token_byte_counts(
            config["num_kv_heads"], config["head_dim"], cache_dtype,
            config.get("kv_cache_dtype", "auto") != "auto"))
        if not token_bytes <= args.pipeline_frame_bytes <= resources["worker_max_payload_bytes"]:
            raise ValueError("pipeline frame must fit one KV token and the payload limit")
        if (not args.m8_comparison and
                (not config.get("enforce_eager") or config.get("enable_chunked_prefill", False))):
            raise ValueError("pipeline requires whole eager prefill")
    repo = Path(args.repo_dir)
    if not repo.is_absolute() or not Path(args.python).is_absolute() or not Path(args.model_path).is_absolute():
        raise ValueError("repo-dir, python and pinned model-path must be absolute")
    cache = output / "cache"
    config["model_name_or_path"] = args.model_path
    config["cache_dir"] = str(cache / "hf-hub")
    config.pop("timing_path", None)
    settings = {"pairs": [], "max_active_requests": resources["gateway_max_active_requests"],
                "max_payload_bytes": resources["gateway_max_payload_bytes"],
                "max_staging_bytes": resources["gateway_max_staging_bytes"], "max_request_bytes": 2 * 1024**2,
                "request_timeout": 600}
    if args.pipeline_min_tokens and (not args.m8_comparison or args.m8_routing_transfer == "pipeline"):
        settings["pipeline_min_tokens"] = args.pipeline_min_tokens
    files = {}
    prefix = "#!/usr/bin/env bash\nset -euo pipefail\n" + f"source {quote(output / 'env.sh')}\n"
    prefix += f"cd {quote(repo)}\n"
    prefix += f"test \"$(git rev-parse HEAD)\" = {quote(args.candidate_commit)}\n"
    prefix += f"test \"$(git rev-parse {quote(args.candidate_tag + '^{commit}')})\" = {quote(args.candidate_commit)}\n"
    exports = {
        "PYTHONPYCACHEPREFIX": cache / "pycache", "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": output / "python", "XDG_CACHE_HOME": cache / "xdg",
        "XDG_CONFIG_HOME": cache / "config", "XDG_DATA_HOME": cache / "data",
        "HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "hf-hub",
        "HF_DATASETS_CACHE": cache / "datasets", "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton", "CUDA_CACHE_PATH": cache / "cuda",
        "VLLM_CACHE_ROOT": cache / "vllm", "VLLM_CONFIG_ROOT": cache / "vllm-config",
        "VLLM_RPC_BASE_PATH": output / "tmp", "MODELSCOPE_CACHE": cache / "modelscope",
        "NUMBA_CACHE_DIR": cache / "numba",
        "TMPDIR": output / "tmp",
    }
    env = "#!/usr/bin/env bash\nset -euo pipefail\n"
    env += "".join(f"export {key}={quote(value)}\n" for key, value in exports.items())
    env += f"export PYTHONPATH={quote(repo / 'src')}\nexport CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
    env += "export VLLM_NO_USAGE_STATS=1\nexport VLLM_DO_NOT_TRACK=1\nexport DO_NOT_TRACK=1\n"
    env += "mkdir -p " + " ".join(quote(value) for value in exports.values()) + "\n"
    files["env.sh"] = env
    python = args.python
    for index, (p_gpu, d_gpu) in enumerate(topology):
        seed = (args.base_seed + index) % 2**32
        pe, de = ["127.0.0.1", 9100 + index], ["127.0.0.1", 9200 + index]
        settings["pairs"].append({"prefill": pe, "decode": de})
        for role, gpu, endpoint in (("prefill", p_gpu, pe), ("decode", d_gpu, de)):
            name = f"{role}-{index}"
            node = 0 if gpu in (0, 1, 2) else 1
            role_config = dict(config, seed=seed, timing_path=str(output / f"{name}.timing.jsonl"))
            if args.m8_comparison:
                if role == "prefill":
                    role_config.update(enable_chunked_prefill=False, enforce_eager=True)
                else:
                    role_config.update(enable_chunked_prefill=True, max_num_batched_tokens=args.chunk_tokens)
            files[f"{name}.json"] = json.dumps(role_config, indent=2) + "\n"
            cmd = [python, "-m", "myvllm.pd.worker", "--role", role,
                   "--config", output / f"{name}.json", "--host", endpoint[0], "--port", endpoint[1],
                   "--max-connections", resources["worker_max_connections"],
                   "--max-transfer-bytes", resources["worker_max_transfer_bytes"],
                   "--max-payload-bytes", resources["worker_max_payload_bytes"], "--request-timeout", 600]
            if role == "prefill":
                cmd += ["--decode-host", de[0], "--decode-port", de[1]]
                if args.m8_comparison:
                    cmd += ["--allowed-decode-endpoints",
                            json.dumps([["127.0.0.1", 9200 + i] for i in range(len(topology))])]
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
                "topology": args.topology, "physical_gpu_pairs": topology,
                "model_revision": config["model_revision"], "model_path": args.model_path,
                "numa": {"gpu_nodes": {str(gpu): 0 if gpu in (0, 1, 2) else 1
                                       for pair in topology for gpu in pair},
                         "gateway_node": 0, "client_node": 0},
                "payload_limit_bytes": resources["gateway_max_payload_bytes"], "resources": resources,
                "gateway_settings": settings,
                "pipeline_frame_bytes": args.pipeline_frame_bytes,
                "workloads": {name: str(dataset) for name, _, dataset in workloads},
                "bench_seed": 7, "warmups": 8, "l1_repetitions": 1, "l2_repetitions": 5}
    if args.m8_comparison:
        for transfer in ("bulk", "pipeline"):
            transfer_settings = dict(settings)
            transfer_settings.pop("pipeline_min_tokens", None)
            if transfer == "pipeline":
                transfer_settings["pipeline_min_tokens"] = args.pipeline_min_tokens
            name = f"fixed-{transfer}-gateway"
            files[f"{name}.json"] = json.dumps(transfer_settings, indent=2) + "\n"
            transfer_model = dict(config, timing_path=str(output / f"{name}.timing.jsonl"))
            files[f"{name}-model.json"] = json.dumps(transfer_model, indent=2) + "\n"
            transfer_gateway = [python, "-m", "myvllm.pd.gateway",
                                "--config", output / f"{name}-model.json",
                                "--gateway-config", output / f"{name}.json",
                                "--served-model-name", "Qwen3-14B", "--host", "127.0.0.1", "--port", 8000]
            files[f"{name}.sh"] = (
                prefix + "export CUDA_VISIBLE_DEVICES=''\nexec "
                + command(["numactl", "--cpunodebind=0", "--membind=0", *transfer_gateway]) + "\n")
        elastic = {
            "enabled": True, "prefill_backlog_tokens": args.prefill_backlog_tokens,
            "tpot_budget_ms": args.tpot_budget_ms, "tpot_margin_ms": args.tpot_margin_ms,
            "step_sample_max_age_ms": args.step_sample_max_age_ms,
            "allow_idle_without_sample": args.idle_without_sample == "allow",
        }
        load = {**settings, "routing_mode": "load", "poll_interval_s": args.poll_interval_s,
                "poll_timeout_s": args.poll_timeout_s, "max_sample_age_s": args.max_sample_age_s}
        allowed = [pair["decode"] for pair in settings["pairs"]]
        for variant in ("load", "elastic"):
            variant_settings = {**load, "elastic": elastic if variant == "elastic" else {"enabled": False}}
            files[f"{variant}-gateway.json"] = json.dumps(variant_settings, indent=2) + "\n"
            variant_model = dict(config, timing_path=str(output / f"{variant}-gateway.timing.jsonl"))
            files[f"{variant}-gateway-model.json"] = json.dumps(variant_model, indent=2) + "\n"
            variant_gateway = [python, "-m", "myvllm.pd.gateway",
                               "--config", output / f"{variant}-gateway-model.json",
                               "--gateway-config", output / f"{variant}-gateway.json",
                               "--served-model-name", "Qwen3-14B", "--host", "127.0.0.1", "--port", 8000]
            files[f"{variant}-gateway.sh"] = (
                prefix + "export CUDA_VISIBLE_DEVICES=''\nexec "
                + command(["numactl", "--cpunodebind=0", "--membind=0", *variant_gateway]) + "\n")
            variant_ready = [python, "-m", "myvllm.pd.ready", "--gateway-config",
                             output / f"{variant}-gateway.json"]
            files[f"{variant}-ready.sh"] = prefix + "export CUDA_VISIBLE_DEVICES=''\nexec " + command(variant_ready) + "\n"
            for index, (p_gpu, d_gpu) in enumerate(topology):
                pe, de = settings["pairs"][index]["prefill"], settings["pairs"][index]["decode"]
                for role, gpu, endpoint in (("prefill", p_gpu, pe), ("decode", d_gpu, de)):
                    name = f"{variant}-{role}-{index}"
                    node = 0 if gpu in (0, 1, 2) else 1
                    role_config = dict(config, seed=seeds[index], timing_path=str(output / f"{name}.timing.jsonl"))
                    if role == "prefill":
                        role_config.update(enable_chunked_prefill=False, enforce_eager=True)
                    else:
                        role_config.update(enable_chunked_prefill=True, max_num_batched_tokens=args.chunk_tokens)
                    files[f"{name}.json"] = json.dumps(role_config, indent=2) + "\n"
                    cmd = [python, "-m", "myvllm.pd.worker", "--role", role,
                           "--config", output / f"{name}.json", "--host", endpoint[0], "--port", endpoint[1],
                           "--max-connections", resources["worker_max_connections"],
                           "--max-transfer-bytes", resources["worker_max_transfer_bytes"],
                           "--max-payload-bytes", resources["worker_max_payload_bytes"], "--request-timeout", 600]
                    if role == "prefill":
                        cmd += ["--decode-host", de[0], "--decode-port", de[1],
                                "--allowed-decode-endpoints", json.dumps(allowed)]
                    if args.pipeline_frame_bytes:
                        cmd += ["--pipeline-frame-bytes", args.pipeline_frame_bytes]
                    files[f"{name}.sh"] = (prefix + f"export CUDA_VISIBLE_DEVICES={gpu}\nexec "
                                           + command(["numactl", f"--cpunodebind={node}",
                                                      f"--membind={node}", *cmd]) + "\n")
        controls = {}
        for variant in ("whole", "chunk"):
            controls[variant] = []
            for index, gpu in enumerate(args.physical_gpus):
                name = f"{variant}-control-{index}"
                port = 8100 + index if variant == "whole" else 8200 + index
                endpoint = f"http://127.0.0.1:{port}"
                controls[variant].append(endpoint)
                control_config = dict(config, seed=(args.base_seed + index) % 2**32,
                                      timing_path=str(output / f"{name}.timing.jsonl"),
                                      enable_chunked_prefill=variant == "chunk")
                if variant == "chunk":
                    control_config["max_num_batched_tokens"] = args.chunk_tokens
                files[f"{name}.json"] = json.dumps(control_config, indent=2) + "\n"
                cmd = [python, "-m", "myvllm.server", "--config", output / f"{name}.json",
                       "--served-model-name", "Qwen3-14B", "--host", "127.0.0.1", "--port", port]
                node = 0 if gpu in (0, 1, 2) else 1
                files[f"{name}.sh"] = (prefix + f"export CUDA_VISIBLE_DEVICES={gpu}\nexec "
                                       + command(["numactl", f"--cpunodebind={node}",
                                                  f"--membind={node}", *cmd]) + "\n")
        manifest["m8_comparison"] = {
            "physical_gpus": args.physical_gpus, "prefill_seeds": seeds,
            "decode_sampling_seeds": [seed ^ 0x80000000 for seed in seeds],
            "chunk_tokens": args.chunk_tokens, "load_gateway_settings": load,
            "routing_transfer": args.m8_routing_transfer,
            "elastic_gateway_settings": {**load, "elastic": elastic},
            "endpoints": {"fixed": ["http://127.0.0.1:8000"],
                          "fixed-bulk": ["http://127.0.0.1:8000"],
                          "fixed-pipeline": ["http://127.0.0.1:8000"],
                          "load": ["http://127.0.0.1:8000"],
                          "elastic": ["http://127.0.0.1:8000"], **controls},
        }
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
    parser.add_argument("--worker-max-connections", type=int, default=4)
    parser.add_argument("--worker-max-transfer-bytes", type=int, default=PAYLOAD_LIMIT)
    parser.add_argument("--worker-max-payload-bytes", type=int, default=PAYLOAD_LIMIT)
    parser.add_argument("--gateway-max-active-requests", type=int, default=8)
    parser.add_argument("--gateway-max-payload-bytes", type=int, default=PAYLOAD_LIMIT)
    parser.add_argument("--gateway-max-staging-bytes", type=int, default=2 * PAYLOAD_LIMIT)
    parser.add_argument("--pipeline-min-tokens", type=int, default=0,
                        help="Opt-in prompt threshold; uncalibrated until measured on target hardware")
    parser.add_argument("--pipeline-frame-bytes", type=int, default=0)
    parser.add_argument("--m8-comparison", action="store_true")
    parser.add_argument("--m8-routing-transfer", choices=("bulk", "pipeline"), default="bulk")
    parser.add_argument("--physical-gpus", type=int, nargs=4, metavar=("P0", "P1", "D0", "D1"))
    parser.add_argument("--poll-interval-s", type=float)
    parser.add_argument("--poll-timeout-s", type=float)
    parser.add_argument("--max-sample-age-s", type=float)
    parser.add_argument("--prefill-backlog-tokens", type=int)
    parser.add_argument("--tpot-budget-ms", type=float)
    parser.add_argument("--tpot-margin-ms", type=float)
    parser.add_argument("--step-sample-max-age-ms", type=float)
    parser.add_argument("--idle-without-sample", choices=("allow", "reject"))
    parser.add_argument("--chunk-tokens", type=int)
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
