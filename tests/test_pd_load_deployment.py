import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from myvllm.pd.worker import PDWorker
from test_pd_deployment import arguments
from test_pd_server import cpu_gateway_engine


ROOT = Path(__file__).parents[1]


def comparison_args():
    return [
        "--m8-comparison", "--physical-gpus", "0", "2", "3", "4",
        "--poll-interval-s", ".1", "--poll-timeout-s", ".5", "--max-sample-age-s", "2",
        "--prefill-backlog-tokens", "100", "--tpot-budget-ms", "50", "--tpot-margin-ms", "5",
        "--step-sample-max-age-ms", "1000", "--idle-without-sample", "allow",
        "--chunk-tokens", "1024", "--pipeline-min-tokens", "1024",
        "--pipeline-frame-bytes", "65536",
    ]


def worker_args(text):
    words = shlex.split(text.splitlines()[-1])
    module = words.index("-m")
    assert words[module + 1] == "myvllm.pd.worker"
    return words[module + 2:]


def value(words, name):
    return words[words.index(name) + 1]


@pytest.mark.parametrize("routing_transfer", ["bulk", "pipeline"])
def test_m8_roles_and_fixed_transfer_ablation_use_real_worker_startup(tmp_path, routing_transfer):
    config = json.loads((ROOT / "configs/qwen3_14b.json").read_text())
    config["enforce_eager"] = False
    base = tmp_path / "base.json"
    base.write_text(json.dumps(config))
    cmd = arguments() + comparison_args() + [
        "--base-config", str(base), "--m8-routing-transfer", routing_transfer,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    files = json.loads(result.stdout)["files"]

    async def run():
        workers = []
        try:
            for variant in ("", "load-", "elastic-"):
                for index in range(2):
                    for role in ("decode", "prefill"):
                        name = f"{variant}{role}-{index}"
                        role_config = json.loads(files[f"{name}.json"])
                        args = worker_args(files[f"{name}.sh"])
                        assert Path(value(args, "--config")).name == f"{name}.json"
                        assert value(args, "--role") == role
                        assert value(args, "--pipeline-frame-bytes") == "65536"
                        assert role_config["enable_chunked_prefill"] == (role == "decode")
                        assert role_config["enforce_eager"] == (role == "prefill")
                        expected_budget = 1024 if role == "decode" else config["max_num_batched_tokens"]
                        assert role_config["max_num_batched_tokens"] == expected_budget

                        def engine_factory(generated):
                            test_config = {
                                **generated, "chunked": generated["enable_chunked_prefill"],
                                "budget": 3 if generated["enable_chunked_prefill"] else 32,
                            }
                            engine = cpu_gateway_engine(test_config)
                            engine.config["enforce_eager"] = generated["enforce_eager"]
                            return engine

                        kwargs = {}
                        if role == "prefill":
                            kwargs["decode_endpoint"] = (
                                value(args, "--decode-host"), int(value(args, "--decode-port")))
                            if "--allowed-decode-endpoints" in args:
                                kwargs["allowed_decode_endpoints"] = json.loads(
                                    value(args, "--allowed-decode-endpoints"))
                        worker = PDWorker(
                            role, role_config, max_connections=int(value(args, "--max-connections")),
                            max_transfer_bytes=int(value(args, "--max-transfer-bytes")),
                            request_timeout=float(value(args, "--request-timeout")),
                            pipeline_frame_bytes=int(value(args, "--pipeline-frame-bytes")),
                            engine_factory=engine_factory, **kwargs,
                        )
                        await worker.start("127.0.0.1", 0)
                        workers.append(worker)
                        assert worker.owner.core.scheduler.enable_chunked_prefill == (role == "decode")
        finally:
            for worker in reversed(workers):
                await worker.close()

    asyncio.run(run())
    gateways = {name: json.loads(files[name]) for name in (
        "gateway.json", "load-gateway.json", "elastic-gateway.json",
        "fixed-bulk-gateway.json", "fixed-pipeline-gateway.json",
    )}
    threshold = 1024 if routing_transfer == "pipeline" else 0
    assert all(gateways[name].get("pipeline_min_tokens", 0) == threshold
               for name in ("gateway.json", "load-gateway.json", "elastic-gateway.json"))
    assert gateways["fixed-bulk-gateway.json"].get("pipeline_min_tokens", 0) == 0
    assert gateways["fixed-pipeline-gateway.json"]["pipeline_min_tokens"] == 1024
    for name in ("fixed-bulk-gateway.sh", "fixed-pipeline-gateway.sh"):
        words = shlex.split(files[name].splitlines()[-1])
        assert Path(value(words, "--gateway-config")).name == name.replace(".sh", ".json")
        model = Path(value(words, "--config")).name
        assert model == name.replace(".sh", "-model.json")
        assert json.loads(files[model])["timing_path"].endswith(name.replace(".sh", ".timing.jsonl"))

    manifest = json.loads(files["manifest.json"])["m8_comparison"]
    assert manifest["physical_gpus"] == [0, 2, 3, 4]
    for variant in ("whole", "chunk"):
        endpoints = manifest["endpoints"][variant]
        assert len(endpoints) == len(set(endpoints)) == 4
        for index, gpu in enumerate((0, 2, 3, 4)):
            name = f"{variant}-control-{index}"
            instance = json.loads(files[f"{name}.json"])
            assert instance["seed"] == index
            assert instance["enable_chunked_prefill"] == (variant == "chunk")
            assert f"CUDA_VISIBLE_DEVICES={gpu}" in files[f"{name}.sh"]
            assert f"--cpunodebind={0 if gpu in (0, 1, 2) else 1}" in files[f"{name}.sh"]
            assert Path(value(shlex.split(files[f"{name}.sh"].splitlines()[-1]), "--config")).name == f"{name}.json"
    for name, text in files.items():
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_generated_env_confines_vllm_and_modelscope_directories(tmp_path):
    output = tmp_path / "fresh deployment"
    cli = (
        "import sys; from pathlib import Path; from tools import prepare_pd; "
        "prepare_pd.ROOT = Path(sys.argv.pop(1)); prepare_pd.main()"
    )
    cmd = [sys.executable, "-c", cli, str(tmp_path), *arguments(output=str(output))[3:], *comparison_args()]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)

    home = tmp_path / "unchanged home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), CUDA_VISIBLE_DEVICES="")
    names = ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "VLLM_CONFIG_ROOT",
             "VLLM_CACHE_ROOT", "VLLM_RPC_BASE_PATH", "MODELSCOPE_CACHE", "TMPDIR")
    for name in names:
        env[name] = str(tmp_path / "outside")
    env.update(VLLM_NO_USAGE_STATS="0", VLLM_DO_NOT_TRACK="0", DO_NOT_TRACK="0")
    probe = (
        "import json, os; from vllm import envs; "
        "from vllm.usage.usage_lib import is_usage_stats_enabled; "
        f"names = {(*names, 'HOME', 'VLLM_NO_USAGE_STATS', 'VLLM_DO_NOT_TRACK', 'DO_NOT_TRACK')!r}; "
        "print(json.dumps({'env': {name: os.environ[name] for name in names}, 'config': envs.VLLM_CONFIG_ROOT, "
        "'cache': envs.VLLM_CACHE_ROOT, 'rpc': envs.VLLM_RPC_BASE_PATH, "
        "'usage_enabled': is_usage_stats_enabled()}))"
    )
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; exec "$2" -c "$3"', "--",
         str(output / "env.sh"), sys.executable, probe],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    observed = json.loads(result.stdout.splitlines()[-1])
    assert observed["env"]["HOME"] == str(home)
    assert not list(home.iterdir())
    for name in names:
        path = Path(observed["env"][name])
        assert path.is_relative_to(output) and path.is_dir()
    for name in ("config", "cache", "rpc"):
        assert Path(observed[name]).is_relative_to(output)
    assert observed["env"]["VLLM_NO_USAGE_STATS"] == "1"
    assert observed["env"]["VLLM_DO_NOT_TRACK"] == "1"
    assert observed["env"]["DO_NOT_TRACK"] == "1"
    assert observed["usage_enabled"] is False
