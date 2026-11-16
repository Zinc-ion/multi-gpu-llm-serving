"""Capture independent reference/P/D runs and analyze frozen logits gates."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid


SCRIPT = Path(__file__).resolve()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write(path, write):
    """Publish a fully written file without ever replacing an existing target."""
    path = Path(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            write(stream)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, value):
    atomic_write(path, lambda stream: stream.write(
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()))


def save_pt(path, value):
    import torch
    atomic_write(path, lambda stream: torch.save(value, stream))


def load_pt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def engine_provenance(source, expected_sha):
    import myvllm
    source = Path(source).resolve()
    module = Path(myvllm.__file__).resolve()
    if module != source / "src/myvllm/__init__.py":
        raise RuntimeError(f"Imported engine {module} is not from requested source {source}")
    root = Path(git(module.parent, "rev-parse", "--show-toplevel")).resolve()
    actual_sha = git(root, "rev-parse", "HEAD")
    if root != source or actual_sha != expected_sha:
        raise RuntimeError(f"Engine source/SHA mismatch: {root} {actual_sha}")
    tracked = git(root, "ls-files", "--", "src/myvllm").splitlines()
    if "src/myvllm/__init__.py" not in tracked:
        raise RuntimeError("Imported engine is not tracked in its owning repository")
    files = {name: sha256(root / name) if (root / name).is_file() else None for name in tracked}
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "--", "src/myvllm").splitlines()
    extras = {name: sha256(root / name) for name in untracked if (root / name).is_file()}
    imports = {}
    for name, imported in list(sys.modules.items()):
        if name == "myvllm" or name.startswith("myvllm."):
            path = getattr(imported, "__file__", None)
            if path:
                resolved = Path(path).resolve()
                if not resolved.is_relative_to(source / "src/myvllm"):
                    raise RuntimeError(f"Engine module imported from another source: {name} {resolved}")
                imports[name] = str(resolved)
    return {
        "source": str(root), "commit": actual_sha, "module": str(module),
        "imports": imports, "tracked_sha256": files, "untracked_sha256": extras,
        "source_status": git(root, "status", "--porcelain", "--untracked-files=all", "--", "src/myvllm"),
    }


def harness_provenance():
    result = {"path": str(SCRIPT), "sha256": sha256(SCRIPT), "commit": None, "source": None}
    try:
        root = Path(git(SCRIPT.parent, "rev-parse", "--show-toplevel")).resolve()
        result.update(source=str(root), commit=git(root, "rev-parse", "HEAD"))
    except subprocess.CalledProcessError:
        pass
    return result


def tokenizer_identity(tokenizer, cases):
    vocab = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()
    encodings = {str(i): tokenizer.encode(case["prompt"]) for i, case in enumerate(cases)}
    decodings = {str(i): tokenizer.decode(case["input_ids"]) for i, case in enumerate(cases)}
    encoding_matches = {str(i): encodings[str(i)] == case["input_ids"] for i, case in enumerate(cases)}
    decoding_matches = {str(i): decodings[str(i)] == case["prompt"] for i, case in enumerate(cases)}
    # prepare truncates IDs before decoding; a partial UTF-8 character need not round-trip.
    mismatches = [case["name"] for i, case in enumerate(cases)
                  if not encoding_matches[str(i)] and not decoding_matches[str(i)]]
    if mismatches:
        raise ValueError(f"Prompt text does not encode to input_ids or match their decoding: {mismatches}")
    return {
        "class": type(tokenizer).__name__, "vocab_sha256": hashlib.sha256(vocab).hexdigest(),
        "special_tokens": str(tokenizer.special_tokens_map), "prompt_encodings": encodings,
        "input_decodings": decodings, "input_matches_prompt_encoding": encoding_matches,
        "prompt_matches_input_decoding": decoding_matches,
        "lossy_nonroundtrip": {index: not matches for index, matches in encoding_matches.items()},
        "model_input": "input_ids",
    }


def execution_affinity(requested_node):
    fields = {}
    for line in Path("/proc/thread-self/status").read_text().splitlines():
        name, separator, value = line.partition(":")
        if separator and name in ("Cpus_allowed", "Cpus_allowed_list", "Mems_allowed", "Mems_allowed_list"):
            fields[name] = value.strip()
    return {"requested_numa_node": requested_node, "thread_id": threading.get_native_id(),
            "allowed_masks": fields, "cpu_affinity": sorted(os.sched_getaffinity(0))}


def cache_format(runner, config):
    import torch
    from myvllm.config import resolve_dtype

    cache = runner.kv_cache
    scales = getattr(runner, "kv_cache_scales", None)
    requested = config.get("kv_cache_dtype", "auto")
    if requested == "auto":
        expected = resolve_dtype(config["dtype"])
        if cache.dtype != expected or scales is not None:
            raise ValueError("Allocated auto KV cache does not match model dtype/no-scales contract")
    elif requested == "float8" + "_e4m3fn":
        expected = getattr(torch, requested)
        if (cache.dtype != expected or scales is None or scales.dtype != torch.float32
                or tuple(scales.shape) != tuple(cache.shape[:-1]) or scales.device != cache.device):
            raise ValueError("Requested FP8 KV cache was not allocated with matching FP32 scales")
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {requested}")
    return {"requested": requested, "dtype": str(cache.dtype), "pool_shape": list(cache.shape),
            "device": str(cache.device), "scale_dtype": str(scales.dtype) if scales is not None else None,
            "scale_pool_shape": list(scales.shape) if scales is not None else None}


def snapshot_kv(runner, seq, valid_tokens):
    import torch
    cache = runner.kv_cache
    scales = getattr(runner, "kv_cache_scales", None)
    chunks = []
    for i, block in enumerate(seq.block_table):
        valid = min(seq.block_size, valid_tokens - i * seq.block_size)
        if valid <= 0:
            break
        data = cache[:, :, block, :valid].detach().contiguous()
        item = {"shape": list(data.shape), "data": data.view(torch.uint8).cpu().clone()}
        if scales is not None:
            scale = scales[:, :, block, :valid].detach().contiguous()
            item.update(scale_shape=list(scale.shape), scales=scale.view(torch.uint8).cpu().clone())
        chunks.append(item)
    return {"valid_tokens": valid_tokens, "blocks": list(seq.block_table[:len(chunks)]),
            "dtype": str(cache.dtype), "chunks": chunks}


def compare_kv(left, right):
    if (left["valid_tokens"], left["dtype"], len(left["chunks"])) != (
            right["valid_tokens"], right["dtype"], len(right["chunks"])):
        raise ValueError("KV snapshots have different formats or valid lengths")
    data_bytes = data_different = scale_bytes = scale_different = 0
    for a, b in zip(left["chunks"], right["chunks"]):
        if a.keys() != b.keys() or a["shape"] != b["shape"]:
            raise ValueError("KV snapshot data shapes differ")
        data_bytes += a["data"].numel()
        data_different += int((a["data"] != b["data"]).sum().item())
        if "scales" in a:
            if a["scale_shape"] != b["scale_shape"]:
                raise ValueError("KV snapshot scale shapes differ")
            scale_bytes += a["scales"].numel()
            scale_different += int((a["scales"] != b["scales"]).sum().item())
    return {"valid_tokens": left["valid_tokens"], "dtype": left["dtype"], "data_bytes": data_bytes,
            "data_different_bytes": data_different, "scale_bytes": scale_bytes,
            "scale_different_bytes": scale_different, "equal": data_different == 0 and scale_different == 0}


def row_path(root, role, index, completion):
    return Path(root) / "partial" / f"{role}-{index:05d}-{completion:06d}.pt"


def restored_path(root, index):
    return Path(root) / "partial" / f"decode-{index:05d}-restored.pt"


def capture_engine_class(role, root, cases, raw_kv, source, expected_sha, numa_node=None):
    import torch
    from myvllm.engine.llm_engine import LLMEngine

    indices = {f"case-{i:05d}": i for i in range(len(cases))}

    class CaptureEngine(LLMEngine):
        def __init__(self, config):
            self.capture_hook = None
            self.capture_head = None
            self.capture_snapshots = {}
            self.capture_restored = set()
            self.capture_format = None
            input_config = dict(config)
            try:
                super().__init__(config)
                self.capture_format = cache_format(self.model_runner, config)
                missing = set(dict(self.model_runner.model.named_parameters())) - self.model_runner.loaded_params
                if missing:
                    raise RuntimeError(f"Unloaded model parameters: {sorted(missing)}")
                self.capture_hook = self.model_runner.model.lm_head.register_forward_hook(self._head)
                runner_call = self.model_runner.call

                def call(method, *args):
                    result = runner_call(method, *args)
                    if raw_kv and method == "run":
                        for seq in args[0]:
                            end = seq.num_cached_tokens + seq.num_scheduled_tokens
                            if end == len(seq):
                                self.capture_snapshots[seq.seq_id] = snapshot_kv(self.model_runner, seq, end)
                    return result

                self.model_runner.call = call
                info = {"role": role, "pid": os.getpid(), "input_config": input_config,
                        "config": dict(config), "cache": self.capture_format,
                        "affinity": execution_affinity(numa_node),
                        "tokenizer": tokenizer_identity(self.tokenizer, cases),
                        "engine": engine_provenance(source, expected_sha),
                        "harness": harness_provenance(), "inference_mode": "explicit caller context around engine.step"}
                save_json(Path(root) / f"{role}.engine.json", info)
            except BaseException as exc:
                try:
                    self.exit()
                except BaseException as cleanup:
                    exc.add_note(f"Capture engine cleanup also failed: {cleanup!r}")
                raise

        def _head(self, module, inputs, output):
            if self.capture_head is not None:
                raise RuntimeError("LM head executed more than once in one engine step")
            self.capture_head = output.detach().float().cpu().clone()

        def step(self, on_tokens=None):
            self.capture_head = None
            self.capture_snapshots = {}
            if raw_kv and role == "decode":
                for seq in self.scheduler.running:
                    if (seq.num_completion_tokens == 1 and seq.num_cached_tokens == seq.num_prompt_tokens
                            and seq.seq_id not in self.capture_restored):
                        index = indices[self.request_ids[seq.seq_id]]
                        save_pt(restored_path(root, index), snapshot_kv(self.model_runner, seq, seq.num_cached_tokens))
                        self.capture_restored.add(seq.seq_id)
            rows = 0

            def record(seq, token):
                nonlocal rows
                request_id = self.request_ids[seq.seq_id]
                index = indices[request_id]
                if self.capture_head is None or rows >= self.capture_head.shape[0]:
                    raise RuntimeError("Token callback has no corresponding LM-head row")
                completion = seq.num_completion_tokens
                if (role == "prefill" and completion != 1) or (role == "decode" and completion <= 1):
                    raise RuntimeError(f"Unexpected {role} completion count {completion}")
                item = {"request_id": request_id, "role": role, "token": token,
                        "completion_tokens": completion, "finished": seq.is_finished,
                        "blocks_after_postprocess": list(seq.block_table),
                        "logits": self.capture_head[rows].clone(),
                        "kv": self.capture_snapshots.get(seq.seq_id) if raw_kv else None}
                save_pt(row_path(root, role, index, completion), item)
                rows += 1
                if on_tokens is not None:
                    on_tokens(seq, token)

            with torch.inference_mode():
                result = super().step(on_tokens=record)
            expected = 0 if self.capture_head is None else self.capture_head.shape[0]
            if rows != expected:
                raise RuntimeError(f"LM-head rows ({expected}) do not match callbacks ({rows})")
            return result

        def exit(self):
            try:
                if self.capture_hook is not None:
                    self.capture_hook.remove()
                    self.capture_hook = None
            finally:
                if getattr(self, "model_runner", None) is not None:
                    super().exit()

    return CaptureEngine


def completed_case(root, role, index, case, config, rows, metadata):
    import torch
    if not rows or any(row["completion_tokens"] != i + 1 for i, row in enumerate(rows)):
        raise RuntimeError(f"Incomplete callback sequence for {case['name']}")
    data = {"metadata": {"config": config, **metadata}, "cases": [{**case,
            "token_ids": [row["token"] for row in rows],
            "logits": torch.stack([row["logits"] for row in rows]),
            "num_prompt_tokens": len(case["input_ids"])}]}
    save_pt(Path(root) / f"{role}-{index:05d}.pt", data)


def read_inputs(args):
    source = Path(args.engine_source).resolve()
    if not source.is_dir() or len(args.expected_sha) != 40 or any(c not in "0123456789abcdef" for c in args.expected_sha):
        raise ValueError("engine source and full lowercase 40-character expected SHA are required")
    configs = {}
    for role in ("reference", "prefill", "decode"):
        path = Path(getattr(args, role + "_config")).resolve()
        configs[role] = json.loads(path.read_text())
        world_size = configs[role]["world_size"]
        if world_size != 1 and not (role == "prefill" and world_size == 2):
            raise ValueError("Capture requires TP1 reference/decode and TP1 or TP2 prefill")
    if configs["prefill"]["world_size"] == 2:
        if args.transfer != "bulk" or args.raw_kv:
            raise ValueError("TP2 prefill capture requires bulk and does not support --raw-kv")
        devices = prefill_gpu_selection(args.prefill_gpu).split(",")
        if len(devices) != 2:
            raise ValueError("TP2 prefill requires exactly two visible GPUs")
    elif "," in args.prefill_gpu:
        raise ValueError("TP1 prefill requires exactly one visible GPU")
    fields = ("model_name_or_path", "model_type", "model_revision", "tokenizer_revision", "dtype",
              "eos", "max_model_length", "block_size", "num_layers", "num_kv_heads", "head_dim", "vocab_size")
    for field in fields:
        if any(configs[role][field] != configs["reference"][field] for role in ("prefill", "decode")):
            raise ValueError(f"Different {field} in capture configs")
    for field in ("hidden_size", "num_heads", "num_qo_heads", "intermediate_size", "rms_norm_epsilon",
                  "qkv_bias", "has_attn_bias", "base", "rope_base", "scale", "max_position",
                  "max_position_embeddings", "ffn_bias", "tie_word_embeddings"):
        if any(configs[role].get(field) != configs["reference"].get(field) for role in ("prefill", "decode")):
            raise ValueError(f"Different {field} in capture configs")
    if len({config.get("kv_cache_dtype", "auto") for config in configs.values()}) != 1:
        raise ValueError("Reference/P/D must request the same KV format")
    manifest = json.loads(Path(args.prompts).read_text())
    for field in ("model_revision", "tokenizer_revision"):
        if manifest[field] != configs["reference"][field]:
            raise ValueError(f"Prompt manifest has a different {field}")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one prompt case is required")
    names = set()
    for case in cases:
        if (not isinstance(case["name"], str) or not case["name"] or case["name"] in names
                or not isinstance(case["prompt"], str) or not isinstance(case["input_ids"], list)
                or not case["input_ids"] or any(type(x) is not int or not 0 <= x < configs["reference"]["vocab_size"]
                                                for x in case["input_ids"])
                or len(case["input_ids"]) >= configs["reference"]["max_model_length"]):
            raise ValueError("Invalid prompt case, duplicate name or token IDs")
        names.add(case["name"])
    if args.transfer == "pipeline" and any(args.max_tokens == 1 or len(case["input_ids"]) + 1 >= configs["reference"]["max_model_length"] for case in cases):
        raise ValueError("Pipeline requires every case to permit a second token")
    if args.transfer == "pipeline" and (not configs["prefill"].get("enforce_eager", False)
                                       or configs["prefill"].get("enable_chunked_prefill", False)):
        raise ValueError("Pipeline requires whole eager prefill")
    return source, configs, cases


def finite_positive(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return number


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def gpu_selection(value):
    if not value.isdecimal() and not (value.startswith("GPU-") and
            all(character in "0123456789abcdefABCDEF-" for character in value[4:]) and len(value) > 4):
        raise argparse.ArgumentTypeError("expected one physical GPU index or GPU UUID")
    return value


def prefill_gpu_selection(value):
    devices = value.split(",")
    if len(devices) > 2:
        raise argparse.ArgumentTypeError("expected one or two prefill GPU indices or UUIDs")
    for device in devices:
        gpu_selection(device)
    if len(devices) == 2:
        if devices[0].isdecimal() != devices[1].isdecimal():
            raise argparse.ArgumentTypeError("use two indices or two UUIDs for TP2 prefill")
        identifiers = [int(device) if device.isdecimal() else device.lower() for device in devices]
        if identifiers[0] == identifiers[1]:
            raise argparse.ArgumentTypeError("TP2 prefill GPUs must be distinct")
    return value


def numa_selection(value):
    if not value.isdecimal():
        raise argparse.ArgumentTypeError("expected a nonnegative NUMA node number")
    return int(value)


def role_child(args):
    from myvllm.config import load_config
    from myvllm.engine.sequence import Sequence
    from myvllm.sampling_parameters import SamplingParams
    from myvllm.pd.worker import PDWorker
    import inspect

    root = Path(args.output_dir)
    cases = json.loads((root / "cases.json").read_text())
    config = load_config(args.config)
    engine_provenance(args.engine_source, args.expected_sha)
    CaptureEngine = capture_engine_class(args.role, root, cases, args.raw_kv, args.engine_source,
                                         args.expected_sha, args.numa_node)
    if args.role == "reference":
        engine = CaptureEngine(config)
        try:
            save_json(root / "reference.ready.json", {"pid": os.getpid()})
            for index, case in enumerate(cases):
                params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                                        ignore_eos=args.ignore_eos, max_model_length=config["max_model_length"])
                seq = Sequence(case["input_ids"], config["block_size"], params)
                engine.scheduler.add_sequence(seq)
                engine.request_ids[seq.seq_id] = f"case-{index:05d}"
                while not seq.is_finished:
                    engine.step()
                rows = [load_pt(row_path(root, "reference", index, i)) for i in range(1, seq.num_completion_tokens + 1)]
                info = json.loads((root / "reference.engine.json").read_text())
                completed_case(root, "reference", index, case, dict(engine.config), rows,
                               {"role": "reference", "sampling": vars(params), "cache": info["cache"],
                                "engine": {key: info["engine"][key] for key in ("source", "commit", "module")}})
                engine.request_ids.pop(seq.seq_id)
        finally:
            engine.exit()
        return

    async def worker_main():
        opts = {"max_connections": 1, "max_transfer_bytes": args.max_transfer_bytes,
                "request_timeout": args.request_timeout, "engine_factory": CaptureEngine}
        if args.role == "prefill":
            opts["decode_endpoint"] = tuple(json.loads((root / "decode.ready.json").read_text())["endpoint"])
        if args.transfer == "pipeline":
            if "pipeline_frame_bytes" not in inspect.signature(PDWorker).parameters:
                raise RuntimeError("Selected engine does not support pipeline")
            opts["pipeline_frame_bytes"] = args.pipeline_frame_bytes
        worker = PDWorker(args.role, config, **opts)
        endpoint = await worker.start("127.0.0.1", 0)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        loop.add_reader(sys.stdin.fileno(), stop.set)
        save_json(root / f"{args.role}.ready.json", {"pid": os.getpid(), "endpoint": endpoint})
        tasks = [asyncio.create_task(stop.wait()), asyncio.create_task(worker.failed.wait())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if worker.failed.is_set():
                raise RuntimeError(f"{args.role} worker failed: {worker.owner.error}")
        finally:
            loop.remove_reader(sys.stdin.fileno())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await worker.close()
    asyncio.run(worker_main())


def process_group_members(group, started):
    members = []
    if started is None:
        return members
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            if int(tail[2]) == group and int(tail[3]) == group and int(tail[19]) >= started:
                members.append({"pid": int(path.name), "state": tail[0], "start_ticks": int(tail[19])})
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    leader = next((item for item in members if item["pid"] == group), None)
    if leader is not None and leader["start_ticks"] != started:
        return []
    return sorted(members, key=lambda item: item["pid"])


def start_role(args, role, source, root):
    gpu = getattr(args, role + "_gpu")
    node = getattr(args, role + "_numa")
    env = os.environ.copy()
    cache = root / "runtime" / role
    for directory in (cache / "tmp", cache / "hf", cache / "xdg", cache / "triton", cache / "inductor"):
        directory.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1",
               TMPDIR=str(cache / "tmp"), HF_HOME=str(cache / "hf"), XDG_CACHE_HOME=str(cache / "xdg"),
               TRITON_CACHE_DIR=str(cache / "triton"), TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-P", str(SCRIPT), "_child", "--role", role,
               "--engine-source", str(source), "--expected-sha", args.expected_sha,
               "--config", str(Path(getattr(args, role + "_config")).resolve()),
               "--output-dir", str(root), "--max-tokens", str(args.max_tokens),
               "--transfer", args.transfer, "--request-timeout", str(args.request_timeout),
               "--max-transfer-bytes", str(args.max_transfer_bytes),
               "--pipeline-frame-bytes", str(args.pipeline_frame_bytes or 0)]
    if args.ignore_eos:
        command.append("--ignore-eos")
    if args.raw_kv:
        command.append("--raw-kv")
    if node is not None:
        command.extend(["--numa-node", str(node)])
        command[:0] = ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
    log = (root / f"{role}.log").open("xb")
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, cwd=str(source), start_new_session=True)
    finally:
        log.close()
    try:
        stat = Path(f"/proc/{process.pid}/stat").read_text()
        started = int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        started = None
    return {"process": process, "pid": process.pid, "group": process.pid, "start_ticks": started,
            "gpu": gpu, "numa_node": node, "command": command, "signals": [], "errors": [], "remaining": []}


def wait_for_file(path, role, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if path.exists():
            return
        exitcode = role["process"].poll()
        if exitcode is not None:
            raise RuntimeError(f"Role {path.name} exited {exitcode} before publishing")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path.name}")
        time.sleep(.05)


def stop_roles(roles, args):
    errors = []
    for name, role in reversed(list(roles.items())):
        process = role["process"]
        if process.poll() is None and process.stdin:
            try:
                process.stdin.write(b"stop\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                role["errors"].append(f"normal stop: {exc!r}")
    for name, role in reversed(list(roles.items())):
        process = role["process"]
        try:
            for sig, timeout in ((None, args.normal_exit_timeout),
                                 (signal.SIGTERM, args.term_timeout), (signal.SIGKILL, args.kill_timeout)):
                members = process_group_members(role["group"], role["start_ticks"])
                live = [member for member in members if member["state"] != "Z"]
                if not live and process.poll() is not None:
                    break
                if sig is not None:
                    if live:
                        os.killpg(role["group"], sig)
                        role["signals"].append(sig.name)
                    elif role["start_ticks"] is None and process.poll() is None:
                        process.send_signal(sig)
                        role["signals"].append(sig.name + " (direct child)")
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    process.poll()
                    members = process_group_members(role["group"], role["start_ticks"])
                    if process.returncode is not None and not any(m["state"] != "Z" for m in members):
                        break
                    time.sleep(.05)
            role["remaining"] = process_group_members(role["group"], role["start_ticks"])
            if process.poll() is None or any(m["state"] != "Z" for m in role["remaining"]):
                role["errors"].append("process group remains after bounded cleanup")
            if role["start_ticks"] is None:
                role["errors"].append("process group identity unavailable; descendants could not be verified")
            role["exitcode"] = process.poll()
            if role["exitcode"] != 0:
                role["errors"].append(f"exit code {role['exitcode']}")
        except BaseException as exc:
            role["errors"].append(f"cleanup: {exc!r}")
        finally:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError as exc:
                    role["errors"].append(f"close stdin: {exc!r}")
        errors.extend(f"{name}: {error}" for error in role["errors"])
    return errors


def public_roles(roles):
    return {name: {key: value for key, value in role.items() if key != "process"}
            for name, role in roles.items()}


def update_manifest(root, manifest, roles):
    manifest["roles"] = public_roles(roles)
    manifest["visible_artifacts"] = sorted(str(path.relative_to(root)) for path in root.rglob("*")
                                           if path.is_file() and "runtime" not in path.parts)
    temp = root / f".manifest.{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp, root / "manifest.json")


def compare_role_metadata(root):
    info = {role: json.loads((root / f"{role}.engine.json").read_text())
            for role in ("reference", "prefill", "decode")}
    for role in ("prefill", "decode"):
        if info[role]["tokenizer"] != info["reference"]["tokenizer"]:
            raise ValueError(f"Tokenizer or prompt encoding/decoding differs for {role}")
        for field in ("requested", "dtype", "scale_dtype"):
            if info[role]["cache"][field] != info["reference"]["cache"][field]:
                raise ValueError(f"Actual KV cache {field} differs for {role}")
    return info


async def run_requests(args, root, cases, config, endpoints, info, roles, manifest):
    from myvllm.pd.client import PDClient
    from myvllm.sampling_parameters import SamplingParams

    if args.transfer == "pipeline" and not all(hasattr(PDClient, method)
            for method in ("prepare_p", "start_pipeline", "handoff_pipeline")):
        raise RuntimeError("Selected engine does not support pipeline")
    pi, di = await asyncio.gather(PDClient.info(endpoints["prefill"], timeout=args.request_timeout),
                                  PDClient.info(endpoints["decode"], timeout=args.request_timeout))
    if pi["role"] != "prefill" or di["role"] != "decode" or pi["model"] != di["model"]:
        raise ValueError("P/D worker roles or model specifications differ")
    for index, case in enumerate(cases):
        p = d = None
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens,
                                ignore_eos=args.ignore_eos, max_model_length=config["max_model_length"])
        rows = []
        client_events = []
        try:
            async with asyncio.timeout(args.request_timeout):
                d = await PDClient.open_d(endpoints["decode"], lifetime=args.request_timeout)
                open_p = PDClient.prepare_p if args.transfer == "pipeline" else PDClient.open_p
                p = await open_p(endpoints["prefill"], d.session_id, f"case-{index:05d}",
                                 case["input_ids"], params, lifetime=args.request_timeout)
                if args.transfer == "pipeline":
                    await PDClient.start_pipeline(p, d)
                first = await p.next_token()
                client_events.append(first)
                rows.append(load_pt(row_path(root, "prefill", index, 1)))
                if not first["finished"]:
                    metadata = await p.describe()
                    if args.transfer == "pipeline":
                        await PDClient.handoff_pipeline(p, d, metadata)
                    else:
                        await PDClient.handoff(p, d, metadata)
                    while True:
                        event = await d.next_token()
                        client_events.append(event)
                        rows.append(load_pt(row_path(root, "decode", index, event["completion_tokens"])))
                        if event["finished"]:
                            break
                for number, (row, event) in enumerate(zip(rows, client_events), 1):
                    if (row["request_id"] != f"case-{index:05d}" or row["completion_tokens"] != number
                            or event["completion_tokens"] != number or row["token"] != event["token"]
                            or event["prompt_tokens"] != len(case["input_ids"])
                            or row["finished"] != event["finished"]):
                        raise RuntimeError(f"Callback and client events disagree for {case['name']}")
                handoff = None
                if args.raw_kv:
                    source_kv = rows[0]["kv"]
                    reference_kv = load_pt(row_path(root, "reference", index, 1))["kv"]
                    kv_report = {"reference_to_prefill": compare_kv(reference_kv, source_kv),
                                 "prefill_to_restored_decode": None}
                    if not first["finished"]:
                        dest_kv = load_pt(restored_path(root, index))
                        kv_report["prefill_to_restored_decode"] = compare_kv(source_kv, dest_kv)
                        handoff = kv_report["prefill_to_restored_decode"]["equal"]
                    save_json(root / f"kv-{index:05d}.json", kv_report)
                    if handoff is False:
                        raise RuntimeError(f"P/D effective KV bytes differ for {case['name']}")
                completed_case(root, "candidate", index, case, info["prefill"]["config"], rows,
                               {"role": "prefill+decode", "sampling": vars(params),
                                "transfer": args.transfer, "kv_handoff_equal": handoff,
                                "cache": info["prefill"]["cache"],
                                "decode_cache": info["decode"]["cache"],
                                "engine": {role: {key: info[role]["engine"][key]
                                                  for key in ("source", "commit", "module")}
                                           for role in ("prefill", "decode")},
                                "client_events": client_events})
                manifest["completed_cases"].append({"index": index, "name": case["name"],
                                                    "tokens": len(rows), "kv_handoff_equal": handoff})
                update_manifest(root, manifest, roles)
        finally:
            original_error = sys.exception()
            failures = []
            for connection in (p, d):
                if connection is not None:
                    try:
                        await asyncio.wait_for(connection.close(), timeout=args.normal_exit_timeout)
                    except Exception as exc:
                        failures.append(repr(exc))
            if failures:
                if original_error is not None:
                    original_error.add_note(f"Connection cleanup also failed: {failures}")
                else:
                    raise RuntimeError(f"Connection cleanup failed: {failures}")


def capture(args):
    source, configs, cases = read_inputs(args)
    if args.transfer == "pipeline" and not args.pipeline_frame_bytes:
        raise ValueError("--pipeline-frame-bytes is required for pipeline")
    if args.transfer == "bulk" and args.pipeline_frame_bytes:
        raise ValueError("--pipeline-frame-bytes applies only to pipeline")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "partial").mkdir()
    save_json(root / "cases.json", cases)
    save_json(root / "inputs.json", {
        "configs": configs, "config_sha256": {role: sha256(getattr(args, role + "_config")) for role in configs},
        "prompts_sha256": sha256(args.prompts), "expected_engine_sha": args.expected_sha,
        "harness": harness_provenance(), "arguments": vars(args),
    })
    roles = {}
    manifest = {"status": "running", "completed_cases": [], "error": None, "cleanup_errors": [],
                "limitations": ["capture only; no approved gate implied", "unflushed data and power-fail durability are not guaranteed",
                                "controller SIGKILL or host failure can leave its processes running"]}
    update_manifest(root, manifest, roles)
    error = None
    interrupted = {"signal": None, "deferred": False}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(sig, frame):
        if interrupted["signal"] is None:
            interrupted["signal"] = sig
        if not interrupted["deferred"]:
            interrupted["deferred"] = True
            raise InterruptedError(f"Controller received {signal.Signals(sig).name}")

    def start_registered(role):
        # Do not interrupt between Popen and registering the owned process group.
        interrupted["deferred"] = True
        roles[role] = start_role(args, role, source, root)
        interrupted["deferred"] = False
        if interrupted["signal"] is not None:
            raise InterruptedError(f"Controller received {signal.Signals(interrupted['signal']).name}")

    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        start_registered("reference")
        update_manifest(root, manifest, roles)
        wait_for_file(root / "reference.ready.json", roles["reference"], args.startup_timeout)
        for index in range(len(cases)):
            wait_for_file(root / f"reference-{index:05d}.pt", roles["reference"], args.request_timeout)
        reference = roles["reference"]["process"]
        try:
            code = reference.wait(timeout=args.normal_exit_timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Reference completed but did not exit in time") from exc
        remaining = process_group_members(roles["reference"]["group"], roles["reference"]["start_ticks"])
        if code != 0 or any(member["state"] != "Z" for member in remaining):
            raise RuntimeError(f"Reference did not fully exit: {code}, {remaining}")
        start_registered("decode")
        update_manifest(root, manifest, roles)
        wait_for_file(root / "decode.ready.json", roles["decode"], args.startup_timeout)
        start_registered("prefill")
        update_manifest(root, manifest, roles)
        wait_for_file(root / "prefill.ready.json", roles["prefill"], args.startup_timeout)
        info = compare_role_metadata(root)
        endpoints = {role: tuple(json.loads((root / f"{role}.ready.json").read_text())["endpoint"])
                     for role in ("prefill", "decode")}
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(source / "src"))
        engine_provenance(source, args.expected_sha)
        asyncio.run(run_requests(args, root, cases, configs["reference"], endpoints, info, roles, manifest))
    except BaseException as exc:
        error = exc
        manifest["error"] = "".join(traceback.format_exception(exc))
    finally:
        interrupted["deferred"] = True
        try:
            manifest["cleanup_errors"] = stop_roles(roles, args)
            if interrupted["signal"] is not None and error is None:
                error = InterruptedError(f"Controller received {signal.Signals(interrupted['signal']).name}")
                manifest["error"] = str(error)
            manifest["controller_signal"] = (signal.Signals(interrupted["signal"]).name
                                             if interrupted["signal"] is not None else None)
            manifest["status"] = "failed" if error or manifest["cleanup_errors"] else "complete"
            update_manifest(root, manifest, roles)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    if error:
        raise RuntimeError(f"Capture failed; inspect {root / 'manifest.json'}") from error
    if manifest["cleanup_errors"]:
        raise RuntimeError(f"Cleanup failed; inspect {root / 'manifest.json'}")


def read_freeze(path):
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    if (not isinstance(data.get("version"), str) or not data["version"]
            or data.get("approved") is not True
            or not isinstance(data.get("approval_declaration"), str) or not data["approval_declaration"].strip()):
        raise ValueError("Gate freeze requires version, approved=true and approval_declaration")
    for key in ("atol", "rtol"):
        value = data[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Gate {key} must be finite and nonnegative")
    return {"version": data["version"], "approved": True,
            "approval_declaration": data["approval_declaration"], "atol": data["atol"],
            "rtol": data["rtol"], "sha256": sha256(path)}


def analyze_data(baseline, candidate, freeze):
    import torch

    for field in ("model_name_or_path", "model_type", "model_revision", "tokenizer_revision", "dtype", "eos",
                  "max_model_length"):
        if baseline["metadata"]["config"][field] != candidate["metadata"]["config"][field]:
            raise ValueError(f"Different {field}")
    if baseline["metadata"]["config"].get("kv_cache_dtype", "auto") != candidate["metadata"]["config"].get("kv_cache_dtype", "auto"):
        raise ValueError("Different KV cache formats")
    if baseline["metadata"].get("sampling") != candidate["metadata"].get("sampling"):
        raise ValueError("Different sampling settings")
    reference_cache = baseline["metadata"].get("cache")
    candidate_cache = candidate["metadata"].get("cache")
    if freeze is not None and (reference_cache is None or candidate_cache is None):
        raise ValueError("Frozen gate requires actual allocated KV cache metadata")
    if reference_cache is not None and candidate_cache is not None:
        for field in ("requested", "dtype", "scale_dtype"):
            if reference_cache[field] != candidate_cache[field]:
                raise ValueError(f"Different actual KV cache {field}")
    if [case["name"] for case in baseline["cases"]] != [case["name"] for case in candidate["cases"]] or not baseline["cases"]:
        raise ValueError("Runs must contain the same nonempty ordered cases")
    reports = []
    for before, after in zip(baseline["cases"], candidate["cases"]):
        if before["input_ids"] != after["input_ids"]:
            raise ValueError(f"Different input tokens for {before['name']}")
        old, new = before["token_ids"], after["token_ids"]
        a, b = before["logits"], after["logits"]
        if a.ndim != 2 or b.ndim != 2 or len(a) != len(old) or len(b) != len(new) or a.shape[1] != b.shape[1]:
            raise ValueError(f"Expected one equal-width logits row per token: {before['name']}")
        common = min(len(old), len(new))
        mismatch = next((i for i in range(common) if old[i] != new[i]), None)
        if mismatch is None and len(old) != len(new):
            mismatch = common
        scored = min(common, mismatch + 1) if mismatch is not None else common
        if scored == 0:
            raise ValueError(f"No comparable positions: {before['name']}")
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        ref, cand = a[:scored].double(), b[:scored].double()
        error = (cand - ref).abs()
        finite_scored = bool(torch.isfinite(error).all())
        greedy = (bool(torch.equal(a.argmax(dim=1).cpu(), torch.tensor(old, dtype=torch.long)))
                  and bool(torch.equal(b.argmax(dim=1).cpu(), torch.tensor(new, dtype=torch.long)))) if finite else False
        gate_passed = None
        failures = None
        if freeze is not None:
            passed = error <= freeze["atol"] + freeze["rtol"] * ref.abs()
            failures = int((~passed).sum().item())
            gate_passed = finite and finite_scored and greedy and old == new and failures == 0
        reports.append({"name": before["name"], "tokens_identical": old == new,
                        "greedy_matches_logits": greedy, "finite": finite, "first_divergence": mismatch,
                        "same_prefix_logit_positions": scored, "reference_tokens": len(old),
                        "candidate_tokens": len(new), "max_abs_error": error.max().item() if finite_scored else None,
                        "mean_abs_error": error.mean().item() if finite_scored else None,
                        "gate_failed_elements": failures, "gate_passed": gate_passed})
    return {"status": "gated" if freeze else "capture_only", "freeze": freeze,
            "tokens_identical": all(item["tokens_identical"] for item in reports),
            "all_finite": all(item["finite"] for item in reports),
            "greedy_matches_logits": all(item["greedy_matches_logits"] for item in reports),
            "gate_passed": all(item["gate_passed"] for item in reports) if freeze else None,
            "cases": reports}


def analyze(args):
    report = {"baseline": str(Path(args.baseline).resolve()), "candidate": str(Path(args.candidate).resolve()),
              "baseline_sha256": sha256(args.baseline), "candidate_sha256": sha256(args.candidate)}
    try:
        freeze = read_freeze(args.gate_freeze)
        report.update(analyze_data(load_pt(args.baseline), load_pt(args.candidate), freeze))
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", gate_passed=False)
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if (report["status"] == "error" or not report["tokens_identical"] or not report["all_finite"]
            or not report["greedy_matches_logits"] or report["gate_passed"] is False):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--engine-source", required=True)
    cap.add_argument("--expected-sha", required=True)
    for role in ("reference", "prefill", "decode"):
        cap.add_argument(f"--{role}-config", required=True)
        cap.add_argument(f"--{role}-gpu", type=prefill_gpu_selection if role == "prefill" else gpu_selection,
                         required=True)
        cap.add_argument(f"--{role}-numa", type=numa_selection)
    cap.add_argument("--prompts", required=True)
    cap.add_argument("--max-tokens", type=positive_int, required=True)
    cap.add_argument("--ignore-eos", action="store_true")
    cap.add_argument("--transfer", choices=("bulk", "pipeline"), required=True)
    cap.add_argument("--pipeline-frame-bytes", type=positive_int)
    cap.add_argument("--max-transfer-bytes", type=positive_int, required=True)
    cap.add_argument("--output-dir", required=True)
    cap.add_argument("--raw-kv", action="store_true")
    for name in ("startup-timeout", "request-timeout", "normal-exit-timeout", "term-timeout", "kill-timeout"):
        cap.add_argument(f"--{name}", type=finite_positive, required=True)
    check = sub.add_parser("analyze")
    check.add_argument("--baseline", required=True)
    check.add_argument("--candidate", required=True)
    check.add_argument("--gate-freeze")
    check.add_argument("--output", required=True)
    child = sub.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--role", choices=("reference", "prefill", "decode"), required=True)
    for name in ("engine-source", "expected-sha", "config", "output-dir"):
        child.add_argument(f"--{name}", required=True)
    child.add_argument("--max-tokens", type=positive_int, required=True)
    child.add_argument("--ignore-eos", action="store_true")
    child.add_argument("--raw-kv", action="store_true")
    child.add_argument("--transfer", choices=("bulk", "pipeline"), required=True)
    child.add_argument("--request-timeout", type=finite_positive, required=True)
    child.add_argument("--max-transfer-bytes", type=positive_int, required=True)
    child.add_argument("--pipeline-frame-bytes", type=int, required=True)
    child.add_argument("--numa-node", type=numa_selection)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        role_child(args)


if __name__ == "__main__":
    main()
