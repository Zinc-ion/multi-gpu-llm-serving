#!/usr/bin/env python3
"""Capture the original tiny Qwen3 TP engine without changing its source.

This is a diagnostic, not a benchmark. Hooks copy real tensors to CPU and can
change synchronization or compilation boundaries. The original captures remain
separate and their strict greedy gate is never replaced by the new run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open


ORIGINAL_SHA = "c7bf6d331b42b1a7e078f6f728126218dc774d51"
PROMPT = [1, 9, 4, 6, 12]
OUTPUT_TOKENS = 3
MAX_RANK_EVENTS = 4000
TOLERANCES = {
    "float32": {"rtol": 2e-4, "atol": 2e-5},
    "bfloat16": {"rtol": 5e-2, "atol": 2e-2},
}
TINY_CONFIG = {
    "model_type": "Qwen3-0.6B", "vocab_size": 67, "hidden_size": 64,
    "num_heads": 4, "num_kv_heads": 2, "head_dim": 16,
    "intermediate_size": 128, "num_layers": 2, "block_size": 2,
    "enforce_eager": True, "qkv_bias": False, "ffn_bias": False,
    "tie_word_embeddings": False, "max_num_sequences": 1,
    "max_num_batched_tokens": 16, "max_model_length": 16,
    "max_cached_blocks": 16, "gpu_memory_utilization": 0.1,
    "scale": 1.0, "base": 10000, "max_position": 64,
    "rms_norm_epsilon": 1e-5, "eos": [66],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_torch(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("xb") as stream:
        torch.save(data, stream)
    os.replace(temp, path)


def tensor_stats(tensor):
    tensor = tensor.detach().cpu()
    finite = torch.isfinite(tensor)
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "numel": tensor.numel(), "finite": int(finite.sum().item()),
            "nonfinite": int((~finite).sum().item())}


def tensor_leaves(value, prefix):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, f"{prefix}.{key}")


def layout_for(module, phase, field, rank, world_size):
    """Classify Qwen3 boundaries; unknown layouts remain raw-only."""
    if module == "model.embed_tokens":
        return "replicated"
    if module == "lm_head":
        if phase == "output" and rank and world_size > 1:
            return "vocab_shard"
        return "replicated"
    if ".self_attn.qkv_projection" in module:
        return "qkv_packed" if phase == "output" else "replicated"
    if ".mlp.gate_up" in module:
        return "gate_up_packed" if phase == "output" else "replicated"
    if ".mlp.activation" in module:
        return "gate_up_packed" if phase == "input" else "feature_shard"
    if ".self_attn.o_proj" in module or ".mlp.down_proj" in module:
        return "feature_shard" if phase == "input" else "replicated"
    if module.endswith((".self_attn.q_norm", ".self_attn.k_norm")):
        return "head_shard"
    if module.endswith(".self_attn.rotary_emb"):
        if phase == "input" and field in ("args.0", "kwargs.positions"):
            return "replicated"
        return "head_shard"
    if module.endswith(".self_attn.attention"):
        return "head_shard" if phase == "input" else "feature_shard"
    if module in ("", "model", "model.norm") or ".layers." in module:
        return "replicated"
    return "unknown"


class Capture:
    """Write each tensor before continuing, so a killed process leaves evidence."""

    def __init__(self, model, output, rank, world_size, max_bytes):
        self.model = model
        self.output = Path(output)
        self.rank = rank
        self.world_size = world_size
        self.limit = max_bytes
        self.bytes = 0
        self.sequence = 0
        self.step = -1
        self.calls = defaultdict(int)
        self.stack = []
        self.handles = []
        self.original_collectives = {}
        self.output.mkdir(parents=True, exist_ok=False)

    def write(self, kind, module, call, phase, field, value, layout):
        for key, tensor in tensor_leaves(value, field):
            size = tensor.numel() * tensor.element_size()
            if self.bytes + size > self.limit:
                raise RuntimeError(f"rank {self.rank} tensor budget exceeded: {self.limit}")
            if self.sequence >= MAX_RANK_EVENTS:
                raise RuntimeError(f"rank {self.rank} event budget exceeded: {MAX_RANK_EVENTS}")
            cpu = tensor.detach().cpu().clone()
            self.sequence += 1
            entry = {"sequence": self.sequence, "step": self.step, "kind": kind,
                     "module": module, "call": call, "phase": phase, "field": key,
                     "layout": layout_for(module, phase, key, self.rank, self.world_size)
                     if layout == "module" else layout,
                     "rank": self.rank, "world_size": self.world_size,
                     "stats": tensor_stats(cpu)}
            name = f"{self.sequence:06d}.pt"
            atomic_torch(self.output / name, {"metadata": entry, "tensor": cpu})
            with (self.output / "index.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**entry, "file": name}, sort_keys=True) + "\n")
            self.bytes += size

    def save_weights(self):
        for name, param in self.model.named_parameters(remove_duplicate=False):
            self.write("weight", name, 0, "loaded", "parameter", param, "weight")

    def _pre(self, name):
        def hook(module, args, kwargs):
            if name == "":
                self.step += 1
            self.calls[name] += 1
            call = self.calls[name]
            self.stack.append((name, call))
            self.write("module", name, call, "input", "args", args, "module")
            self.write("module", name, call, "input", "kwargs", kwargs, "module")
            if name == "":
                from myvllm.utils import get_context
                ctx = get_context()
                atomic_json(self.output / f"step-{self.step}-context.json",
                            {"step": self.step, "is_prefill": ctx.is_prefill})
                for key in ("positions", "slot_mapping", "block_tables", "context_lens", "cu_seqlens_q", "cu_seqlens_k"):
                    self.write("context", name, call, "input", key, getattr(ctx, key), "context")
        return hook

    def _post(self, name):
        def hook(module, args, kwargs, output):
            actual_name, call = self.stack.pop()
            if actual_name != name:
                raise RuntimeError(f"module hook stack mismatch: {actual_name} != {name}")
            self.write("module", name, call, "output", "output", output, "module")
        return hook

    def _collective(self, name, original):
        def wrapped(tensor, *args, **kwargs):
            module, call = self.stack[-1]
            before = "partial_sum" if name == "all_reduce" else "vocab_shard"
            self.write("collective", module, call, name + "_pre", "tensor", tensor, before)
            result = original(tensor, *args, **kwargs)
            if kwargs.get("async_op", False):
                raise RuntimeError("asynchronous collectives are outside this diagnostic")
            if name == "all_reduce":
                self.write("collective", module, call, name + "_post", "tensor", tensor, "replicated")
            else:
                gather_list = kwargs.get("gather_list", args[0] if args else None)
                if gather_list is not None:
                    self.write("collective", module, call, name + "_post", "gather", gather_list, "vocab_shard")
            return result
        return wrapped

    def __enter__(self):
        self.save_weights()
        for name, module in self.model.named_modules():
            self.handles.append(module.register_forward_pre_hook(self._pre(name), with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._post(name), with_kwargs=True))
        for name in ("all_reduce", "gather"):
            original = getattr(dist, name)
            self.original_collectives[name] = original
            setattr(dist, name, self._collective(name, original))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for name, original in self.original_collectives.items():
            setattr(dist, name, original)


def load_events(directory):
    directory = Path(directory)
    index = directory / "index.jsonl"
    if not index.exists():
        return []
    entries = [json.loads(line) for line in index.read_text().splitlines()]
    return [(entry, torch.load(directory / entry["file"], map_location="cpu", weights_only=True)["tensor"])
            for entry in entries]


def slice_reference(reference, actual, layout, rank, world_size, config):
    """Select this rank's actual global Q/K/V, feature, head, or vocab slice."""
    if layout == "replicated":
        return reference, actual, 0
    if layout in ("feature_shard", "head_shard"):
        axis = -1 if layout == "feature_shard" else -2
        if reference.shape[axis] % world_size:
            raise ValueError("sharded reference dimension is not divisible")
        width = reference.shape[axis] // world_size
        return reference.narrow(axis, rank * width, width), actual, 0
    if layout in ("qkv_packed", "gate_up_packed"):
        if layout == "qkv_packed":
            sizes = [config["num_heads"] * config["head_dim"],
                     config["num_kv_heads"] * config["head_dim"]]
            sizes.append(sizes[-1])
        else:
            sizes = [config["intermediate_size"]] * 2
        if any(size % world_size for size in sizes):
            raise ValueError("packed segment is not divisible")
        chunks = reference.split(sizes, dim=-1)
        selected = [chunk.narrow(-1, rank * (size // world_size), size // world_size)
                    for chunk, size in zip(chunks, sizes)]
        return torch.cat(selected, dim=-1), actual, 0
    if layout == "vocab_shard":
        width = (config["vocab_size"] + world_size - 1) // world_size
        start = min(rank * width, config["vocab_size"])
        count = min(width, config["vocab_size"] - start)
        return reference.narrow(-1, start, count), actual.narrow(-1, 0, count), width - count
    raise ValueError(f"layout cannot be compared: {layout}")


def compare_tensors(actual, reference):
    if actual.shape != reference.shape:
        return {"comparable": False, "reason": "shape mismatch",
                "actual_shape": list(actual.shape), "reference_shape": list(reference.shape)}
    if actual.dtype != reference.dtype:
        return {"comparable": False, "reason": "dtype mismatch",
                "actual_dtype": str(actual.dtype), "reference_dtype": str(reference.dtype)}
    a, b = actual.to(torch.float64), reference.to(torch.float64)
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite] - b[finite]).abs()
    return {"comparable": True, "equal": bool(torch.equal(actual, reference)),
            "different_elements": int((actual != reference).sum().item()),
            "finite_pairs": int(finite.sum().item()),
            "max_abs": float(delta.max().item()) if delta.numel() else None,
            "mean_abs": float(delta.mean().item()) if delta.numel() else None}


def original_gate(reference, tp1, tp2, rank1, rank2):
    """Recompute the original token gate and same-prefix logits separately."""
    tolerance = TOLERANCES[reference["dtype"]]
    tokens = reference["tokens"]
    result = {"reference_tokens": tokens, "tp1_tokens": tp1["tokens"], "tp2_tokens": tp2["tokens"],
              "strict_greedy_pass": tp1["tokens"] == tokens and tp2["tokens"] == tokens,
              "tolerances": tolerance, "same_prefix_logits": {}}
    expected_prefixes = [tokens[:len(PROMPT) + i] for i in range(OUTPUT_TOKENS)]
    comparisons = (
        ("tp1_vs_hf", tp1, rank1["logits"], expected_prefixes, reference["logits"]),
        ("tp2_vs_hf", tp2, rank2["logits"], expected_prefixes, reference["logits"]),
        ("tp2_vs_tp1", tp2, rank2["logits"], tp1["prefixes"], rank1["logits"]),
    )
    for name, run, actual_logits, reference_prefixes, reference_logits in comparisons:
        rows = []
        for index, (actual, expected) in enumerate(zip(actual_logits, reference_logits)):
            same = (index < len(run["prefixes"]) and index < len(reference_prefixes)
                    and run["prefixes"][index] == reference_prefixes[index])
            comparable = same and actual.shape == expected.shape and actual.dtype == expected.dtype
            close = bool(torch.allclose(actual, expected, **tolerance)) if comparable else None
            rows.append({"step": index, "same_prefix": same, "within_original_tolerance": close,
                         "comparison": compare_tensors(actual, expected) if same else None})
        result["same_prefix_logits"][name] = rows
    result["original_test_pass"] = (result["strict_greedy_pass"] and
        all(len(rows) == OUTPUT_TOKENS and all(row["within_original_tolerance"] is True for row in rows)
            for rows in result["same_prefix_logits"].values()))
    return result


def verify_weights(rank_dir, checkpoint, config, rank, world_size):
    weights = {}
    for shard in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in weights:
                    raise ValueError(f"duplicate checkpoint tensor: {name}")
                weights[name] = handle.get_tensor(name)
    captured = {entry["module"]: value for entry, value in load_events(rank_dir) if entry["kind"] == "weight"}
    results = {}
    consumed = set()
    for name, actual in captured.items():
        parts = None
        if ".self_attn.qkv_projection." in name:
            parts = [name.replace(".qkv_projection.", f".{part}_proj.") for part in ("q", "k", "v")]
        elif ".mlp.gate_up." in name:
            parts = [name.replace(".gate_up.", f".{part}_proj.") for part in ("gate", "up")]
        if parts is not None:
            pieces = []
            for part in parts:
                source = weights[part]
                width = source.shape[0] // world_size
                pieces.append(source.narrow(0, rank * width, width))
            expected = torch.cat(pieces, dim=0)
            consumed.update(parts)
            semantics = "packed output shards in Q/K/V or gate/up order"
        else:
            source = weights[name]
            consumed.add(name)
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                width = (config["vocab_size"] + world_size - 1) // world_size
                start = min(rank * width, config["vocab_size"])
                length = min(width, config["vocab_size"] - start)
                expected = torch.zeros((width, *source.shape[1:]), dtype=source.dtype)
                expected[:length].copy_(source.narrow(0, start, length))
                semantics = "vocabulary shard with zero padding"
            elif name.endswith("weight") and (".self_attn.o_proj." in name or ".mlp.down_proj." in name):
                width = source.shape[1] // world_size
                expected = source.narrow(1, rank * width, width)
                semantics = "input feature shard"
            else:
                expected = source
                semantics = "replicated"
        expected = expected.to(dtype=actual.dtype)
        results[name] = {"semantics": semantics, "actual": tensor_stats(actual),
                         "checkpoint": parts if parts is not None else [name],
                         "dtype_matches_config": actual.dtype == getattr(torch, config["dtype"]),
                         **compare_tensors(actual, expected)}
    return {"parameters": results, "checkpoint_tensors_not_compared": sorted(set(weights) - consumed),
            "all_equal": bool(results) and all(item.get("equal") is True and item["dtype_matches_config"]
                                               for item in results.values())
                         and consumed == set(weights)}


def event_key(entry):
    return (entry["step"], entry["kind"], entry["module"], entry["call"], entry["phase"], entry["field"])


def compare_events(tp1_dir, tp2_dirs, prefixes1, prefixes2, config):
    reference_events = load_events(tp1_dir)
    rank_events = [dict((event_key(entry), (entry, tensor)) for entry, tensor in load_events(path))
                   for path in tp2_dirs]
    rows, first = [], None
    for entry, reference in reference_events:
        if entry["kind"] != "module" or entry["step"] < 0:
            continue
        step = entry["step"]
        for rank, events in enumerate(rank_events):
            key = event_key(entry)
            other = events.get(key)
            base = {"step": step, "rank": rank, "module": entry["module"],
                    "call": entry["call"], "phase": entry["phase"], "field": entry["field"],
                    "tp1_sequence": entry["sequence"]}
            if step >= len(prefixes1) or step >= len(prefixes2) or prefixes1[step] != prefixes2[step]:
                rows.append({**base, "comparable": False, "reason": "different or missing prefix"})
                continue
            if other is None:
                rows.append({**base, "comparable": False, "reason": "event missing"})
                continue
            other_entry, actual = other
            layout = other_entry["layout"]
            if layout == "unknown":
                rows.append({**base, "comparable": False, "reason": "unclassified layout"})
                continue
            try:
                aligned, actual, padding = slice_reference(reference, actual, layout, rank, len(tp2_dirs), config)
                comparison = compare_tensors(actual, aligned)
            except (RuntimeError, ValueError, IndexError) as exc:
                comparison = {"comparable": False, "reason": str(exc)}
                padding = 0
            row = {**base, "tp2_sequence": other_entry["sequence"], "layout": layout,
                   "excluded_vocab_padding": padding, **comparison}
            rows.append(row)
            if first is None and comparison.get("equal") is False:
                first = row
    return {"first_observed_exact_difference": first, "timeline": rows,
            "limits": ["Exact inequality is descriptive, not a root cause or a numerical gate.",
                       "Row-parallel pre-reduction tensors are partial sums, not feature slices.",
                       "Collective pre/post tensors are retained raw; post-reduction rank agreement is reported separately.",
                       "Comparisons stop when prefixes differ; missing or incompatible events remain visible."]}


def compare_collective_ranks(rank_dirs):
    events = [dict((event_key(entry), (entry, value)) for entry, value in load_events(path)
                   if entry["kind"] == "collective" and entry["phase"] == "all_reduce_post")
              for path in rank_dirs]
    rows = []
    if len(events) < 2:
        return rows
    for key, (entry, value) in events[0].items():
        for rank in range(1, len(events)):
            other = events[rank].get(key)
            comparison = compare_tensors(other[1], value) if other else {"comparable": False, "reason": "event missing"}
            rows.append({"step": entry["step"], "module": entry["module"], "call": entry["call"],
                         "rank": rank, **comparison})
    return rows


def compare_gather_ranks(rank_dirs):
    events = [load_events(path) for path in rank_dirs]
    sends = [{(entry["step"], entry["module"], entry["call"]): tensor
              for entry, tensor in rank_events
              if entry["kind"] == "collective" and entry["phase"] == "gather_pre"}
             for rank_events in events]
    rows = []
    for entry, received in events[0]:
        if entry["kind"] != "collective" or entry["phase"] != "gather_post":
            continue
        rank = int(entry["field"].split(".")[-1])
        key = (entry["step"], entry["module"], entry["call"])
        sent = sends[rank].get(key)
        comparison = compare_tensors(received, sent) if sent is not None else {
            "comparable": False, "reason": "rank send missing"}
        rows.append({"step": entry["step"], "module": entry["module"],
                     "call": entry["call"], "rank": rank, **comparison})
    return rows


def git_source(module_file):
    module_file = Path(module_file).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(module_file.parent), *args], text=True).strip()
    root = Path(git("rev-parse", "--show-toplevel"))
    return {"module": str(module_file), "module_sha256": sha256(module_file),
            "git_root": str(root), "git_sha": git("rev-parse", "HEAD"),
            "git_status_porcelain": git("status", "--porcelain", "--untracked-files=all")}


def import_engine(source, expected_sha):
    source = Path(source).resolve(strict=True)
    sys.path.insert(0, str(source))
    import myvllm.engine.llm_engine as engine_module
    actual = Path(engine_module.__file__).resolve()
    if not actual.is_relative_to(source):
        raise RuntimeError(f"imported engine {actual} is outside supplied source {source}")
    provenance = git_source(actual)
    if provenance["git_sha"] != expected_sha:
        raise RuntimeError(f"engine Git SHA {provenance['git_sha']} != {expected_sha}")
    return engine_module, provenance


def imported_sources(source):
    source = Path(source).resolve()
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if (name == "myvllm" or name.startswith("myvllm.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise RuntimeError(f"imported {name} outside supplied source: {path}")
            modules[name] = {"path": str(path), "sha256": sha256(path)}
    return modules


def observe_runner_calls(runner, directory):
    original = runner.run
    call = 0

    def run(sequences, is_prefill):
        nonlocal call
        result = original(sequences, is_prefill)
        call += 1
        atomic_json(directory / f"runner-call-{call}.json",
                    {"call": call, "is_prefill": is_prefill, "sequences": len(sequences)})
        return result

    runner.run = run


def wait_for_path(path, timeout):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"waiting for {path} exceeded {timeout}s")
        time.sleep(0.01)


def tracked_shared_memory(directory, factory):
    def create(*args, **kwargs):
        if args or kwargs != {"create": True, "size": 2**20}:
            raise ValueError("unexpected original engine shared memory request")
        identity = json.loads((Path(directory) / "owned-shm.json").read_text())
        if os.environ.get("MINIVLLM_P3A_SHM_NAME") != identity["name"]:
            raise ValueError("shared memory reservation does not match this lane")
        shared = factory(name=identity["name"])
        stat = os.fstat(shared._fd)
        if (stat.st_dev, stat.st_ino) != (identity["device"], identity["inode"]):
            from multiprocessing import resource_tracker
            shared.close()
            resource_tracker.unregister(shared._name, "shared_memory")
            raise RuntimeError("shared memory reservation identity changed")
        return shared
    return create


def observed_worker(config, rank, event):
    from myvllm.engine.model_runner import ModelRunner
    import myvllm.engine.llm_engine as engine_module
    runner = ModelRunner(config, rank, event)
    directory = Path(config["_diagnostic_output"]) / f"rank-{rank}"
    with Capture(runner.model, directory, rank, runner.world_size, config["_diagnostic_max_bytes"]) as capture:
        observe_runner_calls(runner, directory)
        atomic_json(directory / "ready.json", {"rank": rank, "pid": os.getpid(),
                    "device": str(next(runner.model.parameters()).device),
                    "kv_cache_shape": list(runner.kv_cache.shape),
                    "kv_cache_dtype": str(runner.kv_cache.dtype),
                    "engine_source": git_source(engine_module.__file__),
                    "imported_modules": imported_sources(config["_diagnostic_engine_source"]),
                    "loaded_params": sorted(runner.loaded_params)})
        runner.loop()
    atomic_json(directory / "exited.json", {"rank": rank})


def engine_child(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine_module, provenance = import_engine(args.engine_source, args.expected_engine_sha)
    from myvllm.sampling_parameters import SamplingParams
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    config["model_name_or_path"] = str(Path(args.checkpoint).resolve(strict=True))
    config["_diagnostic_output"] = str(Path(args.output).resolve())
    config["_diagnostic_max_bytes"] = args.max_rank_bytes
    config["_diagnostic_engine_source"] = str(Path(args.engine_source).resolve())
    atomic_json(Path(args.output) / "engine-provenance.json", provenance)
    engine_module.worker_process = observed_worker
    engine_module.SharedMemory = tracked_shared_memory(args.output, engine_module.SharedMemory)
    engine = engine_module.LLMEngine(config)
    directory = Path(args.output) / "rank-0"
    try:
        with Capture(engine.model_runner.model, directory, 0, config["world_size"], args.max_rank_bytes):
            observe_runner_calls(engine.model_runner, directory)
            atomic_json(directory / "ready.json", {"rank": 0, "pid": os.getpid(),
                        "device": str(next(engine.model_runner.model.parameters()).device),
                        "kv_cache_shape": list(engine.model_runner.kv_cache.shape),
                        "kv_cache_dtype": str(engine.model_runner.kv_cache.dtype),
                        "engine_source": provenance,
                        "imported_modules": imported_sources(args.engine_source),
                        "loaded_params": sorted(engine.model_runner.loaded_params)})
            for rank in range(1, config["world_size"]):
                wait_for_path(Path(args.output) / f"rank-{rank}/ready.json", args.startup_timeout)
            seq = engine.add_prompt(" ".join(f"t{i}" for i in PROMPT),
                                    SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True))
            if seq.prompt_token_ids != PROMPT:
                raise RuntimeError(f"tokenizer did not reproduce original prompt: {seq.prompt_token_ids}")
            prefixes = []
            while not seq.is_finished:
                prefixes.append(list(seq.token_ids))
                before = scheduler_snapshot(engine.scheduler)
                outputs, processed, is_prefill = engine.step()
                for rank in range(1, config["world_size"]):
                    wait_for_path(Path(args.output) / f"rank-{rank}/runner-call-{len(prefixes)}.json",
                                  args.run_timeout)
                atomic_json(Path(args.output) / f"step-{len(prefixes)}.json",
                            {"prefix": prefixes[-1], "tokens_after": list(seq.token_ids),
                             "scheduler_before": before, "scheduler_after": scheduler_snapshot(engine.scheduler),
                             "engine_outputs": outputs, "processed_tokens": processed, "is_prefill": is_prefill})
                if len(prefixes) > OUTPUT_TOKENS:
                    raise RuntimeError("original request exceeded its output budget")
            if len(prefixes) != OUTPUT_TOKENS:
                raise RuntimeError(f"original request ended after {len(prefixes)} steps")
            atomic_json(Path(args.output) / "request-complete.json", {"tokens": list(seq.token_ids),
                        "prefixes": prefixes, "scheduler_finished": engine.scheduler.is_finished(),
                        "used_blocks": sorted(engine.scheduler.block_manager.used_block_ids)})
            engine.exit()
            atomic_json(Path(args.output) / "engine-exit.json", {
                "worker_exitcodes": [process.exitcode for process in engine.processes],
                "process_group_initialized": dist.is_initialized()})
    finally:
        engine.exit()


def scheduler_snapshot(scheduler):
    def sequence(seq):
        return {"id": seq.seq_id, "tokens": list(seq.token_ids),
                "cached_tokens": seq.num_cached_tokens, "scheduled_tokens": seq.num_scheduled_tokens,
                "blocks": list(seq.block_table)}
    return {"waiting": [sequence(seq) for seq in scheduler.waiting],
            "running": [sequence(seq) for seq in scheduler.running],
            "used_blocks": sorted(scheduler.block_manager.used_block_ids)}


def live_group_members(group):
    """Inspect only the session created for this lane; ignore reaped/zombie tasks."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            state, _, process_group, session = stat[stat.rfind(")") + 2:].split()[:4]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(process_group) == group and int(session) == group and state != "Z":
            members.append(int(entry.name))
    return sorted(members)


def terminate_group(process, term_timeout, kill_timeout):
    actions = []
    for sig, timeout in ((signal.SIGTERM, term_timeout), (signal.SIGKILL, kill_timeout)):
        process.poll()
        if not live_group_members(process.pid):
            break
        try:
            os.killpg(process.pid, sig)
            actions.append(signal.Signals(sig).name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            if not live_group_members(process.pid):
                return actions
            time.sleep(0.05)
    process.poll()
    return actions


def tree_size(directory):
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def reserve_shared_memory(directory):
    fd, name = tempfile.mkstemp(prefix="minivllm-p3a-", dir="/dev/shm")
    path = Path(name)
    try:
        os.ftruncate(fd, 2**20)
        stat = os.fstat(fd)
        identity = {"name": path.name, "device": stat.st_dev, "inode": stat.st_ino}
        atomic_json(Path(directory) / "owned-shm.json", identity)
        return fd, identity
    except BaseException:
        try:
            if path.exists() and os.path.samestat(os.fstat(fd), path.lstat()):
                path.unlink()
        finally:
            os.close(fd)
        raise


def cleanup_owned_shared_memory(fd, identity, live_members):
    if fd is None:
        return {"status": "not_created"}
    name = identity["name"]
    if live_members:
        return {"status": "deferred_live_processes", "name": name}
    path = Path("/dev/shm") / name
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return {"status": "already_released", "name": name}
    if not os.path.samestat(os.fstat(fd), stat):
        return {"status": "identity_changed_not_removed", "name": name}
    path.unlink()
    return {"status": "removed_after_exit", "name": name}


def run_lane(command, directory, world_size, env, startup_timeout, run_timeout, exit_timeout,
             term_timeout, kill_timeout, budget_root=None, max_output_bytes=None):
    directory = Path(directory)
    directory.mkdir()
    log_path = directory / "engine.log"
    started = time.monotonic()
    received = []

    def defer_interrupt(signum, frame):
        if not received:
            received.append(signum)

    previous_int = signal.signal(signal.SIGINT, defer_interrupt)
    previous_term = signal.signal(signal.SIGTERM, defer_interrupt)
    previous_mask = None
    fd, identity, process = None, None, None
    try:
        env = env.copy()
        env.pop("MINIVLLM_P3A_SHM_NAME", None)
        if world_size > 1:
            fd, identity = reserve_shared_memory(directory)
            env["MINIVLLM_P3A_SHM_NAME"] = identity["name"]
        try:
            if received:
                raise InterruptedError(f"received signal {received[0]}")
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env=env)
                stage, deadline, failure = "startup", started + startup_timeout, None
                actions = []
                next_budget_check = started
                try:
                    while True:
                        code = process.poll()
                        now = time.monotonic()
                        if received:
                            failure = f"received signal {received[0]}"
                            break
                        if stage == "startup" and all((directory / f"rank-{rank}" / "ready.json").exists()
                                                       for rank in range(world_size)):
                            stage, deadline = "request", now + run_timeout
                        if stage == "request" and (directory / "request-complete.json").exists():
                            stage, deadline = "exit", now + exit_timeout
                        if max_output_bytes is not None and now >= next_budget_check:
                            if tree_size(budget_root) > max_output_bytes:
                                failure = "polled output disk budget exceeded"
                                break
                            next_budget_check = now + 0.5
                        if code is not None:
                            if code != 0:
                                failure = f"engine exited {code} during {stage}"
                                break
                            if stage != "exit" or not (directory / "engine-exit.json").exists():
                                failure = f"engine exited without complete artifacts during {stage}"
                                break
                            exit_data = json.loads((directory / "engine-exit.json").read_text())
                            if (exit_data["worker_exitcodes"] != [0] * (world_size - 1)
                                    or exit_data["process_group_initialized"]):
                                failure = "engine or worker did not exit cleanly"
                                break
                            if not live_group_members(process.pid):
                                break
                        if now >= deadline:
                            failure = f"{stage} deadline exceeded"
                            break
                        time.sleep(0.05)
                finally:
                    actions = terminate_group(process, term_timeout, kill_timeout)
        finally:
            try:
                remaining = live_group_members(process.pid) if process is not None else []
                shared_memory = cleanup_owned_shared_memory(fd, identity, remaining)
                if shared_memory["status"] != "not_created":
                    atomic_json(directory / "owned-shm-cleanup.json", shared_memory)
            finally:
                if fd is not None:
                    os.close(fd)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        if process.poll() is None or remaining:
            failure = failure or f"process group survived KILL deadline: {remaining}"
        elif failure is None and actions:
            failure = "descendant process remained after engine exit"
        elif failure is None and shared_memory["status"] not in ("not_created", "already_released"):
            failure = f"engine exit left shared memory: {shared_memory['status']}"
        if received and failure is None:
            failure = f"received signal {received[0]}"
        status = {"stage": stage, "failure": failure, "exitcode": process.poll(),
                  "process_group": process.pid, "remaining_live_members": remaining,
                  "shared_memory_cleanup": shared_memory,
                  "cleanup_signals": actions, "interrupted_signal": received[0] if received else None,
                  "elapsed_seconds": time.monotonic() - started,
                  "deadlines_seconds": {"startup": startup_timeout, "request": run_timeout,
                                        "exit": exit_timeout, "term": term_timeout, "kill": kill_timeout}}
        atomic_json(directory / "status.json", status)
        return status
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        if previous_mask is not None:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def historical_hashes(manifest, asset_manifest, dtype, checkpoint, reference, originals, configs):
    evidence = json.loads(Path(manifest).read_text())
    assets = json.loads(Path(asset_manifest).read_text())
    if evidence["candidate_commit"] != ORIGINAL_SHA:
        raise ValueError("historical evidence does not identify the original M4 candidate")
    case = "test_two_visible_gpus_hf_check1" if dtype == "bfloat16" else "test_two_visible_gpus_hf_check0"
    case_root = Path(checkpoint).resolve().parent
    expected_paths = {
        Path(reference).resolve(): case_root / "hf-reference.pt",
        Path(originals[0]).resolve(): case_root / "engine-tp1/engine.pt",
        Path(originals[1]).resolve(): case_root / "engine-tp2/engine.pt",
        Path(configs[0]).resolve(): case_root / "engine-tp1/config.json",
        Path(configs[1]).resolve(): case_root / "engine-tp2/config.json",
    }
    if Path(checkpoint).resolve() != case_root / "checkpoint" or any(
            actual != expected for actual, expected in expected_paths.items()):
        raise ValueError("checkpoint, reference, configs and captures must come from the same original case")
    asset_files = []
    for name, wanted in sorted(assets["files"].items()):
        if not name.startswith(case + "/"):
            continue
        relative = Path(name).relative_to(case)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid original asset path: {name}")
        path = case_root / relative
        actual = sha256(path)
        asset_files.append({"input": str(path.resolve()), "manifest_member": name,
                            "actual_sha256": actual, "expected_sha256": wanted,
                            "matches_asset_manifest": actual == wanted})
    if not asset_files:
        raise ValueError("asset manifest does not contain the requested case")
    required = {f"{case}/checkpoint/{path.name}" for path in Path(checkpoint).iterdir() if path.is_file()}
    required.update(f"{case}/{name}" for name in (
        "hf-reference.pt", "engine-tp1/config.json", "engine-tp1/engine.pt", "engine-tp1/rank-0.pt",
        "engine-tp2/config.json", "engine-tp2/engine.pt", "engine-tp2/rank-0.pt", "engine-tp2/rank-1.pt"))
    listed = {item["manifest_member"] for item in asset_files}
    if not required <= listed:
        raise ValueError(f"asset manifest omits required inputs: {sorted(required - listed)}")
    expected = evidence["archive_member_sha256"]
    files = [(f"checkpoint/{path.name}", path) for path in sorted(Path(checkpoint).glob("*.safetensors"))]
    files += [("hf-reference.pt", Path(reference))]
    for tp, path in enumerate(originals, 1):
        files.extend([(f"engine-tp{tp}/engine.pt", Path(path)),
                      (f"engine-tp{tp}/rank-0.pt", Path(path).parent / "rank-0.pt"),
                      (f"engine-tp{tp}/config.json", Path(configs[tp - 1]))])
        if tp == 2:
            files.append(("engine-tp2/rank-1.pt", Path(path).parent / "rank-1.pt"))
    records = []
    for relative, path in files:
        key = f"pytest-tp2/{case}/{relative}"
        actual = sha256(path)
        wanted = expected.get(key)
        records.append({"input": str(path.resolve()), "historical_member": key,
                        "actual_sha256": actual, "historical_sha256": wanted,
                        "matches_historical": wanted == actual if wanted else None})
    return {"evidence": str(Path(manifest).resolve()), "evidence_sha256": sha256(manifest),
            "asset_manifest": str(Path(asset_manifest).resolve()),
            "asset_manifest_sha256": sha256(asset_manifest),
            "asset_archive_sha256": assets["archive_sha256"],
            "asset_remote_root": assets["remote_root"],
            "asset_files": asset_files,
            "all_asset_files_match": all(item["matches_asset_manifest"] for item in asset_files),
            "candidate_commit": evidence["candidate_commit"], "files": records,
            "checkpoint_and_reference_match": all(item["matches_historical"] is True for item in records
                                                   if item["historical_member"].endswith(".safetensors")
                                                   or item["historical_member"].endswith("hf-reference.pt")),
            "original_engine_sequences_match_history": all(item["matches_historical"] is True for item in records
                                                           if item["historical_member"].endswith("/engine.pt")),
            "original_capture_files_match": all(item["matches_historical"] is True for item in records
                                                if item["historical_member"].endswith(".pt")),
            "note": "The later reproduction has its own capture/timing hashes; historical rank and config hashes need not match it."}


def validate_inputs(args):
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if not checkpoint.is_dir() or not list(checkpoint.glob("*.safetensors")):
        raise ValueError("checkpoint must be an existing local safetensors directory")
    configs = [json.loads(Path(path).read_text()) for path in (args.tp1_config, args.tp2_config)]
    for index, config in enumerate(configs, 1):
        if config["world_size"] != index:
            raise ValueError(f"TP{index} config has wrong world_size")
        for key, expected in TINY_CONFIG.items():
            if config[key] != expected:
                raise ValueError(f"TP{index} {key} differs from the original tiny case")
        if config.get("seed", 0) != 0 or config["dtype"] not in TOLERANCES:
            raise ValueError("original seed 0 and BF16 or separate FP32 control are required")
    ignored = {"world_size", "_tp_test_output", "model_name_or_path"}
    if {k: v for k, v in configs[0].items() if k not in ignored} != {
            k: v for k, v in configs[1].items() if k not in ignored}:
        raise ValueError("TP1 and TP2 configurations differ beyond world size and paths")
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    originals = [torch.load(path, map_location="cpu", weights_only=False)
                 for path in (args.original_tp1, args.original_tp2)]
    if reference["dtype"] != configs[0]["dtype"] or reference["tokens"][:len(PROMPT)] != PROMPT:
        raise ValueError("reference dtype or prompt differs from the supplied original config")
    if reference["device"] != "cpu" or reference["buffer_dtypes"]["model.rotary_emb.inv_freq"] != "torch.float32":
        raise ValueError("reference does not match the original CPU / FP32 RoPE provenance")
    if reference["tolerances"] != TOLERANCES[reference["dtype"]]:
        raise ValueError("reference does not carry the original tolerance")
    if len(reference["tokens"]) != len(PROMPT) + OUTPUT_TOKENS or len(reference["logits"]) != OUTPUT_TOKENS:
        raise ValueError("reference does not have the original three output steps")
    for index, original in enumerate(originals, 1):
        if len(original["tokens"]) != len(PROMPT) + OUTPUT_TOKENS or len(original["prefixes"]) != OUTPUT_TOKENS:
            raise ValueError(f"original TP{index} capture is incomplete")
    return configs, reference, originals


def make_environment(output, source):
    output = Path(output).resolve()
    paths = {"TMPDIR": output / "tmp", "XDG_CACHE_HOME": output / "cache",
             "XDG_CONFIG_HOME": output / "cache/config", "XDG_DATA_HOME": output / "cache/data",
             "MPLCONFIGDIR": output / "cache/matplotlib",
             "HF_HOME": output / "cache/huggingface", "HF_HUB_CACHE": output / "cache/hub",
             "PYTHONPYCACHEPREFIX": output / "cache/pycache",
             "TORCHINDUCTOR_CACHE_DIR": output / "cache/torchinductor",
             "TRITON_CACHE_DIR": output / "cache/triton", "CUDA_CACHE_PATH": output / "cache/cuda"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(path) for key, path in paths.items()})
    env.update({"PYTHONPATH": str(Path(source).resolve()), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"})
    return env


def dump_manifest(output):
    output = Path(output)
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "manifest.json.tmp"):
            files.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    atomic_json(output / "manifest.json", {"files": files, "total_bytes": sum(item["bytes"] for item in files)})


def instrumented_logits(rank_dir):
    return [tensor.float() for entry, tensor in load_events(rank_dir)
            if entry["kind"] == "module" and entry["module"] == "lm_head"
            and entry["phase"] == "output" and entry["field"] == "output"]


def capture_issues(lanes, config, instrumented, timeline):
    modules = ["", "model", "model.embed_tokens", "model.norm", "lm_head"]
    suffixes = ("", ".input_layernorm", ".self_attn", ".self_attn.qkv_projection",
                ".self_attn.q_norm", ".self_attn.k_norm", ".self_attn.rotary_emb",
                ".self_attn.attention", ".self_attn.o_proj", ".post_attention_layernorm",
                ".mlp", ".mlp.gate_up", ".mlp.activation", ".mlp.down_proj")
    for layer in range(config["num_layers"]):
        modules.extend(f"model.layers.{layer}{suffix}" for suffix in suffixes)
    issues = []
    for world in (1, 2):
        run = instrumented[world - 1]
        if not run["scheduler_finished"] or run["used_blocks"]:
            issues.append(f"TP{world} scheduler did not finish with all blocks released")
        for rank in range(world):
            entries = [entry for entry, _ in load_events(lanes[world - 1] / f"rank-{rank}")]
            present = {(entry["step"], entry["module"], entry["phase"]) for entry in entries
                       if entry["kind"] == "module"}
            missing = [(step, module, phase) for step in range(OUTPUT_TOKENS)
                       for module in modules for phase in ("input", "output")
                       if (step, module, phase) not in present]
            if missing:
                issues.append(f"TP{world} rank {rank} missing module boundaries: {missing}")
    bad = [row for row in timeline["timeline"] if not row["comparable"]
           and row["reason"] != "different or missing prefix"]
    if bad:
        issues.append(f"{len(bad)} same-prefix module boundaries could not be aligned")
    reductions = timeline["all_reduce_post_rank_agreement"]
    gathers = timeline["gather_received_vs_sent"]
    expected_reductions = OUTPUT_TOKENS * (1 + 2 * config["num_layers"])
    if len(reductions) != expected_reductions or any(row.get("equal") is not True for row in reductions):
        issues.append("TP2 all-reduce post tensors are incomplete or differ between ranks")
    if len(gathers) != OUTPUT_TOKENS * 2 or any(row.get("equal") is not True for row in gathers):
        issues.append("TP2 gathered tensors are incomplete or differ from the corresponding sends")
    return issues


def analyze_complete(output, checkpoint, config, reference, originals, original_ranks):
    output = Path(output)
    lanes = [output / "tp1", output / "tp2"]
    instrumented = [json.loads((lane / "request-complete.json").read_text()) for lane in lanes]
    instrumented_ranks = [{"logits": instrumented_logits(lane / "rank-0")} for lane in lanes]
    gate = original_gate(reference, *originals, *original_ranks)
    new_gate = original_gate(reference, *instrumented, *instrumented_ranks)
    weights = {f"tp{world}-rank{rank}": verify_weights(lanes[world - 1] / f"rank-{rank}", checkpoint,
                                                        config, rank, world)
               for world in (1, 2) for rank in range(world)}
    timeline = compare_events(lanes[0] / "rank-0", [lanes[1] / "rank-0", lanes[1] / "rank-1"],
                              instrumented[0]["prefixes"], instrumented[1]["prefixes"], config)
    timeline["all_reduce_post_rank_agreement"] = compare_collective_ranks(
        [lanes[1] / "rank-0", lanes[1] / "rank-1"])
    timeline["gather_received_vs_sent"] = compare_gather_ranks(
        [lanes[1] / "rank-0", lanes[1] / "rank-1"])
    issues = capture_issues(lanes, config, instrumented, timeline)
    summary = {"original_gate": gate, "instrumented_gate": new_gate,
               "instrumented_tokens_changed_from_original": [
                   new["tokens"] != old["tokens"] for new, old in zip(instrumented, originals)],
               "weights": weights, "first_observed_exact_difference": timeline["first_observed_exact_difference"],
               "capture_complete": not issues, "capture_issues": issues,
               "limitations": timeline["limits"] + [
                   "Hooks and host copies can change synchronization and compilation boundaries.",
                   "Rank 0 waits for the worker's completed-run marker before the next engine step.",
                   "HF reference ran on CPU; this run observes the supplied engine on CUDA.",
                   "Original failed strict greedy remains a failed gate even if instrumentation changes the output.",
                   "The first exact difference is not necessarily significant or the cause of token divergence."]}
    atomic_json(output / "comparisons.json", timeline)
    atomic_json(output / "analysis.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-source", required=True, help="explicit original checkout src directory")
    parser.add_argument("--expected-engine-sha", default=ORIGINAL_SHA)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tp1-config", required=True)
    parser.add_argument("--tp2-config", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--original-tp1", required=True, help="preserved original engine-tp1/engine.pt")
    parser.add_argument("--original-tp2", required=True, help="preserved original engine-tp2/engine.pt")
    parser.add_argument("--historical-evidence", required=True, help="independent recovery numeric.json")
    parser.add_argument("--asset-manifest", required=True, help="manifest of the supplied original reproduction assets")
    parser.add_argument("--output", required=True, help="new directory outside the original checkout and assets")
    parser.add_argument("--startup-timeout", type=float, default=240)
    parser.add_argument("--run-timeout", type=float, default=180)
    parser.add_argument("--exit-timeout", type=float, default=60)
    parser.add_argument("--term-timeout", type=float, default=5)
    parser.add_argument("--kill-timeout", type=float, default=5)
    parser.add_argument("--max-rank-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--max-output-bytes", type=int, default=1024 * 1024 * 1024,
                        help="polling-based disk limit; a fast writer can briefly exceed it")
    parser.add_argument("--preflight-only", action="store_true",
                        help="inspect supplied assets and engine import on CPU without launching an engine")
    args = parser.parse_args(argv)
    deadlines = (args.startup_timeout, args.run_timeout, args.exit_timeout, args.term_timeout, args.kill_timeout)
    if not all(math.isfinite(value) and value > 0 for value in deadlines):
        parser.error("all deadlines must be finite and positive")
    if min(args.max_rank_bytes, args.max_output_bytes) <= 0:
        parser.error("tensor and output budgets must be positive")
    if args.expected_engine_sha != ORIGINAL_SHA:
        parser.error("this diagnostic is restricted to the original M4 commit")
    output = Path(args.output).resolve()
    engine_root = Path(args.engine_source).resolve().parent
    input_root = Path(args.checkpoint).resolve().parent
    if output.is_relative_to(engine_root) or output.is_relative_to(input_root):
        parser.error("output must be outside the engine checkout and original asset directory")
    output.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError(f"received signal {signum}")
    previous_int = signal.signal(signal.SIGINT, interrupted)
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    returncode = 1
    try:
        configs, reference, originals = validate_inputs(args)
        original_ranks = [torch.load(Path(path).parent / "rank-0.pt", map_location="cpu", weights_only=False)
                          for path in (args.original_tp1, args.original_tp2)]
        provenance = historical_hashes(args.historical_evidence, args.asset_manifest, reference["dtype"], args.checkpoint,
                                      args.reference, (args.original_tp1, args.original_tp2),
                                      (args.tp1_config, args.tp2_config))
        original_result = original_gate(reference, *originals, *original_ranks)
        atomic_json(output / "inputs.json", {"harness_path": str(Path(__file__).resolve()),
                    "harness_sha256": sha256(__file__), "engine_source_requested": str(Path(args.engine_source).resolve()),
                    "expected_engine_sha": args.expected_engine_sha, "prompt": PROMPT,
                    "output_tokens": OUTPUT_TOKENS, "sampling_seed": 0, "checkpoint_seed": 12345,
                    "instrumentation_synchronization": "rank 0 waits for rank-local completed-run files between steps",
                    "capture_bounds": {"max_rank_tensor_bytes": args.max_rank_bytes,
                                       "max_rank_tensor_events": MAX_RANK_EVENTS,
                                       "max_output_bytes_polled": args.max_output_bytes},
                    "dtype": reference["dtype"], "historical_provenance": provenance,
                    "config_sha256": [sha256(args.tp1_config), sha256(args.tp2_config)],
                    "original_gate": original_result})
        if (not provenance["checkpoint_and_reference_match"] or not provenance["all_asset_files_match"]
                or not provenance["original_engine_sequences_match_history"]):
            atomic_json(output / "input-validation-error.json", {
                "reason": "original assets do not match the supplied asset manifest and historical evidence"})
            return 3
        if args.preflight_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            sys.dont_write_bytecode = True
            _, engine_provenance = import_engine(args.engine_source, args.expected_engine_sha)
            atomic_json(output / "preflight.json", {"engine_source": engine_provenance,
                        "imported_modules": imported_sources(args.engine_source),
                        "engine_launched": False})
            return 0 if original_result["original_test_pass"] else 2
        env = make_environment(output, args.engine_source)
        statuses = []
        for world, config_path in ((1, args.tp1_config), (2, args.tp2_config)):
            lane = output / f"tp{world}"
            command = [sys.executable, str(Path(__file__).resolve()), "--engine-child",
                       "--engine-source", str(Path(args.engine_source).resolve()),
                       "--expected-engine-sha", args.expected_engine_sha,
                       "--checkpoint", str(Path(args.checkpoint).resolve()),
                       "--config", str(Path(config_path).resolve()), "--output", str(lane),
                       "--startup-timeout", str(args.startup_timeout), "--run-timeout", str(args.run_timeout),
                       "--max-rank-bytes", str(args.max_rank_bytes)]
            statuses.append(run_lane(command, lane, world, env, args.startup_timeout, args.run_timeout,
                                     args.exit_timeout, args.term_timeout, args.kill_timeout,
                                     output, args.max_output_bytes))
            if statuses[-1]["failure"]:
                break
        atomic_json(output / "run-status.json", {"lanes": statuses})
        if len(statuses) == 2 and not any(status["failure"] for status in statuses):
            result = analyze_complete(output, args.checkpoint, configs[0], reference, originals, original_ranks)
            if not all(value["all_equal"] for value in result["weights"].values()):
                returncode = 4
            elif not result["capture_complete"]:
                returncode = 1
            elif not result["original_gate"]["original_test_pass"] or not result["instrumented_gate"]["original_test_pass"]:
                returncode = 2
            else:
                returncode = 0
    except Exception as exc:
        atomic_json(output / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        print(f"diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        dump_manifest(output)
    return returncode


def child_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-source", required=True)
    parser.add_argument("--expected-engine-sha", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rank-bytes", type=int, required=True)
    parser.add_argument("--startup-timeout", type=float, required=True)
    parser.add_argument("--run-timeout", type=float, required=True)
    engine_child(parser.parse_args(argv))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--engine-child":
        child_main(sys.argv[2:])
    else:
        sys.exit(main())
