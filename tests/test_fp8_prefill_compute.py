"""FP8 prefill computation modes, their deployment contract and code generation."""

import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from fp8_helpers import quantize_reference, tolerances
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.layers.attention import Attention, paged_attention_prefill, paged_attention_prefill_kernel
from myvllm.pd.client import PDClient
from myvllm.pd.gateway import PDGateway, _validate_fp8_prefill_mode
from myvllm.pd.ready import check as check_ready
from myvllm.pd.worker import PDWorker
from myvllm.utils import reset_context, set_context
from pd_fp8_helpers import fp8_engine
from test_compare_pd import capture_args
from test_pd_deployment import arguments
from test_pd_server import make_config
from tools import compare_pd


ROOT = Path(__file__).resolve().parents[1]
GPU_ENABLED = os.environ.get("RUN_FP8_PREFILL_GPU") == "1" and torch.cuda.is_available()


def make_inputs(dtype, device, heads=40, kv_heads=8, dim=128,
                q_lengths=(3, 1, 5), k_lengths=(7, 43, 131), block_size=13):
    generator = torch.Generator().manual_seed(20261117)
    sizes = [(n + block_size - 1) // block_size for n in k_lengths]
    count = sum(sizes) + 2
    physical = torch.randperm(count, generator=generator)[:sum(sizes)]
    tables = torch.full((len(sizes), max(sizes)), -1, dtype=torch.int32)
    slots = []
    start = 0
    for i, (size, length) in enumerate(zip(sizes, k_lengths)):
        blocks = physical[start:start + size]
        tables[i, :size] = blocks
        positions = torch.arange(length)
        slots.append(blocks[positions // block_size] * block_size + positions % block_size)
        start += size
    slots = torch.cat(slots)
    assert (physical[1:] - physical[:-1]).abs().ne(1).any()
    assert any(length % block_size for length in k_lengths)

    shape = (sum(k_lengths), kv_heads, dim)
    factors = torch.tensor([.003, .1, 1., 4., 12.])
    factors = factors[torch.arange(shape[0]) % len(factors), None, None]
    keys = (torch.randn(shape, generator=generator) * factors).to(dtype)
    values = (torch.randn(shape, generator=generator) * factors.flip(0)).to(dtype)
    keys[0, 0] = 0
    values[-1, -1] = 0
    cache_shape = (count, block_size, kv_heads, dim)
    kc = torch.zeros(cache_shape, dtype=torch.float8_e4m3fn)
    vc = torch.zeros_like(kc)
    ks = torch.ones(cache_shape[:-1], dtype=torch.float32)
    vs = torch.ones_like(ks)
    for source, data, scales in ((keys, kc, ks), (values, vc, vs)):
        quantized, reference_scales = quantize_reference(source)
        data.view(torch.uint8).flatten(0, 1)[slots] = quantized.view(torch.uint8)
        scales.flatten(0, 1)[slots] = reference_scales
    assert ks.flatten(0, 1)[slots[0], 0] == 1
    assert vs.flatten(0, 1)[slots[-1], -1] == 1

    q = (torch.randn((sum(q_lengths), heads, dim), generator=generator) * .5).to(dtype)
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], dtype=torch.int32)
    return tuple(t.to(device) for t in (q, kc, vc, ks, vs, cu_q, cu_k, tables)), q_lengths, k_lengths, block_size


def reference(inputs, q_lengths, k_lengths, block_size, mode):
    """Independent masked reference with per-tile probability rounding."""
    q, kc, vc, ks, vs, _, _, tables = (t.cpu() for t in inputs)
    heads, dim = q.shape[1:]
    kv_heads = kc.shape[2]
    head_map = torch.arange(heads) // (heads // kv_heads)
    tile = 64 if dim <= 64 else 32 if dim <= 128 else 16
    result = []
    q_start = 0
    for row, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        positions = torch.arange(k_length)
        slots = tables[row, positions // block_size].long() * block_size + positions % block_size
        keys = kc.float().flatten(0, 1)[slots] * ks.flatten(0, 1)[slots, :, None]
        values = vc.float().flatten(0, 1)[slots] * vs.flatten(0, 1)[slots, :, None]
        if mode == "query":
            keys = keys.to(q.dtype).float()
            values = values.to(q.dtype).float()
        keys = keys[:, head_map].transpose(0, 1)
        values = values[:, head_map].transpose(0, 1)
        query = q[q_start:q_start + q_length].float().transpose(0, 1)
        scores = query @ keys.transpose(-1, -2) * (dim ** -.5)
        allowed = positions[None, :] <= k_length - q_length + torch.arange(q_length)[:, None]
        scores = scores.masked_fill(~allowed, -1e10)
        global_max = scores.amax(dim=-1, keepdim=True)
        normalizer = (scores - global_max).exp().sum(dim=-1, keepdim=True)
        output = torch.zeros(heads, q_length, dim)
        for end in range(tile, k_length + tile, tile):
            begin, end = end - tile, min(end, k_length)
            running_max = scores[..., :end].amax(dim=-1, keepdim=True)
            probability = (scores[..., begin:end] - running_max).exp()
            if mode == "query":
                probability = probability.to(q.dtype).float()
            weighted = probability * (running_max - global_max).exp()
            output += weighted @ values[:, begin:end]
        result.append((output / normalizer).transpose(0, 1))
        q_start += q_length
    return torch.cat(result).to(q.dtype)


def exercise_kernel(dtype, device, mode, *, long_prefix=False):
    shape = (4, 2, 32, (1, 4), (769, 17), 17) if long_prefix else (
        40, 8, 128, (3, 1, 5), (7, 43, 131), 13)
    inputs, q_lengths, k_lengths, block_size = make_inputs(dtype, device, *shape)
    q, kc, vc, ks, vs, cu_q, cu_k, tables = inputs
    heads, dim = q.shape[1:]
    kv_heads = kc.shape[2]
    originals = [q.clone(), kc.view(torch.uint8).clone(), vc.view(torch.uint8).clone(),
                 ks.view(torch.uint8).clone(), vs.view(torch.uint8).clone()]
    args = (q, kc, vc, cu_q, cu_k, tables, max(q_lengths), dim ** -.5,
            heads, kv_heads, dim, block_size, ks, vs)
    actual = paged_attention_prefill(*args, fp8_prefill_compute_dtype=mode)
    expected = reference(inputs, q_lengths, k_lengths, block_size, mode)
    torch.testing.assert_close(actual.cpu(), expected, **tolerances(dtype))
    assert torch.isfinite(actual).all()
    if not long_prefix:
        layer = Attention(heads, dim, num_kv_heads=kv_heads, block_size=block_size)
        layer.k_cache, layer.v_cache, layer.k_scale_cache, layer.v_scale_cache = kc, vc, ks, vs
        layer.fp8_prefill_compute_dtype = mode
        set_context(True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=max(q_lengths), block_tables=tables)
        try:
            wrapped = layer(q, torch.empty(0, device=device), torch.empty(0, device=device))
        finally:
            reset_context()
        torch.testing.assert_close(wrapped.reshape_as(actual), actual, rtol=0, atol=0)
    for before, after in zip(originals, (q, kc.view(torch.uint8), vc.view(torch.uint8),
                                         ks.view(torch.uint8), vs.view(torch.uint8))):
        assert torch.equal(before, after)
    return actual.cpu()


def interpreter_checks():
    torch.set_num_threads(1)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        modes = ("float32",) if dtype == torch.bfloat16 else ("float32", "query")
        results = {}
        for mode in modes:
            results[mode] = exercise_kernel(dtype, "cpu", mode)
            exercise_kernel(dtype, "cpu", mode, long_prefix=True)
            exercise_zero_and_nonfinite(dtype, "cpu", mode)
            print(f"interpreter dtype={dtype} mode={mode}: passed", flush=True)
        if dtype == torch.float32:
            assert torch.equal(results["float32"], results["query"])
        if dtype == torch.float16:
            assert not torch.equal(results["float32"], results["query"])
            exercise_cast_ties(dtype, "cpu")
            exercise_fp16_overflow("cpu")
    q = torch.randn(2, 2, 16)
    k, v = torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    cu_k = torch.tensor([0, 4], dtype=torch.int32)
    tables = torch.tensor([[0]], dtype=torch.int32)
    args = (q, k, v, cu_q, cu_k, tables, 2, .25, 2, 1, 16, 4)
    assert torch.equal(paged_attention_prefill(*args),
                       paged_attention_prefill(*args, fp8_prefill_compute_dtype="query"))
    # Triton 3.5.1's CPU interpreter produces incorrect BF16 tl.dot results;
    # BF16 query is covered by the real GPU test and offline SM89 compilation.


def exercise_zero_and_nonfinite(dtype, device, mode):
    q = torch.ones(1, 2, 16, dtype=dtype, device=device)
    kc = torch.zeros(1, 4, 1, 16, dtype=torch.float8_e4m3fn, device=device)
    vc = torch.zeros_like(kc)
    ks = torch.ones(1, 4, 1, device=device)
    vs = torch.ones_like(ks)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 4], dtype=torch.int32, device=device)
    tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    args = (q, kc, vc, cu_q, cu_k, tables, 1, .25, 2, 1, 16, 4, ks, vs)
    zero = paged_attention_prefill(*args, fp8_prefill_compute_dtype=mode)
    assert torch.equal(zero, torch.zeros_like(zero))
    vs[0, 0, 0] = float("nan")
    nan_value = paged_attention_prefill(*args, fp8_prefill_compute_dtype=mode)
    assert torch.isnan(nan_value).all()
    vs[0, 0, 0] = 1
    ks[0, 0, 0] = float("inf")
    inf_key = paged_attention_prefill(*args, fp8_prefill_compute_dtype=mode)
    assert torch.isnan(inf_key).all()


def exercise_cast_ties(dtype, device):
    q = torch.zeros(1, 1, 16, dtype=dtype, device=device)
    kc = torch.zeros(1, 2, 1, 16, dtype=torch.float8_e4m3fn, device=device)
    vc = torch.zeros_like(kc)
    vc.view(torch.uint8)[0, 0, 0].fill_(0x38)  # FP8 1.0
    vc.view(torch.uint8)[0, 1, 0].fill_(0x40)  # FP8 2.0
    ks = torch.ones(1, 2, 1, device=device)
    midpoint = 1 + (2 ** -11 if dtype == torch.float16 else 2 ** -8)
    vs = torch.full_like(ks, midpoint)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 2], dtype=torch.int32, device=device)
    tables = torch.tensor([[0]], dtype=torch.int32, device=device)
    args = (q, kc, vc, cu_q, cu_k, tables, 1, .25, 1, 1, 16, 2, ks, vs)
    original = paged_attention_prefill(*args)
    query = paged_attention_prefill(*args, fp8_prefill_compute_dtype="query")
    assert torch.all(query == 1.5)
    expected = torch.tensor(1.5 * midpoint, dtype=dtype, device=device)
    assert torch.all(original == expected)
    assert not torch.equal(query, original)


def exercise_fp16_overflow(device):
    q = torch.ones(1, 1, 16, dtype=torch.float16, device=device)
    kc = torch.zeros(1, 1, 1, 16, dtype=torch.float8_e4m3fn, device=device)
    kc.view(torch.uint8).fill_(0x38)
    vc = torch.zeros_like(kc)
    ks = torch.full((1, 1, 1), 70000., device=device)
    vs = torch.ones_like(ks)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    table = torch.tensor([[0]], dtype=torch.int32, device=device)
    args = (q, kc, vc, cu, cu, table, 1, .25, 1, 1, 16, 1, ks, vs)
    assert torch.equal(paged_attention_prefill(*args), torch.zeros_like(q))
    assert torch.isnan(paged_attention_prefill(*args, fp8_prefill_compute_dtype="query")).all()


def test_actual_fp8_kernel_in_cpu_interpreter():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"}
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--interpreter"],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dtype=torch.float16 mode=query: passed" in result.stdout
    print(result.stdout)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("query", [False, True])
def test_actual_prefill_sm89_offline_compile(dtype, query):
    signature = {"Q": "*" + dtype, "K": "*fp8e4nv", "V": "*fp8e4nv", "O": "*" + dtype,
                 "cu_seqlens_q_ptr": "*i32", "cu_seqlens_k_ptr": "*i32", "block_tables_ptr": "*i32",
                 "K_scale": "*fp32", "V_scale": "*fp32", "scale": "fp32"}
    constants = {"num_heads": 40, "num_kv_heads": 8, "head_dim": 128,
                 "block_size": 256, "max_num_blocks": 16, "BLOCK_M": 32, "BLOCK_N": 32,
                 "FP8_CACHE": True, "FP8_PREFILL_QUERY": query}
    ptx = triton.compile(ASTSource(paged_attention_prefill_kernel, signature, constants),
                         target=GPUTarget("cuda", 89, 32)).asm["ptx"]
    assert ".entry" in ptx
    if query and dtype in ("fp16", "bf16"):
        assert "mma.sync" in ptx and dtype.replace("fp", "f") in ptx


@pytest.mark.skipif(not GPU_ENABLED, reason="real GPU requires RUN_FP8_PREFILL_GPU=1")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["float32", "query"])
def test_actual_fp8_prefill_on_gpu(dtype, mode):
    exercise_kernel(dtype, "cuda", mode)
    exercise_kernel(dtype, "cuda", mode, long_prefix=True)
    exercise_zero_and_nonfinite(dtype, "cuda", mode)
    if mode == "query" and dtype in (torch.float16, torch.bfloat16):
        exercise_cast_ties(dtype, "cuda")
    if mode == "query" and dtype == torch.float16:
        exercise_fp16_overflow("cuda")


@pytest.mark.skipif(not GPU_ENABLED, reason="real GPU requires RUN_FP8_PREFILL_GPU=1")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_query_model_whole_chunked_eager_and_graph(tmp_path, dtype):
    from test_fp8_model_gpu import exercise_engine, tiny_checkpoint

    base = tiny_checkpoint(tmp_path / "checkpoint")
    results = []
    for chunked, eager in ((False, True), (False, False), (True, True), (True, False)):
        config = dict(base, dtype=dtype, fp8_prefill_compute_dtype="query", enforce_eager=eager,
                      enable_chunked_prefill=chunked, max_num_batched_tokens=5 if chunked else 32)
        results.append(exercise_engine(config))
    for actual in results[1:]:
        assert actual["tokens"] == results[0]["tokens"]
        for logits, expected in zip(actual["logits"], results[0]["logits"]):
            torch.testing.assert_close(logits, expected, **tolerances(getattr(torch, dtype)))


@pytest.mark.skipif(not GPU_ENABLED, reason="real GPU requires RUN_FP8_PREFILL_GPU=1")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_query_pd_bulk_pipeline_and_local(tmp_path, dtype):
    import multiprocessing as mp

    from myvllm.sampling_parameters import SamplingParams
    from pd_helpers import create_tiny_qwen_checkpoint
    from test_pd_fp8_gpu import reference as gpu_reference, same_bits, start_worker
    from test_pd_gpu import _request
    from test_pd_pipeline_gpu import _stop_workers, pipeline_case

    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(dtype=dtype, kv_cache_dtype="float8_e4m3fn", fp8_prefill_compute_dtype="query",
                  scale=1.0, max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=6, seed=0, eos=[63])
    case = {"id": "reference", "prompt": [1, 3, 5, 7, 9],
            "sampling": {"temperature": 0, "max_tokens": 4, "ignore_eos": True}}
    ctx = mp.get_context("spawn")
    expected = gpu_reference(ctx, config, [case], tmp_path / "reference.pt")
    d_path, p_path = tmp_path / "decode.pt", tmp_path / "prefill.pt"
    d = start_worker(ctx, "decode", {**config, "capture_role": "decode", "capture_path": str(d_path),
                                   "enable_chunked_prefill": True, "max_num_batched_tokens": 3})
    p = None

    async def exercise():
        for endpoint in (p[2], d[2]):
            info = await PDClient.info(endpoint)
            assert info["fp8_prefill_compute_dtype"] == "query"
        bulk = await _request(p[2], d[2], {**case, "id": "bulk"})
        pipeline = await pipeline_case(p[2], d[2], {**case, "id": "pipeline"})
        local = await PDClient.open_local(d[2], "local", case["prompt"],
                                          SamplingParams(**case["sampling"]), lifetime=90)
        try:
            local_tokens = []
            while True:
                event = await local.next_token()
                local_tokens.append(event["token"])
                if event["finished"]:
                    break
        finally:
            await local.close()
        for endpoint in (p[2], d[2]):
            stats = await PDClient.stats(endpoint)
            assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0
        return bulk, pipeline, local_tokens

    try:
        p = start_worker(ctx, "prefill", {**config, "capture_role": "prefill",
                                         "capture_path": str(p_path)}, d[2])
        bulk, pipeline, local = asyncio.run(exercise())
    finally:
        _stop_workers([item for item in (p, d) if item is not None])
    pc, dc = torch.load(p_path, weights_only=False), torch.load(d_path, weights_only=False)
    reference_events = expected["requests"]["reference"]["events"]
    tokens = [event["token"] for event in reference_events]
    assert bulk[0] == pipeline[0] == local == tokens
    reference_logits = torch.stack([event["logits"] for event in reference_events])
    for name in ("bulk", "pipeline", "local"):
        events = (pc["requests"].get(name, {}).get("events", []) +
                  dc["requests"].get(name, {}).get("events", []))
        assert [event["token"] for event in events] == tokens
        torch.testing.assert_close(torch.stack([event["logits"] for event in events]), reference_logits,
                                   **tolerances(getattr(torch, dtype)))
        if name != "local":
            assert same_bits(pc["requests"][name]["kv"], dc["requests"][name]["kv"])
    assert any(dc["step_kinds"])


@pytest.mark.parametrize("value", ["fp16", "FLOAT32", "", None, 1, [], {}])
def test_engine_rejects_invalid_mode_before_resources(value):
    with pytest.raises(ValueError, match="Unsupported fp8_prefill_compute_dtype"):
        LLMEngine({"fp8_prefill_compute_dtype": value})
    with pytest.raises(ValueError, match="Unsupported fp8_prefill_compute_dtype"):
        ModelRunner({"fp8_prefill_compute_dtype": value}, rank=0, event=[])


def test_real_runner_propagates_mode_to_each_attention(tmp_path):
    from pd_helpers import create_tiny_qwen_checkpoint
    from pd_tp_compare_cpu_child import cpu_backend

    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    config.update(fp8_prefill_compute_dtype="query", enforce_eager=True)
    previous = torch.get_default_dtype()
    engine = None
    try:
        with cpu_backend():
            engine = LLMEngine(config)
            layers = [m for m in engine.model_runner.model.modules() if isinstance(m, Attention)]
            assert layers and all(m.fp8_prefill_compute_dtype == "query" for m in layers)
            assert engine.model_runner.fp8_prefill_compute_dtype == engine.config["fp8_prefill_compute_dtype"] == "query"
            engine.exit()
            engine = None
    finally:
        if engine is not None:
            engine.exit()
        torch.set_default_dtype(previous)


def cpu_fp8_engine(config):
    engine = fp8_engine(config)
    engine.config["model_name_or_path"] = config["model_name_or_path"]
    engine.config["fp8_prefill_compute_dtype"] = config.get("fp8_prefill_compute_dtype", "float32")
    engine.scheduler.enable_chunked_prefill = config.get("enable_chunked_prefill", False)
    return engine


@pytest.mark.parametrize("routing", ["fixed", "load"])
@pytest.mark.parametrize("p_mode,d_mode,gateway_mode", [
    ("query", "query", "query"), ("query", "float32", "query"),
    ("float32", "query", "float32"), ("float32", "float32", "float32"),
    ("query", "query", "float32")])
def test_real_worker_info_gateway_and_ready_check_mode(tmp_path, routing, p_mode, d_mode, gateway_mode):
    async def run():
        config, _ = make_config(tmp_path / "tokenizer")
        config.update(head_dim=3, kv_cache_dtype="float8_e4m3fn", enforce_eager=True,
                      fp8_prefill_compute_dtype=p_mode, enable_chunked_prefill=True)
        options = dict(max_connections=4, max_transfer_bytes=1_000_000, request_timeout=5,
                       engine_factory=cpu_fp8_engine)
        d = PDWorker("decode", {**config, "fp8_prefill_compute_dtype": d_mode}, **options)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", config, decode_endpoint=de, **options)
        pe = await p.start("127.0.0.1", 0)
        settings = {"pairs": [{"prefill": list(pe), "decode": list(de)}], "max_active_requests": 4,
                    "max_payload_bytes": 1_000_000, "max_staging_bytes": 1_000_000,
                    "max_request_bytes": 4096, "request_timeout": 4}
        if routing == "load":
            settings.update(routing_mode="load", poll_interval_s=1, poll_timeout_s=1,
                            max_sample_age_s=2, elastic={"enabled": False})
        gateway = PDGateway({**config, "fp8_prefill_compute_dtype": gateway_mode}, settings)
        try:
            pi, di = await asyncio.gather(PDClient.info(pe), PDClient.info(de))
            assert pi["fp8_prefill_compute_dtype"] == p_mode
            assert di["fp8_prefill_compute_dtype"] == d_mode
            if p_mode == d_mode:
                result = await check_ready(settings, None)
                assert result["pairs"][0]["decode"]["fp8_prefill_compute_dtype"] == d_mode
            else:
                with pytest.raises(ValueError, match="FP8 prefill compute dtype"):
                    await check_ready(settings, None)
            if p_mode == d_mode == gateway_mode:
                await gateway.start()
                if routing == "load":
                    gateway.poll_task.cancel()
                    await asyncio.gather(gateway.poll_task, return_exceptions=True)
            else:
                with pytest.raises(ValueError, match="FP8 prefill compute dtype"):
                    await gateway.start()
            legacy = copy.deepcopy(pi)
            legacy.pop("fp8_prefill_compute_dtype")
            _validate_fp8_prefill_mode(legacy, "float32")
            with pytest.raises(ValueError, match="FP8 prefill compute dtype"):
                _validate_fp8_prefill_mode(legacy, "query")
        finally:
            await p.close()
            await d.close()
    asyncio.run(run())


def test_prepare_carries_mode_into_all_roles_and_manifest(tmp_path):
    config = json.loads((ROOT / "configs/qwen3_14b.json").read_text())
    config.update(kv_cache_dtype="float8_e4m3fn", fp8_prefill_compute_dtype="query")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = arguments() + ["--base-config", str(config_path), "--m8-comparison",
                          "--physical-gpus", "0", "1", "3", "4",
                          "--pipeline-min-tokens", "1", "--pipeline-frame-bytes", "8192",
                          "--poll-interval-s", "1", "--poll-timeout-s", "1", "--max-sample-age-s", "2",
                          "--prefill-backlog-tokens", "1024", "--tpot-budget-ms", "100",
                          "--tpot-margin-ms", "20", "--step-sample-max-age-ms", "1000",
                          "--idle-without-sample", "allow", "--chunk-tokens", "1024", "--m8-tp2-bulk"]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    files = json.loads(result.stdout)["files"]
    model_files = [name for name in files if name.endswith(".json") and
                   ("control-" in name or name.endswith("-model.json") or
                    name.startswith(("prefill-", "decode-", "load-prefill-", "load-decode-",
                                     "elastic-prefill-", "elastic-decode-", "tp2-bulk-prefill",
                                     "tp2-bulk-decode-")))]
    assert len(model_files) >= 18
    assert all(json.loads(files[name])["fp8_prefill_compute_dtype"] == "query" for name in model_files)
    assert json.loads(files["manifest.json"])["fp8_prefill_compute_dtype"] == "query"
    config["fp8_prefill_compute_dtype"] = "invalid"
    config_path.write_text(json.dumps(config))
    rejected = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0 and "Unsupported fp8_prefill_compute_dtype" in rejected.stderr


def test_capture_and_analyze_require_same_fp8_mode(tmp_path):
    args = capture_args(tmp_path)
    for role in ("reference", "prefill", "decode"):
        path = Path(getattr(args, role + "_config"))
        config = json.loads(path.read_text())
        config["kv_cache_dtype"] = "float8_e4m3fn"
        if role != "reference":
            config["fp8_prefill_compute_dtype"] = "query"
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="same FP8 prefill"):
        compare_pd.read_inputs(args)
    path = Path(args.reference_config)
    config["fp8_prefill_compute_dtype"] = "query"
    path.write_text(json.dumps(config))
    compare_pd.read_inputs(args)
    reference = {"metadata": {"config": config}, "cases": []}
    candidate = copy.deepcopy(reference)
    candidate["metadata"]["config"].pop("fp8_prefill_compute_dtype")
    with pytest.raises(ValueError, match="Different FP8 prefill"):
        compare_pd.analyze_data(reference, candidate, None)
    candidate = copy.deepcopy(reference)
    with pytest.raises(ValueError, match="requires actual prefill compute metadata"):
        compare_pd.analyze_data(reference, candidate, None)
    reference["metadata"]["cache"] = {"fp8_prefill_compute_dtype": "query"}
    candidate["metadata"]["cache"] = {"fp8_prefill_compute_dtype": "float32"}
    with pytest.raises(ValueError, match="Actual FP8 prefill"):
        compare_pd.analyze_data(reference, candidate, None)
    cache = torch.zeros(2, 1, 1, 2, 1, 16, dtype=torch.float8_e4m3fn)
    runner = SimpleNamespace(kv_cache=cache, kv_cache_scales=torch.ones(cache.shape[:-1]))
    with pytest.raises(ValueError, match="Actual FP8 prefill"):
        compare_pd.cache_format(runner, config)
    runner.fp8_prefill_compute_dtype = "query"
    assert compare_pd.cache_format(runner, config)["fp8_prefill_compute_dtype"] == "query"
    del runner.fp8_prefill_compute_dtype
    assert compare_pd.cache_format(runner, {**config, "fp8_prefill_compute_dtype": "float32"})[
        "fp8_prefill_compute_dtype"] == "float32"


@pytest.mark.parametrize("p_mode,d_mode", [("float32", "query"), ("query", "float32")])
def test_capture_checks_actual_worker_info_before_requests(tmp_path, p_mode, d_mode):
    async def run():
        config, _ = make_config(tmp_path / "tokenizer")
        config.update(head_dim=3, kv_cache_dtype="float8_e4m3fn", enforce_eager=True)
        options = dict(max_connections=1, max_transfer_bytes=1_000_000,
                       request_timeout=5, engine_factory=cpu_fp8_engine)
        d = PDWorker("decode", {**config, "fp8_prefill_compute_dtype": d_mode}, **options)
        de = await d.start("127.0.0.1", 0)
        p = PDWorker("prefill", {**config, "fp8_prefill_compute_dtype": p_mode},
                     decode_endpoint=de, **options)
        pe = await p.start("127.0.0.1", 0)
        args = SimpleNamespace(transfer="bulk", request_timeout=4)
        info = {"reference": {"cache": {"requested": "float8_e4m3fn",
                                        "fp8_prefill_compute_dtype": "query"}}}
        try:
            with pytest.raises(ValueError, match="P/D worker roles or model specifications differ"):
                await compare_pd.run_requests(args, tmp_path, [{}], config,
                                              {"prefill": pe, "decode": de}, info, {}, {})
            worker_info = json.loads((tmp_path / "worker-info.json").read_text())
            assert worker_info["prefill"]["fp8_prefill_compute_dtype"] == p_mode
            assert worker_info["decode"]["fp8_prefill_compute_dtype"] == d_mode
            for endpoint in (pe, de):
                stats = await PDClient.stats(endpoint)
                assert stats["open_handles"] == stats["used_blocks"] == 0
        finally:
            await p.close()
            await d.close()
    asyncio.run(run())


if __name__ == "__main__" and sys.argv[1:] == ["--interpreter"]:
    interpreter_checks()
