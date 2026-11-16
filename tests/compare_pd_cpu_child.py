"""CPU-only child bootstrap for the real capture CLI and real P/D workers."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch
import torch.distributed as dist

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(ROOT / "tools"))

import myvllm.engine.model_runner as runner_module
import myvllm.layers.attention as attention
from test_paged_prefill import reference_prefill
import compare_pd


def batch_capture(spec_path):
    from myvllm.config import load_config
    from myvllm.engine.sequence import Sequence
    from myvllm.sampling_parameters import SamplingParams

    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec["output"])
    root.mkdir()
    (root / "partial").mkdir()
    cls = compare_pd.capture_engine_class("reference", root, spec["cases"], False,
                                          spec["source"], spec["sha"])
    engine = cls(load_config(spec["config"]))
    seen = []
    head = []
    hook = engine.model_runner.model.lm_head.register_forward_hook(
        lambda module, inputs, output: head.append((output.detach().float().cpu().clone(), output)))
    try:
        for index, case in enumerate(spec["cases"]):
            seq = Sequence(case["input_ids"], engine.config["block_size"],
                           SamplingParams(temperature=spec.get("temperature", 0), max_tokens=1, ignore_eos=True))
            engine.scheduler.add_sequence(seq)
            engine.request_ids[seq.seq_id] = f"case-{index:05d}"

        def publish(seq, token):
            seen.append((engine.request_ids.pop(seq.seq_id), token, list(seq.block_table)))

        engine.step(on_tokens=publish)
        compare_pd.save_json(root / "callbacks.json", seen)
        compare_pd.save_pt(root / "head.pt",
                           {"before": head[0][0], "after": head[0][1].detach().float().cpu().clone()})
    finally:
        hook.remove()
        engine.exit()


def main():
    tensor = torch.tensor
    original_init = dist.init_process_group
    original_run = runner_module.ModelRunner.run

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)

    def init_gloo(backend, *args, **kwargs):
        assert backend == "nccl"
        return original_init("gloo", *args, **kwargs)

    def allocate(runner):
        config = runner.config
        runner.kv_cache = torch.zeros(2, config["num_layers"], config["max_cached_blocks"],
                                      config["block_size"], config["num_kv_heads"], config["head_dim"])
        runner.kv_cache_scales = None
        layer = 0
        for module in runner.model.modules():
            if isinstance(module, attention.Attention):
                module.k_cache = runner.kv_cache[0, layer]
                module.v_cache = runner.kv_cache[1, layer]
                layer += 1

    def store(k, v, kc, vc, slots, block_size):
        kc.flatten(0, 1)[slots] = k
        vc.flatten(0, 1)[slots] = v

    def delayed_run(runner, *args):
        time.sleep(float(os.environ.get("P4_TEST_STEP_DELAY", "0")))
        return original_run(runner, *args)

    def decode(q, kc, vc, tables, lengths, scale, heads, kv_heads, dim, block_size):
        cu_q = torch.arange(len(q) + 1)
        cu_k = torch.cat([torch.zeros(1, dtype=lengths.dtype), lengths.cumsum(0)])
        return reference_prefill(q, kc, vc, cu_q, cu_k, tables,
                                 1, scale, heads, kv_heads, dim, block_size)

    with ExitStack() as stack:
        stack.enter_context(patch.object(dist, "init_process_group", init_gloo))
        stack.enter_context(patch.object(torch.cuda, "set_device"))
        stack.enter_context(patch.object(torch.cuda, "synchronize"))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        stack.enter_context(patch.object(torch, "set_default_device"))
        stack.enter_context(patch.object(torch, "tensor", cpu_tensor))
        stack.enter_context(patch.object(torch.Tensor, "cuda", lambda self, **kwargs: self))
        stack.enter_context(patch.object(runner_module.Qwen3ForCausalLM, "cuda", lambda self, device=None: self))
        stack.enter_context(patch.object(runner_module.ModelRunner, "warmup_model"))
        stack.enter_context(patch.object(runner_module.ModelRunner, "allocate_kv_cache", allocate))
        stack.enter_context(patch.object(runner_module.ModelRunner, "run", delayed_run))
        stack.enter_context(patch.object(attention, "store_kvcache", store))
        stack.enter_context(patch.object(attention, "paged_attention_prefill", reference_prefill))
        stack.enter_context(patch.object(attention, "paged_attention_decode", decode))
        stack.enter_context(torch.compiler.set_stance("force_eager"))
        if len(sys.argv) == 3 and sys.argv[1] == "--test-batch":
            batch_capture(sys.argv[2])
        else:
            compare_pd.main()


if __name__ == "__main__":
    main()
