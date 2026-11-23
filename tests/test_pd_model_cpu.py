from copy import deepcopy

import pytest
import torch

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import Sequence
from myvllm.layers.attention import Attention
from myvllm.layers.sampler import SamplerLayer
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.pd import DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from test_paged_prefill import cpu_model_backend
from test_pd_core import first_token, make_engine, start


def model_engine(model, dtype, *, chunked=False):
    engine = make_engine(dtype=dtype, chunked=chunked, budget=3 if chunked else 16)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = engine.config
    runner.model = deepcopy(model)
    runner.rank, runner.block_size, runner.world_size = 0, 4, 1
    runner.sampler, runner.timings, runner.enforce_eager = SamplerLayer(), None, True
    runner.kv_cache = torch.zeros(2, 2, 8, 4, 1, 4, dtype=getattr(torch, dtype), device="cpu")
    layer = 0
    for module in runner.model.modules():
        if isinstance(module, Attention):
            module.k_cache = runner.kv_cache[0, layer]
            module.v_cache = runner.kv_cache[1, layer]
            layer += 1
    engine.model_runner = runner
    return engine


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("chunked", [False, True])
def test_qwen_logits_and_tokens_match_after_real_kv_handoff(cpu_model_backend, dtype, chunked):
    torch.manual_seed(20261115)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        model = Qwen3ForCausalLM(
            vocab_size=64, hidden_size=8, num_heads=2, head_dim=4, num_kv_heads=1,
            intermediate_size=16, num_layers=2, max_position=32, block_size=4,
        ).eval()
    finally:
        torch.set_default_dtype(previous_dtype)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=.1)
    reference = model_engine(model, dtype)
    p_engine = model_engine(model, dtype, chunked=chunked)
    d_engine = model_engine(model, dtype, chunked=chunked)
    reference_logits, p_logits, d_logits = [], [], []
    hooks = []
    for engine, collected in ((reference, reference_logits), (p_engine, p_logits),
                              (d_engine, d_logits)):
        hooks.append(engine.model_runner.model.lm_head.register_forward_hook(
            lambda module, args, output, collected=collected:
                collected.extend(output.detach().float().clone())
        ))
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    prompt = [1, 2, 3, 4, 5]
    try:
        with torch.inference_mode():
            seq = Sequence(prompt, 4, params)
            reference.scheduler.add_sequence(seq)
            while not reference.scheduler.is_finished():
                reference.step()
            p = PrefillCore(p_engine, 8, 1_000_000)
            d = DecodeCore(d_engine, 8, 1_000_000)
            ph, dh = start(p, d, prompt, params=params)
            first = first_token(p, ph)
            metadata = p.describe(ph)
            d.reserve(dh, metadata)
            payload = p.export(ph)
            d.restore(dh, payload)
            p.acknowledge(ph)
            d.activate(dh)
            tokens = [first.token]
            while dh.state == "active":
                tokens.extend(event.token for event in d.step())
            assert tokens == seq.completion_token_ids
            actual_logits = torch.stack(p_logits + d_logits)
            expected_logits = torch.stack(reference_logits)
            torch.testing.assert_close(
                actual_logits, expected_logits,
                rtol=1e-4 if dtype == "float32" else .02,
                atol=1e-5 if dtype == "float32" else .02,
            )
            p.close(ph)
            d.close(dh)
    finally:
        for hook in hooks:
            hook.remove()
