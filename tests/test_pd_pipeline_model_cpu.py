import asyncio

import pytest
import torch

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.pd.client import PDClient
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from pd_pipeline_helpers import empty_pipeline
from test_paged_prefill import cpu_model_backend
from test_pd_model_cpu import model_engine
from myvllm.engine.sequence import Sequence


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_actual_runner_layer_hooks_preserve_kv_logits_and_default_path(cpu_model_backend, dtype):
    torch.manual_seed(20261115)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        model = Qwen3ForCausalLM(
            vocab_size=64, hidden_size=8, num_heads=2, head_dim=4, num_kv_heads=1,
            intermediate_size=16, num_layers=2, max_position=32, block_size=4).eval()
    finally:
        torch.set_default_dtype(previous)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=.1)
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    logits = {"reference": [], "prefill": [], "decode": []}
    engines = {}
    hooks = []

    def factory(role):
        def create(config):
            engine = model_engine(model, dtype)
            engine.config["enforce_eager"] = True
            engine.step = torch.inference_mode()(engine.step)
            hooks.append(engine.model_runner.model.lm_head.register_forward_hook(
                lambda module, args, output: logits[role].extend(output.detach().float().clone())))
            engine.exit = lambda: None
            engines[role] = engine
            return engine
        return create

    reference = factory("reference")({})
    seq = Sequence(prompt, 4, params)
    reference.scheduler.add_sequence(seq)
    with torch.inference_mode():
        while not reference.scheduler.is_finished():
            reference.step()

    async def run():
        config = {"world_size": 1, "enforce_eager": True}
        settings = {"max_connections": 4, "max_transfer_bytes": 1_000_000,
                    "request_timeout": 10, "pipeline_frame_bytes": 64}
        dw = PDWorker("decode", config, engine_factory=factory("decode"), **settings)
        de = await dw.start("127.0.0.1", 0)
        pw = PDWorker("prefill", config, engine_factory=factory("prefill"),
                      decode_endpoint=de, **settings)
        pe = await pw.start("127.0.0.1", 0)
        p = d = None
        try:
            d = await PDClient.open_d(de, lifetime=9)
            p = await PDClient.prepare_p(pe, d.session_id, "model", prompt, params, lifetime=9)
            await PDClient.start_pipeline(p, d)
            first = await p.next_token()
            expected = {"payload_bytes": 320, "frames": 6} if dtype == "float32" else {
                "payload_bytes": 160, "frames": 4}
            copied = await p.wait_pipeline()
            assert copied.pop("setup")["data_backend"] == "socket"
            assert copied == expected
            metadata = await p.describe()
            await d.commit_early(metadata)
            ph = next(iter(pw.owner.core._handles.values()))
            dh = next(iter(dw.owner.core._handles.values()))
            for pos in range(len(prompt)):
                a = engines["prefill"].model_runner.kv_cache[
                    :, :, ph.sequence.block_table[pos // 4], pos % 4]
                b = engines["decode"].model_runner.kv_cache[
                    :, :, dh.sequence.block_table[pos // 4], pos % 4]
                assert torch.equal(a, b)
            await p.acknowledge_early()
            await d.activate()
            tokens = [first["token"]]
            while True:
                event = await d.next_token()
                tokens.append(event["token"])
                if event["finished"]:
                    break
            assert tokens == seq.completion_token_ids
            # The temporary hooks have gone before any later ordinary batch.
            assert not engines["prefill"].model_runner.layer_transfers
            assert all(not layer._forward_hooks
                       for layer in engines["prefill"].model_runner.model.model.layers)
        finally:
            if p:
                await p.close()
            if d:
                await d.close()
            try:
                await empty_pipeline(pe)
                await empty_pipeline(de)
            finally:
                await pw.close()
                await dw.close()
    try:
        asyncio.run(run())
        torch.testing.assert_close(
            torch.stack(logits["prefill"] + logits["decode"]), torch.stack(logits["reference"]),
            rtol=1e-4 if dtype == "float32" else .02,
            atol=1e-5 if dtype == "float32" else .02)
    finally:
        for hook in hooks:
            hook.remove()
