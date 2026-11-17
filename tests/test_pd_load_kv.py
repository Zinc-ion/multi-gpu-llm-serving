import asyncio

import pytest
import torch

from myvllm.server import IncrementalDecoder
from pd_network_helpers import SlowCPUExecution
from pd_pipeline_helpers import LayeredCPUExecution, empty_pipeline
from test_pd_core import reference_tokens
from test_pd_load_routing import http_gateway, layered_engine, load_engine, nodes, records, reserved_remote, settings, wait_sample
from test_pd_server import body, drained


class CaptureExecution:
    def __init__(self, *args):
        super().__init__(*args)
        self.prompt_kv = {}
        self.restored_kv = {}
        self.output_tokens = {}

    def snapshot(self, seq):
        return torch.stack([
            self.kv_cache[:, :, seq.block_table[pos // seq.block_size], pos % seq.block_size].clone()
            for pos in range(seq.num_prompt_tokens)
        ])

    def call(self, method, sequences, is_prefill):
        for seq in sequences:
            if seq.num_cached_tokens == seq.num_prompt_tokens and seq.num_completion_tokens == 1:
                self.restored_kv[tuple(seq.prompt_token_ids)] = self.snapshot(seq)
        sampled = [seq for seq in sequences
                   if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)]
        tokens = super().call(method, sequences, is_prefill)
        for seq, token in zip(sampled, tokens.tolist()):
            prompt = tuple(seq.prompt_token_ids)
            self.output_tokens.setdefault(prompt, []).append(token)
            if seq.num_completion_tokens == 0:
                self.prompt_kv[prompt] = self.snapshot(seq)
        return tokens


class CapturingBulk(CaptureExecution, SlowCPUExecution):
    pass


class CapturingPipeline(CaptureExecution, LayeredCPUExecution):
    pass


@pytest.mark.parametrize("pipeline", [False, True])
def test_routed_nondefault_decode_matches_full_tokens_and_raw_kv(tmp_path, pipeline):
    def engine_factory(config):
        if pipeline:
            engine = layered_engine(config)
            engine.model_runner = CapturingPipeline(engine.config, config.get("layer_delay", 0),
                                                     config.get("forced_token"))
        else:
            engine = load_engine(config)
            engine.model_runner = CapturingBulk(engine.config, config.get("delay", 0),
                                                 config.get("forced_token"))
        return engine

    async def run():
        async with nodes(tmp_path, frame=64 if pipeline else 0, engine_factory=engine_factory) as (
                config, tokenizer, ps, ds, workers):
            pairs = list(zip(ps, ds))
            cfg = settings(pairs, poll_interval=.01, pipeline=2 if pipeline else 0)
            async with http_gateway(config, cfg, tmp_path) as (client, gateway, _):
                prefill, decode = await reserved_remote(ps[1], ds[0])
                try:
                    await wait_sample(gateway, ds[0], lambda row: row["resident_requests"] == 1)
                    response = await client.post("/v1/completions", json=body(length=4))
                    assert response.status_code == 200, response.text
                    assigned = records(tmp_path / "gateway.jsonl", "pd_gateway_assigned")[-1]
                    assert tuple(assigned["prefill"]) == ps[0]
                    assert tuple(assigned["decode"]) == ds[1]
                    source = workers[2].owner.engine.model_runner
                    destination = workers[1].owner.engine.model_runner
                    prompt = (1, 2)
                    assert torch.equal(source.prompt_kv[prompt], destination.restored_kv[prompt])
                    tokens = source.output_tokens[prompt] + destination.output_tokens[prompt]
                    assert tokens == reference_tokens(list(prompt), "float32")[:4]
                    decoder = IncrementalDecoder(tokenizer)
                    expected = "".join(decoder.push(token, index == len(tokens) - 1)
                                       for index, token in enumerate(tokens))
                    assert response.json()["choices"][0]["text"] == expected
                finally:
                    await prefill.close()
                    await decode.close()
                await drained(gateway, pairs)
                for endpoint in ps + ds:
                    await empty_pipeline(endpoint)

    asyncio.run(run())
