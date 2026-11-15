"""Layered CPU execution and real socket helpers for the M6 transaction."""

import asyncio
from contextlib import asynccontextmanager
import time

import torch

from myvllm.pd.client import PDClient
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams
from test_pd_core import CPUExecution, make_engine


class LayeredCPUExecution(CPUExecution):
    def __init__(self, config, delay, forced_token):
        super().__init__(config, forced_token)
        self.delay = delay
        self.layer_transfers = {}

    def call(self, method, sequences, is_prefill):
        assert method == "run"
        for seq in sequences:
            self.calls.append((seq.seq_id, seq.num_cached_tokens,
                               seq.num_cached_tokens + seq.num_scheduled_tokens,
                               tuple(seq.block_table), is_prefill))
        active = [self.layer_transfers[s.seq_id] for s in sequences
                  if s.seq_id in self.layer_transfers]
        try:
            for layer in range(self.kv_cache.shape[1]):
                if self.delay:
                    time.sleep(self.delay)
                for seq in sequences:
                    for pos in range(seq.num_cached_tokens,
                                     seq.num_cached_tokens + seq.num_scheduled_tokens):
                        block = seq.block_table[pos // seq.block_size]
                        for kv in range(2):
                            self.kv_cache[kv, layer, block, pos % seq.block_size] = (
                                seq.token_ids[pos] + pos * .25 + layer * .5 + kv)
                for transfer in active:
                    transfer.on_layer(layer)
        finally:
            for transfer in active:
                transfer.drain_copies()
        return torch.tensor([
            self.forced_token if self.forced_token is not None else sum(seq.token_ids) % 60 + 1
            for seq in sequences if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)
        ], device="cpu")


def pipeline_engine(config):
    engine = make_engine(blocks=config.get("blocks", 16), budget=32,
                         sequences=config.get("sequences", 4), model_length=32,
                         dtype=config.get("dtype", "float32"))
    engine.config["enforce_eager"] = True
    engine.model_runner = LayeredCPUExecution(
        engine.config, config.get("layer_delay", 0), config.get("forced_token"))
    engine.exit = lambda: None
    return engine


def pipeline_worker(role, decode_endpoint=None, frame_bytes=64, **config):
    return PDWorker(role, {"world_size": 1, "enforce_eager": True, **config},
                    max_connections=4, max_transfer_bytes=1_000_000,
                    request_timeout=5, pipeline_frame_bytes=frame_bytes,
                    decode_endpoint=decode_endpoint, engine_factory=pipeline_engine)


@asynccontextmanager
async def pipeline_workers(**config):
    d = pipeline_worker("decode", **config)
    de = await d.start("127.0.0.1", 0)
    p = pipeline_worker("prefill", de, **config)
    pe = await p.start("127.0.0.1", 0)
    try:
        yield p, d, pe, de
    finally:
        await p.close()
        await d.close()


async def prepare_pair(pe, de, prompt, params=None, lifetime=4, request_id="pipeline"):
    params = params or SamplingParams(temperature=0, max_tokens=6, ignore_eos=True)
    d = await PDClient.open_d(de, lifetime=lifetime)
    try:
        p = await PDClient.prepare_p(pe, d.session_id, request_id, prompt, params,
                                     lifetime=lifetime)
    except BaseException:
        await d.close()
        raise
    return p, d


async def pipeline_request(pe, de, prompt, params=None, request_id="pipeline"):
    p, d = await prepare_pair(pe, de, prompt, params, request_id=request_id)
    try:
        await PDClient.start_pipeline(p, d)
        event = await p.next_token()
        tokens = [event["token"]]
        if not event["finished"]:
            await PDClient.handoff_pipeline(p, d, await p.describe())
            while True:
                event = await d.next_token()
                tokens.append(event["token"])
                if event["finished"]:
                    break
        return tokens
    finally:
        await p.close()
        await d.close()


async def empty_pipeline(endpoint):
    for _ in range(200):
        stats = await PDClient.stats(endpoint)
        if stats["open_handles"] == stats["used_blocks"] == stats["reserved_transfer_bytes"] == 0:
            assert stats["pipeline_slots"] == stats["pipeline_staging_bytes"] == 0
            assert stats["pipeline_pinned_bytes"] == stats["pipeline_busy_slots"] == 0
            assert stats["queued_events"] == 0
            if "reserved_blocks" in stats:
                assert stats["reserved_blocks"] == stats["resident_requests"] == 0
            return stats
        await asyncio.sleep(.01)
    raise AssertionError(f"Pipeline resources did not drain: {stats}")
