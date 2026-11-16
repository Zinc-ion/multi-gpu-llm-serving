"""Small FP8 CPU producer for real PD cores and socket workers."""

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.pd.worker import PDWorker


class FP8Execution:
    timings = None

    def __init__(self, config, forced_token=None):
        shape = (2, config["num_layers"], config["max_cached_blocks"],
                 config["block_size"], config["num_kv_heads"], config["head_dim"])
        self.kv_cache = torch.zeros(shape, dtype=torch.float8_e4m3fn)
        self.kv_cache_scales = torch.ones(shape[:-1], dtype=torch.float32)
        self.forced_token = forced_token
        self.layer_transfers = {}
        self.calls = []

    def call(self, method, sequences, is_prefill):
        assert method == "run"
        active = [self.layer_transfers[s.seq_id] for s in sequences
                  if s.seq_id in self.layer_transfers]
        try:
            for layer in range(self.kv_cache.shape[1]):
                for seq in sequences:
                    for pos in range(seq.num_cached_tokens,
                                     seq.num_cached_tokens + seq.num_scheduled_tokens):
                        block = seq.block_table[pos // seq.block_size]
                        offset = pos % seq.block_size
                        for kv in range(2):
                            scale = (seq.token_ids[pos] + layer + kv + 1) / 16
                            values = torch.arange(1, self.kv_cache.shape[-1] + 1).float()
                            if pos == layer == kv == 0:
                                values.zero_()
                                scale = 1.0
                            self.kv_cache[kv, layer, block, offset].copy_(
                                (values * (pos + 1)).to(torch.float8_e4m3fn))
                            self.kv_cache_scales[kv, layer, block, offset].fill_(scale)
                for transfer in active:
                    transfer.on_layer(layer)
        finally:
            for transfer in active:
                transfer.drain_copies()
        return torch.tensor([
            self.forced_token if self.forced_token is not None else sum(seq.token_ids) % 60 + 1
            for seq in sequences if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)
        ])


def fp8_engine(config=None):
    config = config or {}
    model = {
        "model_name_or_path": "tiny-model", "model_type": "Qwen3-0.6B",
        "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
        "world_size": 1, "block_size": 4, "vocab_size": 64,
        "hidden_size": 8, "num_heads": 2, "num_kv_heads": 1,
        "head_dim": 3, "num_layers": 2, "intermediate_size": 16, "base": 10000,
        "max_cached_blocks": 16, "dtype": config.get("dtype", "float32"),
        "kv_cache_dtype": "float8_e4m3fn", "enforce_eager": True,
    }
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = model
    engine.scheduler = Scheduler(4, 32, 16, 4, [62, 63], 32)
    engine.model_runner = FP8Execution(model, config.get("forced_token"))
    engine.request_ids = {}
    engine.exit = lambda: None
    return engine


def fp8_worker(role, decode_endpoint=None, frame_bytes=28, **config):
    return PDWorker(role, {"world_size": 1, "enforce_eager": True, **config},
                    max_connections=4, max_transfer_bytes=1_000_000,
                    request_timeout=5, pipeline_frame_bytes=frame_bytes,
                    decode_endpoint=decode_endpoint, engine_factory=fp8_engine)


def valid_kv(engine, seq):
    cache = engine.model_runner.kv_cache
    scales = engine.model_runner.kv_cache_scales
    data = []
    scale = []
    for pos in range(seq.num_prompt_tokens):
        block = seq.block_table[pos // seq.block_size]
        offset = pos % seq.block_size
        data.append(cache[:, :, block, offset].contiguous().view(torch.uint8))
        scale.append(scales[:, :, block, offset].contiguous().view(torch.uint8))
    return torch.stack(data), torch.stack(scale)
