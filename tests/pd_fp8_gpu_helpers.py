"""Capture the actual FP8 data and scale bytes in independent PD workers."""

import asyncio
import os

import torch

from myvllm.pd.worker import PDWorker
from pd_pipeline_gpu_helpers import PipelineCaptureEngine


class FP8CaptureEngine(PipelineCaptureEngine):
    def _snapshot(self, seq):
        blocks = list(seq.block_table[:seq.num_cached_blocks])
        cache = self.model_runner.kv_cache
        scales = self.model_runner.kv_cache_scales
        slices = []
        for index, block in enumerate(blocks):
            valid = min(seq.block_size, seq.num_cached_tokens - index * seq.block_size)
            data = cache[:, :, block, :valid].contiguous().view(torch.uint8).cpu().clone()
            scale = scales[:, :, block, :valid].contiguous().view(torch.uint8).cpu().clone()
            slices.append((data, scale))
        return blocks, slices


def fp8_gpu_worker_process(role, config, ready, stop, decode_endpoint, progress_lock=None):
    if progress_lock is not None:
        from tqdm import tqdm
        tqdm.set_lock(progress_lock)

    async def serve():
        worker = PDWorker(role, config, max_connections=8, max_transfer_bytes=1_000_000,
                          request_timeout=120, decode_endpoint=decode_endpoint,
                          pipeline_frame_bytes=80, engine_factory=FP8CaptureEngine)
        endpoint = await worker.start("127.0.0.1", 0)
        try:
            ready.send(("ready", endpoint, os.getpid()))
            ready.close()
            await asyncio.to_thread(stop.recv)
        finally:
            stop.close()
            await worker.close()
    asyncio.run(serve())
