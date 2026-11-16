"""Test-only capture around the real engine, model head and PD workers."""

import asyncio
from collections import defaultdict
import os

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.sequence import Sequence
from myvllm.pd.worker import PDWorker
from myvllm.sampling_parameters import SamplingParams


class CaptureEngine(LLMEngine):
    def __init__(self, config):
        super().__init__(config)
        self.capture = defaultdict(lambda: {"events": [], "kv": None, "blocks": None})
        self.capture_path = config["capture_path"]
        self.capture_role = config["capture_role"]
        self.step_kinds = []
        self.logits = None
        self.rng = None
        self.device = torch.cuda.current_device()
        self.missing = sorted(set(dict(self.model_runner.model.named_parameters())) -
                              self.model_runner.loaded_params)
        self.head_hook = self.model_runner.model.lm_head.register_forward_hook(self._head)
        self.sampler_hook = self.model_runner.sampler.register_forward_pre_hook(self._sampling)

    def _head(self, module, inputs, output):
        self.logits = output.detach().float().cpu().clone()

    def _sampling(self, module, inputs):
        self.rng = torch.cuda.get_rng_state().cpu().clone()

    def _snapshot(self, seq):
        blocks = list(seq.block_table[:seq.num_cached_blocks])
        cache = self.model_runner.kv_cache
        slices = []
        for index, block in enumerate(blocks):
            valid = min(seq.block_size, seq.num_cached_tokens - index * seq.block_size)
            slices.append(cache[:, :, block, :valid].detach().cpu().clone())
        return blocks, slices

    def step(self, on_tokens=None):
        if self.capture_role == "decode":
            for seq in self.scheduler.running:
                if seq.num_completion_tokens == 1 and seq.num_cached_tokens == seq.num_prompt_tokens:
                    request_id = self.request_ids[seq.seq_id]
                    self.capture[request_id]["blocks"], self.capture[request_id]["kv"] = self._snapshot(seq)
        row = 0

        def capture(seq, token):
            nonlocal row
            request_id = self.request_ids[seq.seq_id]
            record = self.capture[request_id]
            record["events"].append({"token": token, "logits": self.logits[row].clone(),
                                     "rng": self.rng.clone(),
                                     "completion_tokens": seq.num_completion_tokens})
            row += 1
            if self.capture_role == "prefill" and not seq.is_finished:
                record["blocks"], record["kv"] = self._snapshot(seq)
            if on_tokens:
                on_tokens(seq, token)

        result = super().step(on_tokens=capture)
        self.step_kinds.append(result[2])
        return result

    def exit(self):
        if getattr(self, "model_runner", None) is not None and hasattr(self, "capture"):
            self.head_hook.remove()
            self.sampler_hook.remove()
            torch.save({"requests": dict(self.capture), "step_kinds": self.step_kinds,
                        "device": self.device, "missing": self.missing}, self.capture_path)
        super().exit()


def reference_process(config, cases, ready):
    try:
        engine = CaptureEngine(config)
        try:
            for case in cases:
                params = SamplingParams(**case["sampling"])
                seq = Sequence(case["prompt"], config["block_size"], params)
                engine.scheduler.add_sequence(seq)
                engine.request_ids[seq.seq_id] = case["id"]
                while not seq.is_finished:
                    engine.step()
                engine.request_ids.pop(seq.seq_id)
        finally:
            engine.exit()
        ready.send(("ok", os.getpid()))
    except BaseException:
        import traceback
        ready.send(("error", traceback.format_exc()))
    finally:
        ready.close()


def gpu_worker_process(role, config, ready, stop, decode_endpoint, progress_lock=None):
    if progress_lock is not None:
        from tqdm import tqdm
        tqdm.set_lock(progress_lock)

    async def serve():
        worker = PDWorker(role, config, max_connections=8, max_transfer_bytes=1_000_000,
                          request_timeout=120, decode_endpoint=decode_endpoint,
                          engine_factory=CaptureEngine)
        endpoint = await worker.start("127.0.0.1", 0)
        ready.send(("ready", endpoint, os.getpid()))
        ready.close()
        await asyncio.to_thread(stop.recv)
        stop.close()
        await worker.close()
    asyncio.run(serve())
