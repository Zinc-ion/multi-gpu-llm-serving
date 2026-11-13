"""Opt-in step timings; CUDA events are read after the existing token D2H."""

import json
from time import perf_counter

import torch


def write_record(path, record):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


class StepTimings:
    def __init__(self):
        self.step = 0

    def begin(self):
        self.started = perf_counter()
        self.cpu = {}
        self.events = {}

    def prepare(self):
        self.prepare_started = perf_counter()

    def h2d(self):
        self.cpu["prepare_ms"] = (perf_counter() - self.prepare_started) * 1000
        self.start_gpu("h2d")

    def start_gpu(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.events[name] = (start, end)

    def end_gpu(self, name):
        self.events[name][1].record()

    def write(self, path, is_prefill, seqs, token_count, request_ids):
        gpu = {name + "_ms": start.elapsed_time(end) for name, (start, end) in self.events.items()}
        gpu["h2d_ms"] += gpu.pop("sample_h2d_ms")
        write_record(path, {
            "event": "step", "step": self.step, "is_prefill": is_prefill,
            "num_sequences": len(seqs), "num_tokens": token_count,
            "request_ids": [request_ids.get(seq.seq_id, str(seq.seq_id)) for seq in seqs],
            "step_ms": (perf_counter() - self.started) * 1000,
            **self.cpu, **gpu,
        })
        self.step += 1
