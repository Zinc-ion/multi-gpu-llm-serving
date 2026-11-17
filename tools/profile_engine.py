"""Capture a warmed request on every TP rank, separately from HTTP benchmarks."""

import argparse
import json
from pathlib import Path
import subprocess

import torch

import myvllm.engine.llm_engine as engine_module
from myvllm.config import load_config
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


class ProfiledRunner(ModelRunner):
    def begin_profile(self, output_dir):
        torch.cuda.synchronize()
        self.profile_output = Path(output_dir) / f"rank-{self.rank}.json"
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        self.profiler.start()

    def end_profile(self):
        torch.cuda.synchronize()
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(self.profile_output))
        del self.profiler

    def run(self, seqs, is_prefill):
        with torch.profiler.record_function("mini.prefill" if is_prefill else "mini.decode"):
            return super().run(seqs, is_prefill)


# Spawn imports this module before invoking the original engine worker. Both
# ranks therefore keep the production dispatch and shutdown implementation.
engine_module.ModelRunner = ProfiledRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--case", default="tokens-4096")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads(Path(args.prompts).read_text())
    case = next(row for row in manifest["cases"] if row["name"] == args.case)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    engine = engine_module.LLMEngine(config)

    def request():
        seq = Sequence(case["input_ids"], config["block_size"], SamplingParams(
            temperature=0, max_tokens=args.max_tokens, ignore_eos=True,
            max_model_length=config["max_model_length"],
        ))
        engine.scheduler.add_sequence(seq)
        while not seq.is_finished:
            engine.step()
        return seq.completion_token_ids

    try:
        warmup_tokens = request()
        engine.model_runner.call("begin_profile", str(root))
        try:
            tokens = request()
        finally:
            engine.model_runner.call("end_profile")
        (root / "capture.json").write_text(json.dumps({
            "config": config, "case": case, "warmup_requests": 1,
            "warmup_tokens": warmup_tokens, "output_tokens": tokens,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "scope": "one warmed engine request; all TP ranks; no HTTP or PD workers",
            "timing_limit": "profiler perturbs execution; use separate unprofiled HTTP measurements",
        }, ensure_ascii=False, indent=2) + "\n")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
