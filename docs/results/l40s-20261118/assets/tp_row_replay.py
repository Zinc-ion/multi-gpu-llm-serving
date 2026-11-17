"""Diagnostic replay of the original failing tiny case; no production edits."""

import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

import myvllm.engine.llm_engine as engine_module
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import Sequence
from myvllm.layers.linear import RowParallelLinear
from myvllm.sampling_parameters import SamplingParams


if os.environ.get("REPLAY_FP32_ROW") == "1":
    def float_row(self, x):
        output = F.linear(x.float(), self.weight.float())
        if self.tp_size > 1:
            dist.all_reduce(output)
        output = output.to(x.dtype)
        return output if self.bias is None else output + self.bias
    RowParallelLinear.forward = float_row


class ReplayRunner(ModelRunner):
    def __init__(self, config, rank, event=None):
        super().__init__(config, rank, event)
        self.rows = []
        self.replay_dir = Path(config["replay_dir"])
        module = self.model.model.layers[0].self_attn.o_proj
        self.weight = module.weight.detach().cpu().clone()
        module.register_forward_hook(lambda module, args, output: self.rows.append({
            "input": args[0].detach().cpu().clone(), "output": output.detach().cpu().clone(),
        }))

    def exit(self):
        if hasattr(self, "rows") and not self._exited:
            torch.save({"weight": self.weight, "rows": self.rows},
                       self.replay_dir / f"rank-{self.rank}.pt")
        super().exit()


engine_module.ModelRunner = ReplayRunner


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    config["world_size"] = int(sys.argv[2])
    root = Path(sys.argv[3])
    root.mkdir()
    config["replay_dir"] = str(root)
    engine = engine_module.LLMEngine(config)
    logits = []
    engine.model_runner.model.lm_head.register_forward_hook(
        lambda module, args, output: logits.append(output.detach().cpu().clone()))
    try:
        seq = Sequence([1, 9, 4, 6, 12], config["block_size"],
                       SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
        engine.scheduler.add_sequence(seq)
        while not seq.is_finished:
            engine.step()
        torch.save({"tokens": seq.completion_token_ids, "logits": logits}, root / "output.pt")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
