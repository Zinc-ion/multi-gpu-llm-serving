"""Run real Qwen3 TP layers and Gloo collectives on a small CPU fixture."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.diagnose_tp import Capture
from tp_helpers import cpu_flash
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.utils import reset_context, set_context
from myvllm.utils.loader import load_weights_from_checkpoint


def worker(rank, world, rendezvous, checkpoint, output, config, tokens):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world)
    try:
        import myvllm.layers.attention as attention
        attention.flash_attention_prefill = cpu_flash
        torch.set_default_dtype(getattr(torch, config["dtype"]))
        model = Qwen3ForCausalLM(
            vocab_size=config["vocab_size"], hidden_size=config["hidden_size"],
            num_heads=config["num_heads"], head_dim=config["head_dim"], num_kv_heads=config["num_kv_heads"],
            intermediate_size=config["intermediate_size"], num_layers=config["num_layers"],
            max_position=config["max_position"], qkv_bias=False, ffn_bias=False,
            block_size=config["block_size"], base=config["base"],
        ).eval()
        load_weights_from_checkpoint(model, str(checkpoint))
        ids = torch.tensor(tokens, dtype=torch.long)
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, len(tokens)], dtype=torch.int32),
                    max_seqlen_q=len(tokens), positions=torch.arange(len(tokens)))
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            with Capture(model, Path(output) / f"rank-{rank}", rank, world, 1024 * 1024):
                for _ in range(3):
                    logits = model.compute_logits(model(ids))
        torch.save(logits.cpu(), Path(output) / f"logits-{rank}.pt")
    finally:
        reset_context()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    rendezvous = args.output / "gloo-rendezvous"
    config = json.loads(args.config.read_text()) if args.config else {
        "vocab_size": 13, "hidden_size": 8, "num_heads": 2, "head_dim": 4, "num_kv_heads": 2,
        "intermediate_size": 12, "num_layers": 1, "max_position": 16, "block_size": 2,
        "base": 10000, "dtype": "float32",
    }
    tokens = [1, 9, 4, 6, 12] if args.config else [1, 2, 3, 4]
    mp.spawn(worker, nprocs=args.world,
             args=(args.world, str(rendezvous), args.checkpoint, args.output, config, tokens), join=True)


if __name__ == "__main__":
    main()
