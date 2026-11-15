import pytest
import torch
import torch.distributed as dist

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.layers.attention import Attention
from myvllm.layers.sampler import SamplerLayer
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import reset_context


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_cudagraph_replay_matches_eager_and_chunked_tokens(tmp_path, dtype):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        torch.manual_seed(8)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = Qwen3ForCausalLM(
                vocab_size=64, hidden_size=32, num_heads=2, head_dim=16,
                num_kv_heads=1, intermediate_size=64, num_layers=2,
                max_position=32, block_size=4,
            ).cuda().eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        for param in model.parameters():
            torch.nn.init.normal_(param, std=.1)
        for module in model.modules():
            if isinstance(module, Attention):
                module.k_cache = torch.zeros(32, 4, 1, 16, device="cuda", dtype=dtype)
                module.v_cache = torch.zeros_like(module.k_cache)
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = {
            "max_num_sequences": 7, "max_num_batched_tokens": 32,
            "max_model_length": 32, "max_cached_blocks": 32,
            "hidden_size": 32, "vocab_size": 64,
        }
        runner.rank, runner.block_size, runner.world_size = 0, 4, 1
        runner.default_dtype = dtype
        runner.model, runner.sampler, runner.timings = model, SamplerLayer(), None
        runner.capture_cudagraph()
        expected = None
        for chunked, eager in ((False, True), (False, False), (True, True), (True, False)):
            runner.enforce_eager = eager
            engine = LLMEngine.__new__(LLMEngine)
            engine.model_runner = runner
            engine.scheduler = Scheduler(7, 5 if chunked else 32, 32, 4, [63], 32,
                                         enable_chunked_prefill=chunked)
            seqs = [
                Sequence(list(range(1, length + 1)), 4, SamplingParams(
                    temperature=0, max_tokens=output, ignore_eos=True,
                )) for length, output in ((5, 2), (7, 4), (9, 6))
            ]
            for seq in seqs:
                engine.scheduler.add_sequence(seq)
            while not engine.scheduler.is_finished():
                engine.step()
            result = [seq.completion_token_ids for seq in seqs]
            if expected is None:
                expected = result
            else:
                assert result == expected
            assert not engine.scheduler.block_manager.used_block_ids
            assert all(torch.isfinite(value).all() for value in runner.graph_vars.values())
    finally:
        reset_context()
        dist.destroy_process_group()
