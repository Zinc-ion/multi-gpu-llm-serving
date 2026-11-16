import pytest
import torch

from myvllm.engine.sequence import Sequence
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.sampling_parameters import SamplingParams
from test_paged_prefill import cpu_model_backend
from test_pd_model_cpu import model_engine


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_engine_step_supports_sampler_eager_fallback(cpu_model_backend, dtype):
    torch.manual_seed(17)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        model = Qwen3ForCausalLM(
            vocab_size=64, hidden_size=8, num_heads=2, head_dim=4, scale=1.0,
            num_kv_heads=1, intermediate_size=16, num_layers=2,
            max_position=32, block_size=4,
        ).eval()
    finally:
        torch.set_default_dtype(previous)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=.1)
    engine = model_engine(model, dtype, chunked=True)
    seq = Sequence([1, 2, 3, 4, 5], 4,
                   SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    engine.scheduler.add_sequence(seq)
    assert not torch.is_inference_mode_enabled()
    while not engine.scheduler.is_finished():
        engine.step()
    assert len(seq.completion_token_ids) == 3
    assert not engine.scheduler.block_manager.used_block_ids
