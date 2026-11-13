import pytest
import torch
import torch.distributed as dist
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from myvllm.layers.layernorm import LayerNorm
from myvllm.models.qwen3 import Qwen3ForCausalLM


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.001, 300])
def test_rmsnorm_matches_hf(dtype, scale):
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(128, generator=generator).to(dtype)
    x = (torch.randn(3, 128, generator=generator) * scale).to(dtype)
    actual = LayerNorm(weight, eps=1e-6)
    reference = Qwen3RMSNorm(128, eps=1e-6).to(dtype)
    reference.weight.data.copy_(weight)
    with torch.compiler.set_stance("force_eager"):
        result = actual(x)
    torch.testing.assert_close(result, reference(x), rtol=0, atol=0)


def test_qwen_passes_configured_eps_to_all_norms(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        model = Qwen3ForCausalLM(
            vocab_size=32, hidden_size=32, num_heads=2, head_dim=16,
            num_kv_heads=1, intermediate_size=64, num_layers=1,
            max_position=16, rms_norm_epsilon=1e-6,
        )
        norms = [module for module in model.modules() if isinstance(module, LayerNorm)]
        assert len(norms) == 5
        assert all(module.eps == 1e-6 for module in norms)
    finally:
        dist.destroy_process_group()
