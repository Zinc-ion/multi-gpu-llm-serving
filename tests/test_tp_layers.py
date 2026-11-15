import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from myvllm.layers.embedding_head import ParallelLMHead
from myvllm.layers.linear import MergedColumnParallelLinear, QKVColumnParallelLinear, RowParallelLinear
from myvllm.utils import reset_context, set_context
from tp_helpers import cpu_flash, make_model


def _row_worker(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
        bias = torch.tensor([0.3, -0.2, 0.4, 0.8])
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5, 0.7, -0.3]])
        layer = RowParallelLinear(6, 4, bias=True)
        layer.weight.weight_loader(layer.weight, weight)
        if mode == "load":
            layer.bias.weight_loader(layer.bias, bias)
        else:
            with torch.no_grad():
                layer.bias.copy_(bias)
        actual = layer(x[:, rank * 3:(rank + 1) * 3])
        torch.testing.assert_close(actual, F.linear(x, weight, bias), rtol=1e-6, atol=1e-6)
    finally:
        dist.destroy_process_group()


def test_row_bias_load_and_allreduce(tmp_path):
    mp.spawn(_row_worker, nprocs=2, args=(str(tmp_path / "load"), "load"), join=True)


def test_row_bias_is_added_once_after_allreduce(tmp_path):
    mp.spawn(_row_worker, nprocs=2, args=(str(tmp_path / "reduce"), "reduce"), join=True)


def _head_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        head = ParallelLMHead(7, 4).to(dtype=torch.bfloat16)
        weight = torch.arange(28, dtype=torch.float32).reshape(7, 4).to(torch.bfloat16)
        head.weight.weight_loader(head.weight, weight)
        x = torch.tensor([[0.5, 1, -1, 2]], dtype=torch.bfloat16)
        result = head(x)
        if rank == 0:
            torch.testing.assert_close(result, F.linear(x, weight), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_lm_head_gathers_bfloat16_and_trims_vocab(tmp_path):
    mp.spawn(_head_worker, nprocs=2, args=(str(tmp_path / "head"),), join=True)


def _llama_attention_worker(rank, rendezvous):
    import myvllm.layers.attention as attention
    attention.flash_attention_prefill = cpu_flash
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(91)
        model = make_model("llama", tied=False, bias=False).eval()
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.05)
        ids = torch.tensor([1, 3, 5])
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
                    max_seqlen_q=3, positions=torch.arange(3))
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            result = model(ids)
        assert result.shape == (3, 64)
        assert torch.isfinite(result).all()
    finally:
        reset_context()
        dist.destroy_process_group()


def test_llama_attention_uses_local_head_count(tmp_path):
    mp.spawn(_llama_attention_worker, nprocs=2, args=(str(tmp_path / "llama"),), join=True)


def _unsupported_shards_worker(rank, rendezvous):
    import pytest
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        with pytest.raises(ValueError, match="divis"):
            MergedColumnParallelLinear(4, [3, 3])
        with pytest.raises(ValueError, match="divis"):
            QKVColumnParallelLinear(4, head_size=2, num_heads=4, num_kv_heads=1)
    finally:
        dist.destroy_process_group()


def test_individual_merged_and_kv_heads_must_be_divisible(tmp_path):
    mp.spawn(_unsupported_shards_worker, nprocs=2, args=(str(tmp_path / "invalid"),), join=True)


def test_attention_bias_keeps_qk_norm_and_output_bias(tmp_path):
    import myvllm.layers.attention as attention
    previous_flash = attention.flash_attention_prefill
    attention.flash_attention_prefill = cpu_flash
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'attention-bias'}", rank=0, world_size=1)
    try:
        model = make_model("qwen3", tied=False, bias=True).eval()
        calls = []
        handles = [module.register_forward_hook(lambda module, args, output: calls.append(module))
                   for module in (model.model.layers[0].self_attn.q_norm,
                                  model.model.layers[0].self_attn.k_norm)]
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.05)
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
                    max_seqlen_q=2, positions=torch.arange(2))
        with torch.inference_mode(), torch.compiler.set_stance("force_eager"):
            model(torch.tensor([1, 2]))
        assert len(calls) == 2
        assert model.model.layers[0].self_attn.o_proj.bias is not None
        assert make_model("llama", tied=False, bias=True).model.layers[0].self_attn.o_proj.bias is not None
    finally:
        for handle in handles:
            handle.remove()
        reset_context()
        dist.destroy_process_group()
        attention.flash_attention_prefill = previous_flash


def test_parameter_loaders_and_ties_survive_device_conversion(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'conversion'}", rank=0, world_size=1)
    try:
        for family in ("qwen3", "llama"):
            model = make_model(family, tied=True, bias=True).to("meta")
            assert model.lm_head.weight is model.model.embed_tokens.weight
            for module in model.modules():
                if isinstance(module, (RowParallelLinear, MergedColumnParallelLinear,
                                       QKVColumnParallelLinear, ParallelLMHead)):
                    for param in module.parameters(recurse=False):
                        assert callable(param.weight_loader)
    finally:
        dist.destroy_process_group()


def test_layer_weight_loader_survives_device_conversion(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'layer-conversion'}", rank=0, world_size=1)
    try:
        layer = RowParallelLinear(6, 4, bias=True).to("meta")
        assert callable(layer.weight.weight_loader)
        assert callable(layer.bias.weight_loader)
    finally:
        dist.destroy_process_group()


def _column_embedding_worker(rank, rendezvous):
    from myvllm.layers.embedding_head import VocabParallelEmbedding
    from myvllm.layers.linear import ColumnParallelLinear
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        x = torch.tensor([[0.25, 0.5, -1, 2]])
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
        bias = torch.arange(6, dtype=torch.float32) / 5
        layer = ColumnParallelLinear(4, 6, bias=True)
        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        parts = [torch.empty(1, 3) for _ in range(2)]
        dist.all_gather(parts, layer(x))
        torch.testing.assert_close(torch.cat(parts, dim=-1), F.linear(x, weight, bias))
        embedding = VocabParallelEmbedding(7, 4)
        embed_weight = torch.arange(28, dtype=torch.float32).reshape(7, 4)
        embedding.weight.weight_loader(embedding.weight, embed_weight)
        ids = torch.tensor([[0, 3, 4], [6, 1, 5]])
        torch.testing.assert_close(embedding(ids), F.embedding(ids, embed_weight))
    finally:
        dist.destroy_process_group()


def test_column_and_padded_vocab_embedding(tmp_path):
    mp.spawn(_column_embedding_worker, nprocs=2, args=(str(tmp_path / "column"),), join=True)
