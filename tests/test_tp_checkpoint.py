import pytest
import torch
from safetensors.torch import save_file

from myvllm.utils.loader import load_weights_from_checkpoint
from tp_helpers import TOLERANCES, create_checkpoint, reference_run, run_model


@pytest.mark.parametrize("family,tied,bias,dtype_name", [
    ("qwen3", False, False, "float32"),
    ("qwen3", True, True, "bfloat16"),
    ("llama", True, True, "float32"),
    ("llama", False, False, "bfloat16"),
])
def test_hf_checkpoint_tp1_tp2_logits_and_greedy(tmp_path, family, tied, bias, dtype_name):
    checkpoint = tmp_path / "checkpoint"
    reference = create_checkpoint(checkpoint, family, tied, bias)
    ref_tokens, ref_logits = reference_run(reference, dtype_name)
    single = run_model(tmp_path, family, tied, bias, dtype_name, 1)[0]
    shards = run_model(tmp_path, family, tied, bias, dtype_name, 2)
    for result in [single, *shards]:
        assert result["model_names"] <= result["loaded"]
        assert result["checkpoint_names"] <= result["loaded"]
        assert result["tokens"] == ref_tokens
    for result in shards:
        measurement = result["measurement"]
        assert measurement["timing_kind"] == "gloo_wall_diagnostic"
        assert measurement["all_reduce_calls"] > 0 and measurement["gather_calls"] > 0
        assert measurement["all_reduce_ms"] >= 0 and measurement["gather_ms"] >= 0
        assert measurement["peak_allocated_bytes"] is None
    for expected, one, two in zip(ref_logits, single["logits"], shards[0]["logits"]):
        torch.testing.assert_close(one, expected, **TOLERANCES[dtype_name])
        torch.testing.assert_close(two, expected, **TOLERANCES[dtype_name])
        torch.testing.assert_close(two, one, **TOLERANCES[dtype_name])
    tp1_error = max((a - b).abs().max().item() for a, b in zip(ref_logits, single["logits"]))
    tp2_error = max((a - b).abs().max().item() for a, b in zip(ref_logits, shards[0]["logits"]))
    print(f"{family} tied={tied} bias={bias} {dtype_name}: "
          f"parameters={len(single['model_names'])} tensors={len(single['checkpoint_names'])} "
          f"TP1/HF max_abs={tp1_error:.8g} TP2/HF max_abs={tp2_error:.8g} "
          f"greedy={ref_tokens[len(ref_tokens) - 3:]}", flush=True)


@pytest.mark.parametrize("kind", ["shape", "missing", "unexpected"])
def test_loader_rejects_invalid_checkpoint(tmp_path, kind):
    model = torch.nn.Linear(3, 2)
    tensors = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
    if kind == "shape":
        tensors["weight"] = torch.ones(3, 3)
    elif kind == "missing":
        del tensors["bias"]
    else:
        tensors["extra"] = torch.ones(1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises((ValueError, RuntimeError), match="(?i)(shape|missing|unload|unexpected|extra)"):
        load_weights_from_checkpoint(model, str(tmp_path))


def _consolidate_checkpoint(source, target, updates):
    from safetensors import safe_open
    tensors = {}
    for shard in source.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            tensors.update({name: weights.get_tensor(name) for name in weights.keys()})
    for name, value in updates.items():
        if value is None:
            del tensors[name]
        elif value == "embed":
            tensors[name] = tensors["model.embed_tokens.weight"].clone()
        elif value == "different":
            tensors[name] = tensors["model.embed_tokens.weight"].clone() + 1
    target.mkdir()
    save_file(tensors, str(target / "model.safetensors"))


def test_loader_rejects_missing_packed_tensor(tmp_path):
    import torch.distributed as dist
    from tp_helpers import make_model
    source, target = tmp_path / "source", tmp_path / "incomplete"
    create_checkpoint(source, "qwen3", False, False)
    _consolidate_checkpoint(source, target, {"model.layers.0.self_attn.k_proj.weight": None})
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        with pytest.raises(ValueError, match="Missing checkpoint shards"):
            load_weights_from_checkpoint(make_model("qwen3", False, False), str(target))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("conflict", [False, True])
def test_tied_checkpoint_aliases_must_agree(tmp_path, conflict):
    import torch.distributed as dist
    from tp_helpers import make_model
    source, target = tmp_path / "source", tmp_path / "aliases"
    create_checkpoint(source, "qwen3", True, False)
    _consolidate_checkpoint(source, target, {"lm_head.weight": "different" if conflict else "embed"})
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1)
    try:
        model = make_model("qwen3", True, False)
        if conflict:
            with pytest.raises(ValueError, match="Tied checkpoint tensors differ"):
                load_weights_from_checkpoint(model, str(target))
        else:
            loaded = load_weights_from_checkpoint(model, str(target))
            assert {"model.embed_tokens.weight", "lm_head.weight"} <= loaded
            assert model.lm_head.weight is model.model.embed_tokens.weight
    finally:
        dist.destroy_process_group()
