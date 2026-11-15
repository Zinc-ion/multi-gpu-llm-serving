import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str, revision: str | None = None, cache_dir: str | None = None):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.
    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download

    # Try to resolve the path - could be local or from HF cache
    checkpoint_path = None

    # First, try local paths
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # Skip non-safetensors weights
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                if weight_name in hf_weights:
                    raise ValueError(f"Duplicate checkpoint tensor: {weight_name}")
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Now map and load weights into custom model
    params = dict(model.named_parameters(remove_duplicate=False))
    aliases = {}
    for name, param in params.items():
        aliases.setdefault(id(param), set()).add(name)

    # HF stores separate Q/K/V and gate/up tensors; each shard is placed in its
    # own rank-local region by the parallel layer's weight_loader.
    packed = {
        '.self_attn.q_proj.': ('.self_attn.qkv_projection.', 'q', {'q', 'k', 'v'}),
        '.self_attn.k_proj.': ('.self_attn.qkv_projection.', 'k', {'q', 'k', 'v'}),
        '.self_attn.v_proj.': ('.self_attn.qkv_projection.', 'v', {'q', 'k', 'v'}),
        '.mlp.gate_proj.': ('.mlp.gate_up.', 0, {0, 1}),
        '.mlp.up_proj.': ('.mlp.gate_up.', 1, {0, 1}),
    }
    loaded_params = set()
    packed_parts = {}
    tied_sources = {}

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        custom_name = hf_name
        shard_id = None
        expected_parts = None
        for source, (target, part, expected) in packed.items():
            if source in hf_name:
                custom_name = hf_name.replace(source, target, 1)
                shard_id, expected_parts = part, expected
                break
        if custom_name not in params:
            raise ValueError(f"Unexpected checkpoint tensor {hf_name}: no model parameter {custom_name}")
        param = params[custom_name]
        weight_loader = getattr(param, 'weight_loader', default_weight_loader)
        try:
            if shard_id is None:
                previous = tied_sources.get(id(param))
                if previous is not None:
                    if not torch.equal(hf_weights[previous], hf_weight):
                        raise ValueError(f"Tied checkpoint tensors differ: {previous} and {hf_name}")
                else:
                    weight_loader(param, hf_weight)
                    tied_sources[id(param)] = hf_name
                loaded_params.update(aliases[id(param)])
            else:
                weight_loader(param, hf_weight, shard_id)
                parts, _ = packed_parts.setdefault(custom_name, (set(), expected_parts))
                parts.add(shard_id)
        except (ValueError, RuntimeError) as exc:
            exc.add_note(f"Checkpoint tensor {hf_name} -> model parameter {custom_name}")
            raise
        loaded_params.add(hf_name)

    for name, (parts, expected) in packed_parts.items():
        if parts != expected:
            raise ValueError(f"Missing checkpoint shards for {name}: {sorted(expected - parts)}")
        loaded_params.add(name)

    # Check for model parameters that weren't loaded
    unloaded_params = set(dict(model.named_parameters())) - loaded_params
    if unloaded_params:
        raise ValueError(f"Unloaded model parameters: {sorted(unloaded_params)}")

    print(f"Weight loading: {len(dict(model.named_parameters()))} model parameters, "
          f"{len(hf_weights)} checkpoint tensors loaded")
    return loaded_params
