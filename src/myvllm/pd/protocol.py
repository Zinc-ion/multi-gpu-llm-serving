"""The JSON-compatible metadata exchanged before transferring KV bytes."""

import math
import sys

import torch

from myvllm.config import resolve_dtype


_ARCHITECTURE_FIELDS = (
    "vocab_size", "hidden_size", "num_heads", "num_qo_heads", "head_dim",
    "num_kv_heads", "num_layers", "intermediate_size", "rms_norm_epsilon",
    "qkv_bias", "has_attn_bias", "base", "rope_base", "scale", "max_position",
    "max_position_embeddings", "ffn_bias", "tie_word_embeddings", "block_size",
)
_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def model_spec(engine):
    config = engine.config
    if config["world_size"] != 1:
        raise ValueError("PD core requires world_size == 1")
    if not isinstance(config.get("model_revision"), str) or not config["model_revision"]:
        raise ValueError("PD core requires an explicit model_revision")
    cache = engine.model_runner.kv_cache
    dtype = resolve_dtype(config.get("dtype", "float32"))
    if dtype not in _DTYPE_NAMES or cache.dtype != dtype:
        raise ValueError("KV cache dtype does not match the configured PD dtype")
    block_size = config["block_size"]
    heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    shape = (2, config["num_layers"], len(engine.scheduler.block_manager.blocks),
             block_size, heads, head_dim)
    if tuple(cache.shape) != shape or block_size != engine.scheduler.block_manager.block_size:
        raise ValueError("KV cache shape does not match the scheduler and model")
    return {
        "name": config["model_name_or_path"],
        "revision": config["model_revision"],
        "tokenizer_revision": config.get("tokenizer_revision"),
        "type": config["model_type"],
        "dtype": _DTYPE_NAMES[dtype],
        "architecture": {key: config[key] for key in _ARCHITECTURE_FIELDS if key in config},
        "eos": sorted(engine.scheduler.eos),
    }


def payload_shape(spec, cached_tokens):
    arch = spec["architecture"]
    block_size = arch["block_size"]
    return [2, arch["num_layers"], (cached_tokens + block_size - 1) // block_size,
            block_size, arch["num_kv_heads"], arch["head_dim"]]


def payload_nbytes(spec, cached_tokens):
    dtype = _DTYPES[spec["dtype"]]
    return math.prod(payload_shape(spec, cached_tokens)) * dtype.itemsize


def tensor_dtype(spec):
    return _DTYPES[spec["dtype"]]


def make_metadata(handle, spec, sampling_seed):
    seq = handle.sequence
    cached = seq.num_cached_tokens
    return {
        "version": 2,
        "session_id": handle.session_id,
        "transaction_id": handle.transaction_id,
        "request_id": handle.request_id,
        "prefill_seq_id": seq.seq_id,
        "token_ids": list(seq.token_ids),
        "prompt_length": seq.num_prompt_tokens,
        "cached_tokens": cached,
        "next_position": cached,
        "completion_count": seq.num_completion_tokens,
        "sampling_seed": sampling_seed,
        "sampling": {
            "temperature": seq.temperature,
            "max_tokens": seq.max_tokens,
            "ignore_eos": seq.ignore_eos,
            "max_model_length": seq.max_model_length,
        },
        "model": spec,
        "kv": {
            "shape": payload_shape(spec, cached),
            "dtype": spec["dtype"],
            "byte_order": sys.byteorder,
            "nbytes": payload_nbytes(spec, cached),
        },
    }


def validate_metadata(metadata, session_id, spec, max_model_length):
    fields = {
        "version", "session_id", "transaction_id", "request_id", "prefill_seq_id",
        "token_ids", "prompt_length", "cached_tokens", "next_position",
        "completion_count", "sampling_seed", "sampling", "model", "kv",
    }
    if not isinstance(metadata, dict) or set(metadata) != fields:
        raise ValueError("Invalid PD metadata fields")
    if type(metadata["version"]) is not int or metadata["version"] != 2:
        raise ValueError("Unsupported PD protocol version")
    if metadata["session_id"] != session_id:
        raise ValueError("PD session does not match this handle")
    for field in ("transaction_id", "request_id"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"Invalid {field}")
    if type(metadata["prefill_seq_id"]) is not int or metadata["prefill_seq_id"] < 0:
        raise ValueError("Invalid prefill_seq_id")
    if metadata["model"] != spec:
        raise ValueError("Incompatible model, revision, dtype, architecture or EOS")
    if type(metadata["sampling_seed"]) is not int or not 0 <= metadata["sampling_seed"] < 2**32:
        raise ValueError("Invalid sampling_seed")

    sampling = metadata["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != {
        "temperature", "max_tokens", "ignore_eos", "max_model_length",
    }:
        raise ValueError("Invalid sampling metadata")
    temperature = sampling["temperature"]
    if not finite_number(temperature) or temperature < 0:
        raise ValueError("Invalid temperature")
    if type(sampling["max_tokens"]) is not int or sampling["max_tokens"] <= 1:
        raise ValueError("Invalid remaining completion budget")
    if type(sampling["ignore_eos"]) is not bool:
        raise ValueError("Invalid ignore_eos")
    limit = sampling["max_model_length"]
    if type(limit) is not int or not 1 < limit <= max_model_length:
        raise ValueError("Invalid max_model_length")

    prompt_length = metadata["prompt_length"]
    if type(prompt_length) is not int or not 0 < prompt_length < limit - 1:
        raise ValueError("Invalid prompt length or terminal first token")
    if any(type(metadata[field]) is not int or metadata[field] != prompt_length
           for field in ("cached_tokens", "next_position")):
        raise ValueError("Invalid cached progress or next position")
    if type(metadata["completion_count"]) is not int or metadata["completion_count"] != 1:
        raise ValueError("Invalid completion count")
    tokens = metadata["token_ids"]
    if not isinstance(tokens, list) or len(tokens) != prompt_length + 1:
        raise ValueError("Invalid token history length")
    vocab_size = spec["architecture"]["vocab_size"]
    if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise ValueError("Invalid token history")
    if not sampling["ignore_eos"] and tokens[-1] in spec["eos"]:
        raise ValueError("Terminal EOS cannot be handed off")
    expected = {
        "shape": payload_shape(spec, prompt_length),
        "dtype": spec["dtype"],
        "byte_order": sys.byteorder,
        "nbytes": payload_nbytes(spec, prompt_length),
    }
    kv = metadata["kv"]
    if (not isinstance(kv, dict) or set(kv) != set(expected)
            or not isinstance(kv["shape"], list)
            or any(type(dim) is not int for dim in kv["shape"])
            or type(kv["nbytes"]) is not int or kv != expected):
        raise ValueError("Invalid KV shape, dtype or byte count")
