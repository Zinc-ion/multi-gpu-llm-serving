"""Copy only cached positions through owned, ordinary host bytes."""

import torch

from myvllm.pd.protocol import payload_byte_counts, payload_shape, tensor_dtype


def export_kv(cache, block_table, cached_tokens, spec, scales=None):
    shape = payload_shape(spec, cached_tokens)
    host = torch.zeros(shape, dtype=cache.dtype, device="cpu")
    host_scales = torch.ones(shape[:-1], dtype=torch.float32, device="cpu") if scales is not None else None
    block_size = shape[3]
    for i in range(shape[2]):
        valid = min(block_size, cached_tokens - i * block_size)
        host[:, :, i, :valid].copy_(cache[:, :, block_table[i], :valid], non_blocking=False)
        if host_scales is not None:
            host_scales[:, :, i, :valid].copy_(scales[:, :, block_table[i], :valid], non_blocking=False)
    data = host.view(torch.uint8).numpy().tobytes()
    return data + host_scales.view(torch.uint8).numpy().tobytes() if host_scales is not None else data


def restore_kv(cache, block_table, payload, spec, cached_tokens, scales=None):
    shape = payload_shape(spec, cached_tokens)
    data_nbytes, _ = payload_byte_counts(spec, cached_tokens)
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    host = raw[:data_nbytes].view(tensor_dtype(spec)).reshape(shape)
    host_scales = None
    if scales is not None:
        host_scales = torch.empty(shape[:-1], dtype=torch.float32, device="cpu")
        host_scales.view(torch.uint8).reshape(-1).copy_(raw[data_nbytes:])
    for i, block_id in enumerate(block_table[:shape[2]]):
        cache[:, :, block_id].copy_(host[:, :, i], non_blocking=False)
        if host_scales is not None:
            scales[:, :, block_id].copy_(host_scales[:, :, i], non_blocking=False)
