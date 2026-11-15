"""Copy only cached positions through owned, ordinary host bytes."""

import torch

from myvllm.pd.protocol import payload_shape, tensor_dtype


def export_kv(cache, block_table, cached_tokens, spec):
    shape = payload_shape(spec, cached_tokens)
    host = torch.zeros(shape, dtype=cache.dtype, device="cpu")
    block_size = shape[3]
    for i in range(shape[2]):
        valid = min(block_size, cached_tokens - i * block_size)
        host[:, :, i, :valid].copy_(cache[:, :, block_table[i], :valid], non_blocking=False)
    return host.view(torch.uint8).numpy().tobytes()


def restore_kv(cache, block_table, payload, spec, cached_tokens):
    shape = payload_shape(spec, cached_tokens)
    host = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(tensor_dtype(spec)).reshape(shape)
    for i, block_id in enumerate(block_table[:shape[2]]):
        cache[:, :, block_id].copy_(host[:, :, i], non_blocking=False)
