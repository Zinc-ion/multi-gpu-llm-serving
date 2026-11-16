"""Gather TP prefill KV shards as raw host bytes for the TP1 bulk format."""

import time

import torch
import torch.distributed as dist

from myvllm.pd.protocol import payload_byte_counts, payload_shape


def export_tp_kv(cache, scales, block_table, cached_tokens, spec, rank, group):
    shape = payload_shape(spec, cached_tokens)
    local_shape = list(shape)
    local_shape[-2] //= 2
    data_bytes, scale_bytes = payload_byte_counts(spec, cached_tokens)
    local_data_bytes, local_scale_bytes = data_bytes // 2, scale_bytes // 2

    started = time.perf_counter()
    shard = torch.zeros(local_data_bytes + local_scale_bytes, dtype=torch.uint8, device='cpu')
    data = shard[:local_data_bytes].view(cache.dtype).reshape(local_shape)
    host_scales = (torch.ones(local_shape[:-1], dtype=torch.float32, device='cpu')
                   if scales is not None else None)
    for i in range(shape[2]):
        valid = min(shape[3], cached_tokens - i * shape[3])
        data[:, :, i, :valid].copy_(cache[:, :, block_table[i], :valid], non_blocking=False)
        if host_scales is not None:
            host_scales[:, :, i, :valid].copy_(scales[:, :, block_table[i], :valid], non_blocking=False)
    if host_scales is not None:
        shard[local_data_bytes:].copy_(host_scales.view(torch.uint8).reshape(-1))
    del data, host_scales
    export_ms = (time.perf_counter() - started) * 1000
    padded_tokens = shape[2] * shape[3]
    local_metrics = torch.tensor([local_data_bytes, local_scale_bytes,
                                  local_data_bytes * cached_tokens // padded_tokens,
                                  local_scale_bytes * cached_tokens // padded_tokens, export_ms],
                                 dtype=torch.float64, device='cpu')

    started = time.perf_counter()
    shards = [torch.empty_like(shard) for _ in range(2)] if rank == 0 else None
    metrics = [torch.empty_like(local_metrics) for _ in range(2)] if rank == 0 else None
    dist.gather(shard, gather_list=shards, dst=0, group=group)
    dist.gather(local_metrics, gather_list=metrics, dst=0, group=group)
    if rank:
        return None

    packed = torch.empty(data_bytes + scale_bytes, dtype=torch.uint8, device='cpu')
    data = packed[:data_bytes].view(cache.dtype).reshape(shape)
    full_scales = torch.empty(shape[:-1], dtype=torch.float32, device='cpu') if scale_bytes else None
    for index, part in enumerate(shards):
        first = index * local_shape[-2]
        last = first + local_shape[-2]
        data[..., first:last, :].copy_(part[:local_data_bytes].view(cache.dtype).reshape(local_shape))
        if full_scales is not None:
            local_scales = torch.empty(local_shape[:-1], dtype=torch.float32, device='cpu')
            local_scales.view(torch.uint8).reshape(-1).copy_(part[local_data_bytes:])
            full_scales[..., first:last].copy_(local_scales)
    if full_scales is not None:
        packed[data_bytes:].copy_(full_scales.view(torch.uint8).reshape(-1))
    payload = packed.numpy().tobytes()
    return payload, {
        'ranks': [{'rank': index, 'data_bytes': int(values[0]), 'scale_bytes': int(values[1]),
                   'valid_data_bytes': int(values[2]), 'valid_scale_bytes': int(values[3]),
                   'export_host_ms': float(values[4])} for index, values in enumerate(metrics)],
        'gather_pack_host_ms': (time.perf_counter() - started) * 1000,
    }
