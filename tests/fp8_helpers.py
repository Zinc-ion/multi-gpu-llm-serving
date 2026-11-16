"""Independent FP8 quantization and attention references for M7 tests."""

import torch


def quantize_reference(values):
    values = values.float()
    amax = values.abs().amax(dim=-1)
    scales = torch.where(
        amax == 0,
        torch.ones_like(amax),
        torch.clamp(amax / 448.0, min=torch.finfo(torch.float32).tiny),
    )
    quantized = (values / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scales


def attention_reference(queries, keys, values, q_lengths, k_lengths, scale):
    """Causal suffix attention, computed in FP32 from the independent quantizer."""
    qkeys, kscales = quantize_reference(keys)
    qvalues, vscales = quantize_reference(values)
    keys = qkeys.float() * kscales.unsqueeze(-1)
    values = qvalues.float() * vscales.unsqueeze(-1)
    heads, kv_heads = queries.shape[1], keys.shape[1]
    head_map = torch.arange(heads, device=queries.device) // (heads // kv_heads)
    result = []
    q_start = k_start = 0
    for q_length, k_length in zip(q_lengths, k_lengths):
        if not q_length:
            continue
        q = queries[q_start:q_start + q_length].float().transpose(0, 1)
        k = keys[k_start:k_start + k_length, head_map].transpose(0, 1)
        v = values[k_start:k_start + k_length, head_map].transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = (torch.arange(k_length, device=queries.device)[None, :]
                <= k_length - q_length + torch.arange(q_length, device=queries.device)[:, None])
        scores = scores.masked_fill(~mask, float('-inf'))
        result.append(torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1))
        q_start += q_length
        k_start += k_length
    return torch.cat(result).to(queries.dtype)


def tolerances(dtype):
    if dtype == torch.float32:
        return {"rtol": 1e-4, "atol": 1e-5}
    if dtype == torch.float16:
        return {"rtol": 5e-3, "atol": 5e-3}
    return {"rtol": 2e-2, "atol": 2e-2}
