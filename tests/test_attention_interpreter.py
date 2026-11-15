"""Run actual Triton kernels on CPU in an isolated interpreter-mode process."""

import os
from pathlib import Path
import subprocess
import sys


def test_triton_attention_in_cpu_interpreter():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TRITON_INTERPRET": "1"},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def interpreter_checks():
    import torch
    from myvllm.layers.attention import paged_attention_decode, store_kvcache, paged_attention_prefill
    from test_paged_prefill import reference_prefill

    torch.manual_seed(31)
    # Real decode kernel: a graph padding row has no KV and must return zeros.
    q = torch.randn(2, 2, 16)
    kc, vc = torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16)
    out = paged_attention_decode(q, kc, vc, torch.zeros(2, 1, dtype=torch.int32),
                                 torch.tensor([3, 0]), .25, 2, 1, 16, 4)
    assert out.isfinite().all(), "zero-length graph padding produced NaN"
    assert torch.all(out[1] == 0)
    k, v = torch.randn(3, 1, 16), torch.randn(3, 1, 16)
    saved = kc.clone()
    store_kvcache(k, v, kc, vc, torch.tensor([1, -1, 3]), 4)
    torch.testing.assert_close(kc[0, 0], saved[0, 0])
    torch.testing.assert_close(kc[0, 1], k[0])
    torch.testing.assert_close(kc[0, 2], saved[0, 2])
    torch.testing.assert_close(kc[0, 3], k[2])
    # Unequal Q/K, GQA, ragged rows, noncontiguous blocks and a partial block.
    for block_size in (1, 4, 16):
        sizes = [(length + block_size - 1) // block_size for length in (11, 3, 26)]
        physical = torch.randperm(sum(sizes))
        tables = torch.full((3, max(sizes)), -1, dtype=torch.int32)
        offset = 0
        for i, n in enumerate(sizes):
            tables[i, :n] = physical[offset:offset + n]
            offset += n
        q = torch.randn(23, 4, 32)
        kc = torch.randn(sum(sizes), block_size, 2, 32)
        vc = torch.randn_like(kc)
        args = (
            q, kc, vc, torch.tensor([0, 5, 6, 23], dtype=torch.int32),
            torch.tensor([0, 11, 14, 40], dtype=torch.int32), tables,
            17, 32 ** -.5, 4, 2, 32, block_size,
        )
        torch.testing.assert_close(paged_attention_prefill(*args), reference_prefill(*args),
                                   rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    interpreter_checks()
