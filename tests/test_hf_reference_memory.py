import gc
import weakref

import torch
from transformers.integrations import sdpa_attention

from tools.compare_hf import kv_expansion


def test_explicit_kv_expansion_does_not_retain_past_key_tensors():
    original = sdpa_attention.use_gqa_in_sdpa
    with kv_expansion(True):
        key = torch.zeros(1, 8, 4, 128)
        reference = weakref.ref(key)
        assert sdpa_attention.use_gqa_in_sdpa(None, key) is False
        del key
        gc.collect()
        assert reference() is None
    assert sdpa_attention.use_gqa_in_sdpa is original


def test_default_kv_expansion_preserves_hf_policy():
    original = sdpa_attention.use_gqa_in_sdpa
    with kv_expansion(False):
        assert sdpa_attention.use_gqa_in_sdpa is original
