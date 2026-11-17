from copy import deepcopy

import pytest
import torch

from myvllm.pd import CapacityError, DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from pd_fp8_helpers import fp8_engine, valid_kv
from test_pd_pipeline_core import assert_blocks
from test_pd_core import make_engine


def cores():
    return PrefillCore(fp8_engine(), 8, 1_000_000), DecodeCore(fp8_engine(), 8, 1_000_000)


def request(p, d, prompt, early=False):
    dh = d.open()
    method = p.prepare if early else p.open
    ph = method(dh.session_id, "fp8", prompt,
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
    return ph, dh


@pytest.mark.parametrize("length", [3, 4, 5])
def test_bulk_fp8_metadata_exact_bits_and_padding(length):
    p, d = cores()
    ph, dh = request(p, d, list(range(1, length + 1)))
    try:
        assert len(p.step()) == 1
        meta = p.describe(ph)
        kv = meta["kv"]
        assert meta["version"] == 4
        assert meta["model"]["kv_cache"] == {
            "dtype": "float8_e4m3fn", "scale_dtype": "float32",
            "scale_granularity": "token_head"}
        assert kv["scale_shape"] == kv["shape"][:-1]
        assert kv["nbytes"] == kv["data_nbytes"] + kv["scale_nbytes"]
        assert p.stats()["reserved_transfer_bytes"] == kv["nbytes"]
        assert d.reserve(dh, meta)
        payload = p.export(ph)
        assert len(payload) == kv["nbytes"]
        assert d.restore(dh, payload)
        assert all(torch.equal(a, b) for a, b in zip(
            valid_kv(p.engine, ph.sequence), valid_kv(d.engine, dh.sequence)))
        first_block = dh.sequence.block_table[0]
        assert torch.count_nonzero(d.engine.model_runner.kv_cache[0, 0, first_block, 0].view(torch.uint8)) == 0
        assert torch.all(d.engine.model_runner.kv_cache_scales[0, 0, first_block, 0] == 1)
        if length % 4:
            block = dh.sequence.block_table[-1]
            assert torch.count_nonzero(d.engine.model_runner.kv_cache[:, :, block, length % 4:].view(torch.uint8)) == 0
            assert torch.all(d.engine.model_runner.kv_cache_scales[:, :, block, length % 4:] == 1)
        assert p.acknowledge(ph)
        assert d.activate(dh)
        while dh.state == "active":
            d.step()
    finally:
        p.shutdown()
        d.shutdown()
    assert_blocks(p)
    assert_blocks(d)
    assert p.stats()["reserved_transfer_bytes"] == d.stats()["reserved_transfer_bytes"] == 0


@pytest.mark.parametrize("field", ["version", "format", "shape", "bytes"])
def test_bad_fp8_metadata_fails_before_reserving(field):
    p, d = cores()
    ph, dh = request(p, d, [1, 2, 3], early=True)
    try:
        meta = deepcopy(p.describe_early(ph))
        assert meta["version"] == 5
        if field == "version":
            meta["version"] = 3
        elif field == "format":
            meta["model"]["kv_cache"]["scale_granularity"] = "tensor"
        elif field == "shape":
            meta["kv"]["scale_shape"][-1] += 1
        else:
            meta["kv"]["scale_nbytes"] -= 1
        with pytest.raises(ValueError):
            d.reserve_early(dh, meta)
        assert d.stats()["used_blocks"] == d.stats()["reserved_blocks"] == 0
        assert d.stats()["reserved_transfer_bytes"] == 0
    finally:
        p.shutdown()
        d.shutdown()


def region_payload(engine, seq, layer, start, end):
    cache = engine.model_runner.kv_cache
    scales = engine.model_runner.kv_cache_scales
    data = []
    scale = []
    for pos in range(start, end):
        block = seq.block_table[pos // seq.block_size]
        offset = pos % seq.block_size
        data.append(cache[:, layer, block, offset])
        scale.append(scales[:, layer, block, offset])
    return (torch.stack(data, dim=1).contiguous().view(torch.uint8).numpy().tobytes()
            + torch.stack(scale, dim=1).contiguous().view(torch.uint8).numpy().tobytes())


@pytest.mark.parametrize("commit_first", [False, True])
def test_early_direct_regions_transfer_odd_unaligned_fp8_and_scales(commit_first):
    p, d = cores()
    ph, dh = request(p, d, [1, 2, 3, 4, 5], early=True)
    try:
        early = p.describe_early(ph)
        assert d.reserve_early(dh, early)
        assert not d.reserve_early(dh, early)
        assert p.ready(ph)
        assert p.start(ph)
        assert len(p.step()) == 1
        meta = p.describe(ph)
        assert meta["version"] == 4
        if commit_first:
            assert d.commit_early(dh, meta)
        for layer in range(2):
            for start, end in ((0, 1), (1, 4), (4, 5)):
                payload = region_payload(p.engine, ph.sequence, layer, start, end)
                assert len(payload) == 14 * (end - start)
                assert d.restore_region(dh, layer, start, end, payload)
        if not commit_first:
            assert d.commit_early(dh, meta)
        assert dh.state == "restored"
        assert all(torch.equal(a, b) for a, b in zip(
            valid_kv(p.engine, ph.sequence), valid_kv(d.engine, dh.sequence)))
        block = dh.sequence.block_table[-1]
        assert torch.count_nonzero(d.engine.model_runner.kv_cache[:, :, block, 1:].view(torch.uint8)) == 0
        assert torch.all(d.engine.model_runner.kv_cache_scales[:, :, block, 1:] == 1)
        assert p.acknowledge_early(ph)
        assert d.activate(dh)
        while dh.state == "active":
            d.step()
    finally:
        p.shutdown()
        d.shutdown()
    assert_blocks(p)
    assert_blocks(d)


def test_fp8_scale_bytes_count_toward_both_transfer_budgets():
    p = PrefillCore(fp8_engine(), 8, 223)
    d = DecodeCore(fp8_engine(), 8, 223)
    dh = d.open()
    params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
    try:
        with pytest.raises(CapacityError):
            p.open(dh.session_id, "too-large", [1, 2, 3, 4, 5], params)
        source = PrefillCore(fp8_engine(), 8, 224)
        try:
            ph = source.open(dh.session_id, "large", [1, 2, 3, 4, 5], params)
            source.step()
            meta = source.describe(ph)
            assert meta["kv"]["data_nbytes"] == 96
            assert meta["kv"]["scale_nbytes"] == 128
            with pytest.raises(CapacityError):
                d.reserve(dh, meta)
            assert d.stats()["used_blocks"] == d.stats()["reserved_blocks"] == 0
            assert d.stats()["reserved_transfer_bytes"] == 0
        finally:
            source.shutdown()
    finally:
        p.shutdown()
        d.shutdown()


def test_cross_format_metadata_rejected_before_reservation():
    p = PrefillCore(fp8_engine(), 8, 1_000_000)
    d = DecodeCore(make_engine(), 8, 1_000_000)
    ph, dh = request(p, d, [1, 2, 3], early=True)
    try:
        with pytest.raises(ValueError):
            d.reserve_early(dh, p.describe_early(ph))
        assert d.stats()["used_blocks"] == d.stats()["reserved_blocks"] == 0
        assert d.stats()["reserved_transfer_bytes"] == 0
    finally:
        p.shutdown()
        d.shutdown()


def test_early_partial_tail_is_reset_after_real_block_reuse():
    p, d = cores()
    first_block = None
    for _ in range(16):
        ph, dh = request(p, d, [1, 2, 3, 4])
        p.step()
        assert d.reserve(dh, p.describe(ph))
        assert d.restore(dh, p.export(ph))
        if first_block is None:
            first_block = dh.sequence.block_table[0]
            assert torch.count_nonzero(d.engine.model_runner.kv_cache[:, :, first_block, 1:].view(torch.uint8)) > 0
            assert torch.any(d.engine.model_runner.kv_cache_scales[:, :, first_block, 1:] != 1)
        p.acknowledge(ph)
        d.cancel(dh)
        p.close(ph)
        d.close(dh)
    ph, dh = request(p, d, [5], early=True)
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        block = dh.sequence.block_table[0]
        assert block == first_block
        assert torch.count_nonzero(d.engine.model_runner.kv_cache[:, :, block, 1:].view(torch.uint8)) == 0
        assert torch.all(d.engine.model_runner.kv_cache_scales[:, :, block, 1:] == 1)
    finally:
        p.shutdown()
        d.shutdown()
    assert_blocks(p)
    assert_blocks(d)
