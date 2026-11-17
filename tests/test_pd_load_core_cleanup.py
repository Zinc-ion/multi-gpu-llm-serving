import errno

import pytest

from myvllm.pd import DecodeCore, PrefillCore
from test_pd_core import make_engine
from test_pd_pipeline_core import first_token, make_cores, payload_for, prepare


@pytest.mark.parametrize("operation", ["cancel", "close"])
def test_direct_decode_core_releases_capacity_after_receiver_timing_failure(tmp_path, operation):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5])
    path = tmp_path / "receiver.jsonl"
    path.symlink_to("/dev/full")
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        receiver = d.attach_receiver(dh, 64, str(path))
        with pytest.raises(OSError) as error:
            getattr(d, operation)(dh)
        assert error.value.errno == errno.ENOSPC
        stats = d.stats()
        assert stats["used_blocks"] == stats["reserved_blocks"] == stats["resident_requests"] == 0
        assert stats["reserved_transfer_bytes"] == 0 and not receiver.slots
        if operation == "cancel":
            assert d.close(dh)
        assert d.stats()["open_handles"] == 0
    finally:
        path.unlink()
        p.shutdown()
        d.shutdown()


def test_shutdown_releases_other_handles_after_receiver_timing_failure(tmp_path):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5])
    other_ph, other_dh = prepare(p, d, [6, 7, 8])
    path = tmp_path / "receiver.jsonl"
    path.symlink_to("/dev/full")
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        assert d.attach_receiver(dh, 64, str(path))
        assert d.reserve_early(other_dh, p.describe_early(other_ph))
        with pytest.raises(OSError) as error:
            d.shutdown()
        assert error.value.errno == errno.ENOSPC
        stats = d.stats()
        assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0
        assert stats["resident_requests"] == stats["reserved_transfer_bytes"] == 0
    finally:
        path.unlink()
        p.shutdown()
        d.shutdown()


def test_expire_releases_other_handles_after_receiver_timing_failure(tmp_path):
    engine = make_engine()
    engine.config["enforce_eager"] = True
    p = PrefillCore(engine, 2, 1_000_000)
    d = DecodeCore(make_engine(), 2, 1_000_000)
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5], deadline=10)
    other_ph, other_dh = prepare(p, d, [6, 7, 8], deadline=10)
    path = tmp_path / "receiver.jsonl"
    path.symlink_to("/dev/full")
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        assert d.attach_receiver(dh, 64, str(path))
        assert d.reserve_early(other_dh, p.describe_early(other_ph))
        with pytest.raises(OSError) as error:
            d.expire(now=10)
        assert error.value.errno == errno.ENOSPC
        stats = d.stats()
        assert stats["used_blocks"] == stats["reserved_blocks"] == stats["resident_requests"] == 0
        assert stats["reserved_transfer_bytes"] == 0
        assert stats["open_handles"] == 0
        assert d.expire(now=10) == []
        replacement = d.open(deadline=20)
        assert d.close(replacement)
    finally:
        path.unlink()
        p.shutdown()
        d.shutdown()


def test_expire_timing_failure_keeps_unexpired_handle(tmp_path):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5], deadline=10)
    later = d.open(deadline=20)
    path = tmp_path / "receiver.jsonl"
    path.symlink_to("/dev/full")
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        assert d.attach_receiver(dh, 64, str(path))
        with pytest.raises(OSError) as error:
            d.expire(now=10)
        assert error.value.errno == errno.ENOSPC
        assert d.stats()["open_handles"] == 1
        assert d.close(later)
    finally:
        path.unlink()
        p.shutdown()
        d.shutdown()


def test_finished_handle_closes_after_receiver_timing_failure(tmp_path):
    p, d = make_cores()
    ph, dh = prepare(p, d, [1, 2, 3, 4, 5])
    path = tmp_path / "receiver.jsonl"
    try:
        assert d.reserve_early(dh, p.describe_early(ph))
        receiver = d.attach_receiver(dh, 256, str(path))
        assert p.ready(ph) and p.start(ph)
        assert not first_token(p, ph).finished
        for layer in range(2):
            for start, end in ((0, 3), (3, 5)):
                assert d.restore_region(dh, layer, start, end, payload_for(p, ph, layer, start, end))
        assert d.commit_early(dh, p.describe(ph))
        assert p.acknowledge_early(ph) and d.activate(dh)
        while dh.state == "active":
            d.step()
        assert dh.state == "finished"
        path.unlink()
        path.symlink_to("/dev/full")
        with pytest.raises(OSError) as error:
            d.close(dh)
        assert error.value.errno == errno.ENOSPC
        stats = d.stats()
        assert stats["open_handles"] == stats["used_blocks"] == stats["reserved_blocks"] == 0
        assert stats["resident_requests"] == stats["reserved_transfer_bytes"] == 0
        assert not receiver.slots
    finally:
        path.unlink(missing_ok=True)
        p.shutdown()
        d.shutdown()
