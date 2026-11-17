import time
from types import SimpleNamespace

import pytest
import torch

from myvllm.engine.scheduler import Scheduler
from myvllm.pd import DecodeCore, PrefillCore
from myvllm.pd.pipeline import LayerReceiver
from myvllm.sampling_parameters import SamplingParams
from pd_fp8_helpers import fp8_engine, valid_kv
from test_pd_fp8_core import region_payload
from test_pd_pipeline_core import assert_blocks


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_cancelled_fp8_tail_clear_precedes_reused_block_receiver():
    p = PrefillCore(fp8_engine(), 3, 1_000_000)
    shape = (2, 2, 1, 4, 1, 3)
    engine = SimpleNamespace(
        config={**p.engine.config, "max_cached_blocks": 1},
        scheduler=Scheduler(1, 4, 1, 4, [62, 63], 4),
        model_runner=SimpleNamespace(
            kv_cache=torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda"),
            kv_cache_scales=torch.ones(shape[:-1], dtype=torch.float32, device="cuda"),
        ),
        request_ids={},
    )
    d = DecodeCore(engine, 3, 1_000_000)
    params = SamplingParams(temperature=0, max_tokens=2, ignore_eos=True, max_model_length=4)
    a, b = d.open(), d.open()
    pa = p.prepare(a.session_id, "cancelled", [1], params)
    pb = p.prepare(b.session_id, "reused", [5, 6], params)
    early_a, early_b = p.describe_early(pa), p.describe_early(pb)
    try:
        assert p.ready(pb)
        assert p.start(pb)
        assert not p.step()[0].finished
        payloads = [region_payload(p.engine, pb.sequence, layer, 0, 2) for layer in range(2)]
        expected_data, expected_scales = valid_kv(p.engine, pb.sequence)
        assert torch.count_nonzero(expected_data[1]) > 0
        assert torch.all(expected_scales[1].view(torch.float32) != 1)

        # Reuse already allocated staging resources, as in a long-running worker.
        warm_receiver = LayerReceiver(engine.model_runner.kv_cache, 28, {},
                                      scales=engine.model_runner.kv_cache_scales)
        warm_receiver.restore([0], 0, 0, 2, payloads[0])
        warm_receiver.restore([0], 1, 0, 2, payloads[1])
        warm_receiver.finish()
        warm_receiver.close()
        for prompt in ([1], [5, 6]):
            warm_d = d.open()
            warm_p = p.prepare(warm_d.session_id, "warmup", prompt, params)
            assert d.reserve_early(warm_d, p.describe_early(warm_p))
            assert d.close(warm_d)
            assert p.close(warm_p)
        torch.cuda._sleep(10_000)
        torch.cuda.synchronize()
        current = torch.cuda.current_stream()
        with torch.cuda.stream(current):
            started = time.perf_counter()
            torch.cuda._sleep(2_000_000_000)
            assert d.reserve_early(a, early_a)
            block_a = a.sequence.block_table[0]
            tail_done = torch.cuda.Event()
            tail_done.record(current)
            assert not tail_done.query()
            assert d.cancel(a)
            assert d.close(a)
            assert not tail_done.query(), "A's tail clear must still be queued when its block is freed"

            assert d.reserve_early(b, early_b)
            block_b = b.sequence.block_table[0]
            assert block_a == block_b == 0
            receiver = d.attach_receiver(b, frame_bytes=28)
            assert receiver.timing_path is None
            assert not tail_done.query(), "A's tail clear must still be queued when B attaches"
            print(f"attached_ms={(time.perf_counter() - started)*1000:.2f}")
            for layer, payload in enumerate(payloads):
                assert d.restore_region(b, layer, 0, 2, payload)
                print(f"restored_layer={layer} elapsed_ms={(time.perf_counter() - started)*1000:.2f} "
                      f"a_clear_pending={not tail_done.query()}")
            copy_beat_tail = not tail_done.query()
            current.synchronize()
            actual_data, actual_scales = valid_kv(engine, b.sequence)
            actual_data, actual_scales = actual_data.cpu(), actual_scales.cpu()
        print(f"reused_block={block_b} a_clear_pending_after_receiver={copy_beat_tail}")
        print(f"expected_data={expected_data.tolist()} actual_data={actual_data.tolist()}")
        print(f"expected_scales={expected_scales.view(torch.float32).tolist()} "
              f"actual_scales={actual_scales.view(torch.float32).tolist()}")
        assert torch.equal(actual_data, expected_data) and torch.equal(actual_scales, expected_scales)
        assert d.commit_early(b, p.describe(pb))
        assert p.acknowledge_early(pb)
        assert d.activate(b)
        assert d.cancel(b)
    finally:
        p.shutdown()
        d.shutdown()
        torch.cuda.synchronize()
    assert_blocks(p)
    assert_blocks(d)
    assert d.stats()["used_blocks"] == d.stats()["reserved_blocks"] == 0
