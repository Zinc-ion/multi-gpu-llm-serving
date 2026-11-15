"""The scheduler-to-worker protocol must retain decode block geometry."""
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import pytest
import torch

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import get_context, reset_context


@pytest.fixture
def worker_boundary(monkeypatch):
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(*args, **kwargs)

    # Only replace CUDA transport; shared memory, pickle and input preparation
    # are the production implementations used by TP workers.
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    event = mp.get_context("spawn").Event()
    shared = SharedMemory(create=True, size=2**20)
    master, worker = ModelRunner.__new__(ModelRunner), ModelRunner.__new__(ModelRunner)
    master.rank, worker.rank = 0, 1
    master.world_size = worker.world_size = 2
    master.block_size = worker.block_size = 4
    master.timings = worker.timings = None
    master.event, worker.event = [event], event
    master.shm = worker.shm = shared

    def prepare(batch, is_prefill):
        master.write_shm("run", (batch, is_prefill))
        method, (restored, restored_prefill) = worker.read_shm()
        assert method == "run" and restored_prefill == is_prefill
        input_ids = (worker.prepare_prefill(restored) if restored_prefill
                     else worker.prepare_decode(restored))
        if not restored_prefill:
            assert all(seq.block_size == 4 for seq in restored)
        return input_ids, restored, get_context()

    try:
        yield prepare
    finally:
        reset_context()
        shared.close()
        shared.unlink()


def assert_blocks_conserved(scheduler):
    manager = scheduler.block_manager
    free, used = set(manager.free_block_ids), manager.used_block_ids
    assert not free & used
    assert free | used == set(range(len(manager.blocks)))
    for block_id in free:
        block = manager.blocks[block_id]
        assert block.ref_count == 0 and block.hash == -1


def test_tp_worker_prefill_then_decode_crosses_block_and_cancels(worker_boundary):
    scheduler = Scheduler(2, 16, 8, 4, [66], 16)
    first = Sequence([1, 2, 3], 4, SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    cancelled = Sequence([4, 5, 6, 7], 4, SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
    for seq in (first, cancelled):
        scheduler.add_sequence(seq)
    batch, is_prefill = scheduler.schedule()
    assert is_prefill
    ids, restored, context = worker_boundary(batch, is_prefill)
    assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert [seq.token_ids for seq in restored] == [[1, 2, 3], [4, 5, 6, 7]]
    assert context.positions.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert context.slot_mapping.tolist() == [0, 1, 2, 4, 5, 6, 7]
    assert context.cu_seqlens_q.tolist() == context.cu_seqlens_k.tolist() == [0, 3, 7]
    scheduler.postprocess(batch, [8, 9])
    assert_blocks_conserved(scheduler)

    # One request fills its first block; the other needs a new physical block.
    batch, is_prefill = scheduler.schedule()
    assert not is_prefill and batch == [first, cancelled]
    ids, restored, context = worker_boundary(batch, is_prefill)
    assert ids.tolist() == [8, 9]
    assert [seq.token_ids for seq in restored] == [[8], [9]]
    assert [seq.last_block_num_tokens for seq in restored] == [4, 1]
    assert context.context_lens.tolist() == [4, 5]
    assert context.positions.tolist() == [3, 4]
    assert context.block_tables.tolist() == [[0, -1], [1, 2]]
    assert context.slot_mapping.tolist() == [3, 8]
    scheduler.postprocess(batch, [10, 11])
    assert scheduler.cancel(cancelled.seq_id)
    assert cancelled.status == SequenceStatus.CANCELLED and not cancelled.block_table
    assert not scheduler.cancel(cancelled.seq_id)
    assert_blocks_conserved(scheduler)

    batch, is_prefill = scheduler.schedule()
    assert not is_prefill and batch == [first]
    ids, restored, context = worker_boundary(batch, is_prefill)
    assert ids.tolist() == [10] and restored[0].last_block_num_tokens == 1
    assert context.context_lens.tolist() == [5]
    assert context.positions.tolist() == [4]
    assert context.block_tables.tolist() == [[0, 3]]
    assert context.slot_mapping.tolist() == [12]
    scheduler.postprocess(batch, [12])
    assert first.completion_token_ids == [8, 10, 12]
    assert first.status == SequenceStatus.FINISHED and scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids
    assert_blocks_conserved(scheduler)
