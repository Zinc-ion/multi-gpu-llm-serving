import pickle
import random

import pytest

from myvllm.engine.input_batch import build_prefill_batch
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from test_scheduler_lifecycle import assert_block_pool_consistent


def scheduler(budget=3, sequences=4, blocks=8, block_size=4, model_length=16):
    return Scheduler(sequences, budget, blocks, block_size, [0], model_length,
                     enable_chunked_prefill=True)


def samples(seqs, token=99):
    return [token for seq in seqs
            if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)]


def test_chunking_relaxes_only_the_batch_budget_constraint():
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        Scheduler(4, 3, 8, 4, [0], 16)
    with pytest.raises(ValueError, match="KV cache"):
        scheduler(blocks=1, model_length=16)
    assert scheduler().max_model_length == 16


def test_partial_chunks_do_not_generate_and_use_exact_non_aligned_slots():
    sched = scheduler()
    seq = Sequence(list(range(8)), 4, SamplingParams(max_tokens=2))
    sched.add_sequence(seq)
    all_positions, all_slots = [], []
    for step, expected_size in enumerate([3, 3, 2]):
        batch, is_prefill = sched.schedule()
        assert is_prefill
        assert batch == [seq]
        metadata = build_prefill_batch(batch, 4)
        transported = build_prefill_batch(pickle.loads(pickle.dumps(batch)), 4)
        assert transported == metadata
        assert len(metadata["input_ids"]) == expected_size
        assert metadata["cu_seqlens_q"] == [0, expected_size]
        assert metadata["cu_seqlens_k"] == [0, min((step + 1) * 3, 8)]
        assert metadata["logit_indices"] == ([1] if step == 2 else [])
        all_positions.extend(metadata["positions"])
        all_slots.extend(metadata["slot_mapping"])
        sched.postprocess(batch, samples(batch))
        assert seq.num_completion_tokens == (1 if step == 2 else 0)
        assert_block_pool_consistent(sched)
    assert all_positions == list(range(8))
    assert all_slots == list(range(8))
    batch, is_prefill = sched.schedule()
    assert not is_prefill
    sched.postprocess(batch, [0])
    assert sched.is_finished()
    assert not sched.block_manager.used_block_ids


def test_decode_advances_in_every_mixed_batch():
    sched = scheduler(budget=4)
    short = Sequence([1], 4, SamplingParams(max_tokens=5))
    sched.add_sequence(short)
    batch, _ = sched.schedule()
    sched.postprocess(batch, samples(batch))
    long = Sequence(list(range(9)), 4, SamplingParams(max_tokens=1))
    sched.add_sequence(long)
    for expected_cached in (3, 6, 9):
        batch, is_prefill = sched.schedule()
        assert is_prefill
        assert batch == [short, long]
        assert [seq.num_scheduled_tokens for seq in batch] == [1, 3]
        metadata = build_prefill_batch(batch, 4)
        assert build_prefill_batch(pickle.loads(pickle.dumps(batch)), 4) == metadata
        assert metadata["logit_indices"] == ([0, 3] if expected_cached == 9 else [0])
        before = short.num_completion_tokens
        sched.postprocess(batch, samples(batch))
        assert short.num_completion_tokens == before + 1
        assert long.is_finished == (expected_cached == 9)
        assert_block_pool_consistent(sched)
    assert sched.cancel(short.seq_id)
    assert sched.is_finished()


def test_resident_limit_avoids_decode_starvation():
    sched = scheduler(budget=2, sequences=2)
    seqs = [Sequence([i], 4, SamplingParams(max_tokens=4)) for i in range(4)]
    for seq in seqs:
        sched.add_sequence(seq)
    for _ in range(4):
        batch, _ = sched.schedule()
        assert batch == seqs[:2]
        sched.postprocess(batch, samples(batch))
        assert len(sched.running) <= 2
    assert all(seq.is_finished for seq in seqs[:2])
    batch, _ = sched.schedule()
    assert batch == seqs[2:]
    for seq in seqs[2:]:
        assert sched.cancel(seq.seq_id)
    assert_block_pool_consistent(sched)


def test_cancel_partial_prefill_releases_all_reserved_blocks():
    sched = scheduler(budget=3)
    seq = Sequence(list(range(11)), 4)
    sched.add_sequence(seq)
    batch, _ = sched.schedule()
    sched.postprocess(batch, [])
    assert seq.num_cached_tokens == 3
    assert seq.num_completion_tokens == 0
    assert len(seq.block_table) == 3
    assert sched.cancel(seq.seq_id)
    assert seq.status == SequenceStatus.CANCELLED
    assert sched.is_finished()
    assert not sched.block_manager.used_block_ids
    assert_block_pool_consistent(sched)


def test_pickle_retains_generated_prefix_for_recompute_after_preemption():
    sched = scheduler(budget=3, blocks=2, model_length=8)
    seqs = [Sequence([i] * 2, 4) for i in (1, 2)]
    for seq in seqs:
        sched.add_sequence(seq)
    for _ in range(4):
        batch, _ = sched.schedule()
        sched.postprocess(batch, samples(batch))
    assert seqs[1].status == SequenceStatus.WAITING
    assert seqs[1].num_completion_tokens > 0
    # The public preempt/cancel lifecycle resets cached progress, not the tokens.
    assert sched.cancel(seqs[0].seq_id)
    batch, _ = sched.schedule()
    restored = pickle.loads(pickle.dumps(batch[0]))
    assert restored.token_ids == seqs[1].token_ids
    assert restored.num_cached_tokens == 0
    assert restored.num_scheduled_tokens == 3
    assert sched.cancel(seqs[1].seq_id)
    assert_block_pool_consistent(sched)


def test_randomized_chunk_preemption_and_cancellation_conserve_resources():
    preemptions = partial_chunks = mixed_batches = 0
    for seed in range(50):
        rng = random.Random(seed)
        sched = scheduler(budget=3, blocks=3, block_size=2, model_length=6)
        seqs = [Sequence([i + 1] * rng.randint(1, 4), 2,
                         SamplingParams(max_tokens=5)) for i in range(8)]
        for seq in seqs:
            sched.add_sequence(seq)
        for _ in range(300):
            if sched.is_finished():
                break
            was_running = set(sched.running)
            batch, is_prefill = sched.schedule()
            preemptions += sum(seq.status == SequenceStatus.WAITING for seq in was_running)
            assert sum(seq.num_scheduled_tokens for seq in batch) <= 3
            assert len(sched.running) <= 3
            complete = samples(batch)
            partial_chunks += len(batch) - len(complete)
            mixed_batches += is_prefill and 0 < len(complete) < len(batch)
            sched.postprocess(batch, [0 if rng.random() < .05 else 99 for _ in complete])
            active = list(sched.waiting) + list(sched.running)
            if active and rng.random() < .08:
                sched.cancel(rng.choice(active).seq_id)
            assert_block_pool_consistent(sched)
            for seq in seqs:
                assert int(seq in sched.waiting) + int(seq in sched.running) == (not seq.is_finished)
        else:
            pytest.fail(f"chunked scheduler stalled for seed {seed}")
        assert not sched.block_manager.used_block_ids
    assert preemptions > 0
    assert partial_chunks > 0
    assert mixed_batches > 0
