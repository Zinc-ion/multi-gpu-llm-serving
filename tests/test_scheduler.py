import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=4,
    max_model_length=None,
):
    if max_model_length is None:
        max_model_length = min(max_num_batched_tokens, max_cached_blocks * block_size)
    return Scheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
        max_model_length=max_model_length,
    )


def make_sequence(
    token_ids: list[int],
    block_size: int = 4,
    sampling_params: SamplingParams | None = None,
) -> Sequence:
    return Sequence(token_ids, block_size=block_size, sampling_params=sampling_params)


def assert_block_pool_consistent(scheduler: Scheduler) -> None:
    manager = scheduler.block_manager
    all_ids = set(range(len(manager.blocks)))
    free_ids = set(manager.free_block_ids)
    assert free_ids.isdisjoint(manager.used_block_ids)
    assert free_ids | manager.used_block_ids == all_ids
    for block_id, block in enumerate(manager.blocks):
        if block_id in free_ids:
            assert block.ref_count == 0
            assert block.hash == -1
        else:
            assert block.ref_count > 0


def test_scheduler_rejects_an_impossible_model_length():
    with pytest.raises(ValueError, match="max_model_length"):
        make_scheduler(max_num_batched_tokens=4, max_model_length=5)

    with pytest.raises(ValueError, match="KV cache"):
        make_scheduler(max_cached_blocks=1, block_size=4, max_model_length=5)


def test_prefill_allocates_real_blocks():
    scheduler = make_scheduler(max_num_batched_tokens=8, max_cached_blocks=2)
    sequence = make_sequence([1, 2, 3, 4])
    scheduler.add_sequence(sequence)

    scheduled, is_prefill = scheduler.schedule()

    assert is_prefill
    assert scheduled == [sequence]
    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.max_model_length == scheduler.max_model_length
    assert len(sequence.block_table) == 1
    assert scheduler.block_manager.blocks[sequence.block_table[0]].ref_count == 1
    assert_block_pool_consistent(scheduler)


def test_eos_releases_all_blocks():
    scheduler = make_scheduler(max_num_batched_tokens=8, max_cached_blocks=2)
    sequence = make_sequence([1, 2, 3, 4])
    scheduler.add_sequence(sequence)
    scheduled, _ = scheduler.schedule()

    scheduler.postprocess(scheduled, [scheduler.eos])

    assert sequence.status == SequenceStatus.FINISHED
    assert sequence.block_table == []
    assert scheduler.is_finished()
    assert scheduler.block_manager.used_block_ids == set()
    assert_block_pool_consistent(scheduler)


def test_max_tokens_and_max_model_length_release_blocks():
    token_scheduler = make_scheduler(max_num_batched_tokens=8, max_cached_blocks=2)
    token_sequence = make_sequence(
        [1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1, max_model_length=8),
    )
    token_scheduler.add_sequence(token_sequence)
    scheduled, _ = token_scheduler.schedule()
    token_scheduler.postprocess(scheduled, [99])
    assert token_sequence.status == SequenceStatus.FINISHED
    assert token_scheduler.block_manager.used_block_ids == set()

    length_scheduler = make_scheduler(max_num_batched_tokens=8, max_cached_blocks=2)
    length_sequence = make_sequence(
        [1, 2, 3], sampling_params=SamplingParams(max_model_length=4)
    )
    length_scheduler.add_sequence(length_sequence)
    scheduled, _ = length_scheduler.schedule()
    length_scheduler.postprocess(scheduled, [99])
    assert length_sequence.status == SequenceStatus.FINISHED
    assert length_sequence.num_tokens == 4
    assert length_scheduler.block_manager.used_block_ids == set()


def test_nonterminal_decode_keeps_running_order():
    scheduler = make_scheduler(
        max_num_batched_tokens=8, max_num_sequences=2, max_cached_blocks=2
    )
    first = make_sequence(
        [1, 2], sampling_params=SamplingParams(max_tokens=4, max_model_length=8)
    )
    second = make_sequence(
        [3, 4], sampling_params=SamplingParams(max_tokens=4, max_model_length=8)
    )
    scheduler.add_sequence(first)
    scheduler.add_sequence(second)
    scheduled, _ = scheduler.schedule()

    scheduler.postprocess(scheduled, [99, 100])

    assert list(scheduler.running) == [first, second]
    assert first.status == SequenceStatus.RUNNING
    assert second.status == SequenceStatus.RUNNING
    assert_block_pool_consistent(scheduler)


def test_prompt_at_model_limit_is_rejected_before_prefill():
    scheduler = make_scheduler(max_num_batched_tokens=8, max_cached_blocks=2)
    sequence = make_sequence(
        [1, 2, 3, 4], sampling_params=SamplingParams(max_model_length=4)
    )
    with pytest.raises(ValueError, match="leave room for completion"):
        scheduler.add_sequence(sequence)

    assert not scheduler.waiting
    assert not scheduler.running
    assert_block_pool_consistent(scheduler)


def test_cancel_releases_waiting_and_running_sequences_by_id():
    scheduler = make_scheduler(
        max_num_batched_tokens=8, max_num_sequences=1, max_cached_blocks=3
    )
    running = make_sequence([1, 2, 3, 4])
    waiting = make_sequence([5, 6, 7, 8])
    scheduler.add_sequence(running)
    scheduler.add_sequence(waiting)
    scheduled, _ = scheduler.schedule()

    assert scheduler.cancel(waiting.seq_id)
    assert waiting.status == SequenceStatus.CANCELLED
    assert waiting not in scheduler.waiting
    assert scheduler.cancel(running.seq_id)
    assert running.status == SequenceStatus.CANCELLED
    assert scheduler.is_finished()
    assert scheduler.block_manager.used_block_ids == set()
    assert not scheduler.cancel(running.seq_id)
    assert_block_pool_consistent(scheduler)


def test_preemption_releases_the_newest_request_and_retries_the_oldest():
    scheduler = make_scheduler(
        max_num_batched_tokens=8, max_num_sequences=2, max_cached_blocks=2
    )
    oldest = make_sequence([1, 2, 3, 4])
    newest = make_sequence([5, 6, 7, 8])
    scheduler.add_sequence(oldest)
    scheduler.add_sequence(newest)
    prefill, is_prefill = scheduler.schedule()
    assert is_prefill
    scheduler.postprocess(prefill, [99, 99])

    scheduled, is_prefill = scheduler.schedule()

    assert not is_prefill
    assert scheduled == [oldest]
    assert oldest.status == SequenceStatus.RUNNING
    assert newest.status == SequenceStatus.WAITING
    assert list(scheduler.waiting) == [newest]
    assert len(oldest.block_table) == 2
    assert newest.block_table == []
    assert_block_pool_consistent(scheduler)

    scheduler.postprocess([oldest], [scheduler.eos])
    resumed, is_prefill = scheduler.schedule()
    assert is_prefill
    assert resumed == [newest]
    assert newest.status == SequenceStatus.RUNNING
    assert_block_pool_consistent(scheduler)


def test_prefill_budget_and_sequence_limit_preserve_waiters():
    scheduler = make_scheduler(
        max_num_batched_tokens=4,
        max_num_sequences=2,
        max_cached_blocks=4,
        max_model_length=4,
    )
    sequences = [make_sequence([index, index + 1]) for index in (1, 3, 5)]
    for sequence in sequences:
        scheduler.add_sequence(sequence)

    first_batch, _ = scheduler.schedule()
    assert first_batch == sequences[:2]
    assert list(scheduler.waiting) == [sequences[2]]
    scheduler.postprocess(first_batch, [scheduler.eos, scheduler.eos])

    second_batch, is_prefill = scheduler.schedule()
    assert is_prefill
    assert second_batch == [sequences[2]]
    assert sequences[2].status == SequenceStatus.RUNNING
    assert_block_pool_consistent(scheduler)
    scheduler.postprocess(second_batch, [scheduler.eos])
    assert scheduler.is_finished()
    assert_block_pool_consistent(scheduler)


def test_randomized_lifecycle_preserves_block_pool_invariants():
    total_preemptions = 0
    for seed in range(200):
        rng = random.Random(seed)
        scheduler = make_scheduler(
            max_num_batched_tokens=16,
            max_num_sequences=4,
            max_cached_blocks=3,
            max_model_length=8,
        )
        sequences = [
            make_sequence(
                [index] * (1 + index % 3),
                sampling_params=SamplingParams(max_tokens=7, max_model_length=8),
            )
            for index in range(8)
        ]
        for sequence in sequences:
            scheduler.add_sequence(sequence)

        preemptions = 0
        for _ in range(200):
            if scheduler.is_finished():
                break
            was_running = set(scheduler.running)
            scheduled, _ = scheduler.schedule()
            preemptions += sum(
                sequence in was_running and sequence.status == SequenceStatus.WAITING
                for sequence in sequences
            )
            if scheduled:
                tokens = [
                    0 if rng.random() < 0.05 else rng.randrange(1, 100)
                    for _ in scheduled
                ]
                scheduler.postprocess(scheduled, tokens)
            active = list(scheduler.waiting) + list(scheduler.running)
            if active and rng.random() < 0.05:
                scheduler.cancel(rng.choice(active).seq_id)
            assert_block_pool_consistent(scheduler)
            for sequence in sequences:
                memberships = int(sequence in scheduler.waiting) + int(
                    sequence in scheduler.running
                )
                assert memberships == (0 if sequence.is_finished else 1)
        else:
            raise AssertionError(f"randomized lifecycle did not finish for seed {seed}")

        assert scheduler.is_finished()
        assert scheduler.block_manager.used_block_ids == set()
        assert all(sequence.is_finished for sequence in sequences)
        total_preemptions += preemptions

    assert total_preemptions > 0
    assert preemptions > 0
