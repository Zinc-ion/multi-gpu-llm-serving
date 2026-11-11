import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def assert_pool_consistent(manager: BlockManager) -> None:
    all_ids = set(range(len(manager.blocks)))
    free_ids = set(manager.free_block_ids)
    assert len(free_ids) == len(manager.free_block_ids)
    assert free_ids.isdisjoint(manager.used_block_ids)
    assert free_ids | manager.used_block_ids == all_ids

    for block_id, block in enumerate(manager.blocks):
        if block_id in free_ids:
            assert block.ref_count == 0
            assert block.hash == -1
        else:
            assert block.ref_count > 0

    for block_hash, block_id in manager.hash_to_block_id.items():
        block = manager.blocks[block_id]
        assert block_id in manager.used_block_ids
        assert block.hash == block_hash


def test_allocate_and_deallocate_returns_every_block_to_pool():
    manager = BlockManager(num_blocks=3, block_size=4)
    sequence = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9], block_size=4)

    manager.allocate(sequence)

    assert len(sequence.block_table) == 3
    assert manager.used_block_ids == set(sequence.block_table)
    assert [manager.blocks[i].ref_count for i in sequence.block_table] == [1, 1, 1]
    assert_pool_consistent(manager)

    manager.deallocate(sequence)

    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert manager.used_block_ids == set()
    assert len(manager.free_block_ids) == 3
    assert manager.hash_to_block_id == {}
    assert_pool_consistent(manager)


def test_insufficient_capacity_does_not_change_the_block_pool():
    manager = BlockManager(num_blocks=1, block_size=4)
    sequence = Sequence([1, 2, 3, 4, 5], block_size=4)

    assert not manager.can_allocate(sequence)
    assert sequence.block_table == []
    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [0]
    assert_pool_consistent(manager)


def test_append_finalizes_full_block_and_allocates_next_block():
    manager = BlockManager(num_blocks=2, block_size=4)
    sequence = Sequence([1, 2, 3], block_size=4)
    manager.allocate(sequence)
    partial_block_id = sequence.block_table[-1]

    sequence.append_token(4)
    manager.append(sequence)
    full_block_hash = manager.blocks[partial_block_id].hash
    assert full_block_hash != -1
    assert manager.hash_to_block_id[full_block_hash] == partial_block_id

    sequence.append_token(5)
    assert manager.can_append(sequence)
    manager.append(sequence)
    assert len(sequence.block_table) == 2
    assert manager.blocks[sequence.block_table[-1]].hash == -1

    sequence.append_token(6)
    manager.append(sequence)
    assert manager.blocks[sequence.block_table[-1]].hash == -1
    assert_pool_consistent(manager)

    manager.deallocate(sequence)
    assert manager.hash_to_block_id == {}
    assert_pool_consistent(manager)


def test_reusing_a_released_block_drops_old_hash_metadata():
    manager = BlockManager(num_blocks=1, block_size=2)
    original = Sequence([1, 2], block_size=2)
    replacement = Sequence([9, 10], block_size=2)
    original_hash = manager.compute_hash([1, 2], -1)
    replacement_hash = manager.compute_hash([9, 10], -1)

    manager.allocate(original)
    manager.deallocate(original)
    assert original_hash not in manager.hash_to_block_id
    assert manager.blocks[0].hash == -1

    manager.allocate(replacement)
    assert manager.hash_to_block_id == {replacement_hash: 0}
    assert original_hash not in manager.hash_to_block_id
    assert_pool_consistent(manager)


def test_identical_active_prefixes_get_distinct_blocks_and_release_cleanly():
    manager = BlockManager(num_blocks=2, block_size=4)
    first = Sequence([1, 2, 3, 4], block_size=4)
    second = Sequence([1, 2, 3, 4], block_size=4)

    manager.allocate(first)
    manager.allocate(second)

    assert first.block_table != second.block_table
    assert first.num_cached_tokens == 0
    assert second.num_cached_tokens == 0
    assert [block.ref_count for block in manager.blocks] == [1, 1]
    assert_pool_consistent(manager)

    first_hash = manager.blocks[first.block_table[0]].hash
    second_block = second.block_table[0]
    manager.deallocate(first)
    assert manager.blocks[second_block].ref_count == 1
    assert manager.hash_to_block_id.get(first_hash) == second_block
    assert_pool_consistent(manager)

    manager.deallocate(second)
    assert manager.used_block_ids == set()
    assert manager.hash_to_block_id == {}
    assert_pool_consistent(manager)


def test_deallocate_is_idempotent_after_sequence_released():
    manager = BlockManager(num_blocks=1, block_size=4)
    sequence = Sequence([1, 2, 3], block_size=4)
    manager.allocate(sequence)

    manager.deallocate(sequence)
    manager.deallocate(sequence)

    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [0]
    assert_pool_consistent(manager)


def test_block_size_one_allocates_a_new_block_on_each_token():
    manager = BlockManager(num_blocks=3, block_size=1)
    sequence = Sequence([1], block_size=1)
    manager.allocate(sequence)

    for token_id in (2, 3):
        sequence.append_token(token_id)
        assert manager.can_append(sequence)
        manager.append(sequence)

    assert sequence.block_table == [0, 1, 2]
    assert [manager.blocks[block_id].hash for block_id in sequence.block_table] == [
        manager.compute_hash([1], -1),
        manager.compute_hash([2], manager.compute_hash([1], -1)),
        -1,
    ]
    manager.deallocate(sequence)
    assert_pool_consistent(manager)
