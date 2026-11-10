from collections import deque

import numpy as np
import xxhash

from myvllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0

    def update_hash(self, block_hash: int) -> None:
        self.hash = block_hash

    def reset(self) -> None:
        self.hash = -1
        self.ref_count = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        # Prefix reuse is intentionally disabled until prefill consumes cached KV
        # and applies the corresponding position offset. This index is retained so
        # completed-block metadata can still be audited and released correctly.
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, "little"))
        h.update(np.asarray(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _index_block(self, block: Block) -> None:
        if block.hash != -1:
            self.hash_to_block_id[block.hash] = block.block_id

    def _remove_hash_mapping(self, block: Block) -> None:
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    def _allocate_block(self) -> Block:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.reset()
        block.ref_count = 1
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        self._remove_hash_mapping(block)
        block.reset()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        if not self.can_allocate(seq):
            raise RuntimeError(
                f"Sequence {seq.seq_id} needs {seq.num_blocks} KV blocks, "
                f"but only {len(self.free_block_ids)} are free"
            )

        prefix_hash = -1
        for index in range(seq.num_blocks):
            token_ids = seq.block(index)
            block = self._allocate_block()
            if len(token_ids) == self.block_size:
                block_hash = self.compute_hash(token_ids, prefix_hash)
                block.update_hash(block_hash)
                self._index_block(block)
                prefix_hash = block_hash
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.block_table.clear()
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        # A new logical block is needed only after the sequence crosses a block
        # boundary. The normal engine invariant is table_len == num_blocks or
        # table_len == num_blocks - 1 here.
        return seq.num_blocks <= len(seq.block_table) or bool(self.free_block_ids)

    def append(self, seq: Sequence) -> None:
        if seq.num_blocks > len(seq.block_table):
            block_table = seq.block_table
            previous_block = self.blocks[block_table[-1]]
            if previous_block.hash == -1:
                prefix_hash = (
                    -1 if len(block_table) == 1 else self.blocks[block_table[-2]].hash
                )
                previous_hash = self.compute_hash(
                    seq.block(len(block_table) - 1), prefix_hash
                )
                previous_block.update_hash(previous_hash)
                self._index_block(previous_block)
            block = self._allocate_block()
            seq.block_table.append(block.block_id)
            return

        if seq.num_tokens % self.block_size != 0:
            return

        block_table = seq.block_table
        prefix_hash = -1 if len(block_table) == 1 else self.blocks[block_table[-2]].hash
        block = self.blocks[block_table[-1]]
        block_hash = self.compute_hash(seq.block(seq.num_blocks - 1), prefix_hash)
        block.update_hash(block_hash)
        self._index_block(block)
