from collections import deque

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(
        self,
        max_num_sequences: int,
        max_num_batched_tokens: int,
        max_cached_blocks: int,
        block_size: int,
        eos: int,
        max_model_length: int,
    ):
        if max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be positive")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if max_model_length <= 0:
            raise ValueError("max_model_length must be positive")
        if max_model_length > max_num_batched_tokens:
            raise ValueError("max_model_length must not exceed max_num_batched_tokens")
        if max_model_length > max_cached_blocks * block_size:
            raise ValueError("max_model_length must fit within the configured KV cache")

        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.max_model_length = max_model_length
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add_sequence(self, sequence: Sequence) -> None:
        if not sequence.token_ids:
            raise ValueError(
                f"Sequence {sequence.seq_id} requires at least one prompt token"
            )
        if sequence.max_model_length is None:
            sequence.max_model_length = self.max_model_length
        elif sequence.max_model_length > self.max_model_length:
            raise ValueError(
                f"Sequence {sequence.seq_id} max_model_length={sequence.max_model_length} "
                f"exceeds scheduler max_model_length={self.max_model_length}."
            )
        if len(sequence) >= sequence.max_model_length:
            raise ValueError(
                f"Sequence {sequence.seq_id} prompt must leave room for completion."
            )
        self.waiting.append(sequence)

    def _release(self, sequence: Sequence) -> None:
        self.block_manager.deallocate(sequence)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences: list[Sequence] = []
        scheduled_tokens = 0

        # Prefill is prioritized; a prompt outside this batch budget stays queued.
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            sequence = self.waiting[0]
            if len(sequence) + scheduled_tokens > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(sequence):
                break

            self.waiting.popleft()
            self.block_manager.allocate(sequence)
            sequence.status = SequenceStatus.RUNNING
            self.running.append(sequence)
            scheduled_sequences.append(sequence)
            scheduled_tokens += len(sequence)

        if scheduled_sequences:
            return scheduled_sequences, True

        # Process each sequence that was running at the start at most once. A
        # scheduled sequence is put back immediately, keeping queue order stable.
        initial_running_count = len(self.running)
        preempted = False
        for _ in range(initial_running_count):
            if (
                scheduled_tokens >= self.max_num_batched_tokens
                or len(scheduled_sequences) >= self.max_num_sequences
            ):
                break

            sequence = self.running.popleft()
            if not self.block_manager.can_append(sequence):
                preempted = True
                if self.running:
                    # Keep the oldest request to retry after giving up the newest
                    # request's blocks. This preserves FIFO progress under OOM.
                    self.running.appendleft(sequence)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(sequence)
                    break
                continue

            self.block_manager.append(sequence)
            scheduled_sequences.append(sequence)
            scheduled_tokens += 1

        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        if scheduled_sequences:
            return scheduled_sequences, False

        if not preempted and (self.waiting or self.running):
            raise RuntimeError(
                "Scheduler made no progress: "
                f"waiting={len(self.waiting)}, running={len(self.running)}, "
                f"free_blocks={len(self.block_manager.free_block_ids)}/"
                f"{len(self.block_manager.blocks)}."
            )
        return [], False

    def preempt(self, sequence: Sequence) -> None:
        self._release(sequence)
        sequence.status = SequenceStatus.WAITING
        self.waiting.appendleft(sequence)

    def cancel(self, sequence_id: int) -> bool:
        """Cancel a waiting or running sequence by request ID."""

        for queue in (self.waiting, self.running):
            for index, sequence in enumerate(queue):
                if sequence.seq_id == sequence_id:
                    del queue[index]
                    self._release(sequence)
                    sequence.status = SequenceStatus.CANCELLED
                    return True
        return False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for sequence, token_id in zip(seqs, token_ids):
            sequence.append_token(token_id)

            stop_due_to_eos = not sequence.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = (
                sequence.num_completion_tokens >= sequence.max_tokens
            )
            stop_due_to_max_length = (
                sequence.max_model_length is not None
                and sequence.num_tokens >= sequence.max_model_length
            )

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                self.running.remove(sequence)
                self._release(sequence)
                sequence.status = SequenceStatus.FINISHED
                continue

            # Keep a live request in its current queue position. Removing and
            # appending it here would silently rotate the decode order each step.
