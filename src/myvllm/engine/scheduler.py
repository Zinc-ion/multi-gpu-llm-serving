from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int, max_model_length: int):
        if max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be positive")
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if max_model_length <= 0 or max_model_length > max_num_batched_tokens:
            raise ValueError("max_model_length must be positive and must not exceed max_num_batched_tokens")
        if max_model_length > max_cached_blocks * block_size:
            raise ValueError("max_model_length must fit within the configured KV cache")
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.max_model_length = max_model_length
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        # Reject up front what the block manager could never satisfy, otherwise the
        # sequence sits in `waiting` forever and only surfaces as a stalled engine.
        if not sequence.token_ids:
            raise ValueError(f"Sequence {sequence.seq_id} requires at least one prompt token")
        if sequence.max_model_length is None:
            sequence.max_model_length = self.max_model_length
        elif sequence.max_model_length > self.max_model_length:
            raise ValueError(f"Sequence {sequence.seq_id} max_model_length exceeds scheduler max_model_length")
        if len(sequence) >= sequence.max_model_length:
            raise ValueError(f"Sequence {sequence.seq_id} prompt must leave room for completion.")
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # An empty schedule is only legitimate when this call freed blocks by
        # preempting, so the next call can make progress. See the guard below.
        preempted = False
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                preempted = True
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
        elif not preempted and (self.waiting or self.running):
            # Nothing was scheduled and nothing was preempted, so no engine state
            # changed: every later schedule() would take the same decisions and
            # LLMEngine.generate() would spin forever. Fail loudly instead.
            raise RuntimeError(
                "Scheduler made no progress: "
                f"{len(self.waiting)} waiting and {len(self.running)} running sequences, "
                f"{len(self.block_manager.free_block_ids)} of "
                f"{len(self.block_manager.blocks)} blocks free. "
                "This means either a sequence that cannot fit in the KV cache, or "
                "blocks leaked because their ref_count never returned to 0."
            )

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    def cancel(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for i, seq in enumerate(queue):
                if seq.seq_id == seq_id:
                    del queue[i]
                    self.block_manager.deallocate(seq)
                    seq.status = SequenceStatus.CANCELLED
                    return True
        return False


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)