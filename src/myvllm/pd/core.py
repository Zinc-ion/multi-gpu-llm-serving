"""Single-owner P/D lifecycle over the existing engine and scheduler."""

from copy import deepcopy
from dataclasses import dataclass
import time
import uuid

import torch

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.pd.kv import export_kv, restore_kv
from myvllm.pd.protocol import finite_number, make_metadata, model_spec, payload_nbytes, validate_metadata
from myvllm.sampling_parameters import SamplingParams


class CapacityError(ValueError):
    """The request cannot be admitted within the configured bounds."""


class StateError(ValueError):
    """The operation is invalid for this connection-scoped handle."""


@dataclass(frozen=True)
class TokenEvent:
    handle: object
    token: int
    finished: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


class _Handle:
    def __init__(self, session_id, deadline):
        self.session_id = session_id
        self.deadline = deadline
        self.state = "open"
        self.sequence = None
        self.transaction_id = None
        self.request_id = None
        self._metadata = None
        self._payload = None
        self._reserved_bytes = 0
        self._reserved_blocks = 0


class _Core:
    _seed_mask = 0

    def __init__(self, engine, max_open_requests, max_transfer_bytes):
        if type(max_open_requests) is not int or max_open_requests <= 0:
            raise ValueError("max_open_requests must be a positive integer")
        if type(max_transfer_bytes) is not int or max_transfer_bytes <= 0:
            raise ValueError("max_transfer_bytes must be a positive integer")
        seed = engine.config.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a 32-bit unsigned integer")
        self.engine = engine
        self.scheduler = engine.scheduler
        self.manager = self.scheduler.block_manager
        if not self.scheduler.is_finished() or self.manager.used_block_ids:
            raise ValueError("PD core requires an empty, exclusively owned scheduler")
        self.spec = model_spec(engine)
        self.max_open_requests = max_open_requests
        self.max_transfer_bytes = max_transfer_bytes
        self._handles = {}
        self._sequences = {}
        self._reserved_bytes = 0
        self.sampling_seed = seed ^ self._seed_mask
        torch.manual_seed(self.sampling_seed)

    def _open(self, session_id, deadline):
        if len(self._handles) >= self.max_open_requests:
            raise CapacityError("PD connection capacity exhausted")
        if deadline is not None and not finite_number(deadline):
            raise ValueError("deadline must be a finite monotonic time")
        handle = _Handle(session_id, deadline)
        self._handles[session_id] = handle
        return handle

    def _check(self, handle):
        if not isinstance(handle, _Handle) or self._handles.get(handle.session_id) is not handle:
            raise StateError("PD handle is closed or belongs to another core")

    def _release_bytes(self, handle):
        self._reserved_bytes -= handle._reserved_bytes
        handle._reserved_bytes = 0
        handle._payload = None

    def expire(self, now=None):
        if now is None:
            now = time.monotonic()
        expired = []
        for handle in list(self._handles.values()):
            if handle.deadline is not None and now >= handle.deadline:
                if self._cancel(handle, "expired"):
                    expired.append(handle)
        return expired

    def cancel(self, handle):
        self._check(handle)
        return self._cancel(handle, "cancelled")

    def close(self, handle):
        if isinstance(handle, _Handle) and handle.state == "closed":
            return False
        self._check(handle)
        self._cancel(handle, "cancelled")
        del self._handles[handle.session_id]
        handle.sequence = None
        handle._metadata = None
        handle.state = "closed"
        return True

    def shutdown(self):
        for handle in list(self._handles.values()):
            self.close(handle)

    def stats(self):
        return {
            "sampling_seed": self.sampling_seed,
            "open_handles": len(self._handles),
            "reserved_transfer_bytes": self._reserved_bytes,
            "free_blocks": len(self.manager.free_block_ids),
            "used_blocks": len(self.manager.used_block_ids),
        }

    def _event(self, handle, seq, token):
        reason = None
        if seq.is_finished:
            reason = "stop" if not seq.ignore_eos and token in self.scheduler.eos else "length"
        return TokenEvent(handle, token, seq.is_finished, reason,
                          seq.num_prompt_tokens, seq.num_completion_tokens)


class PrefillCore(_Core):
    """Own P requests from admission through a synchronous transfer ACK."""

    def open(self, session_id, request_id, prompt_token_ids, params, deadline=None):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("prompt_token_ids must be a nonempty list")
        vocab_size = self.spec["architecture"]["vocab_size"]
        if any(type(token) is not int or not 0 <= token < vocab_size for token in prompt_token_ids):
            raise ValueError("Invalid prompt token")
        if not isinstance(params, SamplingParams):
            raise ValueError("params must be SamplingParams")
        if not finite_number(params.temperature) or params.temperature < 0:
            raise ValueError("Invalid temperature")
        if type(params.max_tokens) is not int or params.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(params.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be boolean")
        limit = params.max_model_length
        if limit is None:
            limit = self.scheduler.max_model_length
        if type(limit) is not int or not 0 < limit <= self.scheduler.max_model_length:
            raise ValueError("Invalid max_model_length")
        if len(prompt_token_ids) >= limit:
            raise ValueError("Prompt must leave room for completion")

        existing = self._handles.get(session_id)
        if existing is not None:
            seq = existing.sequence
            if (existing.request_id == request_id and seq.prompt_token_ids == prompt_token_ids
                    and (seq.temperature, seq.max_tokens, seq.ignore_eos, seq.max_model_length)
                    == (params.temperature, params.max_tokens, params.ignore_eos, limit)
                    and existing.deadline == deadline):
                return existing
            raise StateError("Session is already bound to another request")

        nbytes = 0
        if params.max_tokens > 1 and len(prompt_token_ids) + 1 < limit:
            nbytes = payload_nbytes(self.spec, len(prompt_token_ids))
        if self._reserved_bytes + nbytes > self.max_transfer_bytes:
            raise CapacityError("P transfer capacity exhausted")
        handle = self._open(session_id, deadline)
        seq = Sequence(prompt_token_ids, self.manager.block_size, params)
        try:
            self.scheduler.add_sequence(seq)
        except ValueError:
            del self._handles[session_id]
            raise
        handle.state = "queued"
        handle.sequence = seq
        handle.transaction_id = uuid.uuid4().hex
        handle.request_id = request_id
        handle._reserved_bytes = nbytes
        self._reserved_bytes += nbytes
        self._sequences[seq.seq_id] = handle
        self.engine.request_ids[seq.seq_id] = request_id
        return handle

    def step(self):
        if self.scheduler.is_finished():
            return []
        if (not self.scheduler.running and self.scheduler.waiting
                and not self.manager.can_allocate(self.scheduler.waiting[0])
                and any(handle.state in ("held", "exported") for handle in self._handles.values())):
            return []
        events = []

        def publish(seq, token):
            handle = self._sequences.pop(seq.seq_id)
            self.engine.request_ids.pop(seq.seq_id, None)
            events.append(self._event(handle, seq, token))
            if seq.is_finished:
                handle.state = "finished"
                self._release_bytes(handle)
            else:
                detached = self.scheduler.detach(seq.seq_id)
                assert detached is seq
                handle.state = "held"

        self.engine.step(on_tokens=publish)
        for seq in self.scheduler.running:
            self._sequences[seq.seq_id].state = "prefill"
        return events

    def describe(self, handle):
        self._check(handle)
        if handle.state not in ("held", "exported", "handed_off"):
            raise StateError("P request has no transferable KV")
        if handle._metadata is None:
            handle._metadata = make_metadata(handle, self.spec, self.sampling_seed)
        return deepcopy(handle._metadata)

    def export(self, handle):
        self._check(handle)
        if handle.state == "exported":
            return handle._payload
        if handle.state != "held":
            raise StateError("P request is not waiting for export")
        if handle._metadata is None:
            handle._metadata = make_metadata(handle, self.spec, self.sampling_seed)
        try:
            payload = export_kv(self.engine.model_runner.kv_cache,
                                handle.sequence.block_table,
                                handle.sequence.num_cached_tokens, self.spec)
        except Exception:
            self._cancel(handle, "failed")
            raise
        handle._payload = payload
        handle.state = "exported"
        return payload

    def acknowledge(self, handle):
        self._check(handle)
        if handle.state == "handed_off":
            return False
        if handle.state != "exported":
            raise StateError("P request has no exported transfer to acknowledge")
        self.manager.deallocate(handle.sequence)
        self._release_bytes(handle)
        handle.state = "handed_off"
        return True

    def _cancel(self, handle, state):
        if handle.state in ("finished", "handed_off", "cancelled", "expired", "failed"):
            return False
        seq = handle.sequence
        if handle.state in ("held", "exported"):
            self.manager.deallocate(seq)
            seq.status = SequenceStatus.CANCELLED
        elif seq is not None:
            self.scheduler.cancel(seq.seq_id)
            self._sequences.pop(seq.seq_id, None)
            self.engine.request_ids.pop(seq.seq_id, None)
        self._release_bytes(handle)
        handle.state = state
        return True

    def stats(self):
        result = super().stats()
        result["held_requests"] = sum(
            handle.state in ("held", "exported") for handle in self._handles.values()
        )
        result["buffered_bytes"] = sum(
            len(handle._payload) for handle in self._handles.values() if handle._payload is not None
        )
        return result


class DecodeCore(_Core):
    """Reserve complete D budgets and activate only after the KV copy completes."""

    _seed_mask = 0x80000000

    def __init__(self, engine, max_open_requests, max_transfer_bytes):
        super().__init__(engine, max_open_requests, max_transfer_bytes)
        self._reserved_blocks = 0
        self._resident = 0
        self._resident_limit = min(self.scheduler.max_num_sequences,
                                   self.scheduler.max_num_batched_tokens)

    def open(self, deadline=None):
        return self._open(uuid.uuid4().hex, deadline)

    def reserve(self, handle, metadata):
        self._check(handle)
        if handle.state in ("reserved", "restored", "active", "finished"):
            validate_metadata(metadata, handle.session_id, self.spec,
                              self.scheduler.max_model_length)
            if metadata == handle._metadata:
                return False
            raise StateError("Transaction metadata changed")
        if handle.state != "open":
            raise StateError("D handle cannot reserve a transaction")
        try:
            validate_metadata(metadata, handle.session_id, self.spec,
                              self.scheduler.max_model_length)
            if metadata["sampling_seed"] == self.sampling_seed:
                raise ValueError("P and D sampling_seed must differ")
            nbytes = metadata["kv"]["nbytes"]
            tokens = min(metadata["prompt_length"] + metadata["sampling"]["max_tokens"],
                         metadata["sampling"]["max_model_length"])
            blocks = (tokens + self.manager.block_size - 1) // self.manager.block_size
            if self._resident >= self._resident_limit:
                raise CapacityError("D resident sequence capacity exhausted")
            if self._reserved_blocks + blocks > len(self.manager.blocks):
                raise CapacityError("D KV block capacity exhausted")
            if self._reserved_bytes + nbytes > self.max_transfer_bytes:
                raise CapacityError("D transfer capacity exhausted")
        except ValueError:
            handle.state = "failed"
            raise
        handle._metadata = deepcopy(metadata)
        handle.transaction_id = metadata["transaction_id"]
        handle.request_id = metadata["request_id"]
        handle._reserved_bytes = nbytes
        handle._reserved_blocks = blocks
        self._reserved_bytes += nbytes
        self._reserved_blocks += blocks
        self._resident += 1
        handle.state = "reserved"
        return True

    def restore(self, handle, payload):
        self._check(handle)
        if handle.state in ("restored", "active", "finished"):
            if type(payload) is bytes and len(payload) == handle._metadata["kv"]["nbytes"]:
                return False
            raise ValueError("Invalid KV payload byte count")
        if handle.state != "reserved":
            raise StateError("D handle has no pending reservation")
        metadata = handle._metadata
        if type(payload) is not bytes or len(payload) != metadata["kv"]["nbytes"]:
            self._cancel(handle, "failed")
            raise ValueError("Invalid KV payload byte count")
        params = SamplingParams(**metadata["sampling"])
        prompt_length = metadata["prompt_length"]
        seq = Sequence(metadata["token_ids"][:prompt_length], self.manager.block_size, params)
        seq.append_token(metadata["token_ids"][-1])
        try:
            self.manager.allocate(seq)
            restore_kv(self.engine.model_runner.kv_cache, seq.block_table, payload,
                       self.spec, metadata["cached_tokens"])
        except Exception:
            self.manager.deallocate(seq)
            self._cancel(handle, "failed")
            raise
        seq.num_cached_tokens = metadata["cached_tokens"]
        seq.num_scheduled_tokens = 1
        handle.sequence = seq
        self._release_bytes(handle)
        handle.state = "restored"
        return True

    def activate(self, handle):
        self._check(handle)
        if handle.state in ("active", "finished"):
            return False
        if handle.state != "restored":
            raise StateError("D handle is not restored")
        seq = handle.sequence
        self.scheduler.activate_restored(seq)
        self._sequences[seq.seq_id] = handle
        self.engine.request_ids[seq.seq_id] = handle.request_id
        handle.state = "active"
        return True

    def step(self):
        if self.scheduler.is_finished():
            return []
        events = []

        def publish(seq, token):
            handle = self._sequences[seq.seq_id]
            events.append(self._event(handle, seq, token))
            if seq.is_finished:
                self._sequences.pop(seq.seq_id)
                self.engine.request_ids.pop(seq.seq_id, None)
                self._release(handle)
                handle.state = "finished"

        self.engine.step(on_tokens=publish)
        return events

    def _release(self, handle):
        self._release_bytes(handle)
        if handle._reserved_blocks:
            self._reserved_blocks -= handle._reserved_blocks
            handle._reserved_blocks = 0
            self._resident -= 1

    def _cancel(self, handle, state):
        if handle.state in ("finished", "cancelled", "expired", "failed"):
            return False
        seq = handle.sequence
        if handle.state == "active":
            self.scheduler.cancel(seq.seq_id)
            self._sequences.pop(seq.seq_id, None)
            self.engine.request_ids.pop(seq.seq_id, None)
        elif handle.state == "restored":
            self.manager.deallocate(seq)
            seq.status = SequenceStatus.CANCELLED
        self._release(handle)
        handle.state = state
        return True

    def stats(self):
        result = super().stats()
        result.update(reserved_blocks=self._reserved_blocks, resident_requests=self._resident,
                      resident_limit=self._resident_limit)
        return result
