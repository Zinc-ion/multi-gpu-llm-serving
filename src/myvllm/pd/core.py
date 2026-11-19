"""Single-owner P/D lifecycle over the existing engine and scheduler."""

from copy import deepcopy
from dataclasses import dataclass
import time
import uuid

import torch

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.pd.kv import export_kv, restore_kv
from myvllm.pd.pipeline import LayerReceiver, PipelineTimingError
from myvllm.pd.protocol import bulk_version, finite_number, make_early_metadata, make_metadata, model_spec, payload_nbytes, tensor_dtype, token_byte_counts, validate_early_metadata, validate_metadata
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
        self._early_metadata = None
        self._commit_metadata = None
        self._next_layer = 0
        self._next_offset = 0
        self._payload = None
        self._receiver = None
        self._reserved_bytes = 0
        self._reserved_blocks = 0
        self._tp_export_metrics = None
        self.local = False


class _Core:
    _seed_mask = 0
    _allow_tp_prefill = False

    def __init__(self, engine, max_open_requests, max_transfer_bytes, max_payload_bytes=None):
        if type(max_open_requests) is not int or max_open_requests <= 0:
            raise ValueError("max_open_requests must be a positive integer")
        if type(max_transfer_bytes) is not int or max_transfer_bytes <= 0:
            raise ValueError("max_transfer_bytes must be a positive integer")
        if max_payload_bytes is None:
            max_payload_bytes = max_transfer_bytes
        if type(max_payload_bytes) is not int or not 0 < max_payload_bytes <= max_transfer_bytes:
            raise ValueError("max_payload_bytes must be positive and no greater than max_transfer_bytes")
        seed = engine.config.get("seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a 32-bit unsigned integer")
        self.engine = engine
        self.scheduler = engine.scheduler
        self.manager = self.scheduler.block_manager
        if not self.scheduler.is_finished() or self.manager.used_block_ids:
            raise ValueError("PD core requires an empty, exclusively owned scheduler")
        self.spec = model_spec(engine, allow_tp_prefill=self._allow_tp_prefill)
        self.max_open_requests = max_open_requests
        self.max_transfer_bytes = max_transfer_bytes
        self.max_payload_bytes = max_payload_bytes
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

    def _validate_request(self, request_id, prompt_token_ids, params):
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
        return limit

    def expire(self, now=None):
        if now is None:
            now = time.monotonic()
        expired = []
        error = None
        for handle in list(self._handles.values()):
            if handle.deadline is not None and now >= handle.deadline:
                try:
                    if self._cancel(handle, "expired"):
                        expired.append(handle)
                except PipelineTimingError as exc:
                    expired.append(handle)
                    if error is None:
                        error = exc
        if error is not None:
            for handle in expired:
                self.close(handle)
            raise error
        return expired

    def cancel(self, handle):
        self._check(handle)
        return self._cancel(handle, "cancelled")

    def close(self, handle):
        if isinstance(handle, _Handle) and handle.state == "closed":
            return False
        self._check(handle)
        error = None
        try:
            self._cancel(handle, "cancelled")
        except PipelineTimingError as exc:
            error = exc
        if handle._receiver is not None:
            try:
                handle._receiver.close()
            except PipelineTimingError as exc:
                error = exc
            handle._receiver = None
        del self._handles[handle.session_id]
        handle.sequence = None
        handle._metadata = None
        handle._early_metadata = None
        handle._commit_metadata = None
        handle.state = "closed"
        if error is not None:
            raise error
        return True

    def shutdown(self):
        error = None
        for handle in list(self._handles.values()):
            try:
                self.close(handle)
            except PipelineTimingError as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def stats(self):
        return {
            "sampling_seed": self.sampling_seed,
            "open_handles": len(self._handles),
            "reserved_transfer_bytes": self._reserved_bytes,
            "free_blocks": len(self.manager.free_block_ids),
            "used_blocks": len(self.manager.used_block_ids),
            "total_blocks": len(self.manager.blocks),
        }

    def _event(self, handle, seq, token):
        reason = None
        if seq.is_finished:
            reason = "stop" if not seq.ignore_eos and token in self.scheduler.eos else "length"
        return TokenEvent(handle, token, seq.is_finished, reason,
                          seq.num_prompt_tokens, seq.num_completion_tokens)


class PrefillCore(_Core):
    """Own P requests from admission through a synchronous transfer ACK."""

    _allow_tp_prefill = True

    def open(self, session_id, request_id, prompt_token_ids, params, deadline=None):
        return self._open_request(session_id, request_id, prompt_token_ids, params, deadline, early=False)

    def prepare(self, session_id, request_id, prompt_token_ids, params, deadline=None):
        if self.engine.config["world_size"] != 1:
            raise ValueError("Early transfer requires TP1 prefill")
        if self.scheduler.enable_chunked_prefill or not self.engine.config.get("enforce_eager", False):
            raise ValueError("Early transfer requires whole eager prefill")
        return self._open_request(session_id, request_id, prompt_token_ids, params, deadline, early=True)

    def _open_request(self, session_id, request_id, prompt_token_ids, params, deadline, early):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        limit = self._validate_request(request_id, prompt_token_ids, params)

        existing = self._handles.get(session_id)
        if existing is not None:
            seq = existing.sequence
            if ((existing._early_metadata is not None) == early
                    and existing.request_id == request_id and seq.prompt_token_ids == prompt_token_ids
                    and (seq.temperature, seq.max_tokens, seq.ignore_eos, seq.max_model_length)
                    == (params.temperature, params.max_tokens, params.ignore_eos, limit)
                    and existing.deadline == deadline):
                return existing
            raise StateError("Session is already bound to another request")

        nbytes = 0
        if params.max_tokens > 1 and len(prompt_token_ids) + 1 < limit:
            nbytes = payload_nbytes(self.spec, len(prompt_token_ids))
        if early and not nbytes:
            raise ValueError("Early transfer requires a possible second token")
        pending = [h for h in self._handles.values()
                   if h.state in ("prepared", "ready", "queued", "prefill", "held", "exported")]
        if early or any(h._early_metadata is not None for h in pending):
            block_size = self.manager.block_size
            blocks = (len(prompt_token_ids) + block_size - 1) // block_size
            blocks += sum((h.sequence.num_prompt_tokens + block_size - 1) // block_size for h in pending)
            if blocks > len(self.manager.blocks):
                raise CapacityError("P prompt block capacity exhausted")
        if nbytes > self.max_payload_bytes or self._reserved_bytes + nbytes > self.max_transfer_bytes:
            raise CapacityError("P transfer capacity exhausted")
        handle = self._open(session_id, deadline)
        seq = Sequence(prompt_token_ids, self.manager.block_size, params)
        if early:
            seq.max_model_length = limit
        else:
            try:
                self.scheduler.add_sequence(seq)
            except ValueError:
                del self._handles[session_id]
                raise
        handle.state = "prepared" if early else "queued"
        handle.sequence = seq
        handle.transaction_id = uuid.uuid4().hex
        handle.request_id = request_id
        handle._reserved_bytes = nbytes
        self._reserved_bytes += nbytes
        if early:
            handle._early_metadata = make_early_metadata(handle, self.spec, self.sampling_seed)
        else:
            self._sequences[seq.seq_id] = handle
            self.engine.request_ids[seq.seq_id] = request_id
        return handle

    def describe_early(self, handle):
        self._check(handle)
        if handle._early_metadata is None or handle.state not in ("prepared", "ready", "queued", "prefill", "held"):
            raise StateError("P request has no early transfer")
        return deepcopy(handle._early_metadata)

    def ready(self, handle):
        self._check(handle)
        if handle._early_metadata is None:
            raise StateError("P request was not prepared for early transfer")
        if handle.state in ("ready", "queued", "prefill", "held"):
            return False
        if handle.state != "prepared":
            raise StateError("P request cannot become ready")
        handle.state = "ready"
        return True

    def start(self, handle):
        self._check(handle)
        if handle._early_metadata is None:
            raise StateError("P request was not prepared for early transfer")
        if handle.state in ("queued", "prefill", "held"):
            return False
        if handle.state != "ready":
            raise StateError("P request has no ready data channel")
        seq = handle.sequence
        self.scheduler.add_sequence(seq)
        self._sequences[seq.seq_id] = handle
        self.engine.request_ids[seq.seq_id] = handle.request_id
        handle.state = "queued"
        return True

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
            if self.engine.config["world_size"] == 2:
                payload, handle._tp_export_metrics = self.engine.model_runner.call(
                    "export_pd_tp_kv", handle.sequence.block_table,
                    handle.sequence.num_cached_tokens, self.spec)
            else:
                scales = self.engine.model_runner.kv_cache_scales if "kv_cache" in self.spec else None
                payload = export_kv(self.engine.model_runner.kv_cache,
                                    handle.sequence.block_table,
                                    handle.sequence.num_cached_tokens, self.spec, scales)
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

    def acknowledge_early(self, handle):
        self._check(handle)
        if handle._early_metadata is None:
            raise StateError("P request has no early transfer")
        if handle.state == "handed_off":
            return False
        if handle.state != "held":
            raise StateError("P request is not waiting for early transfer ACK")
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
        elif handle.state in ("prepared", "ready"):
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
        pending = [h for h in self._handles.values()
                   if h.state in ("prepared", "ready", "queued", "prefill", "held", "exported")]
        result["admitted_requests"] = len(pending)
        result["pending_prefill_tokens"] = sum(
            max(0, h.sequence.num_prompt_tokens - h.sequence.num_cached_tokens)
            for h in pending if h.state not in ("held", "exported")
        )
        result["pending_prompt_blocks"] = sum(
            (h.sequence.num_prompt_tokens + self.manager.block_size - 1) // self.manager.block_size
            for h in pending
        )
        result["reserved_prompt_blocks"] = (
            result["pending_prompt_blocks"] if any(h._early_metadata is not None for h in pending) else 0
        )
        result["buffered_bytes"] = sum(
            len(handle._payload) for handle in self._handles.values() if handle._payload is not None
        )
        return result


class DecodeCore(_Core):
    """Reserve complete D budgets and activate only after the KV copy completes."""

    _seed_mask = 0x80000000

    def __init__(self, engine, max_open_requests, max_transfer_bytes, max_payload_bytes=None):
        super().__init__(engine, max_open_requests, max_transfer_bytes, max_payload_bytes)
        self._reserved_blocks = 0
        self._resident = 0
        self._local_resident = 0
        self._resident_limit = min(self.scheduler.max_num_sequences,
                                   self.scheduler.max_num_batched_tokens)

    def open(self, deadline=None):
        return self._open(uuid.uuid4().hex, deadline)

    def open_local(self, request_id, prompt_token_ids, params, deadline=None):
        if not self.scheduler.enable_chunked_prefill:
            raise ValueError("Local prefill requires enable_chunked_prefill")
        limit = self._validate_request(request_id, prompt_token_ids, params)
        handle = self._open(uuid.uuid4().hex, deadline)
        try:
            self._reserve_capacity(handle, min(len(prompt_token_ids) + params.max_tokens, limit), 0)
            seq = Sequence(prompt_token_ids, self.manager.block_size, params)
            self.scheduler.add_sequence(seq)
        except ValueError:
            self._release(handle)
            del self._handles[handle.session_id]
            raise
        handle.local = True
        self._local_resident += 1
        handle.sequence = seq
        handle.request_id = request_id
        handle.state = "queued"
        self._sequences[seq.seq_id] = handle
        self.engine.request_ids[seq.seq_id] = request_id
        return handle

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
            self._reserve_budget(handle, metadata)
        except ValueError:
            handle.state = "failed"
            raise
        handle.state = "reserved"
        return True

    def _reserve_budget(self, handle, metadata):
        if metadata["sampling_seed"] == self.sampling_seed:
            raise ValueError("P and D sampling_seed must differ")
        nbytes = metadata["kv"]["nbytes"]
        tokens = min(metadata["prompt_length"] + metadata["sampling"]["max_tokens"],
                     metadata["sampling"]["max_model_length"])
        self._reserve_capacity(handle, tokens, nbytes)
        handle._metadata = deepcopy(metadata)
        handle.transaction_id = metadata["transaction_id"]
        handle.request_id = metadata["request_id"]

    def _reserve_capacity(self, handle, tokens, nbytes):
        blocks = (tokens + self.manager.block_size - 1) // self.manager.block_size
        if self._resident >= self._resident_limit:
            raise CapacityError("D resident sequence capacity exhausted")
        if self._reserved_blocks + blocks > len(self.manager.blocks):
            raise CapacityError("D KV block capacity exhausted")
        if nbytes > self.max_payload_bytes or self._reserved_bytes + nbytes > self.max_transfer_bytes:
            raise CapacityError("D transfer capacity exhausted")
        handle._reserved_bytes = nbytes
        handle._reserved_blocks = blocks
        self._reserved_bytes += nbytes
        self._reserved_blocks += blocks
        self._resident += 1

    def reserve_early(self, handle, metadata):
        self._check(handle)
        if handle._early_metadata is not None and handle.state in ("early_reserved", "restored", "active", "finished"):
            validate_early_metadata(metadata, handle.session_id, self.spec,
                                    self.scheduler.max_model_length)
            if metadata == handle._early_metadata:
                return False
            raise StateError("Early transaction metadata changed")
        if handle.state != "open":
            raise StateError("D handle cannot reserve an early transaction")
        try:
            validate_early_metadata(metadata, handle.session_id, self.spec,
                                    self.scheduler.max_model_length)
            self._reserve_budget(handle, metadata)
        except ValueError:
            handle.state = "failed"
            raise
        seq = Sequence(metadata["token_ids"], self.manager.block_size,
                       SamplingParams(**metadata["sampling"]))
        self.manager.allocate(seq)
        if "kv_cache" in self.spec and seq.num_prompt_tokens % self.manager.block_size:
            try:
                block = seq.block_table[-1]
                offset = seq.num_prompt_tokens % self.manager.block_size
                self.engine.model_runner.kv_cache[:, :, block, offset:].zero_()
                self.engine.model_runner.kv_cache_scales[:, :, block, offset:].fill_(1)
            except Exception:
                self.manager.deallocate(seq)
                self._release(handle)
                handle.state = "failed"
                raise
        handle.sequence = seq
        handle._early_metadata = deepcopy(metadata)
        handle.state = "early_reserved"
        return True

    def attach_receiver(self, handle, frame_bytes, timing_path=None, *, mooncake_binding=None):
        self._check(handle)
        if handle.state != "early_reserved" or handle._receiver is not None:
            raise StateError("D handle cannot attach another data channel")
        identity = {"session_id": handle.session_id, "transaction_id": handle.transaction_id,
                    "request_id": handle.request_id, "role": "decode"}
        scales = self.engine.model_runner.kv_cache_scales if "kv_cache" in self.spec else None
        handle._receiver = LayerReceiver(self.engine.model_runner.kv_cache, frame_bytes,
                                         identity, timing_path, scales, mooncake_binding=mooncake_binding)
        return handle._receiver

    def restore_region(self, handle, layer, start, end, payload):
        self._check(handle)
        if handle.state != "early_reserved":
            if handle._early_metadata is not None and handle.state in ("restored", "active"):
                self._cancel(handle, "failed")
            raise StateError("D handle has no early transfer in progress")
        prompt_length = handle._early_metadata["prompt_length"]
        arch = self.spec["architecture"]
        try:
            if (type(layer) is not int or layer != handle._next_layer
                    or not 0 <= layer < arch["num_layers"]
                    or type(start) is not int or type(end) is not int
                    or start != handle._next_offset or not start < end <= prompt_length):
                raise ValueError("Invalid or duplicate KV region")
            data_per_token, scale_per_token = token_byte_counts(self.spec)
            nbytes = (end - start) * (data_per_token + scale_per_token)
            host_payload = type(payload) is bytes or (handle._receiver is not None and isinstance(payload, memoryview))
            if not host_payload or len(payload) != nbytes:
                raise ValueError("Invalid KV region byte count")
            if handle._receiver is not None:
                if nbytes > handle._receiver.frame_bytes:
                    raise ValueError("KV region exceeds pipeline frame limit")
                handle._receiver.restore(handle.sequence.block_table, layer, start, end, payload)
                if layer + 1 == arch["num_layers"] and end == prompt_length:
                    handle._receiver.finish()
            else:
                raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                data_nbytes = (end - start) * data_per_token
                host = raw[:data_nbytes].view(tensor_dtype(self.spec)).reshape(
                    2, end - start, arch["num_kv_heads"], arch["head_dim"])
                host_scales = None
                if scale_per_token:
                    host_scales = torch.empty((2, end - start, arch["num_kv_heads"]),
                                              dtype=torch.float32, device="cpu")
                    host_scales.view(torch.uint8).reshape(-1).copy_(raw[data_nbytes:])
                cache = self.engine.model_runner.kv_cache
                block_size = self.manager.block_size
                pos = start
                while pos < end:
                    block = handle.sequence.block_table[pos // block_size]
                    offset = pos % block_size
                    count = min(end - pos, block_size - offset)
                    cache[:, layer, block, offset:offset + count].copy_(host[:, pos - start:pos - start + count])
                    if host_scales is not None:
                        self.engine.model_runner.kv_cache_scales[:, layer, block, offset:offset + count].copy_(
                            host_scales[:, pos - start:pos - start + count])
                    pos += count
        except Exception:
            self._cancel(handle, "failed")
            raise
        if end == prompt_length:
            handle._next_layer += 1
            handle._next_offset = 0
        else:
            handle._next_offset = end
        self._finish_early_restore(handle)
        return True

    def commit_early(self, handle, metadata):
        self._check(handle)
        if handle._commit_metadata is not None and handle.state in ("early_reserved", "restored", "active", "finished"):
            validate_metadata(metadata, handle.session_id, self.spec,
                              self.scheduler.max_model_length)
            if metadata == handle._commit_metadata:
                return False
            raise StateError("Early transaction commit changed")
        if handle.state != "early_reserved":
            raise StateError("D handle has no early transfer to commit")
        try:
            validate_metadata(metadata, handle.session_id, self.spec,
                              self.scheduler.max_model_length)
            expected = deepcopy(handle._early_metadata)
            expected["version"] = bulk_version(self.spec)
            expected["token_ids"].append(metadata["token_ids"][-1])
            expected["cached_tokens"] = expected["next_position"] = expected["prompt_length"]
            expected["completion_count"] = 1
            if metadata != expected:
                raise ValueError("Early transaction commit does not match its reservation")
        except ValueError:
            self._cancel(handle, "failed")
            raise
        handle._commit_metadata = deepcopy(metadata)
        handle.sequence.append_token(metadata["token_ids"][-1])
        self._finish_early_restore(handle)
        return True

    def _finish_early_restore(self, handle):
        prompt_length = handle._early_metadata["prompt_length"]
        if handle._commit_metadata is not None and handle._next_layer == self.spec["architecture"]["num_layers"]:
            handle.sequence.num_cached_tokens = prompt_length
            handle.sequence.num_scheduled_tokens = 1
            self._release_bytes(handle)
            handle.state = "restored"

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
            scales = self.engine.model_runner.kv_cache_scales if "kv_cache" in self.spec else None
            restore_kv(self.engine.model_runner.kv_cache, seq.block_table, payload,
                       self.spec, metadata["cached_tokens"], scales)
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
            elif handle.local:
                handle.state = "active"

        self.engine.step(on_tokens=publish)
        if self._local_resident:
            for seq in self.scheduler.running:
                handle = self._sequences[seq.seq_id]
                if handle.local and handle.state == "queued":
                    handle.state = "prefill"
        return events

    def _release(self, handle):
        self._release_bytes(handle)
        if handle._reserved_blocks:
            self._reserved_blocks -= handle._reserved_blocks
            handle._reserved_blocks = 0
            self._resident -= 1
            if handle.local:
                self._local_resident -= 1

    def _cancel(self, handle, state):
        if handle.state in ("finished", "cancelled", "expired", "failed"):
            return False
        error = None
        if handle._receiver is not None:
            try:
                handle._receiver.close()
            except PipelineTimingError as exc:
                error = exc
            handle._receiver = None
        seq = handle.sequence
        if handle.state in ("queued", "prefill", "active"):
            self.scheduler.cancel(seq.seq_id)
            self._sequences.pop(seq.seq_id, None)
            self.engine.request_ids.pop(seq.seq_id, None)
        elif handle.state in ("restored", "early_reserved"):
            self.manager.deallocate(seq)
            seq.status = SequenceStatus.CANCELLED
        self._release(handle)
        handle.state = state
        if error is not None:
            raise error
        return True

    def stats(self):
        result = super().stats()
        result.update(reserved_blocks=self._reserved_blocks, resident_requests=self._resident,
                      resident_limit=self._resident_limit)
        result["local_resident_requests"] = self._local_resident
        result["remote_resident_requests"] = self._resident - result["local_resident_requests"]
        return result
