"""PD worker: bounded TCP transactions and one engine-owner thread."""

import argparse
import asyncio
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import json
import logging
import queue
import signal
import socket
import threading
import time
import traceback

from myvllm.config import load_config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.pd import CapacityError, DecodeCore, PrefillCore
from myvllm.pd.pipeline import LayerSender, PipelineTimingError
from myvllm.pd.protocol import descriptor_components, finite_number, token_byte_counts
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, ProtocolError, close_writer, read_body, read_header, write_frame
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils.timing import write_record


logger = logging.getLogger("myvllm.pd.worker")

_OWNER_CLEANUP_COMMANDS = ("cleanup", "cleanup_data", "shutdown")


async def _wait_for_cleanup(future):
    cancelled = None
    while True:
        try:
            await asyncio.shield(future)
            break
        except asyncio.CancelledError as exc:
            if future.cancelled():
                raise
            cancelled = exc
    if cancelled is not None:
        raise cancelled


def _clear_error_frames(error):
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _decode_endpoint(endpoint):
    if (not isinstance(endpoint, (tuple, list)) or len(endpoint) != 2
            or not isinstance(endpoint[0], str) or not endpoint[0]
            or type(endpoint[1]) is not int or not 0 < endpoint[1] < 65536):
        raise ValueError("Invalid decode endpoint")
    return tuple(endpoint)


@dataclass(eq=False)
class _Connection:
    writer: object
    accepted: float
    handle: object = None
    deadline: float = None
    pending: object = None
    events: deque = field(default_factory=deque)
    event_limit: int = 0
    cancelled: bool = False
    claimed: bool = False
    claim_ms: float = 0
    sender: object = None
    data_writer: object = None
    data_target: object = None
    decode_endpoint: object = None
    response_done: object = None
    failure_response: object = None


class _Owner:
    def __init__(self, worker, engine_factory):
        self.worker = worker
        self.engine_factory = engine_factory
        self.commands = queue.Queue(maxsize=2 * worker.max_socket_connections + 2)
        self._command_slots = threading.BoundedSemaphore(worker.max_socket_connections + 1)
        self._submission_lock = threading.Lock()
        self._stopping = False
        self._shutdown_future = None
        self.ready = threading.Event()
        self.cleanup_done = Future()
        self.engine_shutdown_complete = False
        self.thread = threading.Thread(target=self.run, name="pd-model-owner")
        self.error = None
        self.contexts = {}
        self.decode_sessions = {}
        self.core = None
        self.engine = None
        self.decode_step_samples = deque(maxlen=32)

    def submit(self, ctx, op, fields=None, payload=b""):
        future = Future()
        with self._submission_lock:
            if self._stopping:
                error = self.error or RuntimeError("PD worker stopped")
            elif op == "shutdown" and self._shutdown_future is not None:
                return self._shutdown_future
            elif op not in _OWNER_CLEANUP_COMMANDS and not self._command_slots.acquire(blocking=False):
                error = CapacityError("PD owner command capacity exhausted")
            else:
                self.commands.put_nowait((ctx, op, fields or {}, payload, future))
                if op == "shutdown":
                    self._shutdown_future = future
                error = None
        if error is not None:
            future.set_exception(error)
        return future

    def _stop(self, error=None):
        with self._submission_lock:
            if error is not None and self.error is None:
                self.error = error
            self._stopping = True
            failure = self.error or RuntimeError("PD worker stopped")
            pending = []
            while True:
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    break
                if command[1] not in _OWNER_CLEANUP_COMMANDS:
                    self._command_slots.release()
                pending.append(command[-1])
        for future in pending:
            if (not future.done() or future.cancelled()) and future.set_running_or_notify_cancel():
                future.set_exception(failure)

    @staticmethod
    def _resolve(future, result):
        if not future.done():
            future.set_result(result)

    def _record(self, event, ctx, **fields):
        path = self.worker.config.get("timing_path")
        if not path:
            return
        handle = ctx.handle
        if handle and handle.local:
            fields.setdefault("route", "local")
            fields.setdefault("payload_bytes", 0)
        sender_error = ctx.sender.error if ctx.sender is not None else None
        try:
            write_record(path, {
                "event": event, "role": self.worker.role, "time": time.time(),
                "monotonic_ns": time.monotonic_ns(),
                "request_id": handle.request_id if handle else None,
                "session_id": handle.session_id if handle else None,
                "transaction_id": handle.transaction_id if handle else None,
                **fields,
            })
        except OSError:
            if isinstance(sender_error, PipelineTimingError):
                raise sender_error
            raise

    def _info(self):
        scheduler = self.core.scheduler
        return {
            "role": self.worker.role,
            "model": self.core.spec,
            "max_model_length": scheduler.max_model_length,
            "max_completion_tokens": scheduler.max_model_length - 1,
            "max_connections": self.worker.max_connections,
            "max_transfer_bytes": self.worker.max_transfer_bytes,
            "max_payload_bytes": self.worker.max_payload_bytes,
            "max_header_bytes": self.worker.max_header_bytes,
            "request_timeout": self.worker.request_timeout,
            "sampling_seed": self.core.sampling_seed,
            "decode_endpoint": self.worker.decode_endpoint,
            "allowed_decode_endpoints": self.worker.allowed_decode_endpoints,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
            "local_prefill_enabled": self.worker.role == "decode" and scheduler.enable_chunked_prefill,
            "enforce_eager": self.engine.config.get("enforce_eager", False),
            "world_size": self.engine.config["world_size"],
            "max_socket_connections": self.worker.max_socket_connections,
            "pipeline_frame_bytes": self.worker.pipeline_frame_bytes,
        }

    def _stats(self):
        transfers = [ctx.sender or ctx.handle._receiver for ctx in self.contexts.values()
                     if ctx.sender is not None or ctx.handle._receiver is not None]
        staging = [transfer.stats() for transfer in transfers]
        result = {**self.core.stats(), "queued_events": sum(len(c.events) for c in self.contexts.values()),
                "pending_commands": self.commands.qsize(), "open_connections": len(self.contexts),
                "pipeline_staging_bytes": sum(s["staging_bytes"] for s in staging),
                "pipeline_pinned_bytes": sum(s["pinned_bytes"] for s in staging),
                "pipeline_busy_slots": sum(s["busy_slots"] for s in staging),
                "pipeline_slots": sum(s["slot_count"] for s in staging),
                "pipeline_staging_limit_bytes": 2 * self.worker.max_connections * self.worker.pipeline_frame_bytes,
                "pipeline_slot_limit": 2 * self.worker.max_connections if self.worker.pipeline_frame_bytes else 0}
        if self.worker.role == "decode":
            sample = max(self.decode_step_samples, default=None)
            result.update(
                decode_step_wall_ms=sample[0] if sample else None,
                decode_step_sample_count=len(self.decode_step_samples),
                decode_step_sample_source="core.step_wall_decode_including_mixed",
                decode_step_sample_age_ms=(time.perf_counter() - sample[1]) * 1000 if sample else None,
            )
        return result

    def _close(self, ctx):
        if ctx.pending is not None:
            if not ctx.pending.done():
                ctx.pending.set_exception(ConnectionError("PD connection closed"))
            ctx.pending = None
        if ctx.handle is not None:
            handle = ctx.handle
            errors = []
            if ctx.sender is not None:
                try:
                    ctx.sender.close()
                except PipelineTimingError as exc:
                    errors.append(exc)
                if isinstance(ctx.sender.error, PipelineTimingError):
                    errors.insert(0, ctx.sender.error)
                self.engine.model_runner.layer_transfers.pop(handle.sequence.seq_id, None)
                ctx.sender = None
            if ctx.data_writer is not None:
                self.worker.loop.call_soon_threadsafe(ctx.data_writer.transport.abort)
                ctx.data_writer = None
            if handle._receiver is not None:
                try:
                    handle._receiver.close()
                except PipelineTimingError as exc:
                    errors.append(exc)
                handle._receiver = None
            try:
                if handle.state not in ("finished", "handed_off", "cancelled", "expired", "failed"):
                    self._record("pd_cancelled", ctx, reason="connection_closed")
            except OSError as exc:
                errors.append(exc)
            self.core.close(handle)
            self.contexts.pop(handle, None)
            if self.worker.role == "decode":
                self.decode_sessions.pop(handle.session_id, None)
            try:
                self._record("pd_released", ctx, stats=self.core.stats())
            except OSError as exc:
                errors.append(exc)
            ctx.handle = None
            ctx.events.clear()
            if errors:
                raise errors[0]
        else:
            ctx.events.clear()

    def _open(self, ctx, op, fields):
        if ctx.handle is not None:
            raise ValueError("This connection already has a transaction")
        lifetime = fields["lifetime"]
        if not finite_number(lifetime) or not 0 < lifetime <= self.worker.request_timeout:
            raise ValueError("Invalid request lifetime")
        deadline = ctx.accepted + lifetime
        if time.monotonic() >= deadline:
            raise ValueError("Request lifetime expired before admission")
        ctx.deadline = deadline
        if op == "open_d" and self.worker.role == "decode":
            handle = self.core.open(deadline)
        elif ((op in ("open_p", "prepare_p") and self.worker.role == "prefill")
              or (op == "open_local" and self.worker.role == "decode")):
            try:
                params = SamplingParams(**fields["sampling"])
            except (TypeError, AttributeError) as exc:
                raise ValueError("Invalid sampling parameters") from exc
            if op == "prepare_p" and not self.worker.pipeline_frame_bytes:
                raise ValueError("Pipeline is disabled on this worker")
            if op == "open_local":
                handle = self.core.open_local(fields["request_id"], fields["token_ids"], params, deadline)
            else:
                method = self.core.prepare if op == "prepare_p" else self.core.open
                handle = method(fields["session_id"], fields["request_id"],
                                fields["token_ids"], params, deadline)
            limit = params.max_model_length or self.core.scheduler.max_model_length
            ctx.event_limit = min(params.max_tokens, limit - len(fields["token_ids"]))
        else:
            raise ValueError("Operation does not match worker role")
        ctx.handle = handle
        self.contexts[handle] = ctx
        if op == "open_d":
            self.decode_sessions[handle.session_id] = ctx
        if op in ("open_p", "prepare_p", "open_local"):
            if op != "open_local":
                self._record("pd_claim_done", ctx, claim_ms=ctx.claim_ms)
            self._record("pd_admitted", ctx, queue_ms=(time.monotonic() - ctx.accepted) * 1000,
                         prompt_tokens=len(fields["token_ids"]))
        return {"session_id": handle.session_id, "transaction_id": handle.transaction_id,
                "max_transfer_bytes": self.worker.max_transfer_bytes,
                "max_payload_bytes": self.worker.max_payload_bytes}

    def _command(self, ctx, op, fields, payload, future):
        if op == "shutdown":
            self._resolve(future, (True, b""))
            return False
        if op == "cleanup":
            if fields.get("reason") and ctx.handle is not None:
                self._record("pd_failed", ctx, reason=fields["reason"], stats=self.core.stats())
            self._close(ctx)
            self._resolve(future, (True, b""))
            return True
        if op == "cleanup_data":
            target = ctx.data_target
            if target is not None:
                ctx.data_target = None
                try:
                    self._close(target)
                finally:
                    self.worker.loop.call_soon_threadsafe(target.writer.transport.abort)
            self._resolve(future, (True, b""))
            return True
        if op == "info":
            result = self._info()
        elif op == "stats":
            result = self._stats()
        elif op == "claim" and self.worker.role == "decode":
            session_id = fields["session_id"]
            if not isinstance(session_id, str):
                raise ValueError("Invalid decode session claim")
            target = self.decode_sessions.get(session_id)
            if (target is None or target.cancelled or target.claimed or
                    target.handle.state != "open" or time.monotonic() >= target.deadline):
                raise ValueError("Decode session claim rejected")
            target.claimed = True
            result = True
        elif op == "attach_data" and self.worker.role == "decode":
            if (not self.worker.pipeline_frame_bytes or type(fields["frame_bytes"]) is not int
                    or not 0 < fields["frame_bytes"] <= self.worker.pipeline_frame_bytes
                    or not isinstance(fields["session_id"], str)
                    or not isinstance(fields["transaction_id"], str)):
                raise ValueError("Invalid pipeline channel binding or frame limit")
            target = self.decode_sessions.get(fields["session_id"])
            if (target is None or target.cancelled or not target.claimed
                    or target.handle.state != "early_reserved"
                    or target.handle.transaction_id != fields["transaction_id"]
                    or time.monotonic() >= target.deadline):
                raise ValueError("Pipeline channel does not match a live reservation")
            self.core.attach_receiver(target.handle, fields["frame_bytes"],
                                      self.worker.config.get("timing_path"))
            target.data_writer = ctx.writer
            ctx.data_target = target
            result = target
        elif op in ("open_p", "open_d", "prepare_p", "open_local"):
            result = self._open(ctx, op, fields)
        else:
            handle = ctx.handle
            if handle is None:
                raise ValueError("Open a transaction first")
            if ctx.cancelled and op not in ("cancel", "close"):
                raise ValueError("PD connection was cancelled")
            if op == "next":
                if ctx.events:
                    result = ctx.events.popleft()
                elif ((self.worker.role == "prefill" and handle.state not in ("queued", "prefill"))
                      or (self.worker.role == "decode" and handle.state not in ("queued", "prefill", "active"))):
                    raise ValueError("No more token events")
                else:
                    ctx.pending = future
                    return True
            elif op == "describe" and self.worker.role == "prefill":
                result = self.core.describe(handle)
            elif op == "describe_early" and self.worker.role == "prefill":
                result = self.core.describe_early(handle)
            elif op == "connect_pipeline" and self.worker.role == "prefill":
                if ctx.sender is not None or handle.state != "prepared":
                    raise ValueError("P pipeline has already been started")
                identity = {"session_id": handle.session_id, "transaction_id": handle.transaction_id,
                            "request_id": handle.request_id, "role": "prefill"}
                scales = self.engine.model_runner.kv_cache_scales if "kv_cache" in self.core.spec else None
                ctx.sender = LayerSender(self.engine.model_runner.kv_cache, handle.sequence,
                                         ctx.decode_endpoint, self.worker.pipeline_frame_bytes,
                                         identity, handle.deadline, self.worker.config.get("timing_path"), scales)
                result = ctx.sender
            elif op == "start" and self.worker.role == "prefill":
                if ctx.sender is None or not ctx.sender.ready.done() or ctx.sender.ready.exception() is not None:
                    raise ValueError("P pipeline data channel is not ready")
                self.core.ready(handle)
                self.engine.model_runner.layer_transfers[handle.sequence.seq_id] = ctx.sender
                result = self.core.start(handle)
            elif op in ("reserve", "reserve_early") and self.worker.role == "decode":
                if not ctx.claimed:
                    raise ValueError("Decode session has not been claimed by prefill")
                if op == "reserve_early" and not self.worker.pipeline_frame_bytes:
                    raise ValueError("Pipeline is disabled on this worker")
                metadata = fields["metadata"]
                method = self.core.reserve_early if op == "reserve_early" else self.core.reserve
                result = method(handle, metadata)
                if result:
                    sampling = metadata["sampling"]
                    ctx.event_limit = min(sampling["max_tokens"],
                                          sampling["max_model_length"] - metadata["prompt_length"]) - 1
                    self._record("pd_reserved", ctx, payload_bytes=metadata["kv"]["nbytes"],
                                 stats=self.core.stats(), **descriptor_components(metadata["kv"]))
            elif op == "region" and self.worker.role == "decode":
                self.core.restore_region(handle, fields["layer"], fields["start"], fields["end"], payload)
                result = {"copies_done": handle._next_layer == self.core.spec["architecture"]["num_layers"],
                          "payload_bytes": len(payload)}
            elif op == "commit_early" and self.worker.role == "decode":
                result = self.core.commit_early(handle, fields["metadata"])
            elif op == "export" and self.worker.role == "prefill":
                first = handle.state == "held"
                started = time.perf_counter()
                data = self.core.export(handle)
                if first:
                    tp_metrics = ({"tp_export": handle._tp_export_metrics}
                                  if handle._tp_export_metrics is not None else {})
                    components = descriptor_components(handle._metadata["kv"])
                    if tp_metrics and not components:
                        components = {"data_bytes": len(data), "scale_bytes": 0}
                    self._record("pd_export_done", ctx, payload_bytes=len(data),
                                 export_ms=(time.perf_counter() - started) * 1000,
                                 **components, **tp_metrics)
                self._resolve(future, (None, data))
                return True
            elif op == "restore" and self.worker.role == "decode":
                started = time.perf_counter()
                result = self.core.restore(handle, payload)
                if result:
                    self._record("pd_restore_done", ctx, payload_bytes=len(payload),
                                 restore_ms=(time.perf_counter() - started) * 1000,
                                 stats=self.core.stats(), **descriptor_components(handle._metadata["kv"]))
            elif op == "acknowledge" and self.worker.role == "prefill":
                result = self.core.acknowledge(handle)
                if result:
                    self._record("pd_p_released", ctx, stats=self.core.stats())
            elif op == "acknowledge_early" and self.worker.role == "prefill":
                if handle.state != "handed_off":
                    if ctx.sender is None or not ctx.sender.done.done() or ctx.sender.done.exception() is not None:
                        raise ValueError("D has not acknowledged completion of the pipeline copies")
                result = self.core.acknowledge_early(handle)
                if ctx.sender is not None:
                    ctx.sender.close()
                    self.engine.model_runner.layer_transfers.pop(handle.sequence.seq_id, None)
                    ctx.sender = None
                if result:
                    self._record("pd_p_released", ctx, stats=self.core.stats())
            elif op == "activate" and self.worker.role == "decode":
                result = self.core.activate(handle)
            elif op == "cancel":
                if ctx.sender is not None:
                    ctx.sender.abort(ConnectionError("PD request cancelled"))
                queued = bool(ctx.events)
                ctx.events.clear()
                ctx.cancelled = True
                result = self.core.cancel(handle) or queued
                if result:
                    self._record("pd_cancelled", ctx, reason="client", stats=self.core.stats())
            elif op == "close":
                self._close(ctx)
                result = True
            else:
                raise ValueError("Unsupported operation for worker role")
        self._resolve(future, (result, b""))
        return True

    def _poll(self):
        if self.worker.pipeline_frame_bytes and self.worker.role == "prefill":
            for ctx in list(self.contexts.values()):
                if ctx.sender is not None and isinstance(ctx.sender.error, PipelineTimingError):
                    if ctx.pending is not None and not ctx.pending.done():
                        ctx.pending.set_exception(ctx.sender.error)
                        ctx.pending = None
                    raise ctx.sender.error
                if ctx.sender is not None and ctx.sender.error is not None and not ctx.cancelled:
                    ctx.cancelled = True
                    self.core.cancel(ctx.handle)
                    ctx.events.clear()
                    self._record("pd_failed", ctx, reason=str(ctx.sender.error), stats=self.core.stats())
                    if ctx.pending is not None and not ctx.pending.done():
                        ctx.pending.set_exception(ctx.sender.error)
                        ctx.pending = None
                    self.worker.loop.call_soon_threadsafe(ctx.writer.close)
        for handle in self.core.expire():
            ctx = self.contexts[handle]
            self._record("pd_failed", ctx, reason="timeout", stats=self.core.stats())
            if ctx.pending is not None:
                if not ctx.pending.done():
                    ctx.pending.set_exception(TimeoutError("PD request expired"))
                ctx.pending = None
            self.worker.loop.call_soon_threadsafe(ctx.writer.close)

    def _step(self):
        started = time.perf_counter()
        events = self.core.step()
        finished = time.perf_counter()
        if self.worker.role == "decode" and any(event.completion_tokens > 1 for event in events):
            self.decode_step_samples.append(((finished - started) * 1000, finished))
        for event in events:
            ctx = self.contexts[event.handle]
            item = {"token": event.token, "finished": event.finished,
                    "finish_reason": event.finish_reason, "prompt_tokens": event.prompt_tokens,
                    "completion_tokens": event.completion_tokens}
            if event.completion_tokens == 1:
                self._record("pd_first", ctx, elapsed_ms=(time.monotonic() - ctx.accepted) * 1000)
            elif event.completion_tokens == 2:
                self._record("pd_second", ctx, elapsed_ms=(time.monotonic() - ctx.accepted) * 1000)
            if event.finished:
                self._record("pd_finished", ctx, completion_tokens=event.completion_tokens,
                             finish_reason=event.finish_reason, stats=self.core.stats())
            if ctx.pending is not None:
                if not ctx.pending.done():
                    ctx.pending.set_result((item, b""))
                ctx.pending = None
            else:
                assert len(ctx.events) < ctx.event_limit
                ctx.events.append(item)
        return bool(events)

    def run(self):
        current = None
        timing_response = None
        try:
            self.engine = self.engine_factory(self.worker.config)
            core_type = PrefillCore if self.worker.role == "prefill" else DecodeCore
            self.core = core_type(self.engine, self.worker.max_connections,
                                  self.worker.max_transfer_bytes, self.worker.max_payload_bytes)
            if self.worker.pipeline_frame_bytes:
                if self.worker.pipeline_frame_bytes < sum(token_byte_counts(self.core.spec)):
                    raise ValueError("Pipeline frame must hold at least one K/V token")
            if self.worker.pipeline_frame_bytes and self.worker.role == "prefill":
                self.engine.model_runner.layer_transfers = {}
            self.ready.set()
            running = True
            while running:
                self._poll()
                try:
                    scheduler = self.core.scheduler
                    blocked = (scheduler.is_finished() or
                               (not scheduler.running and scheduler.waiting and
                                not scheduler.block_manager.can_allocate(scheduler.waiting[0])))
                    current = self.commands.get(timeout=0.005 if blocked else 0)
                except queue.Empty:
                    current = None
                while current is not None:
                    ctx, op, fields, payload, future = current
                    if op not in _OWNER_CLEANUP_COMMANDS:
                        self._command_slots.release()
                    try:
                        if future.set_running_or_notify_cancel():
                            running = self._command(ctx, op, fields, payload, future)
                    except ValueError as exc:
                        if not future.done():
                            future.set_exception(exc)
                    except BaseException as exc:
                        self._stop(exc)
                        if not future.done():
                            future.set_exception(exc)
                        raise
                    finally:
                        payload = None
                        current = None
                    if not running:
                        break
                    try:
                        current = self.commands.get_nowait()
                    except queue.Empty:
                        pass
                if running:
                    self._poll()
                    if not self.core.scheduler.is_finished():
                        self._step()
        except BaseException as exc:
            self._stop(exc)
            logger.exception("PD model owner failed")
            if current is not None and not current[-1].done():
                current[-1].set_exception(exc)
            if isinstance(exc, PipelineTimingError):
                timing_response = next(
                    ((ctx, ctx.failure_response) for ctx in self.contexts.values()
                     if ctx.sender is not None and ctx.sender.error is exc and ctx.response_done is not None),
                    None,
                )
            if timing_response is None:
                self.worker.loop.call_soon_threadsafe(self.worker.failed.set)
        finally:
            self._stop()
            self.ready.set()
            if self.core is not None:
                for ctx in list(self.contexts.values()):
                    try:
                        self._close(ctx)
                    except Exception:
                        logger.exception("PD cleanup failed")
                    if timing_response is None or ctx is not timing_response[0]:
                        self.worker.loop.call_soon_threadsafe(ctx.writer.close)
            self.cleanup_done.set_result(None)
            if self.engine is not None:
                try:
                    self.engine.exit()
                except Exception:
                    logger.exception("PD engine shutdown failed")
                    if self.error is None:
                        self._stop(RuntimeError("PD engine shutdown failed"))
                        self.worker.loop.call_soon_threadsafe(self.worker.failed.set)
                    # A rank failure can make exit raise after it has reaped every
                    # process and closed the command resources.
                    fields = ("model_runner", "processes", "events", "_shm", "_store")
                    if isinstance(self.engine, LLMEngine) and all(name in vars(self.engine) for name in fields):
                        self.engine_shutdown_complete = not any(getattr(self.engine, name) for name in fields)
                else:
                    self.engine_shutdown_complete = True
            self._stop()
            if timing_response is not None:
                self.worker.loop.call_soon_threadsafe(
                    self.worker._defer_timing_failure, *timing_response, self.worker.failed)


class PDWorker:
    def __init__(self, role, config, *, max_connections, max_transfer_bytes,
                 request_timeout, max_header_bytes=DEFAULT_HEADER_LIMIT,
                 decode_endpoint=None, allowed_decode_endpoints=None,
                 engine_factory=LLMEngine, pipeline_frame_bytes=0, max_payload_bytes=None):
        if role not in ("prefill", "decode"):
            raise ValueError("role must be prefill or decode")
        world_size = config["world_size"]
        if world_size != 1 and not (role == "prefill" and world_size == 2):
            raise ValueError("PD workers require TP1 or TP2 bulk prefill")
        if world_size == 2:
            if pipeline_frame_bytes:
                raise ValueError("TP2 prefill supports bulk only")
            if config["num_kv_heads"] % 2:
                raise ValueError("TP2 prefill requires an even number of KV heads")
        if role == "prefill":
            decode_endpoint = _decode_endpoint(decode_endpoint)
            if allowed_decode_endpoints is None:
                allowed_decode_endpoints = (decode_endpoint,)
            elif not isinstance(allowed_decode_endpoints, (tuple, list)) or not allowed_decode_endpoints:
                raise ValueError("allowed_decode_endpoints must be a nonempty list")
            else:
                allowed_decode_endpoints = tuple(_decode_endpoint(endpoint) for endpoint in allowed_decode_endpoints)
                if len(set(allowed_decode_endpoints)) != len(allowed_decode_endpoints):
                    raise ValueError("allowed_decode_endpoints must be unique")
                if decode_endpoint not in allowed_decode_endpoints:
                    raise ValueError("decode_endpoint must belong to allowed_decode_endpoints")
        elif decode_endpoint is not None or allowed_decode_endpoints is not None:
            raise ValueError("decode workers do not have decode endpoints")
        for name, value in (("max_connections", max_connections),
                            ("max_transfer_bytes", max_transfer_bytes),
                            ("max_header_bytes", max_header_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_payload_bytes is None:
            max_payload_bytes = max_transfer_bytes
        if type(max_payload_bytes) is not int or not 0 < max_payload_bytes <= max_transfer_bytes:
            raise ValueError("max_payload_bytes must be positive and no greater than max_transfer_bytes")
        if not finite_number(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be a positive finite number")
        if type(pipeline_frame_bytes) is not int or not 0 <= pipeline_frame_bytes <= max_payload_bytes:
            raise ValueError("pipeline_frame_bytes must be between 0 and max_payload_bytes")
        if pipeline_frame_bytes and role == "prefill":
            if not config.get("enforce_eager", False) or config.get("enable_chunked_prefill", False):
                raise ValueError("Pipeline requires TP1 whole eager prefill")
        self.role = role
        self.config = {**config, "_pd_tp_bulk": True} if world_size == 2 else config
        self.max_connections = max_connections
        self.max_transfer_bytes = max_transfer_bytes
        self.max_payload_bytes = max_payload_bytes
        self.max_header_bytes = max_header_bytes
        self.decode_endpoint = decode_endpoint
        self.allowed_decode_endpoints = allowed_decode_endpoints
        self.pipeline_frame_bytes = pipeline_frame_bytes
        self.max_socket_connections = max_connections * (
            (3 if pipeline_frame_bytes else 2) if role == "decode" else 1) + 1
        self.request_timeout = request_timeout
        self.engine_factory = engine_factory
        self.server = None
        self._starting = None
        self.connections = {}
        self.sessions = 0

    async def start(self, host, port):
        self.loop = asyncio.get_running_loop()
        self.failed = asyncio.Event()
        self._starting = asyncio.Event()
        self.owner = _Owner(self, self.engine_factory)
        self.owner.thread.start()
        try:
            await asyncio.to_thread(self.owner.ready.wait)
            if self.owner.error is not None:
                raise self.owner.error
            self.server = await asyncio.start_server(self._accept, host, port)
        except BaseException:
            if self.owner.thread.is_alive():
                self.owner.submit(None, "shutdown")
            await asyncio.to_thread(self.owner.thread.join)
            raise
        finally:
            self._starting.set()
        return self.server.sockets[0].getsockname()[:2]

    async def _await_owner(self, reader, future):
        result = asyncio.ensure_future(asyncio.wrap_future(future))
        disconnect = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait((result, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                data = disconnect.result()
                if data:
                    raise ProtocolError("Pipelined commands are not supported")
                raise ConnectionError("Peer disconnected")
            return result.result()
        finally:
            disconnect.cancel()
            if result.done():
                if not result.cancelled():
                    result.exception()
            else:
                result.add_done_callback(lambda item: item.exception() if not item.cancelled() else None)
            await asyncio.gather(disconnect, return_exceptions=True)

    async def _claim(self, session_id, deadline, decode_endpoint):
        try:
            reader, writer = await asyncio.open_connection(*decode_endpoint)
            completed = False
            try:
                await write_frame(writer, {"op": "claim", "session_id": session_id})
                response, size = await read_header(reader, self.max_header_bytes, 0)
                if size or response.get("ok") is not True or response.get("value") is not True:
                    raise ProtocolError("Decode session claim rejected")
                completed = True
            finally:
                await close_writer(writer, deadline, abort=not completed)
        except (OSError, asyncio.IncompleteReadError) as exc:
            raise ProtocolError("Decode session claim failed") from exc

    async def _claim_connected(self, reader, session_id, deadline, decode_endpoint):
        claim = asyncio.create_task(self._claim(session_id, deadline, decode_endpoint))
        disconnect = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait((claim, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if disconnect in done:
                if disconnect.result():
                    raise ProtocolError("Pipelined commands are not supported")
                raise ConnectionError("Peer disconnected")
            return claim.result()
        finally:
            disconnect.cancel()
            if not claim.done():
                claim.cancel()
            await asyncio.gather(disconnect, claim, return_exceptions=True)

    async def _data_connection(self, reader, writer, ctx, header, size):
        if size or set(header) != {"op", "session_id", "transaction_id", "frame_bytes"}:
            raise ProtocolError("Invalid pipeline channel header")
        future = self.owner.submit(ctx, "attach_data", {k: v for k, v in header.items() if k != "op"})
        target, _ = await self._await_owner(reader, future)
        frame_bytes = header["frame_bytes"]
        sock = writer.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, frame_bytes)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, frame_bytes)
        writer.transport.set_write_buffer_limits(high=frame_bytes, low=0)
        async with asyncio.timeout_at(target.deadline):
            await write_frame(writer, {"ok": True, "value": True})
            while True:
                region, length = await read_header(reader, self.max_header_bytes, frame_bytes)
                if set(region) != {"op", "layer", "start", "end"} or region["op"] != "region":
                    raise ProtocolError("Invalid pipeline region header")
                payload = await read_body(reader, length)
                try:
                    future = self.owner.submit(target, "region",
                                               {k: v for k, v in region.items() if k != "op"}, payload)
                    value, _ = await self._await_owner(reader, future)
                finally:
                    payload = None
                    future = None
                await write_frame(writer, {"ok": True, "value": value})
                if value["copies_done"]:
                    ctx.data_target = None
                    target.data_writer = None
                    return

    def _defer_timing_failure(self, ctx, response_done, failed):
        async def finish():
            try:
                async with asyncio.timeout(2):
                    await response_done.wait()
            except TimeoutError:
                pass
            finally:
                ctx.writer.close()
                failed.set()
        asyncio.create_task(finish())

    async def _accept(self, reader, writer):
        if len(self.connections) >= self.max_socket_connections:
            writer.close()
            await writer.wait_closed()
            return
        task = asyncio.current_task()
        self.connections[task] = writer
        owner = self.owner
        ctx = _Connection(writer, time.monotonic(), failure_response=asyncio.Event())
        session = False
        data_channel = False
        expected_payload = None
        cleanup_reason = None
        graceful = False
        deadline = ctx.accepted + self.request_timeout
        try:
            async with asyncio.timeout_at(deadline) as timeout:
                while True:
                    header, size = await read_header(reader, self.max_header_bytes,
                                                     self.max_payload_bytes)
                    op = header.get("op")
                    if op == "pipeline" and not session and self.role == "decode":
                        data_channel = True
                        await self._data_connection(reader, writer, ctx, header, size)
                        graceful = True
                        break
                    expected_fields = {
                        "info": {"op"}, "stats": {"op"}, "open_d": {"op", "lifetime"},
                        "claim": {"op", "session_id"},
                        "open_p": {"op", "lifetime", "session_id", "request_id", "token_ids", "sampling"},
                        "prepare_p": {"op", "lifetime", "session_id", "request_id", "token_ids", "sampling"},
                        "open_local": {"op", "lifetime", "request_id", "token_ids", "sampling"},
                        "reserve": {"op", "metadata"}, "restore": {"op"},
                        "reserve_early": {"op", "metadata"}, "commit_early": {"op", "metadata"},
                        "describe_early": {"op"}, "start": {"op"}, "wait_pipeline": {"op"},
                        "acknowledge_early": {"op"},
                        "next": {"op"}, "describe": {"op"}, "export": {"op"},
                        "acknowledge": {"op"}, "activate": {"op"}, "cancel": {"op"}, "close": {"op"},
                    }
                    fields = expected_fields.get(op) if isinstance(op, str) else None
                    if (fields is None or
                            (set(header) != fields and
                             not (op in ("open_p", "prepare_p") and set(header) == fields | {"decode_endpoint"}))):
                        raise ProtocolError("Invalid operation or control fields")
                    if op == "restore":
                        if expected_payload is None or size != expected_payload:
                            raise ProtocolError("Unexpected KV payload size or state")
                    elif size:
                        raise ProtocolError("Unexpected body")
                    if op in ("info", "stats", "claim") and ctx.handle is not None:
                        raise ProtocolError("Status requires a separate connection")
                    if op in ("open_p", "open_d", "prepare_p", "open_local"):
                        if session or self.sessions >= self.max_connections:
                            raise ProtocolError("PD connection capacity exhausted")
                        session = True
                        self.sessions += 1
                        lifetime = header["lifetime"]
                        if not finite_number(lifetime) or not 0 < lifetime <= self.request_timeout:
                            raise ProtocolError("Invalid request lifetime")
                        deadline = ctx.accepted + lifetime
                        timeout.reschedule(deadline)
                        if op in ("open_p", "prepare_p") and self.role == "prefill":
                            endpoint = _decode_endpoint(header["decode_endpoint"]) if "decode_endpoint" in header else self.decode_endpoint
                            if endpoint not in self.allowed_decode_endpoints:
                                raise ProtocolError("Decode endpoint is not allowed")
                            ctx.decode_endpoint = endpoint
                            started = time.perf_counter()
                            await self._claim_connected(reader, header["session_id"], deadline, endpoint)
                            ctx.claim_ms = (time.perf_counter() - started) * 1000
                    body = await read_body(reader, size)
                    if self.role == "prefill" and (op == "start" or ctx.sender is not None):
                        ctx.response_done = asyncio.Event()
                    future = None
                    try:
                        if op in ("cancel", "close") and ctx.sender is not None:
                            ctx.sender.abort(ConnectionError("PD request cancelled"))
                        if op == "wait_pipeline":
                            sender = ctx.sender
                            if sender is None and isinstance(self.owner.error, PipelineTimingError):
                                raise self.owner.error
                            if sender is None:
                                raise ProtocolError("P has no pipeline to wait for")
                            value, output = await self._await_owner(reader, sender.done)
                        else:
                            if op == "start":
                                future = self.owner.submit(ctx, "connect_pipeline")
                                sender, _ = await self._await_owner(reader, future)
                                await self._await_owner(reader, sender.ready)
                            future = self.owner.submit(ctx, op, {k: v for k, v in header.items() if k != "op"}, body)
                            value, output = await self._await_owner(reader, future)
                    finally:
                        if op in ("info", "stats") and future is not None:
                            future.cancel()
                        body = None
                        future = None
                    if op in ("open_p", "open_d", "prepare_p", "open_local"):
                        timeout.reschedule(ctx.deadline)
                    elif op == "reserve":
                        expected_payload = header["metadata"]["kv"]["nbytes"]
                    elif op == "close":
                        # The owner has released this session; the socket is counted separately.
                        self.sessions -= 1
                        session = False
                    try:
                        await write_frame(writer, {"ok": True, "value": value}, output,
                                          max_header_bytes=self.max_header_bytes)
                    finally:
                        output = None
                    if ctx.response_done is not None:
                        ctx.response_done.set()
                    if op in ("close", "info", "stats", "claim"):
                        graceful = True
                        break
        except (ValueError, KeyError, PipelineTimingError) as exc:
            cleanup_reason = str(exc)
            if not writer.is_closing():
                try:
                    await write_frame(writer, {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                                      max_header_bytes=self.max_header_bytes)
                    graceful = True
                except (ConnectionError, OSError):
                    pass
            if isinstance(exc, PipelineTimingError):
                ctx.failure_response.set()
        except TimeoutError:
            cleanup_reason = "timeout"
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            # An admission or copy may still be executing when the socket goes away.
            # The permit remains held until this ordered cleanup reaches the owner.
            if ctx.response_done is not None:
                ctx.response_done.set()
            ctx.failure_response.set()
            if ctx.sender is not None:
                ctx.sender.abort(ConnectionError("PD control connection closed"))
            try:
                if (session or data_channel) and owner.thread.is_alive() and owner.error is None:
                    future = owner.submit(ctx, "cleanup_data" if data_channel else "cleanup",
                                          {"reason": cleanup_reason})
                    completion = asyncio.wrap_future(future)
                    try:
                        await _wait_for_cleanup(completion)
                    except Exception:
                        logger.exception("PD connection cleanup failed")
            finally:
                try:
                    if (session or data_channel) and (owner.error is not None or owner._stopping):
                        completion = asyncio.wrap_future(owner.cleanup_done)
                        await _wait_for_cleanup(completion)
                finally:
                    try:
                        await close_writer(writer, deadline, abort=not graceful)
                    finally:
                        if session:
                            self.sessions -= 1
                        self.connections.pop(task, None)

    async def close(self):
        if self._starting is not None:
            await self._starting.wait()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        connections = list(self.connections.items())
        for _, writer in connections:
            writer.transport.abort()
        cancelled = None
        if connections:
            completion = asyncio.gather(*(task for task, _ in connections), return_exceptions=True)
            try:
                await _wait_for_cleanup(completion)
            except asyncio.CancelledError as exc:
                cancelled = exc
        if self.owner.thread.is_alive():
            self.owner.submit(None, "shutdown")
            completion = asyncio.create_task(asyncio.to_thread(self.owner.thread.join))
            try:
                await _wait_for_cleanup(completion)
            except asyncio.CancelledError as exc:
                cancelled = exc
        if self.owner.error is not None and self.owner.engine_shutdown_complete:
            _clear_error_frames(self.owner.error)
        if cancelled is not None:
            raise cancelled
        if self.owner.error is not None:
            raise self.owner.error


async def _serve(args):
    config = load_config(args.config)
    if (args.decode_host is None) != (args.decode_port is None):
        raise ValueError("--decode-host and --decode-port must be supplied together")
    decode_endpoint = (args.decode_host, args.decode_port) if args.decode_host else None
    worker = PDWorker(args.role, config, max_connections=args.max_connections,
                      max_transfer_bytes=args.max_transfer_bytes,
                      max_payload_bytes=args.max_payload_bytes,
                      request_timeout=args.request_timeout, max_header_bytes=args.max_header_bytes,
                      decode_endpoint=decode_endpoint, allowed_decode_endpoints=args.allowed_decode_endpoints,
                      pipeline_frame_bytes=args.pipeline_frame_bytes)
    address = await worker.start(args.host, args.port)
    logger.info("PD %s listening on %s:%s", args.role, *address)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopped.set)
    stop_task = asyncio.create_task(stopped.wait())
    fail_task = asyncio.create_task(worker.failed.wait())
    try:
        await asyncio.wait((stop_task, fail_task), return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop_task.cancel()
        fail_task.cancel()
        await worker.close()


def main():
    parser = argparse.ArgumentParser(description="TP=1 prefill or decode worker")
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-connections", type=int, required=True)
    parser.add_argument("--max-transfer-bytes", type=int, required=True)
    parser.add_argument("--max-payload-bytes", type=int)
    parser.add_argument("--request-timeout", type=float, required=True)
    parser.add_argument("--max-header-bytes", type=int, default=DEFAULT_HEADER_LIMIT)
    parser.add_argument("--decode-host")
    parser.add_argument("--decode-port", type=int)
    parser.add_argument("--allowed-decode-endpoints", type=json.loads)
    parser.add_argument("--pipeline-frame-bytes", type=int, default=0)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve(parser.parse_args()))


if __name__ == "__main__":
    main()
