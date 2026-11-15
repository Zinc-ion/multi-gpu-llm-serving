"""TP=1 PD worker: bounded TCP transactions and one engine-owner thread."""

import argparse
import asyncio
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import logging
import queue
import signal
import threading
import time

from myvllm.config import load_config
from myvllm.engine.llm_engine import LLMEngine
from myvllm.pd import DecodeCore, PrefillCore
from myvllm.pd.protocol import finite_number
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, ProtocolError, close_writer, read_body, read_header, write_frame
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils.timing import write_record


logger = logging.getLogger("myvllm.pd.worker")


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


class _Owner:
    def __init__(self, worker, engine_factory):
        self.worker = worker
        self.engine_factory = engine_factory
        self.commands = queue.Queue(maxsize=2 * worker.max_socket_connections + 2)
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, name="pd-model-owner")
        self.error = None
        self.contexts = {}
        self.decode_sessions = {}
        self.core = None
        self.engine = None

    def submit(self, ctx, op, fields=None, payload=b""):
        future = Future()
        if self.error is not None:
            future.set_exception(self.error)
        else:
            self.commands.put_nowait((ctx, op, fields or {}, payload, future))
        return future

    @staticmethod
    def _resolve(future, result):
        if not future.done():
            future.set_result(result)

    def _record(self, event, ctx, **fields):
        path = self.worker.config.get("timing_path")
        if not path:
            return
        handle = ctx.handle
        write_record(path, {
            "event": event, "role": self.worker.role, "time": time.time(),
            "request_id": handle.request_id if handle else None,
            "session_id": handle.session_id if handle else None,
            "transaction_id": handle.transaction_id if handle else None,
            **fields,
        })

    def _info(self):
        scheduler = self.core.scheduler
        return {
            "role": self.worker.role,
            "model": self.core.spec,
            "max_model_length": scheduler.max_model_length,
            "max_completion_tokens": scheduler.max_model_length - 1,
            "max_connections": self.worker.max_connections,
            "max_transfer_bytes": self.worker.max_transfer_bytes,
            "max_header_bytes": self.worker.max_header_bytes,
            "request_timeout": self.worker.request_timeout,
            "sampling_seed": self.core.sampling_seed,
            "decode_endpoint": self.worker.decode_endpoint,
            "max_socket_connections": self.worker.max_socket_connections,
        }

    def _stats(self):
        return {**self.core.stats(), "queued_events": sum(len(c.events) for c in self.contexts.values()),
                "pending_commands": self.commands.qsize(), "open_connections": len(self.contexts)}

    def _close(self, ctx):
        if ctx.pending is not None:
            if not ctx.pending.done():
                ctx.pending.set_exception(ConnectionError("PD connection closed"))
            ctx.pending = None
        if ctx.handle is not None:
            handle = ctx.handle
            if handle.state not in ("finished", "handed_off", "cancelled", "expired", "failed"):
                self._record("pd_cancelled", ctx, reason="connection_closed")
            self.core.close(handle)
            self.contexts.pop(handle, None)
            if self.worker.role == "decode":
                self.decode_sessions.pop(handle.session_id, None)
            self._record("pd_released", ctx, stats=self.core.stats())
            ctx.handle = None
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
        elif op == "open_p" and self.worker.role == "prefill":
            try:
                params = SamplingParams(**fields["sampling"])
            except (TypeError, AttributeError) as exc:
                raise ValueError("Invalid sampling parameters") from exc
            handle = self.core.open(fields["session_id"], fields["request_id"],
                                    fields["token_ids"], params, deadline)
            limit = params.max_model_length or self.core.scheduler.max_model_length
            ctx.event_limit = min(params.max_tokens, limit - len(fields["token_ids"]))
        else:
            raise ValueError("Operation does not match worker role")
        ctx.handle = handle
        self.contexts[handle] = ctx
        if op == "open_d":
            self.decode_sessions[handle.session_id] = ctx
        if op == "open_p":
            self._record("pd_claim_done", ctx, claim_ms=ctx.claim_ms)
            self._record("pd_admitted", ctx, queue_ms=(time.monotonic() - ctx.accepted) * 1000,
                         prompt_tokens=len(fields["token_ids"]))
        return {"session_id": handle.session_id, "transaction_id": handle.transaction_id,
                "max_transfer_bytes": self.worker.max_transfer_bytes}

    def _command(self, ctx, op, fields, payload, future):
        if op == "shutdown":
            return False
        if op == "cleanup":
            if fields.get("reason") and ctx.handle is not None:
                self._record("pd_failed", ctx, reason=fields["reason"], stats=self.core.stats())
            self._close(ctx)
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
        elif op in ("open_p", "open_d"):
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
                      or (self.worker.role == "decode" and handle.state != "active")):
                    raise ValueError("No more token events")
                else:
                    ctx.pending = future
                    return True
            elif op == "describe" and self.worker.role == "prefill":
                result = self.core.describe(handle)
            elif op == "reserve" and self.worker.role == "decode":
                if not ctx.claimed:
                    raise ValueError("Decode session has not been claimed by prefill")
                metadata = fields["metadata"]
                result = self.core.reserve(handle, metadata)
                if result:
                    sampling = metadata["sampling"]
                    ctx.event_limit = min(sampling["max_tokens"],
                                          sampling["max_model_length"] - metadata["prompt_length"]) - 1
                    self._record("pd_reserved", ctx, payload_bytes=metadata["kv"]["nbytes"],
                                 stats=self.core.stats())
            elif op == "export" and self.worker.role == "prefill":
                first = handle.state == "held"
                started = time.perf_counter()
                data = self.core.export(handle)
                if first:
                    self._record("pd_export_done", ctx, payload_bytes=len(data),
                                 export_ms=(time.perf_counter() - started) * 1000)
                self._resolve(future, (None, data))
                return True
            elif op == "restore" and self.worker.role == "decode":
                started = time.perf_counter()
                result = self.core.restore(handle, payload)
                if result:
                    self._record("pd_restore_done", ctx, payload_bytes=len(payload),
                                 restore_ms=(time.perf_counter() - started) * 1000,
                                 stats=self.core.stats())
            elif op == "acknowledge" and self.worker.role == "prefill":
                result = self.core.acknowledge(handle)
                if result:
                    self._record("pd_p_released", ctx, stats=self.core.stats())
            elif op == "activate" and self.worker.role == "decode":
                result = self.core.activate(handle)
            elif op == "cancel":
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
        for handle in self.core.expire():
            ctx = self.contexts[handle]
            self._record("pd_failed", ctx, reason="timeout", stats=self.core.stats())
            if ctx.pending is not None:
                if not ctx.pending.done():
                    ctx.pending.set_exception(TimeoutError("PD request expired"))
                ctx.pending = None
            self.worker.loop.call_soon_threadsafe(ctx.writer.close)

    def _step(self):
        events = self.core.step()
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
        try:
            self.engine = self.engine_factory(self.worker.config)
            core_type = PrefillCore if self.worker.role == "prefill" else DecodeCore
            self.core = core_type(self.engine, self.worker.max_connections,
                                  self.worker.max_transfer_bytes)
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
                    future.set_running_or_notify_cancel()
                    try:
                        running = self._command(ctx, op, fields, payload, future)
                    except ValueError as exc:
                        if not future.done():
                            future.set_exception(exc)
                    except BaseException as exc:
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
            self.error = exc
            logger.exception("PD model owner failed")
            if current is not None and not current[-1].done():
                current[-1].set_exception(exc)
            self.worker.loop.call_soon_threadsafe(self.worker.failed.set)
        finally:
            self.ready.set()
            if self.core is not None:
                for ctx in list(self.contexts.values()):
                    try:
                        self._close(ctx)
                    except Exception:
                        logger.exception("PD cleanup failed")
                    self.worker.loop.call_soon_threadsafe(ctx.writer.close)
            if self.engine is not None:
                try:
                    self.engine.exit()
                except Exception:
                    logger.exception("PD engine shutdown failed")
                    if self.error is None:
                        self.error = RuntimeError("PD engine shutdown failed")
                        self.worker.loop.call_soon_threadsafe(self.worker.failed.set)
            while True:
                try:
                    _, _, _, _, future = self.commands.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_exception(self.error or RuntimeError("PD worker stopped"))


class PDWorker:
    def __init__(self, role, config, *, max_connections, max_transfer_bytes,
                 request_timeout, max_header_bytes=DEFAULT_HEADER_LIMIT,
                 decode_endpoint=None, engine_factory=LLMEngine):
        if role not in ("prefill", "decode"):
            raise ValueError("role must be prefill or decode")
        if config["world_size"] != 1:
            raise ValueError("PD workers require world_size == 1")
        if role == "prefill":
            if (not isinstance(decode_endpoint, (tuple, list)) or len(decode_endpoint) != 2
                    or not isinstance(decode_endpoint[0], str) or not decode_endpoint[0]
                    or type(decode_endpoint[1]) is not int or not 0 < decode_endpoint[1] < 65536):
                raise ValueError("prefill requires a fixed decode_endpoint")
            decode_endpoint = tuple(decode_endpoint)
        elif decode_endpoint is not None:
            raise ValueError("decode workers do not have a decode_endpoint")
        for name, value in (("max_connections", max_connections),
                            ("max_transfer_bytes", max_transfer_bytes),
                            ("max_header_bytes", max_header_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not finite_number(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be a positive finite number")
        self.role = role
        self.config = config
        self.max_connections = max_connections
        self.max_transfer_bytes = max_transfer_bytes
        self.max_header_bytes = max_header_bytes
        self.decode_endpoint = decode_endpoint
        self.max_socket_connections = max_connections * (2 if role == "decode" else 1) + 1
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
            if not result.done():
                result.add_done_callback(lambda item: item.exception() if not item.cancelled() else None)
            await asyncio.gather(disconnect, return_exceptions=True)

    async def _claim(self, session_id, deadline):
        try:
            reader, writer = await asyncio.open_connection(*self.decode_endpoint)
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

    async def _claim_connected(self, reader, session_id, deadline):
        claim = asyncio.create_task(self._claim(session_id, deadline))
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

    async def _accept(self, reader, writer):
        if len(self.connections) >= self.max_socket_connections:
            writer.close()
            await writer.wait_closed()
            return
        task = asyncio.current_task()
        self.connections[task] = writer
        ctx = _Connection(writer, time.monotonic())
        session = False
        expected_payload = None
        cleanup_reason = None
        graceful = False
        deadline = ctx.accepted + self.request_timeout
        try:
            async with asyncio.timeout_at(deadline) as timeout:
                while True:
                    header, size = await read_header(reader, self.max_header_bytes,
                                                     self.max_transfer_bytes)
                    op = header.get("op")
                    expected_fields = {
                        "info": {"op"}, "stats": {"op"}, "open_d": {"op", "lifetime"},
                        "claim": {"op", "session_id"},
                        "open_p": {"op", "lifetime", "session_id", "request_id", "token_ids", "sampling"},
                        "reserve": {"op", "metadata"}, "restore": {"op"},
                        "next": {"op"}, "describe": {"op"}, "export": {"op"},
                        "acknowledge": {"op"}, "activate": {"op"}, "cancel": {"op"}, "close": {"op"},
                    }
                    if not isinstance(op, str) or op not in expected_fields or set(header) != expected_fields[op]:
                        raise ProtocolError("Invalid operation or control fields")
                    if op == "restore":
                        if expected_payload is None or size != expected_payload:
                            raise ProtocolError("Unexpected KV payload size or state")
                    elif size:
                        raise ProtocolError("Unexpected body")
                    if op in ("info", "stats", "claim") and ctx.handle is not None:
                        raise ProtocolError("Status requires a separate connection")
                    if op in ("open_p", "open_d"):
                        if session or self.sessions >= self.max_connections:
                            raise ProtocolError("PD connection capacity exhausted")
                        session = True
                        self.sessions += 1
                        lifetime = header["lifetime"]
                        if not finite_number(lifetime) or not 0 < lifetime <= self.request_timeout:
                            raise ProtocolError("Invalid request lifetime")
                        deadline = ctx.accepted + lifetime
                        timeout.reschedule(deadline)
                        if op == "open_p" and self.role == "prefill":
                            started = time.perf_counter()
                            await self._claim_connected(reader, header["session_id"], deadline)
                            ctx.claim_ms = (time.perf_counter() - started) * 1000
                    body = await read_body(reader, size)
                    try:
                        future = self.owner.submit(ctx, op, {k: v for k, v in header.items() if k != "op"}, body)
                        value, output = await self._await_owner(reader, future)
                    finally:
                        body = None
                        future = None
                    if op in ("open_p", "open_d"):
                        timeout.reschedule(ctx.deadline)
                    elif op == "reserve":
                        expected_payload = header["metadata"]["kv"]["nbytes"]
                    try:
                        await write_frame(writer, {"ok": True, "value": value}, output,
                                          max_header_bytes=self.max_header_bytes)
                    finally:
                        output = None
                    if op in ("close", "info", "stats", "claim"):
                        graceful = True
                        break
        except (ValueError, KeyError) as exc:
            cleanup_reason = str(exc)
            if not writer.is_closing():
                try:
                    await write_frame(writer, {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                                      max_header_bytes=self.max_header_bytes)
                    graceful = True
                except (ConnectionError, OSError):
                    pass
        except TimeoutError:
            cleanup_reason = "timeout"
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            # An admission or copy may still be executing when the socket goes away.
            # The permit remains held until this ordered cleanup reaches the owner.
            try:
                if session and self.owner.thread.is_alive():
                    future = self.owner.submit(ctx, "cleanup", {"reason": cleanup_reason})
                    completion = asyncio.wrap_future(future)
                    try:
                        await asyncio.shield(completion)
                    except asyncio.CancelledError:
                        await asyncio.shield(completion)
                        raise
                    except Exception:
                        logger.exception("PD connection cleanup failed")
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
        if connections:
            await asyncio.gather(*(task for task, _ in connections), return_exceptions=True)
        if self.owner.thread.is_alive():
            self.owner.submit(None, "shutdown")
            await asyncio.to_thread(self.owner.thread.join)
        if self.owner.error is not None:
            raise self.owner.error


async def _serve(args):
    config = load_config(args.config)
    if (args.decode_host is None) != (args.decode_port is None):
        raise ValueError("--decode-host and --decode-port must be supplied together")
    decode_endpoint = (args.decode_host, args.decode_port) if args.decode_host else None
    worker = PDWorker(args.role, config, max_connections=args.max_connections,
                      max_transfer_bytes=args.max_transfer_bytes,
                      request_timeout=args.request_timeout, max_header_bytes=args.max_header_bytes,
                      decode_endpoint=decode_endpoint)
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
    parser.add_argument("--request-timeout", type=float, required=True)
    parser.add_argument("--max-header-bytes", type=int, default=DEFAULT_HEADER_LIMIT)
    parser.add_argument("--decode-host")
    parser.add_argument("--decode-port", type=int)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve(parser.parse_args()))


if __name__ == "__main__":
    main()
