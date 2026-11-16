"""Async client for one connection-scoped transaction per PD worker."""

import asyncio
from dataclasses import asdict
import time

from myvllm.pd.protocol import descriptor_components
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, ProtocolError, close_writer, read_body, read_header, write_frame
from myvllm.utils.timing import write_record


class PDRemoteError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class PDTimingError(OSError):
    """A PD timing record could not be written."""


class PDHeaderLimitError(ProtocolError):
    """A request control header cannot fit the selected worker."""


class PDConnection:
    def __init__(self, reader, writer, deadline, max_header_bytes=DEFAULT_HEADER_LIMIT):
        self._reader = reader
        self._writer = writer
        self._deadline = deadline
        self._max_header_bytes = min(max_header_bytes, DEFAULT_HEADER_LIMIT)
        self._max_body_bytes = 0
        self._lock = asyncio.Lock()
        self.session_id = None
        self.transaction_id = None
        self.request_id = None

    async def _call(self, op, body=b"", **fields):
        async with self._lock:
            try:
                async with asyncio.timeout_at(self._deadline):
                    try:
                        await write_frame(self._writer, {"op": op, **fields}, body,
                                          max_header_bytes=self._max_header_bytes)
                    except ProtocolError as exc:
                        raise PDHeaderLimitError(str(exc)) from exc
                    header, size = await read_header(self._reader, DEFAULT_HEADER_LIMIT,
                                                     self._max_body_bytes)
                    data = await read_body(self._reader, size)
                if set(header) != {"ok", "value"} and set(header) != {"ok", "error", "message"}:
                    raise ProtocolError("Invalid response")
                if header["ok"] is False:
                    raise PDRemoteError(header["error"], header["message"])
                if header["ok"] is not True:
                    raise ProtocolError("Invalid response status")
                return header["value"], data
            except BaseException:
                self._writer.transport.abort()
                raise

    async def next_token(self):
        value, _ = await self._call("next")
        return value

    async def describe(self):
        value, _ = await self._call("describe")
        return value

    async def describe_early(self):
        value, _ = await self._call("describe_early")
        return value

    async def reserve_early(self, metadata):
        value, _ = await self._call("reserve_early", metadata=metadata)
        return value

    async def start(self):
        value, _ = await self._call("start")
        return value

    async def wait_pipeline(self):
        value, _ = await self._call("wait_pipeline")
        return value

    async def commit_early(self, metadata):
        value, _ = await self._call("commit_early", metadata=metadata)
        return value

    async def acknowledge_early(self):
        value, _ = await self._call("acknowledge_early")
        return value

    async def reserve(self, metadata):
        value, _ = await self._call("reserve", metadata=metadata)
        return value

    async def export(self):
        _, data = await self._call("export")
        return data

    async def restore(self, payload):
        if type(payload) is not bytes:
            raise ValueError("payload must be bytes")
        value, _ = await self._call("restore", payload)
        return value

    async def acknowledge(self):
        value, _ = await self._call("acknowledge")
        return value

    async def activate(self):
        value, _ = await self._call("activate")
        return value

    async def cancel(self):
        if self._lock.locked():
            raise RuntimeError("Cannot cancel with an operation in flight; close the connection")
        value, _ = await self._call("cancel")
        return value

    async def close(self):
        try:
            if self._writer.is_closing() or self._lock.locked():
                await close_writer(self._writer, self._deadline, abort=True)
                return
            try:
                await self._call("close")
            except (PDRemoteError, ConnectionError, OSError, asyncio.IncompleteReadError, TimeoutError):
                pass
            await close_writer(self._writer, self._deadline)
        except BaseException:
            self._writer.transport.abort()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class PDClient:
    @staticmethod
    async def _open(endpoint, op, lifetime, *, max_header_bytes=DEFAULT_HEADER_LIMIT, **fields):
        if type(lifetime) not in (int, float) or not 0 < lifetime < float("inf"):
            raise ValueError("lifetime must be a positive finite number")
        deadline = asyncio.get_running_loop().time() + lifetime
        async with asyncio.timeout_at(deadline):
            reader, writer = await asyncio.open_connection(*endpoint)
        connection = PDConnection(reader, writer, deadline, max_header_bytes)
        try:
            value, _ = await connection._call(op, lifetime=lifetime, **fields)
            connection._max_body_bytes = value["max_transfer_bytes"]
            connection.session_id = value["session_id"]
            connection.transaction_id = value.get("transaction_id")
            connection.request_id = fields.get("request_id")
            return connection
        except BaseException:
            await close_writer(writer, deadline, abort=True)
            raise

    @staticmethod
    async def open_d(endpoint, *, lifetime, max_header_bytes=DEFAULT_HEADER_LIMIT):
        return await PDClient._open(endpoint, "open_d", lifetime, max_header_bytes=max_header_bytes)

    @staticmethod
    async def open_p(endpoint, session_id, request_id, token_ids, params, *, lifetime, decode_endpoint=None,
                     max_header_bytes=DEFAULT_HEADER_LIMIT):
        selected = {} if decode_endpoint is None else {"decode_endpoint": decode_endpoint}
        return await PDClient._open(endpoint, "open_p", lifetime, max_header_bytes=max_header_bytes,
                                    session_id=session_id,
                                    request_id=request_id, token_ids=token_ids,
                                    sampling=asdict(params), **selected)

    @staticmethod
    async def prepare_p(endpoint, session_id, request_id, token_ids, params, *, lifetime, decode_endpoint=None,
                        max_header_bytes=DEFAULT_HEADER_LIMIT):
        selected = {} if decode_endpoint is None else {"decode_endpoint": decode_endpoint}
        return await PDClient._open(endpoint, "prepare_p", lifetime, max_header_bytes=max_header_bytes,
                                    session_id=session_id,
                                    request_id=request_id, token_ids=token_ids,
                                    sampling=asdict(params), **selected)

    @staticmethod
    async def open_local(endpoint, request_id, token_ids, params, *, lifetime, max_header_bytes=DEFAULT_HEADER_LIMIT):
        return await PDClient._open(endpoint, "open_local", lifetime, max_header_bytes=max_header_bytes,
                                    request_id=request_id,
                                    token_ids=token_ids, sampling=asdict(params))

    @staticmethod
    async def start_pipeline(prefill, decode, *, on_reserved=None):
        await decode.reserve_early(await prefill.describe_early())
        if on_reserved is not None:
            on_reserved()
        await prefill.start()

    @staticmethod
    async def _status(endpoint, op, timeout):
        deadline = asyncio.get_running_loop().time() + timeout
        async with asyncio.timeout_at(deadline):
            reader, writer = await asyncio.open_connection(*endpoint)
            connection = PDConnection(reader, writer, deadline)
            try:
                value, _ = await connection._call(op)
                return value
            finally:
                await close_writer(writer, deadline)

    @staticmethod
    async def info(endpoint, *, timeout=10):
        return await PDClient._status(endpoint, "info", timeout)

    @staticmethod
    async def stats(endpoint, *, timeout=10):
        return await PDClient._status(endpoint, "stats", timeout)

    @staticmethod
    async def handoff(prefill, decode, metadata, *, timing_path=None, on_reserved=None):
        await decode.reserve(metadata)
        if on_reserved is not None:
            on_reserved()
        started = time.perf_counter()
        payload = await prefill.export()
        nbytes = len(payload)
        try:
            await decode.restore(payload)
        finally:
            del payload
        elapsed = (time.perf_counter() - started) * 1000
        if timing_path:
            try:
                write_record(timing_path, {
                    "event": "pd_transfer_done", "request_id": metadata["request_id"],
                    "session_id": metadata["session_id"],
                    "transaction_id": metadata["transaction_id"],
                    "payload_bytes": nbytes, "transfer_ms": elapsed, **descriptor_components(metadata["kv"]),
                })
            except OSError as exc:
                raise PDTimingError(*exc.args) from exc
        await prefill.acknowledge()
        await decode.activate()

    @staticmethod
    async def handoff_pipeline(prefill, decode, metadata, *, timing_path=None):
        started = time.perf_counter()
        copied = await prefill.wait_pipeline()
        elapsed = (time.perf_counter() - started) * 1000
        await decode.commit_early(metadata)
        if timing_path:
            try:
                write_record(timing_path, {
                    "event": "pd_pipeline_done", "request_id": metadata["request_id"],
                    "session_id": metadata["session_id"],
                    "transaction_id": metadata["transaction_id"],
                    **copied, "residual_wait_ms": elapsed,
                })
            except OSError as exc:
                raise PDTimingError(*exc.args) from exc
        await prefill.acknowledge_early()
        await decode.activate()
