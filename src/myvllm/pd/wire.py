"""Length-delimited JSON control messages and ordinary host bytes."""

import asyncio
import json
import struct


_PREFIX = struct.Struct("!IQ")
DEFAULT_HEADER_LIMIT = 1024 * 1024


class ProtocolError(ValueError):
    pass


async def read_header(reader, max_header_bytes, max_body_bytes):
    prefix = await reader.readexactly(_PREFIX.size)
    header_size, body_size = _PREFIX.unpack(prefix)
    if not 0 < header_size <= max_header_bytes:
        raise ProtocolError("Invalid control frame size")
    if body_size > max_body_bytes:
        raise ProtocolError("Body exceeds transfer limit")
    raw = await reader.readexactly(header_size)
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Invalid JSON control frame") from exc
    if not isinstance(header, dict):
        raise ProtocolError("Control frame must be an object")
    return header, body_size


async def read_body(reader, size):
    return await reader.readexactly(size) if size else b""


async def write_frame(writer, header, body=b"", *, max_header_bytes=DEFAULT_HEADER_LIMIT):
    data = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if not 0 < len(data) <= max_header_bytes:
        raise ProtocolError("Control frame exceeds limit")
    writer.write(_PREFIX.pack(len(data), len(body)))
    writer.write(data)
    if body:
        writer.write(body)
    await writer.drain()


async def close_writer(writer, deadline, *, abort=False):
    if abort:
        writer.transport.abort()
    else:
        writer.close()
    closed = asyncio.create_task(writer.wait_closed())
    try:
        async with asyncio.timeout_at(deadline):
            await asyncio.shield(closed)
    except TimeoutError:
        writer.transport.abort()
        try:
            await closed
        except (ConnectionError, OSError):
            pass
    except asyncio.CancelledError:
        writer.transport.abort()
        await asyncio.gather(closed, return_exceptions=True)
        raise
    except (ConnectionError, OSError):
        pass
