"""HTTP response handling for errors before and after a PD stream starts."""

import json

import anyio
from fastapi.responses import StreamingResponse


class GatewayError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

    def body(self):
        return {"error": {"message": str(self), "type": "pd_error", "code": self.status}}


async def read_json_limited(request, limit):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > limit:
            raise GatewayError(413, "Request body exceeds gateway limit")
        data.extend(chunk)
    return json.loads(data)


class PrefetchResponse(StreamingResponse):
    """Delay headers until the first result, retaining the disconnect listener."""

    async def stream_response(self, send):
        try:
            try:
                first = await self.body_iterator.__anext__()
            except GatewayError as exc:
                data = json.dumps(exc.body()).encode("utf-8")
                await send({"type": "http.response.start", "status": exc.status,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(data)).encode())] +
                                       [(key, value) for key, value in self.raw_headers
                                        if key not in (b"content-type", b"content-length")]})
                await send({"type": "http.response.body", "body": data, "more_body": False})
                return
            except StopAsyncIteration:
                first = None
            await send({"type": "http.response.start", "status": self.status_code,
                        "headers": self.raw_headers})
            if first is not None:
                if not isinstance(first, (bytes, memoryview)):
                    first = first.encode(self.charset)
                await send({"type": "http.response.body", "body": first, "more_body": True})
            async for chunk in self.body_iterator:
                if not isinstance(chunk, (bytes, memoryview)):
                    chunk = chunk.encode(self.charset)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            # Starlette's disconnect watcher cancels this task while it may be
            # suspended between yields; close the iterator even in that scope.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
