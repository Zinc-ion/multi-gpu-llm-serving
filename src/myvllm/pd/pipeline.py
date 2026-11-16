"""Two-slot layer transfers; CUDA events never cross the TCP connection."""

from collections import deque
from concurrent.futures import Future
import json
import socket
import threading
import time

import numpy as np
import torch

from myvllm.pd.protocol import cache_token_byte_counts
from myvllm.pd.wire import DEFAULT_HEADER_LIMIT, ProtocolError, _PREFIX
from myvllm.utils.timing import write_record


class PipelineError(ValueError):
    """One connection-scoped layer transfer failed."""


def contiguous_regions(table, start, end, block_size, max_tokens):
    while start < end:
        stop = min(end, start + max_tokens, (start // block_size + 1) * block_size)
        while stop < min(end, start + max_tokens):
            index = stop // block_size
            if table[index] != table[index - 1] + 1:
                break
            stop = min(end, start + max_tokens, stop + block_size)
        physical = table[start // block_size] * block_size + start % block_size
        yield start, stop, physical
        start = stop


def _recv_exact(sock, size):
    data = bytearray(size)
    view = memoryview(data)
    while view:
        count = sock.recv_into(view)
        if not count:
            raise ConnectionError("Pipeline peer disconnected")
        view = view[count:]
    return data


def _response(sock):
    header_size, body_size = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if not 0 < header_size <= DEFAULT_HEADER_LIMIT or body_size:
        raise ProtocolError("Invalid pipeline response size")
    header = json.loads(_recv_exact(sock, header_size))
    if not isinstance(header, dict) or set(header) != {"ok", "value"} or header["ok"] is not True:
        raise PipelineError(f"Pipeline peer rejected transfer: {header}")
    return header["value"]


def _send(sock, header, body=b""):
    encoded = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    sock.sendall(_PREFIX.pack(len(encoded), len(body)))
    sock.sendall(encoded)
    if body:
        sock.sendall(body)


class _Staging:
    def __init__(self, cache, frame_bytes, identity, timing_path, scales=None):
        self.cache = cache
        self.scales = scales
        self.frame_bytes = frame_bytes
        self.identity = identity
        self.timing_path = timing_path
        self.started = time.perf_counter()
        self.cuda = cache.is_cuda
        self.device = cache.device
        self.data_token_bytes, self.scale_token_bytes = cache_token_byte_counts(
            cache.shape[-2], cache.shape[-1], cache.dtype, scales is not None)
        self.token_bytes = self.data_token_bytes + self.scale_token_bytes
        if type(frame_bytes) is not int or frame_bytes < self.token_bytes:
            raise ValueError("Pipeline frame must hold at least one K/V token")
        self.slots = [torch.empty(frame_bytes, dtype=torch.uint8, device="cpu",
                                  pin_memory=self.cuda) for _ in range(2)]
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self.origin = None
        if self.cuda and timing_path:
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record(torch.cuda.current_stream(self.device))
            self.stream.wait_event(self.origin)
        self.payload_bytes = 0
        self.frames = 0

    def _record(self, event, **fields):
        if self.timing_path:
            write_record(self.timing_path, {
                "event": event, **self.identity,
                "elapsed_ms": (time.perf_counter() - self.started) * 1000,
                "monotonic_ns": time.monotonic_ns(), **fields,
            })

    def drain_copies(self):
        if self.stream is not None:
            self.stream.synchronize()

    def stats(self):
        size = sum(slot.numel() for slot in self.slots)
        return {"staging_bytes": size, "pinned_bytes": size if self.cuda else 0,
                "slot_count": len(self.slots), "frame_bytes": self.frame_bytes,
                "payload_bytes": self.payload_bytes, "frames": self.frames,
                **self.byte_counts(self.payload_bytes)}

    def byte_counts(self, nbytes):
        if self.scales is None:
            return {}
        tokens = nbytes // self.token_bytes
        return {"data_bytes": tokens * self.data_token_bytes,
                "scale_bytes": tokens * self.scale_token_bytes}


class LayerSender(_Staging):
    """The model owner produces; this request's socket thread drains two slots."""

    def __init__(self, cache, sequence, endpoint, frame_bytes, identity, deadline, timing_path=None, scales=None):
        super().__init__(cache, frame_bytes, identity, timing_path, scales)
        self.sequence = sequence
        self.endpoint = endpoint
        self.deadline = deadline
        self.ready = Future()
        self.done = Future()
        self.error = None
        self.condition = threading.Condition()
        self.free = deque(range(2))
        self.pending = deque()
        self.socket = None
        self.sent_bytes = 0
        self.partial_send_unknown = False
        self.peak_slots = 0
        self.thread = threading.Thread(target=self._run, name="pd-layer-sender")
        self.thread.start()

    def abort(self, error):
        # This path also runs on the network loop while the owner waits for a slot.
        with self.condition:
            if self.error is None:
                self.error = PipelineError(str(error))
            for future in (self.ready, self.done):
                if not future.done():
                    future.set_exception(self.error)
            sock = self.socket
            self.condition.notify_all()
        if sock is not None:
            # Only the socket thread closes its fd: close from here can race
            # recv's poll and let it wait on a reused fd until the old timeout.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def on_layer(self, layer):
        if self.error is not None:
            return
        ready = None
        if self.cuda:
            ready = torch.cuda.Event(enable_timing=bool(self.timing_path))
            ready.record(torch.cuda.current_stream(self.device))
        try:
            self._record("pd_layer_ready", layer=layer)
        except OSError as exc:
            self.abort(exc)
            return
        seq = self.sequence
        for start, end, physical in contiguous_regions(
                seq.block_table, 0, seq.num_prompt_tokens, seq.block_size,
                self.frame_bytes // self.token_bytes):
            with self.condition:
                while not self.free and self.error is None:
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.condition.wait(remaining)
                if self.error is not None:
                    return
                if not self.free:
                    timed_out = True
                else:
                    timed_out = False
                    index = self.free.popleft()
                    self.peak_slots = max(self.peak_slots, 2 - len(self.free))
            if timed_out:
                self.abort(TimeoutError("Pipeline slot deadline expired"))
                return
            count = end - start
            size = count * self.token_bytes
            data_size = count * self.data_token_bytes
            host = self.slots[index][:data_size].view(self.cache.dtype).reshape(
                2, count, self.cache.shape[-2], self.cache.shape[-1])
            host_scales = self.slots[index][data_size:size]
            begin = complete = None
            if self.cuda:
                self.stream.wait_event(ready)
                with torch.cuda.stream(self.stream):
                    if self.timing_path:
                        begin = torch.cuda.Event(enable_timing=True)
                        begin.record(self.stream)
                    for kv in range(2):
                        host[kv].copy_(self.cache[kv, layer].flatten(0, 1)[physical:physical + count],
                                       non_blocking=True)
                        if self.scales is not None:
                            width = count * self.scale_token_bytes // 2
                            host_scales[kv * width:(kv + 1) * width].copy_(
                                self.scales[kv, layer].flatten(0, 1)[physical:physical + count]
                                .view(torch.uint8).reshape(-1), non_blocking=True)
                    complete = torch.cuda.Event(enable_timing=bool(self.timing_path))
                    complete.record(self.stream)
            else:
                for kv in range(2):
                    host[kv].copy_(self.cache[kv, layer].flatten(0, 1)[physical:physical + count])
                    if self.scales is not None:
                        width = count * self.scale_token_bytes // 2
                        host_scales[kv * width:(kv + 1) * width].copy_(
                            self.scales[kv, layer].flatten(0, 1)[physical:physical + count]
                            .view(torch.uint8).reshape(-1))
            with self.condition:
                self.pending.append((index, layer, start, end, size, ready, begin, complete))
                self.condition.notify_all()

    def _run(self):
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pipeline connection deadline expired")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(remaining)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.frame_bytes)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.frame_bytes)
                with self.condition:
                    self.socket = sock
                    if self.error is not None:
                        return
                sock.connect(self.endpoint)
                self._record("pd_pipeline_connected",
                             socket_send_buffer=sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
                             socket_receive_buffer=sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
                             **self.stats())
                _send(sock, {"op": "pipeline", "session_id": self.identity["session_id"],
                             "transaction_id": self.identity["transaction_id"],
                             "frame_bytes": self.frame_bytes})
                if _response(sock) is not True:
                    raise ProtocolError("Pipeline channel was not made ready")
                with self.condition:
                    if self.error is not None:
                        return
                    self.ready.set_result((True, b""))
                while True:
                    with self.condition:
                        while not self.pending and self.error is None:
                            remaining = self.deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("Pipeline producer deadline expired")
                            self.condition.wait(remaining)
                        if self.error is not None:
                            return
                        index, layer, start, end, size, ready, begin, complete = self.pending.popleft()
                    if complete is not None:
                        complete.synchronize()
                    timing = {}
                    if self.origin is not None:
                        timing = {"compute_ready_gpu_ms": self.origin.elapsed_time(ready),
                                  "d2h_start_gpu_ms": self.origin.elapsed_time(begin),
                                  "d2h_end_gpu_ms": self.origin.elapsed_time(complete)}
                    self._record("pd_d2h_done", layer=layer, start=start, end=end,
                                 payload_bytes=size, **self.byte_counts(size), **timing)
                    sock.settimeout(max(.001, self.deadline - time.monotonic()))
                    sent = time.perf_counter()
                    self._record("pd_region_send_start", layer=layer, start=start, end=end,
                                 payload_bytes=size, **self.byte_counts(size))
                    body = memoryview(self.slots[index].numpy())[:size]
                    self.partial_send_unknown = True
                    _send(sock, {"op": "region", "layer": layer, "start": start, "end": end}, body)
                    self.partial_send_unknown = False
                    del body
                    self.sent_bytes += size
                    self._record("pd_region_socket_done", layer=layer, start=start, end=end,
                                 payload_bytes=size, **self.byte_counts(size))
                    value = _response(sock)
                    last = layer + 1 == self.cache.shape[1] and end == self.sequence.num_prompt_tokens
                    if value != {"copies_done": last, "payload_bytes": size}:
                        raise ProtocolError("Invalid region completion ACK")
                    self.frames += 1
                    self.payload_bytes += size
                    self._record("pd_region_sent", layer=layer, start=start, end=end,
                                 payload_bytes=size, socket_ms=(time.perf_counter() - sent) * 1000,
                                 **self.byte_counts(size))
                    with self.condition:
                        self.free.append(index)
                        self.condition.notify_all()
                        if last:
                            self.done.set_result(({"payload_bytes": self.payload_bytes,
                                                   "frames": self.frames,
                                                   **self.byte_counts(self.payload_bytes)}, b""))
                            return
        except Exception as exc:
            self.abort(exc)
        finally:
            with self.condition:
                self.socket = None
                self.condition.notify_all()

    def close(self):
        if not self.done.done():
            self.abort(ConnectionError("Pipeline closed"))
        self.drain_copies()
        self.thread.join()
        self._record("pd_pipeline_sender_closed", **self.stats())
        self.pending.clear()
        self.slots.clear()

    def stats(self):
        with self.condition:
            return {**super().stats(), "busy_slots": 2 - len(self.free) if self.slots else 0,
                    "peak_slots": self.peak_slots, "socket_submitted_bytes": self.sent_bytes,
                    "partial_send_unknown": self.partial_send_unknown}


class LayerReceiver(_Staging):
    """The D owner restores into its own table; a slot lives until H2D completes."""

    def __init__(self, cache, frame_bytes, identity, timing_path=None, scales=None):
        super().__init__(cache, frame_bytes, identity, timing_path, scales)
        self.events = [None, None]
        self.next_slot = 0

    def _wait_slot(self, index):
        item = self.events[index]
        if item is not None:
            layer, start, end, begin, complete = item
            complete.synchronize()
            torch.cuda.current_stream(self.device).wait_event(complete)
            self.events[index] = None
            timing = {}
            if self.origin is not None:
                timing = {"h2d_start_gpu_ms": self.origin.elapsed_time(begin),
                          "h2d_end_gpu_ms": self.origin.elapsed_time(complete)}
            self._record("pd_h2d_done", layer=layer, start=start, end=end, **timing)

    def restore(self, table, layer, start, end, payload):
        index = self.next_slot
        self._wait_slot(index)
        size = len(payload)
        self.slots[index][:size].numpy()[:] = np.frombuffer(payload, dtype=np.uint8)
        data_size = (end - start) * self.data_token_bytes
        host = self.slots[index][:data_size].view(self.cache.dtype).reshape(
            2, end - start, self.cache.shape[-2], self.cache.shape[-1])
        host_scales = self.slots[index][data_size:size]
        if self.cuda:
            with torch.cuda.stream(self.stream):
                begin = None
                if self.timing_path:
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record(self.stream)
                self._copy(table, layer, start, end, host, host_scales)
                complete = torch.cuda.Event(enable_timing=bool(self.timing_path))
                complete.record(self.stream)
            self.events[index] = (layer, start, end, begin, complete)
        else:
            self._copy(table, layer, start, end, host, host_scales)
            self._record("pd_h2d_done", layer=layer, start=start, end=end)
        self.frames += 1
        self.payload_bytes += size
        self.next_slot = 1 - index

    def _copy(self, table, layer, start, end, host, host_scales):
        for pos, stop, physical in contiguous_regions(
                table, start, end, self.cache.shape[3], end - start):
            for kv in range(2):
                self.cache[kv, layer].flatten(0, 1)[physical:physical + stop - pos].copy_(
                    host[kv, pos - start:stop - start], non_blocking=self.cuda)
                if self.scales is not None:
                    width = self.scale_token_bytes // 2
                    offset = (kv * (end - start) + pos - start) * width
                    self.scales[kv, layer].flatten(0, 1)[physical:physical + stop - pos].view(
                        torch.uint8).reshape(-1).copy_(
                            host_scales[offset:offset + (stop - pos) * width], non_blocking=self.cuda)

    def finish(self):
        for index in range(2):
            self._wait_slot(index)

    def close(self):
        self.finish()
        self._record("pd_pipeline_receiver_closed", **self.stats())
        self.slots.clear()

    def stats(self):
        return {**super().stats(), "busy_slots": sum(event is not None for event in self.events)}
