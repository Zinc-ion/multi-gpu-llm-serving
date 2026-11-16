# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-slot adaptation of vLLM 0.15.0's consumed-before-reuse protocol.

Sources: distributed/device_communicators/shm_broadcast.py:439-563 and
v1/executor/multiproc_executor.py:376-405. Pipes supply synchronization
instead of the original raw flags/fences.
"""
import math
import mmap
import os
import pickle
import time
from multiprocessing import reduction
from multiprocessing.context import assert_spawning


class CommandSlot:
    def __init__(self, size, fd=None):
        if fd is None:
            fd = os.memfd_create("myvllm-tp-command", os.MFD_CLOEXEC)
            try:
                os.ftruncate(fd, size)
            except BaseException:
                os.close(fd)
                raise
        try:
            self.buf = mmap.mmap(fd, size)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self.size = size

    def __reduce__(self):
        # Creating DupFd here uses spawn's pass_fds rather than resource_sharer.
        assert_spawning(self)
        return _attach_command_slot, (reduction.DupFd(self._fd), self.size)

    def close(self):
        error = None
        if self.buf is not None:
            try:
                self.buf.close()
                self.buf = None
            except Exception as exc:
                error = exc
        if self._fd is not None:
            try:
                os.close(self._fd)
                self._fd = None
            except Exception as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"Command slot FD close failed: {exc!r}")
        if error is not None:
            raise error


def _attach_command_slot(handle, size):
    return CommandSlot(size, handle.detach())


class CommandSignal:
    def __init__(self, context):
        self._ready_recv, self._ready_send = context.Pipe(duplex=False)
        try:
            self._consumed_recv, self._consumed_send = context.Pipe(duplex=False)
        except BaseException:
            self._ready_recv.close()
            self._ready_send.close()
            raise
        self._pending = False

    def __getstate__(self):
        assert_spawning(self)
        return self._ready_recv, self._consumed_send

    def __setstate__(self, state):
        self._ready_recv, self._consumed_send = state
        self._ready_send = self._consumed_recv = None
        self._pending = False

    def close_worker_ends(self):
        self._close_ends(("_ready_recv", "_consumed_send"))

    def close(self):
        self._close_ends(("_ready_recv", "_ready_send", "_consumed_recv", "_consumed_send"))

    def _close_ends(self, names):
        errors = []
        for name in names:
            connection = getattr(self, name)
            if connection is not None:
                try:
                    connection.close()
                    setattr(self, name, None)
                except Exception as exc:
                    errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Command signal close failed", errors)

    def set(self):
        self._ready_send.send_bytes(b"")
        self._pending = True

    def wait(self, timeout):
        if not self._ready_recv.poll(timeout):
            return False
        self._ready_recv.recv_bytes()
        return True

    def clear(self):
        self._consumed_send.send_bytes(b"")

    def is_set(self):
        if self._pending and self._consumed_recv.poll():
            self._consumed_recv.recv_bytes()
            self._pending = False
        return self._pending


def validate_timeout(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def wait_for_consumption(events, timeout, processes=(), exiting=False):
    deadline = time.monotonic() + timeout
    while True:
        pending = []
        for i, event in enumerate(events):
            try:
                if event.is_set():
                    pending.append(i)
            except EOFError as exc:
                if i < len(processes):
                    process = processes[i]
                    process.join(0)
                    if process.exitcode is None:
                        raise RuntimeError(
                            f"TP rank {i + 1} (pid {process.pid}) closed its ACK pipe "
                            "before command ACK"
                        ) from exc
                    raise RuntimeError(
                        f"TP rank {i + 1} (pid {process.pid}) exited with code "
                        f"{process.exitcode} before command ACK"
                    ) from exc
                raise
        for i, process in enumerate(processes):
            code = process.exitcode
            if code is not None and (not exiting or code != 0 or events[i].is_set()):
                raise RuntimeError(
                    f"TP rank {i + 1} (pid {process.pid}) exited with code {code} before command ACK"
                )
        if not pending:
            return
        if time.monotonic() >= deadline:
            ranks = [i + 1 for i in pending]
            raise TimeoutError(f"TP command consumption ACK timed out for ranks {ranks}")
        os.sched_yield()


def write_command(shm, events, method_name, args, timeout, processes=()):
    # Serialize and reject oversized descriptors before touching the live slot.
    data = pickle.dumps((method_name, *args))
    n = len(data)
    if n > shm.size - 4:
        raise ValueError(f"TP command descriptor is {n} bytes; limit is {shm.size - 4}")
    wait_for_consumption(events, timeout, processes)
    shm.buf[4:n+4] = data
    shm.buf[:4] = n.to_bytes(4, 'little')
    for event in events:
        event.set()


def read_command(shm, event, timeout):
    if not event.wait(timeout):
        raise TimeoutError("TP command read timed out")
    n = int.from_bytes(shm.buf[:4], 'little') # read length
    method_name, *args = pickle.loads(bytes(shm.buf[4:n+4]))
    # ACK owned objects before executing a method which may enter a collective.
    event.clear()
    return method_name, args


def stop_workers(processes, exit_timeout=30, term_timeout=10, kill_timeout=10,
                 graceful=True):
    for name, value in (("exit_timeout", exit_timeout), ("term_timeout", term_timeout),
                        ("kill_timeout", kill_timeout)):
        validate_timeout(value, name)
    errors = []

    def wait_for_termination(timeout):
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.join(max(0, deadline - time.monotonic()))
            except Exception as exc:
                errors.append(exc)
        return [process for process in processes if process.is_alive()]

    active = wait_for_termination(exit_timeout) if graceful else [
        process for process in processes if process.is_alive()
    ]
    forced = bool(active)
    for action, timeout in (("terminate", term_timeout), ("kill", kill_timeout)):
        for process in active:
            try:
                getattr(process, action)()
            except Exception as exc:
                errors.append(exc)
        active = wait_for_termination(timeout)
        if not active:
            break
    if active:
        errors.append(RuntimeError(f"TP workers survived SIGKILL: {[p.pid for p in active]}"))
    if errors:
        raise ExceptionGroup("TP worker cleanup failed", errors)
    if graceful and forced:
        raise TimeoutError(f"TP workers required forced termination after {exit_timeout}s")
    if graceful:
        for i, process in enumerate(processes):
            if process.exitcode != 0:
                raise RuntimeError(
                    f"TP rank {i + 1} (pid {process.pid}) exited with code {process.exitcode} during shutdown"
                )
