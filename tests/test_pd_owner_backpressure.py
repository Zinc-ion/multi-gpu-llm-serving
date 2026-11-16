"""Real TCP cleanup and owner admission while a CPU step is slow."""

import asyncio
from concurrent.futures import wait
import errno
import json
import socket
import sys
import threading
import time

import pytest
import torch.distributed as dist

from myvllm.engine.llm_engine import LLMEngine
from myvllm.pd.client import PDClient, PDRemoteError
from myvllm.pd.worker import PDWorker, _Owner
from myvllm.sampling_parameters import SamplingParams
from pd_network_helpers import SlowCPUExecution, cpu_engine
from test_instance_resources import cpu_backend, engine_config


class ObservedSlowCPUExecution(SlowCPUExecution):
    def __init__(self, config, started, delay):
        super().__init__(config, delay=delay)
        self.started = started
        self.second_started = threading.Event()
        self.step_count = 0

    def call(self, method, sequences, is_prefill):
        self.step_count += 1
        self.started.set()
        if self.step_count == 2:
            self.second_started.set()
        return super().call(method, sequences, is_prefill)


def worker_with_slow_step(max_connections, timing_path=None, delay=3):
    started = threading.Event()
    exited = threading.Event()
    status_calls = []

    def engine_factory(config):
        engine = cpu_engine(config)
        engine.model_runner = ObservedSlowCPUExecution(engine.config, started, delay)
        exit_engine = engine.exit

        def profile(frame, event, arg):
            if event == "call" and frame.f_code in (_Owner._stats.__code__, _Owner._info.__code__):
                status_calls.append(frame.f_code.co_name)

        def exit_observed():
            try:
                exit_engine()
            finally:
                sys.setprofile(None)
                exited.set()

        engine.exit = exit_observed
        sys.setprofile(profile)
        return engine

    config = {"world_size": 1, "chunked": True}
    if timing_path is not None:
        config["timing_path"] = str(timing_path)
    worker = PDWorker("decode", config, max_connections=max_connections,
                      max_transfer_bytes=1_000_000, request_timeout=15,
                      engine_factory=engine_factory)
    return worker, started, exited, status_calls


def queued(worker):
    with worker.owner.commands.mutex:
        return list(worker.owner.commands.queue)


async def wait_until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.005)


async def fill_with_timed_out_status(worker, endpoint, count):
    for index in range(count):
        method = PDClient.stats if index % 2 == 0 else PDClient.info
        with pytest.raises(TimeoutError):
            await method(endpoint, timeout=.04)
        await wait_until(lambda: sum(item[1] in ("info", "stats") and item[-1].cancelled()
                                    for item in queued(worker)) == index + 1)


def assert_empty(stats):
    for name in ("open_handles", "used_blocks", "reserved_blocks", "resident_requests",
                 "reserved_transfer_bytes"):
        assert stats[name] == 0, (name, stats)


def test_status_flood_keeps_cleanup_capacity_and_recovers():
    async def run():
        worker, started, exited, status_calls = worker_with_slow_step(1)
        endpoint = await worker.start("127.0.0.1", 0)
        local = survivor = None
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            local = await PDClient.open_local(endpoint, "slow", [1, 2, 3], params, lifetime=12)
            assert await asyncio.to_thread(started.wait, 2)
            slots = worker.max_socket_connections + 1
            await fill_with_timed_out_status(worker, endpoint, slots)
            assert worker.owner.commands.maxsize == 2 * worker.max_socket_connections + 2
            assert len(queued(worker)) == slots
            with pytest.raises(PDRemoteError) as error:
                await PDClient.stats(endpoint, timeout=1)
            assert error.value.code == "CapacityError"
            assert "owner command capacity" in str(error.value)
            assert len(queued(worker)) == slots

            await local.close()
            local = None
            await wait_until(lambda: any(item[1] == "cleanup" for item in queued(worker)))
            assert worker.sessions == 1
            with pytest.raises(PDRemoteError, match="connection capacity"):
                await PDClient.open_d(endpoint, lifetime=5)
            await wait_until(lambda: worker.sessions == 0, timeout=6)
            assert status_calls == []
            assert_empty(await PDClient.stats(endpoint))
            assert status_calls == ["_stats"]

            survivor = await PDClient.open_local(
                endpoint, "after-pressure", [1], SamplingParams(temperature=0, max_tokens=1,
                                                                ignore_eos=True), lifetime=10)
            event = await survivor.next_token()
            assert event["finished"] and event["token"] == 2
            await survivor.close()
            survivor = None
            await wait_until(lambda: worker.sessions == 0, timeout=6)
            assert_empty(await PDClient.stats(endpoint))
        finally:
            for connection in (local, survivor):
                if connection is not None:
                    await connection.close()
            await worker.close()
        assert exited.is_set()
        assert not worker.owner.thread.is_alive()
        await worker.close()

    asyncio.run(run())


def test_shutdown_drains_multiple_sessions_and_queued_disconnected_open(tmp_path):
    async def run():
        timing_path = tmp_path / "timing.jsonl"
        worker, started, exited, status_calls = worker_with_slow_step(3, timing_path)
        endpoint = await worker.start("127.0.0.1", 0)
        idle = active = None
        pending = closing = None
        try:
            idle = await PDClient.open_d(endpoint, lifetime=12)
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            active = await PDClient.open_local(endpoint, "active", [1, 2, 3], params, lifetime=12)
            assert await asyncio.to_thread(started.wait, 2)
            pending = asyncio.create_task(PDClient.open_local(
                endpoint, "queued-disconnect", [4, 5], params, lifetime=12))
            await wait_until(lambda: any(item[1] == "open_local" and
                                         item[2]["request_id"] == "queued-disconnect"
                                         for item in queued(worker)))
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await wait_until(lambda: any(item[1] == "cleanup" and
                                         item[0] is next((op[0] for op in queued(worker)
                                                          if op[1] == "open_local"), None)
                                         for item in queued(worker)))
            assert worker.sessions == 3

            slots = worker.max_socket_connections + 1
            await fill_with_timed_out_status(worker, endpoint, slots - 1)
            with pytest.raises(PDRemoteError) as error:
                await PDClient.info(endpoint, timeout=1)
            assert error.value.code == "CapacityError"
            closing = asyncio.gather(worker.close(), worker.close())
            await wait_until(lambda: sum(item[1] == "cleanup" for item in queued(worker)) == 3)
            assert worker.sessions == 3 and not closing.done()
            assert not exited.is_set()
            await asyncio.wait_for(closing, 8)
            assert status_calls == []
            assert worker.sessions == 0 and not worker.connections
            assert not worker.owner.thread.is_alive() and exited.is_set()
            assert worker.owner.commands.empty()
            assert_empty(worker.owner.core.stats())
            records = [json.loads(line) for line in timing_path.read_text().splitlines()]
            events = [row["event"] for row in records if row["request_id"] == "queued-disconnect"]
            assert events == ["pd_admitted", "pd_cancelled", "pd_released"]
            await worker.close()
        finally:
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            for connection in (idle, active):
                if connection is not None:
                    await connection.close()
            await worker.close()

    asyncio.run(run())


def test_pressure_shutdown_releases_real_cpu_engine_resources(engine_config):
    config = {**engine_config, "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
              "dtype": "float32", "enable_chunked_prefill": True}
    started = threading.Event()

    class CPUResourceRunner:
        def __init__(self, runner, engine_config):
            self.runner = runner
            self.slow = ObservedSlowCPUExecution(engine_config, started, delay=3)
            self.kv_cache = self.slow.kv_cache
            self.timings = runner.timings

        def call(self, method, *args):
            if method == "run":
                return self.slow.call(method, *args)
            return self.runner.call(method, *args)

    def engine_factory(config):
        engine = LLMEngine(config)
        engine.model_runner = CPUResourceRunner(engine.model_runner, engine.config)
        return engine

    async def run():
        worker = PDWorker("decode", config, max_connections=1, max_transfer_bytes=1_000_000,
                          request_timeout=12, engine_factory=engine_factory)
        endpoint = await worker.start("127.0.0.1", 0)
        local = None
        try:
            engine = worker.owner.engine
            runner = engine.model_runner.runner
            port = engine._store.port
            assert dist.is_initialized()
            local = await PDClient.open_local(
                endpoint, "real-engine", [0], SamplingParams(temperature=0, max_tokens=4,
                                                              ignore_eos=True), lifetime=10)
            assert await asyncio.to_thread(started.wait, 2)
            await fill_with_timed_out_status(worker, endpoint, worker.max_socket_connections + 1)
            with pytest.raises(PDRemoteError) as error:
                await PDClient.stats(endpoint, timeout=1)
            assert error.value.code == "CapacityError"
            closing = asyncio.create_task(worker.close())
            await wait_until(lambda: any(item[1] == "cleanup" for item in queued(worker)))
            assert not closing.done() and worker.sessions == 1
            await asyncio.wait_for(closing, 8)
            assert worker.sessions == 0 and not worker.connections
            assert_empty(worker.owner.core.stats())
            assert runner._exited and engine.model_runner is None and engine._store is None
            assert not dist.is_initialized() and not worker.owner.thread.is_alive()
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("::", port))
            await worker.close()
        finally:
            if local is not None:
                await local.close()
            await worker.close()

    with cpu_backend(config):
        asyncio.run(run())


def test_close_ack_allows_immediate_reuse_while_owner_is_busy(tmp_path):
    async def run():
        timing_path = tmp_path / "timing.jsonl"
        worker, started, exited, _ = worker_with_slow_step(2, timing_path, delay=1)
        endpoint = await worker.start("127.0.0.1", 0)
        idle = active = replacement = None
        try:
            idle = await PDClient.open_d(endpoint, lifetime=12)
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            active = await PDClient.open_local(endpoint, "active", [1, 2, 3], params, lifetime=12)
            assert await asyncio.to_thread(started.wait, 2)
            closing = asyncio.create_task(idle.close())
            await wait_until(lambda: any(item[1] == "close" for item in queued(worker)))
            await asyncio.wait_for(closing, 4)
            close_returned = time.monotonic_ns()
            replacement = await PDClient.open_d(endpoint, lifetime=8)
            assert worker.owner.engine.model_runner.second_started.is_set()
            records = [json.loads(line) for line in timing_path.read_text().splitlines()]
            released = next(row for row in records if row["event"] == "pd_released" and
                            row["session_id"] == idle.session_id)
            assert released["monotonic_ns"] <= close_returned
            assert replacement.session_id != idle.session_id
            await replacement.close()
            replacement = None
            await active.close()
            active = None
            assert_empty(await PDClient.stats(endpoint))
        finally:
            for connection in (idle, active, replacement):
                if connection is not None:
                    await connection.close()
            await worker.close()
        assert exited.is_set() and not worker.owner.thread.is_alive()

    asyncio.run(run())


def test_owner_failure_settles_queued_status_and_reclaims_capacity(tmp_path):
    async def run():
        timing_path = tmp_path / "timing.jsonl"
        worker, started, exited, status_calls = worker_with_slow_step(2, timing_path)
        endpoint = await worker.start("127.0.0.1", 0)
        local = pending_status = None
        try:
            params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
            local = await PDClient.open_local(endpoint, "failure", [1, 2, 3], params, lifetime=12)
            assert await asyncio.to_thread(started.wait, 2)
            pending_status = asyncio.create_task(PDClient.stats(endpoint, timeout=10))
            await wait_until(lambda: any(item[1] == "stats" for item in queued(worker)))
            slots = worker.max_socket_connections + 1
            await fill_with_timed_out_status(worker, endpoint, slots - 1)
            futures = [item[-1] for item in queued(worker) if item[1] in ("info", "stats")]
            assert len(futures) == slots
            assert sum(future.cancelled() for future in futures) == slots - 1

            fault = tmp_path / "fault"
            fault.symlink_to("/dev/full")
            fault.replace(timing_path)
            await asyncio.wait_for(worker.failed.wait(), 8)
            result = await asyncio.gather(pending_status, return_exceptions=True)
            assert isinstance(result[0], (ConnectionError, OSError, asyncio.IncompleteReadError))
            with pytest.raises(OSError) as error:
                await worker.close()
            assert error.value.errno == errno.ENOSPC
            assert status_calls == [] and exited.is_set()
            assert worker.sessions == 0 and not worker.connections
            assert not worker.owner.thread.is_alive() and worker.owner.commands.empty()
            assert worker.owner._command_slots._value == slots
            assert all(future.done() for future in futures)
            done, unfinished = wait(futures, timeout=0)
            assert len(done) == slots and not unfinished
            assert_empty(worker.owner.core.stats())
        finally:
            if pending_status is not None:
                await asyncio.gather(pending_status, return_exceptions=True)
            if local is not None:
                await local.close()
            try:
                await worker.close()
            except OSError as exc:
                assert exc.errno == errno.ENOSPC

    asyncio.run(run())
