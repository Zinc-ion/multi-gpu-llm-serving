"""Real spawned consumers exercise production SHM dispatch and engine cleanup."""
import multiprocessing as mp
import pickle
import signal
import socket
import threading
import time
from contextlib import contextmanager, suppress
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import pytest

import myvllm.engine.llm_engine as engine_module
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.ipc import CommandSignal, CommandSlot, read_command, stop_workers, write_command
from pd_tp_ipc_helpers import CPUCommandRunner, cpu_command_worker, waiting_command_reader
from test_instance_resources import cpu_backend, cpu_worker, engine_config


@contextmanager
def command_engine(config, **settings):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    ready = ctx.Event()
    engine = None
    with patch.object(engine_module, "ModelRunner", CPUCommandRunner), \
            patch.object(engine_module, "worker_process", cpu_command_worker):
        try:
            engine = LLMEngine({**config, "world_size": 2, "_ready": ready,
                                "_records": child, **settings})
            assert ready.wait(15), "real command worker did not initialize"
            for event in settings.get("_ready_by_rank", {}).values():
                assert event.wait(15)
            yield engine, parent
        finally:
            if engine is not None and any(p.is_alive() for p in engine.processes):
                gate = settings.get("_read_gate")
                if gate is not None:
                    gate.set()
                for gate in settings.get("_read_gates", {}).values():
                    gate.set()
            if engine is not None:
                with suppress(RuntimeError, TimeoutError):
                    engine.exit()
            parent.close()
            child.close()


def receive(connection):
    assert connection.poll(5), "real consumer did not return a descriptor"
    return connection.recv()


def test_unread_command_cannot_be_overwritten(engine_config):
    gate = mp.get_context("spawn").Event()
    with command_engine(engine_config, _read_gate=gate) as (engine, connection):
        engine.model_runner.write_shm("record", (1,))
        release = threading.Timer(.15, gate.set)
        release.start()
        try:
            engine.model_runner.write_shm("record", (2,))
            assert receive(connection) == ("record", 1)
            assert receive(connection) == ("record", 2)
        finally:
            release.cancel()
            release.join()


def test_oversized_descriptor_leaves_pending_slot_and_events_unchanged(engine_config):
    gate = mp.get_context("spawn").Event()
    with command_engine(engine_config, _read_gate=gate) as (engine, connection):
        runner = engine.model_runner
        runner.write_shm("record", (7,))
        before = bytes(engine._shm.buf)
        with pytest.raises(ValueError):
            runner.write_shm("record", ("x" * 2**20,))
        assert bytes(engine._shm.buf) == before
        assert all(event.is_set() for event in engine.events)
        gate.set()
        assert receive(connection) == ("record", 7)


def test_rank_exit_before_consumption_propagates(engine_config):
    with command_engine(engine_config, _exit_before_read=True,
                        tp_command_timeout_s=.2) as (engine, _):
        engine.processes[0].join(5)
        assert engine.processes[0].exitcode == 17
        with pytest.raises(RuntimeError, match="rank 1.*17"):
            engine.model_runner.call("record", 1)


def test_nonreading_consumer_has_finite_ack_deadline(engine_config):
    gate = mp.get_context("spawn").Event()
    with command_engine(engine_config, _read_gate=gate,
                        tp_command_timeout_s=.2) as (engine, connection):
        engine.model_runner.write_shm("record", (1,))
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="ACK"):
            engine.model_runner.write_shm("record", (2,))
        assert time.monotonic() - start < 1
        gate.set()
        assert receive(connection) == ("record", 1)


def test_exit_reaps_a_worker_that_stops_during_shutdown(engine_config):
    gate, release = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with patch.object(LLMEngine, "_worker_exit_timeout_s", .1, create=True), \
                patch.object(LLMEngine, "_worker_term_timeout_s", .1, create=True), \
                patch.object(LLMEngine, "_worker_kill_timeout_s", .1, create=True):
            with command_engine(engine_config, _exit_gate=gate) as (engine, _):
                slot = engine._shm
                start = time.monotonic()
                with pytest.raises(TimeoutError, match="forced termination"):
                    engine.exit()
                elapsed = time.monotonic() - start
                assert not engine.processes
                assert slot._fd is None and slot.buf is None
                assert elapsed < .6, f"exit waited {elapsed:.3f}s for a stalled worker"
    finally:
        gate.close()
        release.close()


@pytest.mark.parametrize("ignore_term", [False, True], ids=["TERM", "KILL"])
def test_forced_shutdown_reaps_owned_workers_and_can_restart(engine_config, ignore_term):
    gate, release = mp.get_context("spawn").Pipe(duplex=False)
    try:
        with patch.object(LLMEngine, "_worker_exit_timeout_s", .1), \
                patch.object(LLMEngine, "_worker_term_timeout_s", .1), \
                patch.object(LLMEngine, "_worker_kill_timeout_s", 2):
            with command_engine(engine_config, _exit_gate=gate,
                                _ignore_term=ignore_term) as (engine, _):
                slot = engine._shm
                start = time.monotonic()
                with pytest.raises(TimeoutError, match="forced termination"):
                    engine.exit()
                assert time.monotonic() - start < 3
                assert engine.worker_exitcodes == [-(signal.SIGKILL if ignore_term else signal.SIGTERM)]
                assert not engine.processes
                assert slot._fd is None and slot.buf is None
                engine.exit()
    finally:
        gate.close()
        release.close()
    with command_engine(engine_config) as (engine, connection):
        assert engine.model_runner.call("record", 19) == 19
        assert receive(connection) == ("record", 19)


def test_call_waits_for_consumption_before_local_execution(engine_config):
    gate = mp.get_context("spawn").Event()
    with command_engine(engine_config, _read_gate=gate,
                        tp_command_timeout_s=.1) as (engine, connection):
        runner = engine.model_runner
        with pytest.raises(TimeoutError, match="ACK"):
            runner.call("record", 1)
        assert runner.records == []
        gate.set()
        assert receive(connection) == ("record", 1)


def test_consumption_ack_precedes_real_gloo_collective(engine_config):
    with command_engine(engine_config, _gloo=True,
                        tp_command_timeout_s=2) as (engine, connection):
        assert engine.model_runner.call("collective") == 3
        assert receive(connection) == ("collective", 3)
        assert engine.model_runner.call("record", 2) == 2
        assert receive(connection) == ("record", 2)


def test_idle_consumer_survives_read_deadlines(engine_config):
    with command_engine(engine_config, tp_command_timeout_s=.1) as (engine, connection):
        time.sleep(.35)
        assert engine.processes[0].is_alive()
        assert engine.model_runner.call("record", 3) == 3
        assert receive(connection) == ("record", 3)


def test_every_rank_must_ack_before_slot_reuse(engine_config):
    ctx = mp.get_context("spawn")
    gates = {rank: ctx.Event() for rank in (1, 2)}
    ready = {rank: ctx.Event() for rank in (1, 2)}
    gates[1].set()
    with command_engine(engine_config, world_size=3, _read_gates=gates,
                        _ready_by_rank=ready, _rank_records=True,
                        tp_command_timeout_s=.2) as (engine, connection):
        runner = engine.model_runner
        runner.write_shm("record", (1,))
        assert receive(connection) == (1, "record", 1)
        with pytest.raises(TimeoutError, match="ranks \\[2\\]"):
            runner.write_shm("record", (2,))
        gates[2].set()
        assert receive(connection) == (2, "record", 1)
        runner.write_shm("record", (3,))
        assert {receive(connection), receive(connection)} == {
            (1, "record", 3), (2, "record", 3),
        }


def test_exact_capacity_and_rapid_commands_retain_order(engine_config):
    with command_engine(engine_config) as (engine, connection):
        capacity = 2**20 - 4
        sample = b"x" * capacity
        overhead = len(pickle.dumps(("record", sample))) - len(sample)
        value = sample[:capacity - overhead]
        assert len(pickle.dumps(("record", value))) == capacity
        assert engine._shm.size == 2**20
        engine.model_runner.write_shm("record", (value,))
        assert receive(connection) == ("record", value)
        for i in range(40):
            engine.model_runner.write_shm("record", (i,))
        assert [receive(connection) for _ in range(40)] == [
            ("record", i) for i in range(40)
        ]


def test_worker_killed_with_unread_descriptor_fails_and_releases(engine_config):
    gate = mp.get_context("spawn").Event()
    with command_engine(engine_config, _read_gate=gate,
                        tp_command_timeout_s=.2) as (engine, _):
        slot = engine._shm
        engine.model_runner.write_shm("record", (1,))
        engine.processes[0].terminate()
        engine.processes[0].join(5)
        with pytest.raises(RuntimeError, match="rank 1"):
            engine.model_runner.call("record", 2)
        with pytest.raises(RuntimeError, match="rank 1"):
            engine.exit()
        assert engine.worker_exitcodes == [-signal.SIGTERM]
        assert not engine.processes
        assert slot._fd is None and slot.buf is None


def test_worker_nonzero_exit_after_ack_is_reported_and_released(engine_config):
    ctx = mp.get_context("spawn")
    gate, release = ctx.Pipe(duplex=False)
    ready = {rank: ctx.Event() for rank in (1, 2)}
    try:
        with command_engine(engine_config, world_size=3, _ready_by_rank=ready,
                            _exit_fail_rank=2, _exit_fail_gate=gate,
                            _exit_fail_release=release) as (engine, _):
            slot, port = engine._shm, engine._store.port
            try:
                with pytest.raises(RuntimeError, match="rank 2.*code 23"):
                    engine.exit()
            finally:
                assert engine.worker_exitcodes == [0, 23]
                assert not engine.processes
                assert slot._fd is None and slot.buf is None
                with socket.socket() as sock:
                    sock.settimeout(1)
                    assert sock.connect_ex(("127.0.0.1", port)) != 0
            engine.exit()
    finally:
        gate.close()
        release.close()


def test_runner_io_error_does_not_skip_other_workers_or_shm(engine_config, tmp_path):
    sentinel = SharedMemory(create=True, size=16)
    sentinel.buf[:4] = b"keep"
    try:
        with command_engine(engine_config, world_size=3,
                            _exit_io_error=str(tmp_path)) as (engine, _):
            slot, port = engine._shm, engine._store.port
            with pytest.raises(IsADirectoryError):
                engine.exit()
            assert not engine.processes
            assert slot._fd is None and slot.buf is None
            with socket.socket() as sock:
                sock.settimeout(1)
                assert sock.connect_ex(("127.0.0.1", port)) != 0
            attached = SharedMemory(name=sentinel.name)
            try:
                assert bytes(attached.buf[:4]) == b"keep"
            finally:
                attached.close()
            engine.exit()
    finally:
        sentinel.close()
        sentinel.unlink()


def test_real_multirank_initialization_failure_releases_and_restarts(engine_config):
    slots = []
    children = {process.pid for process in mp.active_children()}

    def observe_slot(*args, **kwargs):
        slot = CommandSlot(*args, **kwargs)
        slots.append(slot)
        return slot

    settings = {**engine_config, "world_size": 2, "model_type": "unsupported",
                "tp_command_timeout_s": 10}
    with cpu_backend(settings), patch.object(engine_module, "worker_process", cpu_worker), \
            patch.object(engine_module, "CommandSlot", observe_slot):
        with pytest.raises(Exception, match="Unsupported model:"):
            LLMEngine(settings)
    assert slots
    assert {process.pid for process in mp.active_children()} <= children
    assert all(slot._fd is None and slot.buf is None for slot in slots)
    with command_engine(engine_config) as (engine, connection):
        assert engine.model_runner.call("record", 9) == 9
        assert receive(connection) == ("record", 9)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "30"])
def test_command_deadline_is_validated_before_engine_resources(value):
    with pytest.raises(ValueError, match="tp_command_timeout_s"):
        LLMEngine({"tp_command_timeout_s": value})


@pytest.mark.parametrize("name", ["exit_timeout", "term_timeout", "kill_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_worker_cleanup_deadlines_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=name):
        stop_workers([], **{name: value})


def test_empty_production_channel_has_finite_read_wait():
    event = CommandSignal(mp.get_context("spawn"))
    shm = CommandSlot(2**20)
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="read timed out"):
            read_command(shm, event, .05)
        assert time.monotonic() - start < .5
        assert not event.is_set()
    finally:
        shm.close()
        event.close()


def test_single_rank_exit_uses_runner_call_contract(engine_config):
    class CPUCallRunner:
        def __init__(self, config, rank, event):
            self.calls = []

        def call(self, method, *args):
            self.calls.append(method)

    with patch.object(engine_module, "ModelRunner", CPUCallRunner):
        engine = LLMEngine(engine_config)
        runner = engine.model_runner
        engine.exit()
        assert runner.calls == ["exit"]
        engine.exit()


def test_publication_remains_bounded_after_waiting_consumer_is_killed():
    ctx = mp.get_context("spawn")
    event = CommandSignal(ctx)
    shm = CommandSlot(2**20)
    ready, child_ready = ctx.Pipe(duplex=False)
    consumer = ctx.Process(target=waiting_command_reader, args=(shm, event, child_ready))
    try:
        consumer.start()
        event.close_worker_ends()
        assert ready.poll(10) and ready.recv()
        time.sleep(.2)
        consumer.terminate()
        consumer.join(5)
        assert consumer.exitcode == -signal.SIGTERM
        start = time.monotonic()
        with pytest.raises(BrokenPipeError):
            write_command(shm, [event], "record", (1,), .1)
        assert time.monotonic() - start < 1
    finally:
        if consumer.is_alive():
            consumer.kill()
            consumer.join(5)
        ready.close()
        child_ready.close()
        shm.close()
        event.close()
        consumer.close()
