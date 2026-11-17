"""CPU/Gloo coverage of TP parent death and owned IPC resources."""
import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import socket
import threading
import time
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import myvllm.engine.llm_engine as engine_module
import myvllm.utils.loader as loader
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.ipc import CommandSlot
from myvllm.engine.model_runner import ModelRunner
from test_instance_resources import cpu_backend, engine_config


_worker_process = engine_module.worker_process


def collective_probe(runner):
    value = torch.tensor(runner.rank + 1)
    dist.all_reduce(value)
    return value.item()


def cpu_collective_worker(config, rank, event):
    assert event._ready_send is None and event._consumed_recv is None
    with cpu_backend(config), patch.object(ModelRunner, "collective_probe", collective_probe, create=True):
        _worker_process(config, rank, event)


def fifo_worker(config, rank, event):
    real_loader = loader.load_weights_from_checkpoint
    with cpu_backend(config), patch.object(loader, "load_weights_from_checkpoint", real_loader):
        _worker_process(config, rank, event)


def fifo_owner(config):
    real_loader = loader.load_weights_from_checkpoint
    with cpu_backend(config), patch.object(loader, "load_weights_from_checkpoint", real_loader), \
            patch.object(engine_module, "worker_process", fifo_worker):
        LLMEngine(config)


def prearm_worker(config, rank, event):
    config["_prearm_ready"].send(os.getpid())
    config["_prearm_gate"].recv_bytes()
    with cpu_backend(config):
        _worker_process(config, rank, event)


def prearm_owner(config):
    with cpu_backend(config), patch.object(engine_module, "worker_process", prearm_worker):
        LLMEngine(config)


def identity(fd):
    stat = os.fstat(fd)
    return stat.st_dev, stat.st_ino


def fd_identities(pid):
    result = set()
    for entry in Path(f"/proc/{pid}/fd").iterdir():
        try:
            stat = entry.stat()
            result.add((stat.st_dev, stat.st_ino))
        except FileNotFoundError:
            continue
    return result


def map_contains(pid, descriptor):
    dev, inode = descriptor
    major, minor = os.major(dev), os.minor(dev)
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split()
        if int(fields[4]) == inode and fields[3] == f"{major:02x}:{minor:02x}":
            return True
    return False


def cpu_engine_owner(config, connection, start_in_thread=False):
    try:
        with cpu_backend(config), patch.object(engine_module, "worker_process", cpu_collective_worker), \
                patch.object(ModelRunner, "collective_probe", collective_probe, create=True):
            if start_in_thread:
                result = []
                errors = []

                def start():
                    try:
                        result.append(LLMEngine(config))
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=start)
                thread.start()
                thread.join()
                if errors:
                    raise errors[0]
                engine = result[0]
            else:
                engine = LLMEngine(config)
            event = engine.events[0]
            named = []
            signals = []
            if hasattr(engine._shm, "name"):
                named.append(Path("/dev/shm", engine._shm.name))
                named.extend(Path("/dev/shm", "sem." + sem._semlock.name.lstrip("/"))
                             for sem in (event._ready, event._consumed))
            else:
                signals = [identity(event._ready_send.fileno()), identity(event._consumed_recv.fileno())]
            connection.send((os.getpid(), engine.processes[0].pid, identity(engine._shm._fd),
                             [str(path) for path in named], engine._store.port, signals))
            while True:
                command = connection.recv()
                if command == "collective":
                    connection.send(engine.model_runner.call("collective_probe"))
                else:
                    engine.exit()
                    connection.send(engine.worker_exitcodes)
                    break
    finally:
        connection.close()


def start_owner(config, start_in_thread=False):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=cpu_engine_owner, args=(config, child, start_in_thread))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "TP engine did not initialize"
        return process, parent, parent.recv()
    except BaseException:
        workers = []
        if process.is_alive():
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
            for child_pid in children:
                try:
                    if b"spawn_main" in Path(f"/proc/{child_pid}/cmdline").read_bytes():
                        workers.append((int(child_pid), pid_start(int(child_pid))))
                except FileNotFoundError:
                    pass
            process.kill()
            process.join(5)
        for worker, start in workers:
            stop_owned_worker(worker, start)
        parent.close()
        process.close()
        raise


def pid_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def wait_reaped(pid, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reaped, status = os.waitpid(pid, os.WNOHANG)
        if reaped:
            return status
        time.sleep(.02)
    return None


def stop_owned_worker(pid, start):
    try:
        if pid_start(pid) == start:
            os.kill(pid, signal.SIGKILL)
        wait_reaped(pid, 5)
    except (FileNotFoundError, ProcessLookupError, ChildProcessError):
        pass


def test_slot_reduction_outside_spawn_does_not_create_sharer():
    from multiprocessing import resource_sharer

    listener = resource_sharer._resource_sharer._listener
    slot = CommandSlot(4096)
    try:
        with pytest.raises(RuntimeError, match="inheritance"):
            pickle.dumps(slot)
        assert resource_sharer._resource_sharer._listener is listener
    finally:
        slot.close()


@pytest.fixture
def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        yield
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_worker_exits_if_parent_dies_during_real_checkpoint_fifo(engine_config, tmp_path, subreaper):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    os.mkfifo(checkpoint / "model.safetensors")
    config = {**engine_config, "world_size": 2, "model_name_or_path": str(checkpoint),
              "tp_command_timeout_s": 5}
    owner = mp.get_context("spawn").Process(target=fifo_owner, args=(config,))
    worker = None
    worker_start = None
    reaped = False
    owner.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and owner.is_alive():
            children = Path(f"/proc/{owner.pid}/task/{owner.pid}/children").read_text().split()
            for child in children:
                try:
                    if b"spawn_main" in Path(f"/proc/{child}/cmdline").read_bytes():
                        worker = int(child)
                        worker_start = pid_start(worker)
                except FileNotFoundError:
                    pass
            if worker is not None:
                try:
                    owner_wait = Path(f"/proc/{owner.pid}/wchan").read_text().strip()
                    worker_wait = Path(f"/proc/{worker}/wchan").read_text().strip()
                    if owner_wait == worker_wait == "wait_for_partner":
                        break
                except FileNotFoundError:
                    pass
            time.sleep(.05)
        else:
            pytest.fail("both real checkpoint opens did not block on the FIFO")
        time.sleep(5.2)
        assert owner.is_alive()
        assert pid_start(worker) == worker_start
        assert Path(f"/proc/{worker}/wchan").read_text().strip() == "wait_for_partner"
        owner.kill()
        owner.join(5)
        assert owner.exitcode == -signal.SIGKILL
        reaped = wait_reaped(worker, 3) is not None
        assert reaped, "worker remained blocked in checkpoint open after rank 0 died"
    finally:
        if owner.is_alive():
            owner.kill()
            owner.join(5)
        if worker is not None and not reaped:
            stop_owned_worker(worker, worker_start)
        owner.close()


def test_parent_already_dead_before_sentinel_is_armed(engine_config, subreaper):
    context = mp.get_context("spawn")
    ready, child_ready = context.Pipe(duplex=False)
    child_gate, gate = context.Pipe(duplex=False)
    config = {**engine_config, "world_size": 2, "_prearm_ready": child_ready,
              "_prearm_gate": child_gate}
    owner = context.Process(target=prearm_owner, args=(config,))
    worker = None
    worker_start = None
    reaped = False
    owner.start()
    child_ready.close()
    child_gate.close()
    try:
        assert ready.poll(30), "worker did not reach the pre-arm gate"
        worker = ready.recv()
        worker_start = pid_start(worker)
        owner.kill()
        owner.join(5)
        assert owner.exitcode == -signal.SIGKILL
        gate.send_bytes(b"")
        reaped = wait_reaped(worker, 3) is not None
        assert reaped, "worker survived an already-dead parent at sentinel setup"
    finally:
        if owner.is_alive():
            owner.kill()
            owner.join(5)
        if worker is not None and not reaped:
            stop_owned_worker(worker, worker_start)
        gate.close()
        ready.close()
        owner.close()


def test_creator_thread_exit_keeps_live_parent_and_worker(engine_config, subreaper):
    config = {**engine_config, "world_size": 2}
    owner, connection, (_, worker, _, _, _, _) = start_owner(config, start_in_thread=True)
    worker_start = pid_start(worker)
    try:
        assert owner.is_alive()
        assert pid_start(worker) == worker_start
        connection.send("collective")
        assert connection.poll(5) and connection.recv() == 3
        connection.send("exit")
        assert connection.poll(10) and connection.recv() == [0]
        owner.join(10)
        assert owner.exitcode == 0
    finally:
        if owner.is_alive():
            owner.kill()
            owner.join(5)
        stop_owned_worker(worker, worker_start)
        connection.close()
        owner.close()


def test_abnormal_restarts_release_own_ipc_with_long_lived_parent(engine_config, subreaper):
    config = {**engine_config, "world_size": 2, "tp_command_timeout_s": 10}
    survivor, survivor_conn, survivor_info = start_owner(config)
    survivor_start = pid_start(survivor_info[1])
    try:
        survivor_conn.send("collective")
        assert survivor_conn.poll(5) and survivor_conn.recv() == 3
        for _ in range(2):
            owner, connection, (owner_pid, worker, slot, names, port, signals) = start_owner(config)
            worker_start = pid_start(worker)
            reaped = False
            try:
                assert slot in fd_identities(owner_pid) and slot in fd_identities(worker)
                assert map_contains(owner_pid, slot) and map_contains(worker, slot)
                connection.send("collective")
                assert connection.poll(5) and connection.recv() == 3
                owner.kill()
                owner.join(5)
                assert owner.exitcode == -signal.SIGKILL
                reaped = wait_reaped(worker, 3) is not None
                assert reaped, "worker survived its dead rank 0"
                assert not any(Path(name).exists() for name in names)
                from multiprocessing import resource_tracker

                tracker = resource_tracker._resource_tracker._pid
                assert tracker is not None
                observers = [os.getpid(), survivor_info[0], survivor_info[1], tracker]
                for pid in observers:
                    assert not {slot, *signals} & fd_identities(pid)
                    assert not map_contains(pid, slot)
                with socket.socket() as sock:
                    sock.settimeout(1)
                    assert sock.connect_ex(("127.0.0.1", port)) != 0
                survivor_conn.send("collective")
                assert survivor_conn.poll(5) and survivor_conn.recv() == 3
            finally:
                if owner.is_alive():
                    owner.kill()
                    owner.join(5)
                if not reaped:
                    stop_owned_worker(worker, worker_start)
                connection.close()
                owner.close()
    finally:
        try:
            if survivor.is_alive():
                survivor_conn.send("exit")
                assert survivor_conn.poll(10) and survivor_conn.recv() == [0]
                survivor.join(10)
        finally:
            if survivor.is_alive():
                survivor.kill()
                survivor.join(5)
            stop_owned_worker(survivor_info[1], survivor_start)
            survivor_conn.close()
            survivor.close()


def test_normal_exit_releases_fds_while_engine_and_signal_are_referenced(engine_config):
    config = {**engine_config, "world_size": 2}
    retained = []
    with cpu_backend(config), patch.object(engine_module, "worker_process", cpu_collective_worker), \
            patch.object(ModelRunner, "collective_probe", collective_probe, create=True):
        for _ in range(2):
            engine = LLMEngine(config)
            event = engine.events[0]
            process = engine.processes[0]
            retained.append((engine, event, process))
            assert event._ready_recv is None and event._consumed_send is None
            pid = process.pid
            slot = identity(engine._shm._fd)
            sentinel = identity(process.sentinel)
            signals = [identity(event._ready_send.fileno()), identity(event._consumed_recv.fileno())]
            try:
                assert engine.model_runner.call("collective_probe") == 3
                engine.exit()
            finally:
                if engine.model_runner is not None:
                    engine.exit()
            assert not engine.events
            assert not engine.processes
            assert engine.worker_pids == [pid]
            assert engine.worker_exitcodes == [0]
            assert slot not in fd_identities(os.getpid())
            assert sentinel not in fd_identities(os.getpid())
            assert not set(signals) & fd_identities(os.getpid())
            assert not map_contains(os.getpid(), slot)
            process.close()
            engine.exit()
