import socket
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from test_instance_resources import cpu_backend, engine_config


class CPUEvent:
    ready = False

    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        return 1.0


def accepts_connection(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def store_ports():
    ports = []
    tcp_store = dist.TCPStore

    def observe_store(*args, **kwargs):
        store = tcp_store(*args, **kwargs)
        if kwargs.get("is_master"):
            ports.append(store.port)
        return store

    with patch.object(dist, "TCPStore", observe_store):
        yield ports


@pytest.mark.parametrize("standalone", [False, True], ids=["engine", "standalone"])
def test_timing_io_failure_closes_and_allows_restart(engine_config, tmp_path, standalone):
    timing_dir = tmp_path / "timings"
    timing_dir.mkdir()
    config = {**engine_config, "enforce_eager": False, "timing_path": str(timing_dir)}
    CPUEvent.ready = False
    with cpu_backend(config), patch.object(torch.cuda, "Event", CPUEvent):
        owner = ModelRunner(config, rank=0, event=[]) if standalone else LLMEngine(config)
        runner = owner if standalone else owner.model_runner
        port = runner._standalone_store.port if standalone else owner._store.port
        try:
            timings = runner.timings
            timings.begin()
            for name in ("h2d", "sample_h2d"):
                timings.start_gpu(name)
                timings.end_gpu(name)
            timings.write(str(timing_dir), False, [], 1, {})
            CPUEvent.ready = True
            with pytest.raises(IsADirectoryError) as error:
                owner.exit()
            assert error.value.filename == str(timing_dir)
            owner.exit()
            runner.exit()
            assert not dist.is_initialized()
            assert not accepts_connection(port)

            restarted = ModelRunner(config, rank=0, event=[]) if standalone else LLMEngine(config)
            restarted.exit()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


@pytest.mark.parametrize(
    ("model_type", "rank", "error_type", "message"),
    [
        ("unsupported", 0, Exception, "Unsupported model:"),
        ("Qwen3-0.6B", -1, AssertionError, "rank must be non-negative"),
    ],
)
def test_failed_standalone_runner_releases_resources_and_allows_restart(
    engine_config, store_ports, model_type, rank, error_type, message,
):
    with cpu_backend(engine_config):
        try:
            with pytest.raises(error_type, match=message) as error:
                ModelRunner({**engine_config, "model_type": model_type}, rank=rank, event=[])
            assert error.value.__traceback__ is not None
            assert not dist.is_initialized()
            assert not accepts_connection(store_ports[0])

            runner = ModelRunner(engine_config, rank=0, event=[])
            runner.exit()
            runner.exit()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def test_failed_standalone_runner_does_not_destroy_existing_group(engine_config, store_ports):
    with cpu_backend(engine_config):
        runner = ModelRunner(engine_config, rank=0, event=[])
        try:
            with pytest.raises(ValueError, match="process group") as error:
                ModelRunner(engine_config, rank=0, event=[])
            assert error.value.__traceback__ is not None
            assert dist.is_initialized()
            assert accepts_connection(store_ports[0])
            assert all(not accepts_connection(port) for port in store_ports[1:])
            value = torch.tensor(3)
            dist.all_reduce(value)
            assert value.item() == 3
        finally:
            runner.exit()
