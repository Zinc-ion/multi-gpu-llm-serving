"""Strict checkpoint failures must release startup resources before retry."""
import socket
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from tp_engine_helpers import create_engine_checkpoint
from tp_helpers import checkpoint_names


@contextmanager
def cpu_startup():
    init_process_group = dist.init_process_group
    previous_dtype = torch.get_default_dtype()

    def init_gloo(backend, *args, **kwargs):
        assert backend == "nccl"
        return init_process_group("gloo", *args, **kwargs)

    # Keep the real model and strict loader. Warmup/cache allocation and device
    # transport require CUDA and are outside the startup-failure boundary.
    with ExitStack() as stack:
        stack.enter_context(patch.object(dist, "init_process_group", init_gloo))
        stack.enter_context(patch.object(torch.cuda, "set_device"))
        stack.enter_context(patch.object(torch.cuda, "synchronize"))
        stack.enter_context(patch.object(torch.nn.Module, "cuda", lambda self, device=None: self))
        stack.enter_context(patch.object(torch, "set_default_device"))
        stack.enter_context(patch.object(ModelRunner, "warmup_model"))
        stack.enter_context(patch.object(ModelRunner, "allocate_kv_cache"))
        try:
            yield
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            torch.set_default_dtype(previous_dtype)


def accepts_connection(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.parametrize("standalone", [False, True], ids=["engine", "standalone"])
def test_missing_checkpoint_parameter_releases_resources_and_restarts(tmp_path, standalone):
    complete, incomplete = tmp_path / "complete", tmp_path / "incomplete"
    _, config = create_engine_checkpoint(complete)
    tensors = {}
    for shard in complete.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            tensors.update({name: weights.get_tensor(name) for name in weights.keys()})
    del tensors["model.norm.weight"]
    incomplete.mkdir()
    save_file(tensors, str(incomplete / "model.safetensors"))
    ports, tcp_store = [], dist.TCPStore

    def observe_store(*args, **kwargs):
        store = tcp_store(*args, **kwargs)
        if kwargs.get("is_master"):
            ports.append(store.port)
        return store

    def start(settings):
        return ModelRunner(settings, rank=0, event=[]) if standalone else LLMEngine(settings)

    with cpu_startup(), patch.object(dist, "TCPStore", observe_store):
        with pytest.raises(ValueError, match="Unloaded model parameters:.*model.norm.weight"):
            start({**config, "model_name_or_path": str(incomplete)})
        assert not dist.is_initialized(), "failed strict loading left the process group initialized"
        assert ports and all(not accepts_connection(port) for port in ports)
        owner = start(config)
        try:
            runner = owner if standalone else owner.model_runner
            assert set(dict(runner.model.named_parameters())) <= runner.loaded_params
            assert checkpoint_names(complete) <= runner.loaded_params
            assert dist.is_initialized() and accepts_connection(ports[-1])
        finally:
            owner.exit()
        owner.exit()
        assert not dist.is_initialized()
        assert all(not accepts_connection(port) for port in ports)
