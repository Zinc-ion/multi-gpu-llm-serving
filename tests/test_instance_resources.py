import multiprocessing as mp
import socket
import traceback
from contextlib import ExitStack, contextmanager
from multiprocessing.shared_memory import SharedMemory
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import myvllm.engine.llm_engine as engine_module
import myvllm.engine.model_runner as runner_module
from myvllm.engine.llm_engine import LLMEngine


_worker_process = engine_module.worker_process


class CPUModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def cuda(self, device=None):
        return self


def capture_cpu_graph(runner):
    runner.graphs = {}
    runner.graph_vars = {}


@contextmanager
def cpu_backend(config):
    init_process_group = dist.init_process_group

    def init_gloo(backend, *args, **kwargs):
        assert backend == "nccl"
        return init_process_group("gloo", *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(dist, "init_process_group", init_gloo))
        stack.enter_context(patch.object(torch.cuda, "set_device"))
        stack.enter_context(patch.object(torch.cuda, "synchronize"))
        stack.enter_context(patch.object(torch, "set_default_device"))
        stack.enter_context(patch.object(runner_module, "Qwen3ForCausalLM", CPUModel))
        stack.enter_context(patch.object(runner_module.ModelRunner, "warmup_model"))
        stack.enter_context(patch.object(runner_module.ModelRunner, "allocate_kv_cache"))
        stack.enter_context(patch.object(runner_module.ModelRunner, "capture_cudagraph", capture_cpu_graph))
        stack.enter_context(patch("myvllm.utils.loader.load_weights_from_checkpoint", return_value=set()))
        yield


def cpu_worker(config, rank, event):
    with cpu_backend(config):
        _worker_process(config, rank, event)


def engine_process(config, connection):
    try:
        with cpu_backend(config), patch.object(engine_module, "worker_process", cpu_worker):
            engine = LLMEngine(config)
            shm_name = engine.model_runner.shm.name if config["world_size"] > 1 else None
            assert "_instance_resources" not in config
            store = getattr(engine, "_store", None)
            connection.send(("ready", store.port if store else None, shm_name))
            while (command := connection.recv()) == "ping":
                connection.send(("pong", dist.is_initialized()))
            engine.exit()
            worker_exitcodes = [process.exitcode for process in engine.processes]
            if command == "exit":
                engine.exit()
            connection.send(("closed", worker_exitcodes))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def runner_process(config, connection):
    try:
        with cpu_backend(config):
            runner = runner_module.ModelRunner(config, rank=0, event=[])
            connection.send(("ready", runner._standalone_store.port))
            connection.recv()
            runner.exit()
            runner.exit()
            connection.send(("closed", dist.is_initialized()))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def failed_start_process(config, stage, connection):
    try:
        with cpu_backend(config):
            invalid = {"max_num_sequences": 0} if stage == "scheduler" else {"model_type": "unsupported"}
            try:
                LLMEngine({**config, **invalid})
            except Exception as exc:
                expected = "max_num_sequences must be positive" if stage == "scheduler" else "Unsupported model:"
                assert expected in str(exc)
            else:
                raise AssertionError("invalid configuration was accepted")
            engine = LLMEngine(config)
            engine.exit()
            connection.send(("recovered",))
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def gpu_engine_process(config, connection):
    from myvllm.sampling_parameters import SamplingParams

    engine = None
    try:
        engine = LLMEngine(config)
        missing = set(dict(engine.model_runner.model.named_parameters())) - engine.model_runner.loaded_params
        assert not missing, missing
        connection.send(("ready", engine._store.port, torch.cuda.current_device()))
        while True:
            command = connection.recv()
            if command == "generate":
                seq = engine.add_prompt("hello", SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
                while not seq.is_finished:
                    engine.step()
                assert not engine.scheduler.block_manager.used_block_ids
                connection.send(("tokens", seq.completion_token_ids))
            elif command == "exit":
                engine.exit()
                engine.exit()
                connection.send(("closed",))
                break
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if engine is not None:
            engine.exit()
        connection.close()


@pytest.fixture
def engine_config(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(tmp_path)
    return {
        "model_name_or_path": str(tmp_path), "model_type": "Qwen3-0.6B",
        "world_size": 1, "block_size": 4, "enforce_eager": True,
        "vocab_size": 2, "hidden_size": 4, "num_heads": 1, "head_dim": 4,
        "scale": 0.5, "num_kv_heads": 1, "rms_norm_epsilon": 1e-6,
        "qkv_bias": False, "base": 10000, "max_position": 32,
        "intermediate_size": 8, "ffn_bias": False, "num_layers": 1,
        "tie_word_embeddings": False, "max_num_sequences": 1,
        "max_num_batched_tokens": 8, "max_model_length": 8,
        "max_cached_blocks": 2, "eos": [0],
    }


def start_engine(config):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=engine_process, args=(config, child))
    process.start()
    child.close()
    return process, parent


def receive(connection, timeout=30):
    assert connection.poll(timeout), f"engine did not respond within {timeout} seconds"
    return connection.recv()


def stop_engine(process, connection):
    if process.is_alive():
        try:
            connection.send("exit")
        except (BrokenPipeError, EOFError):
            pass
        process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(10)
    connection.close()


def test_independent_engines_initialize_together_and_exit_idempotently(engine_config):
    first, first_conn = start_engine(engine_config)
    second = second_conn = None
    try:
        first_response = receive(first_conn)
        assert first_response[0] == "ready", first_response
        second, second_conn = start_engine(engine_config)
        response = receive(second_conn)
        assert response[0] == "ready", response
        assert first_response[1] != response[1]
        first_conn.send("exit")
        assert receive(first_conn) == ("closed", [])
        second_conn.send("ping")
        assert receive(second_conn) == ("pong", True)
        second_conn.send("exit")
        assert receive(second_conn) == ("closed", [])
    finally:
        stop_engine(first, first_conn)
        if second is not None:
            stop_engine(second, second_conn)


def test_engine_does_not_unlink_existing_shared_memory(engine_config):
    try:
        sentinel = SharedMemory(name="myvllm", create=True, size=16)
    except FileExistsError:
        pytest.skip("shared memory name myvllm is already in use")
    sentinel.buf[:4] = b"keep"
    config = {**engine_config, "world_size": 2}
    process, connection = start_engine(config)
    try:
        response = receive(connection)
        assert response[0] == "ready", response
        attached = SharedMemory(name=sentinel.name)
        try:
            actual = bytes(attached.buf[:4])
        finally:
            attached.close()
        assert actual == b"keep"
        assert response[2] != sentinel.name
        connection.send("exit")
        assert receive(connection) == ("closed", [0])
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=response[2])
        attached = SharedMemory(name=sentinel.name)
        attached.close()
    finally:
        stop_engine(process, connection)
        sentinel.close()
        try:
            sentinel.unlink()
        except FileNotFoundError:
            pass


def test_graph_worker_exits_once(engine_config):
    try:
        sentinel = SharedMemory(name="myvllm", create=True, size=16)
    except FileExistsError:
        pytest.skip("shared memory name myvllm is already in use")
    config = {**engine_config, "world_size": 2, "enforce_eager": False}
    process, connection = start_engine(config)
    try:
        response = receive(connection)
        assert response[0] == "ready", response
        connection.send("exit-once")
        assert receive(connection) == ("closed", [0])
    finally:
        stop_engine(process, connection)
        sentinel.close()
        try:
            sentinel.unlink()
        except FileNotFoundError:
            pass


def test_multi_rank_instances_keep_separate_resources(engine_config):
    config = {**engine_config, "world_size": 2}
    first, first_conn = start_engine(config)
    second = second_conn = None
    try:
        first_response = receive(first_conn)
        assert first_response[0] == "ready", first_response
        second, second_conn = start_engine(config)
        second_response = receive(second_conn)
        assert second_response[0] == "ready", second_response
        assert first_response[1] != second_response[1]
        assert first_response[2] != second_response[2]
        first_conn.send("exit")
        assert receive(first_conn) == ("closed", [0])
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=first_response[2])
        attached = SharedMemory(name=second_response[2])
        attached.close()
        second_conn.send("ping")
        assert receive(second_conn) == ("pong", True)
        second_conn.send("exit")
        assert receive(second_conn) == ("closed", [0])
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=second_response[2])
    finally:
        stop_engine(first, first_conn)
        if second is not None:
            stop_engine(second, second_conn)


def test_direct_single_rank_model_runner_can_exit_twice(engine_config):
    ctx = mp.get_context("spawn")
    connection, child = ctx.Pipe()
    process = ctx.Process(target=runner_process, args=(engine_config, child))
    process.start()
    child.close()
    try:
        response = receive(connection)
        assert response[0] == "ready", response
        connection.send("exit")
        assert receive(connection) == ("closed", False)
    finally:
        stop_engine(process, connection)


def test_killed_instance_releases_store_and_other_instance_stays_alive(engine_config):
    first, first_conn = start_engine(engine_config)
    second = second_conn = None
    try:
        first_response = receive(first_conn)
        assert first_response[0] == "ready", first_response
        second, second_conn = start_engine(engine_config)
        second_response = receive(second_conn)
        assert second_response[0] == "ready", second_response
        first.kill()
        first.join(10)
        assert not first.is_alive()
        with socket.socket() as sock:
            sock.settimeout(1)
            assert sock.connect_ex(("127.0.0.1", first_response[1])) != 0
        second_conn.send("ping")
        assert receive(second_conn) == ("pong", True)
        second_conn.send("exit")
        assert receive(second_conn) == ("closed", [])
    finally:
        stop_engine(first, first_conn)
        if second is not None:
            stop_engine(second, second_conn)


@pytest.mark.parametrize("stage", ["runner", "scheduler"])
def test_failed_start_does_not_prevent_restart(engine_config, stage):
    ctx = mp.get_context("spawn")
    connection, child = ctx.Pipe()
    process = ctx.Process(target=failed_start_process, args=(engine_config, stage, child))
    process.start()
    child.close()
    try:
        assert receive(connection) == ("recovered",)
    finally:
        stop_engine(process, connection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_gpu_instances_can_initialize_generate_and_close_independently(tmp_path):
    from pd_helpers import create_tiny_qwen_checkpoint

    config = create_tiny_qwen_checkpoint(tmp_path / "checkpoint")
    ctx = mp.get_context("spawn")
    processes = []
    try:
        for _ in range(2):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=gpu_engine_process, args=(config, child))
            process.start()
            child.close()
            processes.append((process, parent))
            response = receive(parent, timeout=120)
            assert response[0] == "ready", response
            assert response[2] == 0
            if len(processes) == 1:
                first_port = response[1]
            else:
                assert response[1] != first_port
        first_conn, second_conn = processes[0][1], processes[1][1]
        first_conn.send("generate")
        first_tokens = receive(first_conn, timeout=120)
        assert first_tokens[0] == "tokens", first_tokens
        first_conn.send("exit")
        assert receive(first_conn, timeout=120) == ("closed",)
        second_conn.send("generate")
        second_tokens = receive(second_conn, timeout=120)
        assert second_tokens == first_tokens
        second_conn.send("exit")
        assert receive(second_conn, timeout=120) == ("closed",)
    finally:
        for process, connection in processes:
            stop_engine(process, connection)
