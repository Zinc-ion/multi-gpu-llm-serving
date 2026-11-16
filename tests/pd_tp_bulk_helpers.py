"""CPU-only model execution around the real TP engine and PD lifecycle."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

import myvllm.engine.llm_engine as engine_module
from myvllm.engine.llm_engine import LLMEngine, worker_process
from myvllm.engine.model_runner import ModelRunner
from test_instance_resources import cpu_backend
from test_pd_core import CPUExecution, make_engine


_export_pd_tp_kv = ModelRunner.export_pd_tp_kv


def _wait_rank_export(runner, *args):
    if runner.rank:
        runner.config['_export_entered'].set()
        runner.config['_export_gate'].recv_bytes()
    return _export_pd_tp_kv(runner, *args)


def _allocate_cpu(runner):
    config = runner.config
    shape = (2, config['num_layers'], config['max_cached_blocks'], config['block_size'],
             config['num_kv_heads'] // runner.world_size, config['head_dim'])
    runner.kv_cache = torch.zeros(shape, dtype=runner.kv_cache_dtype, device='cpu')
    runner.kv_cache_scales = (torch.ones(shape[:-1], dtype=torch.float32, device='cpu')
                              if runner.kv_cache_dtype == torch.float8_e4m3fn else None)


def _value(kv, layer, pos, head):
    return 1 + kv + layer * 2 + pos + head * 10


def _scale(kv, layer, pos, head):
    return (1 + kv + layer * 2 + pos + head * 3) / 16


def _run_cpu(runner, sequences, is_prefill):
    for seq in sequences:
        for pos in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens):
            block = seq.block_table[pos // seq.block_size]
            offset = pos % seq.block_size
            for kv in range(2):
                for layer in range(runner.kv_cache.shape[1]):
                    for local_head in range(runner.kv_cache.shape[4]):
                        head = runner.rank * runner.kv_cache.shape[4] + local_head
                        values = torch.full((runner.kv_cache.shape[-1],),
                                            _value(kv, layer, pos, head), device='cpu')
                        runner.kv_cache[kv, layer, block, offset, local_head].copy_(
                            values.to(runner.kv_cache.dtype))
                        if runner.kv_cache_scales is not None:
                            runner.kv_cache_scales[kv, layer, block, offset, local_head] = _scale(kv, layer, pos, head)
    if runner.rank:
        return None
    return torch.tensor([sum(seq.token_ids) % 60 + 1 for seq in sequences
                         if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)], device='cpu')


@contextmanager
def _cpu_model(config):
    old_dtype = torch.get_default_dtype()
    try:
        with ExitStack() as stack:
            stack.enter_context(cpu_backend(config))
            stack.enter_context(patch.object(ModelRunner, 'allocate_kv_cache', _allocate_cpu))
            stack.enter_context(patch.object(ModelRunner, 'run', _run_cpu))
            stack.enter_context(patch('myvllm.engine.model_runner.StepTimings', lambda: None))
            if '_export_entered' in config:
                stack.enter_context(patch.object(ModelRunner, 'export_pd_tp_kv', _wait_rank_export))
            yield
    finally:
        torch.set_default_dtype(old_dtype)


def cpu_tp_worker(config, rank, event):
    with _cpu_model(config):
        worker_process(config, rank, event)


@contextmanager
def cpu_tp_backend(config):
    with _cpu_model(config), patch.object(engine_module, 'worker_process', cpu_tp_worker):
        yield


@contextmanager
def tp_engine(config):
    with cpu_tp_backend(config):
        engine = LLMEngine(config)
        try:
            yield engine
        finally:
            engine.exit()


def decode_engine(config):
    engine = make_engine(blocks=config['max_cached_blocks'], block_size=config['block_size'],
                         budget=config['max_num_batched_tokens'], sequences=config['max_num_sequences'],
                         model_length=config['max_model_length'], dtype=config['dtype'])
    engine.config.update(config)
    engine.model_runner = CPUExecution(engine.config)
    if config.get('kv_cache_dtype') == 'float8_e4m3fn':
        shape = engine.model_runner.kv_cache.shape
        engine.model_runner.kv_cache = torch.zeros(shape, dtype=torch.float8_e4m3fn, device='cpu')
        engine.model_runner.kv_cache_scales = torch.ones(shape[:-1], dtype=torch.float32, device='cpu')
    else:
        engine.model_runner.kv_cache_scales = None
    engine.exit = lambda: None
    return engine
