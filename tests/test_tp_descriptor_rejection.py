"""Oversized real TP commands must cancel only their scheduled requests."""

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import myvllm.engine.llm_engine as engine_module
from myvllm.engine.ipc import CommandSizeError
from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import SequenceStatus
from myvllm.layers.sampler import SamplerLayer
from myvllm.sampling_parameters import SamplingParams
from myvllm.server import EngineThread, PendingRequest
from myvllm.utils import get_context
from test_instance_resources import cpu_backend


_worker_process = engine_module.worker_process
_tensor = torch.tensor


def _cpu_tensor(*args, **kwargs):
    kwargs.pop('pin_memory', None)
    return _tensor(*args, **kwargs)


def _model(runner, input_ids, is_prefill):
    context = get_context()
    record = {'is_prefill': is_prefill, 'positions': context.positions.tolist(),
              'context_lens': context.context_lens.tolist() if context.context_lens is not None else None}
    with Path(runner.config['_probe_dir'], f'rank-{runner.rank}.jsonl').open('a') as file:
        file.write(json.dumps(record) + '\n')
    return _tensor([[0., 1.]] * len(input_ids))


def _sampler(sampler, logits, temperatures):
    return torch.ones(len(temperatures), dtype=torch.long)


def _cpu_scope(config):
    stack = ExitStack()
    stack.enter_context(cpu_backend(config))
    stack.enter_context(patch.object(torch, 'tensor', _cpu_tensor))
    stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self))
    stack.enter_context(patch.object(ModelRunner, 'run_model', _model))
    stack.enter_context(patch.object(SamplerLayer, 'forward', _sampler))
    return stack


def _cpu_worker(config, rank, event):
    with _cpu_scope(config):
        _worker_process(config, rank, event)


def test_oversized_real_tp_step_cancels_batch_and_preserves_other_requests(tmp_path):
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, 'hello': 1}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]').save_pretrained(tmp_path)
    config = {
        'model_name_or_path': str(tmp_path), 'model_type': 'Qwen3-0.6B', '_probe_dir': str(tmp_path),
        'world_size': 2, 'block_size': 256, 'enforce_eager': True,
        'vocab_size': 151936, 'hidden_size': 4, 'num_heads': 2, 'head_dim': 2,
        'scale': .5, 'num_kv_heads': 2, 'rms_norm_epsilon': 1e-6,
        'qkv_bias': False, 'base': 10000, 'max_position': 32768,
        'intermediate_size': 8, 'ffn_bias': False, 'num_layers': 1,
        'tie_word_embeddings': False, 'max_num_sequences': 8,
        'max_num_batched_tokens': 240000, 'max_model_length': 32768,
        'max_cached_blocks': 1024, 'eos': [0], 'tp_command_timeout_s': 15,
    }
    engine = None
    controller = None
    try:
        with _cpu_scope(config), patch.object(engine_module, 'worker_process', _cpu_worker):
            engine = LLMEngine(config)
            controller = EngineThread({**config, 'timing_path': str(tmp_path / 'admission.jsonl')}, lambda _: engine)
            controller.engine = engine

            async def exercise():
                loop = asyncio.get_running_loop()
                async def admit(name, ids, max_tokens):
                    pending = PendingRequest(name, ids, SamplingParams(temperature=0, max_tokens=max_tokens,
                                             ignore_eos=True), time.perf_counter(), loop,
                                             loop.create_future(), asyncio.Queue())
                    controller.admit(pending)
                    return await pending.admitted

                survivor = await admit('survivor', [1, 2], 2)
                engine.step(on_tokens=controller.publish)
                assert survivor.num_cached_tokens == 2 and survivor.status == SequenceStatus.RUNNING
                oversized = [await admit(f'oversized-{i}', [100000] * 30000, 1) for i in range(8)]
                waiting = await admit('waiting', [3, 4], 1)
                before = bytes(engine._shm.buf)
                with pytest.raises(CommandSizeError, match='TP command descriptor') as captured:
                    engine.step(on_tokens=controller.publish)
                assert bytes(engine._shm.buf) == before
                assert all(seq.status == SequenceStatus.CANCELLED and not seq.block_table
                           and seq.num_cached_tokens == 0 for seq in oversized)
                assert type(captured.value) is CommandSizeError
                assert survivor in engine.scheduler.running and survivor.num_cached_tokens == 2
                assert waiting in engine.scheduler.waiting and waiting.status == SequenceStatus.WAITING
                assert {survivor.seq_id, waiting.seq_id} == set(engine.request_ids)
                manager = engine.scheduler.block_manager
                assert manager.used_block_ids == set(survivor.block_table)
                assert manager.used_block_ids.isdisjoint(manager.free_block_ids)
                assert manager.used_block_ids | set(manager.free_block_ids) == set(range(1024))
                assert all(manager.blocks[i].ref_count == 0 and manager.blocks[i].hash == -1
                           for i in manager.free_block_ids)
                for seq in oversized:
                    controller.command('cancel', seq.seq_id)
                engine.step(on_tokens=controller.publish)
                engine.step(on_tokens=controller.publish)
                assert waiting.status == survivor.status == SequenceStatus.FINISHED
                fresh = await admit('fresh', [5, 6, 7], 1)
                engine.step(on_tokens=controller.publish)
                assert fresh.status == SequenceStatus.FINISHED
                assert not manager.used_block_ids and not engine.request_ids
                assert not controller.requests

            try:
                asyncio.run(exercise())
            finally:
                for seq_id in list(controller.requests):
                    controller.command('cancel', seq_id)
                engine.exit()
            assert engine.worker_exitcodes == [0]
            for rank in (0, 1):
                rows = [json.loads(line) for line in (tmp_path / f'rank-{rank}.jsonl').read_text().splitlines()]
                assert [row['is_prefill'] for row in rows] == [True, True, False, True]
                assert rows[2]['positions'] == [2] and rows[2]['context_lens'] == [3]
    finally:
        if engine is not None:
            engine.exit()
