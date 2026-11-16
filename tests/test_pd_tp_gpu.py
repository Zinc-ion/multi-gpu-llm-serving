"""Opt-in four-GPU TP2 bulk smoke; numerical mismatches are test failures."""

import asyncio
import json
import multiprocessing as mp
import os

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from myvllm.pd.client import PDClient
from myvllm.sampling_parameters import SamplingParams
from pd_helpers import create_tiny_qwen_checkpoint
from pd_tp_gpu_helpers import reference_on_gpu, worker_on_gpus


pytestmark = pytest.mark.skipif(os.environ.get('RUN_PD_TP2_GPU') != '1', reason='four-GPU smoke is opt-in')


def _start(ctx, target, args):
    parent, child = ctx.Pipe(duplex=False)
    stop_child, stop = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(*args, child, stop_child))
    process.start()
    child.close()
    stop_child.close()
    try:
        assert parent.poll(180), f'worker startup timed out: {process.exitcode}'
        status = parent.recv()
        assert status[0] == 'ready', status
        return process, stop, status[1]
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(10)
        stop.close()
        raise
    finally:
        parent.close()


def _worker_entry(visible, role, config, decodes, ready, stop):
    worker_on_gpus(visible, role, config, ready, stop, decodes)


def _stop(item):
    process, stop, _ = item
    try:
        if process.is_alive():
            stop.send('stop')
            process.join(90)
        if process.is_alive():
            process.terminate()
            process.join(10)
            pytest.fail('GPU worker did not exit')
        assert process.exitcode == 0
    finally:
        stop.close()


async def _request(pe, de, case):
    d = await PDClient.open_d(de, lifetime=90)
    p = None
    try:
        p = await PDClient.open_p(pe, d.session_id, case['id'], case['prompt'],
                                  SamplingParams(**case['sampling']), lifetime=90, decode_endpoint=de)
        event = await p.next_token()
        result = [event['token']]
        if not event['finished']:
            await PDClient.handoff(p, d, await p.describe())
            while True:
                event = await d.next_token()
                result.append(event['token'])
                if event['finished']:
                    break
        return result
    finally:
        if p is not None:
            await p.close()
        await d.close()


@pytest.mark.parametrize('fp8', [False, True], ids=['bf16', 'fp8-kv'])
def test_four_gpu_tp2_bulk_against_tp1_reference(tmp_path, fp8):
    config = create_tiny_qwen_checkpoint(tmp_path / 'checkpoint')
    model_config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=16, max_position_embeddings=32, tie_word_embeddings=False)
    torch.manual_seed(20261115)
    Qwen3ForCausalLM(model_config).save_pretrained(tmp_path / 'checkpoint')
    config.update(num_kv_heads=2, dtype='bfloat16', max_num_sequences=1,
                  max_num_batched_tokens=16, max_model_length=16, max_cached_blocks=8, seed=0)
    if fp8:
        config['kv_cache_dtype'] = 'float8_e4m3fn'
    cases = [{'id': f'case-{i}', 'prompt': [1, 2, 3, 4, 5],
              'sampling': {'temperature': 0, 'max_tokens': 3, 'ignore_eos': True}}
             for i in range(2)]
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    reference_path = tmp_path / 'reference.pt'
    reference_config = {**config, 'capture_role': 'reference', 'capture_path': str(reference_path)}
    ref = ctx.Process(target=reference_on_gpu, args=('0', reference_config, cases, child))
    ref.start()
    child.close()
    try:
        assert parent.poll(180) and parent.recv()[0] == 'ok'
        ref.join(30)
        assert ref.exitcode == 0
    finally:
        parent.close()
        if ref.is_alive():
            ref.terminate()
            ref.join(10)
    workers = []
    try:
        decodes = []
        for i, gpu in enumerate(('3', '4')):
            dconfig = {**config, 'seed': i, 'capture_role': 'decode',
                       'capture_path': str(tmp_path / f'decode-{i}.pt')}
            worker = _start(ctx, _worker_entry, (gpu, 'decode', dconfig, None))
            workers.append(worker)
            decodes.append(worker[2])
        pconfig = {**config, 'world_size': 2, 'capture_role': 'prefill',
                   'capture_path': str(tmp_path / 'prefill.pt'),
                   'timing_path': str(tmp_path / 'prefill.jsonl')}
        p = _start(ctx, _worker_entry, ('0,1', 'prefill', pconfig, decodes))
        workers.append(p)
        actual = [asyncio.run(_request(p[2], de, case)) for de, case in zip(decodes, cases)]
    finally:
        for worker in reversed(workers):
            _stop(worker)
    reference = torch.load(reference_path, weights_only=False)['requests']
    prefill = torch.load(tmp_path / 'prefill.pt', weights_only=False)['requests']
    decodes_capture = [torch.load(tmp_path / f'decode-{i}.pt', weights_only=False)['requests'] for i in range(2)]
    reports, failures = {}, []
    for i, case in enumerate(cases):
        name = case['id']
        expected = reference[name]['events']
        events = prefill[name]['events'] + decodes_capture[i][name]['events']
        tokens = [event['token'] for event in expected]
        observed = [event['token'] for event in events]
        same = actual[i] == observed == tokens
        comparable = next((n + 1 for n, (a, b) in enumerate(zip(tokens, observed)) if a != b),
                          min(len(tokens), len(observed)))
        ref_logits = torch.stack([event['logits'] for event in expected[:comparable]])
        got_logits = torch.stack([event['logits'] for event in events[:comparable]])
        close = torch.allclose(ref_logits, got_logits, rtol=.02, atol=.02)
        reports[name] = {'reference_tokens': tokens, 'tokens': observed, 'tokens_identical': same,
                         'logits_within_tolerance': close, 'max_abs_error': (ref_logits - got_logits).abs().max().item()}
        if not same or not close:
            failures.append(name)
    records = [json.loads(line) for line in (tmp_path / 'prefill.jsonl').read_text().splitlines()]
    exports = [row for row in records if row['event'] == 'pd_export_done']
    assert len(exports) == 2 and all(len(row['tp_export']['ranks']) == 2 for row in exports)
    (tmp_path / 'numeric-summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    assert not failures, reports
