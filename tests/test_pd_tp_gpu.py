"""Opt-in four-GPU TP2 bulk smoke; numerical mismatches are test failures."""

import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from myvllm.pd.client import PDClient
from myvllm.sampling_parameters import SamplingParams
from pd_helpers import create_tiny_qwen_checkpoint
from pd_tp_gpu_helpers import reference_on_gpu, worker_on_gpus
from tools.compare_pd import stop_roles


pytestmark = pytest.mark.skipif(os.environ.get('RUN_PD_TP2_GPU') != '1', reason='four-GPU smoke is opt-in')

_START_TIMEOUT = 180
_REFERENCE_TIMEOUT = 30
_STOP_TIMEOUT = 90
_TERM_TIMEOUT = 10
_KILL_TIMEOUT = 10


class _StopPipe:
    def __init__(self, connection):
        self.connection = connection

    def write(self, data):
        try:
            self.connection.send('stop')
        except (ValueError, EOFError) as exc:
            raise OSError(str(exc)) from exc

    def flush(self):
        pass

    def close(self):
        self.connection.close()


class _Process:
    def __init__(self, process, stop):
        self.process = process
        self.stdin = _StopPipe(stop)
        self.returncode = None

    def poll(self):
        self.process.join(0)
        self.returncode = self.process.exitcode
        return self.returncode

    def send_signal(self, sig):
        os.kill(self.process.pid, sig)


def _owned_entry(target, args, ready, stop):
    os.setsid()
    stat = Path('/proc/self/stat').read_text()
    started = int(stat[stat.rfind(')') + 2:].split()[19])
    ready.send(('group', os.getpid(), started))
    try:
        if stop.recv() == 'start':
            target(*args, ready, stop)
    finally:
        ready.close()
        stop.close()


def _receive(connection, process, deadline):
    while True:
        if connection.poll(min(.1, max(0, deadline - time.monotonic()))):
            return connection.recv()
        if process.exitcode is not None:
            raise RuntimeError(f'worker exited during startup: {process.exitcode}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'worker startup timed out: {process.exitcode}')


def _start(ctx, target, args, expected='ready', owners=None):
    parent, child = ctx.Pipe(duplex=False)
    stop_child, stop = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_owned_entry, args=(target, args, child, stop_child))
    item = {'process': _Process(process, stop), 'pid': None, 'group': None,
            'start_ticks': None, 'signals': [], 'errors': [], 'remaining': []}
    if owners is not None:
        owners.append(item)
    try:
        process.start()
        item['pid'] = item['group'] = process.pid
        child.close()
        stop_child.close()
        deadline = time.monotonic() + _START_TIMEOUT
        group = _receive(parent, process, deadline)
        assert group[0] == 'group' and group[1] == process.pid, group
        item['start_ticks'] = group[2]
        stop.send('start')
        status = _receive(parent, process, deadline)
        assert status[0] == expected, status
        item['endpoint'] = status[1]
        return item
    except BaseException:
        _stop_workers([item])
        raise
    finally:
        parent.close()
        child.close()
        stop_child.close()


def _worker_entry(visible, role, config, decodes, ready, stop):
    worker_on_gpus(visible, role, config, ready, stop, decodes)


def _reference_entry(visible, config, cases, ready, stop):
    reference_on_gpu(visible, config, cases, ready)


def _stop_workers(workers):
    original = sys.exception()
    errors = []
    active = {}
    for index, item in enumerate(workers):
        if item.get('closed'):
            continue
        process = item['process'].process
        try:
            if item['pid'] is None:
                item['pid'] = item['group'] = process.pid
            if item['pid'] is not None:
                active[str(index)] = item
        except BaseException as exc:
            errors.append(f"process registration: {exc!r}")
    timeouts = SimpleNamespace(normal_exit_timeout=_STOP_TIMEOUT, term_timeout=_TERM_TIMEOUT,
                               kill_timeout=_KILL_TIMEOUT)
    try:
        errors.extend(stop_roles(active, timeouts))
    except BaseException as exc:
        errors.append(repr(exc))
        for name, item in active.items():
            try:
                errors.extend(stop_roles({name: item}, timeouts))
            except BaseException as cleanup:
                errors.append(repr(cleanup))
    reports = []
    for item in workers:
        if item.get('closed'):
            continue
        process = item['process'].process
        try:
            if item['pid'] is None:
                item['process'].stdin.close()
            if not process.is_alive():
                item['exitcode'] = process.exitcode
                process.close()
                item['closed'] = True
            else:
                item['closed'] = False
                errors.append(f"pid {item['pid']} remains alive; process handle retained")
        except BaseException as exc:
            errors.append(f"pid {item['pid']} close: {exc!r}")
        if item['signals']:
            errors.append(f"pid {item['pid']} required signals {item['signals']}")
        reports.append(f"GPU smoke pid={item['pid']} exitcode={item.get('exitcode')} "
                       f"signals={item['signals']} remaining={item['remaining']}")
    if reports:
        try:
            print('\n'.join(reports), flush=True)
        except BaseException as exc:
            errors.append(f"cleanup report: {exc!r}")
    if errors:
        message = f"GPU smoke cleanup failed: {errors}"
        if original is not None:
            original.add_note(message)
        else:
            raise RuntimeError(message)


def _reference(ctx, visible, config, cases):
    owners = []
    try:
        ref = _start(ctx, _reference_entry, (visible, config, cases), expected='ok', owners=owners)
        process = ref['process'].process
        process.join(_REFERENCE_TIMEOUT)
        assert process.exitcode == 0, f'reference did not exit successfully: {process.exitcode}'
    finally:
        _stop_workers(owners)


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
    reference_path = tmp_path / 'reference.pt'
    reference_config = {**config, 'capture_role': 'reference', 'capture_path': str(reference_path)}
    _reference(ctx, '0', reference_config, cases)
    workers = []
    try:
        decodes = []
        for i, gpu in enumerate(('3', '4')):
            dconfig = {**config, 'seed': i, 'capture_role': 'decode',
                       'capture_path': str(tmp_path / f'decode-{i}.pt')}
            worker = _start(ctx, _worker_entry, (gpu, 'decode', dconfig, None), owners=workers)
            decodes.append(worker['endpoint'])
        pconfig = {**config, 'world_size': 2, 'capture_role': 'prefill',
                   'capture_path': str(tmp_path / 'prefill.pt'),
                   'timing_path': str(tmp_path / 'prefill.jsonl')}
        p = _start(ctx, _worker_entry, ('0,1', 'prefill', pconfig, decodes), owners=workers)
        actual = [asyncio.run(_request(p['endpoint'], de, case)) for de, case in zip(decodes, cases)]
    finally:
        _stop_workers(workers)
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
