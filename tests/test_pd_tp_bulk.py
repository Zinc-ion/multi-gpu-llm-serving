"""Two real CPU ranks exercise the production TP command and PD bulk path."""

import pytest
import torch

from myvllm.pd import DecodeCore, PrefillCore
from myvllm.sampling_parameters import SamplingParams
from pd_tp_bulk_helpers import _scale, _value, decode_engine, tp_engine
from test_pd_server import make_config


def config_for(path, fp8):
    config, _ = make_config(path)
    config.update(world_size=2, _pd_tp_bulk=True, dtype='bfloat16', num_heads=4,
                  num_kv_heads=4, head_dim=3, num_layers=2, max_cached_blocks=3,
                  max_num_sequences=4, max_num_batched_tokens=12, max_model_length=12,
                  enforce_eager=True, tp_command_timeout_s=5, scale=0.5,
                  rms_norm_epsilon=1e-6, qkv_bias=False, max_position=32,
                  ffn_bias=False, tie_word_embeddings=False)
    if fp8:
        config['kv_cache_dtype'] = 'float8_e4m3fn'
    return config


@pytest.mark.parametrize('fp8', [False, True], ids=['bf16', 'fp8-scales'])
def test_real_two_rank_prefill_export_restores_all_heads_and_padding(tmp_path, fp8):
    config = config_for(tmp_path, fp8)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    with tp_engine(config) as engine:
        p = PrefillCore(engine, 4, 100000)
        d = DecodeCore(decode_engine({**config, 'world_size': 1}), 4, 100000)
        try:
            first = p.open('one', 'one', [1], params)
            second = p.open('two', 'two', [2], params)
            assert len(p.step()) == 2
            assert p.cancel(first)
            dh = d.open()
            ph = p.open(dh.session_id, 'target', [1, 2, 3, 4, 5], params)
            assert len(p.step()) == 1
            assert ph.sequence.block_table == [2, 0]
            meta = p.describe(ph)
            assert meta['model']['architecture']['num_kv_heads'] == 4
            assert d.reserve(dh, meta)
            payload = p.export(ph)
            assert d.restore(dh, payload)
            assert dh.sequence.block_table != ph.sequence.block_table
            assert p.acknowledge(ph)
            cache = d.engine.model_runner.kv_cache
            scales = d.engine.model_runner.kv_cache_scales
            for pos in range(8):
                block = dh.sequence.block_table[pos // 4]
                offset = pos % 4
                for kv in range(2):
                    for layer in range(2):
                        for head in range(4):
                            expected = _value(kv, layer, pos, head) if pos < 5 else 0
                            value = torch.full((3,), expected).to(cache.dtype)
                            assert torch.equal(cache[kv, layer, block, offset, head].view(torch.uint8),
                                               value.view(torch.uint8))
                            if fp8:
                                scale = _scale(kv, layer, pos, head) if pos < 5 else 1
                                assert scales[kv, layer, block, offset, head].item() == scale
            assert d.activate(dh)
            while dh.state == 'active':
                d.step()
            assert p.cancel(second)
            assert p.stats()['used_blocks'] == d.stats()['used_blocks'] == 0
        finally:
            p.shutdown()
            d.shutdown()


def test_rejects_tp2_early_and_decode_at_real_core_and_worker_boundaries(tmp_path):
    from myvllm.pd.worker import PDWorker

    config = config_for(tmp_path, False)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    factory_called = []
    def factory(_):
        factory_called.append(True)
        raise AssertionError('engine allocation must not start')
    with pytest.raises(ValueError, match='bulk only'):
        PDWorker('prefill', config, max_connections=1, max_transfer_bytes=1000,
                 request_timeout=5, decode_endpoint=('127.0.0.1', 9999),
                 pipeline_frame_bytes=100, engine_factory=factory)
    with pytest.raises(ValueError, match='TP1 or TP2'):
        PDWorker('decode', config, max_connections=1, max_transfer_bytes=1000,
                 request_timeout=5, engine_factory=factory)
    assert not factory_called
    with tp_engine(config) as engine:
        p = PrefillCore(engine, 4, 100000)
        try:
            with pytest.raises(ValueError, match='TP1'):
                p.prepare('early', 'early', [1], params)
            with pytest.raises(ValueError, match='world_size'):
                DecodeCore(engine, 4, 100000)
        finally:
            p.shutdown()


def test_normal_tp_engine_does_not_create_export_group(tmp_path):
    config = config_for(tmp_path, False)
    del config['_pd_tp_bulk']
    with tp_engine(config) as engine:
        assert engine.model_runner.tp_export_group is None
        with pytest.raises(ValueError, match='world_size'):
            PrefillCore(engine, 4, 100000)


def test_rank_exits_during_export_bounded_cleanup_and_group_restart(tmp_path):
    import multiprocessing as mp
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch.distributed as dist

    config = config_for(tmp_path, False)
    ctx = mp.get_context('spawn')
    entered = ctx.Event()
    gate, release = ctx.Pipe(duplex=False)
    config.update(_export_entered=entered, _export_gate=gate, tp_command_timeout_s=10)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    try:
        with tp_engine(config) as engine:
            slot = engine._shm
            p = PrefillCore(engine, 4, 100000)
            ph = p.open('failure', 'failure', [1, 2, 3, 4, 5], params)
            assert len(p.step()) == 1
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(p.export, ph)
                assert entered.wait(5), 'rank 1 never consumed the export command'
                engine.processes[0].kill()
                engine.processes[0].join(5)
                with pytest.raises((RuntimeError, OSError)):
                    future.result(timeout=12)
            assert time.monotonic() - start < 12
            assert ph.state == 'failed' and p.stats()['used_blocks'] == 0
            p.shutdown()
            with pytest.raises((RuntimeError, BrokenPipeError)):
                engine.exit()
            assert not engine.processes and slot._fd is None and slot.buf is None
            assert engine.worker_exitcodes and engine.worker_exitcodes[0] != 0
            assert not dist.is_initialized()
    finally:
        gate.close()
        release.close()
    config.pop('_export_entered')
    config.pop('_export_gate')
    with tp_engine(config) as engine:
        p = PrefillCore(engine, 4, 100000)
        try:
            ph = p.open('restart', 'restart', [1, 2, 3, 4, 5], params)
            assert len(p.step()) == 1
            assert len(p.export(ph)) == p.describe(ph)['kv']['nbytes']
            assert p.acknowledge(ph)
            assert p.stats()['used_blocks'] == 0
        finally:
            p.shutdown()


@pytest.mark.parametrize('fp8', [False, True], ids=['bf16', 'fp8-scales'])
def test_real_workers_gateway_and_two_selected_decodes(tmp_path, fp8):
    import asyncio
    import json

    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.pd.client import PDClient
    from myvllm.pd.ready import check
    from myvllm.pd.worker import PDWorker
    from pd_tp_bulk_helpers import cpu_tp_backend
    from test_pd_load_routing import http_gateway, records, settings
    from test_pd_server import body, drained

    config = config_for(tmp_path, fp8)
    config.pop('_pd_tp_bulk')
    config['max_cached_blocks'] = 8
    config['timing_path'] = str(tmp_path / 'prefill.jsonl')
    gateway_config = {**config, 'world_size': 1}
    workers = []

    async def run():
        try:
            decodes = []
            for index in range(2):
                d = PDWorker('decode', {**gateway_config, 'seed': index}, max_connections=4,
                             max_transfer_bytes=1000000, request_timeout=30, engine_factory=decode_engine)
                decodes.append(await d.start('127.0.0.1', 0))
                workers.append(d)
            p = PDWorker('prefill', config, max_connections=4, max_transfer_bytes=1000000,
                         request_timeout=30, decode_endpoint=decodes[0],
                         allowed_decode_endpoints=decodes, engine_factory=LLMEngine)
            assert '_pd_tp_bulk' not in config and p.config['_pd_tp_bulk'] is True
            pe = await p.start('127.0.0.1', 0)
            workers.append(p)
            pairs = [(pe, de) for de in decodes]
            cfg = settings(pairs, poll_interval=.01)
            ready = await check(cfg, None)
            assert len(ready['allowed_edges']) == 2
            async with http_gateway(gateway_config, cfg, tmp_path) as (client, gateway, _):
                responses = await asyncio.gather(*(
                    client.post('/v1/completions', json=body(length=3),
                                headers={'x-request-id': f'tp-{index}'}) for index in range(2)))
                assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
                assert all(response.json()['usage']['completion_tokens'] == 3 for response in responses)
                await drained(gateway, pairs)
                assigned = records(tmp_path / 'gateway.jsonl', 'pd_gateway_assigned')
                assert {tuple(row['decode']) for row in assigned} == set(decodes)
            pinfo = await PDClient.info(pe)
            assert pinfo['world_size'] == 2 and pinfo['pipeline_frame_bytes'] == 0
            exports = records(tmp_path / 'prefill.jsonl', 'pd_export_done')
            assert len(exports) == 2
            for row in exports:
                ranks = row['tp_export']['ranks']
                assert [item['rank'] for item in ranks] == [0, 1]
                assert sum(item['data_bytes'] for item in ranks) == row['data_bytes']
                assert sum(item['scale_bytes'] for item in ranks) == row['scale_bytes']
                assert row['payload_bytes'] == row['data_bytes'] + row['scale_bytes']
                assert (row['scale_bytes'] > 0) == fp8
                assert all(item['valid_data_bytes'] < item['data_bytes'] and item['export_host_ms'] >= 0
                           for item in ranks)
                assert row['tp_export']['gather_pack_host_ms'] >= 0
        finally:
            for worker in reversed(workers):
                await worker.close()

    with cpu_tp_backend(config):
        asyncio.run(run())
