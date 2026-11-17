import json
import multiprocessing as mp

import pytest
import torch

from pd_gpu_helpers import reference_process
from pd_helpers import create_tiny_qwen_checkpoint
from pd_local_gpu_helpers import decode_process, prefill_process
from pd_fp8_helpers import fp8_engine


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')


def wait_for(process, channel, timeout=180):
    assert channel.poll(timeout), f'process did not respond: pid={process.pid}, exit={process.exitcode}'
    message = channel.recv()
    assert message[0] != 'error', message
    return message


def finish(process):
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail('test process failed to exit')
    assert process.exitcode == 0


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
@pytest.mark.parametrize('eager', [True, False])
@pytest.mark.parametrize('cache_format', ['auto', 'fp8'])
def test_real_gpu_local_and_restored_decode_mixed(dtype, eager, cache_format, tmp_path):
    config = create_tiny_qwen_checkpoint(tmp_path / 'checkpoint')
    config.update(scale=1.0, dtype=dtype, max_num_sequences=2, max_num_batched_tokens=24,
                  max_model_length=24, max_cached_blocks=12, seed=0, eos=[63])
    if cache_format == 'fp8':
        config['kv_cache_dtype'] = fp8_engine().config['kv_cache_dtype']
    remote = {'id': 'remote', 'prompt': [1, 2, 3],
              'sampling': {'temperature': 0, 'max_tokens': 8, 'ignore_eos': True}}
    local = {'id': 'local', 'prompt': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'sampling': {'temperature': 0, 'max_tokens': 4, 'ignore_eos': True}}
    ctx = mp.get_context('spawn')
    reference_path, p_path, d_path = (tmp_path / name for name in ('reference.pt', 'p.pt', 'd.pt'))
    parent, child = ctx.Pipe(duplex=False)
    ref = ctx.Process(target=reference_process, args=(
        {**config, 'capture_role': 'reference', 'capture_path': str(reference_path)},
        [remote, local], child))
    ref.start()
    child.close()
    try:
        assert wait_for(ref, parent)[0] == 'ok'
        finish(ref)
    finally:
        parent.close()
        if ref.is_alive():
            ref.terminate()
            ref.join(10)

    control, dchild = ctx.Pipe()
    d = ctx.Process(target=decode_process, args=(
        {**config, 'capture_role': 'decode', 'capture_path': str(d_path),
         'enable_chunked_prefill': True, 'max_num_batched_tokens': 3, 'enforce_eager': eager},
        remote, local, dchild))
    d.start()
    dchild.close()
    p = None
    pparent = None
    try:
        status, session = wait_for(d, control)
        assert status == 'ready'
        pparent, pchild = ctx.Pipe(duplex=False)
        p = ctx.Process(target=prefill_process, args=(
            {**config, 'capture_role': 'prefill', 'capture_path': str(p_path), 'enforce_eager': True},
            session, remote, pchild))
        p.start()
        pchild.close()
        status, metadata, payload = wait_for(p, pparent)
        assert status == 'ok'
        finish(p)
        control.send((metadata, payload))
        status, observed = wait_for(d, control)
        assert status == 'ok'
        finish(d)
    finally:
        control.close()
        if pparent:
            pparent.close()
        for process in (p, d):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(10)

    ref_capture = torch.load(reference_path, weights_only=False)
    p_capture = torch.load(p_path, weights_only=False)
    d_capture = torch.load(d_path, weights_only=False)
    batches = torch.load(str(d_path) + '.batches.pt', weights_only=False)
    assert all(c['device'] == 0 and not c['missing'] for c in (ref_capture, p_capture, d_capture))
    assert batches[0]['is_prefill'] and batches[0]['sequences'] == [('remote', 3, 4), ('local', 0, 2)]
    assert any(not batch['is_prefill'] for batch in batches)
    assert all(start >= 3 and end == start + 1 for batch in batches
               for request, start, end in batch['sequences'] if request == 'remote')
    rtol, atol = (1e-4, 1e-5) if dtype == 'float32' else (.02, .02)
    report = {'dtype': dtype, 'cache_format': cache_format, 'decode_eager': eager,
              'rtol': rtol, 'atol': atol, 'cases': {}}
    for case in (remote, local):
        name = case['id']
        expected = ref_capture['requests'][name]['events']
        actual = d_capture['requests'][name]['events']
        if name == 'remote':
            actual = p_capture['requests'][name]['events'] + actual
        assert [e['token'] for e in actual] == [e['token'] for e in expected]
        assert observed[name] == [e['token'] for e in actual][1 if name == 'remote' else 0:]
        ref_logits = torch.stack([e['logits'] for e in expected])
        actual_logits = torch.stack([e['logits'] for e in actual])
        torch.testing.assert_close(actual_logits, ref_logits, rtol=rtol, atol=atol)
        report['cases'][name] = {'tokens': [e['token'] for e in actual],
                                 'max_abs_error': (ref_logits - actual_logits).abs().max().item()}
    (tmp_path / 'numeric-summary.json').write_text(json.dumps(report, indent=2) + '\n')
