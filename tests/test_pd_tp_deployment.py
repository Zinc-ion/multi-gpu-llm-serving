"""The opt-in TP2 profile keeps the existing four-card comparison intact."""

import json
from pathlib import Path
import shlex
import subprocess

from test_pd_deployment import ROOT, arguments
from test_pd_load_deployment import comparison_args, value


def test_tp2_bulk_profile_uses_one_prefill_and_two_decode_workers():
    comparison = comparison_args()
    index = comparison.index('--physical-gpus')
    comparison[index + 1:index + 5] = ['0', '1', '3', '4']
    result = subprocess.run(arguments() + comparison + ['--m8-tp2-bulk'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    files = json.loads(result.stdout)['files']
    settings = json.loads(files['tp2-bulk-gateway.json'])
    manifest = json.loads(files['manifest.json'])['m8_comparison']['tp2_bulk']
    assert settings['routing_mode'] == 'load' and settings['elastic'] == {'enabled': False}
    assert 'pipeline_min_tokens' not in settings
    assert len(settings['pairs']) == 2
    assert settings['pairs'][0]['prefill'] == settings['pairs'][1]['prefill']
    decodes = [entry['decode'] for entry in settings['pairs']]
    assert len({tuple(decode) for decode in decodes}) == 2
    assert manifest['prefill_gpus'] == [0, 1] and manifest['decode_gpus'] == [3, 4]
    assert manifest['gateway_settings'] == settings and manifest['chunk_tokens'] == 1024
    assert json.loads(files['tp2-bulk-gateway-model.json'])['world_size'] == 1
    prefill = json.loads(files['tp2-bulk-prefill.json'])
    assert prefill['world_size'] == 2 and prefill['enforce_eager']
    assert not prefill['enable_chunked_prefill']
    assert 'CUDA_VISIBLE_DEVICES=0,1' in files['tp2-bulk-prefill.sh']
    words = shlex.split(files['tp2-bulk-prefill.sh'].splitlines()[-1])
    assert json.loads(value(words, '--allowed-decode-endpoints')) == decodes
    assert '--pipeline-frame-bytes' not in words
    for index, gpu in enumerate((3, 4)):
        name = f'tp2-bulk-decode-{index}'
        decode = json.loads(files[name + '.json'])
        assert decode['world_size'] == 1 and decode['enable_chunked_prefill']
        assert decode['max_num_batched_tokens'] == 1024
        assert (decode['seed'] ^ 0x80000000) != prefill['seed']
        assert f'CUDA_VISIBLE_DEVICES={gpu}' in files[name + '.sh']
        assert '--pipeline-frame-bytes' not in files[name + '.sh']
    assert not any(name.startswith('tp2-bulk-prefill-') for name in files)
    for name, content in files.items():
        if name.endswith('.sh'):
            subprocess.run(['bash', '-n'], input=content, text=True, check=True)
    # The original TP1 load and its two P workers remain independently runnable.
    assert json.loads(files['load-gateway.json'])['pairs'][0]['prefill'] != json.loads(
        files['load-gateway.json'])['pairs'][1]['prefill']
    assert all(json.loads(files[f'load-prefill-{i}.json'])['world_size'] == 1 for i in range(2))


def test_tp2_profile_requires_explicit_same_four_cards():
    result = subprocess.run(arguments() + comparison_args() + ['--m8-tp2-bulk'], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'physical-gpus 0 1 3 4' in result.stderr
