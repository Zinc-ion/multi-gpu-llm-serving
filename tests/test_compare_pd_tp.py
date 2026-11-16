"""TP2 capture validates explicit device selection and preserves real LM-head capture."""

import argparse
import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from tools import compare_pd
from test_compare_pd import ROOT, capture_args


def tp_args(tmp_path):
    args = capture_args(tmp_path)
    model_config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=16, max_position_embeddings=32, tie_word_embeddings=False)
    torch.manual_seed(20261115)
    Qwen3ForCausalLM(model_config).save_pretrained(tmp_path / 'checkpoint')
    for role in ('reference', 'prefill', 'decode'):
        path = Path(getattr(args, role + '_config'))
        config = json.loads(path.read_text())
        config.update(num_kv_heads=2, world_size=2 if role == 'prefill' else 1,
                      enable_chunked_prefill=False, max_num_batched_tokens=8)
        path.write_text(json.dumps(config))
    args.raw_kv = False
    args.prefill_gpu = '0,1'
    return args


def test_tp2_bulk_capture_on_real_cpu_gloo_models(tmp_path, monkeypatch):
    args = tp_args(tmp_path)
    monkeypatch.setattr(compare_pd, 'SCRIPT', ROOT / 'tests/pd_tp_compare_cpu_child.py')
    compare_pd.capture(args)
    root = Path(args.output_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['roles']['prefill']['gpu'] == '0,1'
    assert all(role['exitcode'] == 0 and not role['remaining'] for role in manifest['roles'].values())
    prefill = json.loads((root / 'prefill.engine.json').read_text())
    assert prefill['config']['world_size'] == 2 and prefill['config']['_pd_tp_bulk'] is True
    assert prefill['cache']['pool_shape'][-2] == 1
    for index in range(2):
        reference = compare_pd.load_pt(root / f'reference-{index:05d}.pt')
        candidate = compare_pd.load_pt(root / f'candidate-{index:05d}.pt')
        result = compare_pd.analyze_data(reference, candidate, None)
        assert result['tokens_identical'] and result['all_finite'] and result['greedy_matches_logits']
        torch.testing.assert_close(reference['cases'][0]['logits'], candidate['cases'][0]['logits'],
                                   rtol=1e-4, atol=1e-5)
        assert candidate['metadata']['kv_handoff_equal'] is None


@pytest.mark.parametrize('reason', ['one-gpu', 'duplicate-gpu', 'mixed-ids', 'raw-kv', 'pipeline', 'tp2-decode', 'tp2-reference'])
def test_tp2_capture_rejects_unsupported_configuration_before_start(tmp_path, reason):
    args = tp_args(tmp_path)
    if reason == 'one-gpu':
        args.prefill_gpu = '0'
    elif reason == 'duplicate-gpu':
        args.prefill_gpu = '0,0'
    elif reason == 'mixed-ids':
        args.prefill_gpu = '0,GPU-abcd'
    elif reason == 'raw-kv':
        args.raw_kv = True
    elif reason == 'pipeline':
        args.transfer = 'pipeline'
    else:
        path = Path(getattr(args, reason.removeprefix('tp2-') + '_config'))
        config = json.loads(path.read_text())
        config['world_size'] = 2
        path.write_text(json.dumps(config))
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        compare_pd.read_inputs(args)
    assert not Path(args.output_dir).exists()


def test_tp2_accepts_two_uuid_devices():
    assert compare_pd.prefill_gpu_selection('GPU-abcd,GPU-ef01') == 'GPU-abcd,GPU-ef01'
