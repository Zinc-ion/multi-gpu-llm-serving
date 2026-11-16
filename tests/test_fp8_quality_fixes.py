import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import torch
import uvicorn

from myvllm.engine.llm_engine import LLMEngine
from myvllm.server import create_app
from test_fp8_eval_tools import (
    COMPARE, REPO, cases, compare, fixture_runs, inputs, mutate_jsonl, native_run, prepared, run, sha, write_json,
)
from test_server import tokenizer


def test_variant_aliases_reject_actual_failed_candidate_on_baseline_port(inputs, tmp_path, monkeypatch, tokenizer):
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    _, output = prepared(inputs, '--api-url', f'http://127.0.0.1:{port}/v1')
    baseline = json.loads((output / 'eval-baseline.json').read_text())
    candidate = json.loads((output / 'eval-candidate.json').read_text())
    assert baseline['model'] != candidate['model']
    assert baseline['model_id'] == candidate['model_id'] == 'Qwen3-14B'
    second_output = inputs[0] / 'second-run'
    second = run(*inputs[2], '--output-dir', second_output)
    assert second.returncode == 0, second.stderr
    assert json.loads((second_output / 'eval-baseline.json').read_text())['model'] != baseline['model']

    class CPUModelRunner:
        def __init__(self, config, rank, event):
            self.timings = None

        def call(self, method, *args):
            if method == 'exit':
                return
            sequences, _ = args
            return torch.tensor([tokenizer.eos_token_id for seq in sequences
                                 if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq)])

    monkeypatch.setattr('myvllm.engine.llm_engine.ModelRunner', CPUModelRunner)
    config = {'model_name_or_path': str(tmp_path), 'world_size': 1, 'block_size': 4,
              'max_num_sequences': 2, 'max_num_batched_tokens': 256, 'max_model_length': 128,
              'max_cached_blocks': 64, 'eos': [tokenizer.eos_token_id], 'enable_chunked_prefill': False}
    server = uvicorn.Server(uvicorn.Config(create_app(config, baseline['model'], engine_factory=LLMEngine),
                                           log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(.01)
        code = '''
import json, sys, uvicorn
import myvllm.engine.llm_engine as engine
from myvllm.server import create_app
class CPUModelRunner:
    def __init__(self, config, rank, event):
        self.timings = None
    def call(self, method, *args):
        pass
engine.ModelRunner = CPUModelRunner
uvicorn.run(create_app(json.loads(sys.argv[1]), sys.argv[2]), host="127.0.0.1", port=int(sys.argv[3]), log_level="error")
'''
        failed = subprocess.run([sys.executable, '-c', code, json.dumps(config), candidate['model'], str(port)],
                                cwd=REPO, text=True, capture_output=True, timeout=30)
        assert failed.returncode != 0 and 'address already in use' in failed.stderr.lower()
        body = {'model': candidate['model'], 'messages': [{'role': 'user', 'content': 'hello'}],
                'stream': False, 'temperature': 0, 'max_tokens': 2}
        with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=10, trust_env=False) as client:
            wrong = client.post('/v1/chat/completions', json=body)
            assert wrong.status_code == 400
            body['model'] = baseline['model']
            right = client.post('/v1/chat/completions', json=body)
            assert right.status_code == 200 and right.json()['model'] == baseline['model']
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


@pytest.mark.parametrize('field', ['model', 'model_output'])
def test_native_prediction_model_must_match_variant_alias(inputs, field):
    output = fixture_runs(inputs)
    path = output / 'candidate/predictions/Qwen3-14B/math_500_Level 1.jsonl'
    if field == 'model':
        mutate_jsonl(path, lambda rows: rows[0].update(model='wrong-alias'))
    else:
        mutate_jsonl(path, lambda rows: rows[0]['model_output'].update(model='wrong-alias'))
    result = compare(output)
    assert result.returncode == 2 and 'planned alias' in result.stderr


def test_native_content_list_matches_evalscope_text_and_preserves_artifacts(inputs):
    output = fixture_runs(inputs)
    prediction = output / 'candidate/predictions/Qwen3-14B/math_500_Level 1.jsonl'
    review = output / 'candidate/reviews/Qwen3-14B/math_500_Level 1.jsonl'
    content = [{'internal': None, 'type': 'reasoning', 'reasoning': 'hidden', 'signature': None, 'redacted': False},
               {'internal': None, 'type': 'text', 'text': ' first ', 'refusal': None},
               {'internal': None, 'type': 'text', 'text': 'second\n', 'refusal': None}]
    mutate_jsonl(prediction, lambda rows: rows[0]['model_output']['choices'][0]['message'].update(content=content))
    mutate_jsonl(review, lambda rows: rows[0]['sample_score']['score'].update(prediction=' first \nsecond\n'))
    original = prediction.read_bytes()
    result = compare(output)
    assert result.returncode == 0, result.stderr
    assert prediction.read_bytes() == original


@pytest.mark.parametrize('content', [
    [{'type': 'unknown', 'text': 'answer'}],
    [{'type': 'reasoning'}],
    [{'type': 'text', 'text': 1}],
    [{'type': 'audio', 'audio': 'blob', 'format': 'invalid'}],
])
def test_native_content_rejects_invalid_parts(inputs, content):
    output = fixture_runs(inputs)
    prediction = output / 'candidate/predictions/Qwen3-14B/math_500_Level 1.jsonl'
    mutate_jsonl(prediction, lambda rows: rows[0]['model_output']['choices'][0]['message'].update(content=content))
    result = compare(output)
    assert result.returncode == 2 and 'invalid native prediction content' in result.stderr


def candidate_math_seventy_percent(output):
    for subset in [f'Level {i}' for i in range(1, 6)]:
        path = output / 'candidate/reviews/Qwen3-14B' / f'math_500_{subset}.jsonl'
        def edit(rows):
            for row in rows:
                if row['index'] % 10 == 1:
                    row['sample_score']['score']['value']['acc'] = 1
        mutate_jsonl(path, edit)
    path = output / 'candidate/reports/Qwen3-14B/math_500.json'
    report = json.loads(path.read_text())
    report['score'] = report['metrics'][0]['score'] = .7
    category = report['metrics'][0]['categories'][0]
    category['score'] = .7
    for subset in category['subsets']:
        subset['score'] = .7
    write_json(path, report)


@pytest.mark.parametrize('gate, exit_code', [
    ('0.1', 0),
    ('0.099999999999999999999999999999', 1),
])
def test_exact_math_gate_boundary_uses_original_decimal(inputs, gate, exit_code):
    output = fixture_runs(inputs, '--math-gate', gate)
    candidate_math_seventy_percent(output)
    result = compare(output)
    assert result.returncode == exit_code, result.stderr
    data = json.loads((output / 'comparison.json').read_text())
    math = data['tasks']['math_500']
    assert (data['baseline']['tasks']['math_500']['correct'],
            data['candidate']['tasks']['math_500']['correct']) == (400, 350)
    assert math['accuracy_drop'] == .1 and math['gate_fraction_exact'] == gate
    assert math['gate_passed'] is (exit_code == 0)


@pytest.mark.parametrize('gate, exit_code', [
    ('0.1111111111111111111111111111', 1),
    ('0.1111111111111111111111111112', 0),
])
def test_exact_needle_weighted_gate_and_cell_results(inputs, gate, exit_code):
    output = fixture_runs(inputs, '--needle-gate', gate)
    result = compare(output)
    assert result.returncode == exit_code, result.stderr
    data = json.loads((output / 'comparison.json').read_text())
    needle = data['tasks']['needle_haystack']
    assert needle['gate_passed'] is (exit_code == 0) and needle['gate_fraction_exact'] == gate
    assert len(needle['cells']) == 9 and sum(cell['count'] for cell in needle['cells'].values()) == 18
    for name, cell in needle['cells'].items():
        before = data['baseline']['tasks']['needle_haystack']['cells'][name]
        after = data['candidate']['tasks']['needle_haystack']['cells'][name]
        assert cell['accuracy_drop'] == before['accuracy'] - after['accuracy']


def test_binary_sample_scores_are_required_for_exact_gate(inputs):
    output = fixture_runs(inputs)
    path = output / 'candidate/reviews/Qwen3-14B/math_500_Level 1.jsonl'
    mutate_jsonl(path, lambda rows: rows[0]['sample_score']['score']['value'].update(acc=.5))
    result = compare(output)
    assert result.returncode == 2 and 'sample accuracy is not binary' in result.stderr


@pytest.fixture
def copied_repo(inputs, tmp_path):
    repo = tmp_path / 'candidate-repo'
    repo.mkdir()
    git_dir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=REPO, text=True).strip()
    (repo / '.git').write_text(f'gitdir: {git_dir}\n')
    shutil.copytree(REPO / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(REPO / 'tools', repo / 'tools', ignore=shutil.ignore_patterns('__pycache__'))
    args = inputs[2]
    args[0] = repo / 'tools/prepare_fp8_eval.py'
    args[args.index('--repo-dir') + 1] = repo
    return repo, repo / 'tools/compare_fp8_eval.py'


def test_prepare_rejects_serving_source_that_differs_from_candidate_blob(inputs, copied_repo):
    repo, _ = copied_repo
    path = repo / 'src/myvllm/server.py'
    path.write_text(path.read_text() + '\n# changed candidate source\n')
    result = run(*inputs[2])
    assert result.returncode != 0 and 'candidate Git blob' in result.stderr
    assert not inputs[1].exists()


@pytest.mark.parametrize('change', ['tracked', 'removed', 'untracked', 'prepare_tool', 'compare_tool'])
def test_source_or_tool_change_blocks_actual_cli_checks(inputs, copied_repo, change):
    repo, compare_tool = copied_repo
    _, output = prepared(inputs)
    rows = cases(output)
    native_run(output, rows, 'baseline')
    result = run(compare_tool, 'freeze', '--plan', output / 'plan.json')
    assert result.returncode == 0, result.stderr
    native_run(output, rows, 'candidate')
    if change == 'untracked':
        path = repo / 'src/myvllm/new_importable_module.py'
        path.write_text('value = 1\n')
    elif change == 'removed':
        (repo / 'src/myvllm/server.py').unlink()
    else:
        path = repo / {'tracked': 'src/myvllm/server.py', 'prepare_tool': 'tools/prepare_fp8_eval.py',
                       'compare_tool': 'tools/compare_fp8_eval.py'}[change]
        path.write_text(path.read_text() + '\n# changed after baseline\n')
    for args in (('check', '--stage', 'assets'), ('check', '--stage', 'baseline'),
                 ('check', '--stage', 'candidate'), ('compare', '--output', str(output / 'comparison.json'))):
        result = run(compare_tool, args[0], '--plan', output / 'plan.json', *args[1:])
        assert result.returncode == 2
        assert 'serving Python source' in result.stderr or 'quality tools differ' in result.stderr
    assert not (output / 'comparison.json').exists()


def test_generated_evaluation_rechecks_source_after_client_returns(inputs, copied_repo):
    repo, _ = copied_repo
    _, output = prepared(inputs)
    rows = cases(output)
    native_run(output, rows, 'baseline')
    shutil.rmtree(output / 'baseline')
    client = output / 'client/bin/python'
    client.parent.mkdir(parents=True)
    source = repo / 'src/myvllm/server.py'
    client.write_text('#!/bin/sh\n'
                      'if [ "$1" = "-c" ]; then\n'
                      f"    printf '\\n# changed during evaluation\\n' >> {shlex.quote(str(source))}\n"
                      '    exit 0\n'
                      'fi\n'
                      f'exec {shlex.quote(sys.executable)} "$@"\n')
    client.chmod(0o755)
    result = subprocess.run(['bash', str(output / 'evaluate-baseline.sh')],
                            cwd=repo, text=True, capture_output=True)
    assert result.returncode != 0 and 'candidate Git blob' in result.stderr
    assert not (output / 'baseline-cases.json').exists()


def test_serving_tokenizer_preflight_does_not_import_repo_root_shadow(inputs, copied_repo):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    repo, _ = copied_repo
    shadow = repo / 'myvllm'
    shadow.mkdir()
    (shadow / '__init__.py').write_text("raise RuntimeError('cwd shadow imported')\n")
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'user': 1, 'hello': 2, 'assistant': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    tokenizer.chat_template = (
        '{% for message in messages %}{{ message["role"] }} {{ message["content"] }} {% endfor %}'
        '{% if add_generation_prompt %}assistant{% endif %}'
    )
    tokenizer.save_pretrained(inputs[0] / 'model')
    _, output = prepared(inputs)
    code = (
        'import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); '
        'import compare_fp8_eval as c; root=Path(sys.argv[2]); '
        'print(json.dumps(c.server_token_counts(json.load(open(root / "plan.json")), root, '
        '[[{"role":"user","content":"hello"}]])))'
    )
    result = subprocess.run([sys.executable, '-c', code, str(repo / 'tools'), str(output)], cwd=repo,
                            env=dict(os.environ, PYTHONPATH=''), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [3]
    assert ' -P -m myvllm.server ' in (output / 'server-baseline.sh').read_text()


def test_model_revision_is_unverified_even_when_numeric_gates_pass(inputs):
    output = fixture_runs(inputs, '--math-gate', '0.25', '--needle-gate', '0.25')
    result = compare(output)
    assert result.returncode == 0, result.stderr
    assets = json.loads((output / 'assets.json').read_text())
    data = json.loads((output / 'comparison.json').read_text())
    assert assets['model']['revision_status'] == 'UNVERIFIED_USER_DECLARATION'
    assert data['model_revision'] == {'declared': assets['model']['revision'],
                                      'status': 'UNVERIFIED_USER_DECLARATION'}
    assert all(task['gate_passed'] for task in data['tasks'].values())
    assert data['formal_m7_acceptance'] is None


def linked_server_package(repo, tmp_path):
    target = tmp_path / 'linked-server'
    target.mkdir()
    (target / '__init__.py').write_text('def tokenize_request(tokenizer, body, chat):\n    return [777]\n')
    (target / '__main__.py').write_text("print('linked-server-entrypoint')\n")
    link = repo / 'src/myvllm/server'
    link.symlink_to(target, target_is_directory=True)
    return link, target


def test_linked_server_shadows_actual_import_and_prepare_rejects_it(inputs, copied_repo, tmp_path):
    repo, _ = copied_repo
    env = dict(os.environ, PYTHONPATH=str(repo / 'src'), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    code = '''
import json
from myvllm import server
class Tokenizer:
    def apply_chat_template(self, messages, **kwargs): return 'hello'
    def encode(self, prompt): return [42]
print(json.dumps({'module': server.__file__, 'tokens': server.tokenize_request(Tokenizer(), {'messages': []}, True)}))
'''

    def probe(*args):
        result = subprocess.run([sys.executable, '-P', *args], cwd=repo, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return result.stdout

    before = json.loads(probe('-c', code))
    assert before == {'module': str(repo / 'src/myvllm/server.py'), 'tokens': [42]}
    assert '--config' in probe('-m', 'myvllm.server', '--help')
    files_before = {str(path.relative_to(repo)): sha(path)
                    for path in (repo / 'src').rglob('*.py') if path.is_file()}
    link, _ = linked_server_package(repo, tmp_path)
    after = json.loads(probe('-c', code))
    assert after == {'module': str(link / '__init__.py'), 'tokens': [777]}
    assert probe('-m', 'myvllm.server', '--help').strip() == 'linked-server-entrypoint'
    assert {str(path.relative_to(repo)): sha(path)
            for path in (repo / 'src').rglob('*.py') if path.is_file()} == files_before
    result = run(*inputs[2])
    assert result.returncode == 2 and 'serving source directory symlink' in result.stderr
    assert not inputs[1].exists()


def test_link_after_prepare_blocks_freeze_and_all_acceptance_checks(inputs, copied_repo, tmp_path):
    repo, compare_tool = copied_repo
    _, output = prepared(inputs)
    rows = cases(output)
    native_run(output, rows, 'baseline')
    plan = output / 'plan.json'
    for stage in ('assets', 'baseline'):
        result = run(compare_tool, 'check', '--plan', plan, '--stage', stage)
        assert result.returncode == 0, result.stderr
    link, target = linked_server_package(repo, tmp_path)
    for args in (('check', '--stage', 'assets'), ('check', '--stage', 'baseline'),
                 ('check', '--stage', 'candidate'), ('freeze',),
                 ('compare', '--output', str(output / 'comparison-linked.json'))):
        result = run(compare_tool, args[0], '--plan', plan, *args[1:])
        assert result.returncode == 2 and 'serving source directory symlink' in result.stderr
    assert not (output / 'baseline-cases.json').exists()
    assert not (output / 'comparison-linked.json').exists()

    link.unlink()
    result = run(compare_tool, 'freeze', '--plan', plan)
    assert result.returncode == 0, result.stderr
    native_run(output, rows, 'candidate')
    result = run(compare_tool, 'check', '--plan', plan, '--stage', 'candidate')
    assert result.returncode == 0, result.stderr
    result = run(compare_tool, 'compare', '--plan', plan, '--output', output / 'comparison-before-link.json')
    assert result.returncode == 0, result.stderr
    link.symlink_to(target, target_is_directory=True)
    for args in (('check', '--stage', 'assets'), ('check', '--stage', 'baseline'),
                 ('check', '--stage', 'candidate'), ('freeze',),
                 ('compare', '--output', str(output / 'comparison-linked.json'))):
        result = run(compare_tool, args[0], '--plan', plan, *args[1:])
        assert result.returncode == 2 and 'serving source directory symlink' in result.stderr
    assert not (output / 'comparison-linked.json').exists()


@pytest.mark.parametrize('location', ['root', 'empty_subdirectory'])
def test_prepare_rejects_source_directory_symlinks(inputs, copied_repo, tmp_path, location):
    repo, _ = copied_repo
    source = repo / 'src'
    if location == 'root':
        target = repo / 'source-original'
        source.rename(target)
        source.symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / 'empty-directory'
        target.mkdir()
        (source / 'myvllm/engine/empty-link').symlink_to(target, target_is_directory=True)
    result = run(*inputs[2])
    assert result.returncode == 2 and 'serving source directory symlink' in result.stderr
    assert not inputs[1].exists()


def test_source_file_symlink_still_requires_matching_git_blob(inputs, copied_repo, tmp_path):
    repo, compare_tool = copied_repo
    source = repo / 'src/myvllm/server.py'
    target = tmp_path / 'server-source.py'
    shutil.copyfile(source, target)
    source.unlink()
    source.symlink_to(target)
    _, output = prepared(inputs)
    result = run(compare_tool, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode == 0, result.stderr
    target.write_text(target.read_text() + '\n# changed target\n')
    result = run(compare_tool, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode == 2 and 'candidate Git blob' in result.stderr


def test_model_asset_file_symlinks_remain_supported(inputs, copied_repo):
    _, compare_tool = copied_repo
    blobs = inputs[0] / 'model-blobs'
    blobs.mkdir()
    for asset in (inputs[0] / 'model').iterdir():
        target = blobs / asset.name
        asset.rename(target)
        asset.symlink_to(Path('..') / blobs.name / asset.name)
    _, output = prepared(inputs)
    assets = json.loads((output / 'assets.json').read_text())
    assert assets['model']['files'] == {path.name: sha(path) for path in blobs.iterdir()}
    result = run(compare_tool, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode == 0, result.stderr
    rows = cases(output)
    native_run(output, rows, 'baseline')
    result = run(compare_tool, 'freeze', '--plan', output / 'plan.json')
    assert result.returncode == 0, result.stderr
    native_run(output, rows, 'candidate')
    result = run(compare_tool, 'compare', '--plan', output / 'plan.json', '--output', output / 'comparison.json')
    assert result.returncode == 0, result.stderr
    (blobs / 'model.safetensors').write_bytes(b'changed fixture weights')
    result = run(compare_tool, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode == 2 and 'asset files changed' in result.stderr
