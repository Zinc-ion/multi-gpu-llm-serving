from collections import Counter
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import shutil
import subprocess
import sys
import threading

import pytest

from test_fp8_eval_tools import (
    COMPARE, REPO, cases, inputs, mutate_jsonl, native_run, prepared, run, write_json,
)


def test_shared_8192_budget_rejects_long_needle(inputs):
    result = run(*inputs[2], '--max-tokens', '8192')
    assert result.returncode == 2
    assert 'context, chat budget and reply must fit strictly below' in result.stderr
    assert not inputs[1].exists()


def test_separate_budgets_prepare_math_and_long_needle(inputs):
    result = run(*inputs[2], '--math-max-tokens', '8192', '--needle-max-tokens', '1024')
    assert result.returncode == 0, result.stderr
    plan = json.loads((inputs[1] / 'plan.json').read_text())
    assert plan['task_max_tokens'] == {'math_500': 8192, 'needle_haystack': 1024}


@pytest.mark.parametrize('options, expected', [
    (['--math-max-tokens', '8192'], {'math_500': 8192, 'needle_haystack': 1024}),
    (['--max-tokens', '768', '--needle-max-tokens', '512'], {'math_500': 768, 'needle_haystack': 512}),
])
def test_unspecified_task_uses_shared_budget(inputs, options, expected):
    _, output = prepared(inputs, *options)
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['task_max_tokens'] == expected
    for task, budget in expected.items():
        config = json.loads((output / f'eval-baseline-{task}.json').read_text())
        assert config['datasets'] == [task]
        assert config['generation_config']['max_tokens'] == budget
        assert config['seed'] == 7 and 'seed' not in config['generation_config']
        assert config['generation_config']['temperature'] == 0 and config['generation_config']['top_p'] == 1
    assert run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'assets').returncode == 0


@pytest.mark.parametrize('options', [
    ['--math-max-tokens', '0'],
    ['--needle-max-tokens', '-1'],
    ['--math-max-tokens', '8192', '--needle-max-tokens', '2048'],
])
def test_invalid_task_budgets_fail_before_writing(inputs, options):
    result = run(*inputs[2], *options)
    assert result.returncode == 2
    assert not inputs[1].exists()


def test_old_plan_version_cannot_hide_task_budgets(inputs):
    _, output = prepared(inputs, '--math-max-tokens', '8192', '--needle-max-tokens', '1024')
    path = output / 'plan.json'
    plan = json.loads(path.read_text())
    plan['version'] = 1
    write_json(path, plan)
    result = run(COMPARE, 'check', '--plan', path, '--stage', 'assets')
    assert result.returncode == 2 and 'version 1 task output budgets must match' in result.stderr


def test_legacy_plan_without_task_budget_field_remains_readable(inputs):
    _, output = prepared(inputs)
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['task_max_tokens'] == {'math_500': 1024, 'needle_haystack': 1024}
    plan.pop('task_max_tokens')
    write_json(plan_path, plan)
    rows = cases(output)
    native_run(output, rows, 'baseline')
    frozen = run(COMPARE, 'freeze', '--plan', plan_path)
    assert frozen.returncode == 0, frozen.stderr
    native_run(output, rows, 'candidate')
    result = run(COMPARE, 'compare', '--plan', plan_path, '--output', output / 'comparison.json')
    assert result.returncode == 0, result.stderr


def test_frozen_legacy_plan_requires_its_original_tools(inputs, tmp_path):
    repo = tmp_path / 'legacy-repo'
    repo.mkdir()
    git_dir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=REPO, text=True).strip()
    (repo / '.git').write_text(f'gitdir: {git_dir}\n')
    shutil.copytree(REPO / 'src', repo / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    (repo / 'tools').mkdir()
    originals = {}
    for name in ('prepare_fp8_eval.py', 'compare_fp8_eval.py'):
        originals[name] = subprocess.check_output(['git', 'show', f'9aaf445:tools/{name}'], cwd=REPO)
        (repo / 'tools' / name).write_bytes(originals[name])
    args = inputs[2]
    args[0] = repo / 'tools/prepare_fp8_eval.py'
    args[args.index('--repo-dir') + 1] = repo
    _, output = prepared(inputs)
    assert json.loads((output / 'plan.json').read_text())['version'] == 1
    rows = cases(output)
    native_run(output, rows, 'baseline')
    old_compare = repo / 'tools/compare_fp8_eval.py'
    frozen = run(old_compare, 'freeze', '--plan', output / 'plan.json')
    assert frozen.returncode == 0, frozen.stderr
    native_run(output, rows, 'candidate')
    for name in originals:
        (repo / 'tools' / name).write_bytes((REPO / 'tools' / name).read_bytes())
    rejected = run(old_compare, 'check', '--plan', output / 'plan.json', '--stage', 'candidate')
    assert rejected.returncode == 2 and 'quality tools differ' in rejected.stderr
    for name, content in originals.items():
        (repo / 'tools' / name).write_bytes(content)
    result = run(old_compare, 'compare', '--plan', output / 'plan.json', '--output', output / 'comparison.json')
    assert result.returncode == 0, result.stderr


def make_stub_client(output, samples):
    package = output.parent / 'fake-evalscope'
    for name in ('evalscope', 'evalscope/api', 'evalscope/utils'):
        directory = package / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / '__init__.py').write_text('')
    modules = {
        'evalscope/__init__.py': 'from fp8_budget_client_stub import run_task\n',
        'evalscope/config.py': 'from fp8_budget_client_stub import TaskConfig\n',
        'evalscope/api/evaluator.py': 'from fp8_budget_client_stub import TaskState\n',
        'evalscope/api/registry.py': 'from fp8_budget_client_stub import get_benchmark\n',
        'evalscope/utils/model_utils.py': 'from fp8_budget_client_stub import seed_everything\n',
        'evalscope-1.5.1.dist-info/METADATA': 'Metadata-Version: 2.1\nName: evalscope\nVersion: 1.5.1\n',
    }
    for name, content in modules.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    client = output / 'client/bin/python'
    client.parent.mkdir(parents=True)
    pythonpath = f'{package}:{REPO / "tests"}'
    client.write_text('#!/bin/sh\n'
                      f'export PYTHONPATH={shlex.quote(pythonpath)}\n'
                      f'export FP8_BUDGET_SAMPLES={shlex.quote(str(samples))}\n'
                      f'exec {shlex.quote(sys.executable)} "$@"\n')
    client.chmod(0o755)


def run_script(output, name):
    result = subprocess.run(['bash', str(output / name)], cwd=REPO, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result


def test_generated_scripts_materialize_send_freeze_and_compare(inputs):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({'[UNK]': 0, 'user': 1, 'assistant': 2, 'x': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
    tokenizer.chat_template = (
        '{% for message in messages %}{{ message["role"] }} {{ message["content"] }} {% endfor %}'
        '{% if add_generation_prompt %}assistant{% endif %}'
    )
    tokenizer.save_pretrained(inputs[0] / 'model')

    requests = []
    counts = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            prompt_tokens = counts[body['messages'][0]['content']]
            response = json.dumps({'model': body['model'], 'choices': [{'message': {'content': 'answer'}}],
                                   'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 3}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        port = server.server_address[1]
        _, output = prepared(inputs, '--api-url', f'http://127.0.0.1:{port}/v1',
                             '--math-max-tokens', '8192', '--needle-max-tokens', '1024')
        original = cases(output)
        for row in original:
            text = f"{row['task']} {row['subset']} {row['index']}"
            if row['task'] == 'needle_haystack':
                text += ' ' + 'x ' * row['context_length']
            row['input'] = text
            row['messages'] = [{'role': 'user', 'content': text}]
        sample_path = output.parent / 'synthetic-samples.json'
        write_json(sample_path, original)
        (output / 'materialized.json').unlink()
        make_stub_client(output, sample_path)

        run_script(output, 'materialize.sh')
        data = json.loads((output / 'materialized.json').read_text())
        rows = data['cases']
        assert [(r['task'], r['subset'], r['index'], r['input'], r['target'], r['messages']) for r in rows] == [
            (r['task'], r['subset'], r['index'], r['input'], r['target'], r['messages']) for r in original]
        assert len(rows) == 518
        assert all(row['prompt_tokens'] + (8192 if row['task'] == 'math_500' else 1024) < 17408 for row in rows)
        assert any(row['prompt_tokens'] + 8192 >= 17408 for row in rows if row['task'] == 'needle_haystack')
        for row in rows:
            counts[row['messages'][0]['content']] = row['prompt_tokens']

        original_materialized = (output / 'materialized.json').read_text()
        for task, too_long in (('math_500', 17408 - 8192), ('needle_haystack', 17408 - 1024)):
            data = json.loads(original_materialized)
            next(row for row in data['cases'] if row['task'] == task)['prompt_tokens'] = too_long
            write_json(output / 'materialized.json', data)
            rejected = run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'baseline')
            assert rejected.returncode == 2 and 'context budget' in rejected.stderr
        (output / 'materialized.json').write_text(original_materialized)

        plan = json.loads((output / 'plan.json').read_text())
        for variant in ('baseline', 'candidate'):
            config = json.loads((output / f'server-{variant}.json').read_text())
            event = {'event': 'server_start', 'config': config, 'arguments': {
                'config': str(output / f'server-{variant}.json'),
                'served_model_name': plan['api_model_aliases'][variant], 'host': '127.0.0.1', 'port': port}}
            (output / f'server-{variant}.log').write_text(json.dumps(event) + '\n')
            run_script(output, f'evaluate-{variant}.sh')
        run_script(output, 'compare.sh')

        expected = Counter((plan['api_model_aliases'][variant], row['messages'][0]['content'],
                            8192 if row['task'] == 'math_500' else 1024)
                           for variant in ('baseline', 'candidate') for row in rows)
        actual = Counter((body['model'], body['messages'][0]['content'], body['max_tokens']) for body in requests)
        assert actual == expected
        assert len(requests) == 1036
        assert all(body['temperature'] == 0 and body['top_p'] == 1 and body['stream'] is False
                   and 'seed' not in body for body in requests)
        evidence = [{'model': model, 'message_sha256': hashlib.sha256(text.encode()).hexdigest(),
                     'max_tokens': budget, 'count': count} for (model, text, budget), count in actual.items()]
        write_json(output / 'synthetic-http-evidence.json', evidence)
        comparison = json.loads((output / 'comparison.json').read_text())
        assert comparison['baseline']['termination_counts_by_task'] == {
            'math_500': {'stop': 500}, 'needle_haystack': {'stop': 18}}
        assert comparison['candidate']['output_tokens_by_task'] == {'math_500': 1500, 'needle_haystack': 54}
        assert comparison['formal_m7_acceptance'] is None
        assert len(comparison['baseline']['native_sha256']) == 16

        math_prediction = output / 'candidate/math_500/predictions/Qwen3-14B/math_500_Level 1.jsonl'
        needle_prediction = output / 'candidate/needle_haystack/predictions/Qwen3-14B/needle_haystack_english.jsonl'
        for path, field, value in ((math_prediction, 'input_tokens', -1),
                                   (math_prediction, 'output_tokens', 8193),
                                   (needle_prediction, 'output_tokens', 1025)):
            original_prediction = path.read_text()
            mutate_jsonl(path, lambda records: records[0]['model_output']['usage'].update({field: value}))
            rejected = run(COMPARE, 'compare', '--plan', output / 'plan.json',
                           '--output', output / 'rejected-usage.json')
            assert rejected.returncode == 2 and 'native token usage' in rejected.stderr
            path.write_text(original_prediction)

        wrong_task = output / 'candidate/needle_haystack/reports/Qwen3-14B/needle_haystack.json'
        original_report = wrong_task.read_bytes()
        wrong_task.write_bytes((output / 'candidate/math_500/reports/Qwen3-14B/math_500.json').read_bytes())
        rejected = run(COMPARE, 'compare', '--plan', output / 'plan.json',
                       '--output', output / 'rejected-wrong-task.json')
        assert rejected.returncode == 2 and 'report model or dataset mismatch' in rejected.stderr
        wrong_task.write_bytes(original_report)
        moved = wrong_task.with_suffix('.moved')
        wrong_task.rename(moved)
        rejected = run(COMPARE, 'compare', '--plan', output / 'plan.json',
                       '--output', output / 'rejected-missing-report.json')
        assert rejected.returncode == 2
        moved.rename(wrong_task)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)
        assert not thread.is_alive()
