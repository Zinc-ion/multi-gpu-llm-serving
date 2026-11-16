import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


REPO = Path(__file__).resolve().parents[1]
PREPARE = REPO / 'tools/prepare_fp8_eval.py'
COMPARE = REPO / 'tools/compare_fp8_eval.py'


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, text=True, capture_output=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(rows):
    identity = [{key: row[key] for key in ('task', 'subset', 'index', 'input', 'target')}
                for row in sorted(rows, key=lambda item: (item['task'], item['subset'], item['index']))]
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'workspace'
    model, math_data, needle = (root / name for name in ('model', 'math', 'needle'))
    for path in (model, math_data, needle):
        path.mkdir(parents=True)
    (model / 'config.json').write_text(json.dumps({'hidden_size': 5120, 'num_hidden_layers': 40,
                                                   'num_attention_heads': 40, 'num_key_value_heads': 8}))
    (model / 'tokenizer.json').write_text('fixture tokenizer')
    (model / 'model.safetensors').write_bytes(b'fixture weights')
    (math_data / 'test.jsonl').write_text('fixture local math dataset')
    (needle / 'PaulGraham_Essays.txt').write_text('fixture English corpus')
    (needle / 'Journey_to_the_West.txt').write_text('fixture Chinese corpus')
    output = root / 'run'
    args = [PREPARE, '--workspace-root', root, '--output-dir', output, '--repo-dir', REPO,
            '--server-python', sys.executable, '--model-path', model, '--math-data', math_data,
            '--needle-data', needle, '--math-revision', 'math-revision-123',
            '--needle-revision', 'needle-revision-456', '--candidate-commit', 'a' * 40]
    return root, output, args


def prepared(inputs, *extra):
    root, output, args = inputs
    result = run(*args, *extra)
    assert result.returncode == 0, result.stderr
    return root, output


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def cases(output):
    plan = json.loads((output / 'plan.json').read_text())
    rows = []
    for subset in plan['math_subsets']:
        for i in range(100):
            rows.append({'task': 'math_500', 'subset': subset, 'index': i,
                         'input': f'original math prompt {subset}/{i}', 'target': f'target {i}',
                         'messages': [{'role': 'user', 'content': f'math {subset}/{i}'}], 'prompt_tokens': 40})
    for subset in plan['needle_subsets']:
        for i, (context, depth) in enumerate((c, d) for c in plan['context_lengths'] for d in plan['depths']):
            rows.append({'task': 'needle_haystack', 'subset': subset, 'index': i,
                         'input': f'original needle prompt {subset}/{i}', 'target': 'needle target',
                         'messages': [{'role': 'user', 'content': f'needle {subset}/{i}'}],
                         'prompt_tokens': context + 20, 'context_length': context, 'depth_percent': depth})
    write_json(output / 'materialized.json', {'plan_sha256': sha(output / 'plan.json'),
                                              'input_sha256': fingerprint(rows), 'cases': rows,
                                              'client_environment': {
                                                  'python': sys.version,
                                                  'packages': sorted(f"{item.metadata['Name']}=={item.version}"
                                                                     for item in distributions(path=sorted({
                                                                         sysconfig.get_path('purelib'),
                                                                         sysconfig.get_path('platlib')})))}})
    return rows


def report_metric(name, subsets):
    subset_data = []
    for subset, values in subsets.items():
        subset_data.append({'name': subset, 'num': len(values), 'score': round(sum(values) / len(values), 4)})
    count = sum(row['num'] for row in subset_data)
    score = round(sum(row['score'] * row['num'] for row in subset_data) / count, 4)
    category = {'name': ['default'], 'num': count, 'score': score, 'macro_score': score, 'subsets': subset_data}
    return {'name': name, 'num': count, 'score': score, 'macro_score': score, 'categories': [category]}


def native_run(output, rows, variant):
    config = json.loads((output / f'server-{variant}.json').read_text())
    event = {'event': 'server_start', 'config': config, 'arguments': {
        'config': str(output / f'server-{variant}.json'), 'served_model_name': 'Qwen3-14B',
        'host': '127.0.0.1', 'port': 8000}}
    (output / f'server-{variant}.log').write_text('INFO:myvllm:' + json.dumps(event) + '\n')
    for task in ('math_500', 'needle_haystack'):
        task_rows = [row for row in rows if row['task'] == task]
        subsets = list(dict.fromkeys(row['subset'] for row in task_rows))
        grouped = {}
        for subset in subsets:
            reviews, predictions = [], []
            for row in (row for row in task_rows if row['subset'] == subset):
                if task == 'math_500':
                    metric = 'acc'
                    value = int(row['index'] % 5 not in ((0,) if variant == 'baseline' else (0, 1)))
                    metadata = {}
                else:
                    metric = f"Context#{row['context_length']} Depth#{row['depth_percent']}"
                    value = int(row['index'] != 0 and (variant == 'baseline' or row['index'] != 8))
                    metadata = {'context_length': row['context_length'], 'depth_percent': row['depth_percent']}
                grouped.setdefault(metric, {}).setdefault(subset, []).append(value)
                predictions.append({'index': row['index'], 'model': 'Qwen3-14B',
                                    'model_output': {'error': None, 'choices': [{
                                        'stop_reason': 'stop', 'message': {'role': 'assistant', 'content': 'answer'}}]},
                                    'messages': [{'role': 'user', 'content': row['input']},
                                                 {'role': 'assistant', 'content': 'answer'}]})
                reviews.append({'index': row['index'], 'input': row['input'], 'target': row['target'],
                                'sample_score': {'sample_id': row['index'], 'sample_metadata': metadata,
                                                 'score': {'value': {metric: value}, 'prediction': 'answer',
                                                           'extracted_prediction': 'answer', 'metadata': {}}}})
            for kind, records in (('reviews', reviews), ('predictions', predictions)):
                path = output / variant / kind / 'Qwen3-14B' / f'{task}_{subset}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        metrics = [report_metric('mean_acc' if name == 'acc' else name, values)
                   for name, values in grouped.items()]
        write_json(output / variant / 'reports/Qwen3-14B' / f'{task}.json', {
            'dataset_name': task, 'model_name': 'Qwen3-14B', 'score': metrics[0]['score'], 'metrics': metrics})


def fixture_runs(inputs, *extra):
    _, output = prepared(inputs, *extra)
    rows = cases(output)
    native_run(output, rows, 'baseline')
    frozen = run(COMPARE, 'freeze', '--plan', output / 'plan.json')
    assert frozen.returncode == 0, frozen.stderr
    native_run(output, rows, 'candidate')
    return output


def compare(output):
    return run(COMPARE, 'compare', '--plan', output / 'plan.json', '--output', output / 'comparison.json')


def test_cli_prepares_native_isolated_configs_and_scripts(inputs):
    _, output = prepared(inputs)
    plan = json.loads((output / 'plan.json').read_text())
    baseline = json.loads((output / 'eval-baseline.json').read_text())
    candidate = json.loads((output / 'eval-candidate.json').read_text())
    assert baseline['use_cache'] is None and candidate['use_cache'] is None
    assert baseline.pop('work_dir') != candidate.pop('work_dir') and baseline == candidate
    assert baseline['judge_strategy'] == 'rule' and baseline['datasets'] == ['math_500', 'needle_haystack']
    generation = baseline['generation_config']
    assert generation['retries'] == 1 and generation['max_retries'] == baseline['model_args']['max_retries'] == 0
    assert 'seed' not in generation and generation['stream'] is False and generation['top_p'] == 1
    assert plan['accuracy_drop_gates'] == {'math_500': None, 'needle_haystack': None}
    assert plan['context_lengths'] == [1024, 7680, 14336]
    assert plan['context_lengths'][-1] + plan['chat_budget'] + plan['max_tokens'] < plan['max_model_length']
    for script in output.glob('*.sh'):
        subprocess.run(['bash', '-n', str(script)], check=True)
    assert run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'assets').returncode == 0


def test_preflight_uses_real_serving_tokenizer_and_chat_template(inputs):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    model = inputs[0] / 'model'
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'user': 1, 'hello': 2, 'assistant': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    tokenizer.chat_template = (
        '{% for message in messages %}{{ message["role"] }} {{ message["content"] }} {% endfor %}'
        '{% if add_generation_prompt %}assistant{% endif %}'
    )
    tokenizer.save_pretrained(model)
    _, output = prepared(inputs)
    messages = [[{'role': 'user', 'content': 'hello'}], [{'role': 'user', 'content': 'hello hello'}]]
    code = (
        'import json,sys; sys.path.insert(0, sys.argv[1]); import compare_fp8_eval as c; '
        'from pathlib import Path; root=Path(sys.argv[2]); '
        'print(json.dumps(c.server_token_counts(json.load(open(root / \"plan.json\")), root, json.load(sys.stdin))))'
    )
    result = subprocess.run([sys.executable, '-c', code, str(REPO / 'tools'), str(output)],
                            input=json.dumps(messages), text=True, capture_output=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    expected = [len(tokenizer.encode(tokenizer.apply_chat_template(
        item, tokenize=False, add_generation_prompt=True))) for item in messages]
    assert json.loads(result.stdout) == expected == [3, 4]


@pytest.mark.parametrize('change, message', [
    ('path', 'below workspace-root'), ('budget', 'strictly below'), ('temperature', 'temperature must be 0'),
    ('gate', 'accuracy-drop gates'), ('url', 'api-url'), ('revision', 'commit SHA'),
    ('math_info', 'evalscope would delete'), ('model', 'model snapshot does not match'),
])
def test_prepare_rejects_invalid_boundary_without_writing(inputs, change, message):
    root, output, args = inputs
    if change == 'path':
        args += ['--output-dir', str(root / '..' / 'outside')]
    elif change == 'budget':
        args += ['--context-max', '16384']
    elif change == 'temperature':
        args += ['--temperature', '0.6']
    elif change == 'gate':
        args += ['--math-gate', '1.1']
    elif change == 'url':
        args += ['--api-url', 'https://example.com/v1']
    elif change == 'revision':
        args += ['--candidate-commit', 'bad']
    elif change == 'math_info':
        (root / 'math/dataset_infos.json').write_text('{}')
    else:
        (root / 'model/config.json').write_text(json.dumps({'hidden_size': 1024}))
    result = run(*args)
    assert result.returncode != 0 and message in result.stderr
    assert not output.exists()


def test_output_symlink_cannot_escape_workspace(inputs):
    root, output, args = inputs
    (root / 'escape').symlink_to(root.parent, target_is_directory=True)
    result = run(*args, '--output-dir', root / 'escape/outside')
    assert result.returncode != 0 and 'below workspace-root' in result.stderr
    assert not output.exists() and not (root.parent / 'outside').exists()


def test_native_cli_capture_is_weighted_and_survives_log_append(inputs):
    output = fixture_runs(inputs)
    with (output / 'server-baseline.log').open('a') as handle:
        handle.write('server shutdown after baseline\n')
    assert run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'candidate').returncode == 0
    result = compare(output)
    assert result.returncode == 0, result.stderr
    data = json.loads((output / 'comparison.json').read_text())
    math = data['tasks']['math_500']
    needle = data['tasks']['needle_haystack']
    assert math['baseline_accuracy'] == pytest.approx(.8)
    assert math['candidate_accuracy'] == pytest.approx(.6)
    assert math['count'] == 500 and math['gate_passed'] is None and math['status'] == 'capture_only'
    assert needle['baseline_accuracy'] == pytest.approx(8 / 9)
    assert needle['candidate_accuracy'] == pytest.approx(7 / 9)
    assert needle['accuracy_drop'] == pytest.approx(1 / 9) and needle['count'] == 18
    assert len(needle['cells']) == 9
    assert needle['cells']['Context#1024 Depth#0']['baseline_accuracy'] == 0
    assert needle['cells']['Context#14336 Depth#100']['accuracy_drop'] == 1
    assert data['baseline']['termination_counts'] == {'stop': 518}


@pytest.mark.parametrize('gate, code, status', [('0.25', 0, 'pass'), ('0.1', 1, 'fail')])
def test_fixed_gate_cli_writes_report_and_uses_exit_status(inputs, gate, code, status):
    output = fixture_runs(inputs, '--math-gate', gate)
    result = compare(output)
    assert result.returncode == code, result.stderr
    data = json.loads((output / 'comparison.json').read_text())
    assert data['tasks']['math_500']['status'] == status
    assert data['tasks']['needle_haystack']['gate_passed'] is None


def mutate_jsonl(path, edit):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    edit(lines)
    path.write_text(''.join(json.dumps(line) + '\n' for line in lines))


@pytest.mark.parametrize('change, message', [
    ('input', 'input or target'), ('target', 'input or target'), ('index', 'unexpected, duplicate or unpaired'),
    ('missing_review', 'prediction and review counts'), ('prediction_error', 'failed or incomplete'),
    ('scoring_error', 'native scoring failed'), ('score_nan', 'invalid accuracy'),
    ('cell_name', 'unexpected native sample metric'), ('cell_metadata', 'metadata differs'),
    ('report_count', 'count disagrees'), ('report_score', 'score or count disagrees'),
    ('missing_metric', 'metrics do not match'), ('bad_server', 'server startup configuration'),
])
def test_native_cli_rejects_incomplete_mismatched_or_failed_runs(inputs, change, message):
    output = fixture_runs(inputs)
    needle = output / 'candidate/reviews/Qwen3-14B/needle_haystack_english.jsonl'
    math = output / 'candidate/reviews/Qwen3-14B/math_500_Level 1.jsonl'
    predictions = output / 'candidate/predictions/Qwen3-14B/math_500_Level 1.jsonl'
    if change == 'input':
        mutate_jsonl(math, lambda rows: rows[0].update(input='other prompt'))
    elif change == 'target':
        mutate_jsonl(math, lambda rows: rows[0].update(target='other target'))
    elif change == 'index':
        mutate_jsonl(math, lambda rows: rows[0].update(index=9999))
    elif change == 'missing_review':
        mutate_jsonl(math, lambda rows: rows.pop())
    elif change == 'prediction_error':
        mutate_jsonl(predictions, lambda rows: rows[0]['model_output'].update(error='HTTP error'))
    elif change == 'scoring_error':
        mutate_jsonl(math, lambda rows: rows[0]['sample_score']['score']['metadata'].update(acc='error: invalid'))
    elif change == 'score_nan':
        mutate_jsonl(math, lambda rows: rows[0]['sample_score']['score']['value'].update(acc=float('nan')))
    elif change == 'cell_name':
        mutate_jsonl(needle, lambda rows: rows[0]['sample_score']['score'].update(value={'acc': 1}))
    elif change == 'cell_metadata':
        mutate_jsonl(needle, lambda rows: rows[0]['sample_score']['sample_metadata'].update(depth_percent=99))
    elif change in ('report_count', 'report_score', 'missing_metric'):
        path = output / 'candidate/reports/Qwen3-14B/needle_haystack.json'
        report = json.loads(path.read_text())
        if change == 'report_count':
            report['metrics'][1]['categories'][0]['subsets'][0]['num'] = 99
        elif change == 'report_score':
            report['metrics'][1]['categories'][0]['subsets'][0]['score'] = .5
        else:
            report['metrics'].pop()
        write_json(path, report)
    else:
        (output / 'server-candidate.log').write_text('no startup evidence\n')
    result = compare(output)
    assert result.returncode != 0 and message in result.stderr
    assert not (output / 'comparison.json').exists()


def test_plan_assets_materialization_and_baseline_are_frozen(inputs):
    output = fixture_runs(inputs)
    math = output / 'baseline/reviews/Qwen3-14B/math_500_Level 1.jsonl'
    mutate_jsonl(math, lambda rows: rows[0]['sample_score']['score'].update(prediction='modified'))
    result = run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'candidate')
    assert result.returncode != 0 and 'prediction and review disagree' in result.stderr
    (inputs[0] / 'needle/PaulGraham_Essays.txt').write_text('changed corpus')
    result = run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode != 0 and 'asset files changed' in result.stderr


def test_changed_client_environment_blocks_the_candidate(inputs):
    output = fixture_runs(inputs)
    path = output / 'materialized.json'
    data = json.loads(path.read_text())
    data['client_environment']['packages'].append('changed-package==1')
    write_json(path, data)
    result = run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'candidate')
    assert result.returncode != 0 and 'client environment changed' in result.stderr


def test_rejects_context_overflow_and_tampered_prepared_config(inputs):
    _, output = prepared(inputs)
    rows = cases(output)
    rows[0]['prompt_tokens'] = 17000
    data = json.loads((output / 'materialized.json').read_text())
    data.update(input_sha256=fingerprint(rows), cases=rows)
    write_json(output / 'materialized.json', data)
    native_run(output, rows, 'baseline')
    result = run(COMPARE, 'freeze', '--plan', output / 'plan.json')
    assert result.returncode != 0 and 'context budget' in result.stderr
    config = output / 'eval-candidate.json'
    config.write_text(config.read_text().replace('"use_cache": null', '"use_cache": "wrong"'))
    result = run(COMPARE, 'check', '--plan', output / 'plan.json', '--stage', 'assets')
    assert result.returncode != 0 and 'prepared file changed' in result.stderr
