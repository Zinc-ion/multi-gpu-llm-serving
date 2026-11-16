"""Freeze and compare the pinned evalscope native FP8 quality artifacts."""

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
from urllib.parse import urlparse

from prepare_fp8_eval import asset_tree, json_text, serving_source, sha_file, tool_hashes


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def load_json(path):
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle)


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json_text(value))


def load_plan(path):
    path = Path(path).resolve()
    plan = load_json(path)
    root = path.parent
    if plan['version'] != 1 or plan['output_dir'] != str(root):
        raise ValueError('plan location or version does not match')
    repo = Path(plan['repo_dir'])
    if Path(__file__).resolve() != repo / 'tools/compare_fp8_eval.py' or tool_hashes(repo) != plan['tool_sha256']:
        raise ValueError('quality tools differ from the prepared versions')
    if serving_source(repo, plan['candidate_commit']) != plan['serving_source']:
        raise ValueError('serving Python source differs from the prepared version')
    for name, expected in plan['files_sha256'].items():
        if Path(name).name != name or sha_file(root / name) != expected:
            raise ValueError(f'prepared file changed: {name}')
    servers = [load_json(root / f'server-{variant}.json') for variant in ('baseline', 'candidate')]
    if [server.pop('kv_cache_dtype') for server in servers] != ['auto', 'float8_e4m3fn'] or servers[0] != servers[1]:
        raise ValueError('server configurations must differ only in KV format')
    tasks = [load_json(root / f'eval-{variant}.json') for variant in ('baseline', 'candidate')]
    work_dirs = [task.pop('work_dir') for task in tasks]
    aliases = [task.pop('model') for task in tasks]
    run_id = plan['run_id']
    if (not isinstance(run_id, str) or not re.fullmatch(r'[0-9a-f]{32}', run_id)
            or plan['api_model_aliases'] != {
                variant: f'Qwen3-14B-{run_id}-{variant}' for variant in ('baseline', 'candidate')}
            or aliases != [plan['api_model_aliases'][variant] for variant in ('baseline', 'candidate')]
            or work_dirs != [str(root / 'baseline'), str(root / 'candidate')] or tasks[0] != tasks[1]):
        raise ValueError('eval configurations must differ only in work directory and planned model alias')
    task = tasks[0]
    gen = task['generation_config']
    if (task['model_id'] != 'Qwen3-14B' or task['datasets'] != plan['tasks'] or task['use_cache'] is not None
            or task['ignore_errors'] or task['analysis_report'] or task['judge_strategy'] != 'rule'
            or not task['no_timestamp'] or task['model_args']['max_retries'] != 0
            or gen != {'stream': False, 'temperature': 0, 'top_p': 1, 'max_tokens': plan['max_tokens'],
                       'timeout': gen['timeout'], 'retries': 1, 'max_retries': 0}):
        raise ValueError('eval configuration violates the fixed quality contract')
    assets = load_json(root / 'assets.json')
    if (plan['model_revision_status'] != 'UNVERIFIED_USER_DECLARATION'
            or assets['model']['revision_status'] != plan['model_revision_status']
            or assets['model']['revision'] != plan['model_revision']):
        raise ValueError('model revision declaration does not match the asset manifest')
    for asset in assets.values():
        if asset_tree(asset['path']) != asset['files']:
            raise ValueError(f"asset files changed: {asset['path']}")
    return plan, root, sha_file(path)


def sample_key(row):
    return row['task'], row['subset'], row['index']


def case_identity(row):
    return {key: row[key] for key in ('task', 'subset', 'index', 'input', 'target')}


def fingerprint(rows):
    return digest([case_identity(row) for row in sorted(rows, key=sample_key)])


def client_environment():
    package_dirs = sorted({sysconfig.get_path('purelib'), sysconfig.get_path('platlib')})
    packages = sorted(f"{item.metadata['Name']}=={item.version}" for item in distributions(path=package_dirs))
    return {'python': sys.version, 'packages': packages}


def validate_cases(plan, rows):
    seen = set()
    math_subsets = {subset: 0 for subset in plan['math_subsets']}
    cells = {(subset, context, depth): 0 for subset in plan['needle_subsets']
             for context in plan['context_lengths'] for depth in plan['depths']}
    for row in rows:
        key = sample_key(row)
        if (key in seen or type(row['index']) is not int or row['index'] < 0
                or not isinstance(row['input'], str) or not row['input']
                or not isinstance(row['target'], str) or not row['target']):
            raise ValueError(f'duplicate or incomplete native input: {key}')
        seen.add(key)
        if row['task'] == 'math_500' and row['subset'] in math_subsets:
            math_subsets[row['subset']] += 1
        elif row['task'] == 'needle_haystack':
            cell = (row['subset'], row['context_length'], row['depth_percent'])
            if cell not in cells:
                raise ValueError(f'unplanned needle cell: {cell}')
            cells[cell] += 1
        else:
            raise ValueError(f'unplanned task or subset: {key}')
        if type(row['prompt_tokens']) is not int or row['prompt_tokens'] < 1:
            raise ValueError(f'missing actual prompt token count: {key}')
        if (not isinstance(row['messages'], list) or not row['messages']
                or any(set(message) != {'role', 'content'} or message['role'] not in ('system', 'user', 'assistant')
                       or not isinstance(message['content'], str) for message in row['messages'])):
            raise ValueError(f'invalid materialized chat messages: {key}')
        if row['prompt_tokens'] + plan['max_tokens'] >= plan['max_model_length']:
            raise ValueError(f'actual prompt and reply exceed the strict context budget: {key}')
    if sum(math_subsets.values()) != plan['math_count'] or not all(math_subsets.values()):
        raise ValueError(f'incomplete math_500: {math_subsets}')
    if any(count != 1 for count in cells.values()):
        raise ValueError(f'incomplete or duplicate needle grid: {cells}')


def materialized(plan, root, plan_sha):
    data = load_json(root / 'materialized.json')
    if data['plan_sha256'] != plan_sha or data['input_sha256'] != fingerprint(data['cases']):
        raise ValueError('materialized cases or plan changed')
    if data['client_environment'] != client_environment():
        raise ValueError('isolated client environment changed after materialization')
    validate_cases(plan, data['cases'])
    return data


def materialize(plan, root, plan_sha):
    from importlib.metadata import version
    from evalscope.api.evaluator import TaskState
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig
    from evalscope.utils.model_utils import seed_everything
    if version('evalscope') != '1.5.1':
        raise ValueError('materialization requires evalscope==1.5.1')
    config = TaskConfig.from_json(str(root / 'eval-baseline.json'))
    seed_everything(config.seed)
    rows = []
    for task in plan['tasks']:
        benchmark = get_benchmark(task, config)
        for subset, samples in benchmark.load_dataset().items():
            for sample in samples:
                state = TaskState(model=config.model, sample=sample)
                messages = [{'role': message.role, 'content': message.content} for message in sample.input]
                if any(not isinstance(message['content'], str) for message in messages):
                    raise ValueError('only text messages are supported by the serving API')
                row = {'task': task, 'subset': subset, 'index': state.sample_id,
                       'input': state.input_markdown, 'target': state.target, 'messages': messages}
                if task == 'needle_haystack':
                    row['context_length'] = sample.metadata['context_length']
                    row['depth_percent'] = sample.metadata['depth_percent']
                rows.append(row)
    counts = server_token_counts(plan, root, [row['messages'] for row in rows])
    for row, count in zip(rows, counts):
        row['prompt_tokens'] = count
    validate_cases(plan, rows)
    save_new(root / 'materialized.json', {'plan_sha256': plan_sha, 'input_sha256': fingerprint(rows),
                                         'client_environment': client_environment(), 'cases': rows})
    print(f'materialized {len(rows)} native cases')


def server_token_counts(plan, root, messages):
    code = """
import json, sys
from transformers import AutoTokenizer
from myvllm.server import tokenize_request
config = json.load(open(sys.argv[1], encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained(
    config['model_name_or_path'], revision=config['tokenizer_revision'],
    cache_dir=config['cache_dir'], local_files_only=True)
print(json.dumps([len(tokenize_request(tokenizer, {'messages': item}, True))
                  for item in json.load(sys.stdin)]))
"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = str(Path(plan['repo_dir']) / 'src')
    result = subprocess.run([plan['server_python'], '-P', '-c', code, str(root / 'server-baseline.json')],
                            input=json.dumps(messages, ensure_ascii=False), text=True, capture_output=True, env=env)
    if result.returncode:
        raise ValueError(f'serving tokenizer preflight failed: {result.stderr.strip()}')
    counts = json.loads(result.stdout)
    if len(counts) != len(messages):
        raise ValueError('serving tokenizer returned a different number of prompts')
    return counts


def check_server(root, variant):
    path = root / f'server-{variant}.log'
    matches = []
    for line in path.read_text(encoding='utf-8').splitlines():
        start = line.find('{')
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if event.get('event') == 'server_start':
            matches.append(event)
    if len(matches) != 1 or matches[0]['config'] != load_json(root / f'server-{variant}.json'):
        raise ValueError(f'{variant} server startup configuration is missing or mismatched')
    arguments = matches[0]['arguments']
    task = load_json(root / f'eval-{variant}.json')
    url = urlparse(task['api_url'])
    if (arguments['config'] != str(root / f'server-{variant}.json')
            or arguments['served_model_name'] != task['model']
            or arguments['host'] != url.hostname or arguments['port'] != url.port):
        raise ValueError(f'{variant} server startup arguments do not match')
    return digest(matches[0])


def read_jsonl(path):
    result = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    return result


def score_number(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'invalid accuracy: {value!r}')
    return float(value)


def prediction_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError('invalid native prediction content')
    fields = {'text': ('text', str), 'reasoning': ('reasoning', str), 'image': ('image', str),
              'audio': ('audio', str), 'video': ('video', str), 'data': ('data', dict)}
    texts = []
    for part in content:
        if not isinstance(part, dict) or part.get('type') not in fields:
            raise ValueError('invalid native prediction content')
        field, field_type = fields[part['type']]
        if not isinstance(part.get(field), field_type):
            raise ValueError('invalid native prediction content')
        if (part['type'] == 'audio' and part.get('format') not in ('wav', 'mp3')
                or part['type'] == 'video' and part.get('format') not in ('mp4', 'mpeg', 'mov')
                or part['type'] == 'image' and part.get('detail', 'auto') not in ('auto', 'low', 'high')):
            raise ValueError('invalid native prediction content')
        if part['type'] == 'text':
            texts.append(part['text'])
    return '\n'.join(texts)


def close_report(actual, expected):
    return abs(score_number(actual) - expected) <= 0.000051


def check_metric(metric, by_subset):
    if not isinstance(metric['categories'], list) or len(metric['categories']) != 1:
        raise ValueError('unexpected native report categories')
    category = metric['categories'][0]
    if category['name'] != ['default']:
        raise ValueError('unexpected native report category name')
    subsets = category['subsets']
    if len(subsets) != len(by_subset) or {row['name'] for row in subsets} != set(by_subset):
        raise ValueError('native report subsets do not match reviews')
    for subset in subsets:
        values = by_subset[subset['name']]
        if subset['num'] != len(values) or not close_report(subset['score'], sum(values) / len(values)):
            raise ValueError('native subset score or count disagrees with reviews')
    count = sum(item['num'] for item in subsets)
    weighted = sum(item['score'] * item['num'] for item in subsets) / count
    if (category['num'] != count or metric['num'] != count
            or not close_report(category['score'], weighted)
            or not close_report(metric['score'], category['score'])):
        raise ValueError('native report metric score or count disagrees with subsets')


def read_run(plan, root, variant, cases):
    work = root / variant
    alias = plan['api_model_aliases'][variant]
    expected = {sample_key(row): row for row in cases}
    grouped = {task: {} for task in plan['tasks']}
    native_hashes = {}
    terminations = {}
    actual_keys = set()
    for task in plan['tasks']:
        subsets = plan['math_subsets'] if task == 'math_500' else plan['needle_subsets']
        for subset in subsets:
            review_path = work / 'reviews/Qwen3-14B' / f'{task}_{subset}.jsonl'
            prediction_path = work / 'predictions/Qwen3-14B' / f'{task}_{subset}.jsonl'
            reviews, predictions = read_jsonl(review_path), read_jsonl(prediction_path)
            native_hashes[str(review_path.relative_to(root))] = sha_file(review_path)
            native_hashes[str(prediction_path.relative_to(root))] = sha_file(prediction_path)
            preds = {}
            for prediction in predictions:
                index = prediction['index']
                if type(index) is not int or index in preds:
                    raise ValueError('duplicate or invalid prediction index')
                output = prediction['model_output']
                choices = output['choices']
                if output['error'] is not None or len(choices) != 1:
                    raise ValueError('failed or incomplete native prediction')
                if prediction['model'] != alias or output['model'] != alias:
                    raise ValueError('native prediction model does not match the planned alias')
                reason = choices[0]['stop_reason']
                if reason not in ('stop', 'max_tokens'):
                    raise ValueError(f'failed or unexpected prediction stop reason: {reason}')
                content = choices[0]['message']['content']
                terminations[reason] = terminations.get(reason, 0) + 1
                preds[index] = prediction_text(content)
            review_ids = set()
            for review in reviews:
                index = review['index']
                key = (task, subset, index)
                if key not in expected or key in actual_keys or index not in preds:
                    raise ValueError(f'unexpected, duplicate or unpaired review: {key}')
                row = expected[key]
                if review['input'] != row['input'] or review['target'] != row['target']:
                    raise ValueError(f'native input or target differs from materialized case: {key}')
                sample_score = review['sample_score']
                score = sample_score['score']
                if sample_score['sample_id'] != index or score['prediction'] != preds[index]:
                    raise ValueError(f'native prediction and review disagree: {key}')
                if any(str(value).startswith('error:') for value in (score['metadata'] or {}).values()):
                    raise ValueError(f'native scoring failed: {key}')
                if task == 'math_500':
                    metric_name = 'acc'
                else:
                    metadata = sample_score['sample_metadata']
                    if (metadata['context_length'] != row['context_length']
                            or metadata['depth_percent'] != row['depth_percent']):
                        raise ValueError(f'needle review metadata differs from materialized case: {key}')
                    metric_name = f"Context#{row['context_length']} Depth#{row['depth_percent']}"
                if set(score['value']) != {metric_name}:
                    raise ValueError(f'unexpected native sample metric: {key}')
                value = score_number(score['value'][metric_name])
                if value not in (0, 1):
                    raise ValueError(f'native sample accuracy is not binary: {key}')
                grouped[task].setdefault(metric_name, {}).setdefault(subset, []).append(value)
                actual_keys.add(key)
                review_ids.add(index)
            if review_ids != set(preds):
                raise ValueError(f'prediction and review counts differ: {task}/{subset}')
    if actual_keys != set(expected):
        raise ValueError('run is incomplete or has different native case identities')
    results = {}
    for task in plan['tasks']:
        path = work / 'reports/Qwen3-14B' / f'{task}.json'
        report = load_json(path)
        native_hashes[str(path.relative_to(root))] = sha_file(path)
        if report['dataset_name'] != task or report['model_name'] != 'Qwen3-14B':
            raise ValueError('native report model or dataset mismatch')
        metrics = report['metrics']
        metric_map = {metric['name']: metric for metric in metrics}
        names = {'mean_acc'} if task == 'math_500' else set(grouped[task])
        if len(metric_map) != len(metrics) or set(metric_map) != names:
            raise ValueError(f'native report metrics do not match planned {task} metrics')
        if not close_report(report['score'], metrics[0]['score']):
            raise ValueError('native report first-metric score is inconsistent')
        for name, metric in metric_map.items():
            sample_metric = 'acc' if task == 'math_500' else name
            check_metric(metric, grouped[task][sample_metric])
        cell_results = {}
        all_scores = []
        for name, by_subset in grouped[task].items():
            values = [value for subset_values in by_subset.values() for value in subset_values]
            all_scores.extend(values)
            cell_results[name] = {'accuracy': sum(values) / len(values), 'count': len(values),
                                  'correct': int(sum(values))}
        results[task] = {'accuracy': sum(all_scores) / len(all_scores), 'count': len(all_scores),
                         'correct': int(sum(all_scores)),
                         'cells': cell_results if task == 'needle_haystack' else {}}
    return {'tasks': results, 'native_sha256': native_hashes, 'termination_counts': terminations,
            'server_start_sha256': check_server(root, variant)}


def freeze(plan, root, plan_sha):
    data = materialized(plan, root, plan_sha)
    baseline = read_run(plan, root, 'baseline', data['cases'])
    save_new(root / 'baseline-cases.json', {'plan_sha256': plan_sha,
             'materialized_sha256': sha_file(root / 'materialized.json'),
             'input_sha256': data['input_sha256'], 'cases': data['cases'], 'baseline': baseline})
    print('baseline native cases and evidence frozen')


def frozen_baseline(plan, root, plan_sha, data):
    frozen = load_json(root / 'baseline-cases.json')
    if (frozen['plan_sha256'] != plan_sha or frozen['materialized_sha256'] != sha_file(root / 'materialized.json')
            or frozen['input_sha256'] != fingerprint(data['cases']) or frozen['cases'] != data['cases']):
        raise ValueError('baseline case freeze does not match the materialized plan')
    baseline = read_run(plan, root, 'baseline', data['cases'])
    if baseline != frozen['baseline']:
        raise ValueError('baseline native evidence changed after freezing')
    return baseline


def compare(plan, root, plan_sha, output):
    output = Path(output).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError('comparison output must be below the prepared output directory')
    data = materialized(plan, root, plan_sha)
    baseline = frozen_baseline(plan, root, plan_sha, data)
    candidate = read_run(plan, root, 'candidate', data['cases'])
    tasks = {}
    failed = False
    for task in plan['tasks']:
        before, after = baseline['tasks'][task], candidate['tasks'][task]
        drop = Fraction(before['correct'], before['count']) - Fraction(after['correct'], after['count'])
        gate = plan['accuracy_drop_gates'][task]
        passed = None if gate is None else drop <= Fraction(Decimal(gate))
        failed |= passed is False
        cells = {}
        for name in before['cells']:
            old, new = before['cells'][name], after['cells'][name]
            if old['count'] != new['count']:
                raise ValueError(f'needle cell count mismatch: {name}')
            cell_drop = Fraction(old['correct'], old['count']) - Fraction(new['correct'], new['count'])
            cells[name] = {'baseline_accuracy': old['accuracy'], 'candidate_accuracy': new['accuracy'],
                           'accuracy_drop': float(cell_drop), 'count': old['count']}
        tasks[task] = {'baseline_accuracy': before['accuracy'], 'candidate_accuracy': after['accuracy'],
                       'accuracy_drop': float(drop), 'count': before['count'],
                       'gate_fraction': None if gate is None else float(Decimal(gate)), 'gate_fraction_exact': gate,
                       'gate_passed': passed, 'status': 'capture_only' if passed is None else ('pass' if passed else 'fail'),
                       'cells': cells}
    result = {'accuracy_unit': 'fraction', 'accuracy_drop_unit': 'fraction (multiply by 100 for percentage points)',
              'needle_total': 'count-weighted accuracy across all planned context/depth samples',
              'plan_sha256': plan_sha, 'materialized_sha256': sha_file(root / 'materialized.json'),
              'asset_manifest_sha256': sha_file(root / 'assets.json'),
              'model_revision': {'declared': plan['model_revision'], 'status': plan['model_revision_status']},
              'formal_m7_acceptance': None,
              'baseline_freeze_sha256': sha_file(root / 'baseline-cases.json'),
              'server_log_sha256': {variant: sha_file(root / f'server-{variant}.log')
                                    for variant in ('baseline', 'candidate')},
              'input_sha256': data['input_sha256'],
              'input_sha256_by_task': {task: fingerprint([row for row in data['cases'] if row['task'] == task])
                                       for task in plan['tasks']},
              'baseline': baseline, 'candidate': candidate, 'tasks': tasks}
    save_new(output, result)
    print(json_text(result), end='')
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('materialize', 'freeze', 'check', 'compare'):
        cmd = sub.add_parser(action)
        cmd.add_argument('--plan', required=True)
        if action == 'check':
            cmd.add_argument('--stage', required=True, choices=('assets', 'baseline', 'candidate'))
        if action == 'compare':
            cmd.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        plan, root, plan_sha = load_plan(args.plan)
        if args.action == 'materialize':
            materialize(plan, root, plan_sha)
        elif args.action == 'freeze':
            freeze(plan, root, plan_sha)
        elif args.action == 'check':
            if args.stage != 'assets':
                data = materialized(plan, root, plan_sha)
                if args.stage == 'candidate':
                    frozen_baseline(plan, root, plan_sha, data)
                check_server(root, args.stage)
            print('quality preparation checks passed')
        elif compare(plan, root, plan_sha, args.output):
            raise SystemExit(1)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
