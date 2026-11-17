"""Prepare an isolated, reproducible evalscope 1.5.1 FP8 quality comparison."""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import urlparse
import uuid


ROOT = Path('/Data2/zincion/ai-infra')
MODEL_REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
EVALSCOPE_COMMIT = 'e7c85f77e62329e584e3041dbf0883f33144334b'


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def asset_tree(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'asset directory does not exist: {path}')
    files = {str(file.relative_to(path)): sha_file(file)
             for file in sorted(path.rglob('*')) if file.is_file()}
    if not files:
        raise ValueError(f'asset directory is empty: {path}')
    return files


def serving_source(repo, commit):
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'cannot inspect candidate Git source: {exc.stderr.decode().strip()}') from exc

    if git('rev-parse', 'HEAD').decode().strip() != commit:
        raise ValueError('repository HEAD does not match candidate commit')
    algorithm = git('rev-parse', '--show-object-format').decode().strip()
    blobs = {}
    for entry in git('ls-tree', '-r', '-z', '--full-tree', commit, '--', 'src').split(b'\0'):
        if entry:
            metadata, path = entry.split(b'\t', 1)
            _, kind, blob = metadata.split(b' ')
            path = os.fsdecode(path)
            if path.endswith('.py') and kind == b'blob':
                blobs[path] = blob.decode()
    src = repo / 'src'
    if src.is_symlink():
        raise ValueError(f'serving source directory symlink is not supported: {src}')
    for path in src.rglob('*'):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f'serving source directory symlink is not supported: {path}')
    files = {str(path.relative_to(repo)): path for path in src.rglob('*.py') if path.is_file()}
    if not blobs or files.keys() != blobs.keys():
        raise ValueError('serving Python source file set differs from candidate commit')
    result = {}
    for name in sorted(files):
        content = files[name].read_bytes()
        blob = hashlib.new(algorithm, f'blob {len(content)}\0'.encode() + content).hexdigest()
        if blob != blobs[name]:
            raise ValueError(f'serving Python source differs from candidate Git blob: {name}')
        result[name] = {'git_blob': blob, 'sha256': hashlib.sha256(content).hexdigest()}
    return result


def tool_hashes(repo):
    return {name: sha_file(repo / 'tools' / name)
            for name in ('prepare_fp8_eval.py', 'compare_fp8_eval.py')}


def command(args):
    return ' '.join(shlex.quote(str(arg)) for arg in args)


def generate(args):
    root = Path(args.workspace_root).resolve()
    output = Path(args.output_dir).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError('output-dir must be a new directory below workspace-root')
    if output.exists():
        raise ValueError('output-dir already exists')
    if not re.fullmatch(r'[0-9a-f]{40}', args.candidate_commit):
        raise ValueError('candidate-commit must be a full 40-character commit SHA')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be a 32-bit unsigned integer')
    task_max_tokens = {
        'math_500': args.max_tokens if args.math_max_tokens is None else args.math_max_tokens,
        'needle_haystack': args.max_tokens if args.needle_max_tokens is None else args.needle_max_tokens,
    }
    separate_tasks = args.math_max_tokens is not None or args.needle_max_tokens is not None
    if (args.batch_size < 1 or args.max_tokens < 1 or min(task_max_tokens.values()) < 1 or args.timeout <= 0
            or not math.isfinite(args.timeout) or not math.isfinite(args.temperature)
            or args.temperature != 0):
        raise ValueError('batch-size, max-tokens and timeout must be positive; temperature must be 0')
    for gate in (args.math_gate, args.needle_gate):
        if gate is not None:
            try:
                value = Decimal(gate)
            except InvalidOperation as exc:
                raise ValueError('accuracy-drop gates must be fractions between 0 and 1') from exc
            if not value.is_finite() or not 0 <= value <= 1:
                raise ValueError('accuracy-drop gates must be fractions between 0 and 1')
    url = urlparse(args.api_url)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path != '/v1' or not url.port:
        raise ValueError('api-url must be http://127.0.0.1:PORT/v1')
    if not args.math_revision or not args.needle_revision:
        raise ValueError('both corpus revisions must be supplied')
    for value in (args.repo_dir, args.server_python, args.model_path, args.math_data, args.needle_data):
        if not Path(value).is_absolute():
            raise ValueError('repo, python, model and corpus paths must be absolute')

    config = json.loads(Path(args.base_config).read_text(encoding='utf-8'))
    if (config['model_type'] != 'Qwen3-14B' or config['world_size'] != 1
            or config['dtype'] != 'bfloat16' or config['model_revision'] != MODEL_REVISION
            or config['tokenizer_revision'] != MODEL_REVISION):
        raise ValueError('base config must be the pinned TP=1 Qwen3-14B BF16 configuration')
    if args.context_max <= 1024 or (args.context_max - 1024) % 2:
        raise ValueError('context-max must be greater than 1024 with an even difference')
    if (args.chat_budget < 1
            or args.context_max + args.chat_budget + task_max_tokens['needle_haystack'] >= config['max_model_length']):
        raise ValueError('context, chat budget and reply must fit strictly below max_model_length')
    if task_max_tokens['math_500'] + 1 >= config['max_model_length']:
        raise ValueError('math reply must leave room below max_model_length for a prompt')
    model, math_data, needle = map(lambda p: Path(p).resolve(),
                                   (args.model_path, args.math_data, args.needle_data))
    if (math_data / 'dataset_infos.json').exists():
        raise ValueError('math corpus contains dataset_infos.json, which evalscope would delete')
    model_config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    for model_field, server_field in (('hidden_size', 'hidden_size'), ('num_hidden_layers', 'num_layers'),
                                      ('num_attention_heads', 'num_heads'), ('num_key_value_heads', 'num_kv_heads')):
        if model_config[model_field] != config[server_field]:
            raise ValueError(f'model snapshot does not match 14B config: {model_field}')
    for name in ('PaulGraham_Essays.txt', 'Journey_to_the_West.txt'):
        if not (needle / name).is_file():
            raise ValueError(f'needle corpus is missing {name}')
    assets = {
        'model': {'path': str(model), 'revision': MODEL_REVISION,
                  'revision_status': 'UNVERIFIED_USER_DECLARATION', 'files': asset_tree(model)},
        'math_500': {'path': str(math_data), 'revision': args.math_revision, 'files': asset_tree(math_data)},
        'needle_haystack': {'path': str(needle), 'revision': args.needle_revision, 'files': asset_tree(needle)},
    }
    config['model_name_or_path'] = str(model)
    config['cache_dir'] = str(output / 'cache/hf-hub')
    config.pop('timing_path', None)
    repo = Path(args.repo_dir).resolve()
    if Path(__file__).resolve() != repo / 'tools/prepare_fp8_eval.py':
        raise ValueError('prepare tool must run from the candidate repository')
    source = serving_source(repo, args.candidate_commit)
    tools = tool_hashes(repo)
    comparison = repo / 'tools/compare_fp8_eval.py'
    env = '#!/usr/bin/env bash\nset -euo pipefail\n'
    exports = {
        'PYTHONPYCACHEPREFIX': output / 'cache/pycache', 'UV_CACHE_DIR': output / 'cache/uv',
        'UV_PYTHON_INSTALL_DIR': output / 'cache/uv-python', 'PIP_CACHE_DIR': output / 'cache/pip',
        'PYTHONUSERBASE': output / 'cache/python-user', 'MPLCONFIGDIR': output / 'cache/matplotlib',
        'XDG_CACHE_HOME': output / 'cache/xdg', 'HF_HOME': output / 'cache/huggingface',
        'XDG_CONFIG_HOME': output / 'cache/xdg-config', 'XDG_DATA_HOME': output / 'cache/xdg-data',
        'HF_HUB_CACHE': output / 'cache/hf-hub', 'HF_DATASETS_CACHE': output / 'cache/hf-datasets',
        'HF_MODULES_CACHE': output / 'cache/hf-modules', 'HF_ASSETS_CACHE': output / 'cache/hf-assets',
        'MODELSCOPE_CACHE': output / 'cache/modelscope', 'EVALSCOPE_CACHE': output / 'cache/evalscope',
        'TORCHINDUCTOR_CACHE_DIR': output / 'cache/torchinductor', 'TRITON_CACHE_DIR': output / 'cache/triton',
        'CUDA_CACHE_PATH': output / 'cache/cuda', 'NUMBA_CACHE_DIR': output / 'cache/numba',
        'TMPDIR': output / 'tmp',
    }
    env += ''.join(f'export {name}={shlex.quote(str(path))}\n' for name, path in exports.items())
    env += 'export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
    env += 'mkdir -p ' + ' '.join(shlex.quote(str(path)) for path in exports.values()) + '\n'
    files = {'assets.json': json_text(assets), 'env.sh': env}
    prefix = '#!/usr/bin/env bash\nset -euo pipefail\n'
    prefix += f'source {shlex.quote(str(output / "env.sh"))}\n'
    prefix += f'cd {shlex.quote(str(repo))}\n'
    prefix += f'test "$(git rev-parse HEAD)" = {shlex.quote(args.candidate_commit)}\n'
    plan_path = output / 'plan.json'
    client = output / 'client/bin/python'
    run_id = uuid.uuid4().hex
    aliases = {variant: f'Qwen3-14B-{run_id}-{variant}' for variant in ('baseline', 'candidate')}
    for variant, kv in (('baseline', 'auto'), ('candidate', 'float8_e4m3fn')):
        server_config = dict(config, kv_cache_dtype=kv)
        files[f'server-{variant}.json'] = json_text(server_config)
        task = {
            'model': aliases[variant], 'model_id': 'Qwen3-14B', 'api_url': args.api_url, 'api_key': 'EMPTY',
            'model_args': {'max_retries': 0}, 'datasets': ['math_500', 'needle_haystack'],
            'dataset_args': {
                'math_500': {'local_path': str(math_data), 'shuffle': False},
                'needle_haystack': {'local_path': str(needle), 'subset_list': ['english', 'chinese'],
                                    'shuffle': False, 'extra_params': {
                    'tokenizer_path': str(model), 'context_lengths_min': 1024,
                    'context_lengths_max': args.context_max, 'context_lengths_num_intervals': 3,
                    'document_depth_percent_min': 0, 'document_depth_percent_max': 100,
                    'document_depth_percent_intervals': 3,
                }},
            },
            'dataset_dir': str(output / 'dataset-cache'), 'work_dir': str(output / variant),
            'no_timestamp': True, 'use_cache': None, 'seed': args.seed, 'eval_batch_size': args.batch_size,
            'ignore_errors': False, 'analysis_report': False, 'judge_strategy': 'rule',
            'generation_config': {'stream': False, 'temperature': args.temperature, 'top_p': 1,
                                  'max_tokens': args.max_tokens, 'timeout': args.timeout,
                                  'retries': 1, 'max_retries': 0},
        }
        if separate_tasks:
            for name, max_tokens in task_max_tokens.items():
                task_config = dict(task, datasets=[name], dataset_args={name: task['dataset_args'][name]},
                                   work_dir=str(output / variant / name),
                                   generation_config=dict(task['generation_config'], max_tokens=max_tokens))
                files[f'eval-{variant}-{name}.json'] = json_text(task_config)
        else:
            files[f'eval-{variant}.json'] = json_text(task)
        validate = command([client, comparison, 'check', '--plan', plan_path, '--stage', variant])
        server_cmd = [args.server_python, '-P', '-m', 'myvllm.server', '--config', output / f'server-{variant}.json',
                      '--served-model-name', aliases[variant], '--host', '127.0.0.1', '--port', url.port]
        server = prefix + f'test ! -e {shlex.quote(str(output / f"server-{variant}.log"))}\n'
        server += "PYTHONPATH='' " + command([client, comparison, 'check', '--plan', plan_path, '--stage', 'assets']) + '\n'
        server += f'export PYTHONPATH={shlex.quote(str(repo / "src"))}\nexec ' + command(server_cmd)
        server += f' > {shlex.quote(str(output / f"server-{variant}.log"))} 2>&1\n'
        files[f'server-{variant}.sh'] = server
        run_code = ('from importlib.metadata import version; '
                    'assert version("evalscope") == "1.5.1"; '
                    'from evalscope import run_task; ')
        if separate_tasks:
            paths = [str(output / f'eval-{variant}-{name}.json') for name in task_max_tokens]
            run_code += ('from evalscope.config import TaskConfig; '
                         f'run_task(task_cfg=[TaskConfig.from_json(path) for path in {paths!r}])')
        else:
            run_code += f'run_task(task_cfg={str(output / f"eval-{variant}.json")!r})'
        run = prefix + "export CUDA_VISIBLE_DEVICES='' PYTHONPATH=''\n" + validate + '\n'
        run += f'test ! -e {shlex.quote(str(output / variant))}\n'
        run += command([client, '-c', run_code]) + f' > {shlex.quote(str(output / f"eval-{variant}.log"))} 2>&1\n'
        run += command([client, comparison, 'check', '--plan', plan_path, '--stage', 'assets']) + '\n'
        if variant == 'baseline':
            run += command([client, comparison, 'freeze', '--plan', plan_path]) + '\n'
        files[f'evaluate-{variant}.sh'] = run
    materialize = prefix + "export CUDA_VISIBLE_DEVICES='' PYTHONPATH=''\n"
    materialize += command([client, comparison, 'materialize', '--plan', plan_path]) + '\n'
    files['materialize.sh'] = materialize
    files['compare.sh'] = prefix + "export CUDA_VISIBLE_DEVICES='' PYTHONPATH=''\n" + command(
        [client, comparison, 'compare', '--plan', plan_path, '--output', output / 'comparison.json']) + '\n'
    plan = {
        'version': 2 if separate_tasks else 1, 'evalscope_version': '1.5.1', 'evalscope_commit': EVALSCOPE_COMMIT,
        'candidate_commit': args.candidate_commit, 'output_dir': str(output), 'model_revision': MODEL_REVISION,
        'model_revision_status': 'UNVERIFIED_USER_DECLARATION', 'run_id': run_id, 'api_model_aliases': aliases,
        'repo_dir': str(repo), 'server_python': args.server_python,
        'serving_source': source, 'tool_sha256': tools,
        'tasks': ['math_500', 'needle_haystack'], 'math_count': 500,
        'math_subsets': [f'Level {i}' for i in range(1, 6)],
        'needle_subsets': ['english', 'chinese'],
        'context_lengths': [1024, (1024 + args.context_max) // 2, args.context_max],
        'depths': [0, 50, 100], 'max_model_length': config['max_model_length'],
        'chat_budget': args.chat_budget, 'max_tokens': args.max_tokens, 'task_max_tokens': task_max_tokens,
        'accuracy_drop_gates': {'math_500': args.math_gate, 'needle_haystack': args.needle_gate},
        'files_sha256': {name: sha_text(text) for name, text in files.items()},
    }
    files['plan.json'] = json_text(plan)
    return output, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', default=str(ROOT), help='Server workspace root; local tests may use a temporary analogue')
    for name in ('output-dir', 'repo-dir', 'server-python', 'model-path', 'math-data', 'needle-data',
                 'math-revision', 'needle-revision', 'candidate-commit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--base-config', default='configs/qwen3_14b.json')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--math-max-tokens', type=int, help='Math output budget; defaults to --max-tokens')
    parser.add_argument('--needle-max-tokens', type=int, help='Needle output budget; defaults to --max-tokens')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--context-max', type=int, default=14336)
    parser.add_argument('--chat-budget', type=int, default=1024)
    parser.add_argument('--math-gate', help='Maximum accuracy drop as a decimal fraction; omit for capture-only')
    parser.add_argument('--needle-gate', help='Maximum accuracy drop as a decimal fraction; omit for capture-only')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        output, files = generate(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json_text({'output_dir': str(output), 'files': files}), end='')
        return
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (output / name).write_text(content, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
