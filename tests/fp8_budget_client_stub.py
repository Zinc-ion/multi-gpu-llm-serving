"""Small evalscope substitute for exercising generated scripts over CPU HTTP."""

import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from urllib.request import Request, urlopen

from test_fp8_eval_tools import report_metric


class TaskConfig:
    @classmethod
    def from_json(cls, path):
        config = cls()
        config.__dict__.update(json.loads(Path(path).read_text()))
        return config


class TaskState:
    def __init__(self, model, sample):
        self.sample_id = sample.index
        self.input_markdown = sample.input_markdown
        self.target = sample.target


def seed_everything(seed):
    random.seed(seed)


def get_benchmark(task, config):
    assert config.datasets == [task]
    assert set(config.dataset_args) == {task}

    class Benchmark:
        def load_dataset(self):
            groups = {}
            for row in json.loads(Path(os.environ['FP8_BUDGET_SAMPLES']).read_text()):
                if row['task'] != task:
                    continue
                sample = SimpleNamespace(
                    index=row['index'], input_markdown=row['input'], target=row['target'],
                    input=[SimpleNamespace(**message) for message in row['messages']],
                    metadata={key: row[key] for key in ('context_length', 'depth_percent') if key in row},
                )
                groups.setdefault(row['subset'], []).append(sample)
            return groups

    return Benchmark()


def run_task(task_cfg):
    assert isinstance(task_cfg, list)
    for config in task_cfg:
        assert isinstance(config, TaskConfig)
        seed_everything(config.seed)
        task = config.datasets[0]
        work = Path(config.work_dir)
        metrics = {}
        for subset, samples in get_benchmark(task, config).load_dataset().items():
            predictions = []
            reviews = []
            for sample in samples:
                messages = [vars(message) for message in sample.input]
                body = {'model': config.model, 'messages': messages,
                        'max_tokens': config.generation_config['max_tokens'],
                        'temperature': config.generation_config['temperature'],
                        'top_p': config.generation_config['top_p'], 'stream': config.generation_config['stream']}
                request = Request(config.api_url + '/chat/completions',
                                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=10) as response:
                    answer = json.load(response)
                usage = answer['usage']
                text = answer['choices'][0]['message']['content']
                predictions.append({'index': sample.index, 'model': config.model,
                                    'model_output': {'model': answer['model'], 'error': None,
                                                     'usage': {'input_tokens': usage['prompt_tokens'],
                                                               'output_tokens': usage['completion_tokens']},
                                                     'choices': [{'stop_reason': 'stop',
                                                                  'message': {'content': text}}]}})
                metric = ('acc' if task == 'math_500' else
                          f"Context#{sample.metadata['context_length']} Depth#{sample.metadata['depth_percent']}")
                metrics.setdefault(metric, {}).setdefault(subset, []).append(1)
                reviews.append({'index': sample.index, 'input': sample.input_markdown, 'target': sample.target,
                                'sample_score': {'sample_id': sample.index, 'sample_metadata': sample.metadata,
                                                 'score': {'value': {metric: 1}, 'prediction': text,
                                                           'metadata': {}}}})
            for kind, records in (('predictions', predictions), ('reviews', reviews)):
                path = work / kind / 'Qwen3-14B' / f'{task}_{subset}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        reports = [report_metric('mean_acc' if name == 'acc' else name, groups)
                   for name, groups in metrics.items()]
        path = work / 'reports/Qwen3-14B' / f'{task}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'dataset_name': task, 'model_name': 'Qwen3-14B',
                                    'score': reports[0]['score'], 'metrics': reports}))
