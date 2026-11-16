"""CPU bootstrap for the production capture process with real two-rank models."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

import myvllm.engine.llm_engine as engine_module
import myvllm.engine.model_runner as runner_module
import myvllm.layers.attention as attention
from test_paged_prefill import reference_prefill
from tools import compare_pd


_worker_process = engine_module.worker_process


@contextmanager
def cpu_backend():
    tensor = torch.tensor
    init_process_group = dist.init_process_group

    def cpu_tensor(*args, **kwargs):
        kwargs.pop('pin_memory', None)
        return tensor(*args, **kwargs)

    def init_gloo(backend, *args, **kwargs):
        assert backend == 'nccl'
        return init_process_group('gloo', *args, **kwargs)

    def allocate(runner):
        config = runner.config
        runner.kv_cache = torch.zeros(2, config['num_layers'], config['max_cached_blocks'],
                                      config['block_size'], config['num_kv_heads'] // runner.world_size,
                                      config['head_dim'], dtype=runner.kv_cache_dtype, device='cpu')
        runner.kv_cache_scales = None
        layer = 0
        for module in runner.model.modules():
            if isinstance(module, attention.Attention):
                module.k_cache = runner.kv_cache[0, layer]
                module.v_cache = runner.kv_cache[1, layer]
                layer += 1

    def store(k, v, kc, vc, slots, block_size):
        kc.flatten(0, 1)[slots] = k
        vc.flatten(0, 1)[slots] = v

    def decode(q, kc, vc, tables, lengths, scale, heads, kv_heads, dim, block_size):
        cu_q = torch.arange(len(q) + 1)
        cu_k = torch.cat([torch.zeros(1, dtype=lengths.dtype), lengths.cumsum(0)])
        return reference_prefill(q, kc, vc, cu_q, cu_k, tables,
                                 1, scale, heads, kv_heads, dim, block_size)

    with ExitStack() as stack:
        stack.enter_context(patch.object(dist, 'init_process_group', init_gloo))
        stack.enter_context(patch.object(torch.cuda, 'set_device'))
        stack.enter_context(patch.object(torch.cuda, 'synchronize'))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        stack.enter_context(patch.object(torch, 'set_default_device'))
        stack.enter_context(patch.object(torch, 'tensor', cpu_tensor))
        stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, **kwargs: self))
        stack.enter_context(patch.object(runner_module.Qwen3ForCausalLM, 'cuda', lambda self, device=None: self))
        stack.enter_context(patch.object(runner_module.ModelRunner, 'warmup_model'))
        stack.enter_context(patch.object(runner_module.ModelRunner, 'allocate_kv_cache', allocate))
        stack.enter_context(patch.object(attention, 'store_kvcache', store))
        stack.enter_context(patch.object(attention, 'paged_attention_prefill', reference_prefill))
        stack.enter_context(patch.object(attention, 'paged_attention_decode', decode))
        stack.enter_context(torch.compiler.set_stance('force_eager'))
        yield


def cpu_worker(config, rank, event):
    with cpu_backend():
        _worker_process(config, rank, event)


if __name__ == '__main__':
    with cpu_backend(), patch.object(engine_module, 'worker_process', cpu_worker):
        compare_pd.main()
