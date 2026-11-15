from contextlib import contextmanager, nullcontext

import pytest
import torch

from myvllm.engine.model_runner import ModelRunner
from myvllm.utils import get_context, reset_context, set_context


@pytest.fixture
def cpu_cuda_backend(monkeypatch):
    """Execute capture/replay buffer logic on CPU; this does not test CUDA Graphs."""
    for name in ("zeros", "ones", "full", "empty"):
        factory = getattr(torch, name)

        def cpu_factory(*args, _factory=factory, **kwargs):
            if str(kwargs.get("device", "")).startswith("cuda"):
                kwargs["device"] = "cpu"
            return _factory(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_factory)

    class CPUStream:
        def wait_stream(self, other):
            pass

    class CPUModel:
        def __init__(self, width):
            self.width = width
            self.cache = torch.full((64,), -1, dtype=torch.long)

        def __call__(self, input_ids):
            slots = get_context().slot_mapping
            active = slots >= 0
            self.cache[slots[active]] = input_ids[active]
            return input_ids[:, None].float() + torch.arange(self.width).float()[None, :]

        def compute_logits(self, hidden):
            return hidden

    class CPUGraph:
        def pool(self):
            return 0

        def replay(self):
            previous = get_context()
            set_context(**vars(self.context))
            try:
                variables = self.runner.graph_vars
                batch_size = len(self.context.slot_mapping)
                self.runner.replayed_batch_size = batch_size
                variables["outputs"][:batch_size].copy_(self.runner.model(variables["input_ids"][:batch_size]))
            finally:
                set_context(**vars(previous))

    @contextmanager
    def capture(graph, pool=None, **kwargs):
        graph.context = get_context()
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", CPUGraph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: CPUStream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **kwargs: CPUStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    def make_runner(max_bs=8, hidden=5, vocab=5, budget=64, blocks=64):
        runner = ModelRunner.__new__(ModelRunner)
        runner.rank, runner.block_size, runner.enforce_eager = 0, 4, False
        runner.default_dtype = torch.float32
        runner.config = {
            "max_num_sequences": max_bs, "max_model_length": 16,
            "max_num_batched_tokens": budget, "max_cached_blocks": blocks,
            "hidden_size": hidden, "vocab_size": vocab,
        }
        runner.model = CPUModel(hidden)
        runner.capture_cudagraph()
        for graph in runner.graphs.values():
            graph.runner = runner
        return runner

    try:
        yield make_runner
    finally:
        reset_context()


def decode(runner, tokens, slots, lengths, tables):
    set_context(
        False, slot_mapping=torch.tensor(slots), context_lens=torch.tensor(lengths),
        block_tables=torch.tensor(tables, dtype=torch.int32),
        positions=torch.tensor(lengths) - 1,
    )
    return runner.run_model(torch.tensor(tokens), False)


def test_capture_stores_hidden_states_not_vocab_logits(cpu_cuda_backend):
    runner = cpu_cuda_backend(hidden=7, vocab=11)
    assert runner.graph_vars["outputs"].shape == (8, 7)


@pytest.mark.parametrize("max_bs", [1, 3, 7, 18])
def test_capture_sizes_are_bounded_and_include_the_limit(cpu_cuda_backend, max_bs):
    runner = cpu_cuda_backend(max_bs=max_bs)
    assert max(runner.graphs) == max_bs
    assert min(runner.graphs) == 1
    assert all(size <= max_bs for size in runner.graphs)


def test_replay_chooses_the_smallest_fitting_capture(cpu_cuda_backend):
    runner = cpu_cuda_backend()
    actual = decode(runner, [4, 5, 6], [7, 9, 12], [2, 4, 7], [[0, -1], [1, -1], [2, 3]])
    assert runner.replayed_batch_size == 4
    assert actual.shape == (3, 5)
    assert actual[:, 0].tolist() == [4, 5, 6]


@pytest.mark.parametrize("budget,blocks,limit", [(3, 64, 3), (64, 5, 5)])
def test_capture_limit_respects_token_budget_and_profiled_blocks(cpu_cuda_backend, budget, blocks, limit):
    runner = cpu_cuda_backend(max_bs=20, budget=budget, blocks=blocks)
    assert max(runner.graphs) == limit
    assert runner.graph_vars["input_ids"].shape == (limit,)


def test_padded_replay_does_not_write_live_slots_or_retain_old_tables(cpu_cuda_backend):
    runner = cpu_cuda_backend()
    runner.model.cache.fill_(-1)
    decode(runner, [4, 5, 6], [7, 9, 12], [2, 4, 7], [[0, -1], [1, -1], [2, 3]])
    decode(runner, [8], [13], [1], [[4]])
    variables = runner.graph_vars
    assert torch.all(variables["slot_mapping"][1:] == -1)
    assert torch.all(variables["context_lens"][1:] == 0)
    assert torch.all(variables["block_tables"][1:] == -1)
    assert torch.all(variables["block_tables"][0, 1:] == -1)
    assert variables["positions"][0] == 0
    assert torch.all(variables["positions"][1:] == 0)
    assert runner.model.cache[0] == -1
    assert runner.model.cache[7] == 4 and runner.model.cache[13] == 8
