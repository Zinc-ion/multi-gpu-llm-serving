import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


def test_step_counts_input_tokens_before_appending_completion():
    class CPUModelRunner:
        timings = None

        def call(self, method, sequences, is_prefill):
            return torch.tensor([99 for _ in sequences])

    # Use the real step and scheduler; replace only GPU execution.
    engine = LLMEngine.__new__(LLMEngine)
    engine.model_runner = CPUModelRunner()
    engine.scheduler = Scheduler(4, 16, 4, 4, [0], 16)
    seq = Sequence([1, 2, 3], 4, SamplingParams(max_tokens=2))
    engine.scheduler.add_sequence(seq)

    _, processed, is_prefill = engine.step()
    assert is_prefill
    assert processed == 3
    assert seq.completion_token_ids == [99]
    _, processed, is_prefill = engine.step()
    assert not is_prefill
    assert processed == 1
    assert engine.scheduler.is_finished()
