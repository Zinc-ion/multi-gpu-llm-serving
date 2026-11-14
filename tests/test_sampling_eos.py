import pytest
import torch

from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.layers.sampler import SamplerLayer
from myvllm.sampling_parameters import SamplingParams


def test_greedy_and_positive_temperature_can_share_a_batch():
    SamplingParams(temperature=0)
    logits = torch.tensor([[2., 8., -1.], [-1., 3., 6.]])
    with torch.compiler.set_stance("force_eager"):
        for seed in range(8):
            torch.manual_seed(seed)
            tokens = SamplerLayer()(logits.clone(), torch.tensor([0., 0.7]))
            assert tokens[0].item() == 1
            assert 0 <= tokens[1].item() < 3


def test_positive_temperature_keeps_original_sampling():
    logits = torch.tensor([[1., 2., 3.], [5., 3., 2.]])
    temperature = torch.tensor([0.6, 1.2])
    torch.manual_seed(3)
    probs = torch.softmax(logits / temperature[:, None], dim=-1)
    expected = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(-1)
    torch.manual_seed(3)
    with torch.compiler.set_stance("force_eager"):
        actual = SamplerLayer()(logits.clone(), temperature)
    assert torch.equal(actual, expected)


def test_negative_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-0.1)


@pytest.mark.parametrize("eos", [10, 11])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_each_eos_stops_and_releases_real_blocks(eos, ignore_eos):
    scheduler = Scheduler(2, 16, 4, 4, [10, 11], 16)
    seq = Sequence([1, 2, 3], 4, SamplingParams(ignore_eos=ignore_eos))
    scheduler.add_sequence(seq)
    batch, _ = scheduler.schedule()
    scheduler.postprocess(batch, [eos])
    assert seq.is_finished == (not ignore_eos)
    if ignore_eos:
        scheduler.cancel(seq.seq_id)
    assert not scheduler.block_manager.used_block_ids
    for block in scheduler.block_manager.blocks:
        assert block.ref_count == 0 and block.hash == -1
