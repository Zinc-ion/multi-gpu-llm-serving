import random

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tools.prepare_workloads import random_examples


def test_random_workload_has_distinct_exact_length_prompts_and_replays():
    vocab = {"[UNK]": 0, **{f"word{i}": i + 1 for i in range(64)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

    examples = random_examples(tokenizer, random.Random(7), 32, 32)
    assert examples == random_examples(tokenizer, random.Random(7), 32, 32)
    assert examples != random_examples(tokenizer, random.Random(8), 32, 32)
    token_ids = [tuple(tokenizer.encode(prompt)) for prompt, _ in examples]
    assert len(set(token_ids)) == 32
    assert all(len(ids) == 32 and 0 not in ids for ids in token_ids)
