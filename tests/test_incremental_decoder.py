import random

import pytest
from transformers import PreTrainedTokenizerFast

from myvllm.server import IncrementalDecoder
from test_server import tokenizer


@pytest.mark.parametrize("text", ["中文没有空格也应逐步输出" * 20, "longword" * 100], ids=["chinese", "long_word"])
def test_text_is_visible_before_completion(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoder = IncrementalDecoder(tokenizer)
    chunks = [decoder.push(token) for token in ids[:-1]]
    assert any(chunks), "text was held until the final token"
    chunks.append(decoder.push(ids[-1], finished=True))
    assert "".join(chunks) == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def test_decoding_work_does_not_grow_with_a_long_line(tokenizer, monkeypatch):
    ids = tokenizer.encode("abcd" * 1024, add_special_tokens=False)
    decode = PreTrainedTokenizerFast.decode
    lengths = []

    def observed_decode(self, token_ids, *args, **kwargs):
        lengths.append(len(token_ids))
        return decode(self, token_ids, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "decode", observed_decode)
    decoder = IncrementalDecoder(tokenizer)
    chunks = [decoder.push(token, i == len(ids) - 1) for i, token in enumerate(ids)]
    assert "".join(chunks) == "abcd" * 1024
    assert max(lengths) <= 8
    assert sum(lengths) <= 8 * len(ids)


def test_split_bytes_special_tokens_and_unfinished_final_character(tokenizer):
    ids = tokenizer.encode("👩‍💻中文𠮷\ncafé�", add_special_tokens=False)
    ids.insert(2, tokenizer.eos_token_id)
    # ByteLevel's byte 0xF0 starts a four-byte character, incomplete at completion.
    ids.append(tokenizer.convert_tokens_to_ids("ð"))
    decoder = IncrementalDecoder(tokenizer)
    text = "".join(decoder.push(token, i == len(ids) - 1) for i, token in enumerate(ids))
    assert text == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert text.endswith("�")


def test_arbitrary_byte_tokens_match_whole_decode(tokenizer):
    rng = random.Random(7)
    vocab_size = len(tokenizer)
    for _ in range(100):
        ids = [rng.randrange(vocab_size) for _ in range(32)]
        decoder = IncrementalDecoder(tokenizer)
        actual = "".join(decoder.push(token, i == len(ids) - 1) for i, token in enumerate(ids))
        assert actual == tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
