import json

from vllm.benchmarks.datasets import CustomDataset

from test_server import tokenizer


def test_locked_custom_dataset_preserves_text_order_and_each_output_budget(tmp_path, tokenizer):
    requests = [
        {"prompt": "中文\u2028😀\nhello", "output_tokens": 4},
        {"prompt": "a different prompt", "output_tokens": 7},
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests))
    dataset = CustomDataset(dataset_path=str(path), disable_shuffle=True, random_seed=7)
    samples = dataset.sample(
        tokenizer=tokenizer, num_requests=2, skip_chat_template=True, no_oversample=True,
    )
    assert [sample.prompt for sample in samples] == [row["prompt"] for row in requests]
    assert [sample.expected_output_len for sample in samples] == [4, 7]
    assert [sample.prompt_len for sample in samples] == [len(tokenizer.encode(row["prompt"])) for row in requests]
