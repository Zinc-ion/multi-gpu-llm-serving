import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.config import (
    load_config,
    resolve_dtype,
    sampling_params_from_config,
    seed_everything,
)


ROOT = Path(__file__).parents[1]


def test_checked_in_configs_are_pinned_and_loadable():
    for path in (
        ROOT / "configs" / "qwen3_0_6b.json",
        ROOT / "configs" / "llama3_2_1b.json",
    ):
        config = load_config(path)
        assert isinstance(config, dict)
        assert len(config["model_revision"]) == 40
        assert config["model_revision"] == config["tokenizer_revision"]
        assert resolve_dtype(config["dtype"]) == torch.float32
        assert config["seed"] == 0
        assert "warmup_tokens" not in config
        assert config["sampling"]["temperature"] > 0
        assert config["prompts"]
        assert (
            sampling_params_from_config(config).max_model_length
            == config["max_model_length"]
        )


def test_seed_everything_reproduces_python_numpy_and_torch_streams():
    seed_everything(1234)
    first = (np.random.rand(), torch.rand(3))
    seed_everything(1234)
    second = (np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_config_loader_rejects_unpinned_revision(tmp_path):
    source = ROOT / "configs" / "qwen3_0_6b.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["model_revision"] = "main"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    try:
        load_config(path)
    except ValueError as exc:
        assert "40-character commit SHA" in str(exc)
    else:
        raise AssertionError("unpinned model revision was accepted")


