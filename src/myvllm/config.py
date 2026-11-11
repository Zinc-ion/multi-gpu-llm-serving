"""Load the checked-in JSON configuration files used by the demo entrypoints."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from myvllm.sampling_parameters import SamplingParams


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def load_config(path: str | Path) -> dict:
    with Path(path).expanduser().open(encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")

    for field in ("model_revision", "tokenizer_revision"):
        revision = config[field]
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{field} must be a 40-character commit SHA")
    return config


def sampling_params_from_config(config: dict) -> SamplingParams:
    return SamplingParams(
        **config["sampling"],
        max_model_length=config["max_model_length"],
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
