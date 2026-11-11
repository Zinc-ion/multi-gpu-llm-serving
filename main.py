import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams
from myvllm.config import load_config, sampling_params_from_config
import argparse

DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'qwen3_0_6b.json'

def main(config_path=DEFAULT_CONFIG):
    config = load_config(config_path)
    llm = LLM(config=config)
    tokenizer = llm.tokenizer
    
    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = sampling_params_from_config(config)
    prompts = config['prompts']
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)