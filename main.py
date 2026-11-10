import argparse
import sys
from pathlib import Path

# Add src to Python path when running this file directly.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.config import load_config, sampling_params_from_config
from myvllm.engine.llm_engine import LLMEngine as LLM


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "qwen3_0_6b.json"


def main(config_path: str | Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    llm = LLM(config=config)
    tokenizer = llm.tokenizer
    sampling_params = sampling_params_from_config(config)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in config["prompts"]
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs["text"]):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the configured Qwen3 demo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    main(parser.parse_args().config)
