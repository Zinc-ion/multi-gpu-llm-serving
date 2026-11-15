import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM


def create_tiny_qwen_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    model_config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=32, tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(20261115)
        Qwen3ForCausalLM(model_config).save_pretrained(path)
    vocab = {"[UNK]": 0, "hello": 1, "[EOS]": 63}
    vocab.update({f"token{i}": i for i in range(2, 63)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]",
    ).save_pretrained(path)
    return {
        "model_name_or_path": str(path), "model_type": "Qwen3-0.6B",
        "model_revision": "a" * 40, "tokenizer_revision": "b" * 40,
        "world_size": 1, "block_size": 4, "enforce_eager": True,
        "vocab_size": 64, "hidden_size": 32, "num_heads": 2, "head_dim": 16,
        "scale": model_config.head_dim ** -0.5, "num_kv_heads": 1,
        "rms_norm_epsilon": model_config.rms_norm_eps,
        "qkv_bias": False, "base": model_config.rope_theta, "max_position": 32,
        "intermediate_size": 64, "ffn_bias": False, "num_layers": 2,
        "tie_word_embeddings": False, "max_num_sequences": 1,
        "max_num_batched_tokens": 8, "max_model_length": 8,
        "max_cached_blocks": 8, "gpu_memory_utilization": 0.1,
        "dtype": "float32", "eos": [63],
    }
