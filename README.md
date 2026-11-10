<p align="center">
  <img src="./assets/minivllm.png" alt="MinivLLM" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

Benchmarking on flash attention in prefilling time and paged attention in decoding time are provided.

**New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the main inference engine
uv run python main.py

# Select an explicit config (this is also the default)
uv run python main.py --config configs/qwen3_0_6b.json

# Run prefilling benchmark
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py
```

To select a different model or parallel size, edit a JSON config under `configs/`
and pass it with `--config`. Multi-GPU correctness is tracked separately from the
single-GPU M0/M1 baseline.

## What Each Script Does

```bash
uv run python main.py
```

This is the main inference engine demo

Demonstrates the complete LLM inference pipeline using a custom engine implementation:
- Load the pinned Qwen3-0.6B checkpoint from `configs/qwen3_0_6b.json`
- Use the fixed prompts, sampling parameters, and random seed from the config
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Applies the configured completion and total-length limits with temperature sampling

This showcases how the custom vLLM implementation handles batched text generation with memory-efficient attention.

```bash
uv run python benchmark_prefilling.py
```

This is the prefilling phase comparison

Compares three attention implementations during the **prefilling phase** (processing input prompts):

1. **PyTorch Standard (O(N²) memory)**: Traditional attention that materializes full attention matrix
2. **Naive Triton (O(N²) memory)**: GPU kernel that also uses O(N²) memory, limited by shared memory constraints (≤128 tokens)
3. **Flash Attention (O(N) memory)**: Memory-efficient online softmax algorithm that processes attention in blocks

```bash
uv run python benchmark_decoding.py
```

This is the decoding phase comparison

Compares three implementations during the **decoding phase** (generating output tokens one at a time):

1. **Naive PyTorch**: Simple loop-based implementation using paged KV cache
2. **Optimized PyTorch**: Vectorized implementation with batch gathering and masking
3. **Triton Kernel**: Custom GPU kernel optimized for paged attention decode


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Model layer components (activation, attention, embeddings, etc.)
│       ├── utils/        # Utility helpers and inference context management
│       └── sampling_parameters.py
├── main.py              # Full inference demo
├── benchmark_prefilling.py   # Prefilling attention comparison
└── benchmark_decoding.py     # Decoding attention comparison
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU
- Dependencies: `transformers`, `torch`, `xxhash` (managed by uv)


## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://star-history.dera.page/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)
