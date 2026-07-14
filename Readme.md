# FasterKernels ⚡

A collection of open, highly-optimized attention and decoding kernels for AI inference implemented using **Triton**, **TileLang**, and **CUDA / C++**. 

This repository targets next-generation LLM inference architectures, showcasing optimizations for **FlashAttention-2** (prefill) and **Flash Decoding** (GQA + varlen + attention sinks) on modern NVIDIA GPU architectures (e.g. NVIDIA L4 / Ada Lovelace). Direct support for **Huggingface 🤗** models using GQA with dynamic cache.

---

## Key Features

* **FlashAttention-2 (Prefill)**: Implementation of the forward pass of the FlashAttention-2 algorithm supporting causal/non-causal attention.
* **Flash Decoding (GQA + Varlen)**: High-efficiency decoding kernels designed for Grouped Query Attention (GQA), variable sequence lengths per batch item (`varlen`), and long-context processing.
* **Attention Sinks Support**: First-class support for attention sinks (used to keep initial token activations for long-context generation) natively integrated into the online softmax computation.
* **Multi-Backend implementations**: Cross-compare identical algorithms written in Triton, TileLang (compiled to optimized CUDA via TVM-like DSL), and raw C++/CUDA templates.

---

## Kernel Registry

| Algorithm | Triton | TileLang | CUDA/C++ | Key Features |
|---|---|---|---|---|
| **FlashAttention-2** | [triton_fa2.py](fskernels/triton/triton_fa2.py) | [tilelang_fa2.py](fskernels/tilelang/tilelang_fa2.py) | [cuda_fa2.py](fskernels/cuda/cuda_fa2.py) | Causal/Non-causal, online softmax |
| **Flash Decoding** | [triton_gqa_decode.py](fskernels/triton/triton_gqa_decode.py) | [tilelang_gqa_decode.py](fskernels/tilelang/tilelang_gqa_decode.py) | [cuda_fd_gqa.py](fskernels/cuda/cuda_fd_gqa.py) | GQA (arbitrary group size), varlen layout, Attention Sinks (`s_aux`) |

---

## Performance Benchmarks

Below are benchmarking results evaluated on an **NVIDIA L4 GPU** (24GB VRAM, sm_89) under **FP16** precision. The benchmark sweeps Flash Decoding performance with Query Heads = 32, KV Heads = 8 (GQA Group Size = 4), Head Dim = 128.

### Latency Visualizations

![GQA Varlen Decode Latency vs Max Context Length](benchmarks/benchmark_plots/latency_vs_seqlen.png)

![GQA Varlen Decode Latency vs Batch Size](benchmarks/benchmark_plots/latency_vs_batch.png)

### 1. Sequence Length Sweep (Fixed Batch Size = 4)
*Measures latency (ms) for varying context/sequence lengths.*

| Max Sequence Length | PyTorch Compile (ms) | PyTorch SDPA (ms) | Triton (ms) | TileLang (ms) | Native CUDA (ms) |
|---|---|---|---|---|---|
| **512** | 1.982 | 1.086 | 0.141 | **0.057** | 0.801 |
| **2048** | 2.636 | 2.258 | 0.268 | **0.152** | 2.441 |
| **8192** | 2.803 | 4.801 | 0.409 | **0.351** | 6.014 |
| **16384** | 7.433 | 10.706 | 0.931 | **0.865** | 14.052 |
| **32768** | 10.283 | 22.856 | **1.370** | 1.518 | 32.299 |

### 2. Batch Size Sweep (Fixed Max Sequence Length = 16384)
*Measures latency (ms) for varying batch sizes.*

| Batch Size | PyTorch Compile (ms) | PyTorch SDPA (ms) | Triton (ms) | TileLang (ms) | Native CUDA (ms) |
|---|---|---|---|---|---|
| **1** | 2.417 | 2.570 | **0.436** | 0.472 | 12.919 |
| **4** | 7.510 | 10.749 | 0.943 | **0.867** | 14.107 |
| **16** | 30.008 | 46.762 | 3.110 | **3.099** | 22.181 |
| **32** | 58.456 | 99.993 | **5.629** | 5.703 | 36.543 |

### Key Optimization Insights
* **Triton/TileLang vs. PyTorch SDPA**: Triton and TileLang are **15x - 18x faster** than PyTorch SDPA at larger batch sizes and context lengths.
* **Why is Native CUDA slower?**: The native CUDA implementation included is a template-free cooperative parallel reduction kernel. Unlike Triton and TileLang, it does not utilize specialized CUDA Tensor Cores or compiler-optimized double-buffered pipelines (`cp.async`).
* **Attention Sinks**: Standard PyTorch SDPA cannot support attention sinks (biasing softmax by keeping initial key/value activations) without falling back to un-fused computation. Triton, TileLang, and CUDA kernels support this natively using the `s_aux` parameter.

---

## Installation

Ensure you have Python 3.9+ and PyTorch installed with CUDA support.

```bash
# Clone the repository
git clone https://github.com/coder0143/fasterkernels.git
cd fasterkernels

# Editable installation of the package
pip install -e .

# Install with benchmarking dependencies (pandas, matplotlib, seaborn)
pip install -e ".[benchmarks]"

# Install with development/testing dependencies (pytest)
pip install -e ".[dev]"
```

---

## Usage Examples

### 1. Flash Decoding with GQA and Varlen (Triton)
```python
import torch
from fasterkernels.fskernels.triton.triton_gqa_decode import flash_decode_gqa_varlen

# Batch setup: 4 requests, Q Head=32, KV Head=8, Head Dim=128
B, Q_H, KV_H, D = 4, 32, 8, 128

# Set up variable sequence lengths per batch element
k_seqlens = torch.tensor([512, 1024, 256, 2048], dtype=torch.int32, device="cuda")
cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
cu_seqlens_k[1:] = torch.cumsum(k_seqlens, dim=0)
total_kv_tokens = cu_seqlens_k[-1].item()

# Input tensors
q = torch.randn(B, Q_H, D, dtype=torch.float16, device="cuda")
k_varlen = torch.randn(total_kv_tokens, KV_H, D, dtype=torch.float16, device="cuda")
v_varlen = torch.randn(total_kv_tokens, KV_H, D, dtype=torch.float16, device="cuda")

# (Optional) Attention Sink Bias
sink = torch.randn(Q_H, dtype=torch.float32, device="cuda") * 0.1

# Invoke the kernel
output = flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, s_aux=sink)
print("Output shape:", output.shape) # Expected: [4, 32, 128]
```

### 2. FlashAttention-2 Prefill (TileLang)
```python
import torch
from fasterkernels.fskernels.tilelang.tilelang_fa2 import flash_attn_mha_bhsd

# Shape setup: [Batch, Heads, SeqLen, HeadDim]
q = torch.randn(4, 16, 4096, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(4, 16, 4096, 128, dtype=torch.bfloat16, device="cuda")
v = torch.randn(4, 16, 4096, 128, dtype=torch.bfloat16, device="cuda")

# Invoke the JIT-compiled TileLang FlashAttention kernel
output = flash_attn_mha_bhsd(q, k, v, is_causal=False)
print("Output shape:", output.shape) # Expected: [4, 16, 4096, 128]
```

### 3. Use with HuggingFace transformers 🤗 (Support with hf dynamic cache)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

# Kernel import
from fasterkernels.fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward

# Intercept the execution registry globally
qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward

model_id = "Qwen/Qwen3-4B"
device = "cuda"
dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.clean_up_tokenization_spaces = False 

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=dtype, 
    device_map=device
)

prompt = "Explain the hardware roofline constraints of custom CUDA attention kernels:"
print(f"Prompt: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Using text-streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("\nAnswer: ")

with torch.no_grad():
    _ = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        use_cache=True  # Dictates routing directly through your patched decode path
    )

print("Generated 400 tokens")
```

---

## Testing, Benchmarking & Profiling

### Running Unit Tests
Validate precision and correctness across all backends against PyTorch references:
```bash
pytest tests/
```

### Running Performance Benchmarks
Run sequence length and batch sweeps, generate latency plots, and output CSV metrics:
```bash
python benchmarks/benchmark.py --q-heads 32 --kv-heads 8 --head-dim 128 --dtype fp16
```
Plots and CSV results will be saved to `benchmarks/benchmark_plots/`.

### Profiling Triton Kernels
Compile lower representation code and generate performance summaries for Triton backends:
```bash
python profiling/profile_triton_fa2.py
python profiling/profile_triton_fd_gqa.py
```
Outputs are written under `profile/logs/`.
