import argparse
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from triton.testing import do_bench

# Kernel Imports
from fskernels.triton.triton_gqa_decode import flash_decode_gqa_varlen as triton_kernel
from fskernels.tilelang.tilelang_gqa_decode import flash_decode_gqa_varlen as tilelang_kernel
from fskernels.cuda.cuda_fd_gqa import flash_decode_gqa_varlen as cuda_kernel
from tests.test_triton_fd_gqa import ref_attention_compiled as compile_kernel

# for @torch.compile
torch._dynamo.config.capture_scalar_outputs = True

def setup_inputs(b, q_h, kv_h, max_s, d, device="cuda", dtype=torch.float16):
    torch.manual_seed(42)
    
    k_seqlens = torch.randint(max(64, max_s // 4), max_s + 1, size=(b,), device=device)
    cu_seqlens_k = torch.zeros(b + 1, device=device, dtype=torch.int32)
    cu_seqlens_k[1:] = torch.cumsum(k_seqlens, dim=0).to(torch.int32)
    total_tokens = cu_seqlens_k[-1].item()

    q = torch.randn(b, q_h, d, device=device, dtype=dtype)
    k = torch.randn(total_tokens, kv_h, d, device=device, dtype=dtype)
    v = torch.randn(total_tokens, kv_h, d, device=device, dtype=dtype)
    sink = torch.randn(q_h, device=device, dtype=torch.float32) * 0.1
    
    return q, k, v, cu_seqlens_k, sink, k_seqlens

@torch.compile
def run_sdpa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, k_seqlens, q_heads, kv_heads, head_dim):
    """
    Helper to run torch SDPA by preparing a padded dense representation and mask
    to mimic how an application layer feeds varlen data into SDPA.
    """
    b = q.shape[0]
    max_s = int(k_seqlens.max().item())
    
    # 1. Materialize padded K and V blocks from varlen layout
    k_padded = torch.zeros(b, kv_heads, max_s, head_dim, device=q.device, dtype=q.dtype)
    v_padded = torch.zeros(b, kv_heads, max_s, head_dim, device=q.device, dtype=q.dtype)
    for i in range(b):
        start, end = cu_seqlens_k[i], cu_seqlens_k[i+1]
        k_padded[i, :, :end-start] = k_varlen[start:end].transpose(0, 1)
        v_padded[i, :, :end-start] = v_varlen[start:end].transpose(0, 1)
        
    # 2. Expand KV heads to match Q heads (GQA replication)
    g = q_heads // kv_heads
    if g > 1:
        k_padded = k_padded.unsqueeze(2).expand(b, kv_heads, g, max_s, head_dim).reshape(b, q_heads, max_s, head_dim)
        v_padded = v_padded.unsqueeze(2).expand(b, kv_heads, g, max_s, head_dim).reshape(b, q_heads, max_s, head_dim)
        
    # 3. Formulate the explicit additive bool mask [B, 1, 1, Max_S]
    mask = torch.arange(max_s, device=q.device).expand(b, -1) >= k_seqlens.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(2) # Broad-castable to [B, Q_H, 1, Max_S]
    
    # 4. Reshape Q to match SDPA expectations: [B, Q_H, 1, D]
    q_sdpa = q.unsqueeze(2)
    
    # Note: Attention sink logic cannot be natively injected into SDPA without breaking 
    # the fused backend, so this represents standard SDPA baseline performance.
    out = F.scaled_dot_product_attention(q_sdpa, k_padded, v_padded, attn_mask=~mask)
    return out.squeeze(2)

def run_benchmark(args):
    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    
    print("Pre-compiling baseline kernel...")
    q, k, v, cu_s, sink, k_seqs = setup_inputs(4, args.q_heads, args.kv_heads, 1024, args.head_dim, device, dtype)
    for _ in range(5):
        _ = compile_kernel(q, k, v, cu_s, sink)
    torch.cuda.synchronize()

    results = []

    # Sweep 1: Max Sequence Length (Fixed Batch Size)
    print(f"\n--- Sweeping Sequence Length (Batch Size = {args.fixed_batch}) ---")
    for s in args.seq_lens:
        q, k, v, cu_s, sink, k_seqs = setup_inputs(args.fixed_batch, args.q_heads, args.kv_heads, s, args.head_dim, device, dtype)
        
        t_compile = do_bench(lambda: compile_kernel(q, k, v, cu_s, sink))
        t_triton  = do_bench(lambda: triton_kernel(q, k, v, cu_s, sink))
        t_tile    = do_bench(lambda: tilelang_kernel(q, k, v, cu_s, sink))
        t_cuda    = do_bench(lambda: cuda_kernel(q, k, v, cu_s, sink))
        t_sdpa    = do_bench(lambda: run_sdpa_varlen(q, k, v, cu_s, k_seqs, args.q_heads, args.kv_heads, args.head_dim))

        print(f"SeqLen {s:5d} | Compile: {t_compile:.3f}ms | SDPA: {t_sdpa:.3f}ms | Triton: {t_triton:.3f}ms | TileLang: {t_tile:.3f}ms | CUDA: {t_cuda:.3f}ms")
        
        for k_name, latency in [("torch.compile", t_compile), ("Torch SDPA", t_sdpa), ("Triton", t_triton), ("TileLang", t_tile), ("CUDA (Native)", t_cuda)]:
            results.append({
                "Type": "Sweep_SeqLen", "Param": s, "Kernel": k_name, "Latency (ms)": latency
            })

    # Sweep 2: Batch Size (Fixed Sequence Length)
    print(f"\n--- Sweeping Batch Size (Max SeqLen = {args.fixed_seqlen}) ---")
    for b in args.batch_sizes:
        q, k, v, cu_s, sink, k_seqs = setup_inputs(b, args.q_heads, args.kv_heads, args.fixed_seqlen, args.head_dim, device, dtype)
        
        t_compile = do_bench(lambda: compile_kernel(q, k, v, cu_s, sink))
        t_triton  = do_bench(lambda: triton_kernel(q, k, v, cu_s, sink))
        t_tile    = do_bench(lambda: tilelang_kernel(q, k, v, cu_s, sink))
        t_cuda    = do_bench(lambda: cuda_kernel(q, k, v, cu_s, sink))
        t_sdpa    = do_bench(lambda: run_sdpa_varlen(q, k, v, cu_s, k_seqs, args.q_heads, args.kv_heads, args.head_dim))

        print(f"Batch {b:2d}  | Compile: {t_compile:.3f}ms | SDPA: {t_sdpa:.3f}ms | Triton: {t_triton:.3f}ms | TileLang: {t_tile:.3f}ms | CUDA: {t_cuda:.3f}ms")
        
        for k_name, latency in [("torch.compile", t_compile), ("Torch SDPA", t_sdpa), ("Triton", t_triton), ("TileLang", t_tile), ("CUDA (Native)", t_cuda)]:
            results.append({
                "Type": "Sweep_Batch", "Param": b, "Kernel": k_name, "Latency (ms)": latency
            })

    return pd.DataFrame(results)

def plot_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Line Plot: Sequence Length Sweep
    plt.figure(figsize=(10, 5))
    df_seq = df[df["Type"] == "Sweep_SeqLen"]
    sns.lineplot(data=df_seq, x="Param", y="Latency (ms)", hue="Kernel", marker="o", linewidth=2)
    plt.title("GQA Varlen Decode Latency vs Max Context Length")
    plt.xlabel("Max KV Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_seqlen.png"), dpi=200)
    plt.close()

    # 2. Bar Plot: Batch Size Sweep
    plt.figure(figsize=(10, 5))
    df_batch = df[df["Type"] == "Sweep_Batch"]
    sns.barplot(data=df_batch, x="Param", y="Latency (ms)", hue="Kernel")
    plt.title("GQA Varlen Decode Latency vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_batch.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--output-dir", type=str, default="./benchmarks/benchmark_plots")
    args = parser.parse_args()

    args.seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    args.fixed_batch = 4
    args.batch_sizes = [1, 2, 4, 8, 16, 32]
    args.fixed_seqlen = 16384

    df_metrics = run_benchmark(args)
    plot_results(df_metrics, args.output_dir)
    df_metrics.to_csv(os.path.join(args.output_dir, "raw_benchmarks.csv"), index=False)
    print(f"\nExecution finished. Plots and CSV saved to ./{args.output_dir}")