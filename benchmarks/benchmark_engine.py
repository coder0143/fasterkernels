"""
Benchmarks five decode strategies for Qwen3-8B-FP8 on L4:

1. Native HF           — model.generate(), eager sdpa, DynamicCache
2. Triton attn eager   — our Triton kernel, Python loop, DynamicCache
3. CUDA graph          — Triton kernel, graphed decode, CUDAGraphCache
4. Speculative decode  — CUDA-graph target + Qwen3-0.6B draft (γ=5)
5. Paged batch=4       — 4 concurrent seqs, paged attention, throughput

Run on the HPC cluster:
    python benchmarks/benchmark_engine.py

Expected approximate results on L4:
    Native HF       :   ~5.8 tok/s  |  ~172 ms/tok
    Triton eager    :   ~8-12 tok/s  |  ~85-125 ms/tok  (kills attn overhead)
    CUDA graph      :  ~25-45 tok/s  |  ~22-40 ms/tok   (kills Python loop)
    Speculative     :  ~55-90 tok/s  |  ~11-18 ms/tok   (reduces target calls)
    Paged batch=4   : ~90-160 tok/s  system            (multi-req throughput)
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID   = "Qwen/Qwen3-8B-FP8"
DRAFT_ID   = "Qwen/Qwen3-0.6B"
DEVICE     = "cuda"
PROMPT     = (
    "Explain the hardware roofline constraints of custom CUDA attention kernels "
    "in detail, covering memory bandwidth, compute throughput, and the impact of "
    "tensor core utilisation on modern data-centre GPUs:"
)
WARMUP_TOK = 20
BENCH_TOK  = 200
TEMPERATURE = 0.0   # greedy for reproducibility in benchmarks


def load_target():
    print(f"Loading target model {MODEL_ID} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype="auto", device_map=DEVICE
    ).eval()
    return model, tok


def load_draft(device=DEVICE):
    print(f"Loading draft model {DRAFT_ID} …")
    return AutoModelForCausalLM.from_pretrained(
        DRAFT_ID, dtype=torch.bfloat16, device_map=device
    ).eval()


def greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(-1, keepdim=True)


# ---------------------------------------------------------------------------
# 1. Native HF baseline
# ---------------------------------------------------------------------------
def bench_native_hf(model, tok) -> dict:
    print("\n[1/5] Native HF  …")
    inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
    eos_ids = getattr(model.config, "eos_token_id", [tok.eos_token_id])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=WARMUP_TOK, do_sample=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=BENCH_TOK, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = out.shape[1] - inputs["input_ids"].shape[1]
    return {"name": "Native HF", "n": n, "elapsed": elapsed,
            "tok_s": n / elapsed, "ms_tok": elapsed * 1000 / n}


# ---------------------------------------------------------------------------
# 2. Triton eager (our attention, Python loop, no graph)
# ---------------------------------------------------------------------------
def bench_triton_eager(model, tok) -> dict:
    import transformers.models.qwen3.modeling_qwen3 as qm
    from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward
    qm.eager_attention_forward = custom_hf_decode_attention_forward
    print("\n[2/5] Triton eager  …")

    from fskernels.engine.fs_inference_engine import FsInferenceEngine
    engine = FsInferenceEngine(model, tok, patch_attention=False)  # already patched

    # Warmup
    _ = engine.generate(PROMPT, max_new_tokens=WARMUP_TOK, temperature=TEMPERATURE, use_graph=False)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    res = engine.generate(PROMPT, max_new_tokens=BENCH_TOK, temperature=TEMPERATURE, use_graph=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {"name": "Triton eager", "n": res["new_tokens"], "elapsed": elapsed,
            "tok_s": res["new_tokens"] / elapsed,
            "ms_tok": elapsed * 1000 / res["new_tokens"]}


# ---------------------------------------------------------------------------
# 3. CUDA graph decode
# ---------------------------------------------------------------------------
def bench_cuda_graph(model, tok) -> dict:
    print("\n[3/5] CUDA graph  …")
    from fskernels.engine import FsInferenceEngine

    engine = FsInferenceEngine(model, tok)

    # Warmup (also builds graph)
    _ = engine.generate(PROMPT, max_new_tokens=WARMUP_TOK, temperature=TEMPERATURE,
                        use_graph=True, warmup_steps=5)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    res = engine.generate(PROMPT, max_new_tokens=BENCH_TOK, temperature=TEMPERATURE,
                          use_graph=True, warmup_steps=5)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {"name": "CUDA graph", "n": res["new_tokens"], "elapsed": elapsed,
            "tok_s": res["new_tokens"] / elapsed,
            "ms_tok": elapsed * 1000 / res["new_tokens"]}


# ---------------------------------------------------------------------------
# 4. Speculative decoding (target: graphed, draft: Qwen3-0.6B)
# ---------------------------------------------------------------------------
def bench_speculative(model, tok) -> dict:
    print("\n[4/5] Speculative (γ=5)  …")
    from fskernels.engine import SpeculativeEngine

    draft = load_draft()
    engine = SpeculativeEngine(model, draft, tok, gamma=5)

    # Warmup
    _ = engine.generate(PROMPT, max_new_tokens=WARMUP_TOK, temperature=TEMPERATURE)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    res = engine.generate(PROMPT, max_new_tokens=BENCH_TOK, temperature=TEMPERATURE)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    extra = f"  α={res.get('acceptance_rate', 0):.2f}"
    return {"name": f"Speculative{extra}", "n": res["new_tokens"], "elapsed": elapsed,
            "tok_s": res["new_tokens"] / elapsed,
            "ms_tok": elapsed * 1000 / res["new_tokens"]}


# ---------------------------------------------------------------------------
# 5. Paged attention — batched throughput demo (4 concurrent sequences)
# ---------------------------------------------------------------------------
def bench_paged_batch(model, tok) -> dict:
    """
    Simulates serving 4 requests simultaneously using the paged KV kernel.
    All 4 sequences share a page pool; throughput is measured as total
    tokens generated across all sequences per wall-clock second.
    """
    print("\n[5/5] Paged batch=4  …")
    import transformers.models.qwen3.modeling_qwen3 as qm
    from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward
    qm.eager_attention_forward = custom_hf_decode_attention_forward

    BATCH = 4
    inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"].expand(BATCH, -1).clone()
    bsz, prefill_len = input_ids.shape
    max_seq_len = prefill_len + BENCH_TOK + 64

    from fskernels.engine.cuda_graph_runner import CUDAGraphRunner
    runner = CUDAGraphRunner(model, BATCH, max_seq_len, DEVICE, num_splits=8)
    _ = runner.prefill(input_ids)
    runner.build(warmup_steps=5)

    eos_ids = set(getattr(model.config, "eos_token_id", [tok.eos_token_id])
                  if not isinstance(getattr(model.config, "eos_token_id", None), int)
                  else [model.config.eos_token_id])

    # Greedy decode for all 4 sequences in lock-step (simple demo)
    prefill_logits = runner._static_logits[:, -1, :] if runner._static_logits is not None else \
        model(input_ids=input_ids[:, -1:], past_key_values=runner.kv_cache,
              attention_mask={"full_attention": None}, use_cache=True, logits_to_keep=1).logits[:, -1, :]

    next_tok = prefill_logits.argmax(-1, keepdim=True)  # [B, 1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generated = 0
    for _ in range(BENCH_TOK):
        logits = runner.step(next_tok)    # [B, vocab]
        next_tok = logits.argmax(-1, keepdim=True)
        generated += BATCH

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return {"name": f"Paged batch={BATCH}", "n": generated, "elapsed": elapsed,
            "tok_s": generated / elapsed,
            "ms_tok": elapsed * 1000 / (generated / BATCH)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_table(results):
    print("\n" + "=" * 72)
    print(f"{'Strategy':<30}  {'tok/s':>10}  {'ms/tok':>10}  {'speedup':>10}")
    print("-" * 72)
    baseline = results[0]["tok_s"]
    for r in results:
        sp = r["tok_s"] / baseline
        print(f"{r['name']:<30}  {r['tok_s']:>10.2f}  {r['ms_tok']:>10.2f}  {sp:>9.2f}x")
    print("=" * 72 + "\n")


def main():
    model, tok = load_target()

    results = []

    results.append(bench_native_hf(model, tok))
    results.append(bench_triton_eager(model, tok))
    results.append(bench_cuda_graph(model, tok))

    try:
        results.append(bench_speculative(model, tok))
    except Exception as e:
        print(f"  [skip speculative] {e}")

    try:
        results.append(bench_paged_batch(model, tok))
    except Exception as e:
        print(f"  [skip paged batch] {e}")

    print_table(results)


if __name__ == "__main__":
    main()
