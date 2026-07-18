import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

model_card = "Qwen/Qwen3-8B-FP8"
device = "cuda"
NUM_NEW_TOKENS = 200
PROMPT = "Explain the hardware roofline constraints of custom CUDA attention kernels in detail:"


def load_model(patch_fp8: bool):
    if patch_fp8:
        from fskernels.triton.triton_gqa_decode_hf_fp8 import custom_hf_decode_attention_forward_fp8
        qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward_fp8
        tag = "Triton-FP8"
    else:
        # Reset to native in case a previous run patched it
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as _native
        qwen3_mod.eager_attention_forward = _native
        tag = "Native-HF"

    tokenizer = AutoTokenizer.from_pretrained(model_card)
    tokenizer.clean_up_tokenization_spaces = False

    model = AutoModelForCausalLM.from_pretrained(
        model_card,
        device_map=device,
    )
    model.eval()
    return model, tokenizer, tag


def run_generation(model, tokenizer, tag):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=NUM_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_generated = out.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = n_generated / elapsed
    ms_per_token = elapsed * 1000 / n_generated

    print(f"\n[{tag}]")
    print(f"  Tokens generated : {n_generated}")
    print(f"  Total time       : {elapsed:.3f} s")
    print(f"  Throughput       : {tokens_per_sec:.2f} tok/s")
    print(f"  Latency/token    : {ms_per_token:.3f} ms")

    decoded = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"tag": tag, "tok_per_sec": tokens_per_sec, "ms_per_tok": ms_per_token, "text": decoded}


if __name__ == "__main__":
    print(f"Model  : {model_card}")
    print(f"Device : {device}")
    print(f"Prompt : {PROMPT}\n")
    print("=" * 60)

    results = []

    # Native HF baseline
    model_native, tok_native, tag_native = load_model(patch_fp8=False)
    r_native = run_generation(model_native, tok_native, tag_native)
    results.append(r_native)

    del model_native
    torch.cuda.empty_cache()

    # Triton FP8 kernel
    model_fp8, tok_fp8, tag_fp8 = load_model(patch_fp8=True)
    r_fp8 = run_generation(model_fp8, tok_fp8, tag_fp8)
    results.append(r_fp8)

    del model_fp8
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Backend':<20} {'tok/s':>10} {'ms/tok':>10}")
    print("-" * 42)
    for r in results:
        print(f"{r['tag']:<20} {r['tok_per_sec']:>10.2f} {r['ms_per_tok']:>10.3f}")

    speedup = results[1]["tok_per_sec"] / results[0]["tok_per_sec"]
    print(f"\nTriton-FP8 speedup vs Native-HF: {speedup:.2f}x")
    print("=" * 60)
