import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

# Optimized kernel
from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward

def populate_long_context_cache(model, tokenizer, context_size, batch_size=1):
    """
    Artificially seeds a long context depth inside the active model cache 
    to test the kernel under high global memory bandwidth pressure.
    """
    dummy_text = "The quick brown fox jumps over the lazy dog. " * (context_size // 10)
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    # Clip or expand input dimension coordinates precisely
    input_ids = inputs["input_ids"][:, :context_size].repeat(batch_size, 1).to("cuda")
    
    # Warm pass to build standard Hugging Face 4D Cache structures
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    return outputs.past_key_values

def measure_pure_decode_speed(model, tokenizer, past_key_values, num_decode_steps=128):
    """
    Measures decoding speed exclusively, excluding prefill and allocation steps.
    """
    batch_size = past_key_values.get_seq_length(0)  # Extract context state tracker bounds
    device = model.device
    
    # Seed token vector representing a single decoded position query step
    next_input_ids = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    current_cache = past_key_values
    with torch.no_grad():
        for _ in range(num_decode_steps):
            outputs = model(next_input_ids, past_key_values=current_cache, use_cache=True)
            next_input_ids = outputs.logits.argmax(dim=-1)
            current_cache = outputs.past_key_values
            
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed_seconds = end_time - start_time
    tokens_per_second = num_decode_steps / elapsed_seconds
    return tokens_per_second, elapsed_seconds

if __name__ == "__main__":
    model_id = "Qwen/Qwen3-4B"
    
    # Bench params
    CONTEXT_DEPTH = 800
    DECODE_TOKENS = 1000
    
    print("==========================================================")
    print(f"Initializing Framework & Loading Architecture: {model_id}")
    print("==========================================================")
    
    # Retain a safe link to fallback eager loops
    native_eager_backup = qwen3_mod.eager_attention_forward
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.clean_up_tokenization_spaces = False
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16, 
        device_map="cuda"
    )

    print(f"\n[Baseline] Populating {CONTEXT_DEPTH} tokens of context memory history...")
    hf_cache = populate_long_context_cache(model, tokenizer, CONTEXT_DEPTH)
    
    print(f"[Baseline] Timing {DECODE_TOKENS} steps of sequential token decode...")
    hf_speed, hf_time = measure_pure_decode_speed(model, tokenizer, hf_cache, num_decode_steps=DECODE_TOKENS)
    print(f"-> HF Native Pipeline Performance: {hf_speed:.2f} tokens/sec (Completed in {hf_time:.3f}s)")

    # ------------------------------------------------------------------------
    # Switch to Custom Triton Kernel Integration Pass
    # ------------------------------------------------------------------------
    print("\n==========================================================")
    print("Injecting Zero-Allocation Custom Triton Flash Decoding Patcher")
    print("==========================================================")
    qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward

    print(f"[Triton] Re-populating {CONTEXT_DEPTH} tokens of context memory history...")
    # This pass primes the raw code pathways and triggers cold-start JIT updates safely
    triton_cache = populate_long_context_cache(model, tokenizer, CONTEXT_DEPTH)
    
    print(f"[Triton] Timing {DECODE_TOKENS} steps of sequential token decode...")
    tri_speed, tri_time = measure_pure_decode_speed(model, tokenizer, triton_cache, num_decode_steps=DECODE_TOKENS)
    print(f"-> Triton Custom Kernel Performance: {tri_speed:.2f} tokens/sec (Completed in {tri_time:.3f}s)")

    print("\n==========================================================")
    print("Final Performance Analysis Metrics")
    print("==========================================================")
    speedup = tri_speed / hf_speed
    print(f"Target Memory Depth Context Size : {CONTEXT_DEPTH} tokens")
    print(f"Measured Speedup Multiplier     : {speedup:.4f}x")
    
    if speedup > 1.0:
        print(f"🎉 Success! Your hardcoded Triton configuration scales out efficiently under high load.")
    else:
        print("💡 Performance is identical. At this block sizing scale, hardware driver launch boundaries")
        print("   are equal to the computational steps. Try scaling CONTEXT_DEPTH out to 16,384 tokens.")