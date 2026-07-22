import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Config
model_card = "Qwen/Qwen3-14B-FP8"
device = "cuda"

prompt = (
    "Explain the hardware roofline constraints of custom CUDA attention kernels in detail:"
)

max_new_tokens = 400
temperature = 0.7


# Load model
print(f"Loading {model_card}...")

tokenizer = AutoTokenizer.from_pretrained(model_card)
tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    model_card,
    device_map=device,
    torch_dtype="auto",
).eval()


# Compile
print("Compiling model...")

model = torch.compile(
    model,
    mode="max-autotune",
    fullgraph=False,
    dynamic=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)


# Warmup
print("Warmup...")

with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=temperature,
        use_cache=True,
    )

torch.cuda.synchronize()

# Benchmarking
print("Benchmarking...")

torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        use_cache=True,
    )

torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start

# Metrics
prompt_len = inputs.input_ids.shape[1]
new_tokens = outputs.shape[1] - prompt_len

generated_text = tokenizer.decode(
    outputs[0][prompt_len:],
    skip_special_tokens=True,
)

tok_per_sec = new_tokens / elapsed
ms_per_tok = elapsed * 1000 / new_tokens

# Results
print("\nAnswer:\n")
print(generated_text)

print("\n========== HF torch.compile ==========")
print(f"Tokens Generated : {new_tokens}")
print(f"Time Taken       : {elapsed:.2f} s")
print(f"Throughput       : {tok_per_sec:.2f} tok/s")
print(f"Latency/token    : {ms_per_tok:.2f} ms")