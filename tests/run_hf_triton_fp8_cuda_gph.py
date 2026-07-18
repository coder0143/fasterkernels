from transformers import AutoModelForCausalLM, AutoTokenizer

from fskernels.engine import FsInferenceEngine

model_card = "Qwen/Qwen3-8B-FP8"
device = "cuda"

print(f"Loading tokenizer & model {model_card} …")
tokenizer = AutoTokenizer.from_pretrained(model_card)
tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    model_card,
    device_map=device,
).eval()

print("Initializing FsInferenceEngine (CUDA Graph mode) …")
# Initialize engine (this will patch attention to use custom Triton kernel)
engine = FsInferenceEngine(model, tokenizer, device=device)

prompt = "Explain the hardware roofline constraints of custom CUDA attention kernels in detail:"
print(f"\nPrompt: {prompt}")

print("\nGenerating with CUDA Graph decode …")
# The first generate call will trigger CUDA graph capture (and JIT compile the kernels)
result = engine.generate(
    prompt,
    max_new_tokens=400,
    temperature=0.7,
    use_graph=True,
    warmup_steps=5,
)

print(f"\nAnswer:\n{result['text']}")
print("\n" + "=" * 50)
print(f"Tokens Generated : {result['new_tokens']}")
print(f"Time Taken       : {result['elapsed_s']:.2f} seconds")
print(f"Throughput       : {result['tok_per_sec']:.2f} tokens/sec")
print(f"Latency per token: {result['ms_per_tok']:.2f} ms/token")
print("=" * 50)
