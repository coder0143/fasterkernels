import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

# Kernel import
from fskernels.triton.triton_gqa_decode_hf import custom_hf_decode_attention_forward

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