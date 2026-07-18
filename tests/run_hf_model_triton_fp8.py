import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

from fskernels.triton.triton_gqa_decode_hf_fp8 import custom_hf_decode_attention_forward_fp8

qwen3_mod.eager_attention_forward = custom_hf_decode_attention_forward_fp8

model_card = "Qwen/Qwen3-8B-FP8"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_card)
tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    model_card,
    device_map=device,
)

prompt = "Explain the hardware roofline constraints of custom CUDA attention kernels:"
print(f"Prompt: {prompt}")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
        use_cache=True,
    )

print("Generated 400 tokens")
