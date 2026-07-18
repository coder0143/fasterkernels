import torch
import math
from fskernels.triton.triton_gqa_decode_hf_fp8 import flash_decode_gqa_hf_fp8, quantize_to_fp8

device = "cuda"
dtype = torch.float16

def repeat_kv(x, n_rep):
    B, KV_H, S, D = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, KV_H, n_rep, S, D).reshape(B, KV_H * n_rep, S, D)


def ref_attention_fp16(q, k, v, cache_lens):
    B, Q_H, D = q.shape
    KV_H = k.shape[1]
    S = k.shape[2]
    scale = 1.0 / math.sqrt(D)

    k_exp = repeat_kv(k, Q_H // KV_H)
    v_exp = repeat_kv(v, Q_H // KV_H)

    logits = torch.matmul(q.unsqueeze(2), k_exp.transpose(-2, -1)) * scale

    mask = torch.arange(S, device=device).expand(B, -1) >= cache_lens.unsqueeze(1)
    logits.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    attn = logits.softmax(dim=-1)
    attn.masked_fill_(mask.unsqueeze(1).unsqueeze(2), 0.0)
    return torch.matmul(attn, v_exp).squeeze(2)


B, Q_H, KV_H, MAX_S, D = 4, 32, 8, 2048, 128

torch.manual_seed(42)
q = torch.randn(B, Q_H, D, device=device, dtype=dtype)
k = torch.randn(B, KV_H, MAX_S, D, device=device, dtype=dtype)
v = torch.randn(B, KV_H, MAX_S, D, device=device, dtype=dtype)
cache_lens = torch.randint(MAX_S // 4, MAX_S + 1, (B,), device=device, dtype=torch.int32)

q_fp8, sq = quantize_to_fp8(q.float())
k_fp8, sk = quantize_to_fp8(k.float())
v_fp8, sv = quantize_to_fp8(v.float())

tri_out = flash_decode_gqa_hf_fp8(q_fp8, k_fp8, v_fp8, cache_lens, sq, sk, sv, dtype)

# Reference in fp16 (dequantized inputs to match kernel's effective precision)
q_deq = q_fp8.to(torch.float32) * sq
k_deq = k_fp8.to(torch.float32) * sk
v_deq = v_fp8.to(torch.float32) * sv
ref_out = ref_attention_fp16(q_deq.half(), k_deq.half(), v_deq.half(), cache_lens)

torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
print("✅ FP8 GQA Decode Kernel Match Confirmed!")

# Also sanity-check with split-KV path
cache_lens_long = torch.full((B,), 5000, device=device, dtype=torch.int32)
MAX_S_LONG = 6000
k_long = torch.randn(B, KV_H, MAX_S_LONG, D, device=device, dtype=dtype)
v_long = torch.randn(B, KV_H, MAX_S_LONG, D, device=device, dtype=dtype)

k_long_fp8, sk_l = quantize_to_fp8(k_long.float())
v_long_fp8, sv_l = quantize_to_fp8(v_long.float())

tri_out_long = flash_decode_gqa_hf_fp8(q_fp8, k_long_fp8, v_long_fp8, cache_lens_long, sq, sk_l, sv_l, dtype)

k_deq_long = k_long_fp8.to(torch.float32) * sk_l
v_deq_long = v_long_fp8.to(torch.float32) * sv_l
ref_out_long = ref_attention_fp16(q_deq.half(), k_deq_long.half(), v_deq_long.half(), cache_lens_long)

torch.testing.assert_close(tri_out_long, ref_out_long, atol=1e-3, rtol=1e-3)
print("✅ FP8 GQA Decode Split-KV Path Match Confirmed!")
