import torch
import math
from fskernels.cuda.cuda_fd_gqa import flash_decode_gqa_varlen

device = "cuda"
dtype = torch.float16

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def ref_attention(q, k, v, k_seqlens, q_heads, sink=None):
    batch_size, kv_heads, max_seqlen, head_size = k.shape
    softmax_scale = 1.0 / math.sqrt(head_size)

    k = repeat_kv(k, q_heads // kv_heads)
    v = repeat_kv(v, q_heads // kv_heads)
    logits = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) * softmax_scale 

    mask = torch.arange(max_seqlen, device=q.device).expand(batch_size, -1) >= k_seqlens.unsqueeze(1)
    logits.masked_fill_(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    if sink is None:
        attn_weights = logits.softmax(dim=-1)
    else:
        sink_expanded = sink.view(1, q_heads, 1, 1)
        logits_max = torch.maximum(logits.max(dim=-1, keepdim=True).values, sink_expanded)
        exp_logits = torch.exp(logits - logits_max)
        attn_weights = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + torch.exp(sink_expanded - logits_max))

    attn_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(2), 0.0)
    output = torch.matmul(attn_weights.to(v.dtype), v).squeeze(2)
    return output

# Setup configuration scenario mimicking TileLang
B, Q_H, KV_H, MAX_S, D = 4, 32, 8, 4096, 128

torch.manual_seed(0)
k_seqlens = torch.randint(MAX_S // 4, MAX_S + 1, size=(B,), device=device)
cu_seqlens_k = torch.zeros(B + 1, device=device, dtype=torch.int32)
cu_seqlens_k[1:] = torch.cumsum(k_seqlens, dim=0).to(torch.int32)
total_k_tokens = cu_seqlens_k[-1].item()

# Generate input tensors
q = torch.randn(B, Q_H, D, device=device, dtype=dtype)
k_varlen = torch.randn(total_k_tokens, KV_H, D, device=device, dtype=dtype)
v_varlen = torch.randn(total_k_tokens, KV_H, D, device=device, dtype=dtype)
sink = torch.randn(Q_H, device=device, dtype=torch.float32) * 0.1

# Run Triton Decoding Kernel
tri_out = flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, sink=sink)

# Formulate padded reference tracking
actual_max = int(k_seqlens.max())
k_padded = torch.zeros(B, KV_H, actual_max, D, device=device, dtype=dtype)
v_padded = torch.zeros(B, KV_H, actual_max, D, device=device, dtype=dtype)
for i in range(B):
    seq_len = k_seqlens[i].item()
    k_padded[i, :, :seq_len] = k_varlen[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)
    v_padded[i, :, :seq_len] = v_varlen[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1)

ref_out = ref_attention(q, k_padded, v_padded, k_seqlens, Q_H, sink)

# Precision confirmation check
torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
print("✅ CUDA GQA Varlen Decode Kernel Match Confirmed!")
