import math
import random
import torch
from fskernels.paged_attn import paged_flash_decode, PAGE_SIZE

device = "cuda"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def ref_paged_attention(q, k_padded, v_padded, seq_lens):
    B, Q_H, D = q.shape
    _, KV_H, max_len, _ = k_padded.shape
    G = Q_H // KV_H
    scale = 1.0 / math.sqrt(D)

    k_rep = repeat_kv(k_padded, G)
    v_rep = repeat_kv(v_padded, G)

    out = torch.zeros_like(q)
    for b in range(B):
        slen = int(seq_lens[b].item())
        q_b = q[b : b + 1].unsqueeze(2)
        k_b = k_rep[b : b + 1, :, :slen, :]
        v_b = v_rep[b : b + 1, :, :slen, :]

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        res = torch.matmul(probs, v_b).squeeze(2).squeeze(0)
        out[b] = res

    return out


def test_triton_paged_gqa_decode(dtype=torch.float16):
    B = 4
    Q_H = 32
    KV_H = 8
    D = 128
    seq_lens_list = [64, 512, 1024, 250]
    max_len = max(seq_lens_list)

    torch.manual_seed(42)
    random.seed(42)

    q = torch.randn(B, Q_H, D, device=device, dtype=dtype)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    # Unpaged reference tensors
    k_padded = torch.zeros(B, KV_H, max_len, D, device=device, dtype=dtype)
    v_padded = torch.zeros(B, KV_H, max_len, D, device=device, dtype=dtype)

    for b in range(B):
        slen = seq_lens_list[b]
        k_padded[b, :, :slen, :] = torch.randn(KV_H, slen, D, device=device, dtype=dtype)
        v_padded[b, :, :slen, :] = torch.randn(KV_H, slen, D, device=device, dtype=dtype)

    # Allocate physical pages non-contiguously to verify block table mapping
    total_blocks_needed = sum((slen + PAGE_SIZE - 1) // PAGE_SIZE for slen in seq_lens_list)
    num_pages = total_blocks_needed + 10
    max_blocks_per_seq = (max_len + PAGE_SIZE - 1) // PAGE_SIZE

    k_pool = torch.zeros(num_pages, KV_H, PAGE_SIZE, D, device=device, dtype=dtype)
    v_pool = torch.zeros_like(k_pool)
    block_table = torch.full((B, max_blocks_per_seq), -1, dtype=torch.int32, device=device)

    available_pages = list(range(num_pages))
    random.shuffle(available_pages)

    for b in range(B):
        slen = seq_lens_list[b]
        num_b_pages = (slen + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(num_b_pages):
            pp = available_pages.pop()
            block_table[b, p] = pp
            start_t = p * PAGE_SIZE
            end_t = min(start_t + PAGE_SIZE, slen)
            tok_len = end_t - start_t

            k_pool[pp, :, :tok_len, :] = k_padded[b, :, start_t:end_t, :]
            v_pool[pp, :, :tok_len, :] = v_padded[b, :, start_t:end_t, :]

    # Kernel execution
    tri_out = paged_flash_decode(q, k_pool, v_pool, block_table, seq_lens, PAGE_SIZE)

    # PyTorch reference
    ref_out = ref_paged_attention(q, k_padded, v_padded, seq_lens)

    # Match verification
    torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
    print(f"✅ Paged GQA Decode Kernel Match Confirmed! ({dtype})")


if __name__ == "__main__":
    test_triton_paged_gqa_decode(torch.float16)
    test_triton_paged_gqa_decode(torch.bfloat16)
