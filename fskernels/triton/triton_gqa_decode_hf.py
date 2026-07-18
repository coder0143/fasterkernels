import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as native_eager_attention_forward
from typing import Optional, Tuple
import triton
import triton.language as tl

# Controls split-KV at capture time (set by CUDAGraphRunner before capture)
_DECODE_NUM_SPLITS: int = 1


def set_num_splits(n: int):
    global _DECODE_NUM_SPLITS
    _DECODE_NUM_SPLITS = n


# First pass — combined single-pass / split-KV kernel with sliding-window support
@triton.jit
def _flash_decode_gqa_hf_dense_kernel(
    Q, K_Cache, V_Cache, Cache_Lens, Start_K_Offsets, Out,
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_wb, stride_wh, stride_ws, stride_wd,
    NUM_Q_HEADS, NUM_KV_HEADS, GROUP_SIZE, MAX_SEQLEN_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_hd = tl.program_id(1)
    split_id = tl.program_id(2)

    start_q_head = kv_hd * GROUP_SIZE
    offs_h = start_q_head + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_mask = (offs_h[:, None] < (start_q_head + GROUP_SIZE)) & (offs_h[:, None] < NUM_Q_HEADS)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    cur_seqlen_k = tl.load(Cache_Lens + bid)
    start_k_global = tl.load(Start_K_Offsets + bid)  # 0 for full attn, >0 for sliding window
    effective_len = cur_seqlen_k - start_k_global

    sm_scale = 1.0 / tl.math.sqrt(HEAD_DIM * 1.0)

    tokens_per_split = (effective_len + NUM_SPLITS - 1) // NUM_SPLITS
    start_n_split = start_k_global + split_id * tokens_per_split
    end_n_split = tl.minimum(start_n_split + tokens_per_split, cur_seqlen_k)

    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    if start_n_split < end_n_split:
        for start_n in range(start_n_split, end_n_split, BLOCK_N):
            k_ptrs = (K_Cache + bid * stride_kb + kv_hd * stride_kh +
                      (start_n + offs_n)[:, None] * stride_ks + offs_d[None, :] * stride_kd)
            k_mask = ((start_n + offs_n)[:, None] < end_n_split)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            s = tl.dot(q, tl.trans(k))
            s *= sm_scale
            s = tl.where((start_n + offs_n)[None, :] < end_n_split, s, float('-inf'))

            m_ij = tl.max(s, axis=1)
            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp(m_i - m_next)
            p = tl.math.exp(s - m_next[:, None])

            acc = acc * alpha[:, None]
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v_ptrs = (V_Cache + bid * stride_vb + kv_hd * stride_vh +
                      (start_n + offs_n)[:, None] * stride_vs + offs_d[None, :] * stride_vd)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

            acc = tl.dot(p.to(q.dtype), v, acc=acc)
            m_i = m_next

    if NUM_SPLITS == 1:
        acc = acc / l_i[:, None]
        out_ptrs = Out + bid * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=q_mask)
    else:
        w_offset = bid * stride_wb + kv_hd * stride_wh + split_id * stride_ws
        w_out_ptrs = Gl_Out_Buf + w_offset + offs_h[:, None] * stride_wd + offs_d[None, :]
        tl.store(w_out_ptrs, acc.to(Gl_Out_Buf.type.element_ty), mask=q_mask)

        w_l_ptrs = Gl_L_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (offs_h * NUM_SPLITS) + split_id
        w_m_ptrs = Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (offs_h * NUM_SPLITS) + split_id
        tl.store(w_l_ptrs, l_i, mask=(offs_h < NUM_Q_HEADS))
        tl.store(w_m_ptrs, m_i, mask=(offs_h < NUM_Q_HEADS))


# Second pass — global split-KV reduction kernel
@triton.jit
def _flash_decode_gqa_hf_reduction_kernel(
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf, Out,
    stride_wb, stride_wh, stride_ws, stride_wd,
    stride_ob, stride_oh, stride_od,
    NUM_Q_HEADS, NUM_KV_HEADS, NUM_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    kv_hd = hid // (NUM_Q_HEADS // NUM_KV_HEADS)
    w_base = bid * stride_wb + kv_hd * stride_wh

    m_max = -float('inf')
    l_sum = 0.0
    acc_final = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for s in range(0, NUM_SPLITS):
        m_s = tl.load(Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        m_max = tl.maximum(m_max, m_s)

    for s in range(0, NUM_SPLITS):
        m_s = tl.load(Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        l_s = tl.load(Gl_L_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        alpha = tl.math.exp(m_s - m_max)
        l_sum += l_s * alpha

        w_out_ptrs = Gl_Out_Buf + w_base + s * stride_ws + hid * stride_wd + offs_d
        p_out = tl.load(w_out_ptrs)
        acc_final += p_out * alpha

    acc_final = acc_final / l_sum
    out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_d
    tl.store(out_ptrs, acc_final.to(Out.type.element_ty))


_GLOBAL_SPLIT_OUT = None
_GLOBAL_SPLIT_L = None
_GLOBAL_SPLIT_M = None
_GLOBAL_START_OFFSETS = None


def flash_decode_gqa_hf_dense_optimized(q, k_cache, v_cache, cache_lens, start_k_offsets=None, num_splits=None):
    global _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M, _GLOBAL_START_OFFSETS

    batch_size, q_heads, head_size = q.shape
    kv_heads = k_cache.shape[1]
    max_seqlen_kv = k_cache.shape[2]
    group_size = q_heads // kv_heads
    block_h = max(16, triton.next_power_of_2(group_size))

    if num_splits is None:
        num_splits = _DECODE_NUM_SPLITS

    # Pre-allocate static start_k_offsets buffer if not provided
    if start_k_offsets is None:
        if _GLOBAL_START_OFFSETS is None or _GLOBAL_START_OFFSETS.shape[0] < batch_size:
            _GLOBAL_START_OFFSETS = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
        start_k_offsets = _GLOBAL_START_OFFSETS[:batch_size]

    out = torch.empty_like(q)

    if num_splits == 1:
        grid = (batch_size, kv_heads, 1)
        _flash_decode_gqa_hf_dense_kernel[grid](
            q, k_cache, v_cache, cache_lens, start_k_offsets, out,
            None, None, None,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            0, 0, 0, 0,
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=1,
            num_warps=4, num_stages=3,
        )
    else:
        if _GLOBAL_SPLIT_OUT is None or _GLOBAL_SPLIT_OUT.shape[0] < batch_size:
            _GLOBAL_SPLIT_OUT = torch.zeros((batch_size, kv_heads, num_splits, q_heads, head_size), dtype=q.dtype, device=q.device)
            _GLOBAL_SPLIT_L = torch.zeros((batch_size, q_heads * num_splits), dtype=torch.float32, device=q.device)
            _GLOBAL_SPLIT_M = torch.full((batch_size, q_heads * num_splits), float('-inf'), dtype=torch.float32, device=q.device)

        grid_p1 = (batch_size, kv_heads, num_splits)
        _flash_decode_gqa_hf_dense_kernel[grid_p1](
            q, k_cache, v_cache, cache_lens, start_k_offsets, out,
            _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            _GLOBAL_SPLIT_OUT.stride(0), _GLOBAL_SPLIT_OUT.stride(1),
            _GLOBAL_SPLIT_OUT.stride(2), _GLOBAL_SPLIT_OUT.stride(3),
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=num_splits,
            num_warps=4, num_stages=3,
        )

        grid_p2 = (batch_size, q_heads)
        _flash_decode_gqa_hf_reduction_kernel[grid_p2](
            _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M, out,
            _GLOBAL_SPLIT_OUT.stride(0), _GLOBAL_SPLIT_OUT.stride(1),
            _GLOBAL_SPLIT_OUT.stride(2), _GLOBAL_SPLIT_OUT.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, NUM_SPLITS=num_splits, HEAD_DIM=head_size,
            num_warps=4,
        )

    return out


# Sliding-window pre-allocated offset buffer (per-layer reuse across steps)
_GLOBAL_SW_START_OFFSETS = None


def custom_hf_decode_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    global _GLOBAL_SW_START_OFFSETS

    bsz, q_heads, q_len, head_dim = query_states.shape
    past_key_values = kwargs.get("past_key_values", None)
    cache_position = kwargs.get("cache_position", None)  # [q_len] tensor
    sliding_window = kwargs.get("sliding_window", None)

    kv_heads = getattr(module, "num_key_value_heads", key_states.shape[1])

    if past_key_values is not None and q_len == 1 and cache_position is not None:
        q_2d = query_states.squeeze(2)  # [B, Q_H, D]

        # Derive cache_lens as a GPU tensor — safe for CUDA graph replay
        # cache_position[-1] is the index just written, so KV length = cache_position[-1] + 1
        cache_lens = (cache_position[-1:] + 1).to(torch.int32).expand(bsz).contiguous()

        if sliding_window is not None:
            # start reading from max(0, kv_len - window) — pure tensor ops
            if _GLOBAL_SW_START_OFFSETS is None or _GLOBAL_SW_START_OFFSETS.shape[0] < bsz:
                _GLOBAL_SW_START_OFFSETS = torch.zeros(bsz, dtype=torch.int32, device=q_2d.device)
            start_offsets = torch.clamp(cache_lens - sliding_window, min=0)
            eff_cache_lens = torch.clamp(cache_lens, max=sliding_window)
            attn_output = flash_decode_gqa_hf_dense_optimized(q_2d, key_states, value_states, eff_cache_lens, start_offsets)
        else:
            attn_output = flash_decode_gqa_hf_dense_optimized(q_2d, key_states, value_states, cache_lens)

        return attn_output, None

    return native_eager_attention_forward(
        module, query_states, key_states, value_states, attention_mask, scaling, dropout, **kwargs
    )


if __name__ == "__main__":
    from triton.testing import do_bench
    B, Q_H, KV_H, MAX_S, D = 2, 32, 8, 8192, 128
    q = torch.randn(B, Q_H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    lens = torch.full((B,), MAX_S - 100, device="cuda", dtype=torch.int32)
    res = flash_decode_gqa_hf_dense_optimized(q, k, v, lens)
    ms = do_bench(lambda: flash_decode_gqa_hf_dense_optimized(q, k, v, lens))
    print(f"Output: {res.shape}  |  {ms:.4f} ms")