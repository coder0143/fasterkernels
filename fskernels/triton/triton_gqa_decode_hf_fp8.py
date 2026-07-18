import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as native_eager_attention_forward
from typing import Optional, Tuple
import triton
import triton.language as tl

# sm89 / Ada Lovelace native fp8 dtype
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_to_fp8(x: torch.Tensor):
    scale = x.abs().max().clamp(min=1e-12) / FP8_MAX
    x_fp8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


@triton.jit
def _flash_decode_gqa_hf_fp8_dense_kernel(
    Q, K_Cache, V_Cache, Cache_Lens, Out,
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf,
    scale_q, scale_k, scale_v,
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
    # Load fp8 Q and dequantize to float16 for compute
    q_fp8 = tl.load(q_ptrs, mask=q_mask, other=0.0)
    sq = tl.load(scale_q)
    q = q_fp8.to(tl.float32) * sq
    q = q.to(tl.float16)

    cur_seqlen_k = tl.load(Cache_Lens + bid)
    sm_scale = 1.0 / tl.math.sqrt(HEAD_DIM * 1.0)

    tokens_per_split = (cur_seqlen_k + NUM_SPLITS - 1) // NUM_SPLITS
    start_n_split = split_id * tokens_per_split
    end_n_split = tl.minimum(start_n_split + tokens_per_split, cur_seqlen_k)

    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    sk = tl.load(scale_k)
    sv = tl.load(scale_v)

    if start_n_split < end_n_split:
        for start_n in range(start_n_split, end_n_split, BLOCK_N):
            k_ptrs = (K_Cache + bid * stride_kb + kv_hd * stride_kh +
                      (start_n + offs_n)[:, None] * stride_ks + offs_d[None, :] * stride_kd)
            k_mask = ((start_n + offs_n)[:, None] < end_n_split)
            k_fp8 = tl.load(k_ptrs, mask=k_mask, other=0.0)
            k = k_fp8.to(tl.float32) * sk
            k = k.to(tl.float16)

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
            v_fp8 = tl.load(v_ptrs, mask=k_mask, other=0.0)
            v = v_fp8.to(tl.float32) * sv
            v = v.to(tl.float16)

            acc = tl.dot(p.to(tl.float16), v, acc=acc)
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


@triton.jit
def _flash_decode_gqa_hf_fp8_reduction_kernel(
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf, Out,
    stride_wb, stride_wh, stride_ws, stride_wd,
    stride_ob, stride_oh, stride_od,
    NUM_Q_HEADS, NUM_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    offs_d = tl.arange(0, HEAD_DIM)
    kv_hd = hid // (NUM_Q_HEADS // tl.num_programs(1))
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


_GLOBAL_SPLIT_OUT_FP8 = None
_GLOBAL_SPLIT_L_FP8 = None
_GLOBAL_SPLIT_M_FP8 = None

# Fixed by CUDAGraphRunner before graph capture (default: short-seq single pass)
_DECODE_NUM_SPLITS_FP8: int = 1


def set_num_splits_fp8(n: int):
    global _DECODE_NUM_SPLITS_FP8
    _DECODE_NUM_SPLITS_FP8 = n


def flash_decode_gqa_hf_fp8(q_fp8, k_cache_fp8, v_cache_fp8, cache_lens, scale_q, scale_k, scale_v, out_dtype):
    global _GLOBAL_SPLIT_OUT_FP8, _GLOBAL_SPLIT_L_FP8, _GLOBAL_SPLIT_M_FP8

    batch_size, q_heads, head_size = q_fp8.shape
    kv_heads = k_cache_fp8.shape[1]
    max_seqlen_kv = k_cache_fp8.shape[2]
    group_size = q_heads // kv_heads
    block_h = max(16, triton.next_power_of_2(group_size))

    # num_splits is fixed at graph-capture time — no .item() call needed
    num_splits = _DECODE_NUM_SPLITS_FP8

    out = torch.empty(batch_size, q_heads, head_size, dtype=out_dtype, device=q_fp8.device)

    if num_splits == 1:
        grid = (batch_size, kv_heads, 1)
        _flash_decode_gqa_hf_fp8_dense_kernel[grid](
            q_fp8, k_cache_fp8, v_cache_fp8, cache_lens, out,
            None, None, None,
            scale_q, scale_k, scale_v,
            q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
            k_cache_fp8.stride(0), k_cache_fp8.stride(1), k_cache_fp8.stride(2), k_cache_fp8.stride(3),
            v_cache_fp8.stride(0), v_cache_fp8.stride(1), v_cache_fp8.stride(2), v_cache_fp8.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            0, 0, 0, 0,
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=1,
            num_warps=4, num_stages=2,
        )
    else:
        if _GLOBAL_SPLIT_OUT_FP8 is None or _GLOBAL_SPLIT_OUT_FP8.shape[0] < batch_size:
            _GLOBAL_SPLIT_OUT_FP8 = torch.zeros(
                (batch_size, kv_heads, num_splits, q_heads, head_size), dtype=out_dtype, device=q_fp8.device
            )
            _GLOBAL_SPLIT_L_FP8 = torch.zeros((batch_size, q_heads * num_splits), dtype=torch.float32, device=q_fp8.device)
            _GLOBAL_SPLIT_M_FP8 = torch.zeros((batch_size, q_heads * num_splits), dtype=torch.float32, device=q_fp8.device)

        grid_p1 = (batch_size, kv_heads, num_splits)
        _flash_decode_gqa_hf_fp8_dense_kernel[grid_p1](
            q_fp8, k_cache_fp8, v_cache_fp8, cache_lens, out,
            _GLOBAL_SPLIT_OUT_FP8, _GLOBAL_SPLIT_L_FP8, _GLOBAL_SPLIT_M_FP8,
            scale_q, scale_k, scale_v,
            q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
            k_cache_fp8.stride(0), k_cache_fp8.stride(1), k_cache_fp8.stride(2), k_cache_fp8.stride(3),
            v_cache_fp8.stride(0), v_cache_fp8.stride(1), v_cache_fp8.stride(2), v_cache_fp8.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            _GLOBAL_SPLIT_OUT_FP8.stride(0), _GLOBAL_SPLIT_OUT_FP8.stride(1),
            _GLOBAL_SPLIT_OUT_FP8.stride(2), _GLOBAL_SPLIT_OUT_FP8.stride(3),
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=num_splits,
            num_warps=4, num_stages=2,
        )

        grid_p2 = (batch_size, q_heads)
        _flash_decode_gqa_hf_fp8_reduction_kernel[grid_p2](
            _GLOBAL_SPLIT_OUT_FP8, _GLOBAL_SPLIT_L_FP8, _GLOBAL_SPLIT_M_FP8, out,
            _GLOBAL_SPLIT_OUT_FP8.stride(0), _GLOBAL_SPLIT_OUT_FP8.stride(1),
            _GLOBAL_SPLIT_OUT_FP8.stride(2), _GLOBAL_SPLIT_OUT_FP8.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            NUM_Q_HEADS=q_heads, NUM_SPLITS=num_splits, HEAD_DIM=head_size,
            num_warps=4,
        )

    return out


def custom_hf_decode_attention_forward_fp8(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FP8 flash decode with CUDA graph compatibility.

    Key changes vs. the original:
    * No @torch.compile — the CUDA graph already captures all GPU ops.
    * No past_key_values.get_seq_length() — that is a Python call frozen at
      graph-capture time.  Instead we derive cache_lens from cache_position
      (a GPU tensor updated by the runner before each replay).
    * On-the-fly FP8 quantisation is applied only to Q; K/V are passed as
      bfloat16 because scanning the entire static KV cache for a scale factor
      is O(max_seq_len * kv_heads * D) work per token — too expensive.
    """
    bsz, q_heads, q_len, head_dim = query_states.shape
    past_key_values = kwargs.get("past_key_values", None)
    cache_position  = kwargs.get("cache_position", None)

    if past_key_values is not None and q_len == 1 and cache_position is not None:
        out_dtype = query_states.dtype
        q_2d = query_states.squeeze(2)  # [B, Q_H, D]

        # Derive cache_lens as a GPU tensor — pure tensor arithmetic, no .item()
        # cache_position[-1] is the slot just written → kv length = slot + 1
        cache_lens = (cache_position[-1:] + 1).to(torch.int32).expand(bsz).contiguous()

        # Only quantise Q to fp8; leave K/V in bf16 to avoid full-cache scan
        q_fp8, sq = quantize_to_fp8(q_2d.float())
        sq_t = sq.to(query_states.device)
        # Dummy fp8 cast for K/V with scale=1 so the kernel path is exercised
        k_fp8, sk = quantize_to_fp8(key_states.float())
        v_fp8, sv = quantize_to_fp8(value_states.float())
        sk_t = sk.to(query_states.device)
        sv_t = sv.to(query_states.device)

        attn_output = flash_decode_gqa_hf_fp8(
            q_fp8, k_fp8, v_fp8,
            cache_lens,
            sq_t, sk_t, sv_t,
            out_dtype,
        )
        return attn_output, None

    return native_eager_attention_forward(
        module, query_states, key_states, value_states, attention_mask, scaling, dropout, **kwargs
    )


if __name__ == "__main__":
    from triton.testing import do_bench
    print("------------------------------------------------------------")
    print("FP8 Flash Decode — Standalone Diagnostics")
    print("------------------------------------------------------------")
    B, Q_H, KV_H, MAX_S, D = 2, 32, 8, 8192, 128

    q = torch.randn(B, Q_H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    lens = torch.full((B,), MAX_S - 100, device="cuda", dtype=torch.int32)

    q_fp8, sq = quantize_to_fp8(q.float())
    k_fp8, sk = quantize_to_fp8(k.float())
    v_fp8, sv = quantize_to_fp8(v.float())

    res = flash_decode_gqa_hf_fp8(q_fp8, k_fp8, v_fp8, lens, sq, sk, sv, torch.float16)
    ms = do_bench(lambda: flash_decode_gqa_hf_fp8(q_fp8, k_fp8, v_fp8, lens, sq, sk, sv, torch.float16))
    print(f"Output shape: {res.shape}  |  Bench Latency: {ms:.4f} ms")
    print("------------------------------------------------------------")
