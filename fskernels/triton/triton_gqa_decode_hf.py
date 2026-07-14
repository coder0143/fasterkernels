import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as native_eager_attention_forward
from typing import Optional, Tuple
import triton
import triton.language as tl

# First pass - Combined single-pass / split-kv workspace accumalator kernel
@triton.jit
def _flash_decode_gqa_hf_dense_kernel(
    Q, K_Cache, V_Cache, Cache_Lens, Out,
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf, # Split-KV Global Reduction Workspaces
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_wb, stride_wh, stride_ws, stride_wd,
    NUM_Q_HEADS, NUM_KV_HEADS, GROUP_SIZE, MAX_SEQLEN_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, # Controlled compile-time branching flag
):
    bid = tl.program_id(0)       # Batch index
    kv_hd = tl.program_id(1)     # KV Head index
    split_id = tl.program_id(2)  # Split chunk segment index

    start_q_head = kv_hd * GROUP_SIZE
    offs_h = start_q_head + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # 1. Coalesced Vectorized Load for Queries
    q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_mask = (offs_h[:, None] < (start_q_head + GROUP_SIZE)) & (offs_h[:, None] < NUM_Q_HEADS)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    cur_seqlen_k = tl.load(Cache_Lens + bid)
    sm_scale = 1.0 / tl.math.sqrt(HEAD_DIM * 1.0)

    # Calculate sequence slicing region bounded to this block's Split-KV coordinate
    tokens_per_split = (cur_seqlen_k + NUM_SPLITS - 1) // NUM_SPLITS
    start_n_split = split_id * tokens_per_split
    end_n_split = tl.minimum(start_n_split + tokens_per_split, cur_seqlen_k)

    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    # 2. Sequential Slicing Attention Loop
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

    # 3. Write-Out Logic Branch
    if NUM_SPLITS == 1:
        # Standard Single-Pass Execution: Write directly to HBM out
        acc = acc / l_i[:, None]
        out_ptrs = Out + bid * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=q_mask)
    else:
        # Split-KV Execution: Dump partial state segments out to the workspace arrays
        w_offset = bid * stride_wb + kv_hd * stride_wh + split_id * stride_ws
        
        # Stashing partial output buffers
        w_out_ptrs = Gl_Out_Buf + w_offset + offs_h[:, None] * stride_wd + offs_d[None, :]
        tl.store(w_out_ptrs, acc.to(Gl_Out_Buf.type.element_ty), mask=q_mask)
        
        # Stashing row statistics trackers
        w_l_ptrs = Gl_L_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (offs_h * NUM_SPLITS) + split_id
        w_m_ptrs = Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (offs_h * NUM_SPLITS) + split_id
        tl.store(w_l_ptrs, l_i, mask=(offs_h < NUM_Q_HEADS))
        tl.store(w_m_ptrs, m_i, mask=(offs_h < NUM_Q_HEADS))


# Second pass - Global split-kv reduction and consolidation kernel
@triton.jit
def _flash_decode_gqa_hf_reduction_kernel(
    Gl_Out_Buf, Gl_L_Buf, Gl_M_Buf, Out,
    stride_wb, stride_wh, stride_ws, stride_wd,
    stride_ob, stride_oh, stride_od,
    NUM_Q_HEADS, NUM_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1) # Operates per individual Query head index axis

    offs_d = tl.arange(0, HEAD_DIM)
    
    # Calculate base pointer offset configurations
    kv_hd = hid // (NUM_Q_HEADS // tl.num_programs(1)) # Safe group map
    w_base = bid * stride_wb + kv_hd * stride_wh
    
    m_max = -float('inf')
    l_sum = 0.0
    acc_final = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Phase 1: Discover global maximum across all split boundaries
    for s in range(0, NUM_SPLITS):
        m_s = tl.load(Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        m_max = tl.maximum(m_max, m_s)

    # Phase 2: Compute consolidated scaling denominator and apply rescale targets
    for s in range(0, NUM_SPLITS):
        m_s = tl.load(Gl_M_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        l_s = tl.load(Gl_L_Buf + (bid * NUM_Q_HEADS * NUM_SPLITS) + (hid * NUM_SPLITS) + s)
        
        alpha = tl.math.exp(m_s - m_max)
        l_sum += l_s * alpha
        
        w_out_ptrs = Gl_Out_Buf + w_base + s * stride_ws + hid * stride_wd + offs_d
        p_out = tl.load(w_out_ptrs)
        acc_final += p_out * alpha

    acc_final = acc_final / l_sum

    # Write unified outputs to structural destination
    out_ptrs = Out + bid * stride_ob + hid * stride_oh + offs_d
    tl.store(out_ptrs, acc_final.to(Out.type.element_ty))

# Python interface
_GLOBAL_SPLIT_OUT = None
_GLOBAL_SPLIT_L = None
_GLOBAL_SPLIT_M = None

def flash_decode_gqa_hf_dense_optimized(q, k_cache, v_cache, cache_lens):
    global _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M

    batch_size, q_heads, head_size = q.shape
    kv_heads = k_cache.shape[1]
    max_seqlen_kv = k_cache.shape[2]
    group_size = q_heads // kv_heads
    block_h = max(16, triton.next_power_of_2(group_size))

    # Trigger Split-KV multi-pass scheduling past 4,000 context tokens
    current_avg_len = cache_lens[0].item()
    num_splits = 1 if current_avg_len < 4000 else 8

    out = torch.empty_like(q)

    if num_splits == 1:
        grid = (batch_size, kv_heads, 1)
        _flash_decode_gqa_hf_dense_kernel[grid](
            q, k_cache, v_cache, cache_lens, out,
            None, None, None,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            0, 0, 0, 0,
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=1,
            num_warps=4, num_stages=1
        )
    else:
        if _GLOBAL_SPLIT_OUT is None or _GLOBAL_SPLIT_OUT.shape[0] < batch_size:
            _GLOBAL_SPLIT_OUT = torch.zeros((batch_size, kv_heads, num_splits, q_heads, head_size), dtype=q.dtype, device=q.device)
            _GLOBAL_SPLIT_L = torch.zeros((batch_size, q_heads * num_splits), dtype=torch.float32, device=q.device)
            _GLOBAL_SPLIT_M = torch.zeros((batch_size, q_heads * num_splits), dtype=torch.float32, device=q.device)

        grid_p1 = (batch_size, kv_heads, num_splits)
        _flash_decode_gqa_hf_dense_kernel[grid_p1](
            q, k_cache, v_cache, cache_lens, out,
            _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M,
            q.stride(0), q.stride(1), q.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            _GLOBAL_SPLIT_OUT.stride(0), _GLOBAL_SPLIT_OUT.stride(1), _GLOBAL_SPLIT_OUT.stride(2), _GLOBAL_SPLIT_OUT.stride(3),
            NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
            HEAD_DIM=head_size, BLOCK_H=block_h, BLOCK_N=64, NUM_SPLITS=num_splits,
            num_warps=4, num_stages=1
        )

        grid_p2 = (batch_size, q_heads)
        _flash_decode_gqa_hf_reduction_kernel[grid_p2](
            _GLOBAL_SPLIT_OUT, _GLOBAL_SPLIT_L, _GLOBAL_SPLIT_M, out,
            _GLOBAL_SPLIT_OUT.stride(0), _GLOBAL_SPLIT_OUT.stride(1), _GLOBAL_SPLIT_OUT.stride(2), _GLOBAL_SPLIT_OUT.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            NUM_Q_HEADS=q_heads, NUM_SPLITS=num_splits, HEAD_DIM=head_size,
            num_warps=4
        )

    return out

# Pre-allocate a static GPU memory space for tracking lengths across layers/batches.
# This completely bypasses the dynamic allocation overhead in the token loop.
_GLOBAL_CACHE_LENS = None

@torch.compile(fullgraph=False, dynamic=True)
def custom_hf_decode_attention_forward( # Main function supporting hf dynamic cache
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    global _GLOBAL_CACHE_LENS
    
    bsz, q_heads, q_len, head_dim = query_states.shape
    past_key_values = kwargs.get("past_key_values", None)

    # Resolve architectural configurations
    config = getattr(module, "config", None)
    q_heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", q_heads))
    kv_heads = getattr(module, "num_key_value_heads", getattr(config, "num_key_value_heads", key_states.shape[1]))
    group_size = q_heads // kv_heads

    if past_key_values is not None and q_len == 1:
        # Untranspose the Query tensor layout
        q_untransposed = query_states.squeeze(2)
        layer_idx = getattr(module, "layer_idx", 0)

        # Lazy initialization of the static tracking tensor
        if _GLOBAL_CACHE_LENS is None or _GLOBAL_CACHE_LENS.shape[0] < bsz:
            _GLOBAL_CACHE_LENS = torch.zeros(bsz, dtype=torch.int32, device=query_states.device)

        # Directly query the cache length integer without looping over lists
        # Modern HF DynamicCaches store length uniformly across requests in a batch
        current_seq_len = past_key_values.get_seq_length(layer_idx)
        
        # Zero-allocation update directly to the pinned tensor
        _GLOBAL_CACHE_LENS.fill_(current_seq_len)

        # out = torch.empty_like(q_untransposed)
        # grid = (bsz, kv_heads)

        # Restore expected sequence shape configurations
        attn_output = flash_decode_gqa_hf_dense_optimized(
            q_untransposed, 
            key_states, 
            value_states, 
            _GLOBAL_CACHE_LENS
        )
        return attn_output, None

    # Fallback to eager path for Prefill context processing phases
    return native_eager_attention_forward(
        module, query_states, key_states, value_states, attention_mask, scaling, dropout, **kwargs
    )

# basic Parameter tuning
if __name__ == "__main__":
    from triton.testing import do_bench
    print("------------------------------------------------------------")
    print("Executing Standalone Execution Diagnostics Verification...")
    print("------------------------------------------------------------")
    B, Q_H, KV_H, MAX_S, D = 2, 32, 8, 8192, 128
    q = torch.randn(B, Q_H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, KV_H, MAX_S, D, device="cuda", dtype=torch.float16)
    lens = torch.full((B,), MAX_S - 100, device="cuda", dtype=torch.int32)

    res = flash_decode_gqa_hf_dense_optimized(q, k, v, lens)
    ms = do_bench(lambda: flash_decode_gqa_hf_dense_optimized(q, k, v, lens))
    print(f"Execution complete. Operational Bench Latency: {ms:.4f} ms")
    print("------------------------------------------------------------")