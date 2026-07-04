import itertools
import torch
import triton
import triton.language as tl

def generate_configs():
    block_m_choices = [64, 128, 256]
    block_n_choices = [64, 128, 256]
    
    # Execution wave configurations
    warp_choices = [4, 8]
    stage_choices = [2, 3, 4, 5]
    
    configs = []
    
    # Generate Cartesian product of all parameters
    for bm, bn, warps, stages in itertools.product(
        block_m_choices, block_n_choices, warp_choices, stage_choices
    ):
        # Hardware Guardrails: 
        # Skip very large tiles
        if (bm == 256 and bn == 256) and stages > 2:
            continue
        # Skip very small thread worloads 
        if (bm * bn) // (warps * 32) < 16:
            continue
            
        configs.append(
            triton.Config(
                {'BLOCK_M': bm, 'BLOCK_N': bn}, 
                num_warps=warps, 
                num_stages=stages
            )
        )
    return configs

# best config on L4 (sm89, 24gb vram) := BLOCK_M: 128, BLOCK_N: 128, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None;
# @triton.autotune(
#     configs=generate_configs(),
#     key=['SEQ_LEN_Q', 'SEQ_LEN_K', 'HEAD_DIM'],
# )
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH, NUM_HEADS, SEQ_LEN_Q, SEQ_LEN_K,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    off_b = off_bh // NUM_HEADS
    off_h = off_bh % NUM_HEADS

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Local accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN_Q, other=0.0)

    for start_n in range(0, SEQ_LEN_K, BLOCK_N):
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < SEQ_LEN_K, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < SEQ_LEN_K, other=0.0)

        # S = QK^T
        s = tl.dot(q, tl.trans(k))
        s *= sm_scale
        # Zero out masked-out attention weights (vital for arbitrary sequences)
        s = tl.where((start_n + offs_n)[None, :] < SEQ_LEN_K, s, float('-inf'))

        # Find maximum values in current block row
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        # Calculate strict scaling values
        alpha = tl.math.exp(m_i - m_next)
        p = tl.math.exp(s - m_next[:, None])

        # Step 1: Scale previous accumulated value tracking
        acc = acc * alpha[:, None]
        
        # Step 2: Scale and accumulate previous denominator sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Step 3: Compute current dot product and combine with previous accumulation
        acc = tl.dot(p.to(q.dtype), v, acc=acc)
        
        # Update running maximum state
        m_i = m_next

    # Normalize accumulation at the end of the loop reduction
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < SEQ_LEN_Q)


def flash_attention_2_fwd(q, k, v, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
        
    b, h, s_q, d = q.shape
    s_k = k.shape[2]
    
    out = torch.empty_like(q)
    
    grid = lambda META: (
        triton.cdiv(s_q, META['BLOCK_M']),
        b * h
    )
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        b, h, s_q, s_k,
        HEAD_DIM=d,
        # Below params are post autotuning
        BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2,        
    )
    return out

