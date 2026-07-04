import itertools
import torch
import triton
import triton.language as tl

def generate_configs():
    # Power-of-two matrix block sizes up to 256
    block_n_choices = [64, 128, 256]
    
    # Execution wave configurations
    warp_choices = [4, 8]
    stage_choices = [2, 3, 4, 5]
    
    configs = []
    
    # Generate Cartesian product of all parameters
    for bn, warps, stages in itertools.product(
        block_n_choices, warp_choices, stage_choices
    ):
            
        configs.append(
            triton.Config(
                {'BLOCK_N': bn}, 
                num_warps=warps, 
                num_stages=stages
            )
        )
    return configs

# best config on L4 (sm89, 24gb vram) := BLOCK_N: 64, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None;
# @triton.autotune(
#     configs= generate_configs(),
#     key=['MAX_SEQLEN_KV', 'HEAD_DIM'],
# )
@triton.jit
def _flash_decode_gqa_varlen_kernel(
    Q, K, V, cu_seqlens_k, s_aux, Out,
    stride_qb, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    NUM_Q_HEADS, NUM_KV_HEADS, GROUP_SIZE, MAX_SEQLEN_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,  # Padded block size for GQA group
    BLOCK_N: tl.constexpr,  # Sequence chunk block size
    HAS_SINK: tl.constexpr,
):
    # Map program execution IDs
    bid = tl.program_id(0)     # Batch index
    kv_hd = tl.program_id(1)  # KV Head index

    # Track structural start positions for the GQA group
    start_q_head = kv_hd * GROUP_SIZE

    # Define memory tracking offsets
    offs_h = start_q_head + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Load entire group of Q vectors (Shape: [BLOCK_H, HEAD_DIM])
    q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_mask = (offs_h[:, None] < (start_q_head + GROUP_SIZE)) & (offs_h[:, None] < NUM_Q_HEADS)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Fetch variable sequence boundaries for current batch item
    start_k = tl.load(cu_seqlens_k + bid)
    end_k = tl.load(cu_seqlens_k + bid + 1)
    cur_seqlen_k = end_k - start_k

    # Scaling Factor (1 / sqrt(d))
    sm_scale = 1.0 / tl.math.sqrt(HEAD_DIM * 1.0)

    # Online Softmax Accumulators
    m_i = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    # Loop over the variable sequence tokens of K/V chunks
    for start_n in range(0, cur_seqlen_k, BLOCK_N):
        # Load Key block (Shape: [BLOCK_N, HEAD_DIM])
        k_ptrs = K + (start_k + start_n + offs_n)[:, None] * stride_kt + kv_hd * stride_kh + offs_d[None, :] * stride_kd
        k_mask = ((start_n + offs_n)[:, None] < cur_seqlen_k)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute Attention Logits: S = Q @ K^T
        s = tl.dot(q, tl.trans(k))
        s *= sm_scale
        
        # Mask out-of-bounds tokens
        s = tl.where((start_n + offs_n)[None, :] < cur_seqlen_k, s, float('-inf'))

        # Online Softmax Stability Protocol
        m_ij = tl.max(s, axis=1)
        m_next = tl.maximum(m_i, m_ij)
        
        alpha = tl.math.exp(m_i - m_next)
        p = tl.math.exp(s - m_next[:, None])

        # Rescale running outputs
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Load Value Block (Shape: [BLOCK_N, HEAD_DIM])
        v_ptrs = V + (start_k + start_n + offs_n)[:, None] * stride_vt + kv_hd * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Accumulate Output: O = P @ V
        acc = tl.dot(p.to(q.dtype), v, acc=acc)
        m_i = m_next

    # Apply Optional Attention Sink Modification
    if HAS_SINK:
        s_aux_ptrs = s_aux + offs_h
        s_aux_val = tl.load(s_aux_ptrs, mask=offs_h < NUM_Q_HEADS, other=0.0)
        l_i += s_aux_val

    # Final Normalization Step
    acc = acc / l_i[:, None]

    # Write results to HBM
    out_ptrs = Out + bid * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=q_mask)


# Python binding wrapper
def flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, s_aux=None):
    batch_size, q_heads, head_size = q.shape
    kv_heads = k_varlen.shape[1]
    group_size = q_heads // kv_heads
    
    # Enforce power-of-two constraint for Triton's execution block
    block_h = max(16, triton.next_power_of_2(group_size))
    
    # Find max sequence length across variable batch lengths
    max_seqlen_kv = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())
    
    out = torch.empty_like(q)
    has_sink = s_aux is not None

    grid = lambda META: (batch_size, kv_heads)

    _flash_decode_gqa_varlen_kernel[grid](
        q, k_varlen, v_varlen, cu_seqlens_k, s_aux, out,
        q.stride(0), q.stride(1), q.stride(2),
        k_varlen.stride(0), k_varlen.stride(1), k_varlen.stride(2),
        v_varlen.stride(0), v_varlen.stride(1), v_varlen.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_Q_HEADS=q_heads, NUM_KV_HEADS=kv_heads, GROUP_SIZE=group_size, MAX_SEQLEN_KV=max_seqlen_kv,
        HEAD_DIM=head_size,
        BLOCK_H=block_h,
        HAS_SINK=has_sink,
        # Autotuned params
        BLOCK_N=64, num_warps=4, num_stages=2,
    )
    return out
