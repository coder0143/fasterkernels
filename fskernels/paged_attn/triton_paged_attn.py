import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Paged Flash Decode Kernel
# ---------------------------------------------------------------------------
# Memory layout
#   K_Pool / V_Pool : [num_pages, num_kv_heads, PAGE_SIZE, head_dim]
#   Block_Table     : [batch_size, max_blocks_per_seq]   (logical → physical)
#   Seq_Lens        : [batch_size]   (number of valid tokens per sequence)
#
# Each CTA processes one (sequence, kv_head) pair, iterating over logical
# pages via the block table.  No split-KV reduction — designed for
# continuous batching where individual sequences are long but page counts fit.
# ---------------------------------------------------------------------------

@triton.jit
def _paged_flash_decode_kernel(
    Q,             # [B, Q_H, D]          flattened query (decode step)
    K_Pool,        # [num_pages, KV_H, PAGE_SIZE, D]
    V_Pool,        # [num_pages, KV_H, PAGE_SIZE, D]
    Block_Table,   # [B, max_blocks]
    Seq_Lens,      # [B]
    Out,           # [B, Q_H, D]
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_ks, stride_kd,
    stride_vp, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_btb, stride_btbl,
    NUM_Q_HEADS, NUM_KV_HEADS, GROUP_SIZE,
    PAGE_SIZE:    tl.constexpr,
    BLOCK_H:      tl.constexpr,
    HEAD_DIM:     tl.constexpr,
    MAX_BLOCKS:   tl.constexpr,
):
    bid    = tl.program_id(0)
    kv_hid = tl.program_id(1)

    start_qh = kv_hid * GROUP_SIZE
    offs_h   = start_qh + tl.arange(0, BLOCK_H)
    offs_d   = tl.arange(0, HEAD_DIM)
    offs_s   = tl.arange(0, PAGE_SIZE)

    q_mask = (offs_h[:, None] < (start_qh + GROUP_SIZE)) & (offs_h[:, None] < NUM_Q_HEADS)
    q_ptrs = Q + bid * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q      = tl.load(q_ptrs, mask=q_mask, other=0.0)

    seq_len = tl.load(Seq_Lens + bid)
    sm_scale = 1.0 / tl.math.sqrt(HEAD_DIM * 1.0)

    m_i  = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    l_i  = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    # Walk logical pages
    for page_idx in range(0, MAX_BLOCKS):
        page_start = page_idx * PAGE_SIZE
        if page_start >= seq_len:
            break

        phys_page = tl.load(Block_Table + bid * stride_btb + page_idx * stride_btbl)

        k_ptrs = (K_Pool + phys_page * stride_kp + kv_hid * stride_kh +
                  offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd)
        v_ptrs = (V_Pool + phys_page * stride_vp + kv_hid * stride_vh +
                  offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd)

        # Mask tokens beyond seq_len for the last (partial) page
        tok_idx  = page_start + offs_s
        k_mask   = tok_idx[:, None] < seq_len
        k        = tl.load(k_ptrs, mask=k_mask, other=0.0)   # [PAGE_SIZE, D]
        v        = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # QK^T
        s     = tl.dot(q, tl.trans(k))   # [BLOCK_H, PAGE_SIZE]
        s    *= sm_scale
        s     = tl.where(tok_idx[None, :] < seq_len, s, float('-inf'))

        m_ij  = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp(m_i - m_new)
        p     = tl.math.exp(s - m_new[:, None])

        acc  = acc * alpha[:, None]
        l_i  = l_i * alpha + tl.sum(p, axis=1)
        acc  = tl.dot(p.to(q.dtype), v, acc=acc)
        m_i  = m_new

    acc = acc / l_i[:, None]
    out_ptrs = Out + bid * stride_ob + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=q_mask)


def paged_flash_decode(
    q:           torch.Tensor,    # [B, Q_H, D]
    k_pool:      torch.Tensor,    # [num_pages, KV_H, PAGE_SIZE, D]
    v_pool:      torch.Tensor,    # [num_pages, KV_H, PAGE_SIZE, D]
    block_table: torch.Tensor,    # [B, max_blocks] int32
    seq_lens:    torch.Tensor,    # [B] int32
    page_size:   int,
) -> torch.Tensor:
    B, Q_H, D   = q.shape
    _, KV_H, PS, _ = k_pool.shape
    group_size  = Q_H // KV_H
    block_h     = max(16, triton.next_power_of_2(group_size))
    max_blocks  = block_table.shape[1]

    out = torch.empty_like(q)

    grid = (B, KV_H)
    _paged_flash_decode_kernel[grid](
        q, k_pool, v_pool, block_table, seq_lens, out,
        q.stride(0),       q.stride(1),       q.stride(2),
        k_pool.stride(0),  k_pool.stride(1),  k_pool.stride(2),  k_pool.stride(3),
        v_pool.stride(0),  v_pool.stride(1),  v_pool.stride(2),  v_pool.stride(3),
        out.stride(0),     out.stride(1),     out.stride(2),
        block_table.stride(0), block_table.stride(1),
        NUM_Q_HEADS=Q_H, NUM_KV_HEADS=KV_H, GROUP_SIZE=group_size,
        PAGE_SIZE=page_size, BLOCK_H=block_h, HEAD_DIM=D,
        MAX_BLOCKS=max_blocks,
        num_warps=4, num_stages=2,
    )
    return out
