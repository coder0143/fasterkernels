#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#define BC 64 // Reduced from 128 to fix the 48KB ptxas shared memory overflow

// Shared memory helper utilities for cooperative parallel reductions
__device__ inline float blockReduceMax(float val, float* s_mem) {
    int tid = threadIdx.x;
    s_mem[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] = fmaxf(s_mem[tid], s_mem[tid + stride]);
        }
        __syncthreads();
    }
    return s_mem[0];
}

__device__ inline float blockReduceSum(float val, float* s_mem) {
    int tid = threadIdx.x;
    s_mem[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    return s_mem[0];
}

template <int D, int G>
__global__ void flash_decode_gqa_varlen_kernel(
    const half* __restrict__ Q,           // [B, num_q_heads, D]
    const half* __restrict__ K,           // [total_tokens, num_kv_heads, D]
    const half* __restrict__ V,           // [total_tokens, num_kv_heads, D]
    const int* __restrict__ cu_seqlens_k, // [B + 1]
    const float* __restrict__ s_aux,      // [num_q_heads]
    half* __restrict__ Out,               // [B, num_q_heads, D]
    const int num_q_heads, const int num_kv_heads, const int max_seqlen,
    const int stride_qb, const int stride_qh,
    const int stride_kt, const int stride_kh,
    const int stride_vt, const int stride_vh,
    const int stride_ob, const int stride_oh,
    const bool has_sink
) {
    const int bid = blockIdx.x;     // Batch Index
    const int kv_hd = blockIdx.y;   // KV Head Index
    const int tid = threadIdx.x;    // Thread matching token position index

    const int start_q_head = kv_hd * G;
    const float sm_scale = 1.0f / sqrtf(D * 1.0f);

    // Shared Memory allocations (Now fits safely within ~34.2 KB)
    __shared__ half sQ[G][D];
    __shared__ half sK[BC][D];
    __shared__ half sV[BC][D];
    __shared__ float s_reduce[BC];

    // Load Group Queries into shared space cooperatively
    for (int idx = tid; idx < G * D; idx += blockDim.x) {
        int g = idx / D;
        int d = idx % D;
        int q_head = start_q_head + g;
        if (q_head < num_q_heads) {
            sQ[g][d] = Q[bid * stride_qb + q_head * stride_qh + d];
        } else {
            sQ[g][d] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Bound tracking metrics for variable lengths
    const int start_k = cu_seqlens_k[bid];
    const int end_k = cu_seqlens_k[bid + 1];
    const int cur_seqlen_k = end_k - start_k;

    // Allocate thread local registers for online softmax
    float m_i[G];
    float l_i_partial[G];
    float acc_partial[G][D];

    #pragma unroll
    for (int g = 0; g < G; ++g) {
        m_i[g] = -CUDART_INF_F;
        l_i_partial[g] = 0.0f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            acc_partial[g][d] = 0.0f;
        }
    }

    // Streaming loop over Key-Value token sequence slices
    for (int block_n_idx = 0; block_n_idx < (cur_seqlen_k + BC - 1) / BC; ++block_n_idx) {
        const int token_offset = start_k + block_n_idx * BC;

        // Load K & V slices into Shared Memory
        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int t = idx / D;
            int d = idx % D;
            if ((block_n_idx * BC + t) < cur_seqlen_k) {
                sK[t][d] = K[(token_offset + t) * stride_kt + kv_hd * stride_kh + d];
                sV[t][d] = V[(token_offset + t) * stride_vt + kv_hd * stride_vh + d];
            } else {
                sK[t][d] = __float2half(0.0f);
                sV[t][d] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Thread local dot products and online update tracks
        #pragma unroll
        for (int g = 0; g < G; ++g) {
            float s_g_tid = -CUDART_INF_F;
            if ((block_n_idx * BC + tid) < cur_seqlen_k && (start_q_head + g) < num_q_heads) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < D; ++d) {
                    dot += __half2float(sQ[g][d]) * __half2float(sK[tid][d]);
                }
                s_g_tid = dot * sm_scale;
            }

            // Execute synchronous block-wide max selection
            float m_tile = blockReduceMax(s_g_tid, s_reduce);
            float m_next = fmaxf(m_i[g], m_tile);
            float alpha = expf(m_i[g] - m_next);
            
            float p_g_tid = ((block_n_idx * BC + tid) < cur_seqlen_k) ? expf(s_g_tid - m_next) : 0.0f;

            // Continuous register rescale
            l_i_partial[g] = l_i_partial[g] * alpha + p_g_tid;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                acc_partial[g][d] = acc_partial[g][d] * alpha + p_g_tid * __half2float(sV[tid][d]);
            }
            m_i[g] = m_next;
        }
    }

    // Final normalization reduction phase
    #pragma unroll
    for (int g = 0; g < G; ++g) {
        int q_head = start_q_head + g;
        if (q_head >= num_q_heads) break;

        // Accumulate denominators and numerators from local registers
        float l_i_total = blockReduceSum(l_i_partial[g], s_reduce);
        if (has_sink && tid == 0) {
            l_i_total += s_aux[q_head];
        }
        __syncthreads();

        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float res_d = blockReduceSum(acc_partial[g][d], s_reduce);
            if (tid == 0) {
                Out[bid * stride_ob + q_head * stride_oh + d] = __float2half(res_d / l_i_total);
            }
            __syncthreads();
        }
    }
}

// Host dispatcher linking mechanism
extern "C" void launch_flash_decode_gqa(
    const half* q, const half* k, const half* v, const int* cu_seqlens, const float* s_aux, half* out,
    int b, int q_heads, int kv_heads, int group_size, int max_seqlen, int d, bool has_sink,
    int sq_b, int sq_h, int sk_t, int sk_h, int sv_t, int sv_h, int so_b, int so_h
) {
    dim3 grid(b, kv_heads);
    dim3 block(BC); // Dynamically launches with 64 threads per block

    if (d == 128) {
        if (group_size == 4) {
            flash_decode_gqa_varlen_kernel<128, 4><<<grid, block>>>(
                q, k, v, cu_seqlens, s_aux, out, q_heads, kv_heads, max_seqlen,
                sq_b, sq_h, sk_t, sk_h, sv_t, sv_h, so_b, so_h, has_sink
            );
        } else if (group_size == 8) {
            flash_decode_gqa_varlen_kernel<128, 8><<<grid, block>>>(
                q, k, v, cu_seqlens, s_aux, out, q_heads, kv_heads, max_seqlen,
                sq_b, sq_h, sk_t, sk_h, sv_t, sv_h, so_b, so_h, has_sink
            );
        }
    }
}