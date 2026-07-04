#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h> // Fixes: CUDART_INF_F undefined

#define BR 64
#define BC 64

template <int D>
__global__ void flash_attn_fwd_cuda_kernel(
    const half* __restrict__ Q,    
    const half* __restrict__ K,    
    const half* __restrict__ V,    
    const float sm_scale,
    half* __restrict__ Out,        
    const int B, const int H, const int N_Q, const int N_K,
    const int stride_qb, const int stride_qh, const int stride_qm,
    const int stride_kb, const int stride_kh, const int stride_kn,
    const int stride_vb, const int stride_vh, const int stride_vn,
    const int stride_ob, const int stride_oh, const int stride_om
) {
    const int block_m_idx = blockIdx.x; 
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / H;
    const int h_idx = batch_head_idx % H;
    const int tid = threadIdx.x; 

    __shared__ half sQ[BR][D];
    __shared__ half sK[BC][D];
    __shared__ half sV[BC][D];

    const half* Q_block_ptr = Q + b_idx * stride_qb + h_idx * stride_qh + block_m_idx * BR * stride_qm;
    half* O_block_ptr = Out + b_idx * stride_ob + h_idx * stride_oh + block_m_idx * BR * stride_om;

    for (int idx = tid; idx < BR * D; idx += blockDim.x) {
        int r = idx / D;
        int c = idx % D;
        if ((block_m_idx * BR + r) < N_Q) {
            sQ[r][c] = Q_block_ptr[r * stride_qm + c];
        } else {
            sQ[r][c] = __float2half(0.0f);
        }
    }
    __syncthreads();

    float m_i = -CUDART_INF_F;
    float l_i = 0.0f;
    float acc[D] = {0.0f};

    for (int block_n_idx = 0; block_n_idx < (N_K + BC - 1) / BC; ++block_n_idx) {
        const half* K_block_ptr = K + b_idx * stride_kb + h_idx * stride_kh + block_n_idx * BC * stride_kn;
        const half* V_block_ptr = V + b_idx * stride_vb + h_idx * stride_vh + block_n_idx * BC * stride_vn;

        for (int idx = tid; idx < BC * D; idx += blockDim.x) {
            int r = idx / D;
            int c = idx % D;
            if ((block_n_idx * BC + r) < N_K) {
                sK[r][c] = K_block_ptr[r * stride_kn + c];
                sV[r][c] = V_block_ptr[r * stride_vn + c];
            } else {
                sK[r][c] = __float2half(0.0f);
                sV[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if ((block_m_idx * BR + tid) < N_Q) {
            for (int j = 0; j < BC; ++j) {
                if ((block_n_idx * BC + j) >= N_K) break;

                float s_ij = 0.0f;
                for (int d = 0; d < D; ++d) {
                    s_ij += __half2float(sQ[tid][d]) * __half2float(sK[j][d]);
                }
                s_ij *= sm_scale;

                float m_next = fmaxf(m_i, s_ij);
                float alpha = expf(m_i - m_next);
                float p = expf(s_ij - m_next);

                l_i = l_i * alpha + p;

                for (int d = 0; d < D; ++d) {
                    acc[d] = acc[d] * alpha + p * __half2float(sV[j][d]);
                }
                m_i = m_next;
            }
        }
        __syncthreads();
    }

    if ((block_m_idx * BR + tid) < N_Q) {
        for (int d = 0; d < D; ++d) {
            O_block_ptr[tid * stride_om + d] = __float2half(acc[d] / l_i);
        }
    }
}

// Plain C host function declaration to bridge with the CPP file
extern "C" void launch_flash_attn_fwd(
    const half* q, const half* k, const half* v, float sm_scale, half* out,
    int b, int h, int n_q, int n_k, int d,
    int sq_b, int sq_h, int sq_m,
    int sk_b, int sk_h, int sk_n,
    int sv_b, int sv_h, int sv_n,
    int so_b, int so_h, int so_m
) {
    dim3 grid((n_q + BR - 1) / BR, b * h);
    dim3 block(BR); 

    if (d == 128) {
        flash_attn_fwd_cuda_kernel<128><<<grid, block>>>(
            q, k, v, sm_scale, out, b, h, n_q, n_k,
            sq_b, sq_h, sq_m, sk_b, sk_h, sk_n, sv_b, sv_h, sv_n, so_b, so_h, so_m
        );
    } else if (d == 64) {
        flash_attn_fwd_cuda_kernel<64><<<grid, block>>>(
            q, k, v, sm_scale, out, b, h, n_q, n_k,
            sq_b, sq_h, sq_m, sk_b, sk_h, sk_n, sv_b, sv_h, sv_n, so_b, so_h, so_m
        );
    }
}