#include <torch/extension.h>
#include <cuda_fp16.h>

// Forward declaration of our external pure CUDA launcher function
extern "C" void launch_flash_attn_fwd(
    const half* q, const half* k, const half* v, float sm_scale, half* out,
    int b, int h, int n_q, int n_k, int d,
    int sq_b, int sq_h, int sq_m,
    int sk_b, int sk_h, int sk_n,
    int sv_b, int sv_h, int sv_n,
    int so_b, int so_h, int so_m
);

at::Tensor flash_attn_fwd(at::Tensor q, at::Tensor k, at::Tensor v, float sm_scale) {
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA device");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA device");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA device");
    
    const int b = q.size(0);
    const int h = q.size(1);
    const int n_q = q.size(2);
    const int d = q.size(3);
    const int n_k = k.size(2);

    // Corrected syntax: scope resolution operator (::)
    auto out = torch::empty_like(q);

    launch_flash_attn_fwd(
        (const half*)q.data_ptr<at::Half>(), 
        (const half*)k.data_ptr<at::Half>(), 
        (const half*)v.data_ptr<at::Half>(),
        sm_scale, 
        (half*)out.data_ptr<at::Half>(),
        b, h, n_q, n_k, d,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2)
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_fwd, "FlashAttention-2 Forward (CUDA)");
}