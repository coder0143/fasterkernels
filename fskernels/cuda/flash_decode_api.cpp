#include <torch/extension.h>
#include <cuda_fp16.h>

extern "C" void launch_flash_decode_gqa(
    const half* q, const half* k, const half* v, const int* cu_seqlens, const float* s_aux, half* out,
    int b, int q_heads, int kv_heads, int group_size, int max_seqlen, int d, bool has_sink,
    int sq_b, int sq_h, int sk_t, int sk_h, int sv_t, int sv_h, int so_b, int so_h
);

at::Tensor flash_decode_gqa(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor cu_seqlens, torch::optional<at::Tensor> s_aux) {
    TORCH_CHECK(q.is_cuda(), "Q tensor must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K tensor must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V tensor must be on CUDA");

    const int b = q.size(0);
    const int q_heads = q.size(1);
    const int d = q.size(2);
    const int kv_heads = k.size(1);
    const int group_size = q_heads / kv_heads;
    
    // Find maximum length of variable blocks
    auto seqlens_cpu = cu_seqlens.to(torch::kCPU);
    const int* ptr = seqlens_cpu.data_ptr<int>();
    int max_seqlen = 0;
    for (int i = 0; i < b; ++i) {
        max_seqlen = std::max(max_seqlen, ptr[i+1] - ptr[i]);
    }

    auto out = torch::empty_like(q);
    bool has_sink = s_aux.has_value();
    const float* s_aux_ptr = has_sink ? s_aux.value().data_ptr<float>() : nullptr;

    launch_flash_decode_gqa(
        (const half*)q.data_ptr<at::Half>(),
        (const half*)k.data_ptr<at::Half>(),
        (const half*)v.data_ptr<at::Half>(),
        cu_seqlens.data_ptr<int>(),
        s_aux_ptr,
        (half*)out.data_ptr<at::Half>(),
        b, q_heads, kv_heads, group_size, max_seqlen, d, has_sink,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1)
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_decode_gqa, "GQA Varlen Flash Decoding (CUDA)");
}