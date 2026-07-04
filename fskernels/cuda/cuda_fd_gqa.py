import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
cuda_decode_native = load(
    name="cuda_decode_native",
    sources=[
        os.path.join(current_dir, "flash_decode_api.cpp"),
        os.path.join(current_dir, "flash_decode_cuda.cu")
    ],
    verbose=True
)

def flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, sink):
    return cuda_decode_native.forward(q, k_varlen, v_varlen, cu_seqlens_k, sink)
