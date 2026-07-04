import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
cuda_flash_native = load(
    name="cuda_flash_native",
    sources=[
        os.path.join(current_dir, "flash_attn_api.cpp"),
        os.path.join(current_dir, "flash_attn_cuda.cu")
    ],
    verbose=True
)

def flash_attention_2_fwd(q, k, v, sm_scale):
    return cuda_flash_native.forward(q, k, v, sm_scale)

