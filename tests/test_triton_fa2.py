import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from fskernels.triton.triton_fa2 import flash_attention_2_fwd

device = "cuda"
dtype = torch.bfloat16

B, H, S, D = 4, 16, 4096, 128

# Standard deterministic seeds for verification
torch.manual_seed(42)
q = torch.randn((B, H, S, D), dtype=dtype, device=device)
k = torch.randn((B, H, S, D), dtype=dtype, device=device)
v = torch.randn((B, H, S, D), dtype=dtype, device=device)


with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    
tri_out = flash_attention_2_fwd(q, k, v)

# Using atol=1e-3 due to bfloat16 accumulated rounding error variations vs native Flash
torch.testing.assert_close(tri_out, ref_out, atol=1e-3, rtol=1e-3)
print("✅ Triton FA2 Kernel Precision Match Confirmed!")