# Base imports
import tilelang
import torch
import math

# Kernels
from fskernels.triton.triton_gqa_decode import flash_decode_gqa_varlen as triton_kernel
from fskernels.tilelang.tilelang_gqa_decode import flash_decode_gqa_varlen as tilelang_kernel
from fskernels.cuda.cuda_fd_gqa import flash_decode_gqa_varlen as cuda_kernel
from tests.test_triton_fd_gqa import ref_attention, ref_attention_varlen

# Benchmark
print(triton_kernel)
print(tilelang_kernel)
print(cuda_kernel)

