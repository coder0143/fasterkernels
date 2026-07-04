# Triton profiler for lower representation code

import os
import torch
from fskernels.triton.triton_gqa_decode import flash_decode_gqa_varlen
import tritonparse.parse.utils
import tritonparse.structured_logging

device = "cuda"
dtype = torch.bfloat16

logs_path = "./profile/logs/triton_fd_gqa"
tritonparse.structured_logging.init(logs_path, enable_trace_launch=True)

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

# Setup configuration scenario mimicking TileLang
B, Q_H, KV_H, MAX_S, D = 4, 32, 8, 4096, 128

torch.manual_seed(0)
k_seqlens = torch.randint(MAX_S // 4, MAX_S + 1, size=(B,), device=device)
cu_seqlens_k = torch.zeros(B + 1, device=device, dtype=torch.int32)
cu_seqlens_k[1:] = torch.cumsum(k_seqlens, dim=0).to(torch.int32)
total_k_tokens = cu_seqlens_k[-1].item()

# Generate input tensors
q = torch.randn(B, Q_H, D, device=device, dtype=dtype)
k_varlen = torch.randn(total_k_tokens, KV_H, D, device=device, dtype=dtype)
v_varlen = torch.randn(total_k_tokens, KV_H, D, device=device, dtype=dtype)
sink = torch.randn(Q_H, device=device, dtype=torch.float32) * 0.1

def warmup():
    for _ in range(5):
        flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, s_aux=sink)

warmup()

tritonparse.parse.utils.unified_parse(
    source=logs_path, out="./profile/logs/triton_fd_gqa", overwrite=True,
)

# Pytorch profiler for execution table

def step():
    with torch.profiler.record_function("flash_decode_gqa_varlen"):
        return flash_decode_gqa_varlen(q, k_varlen, v_varlen, cu_seqlens_k, s_aux=sink)
    
    for _ in range(3):
        step()
    
    torch.cuda.synchronize()

trace_dir = "./profile/logs/triton_fd_gqa"
os.makedirs(trace_dir, exist_ok=True)

table_path = os.path.join(trace_dir, "table.txt")

schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=schedule,
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
) as prof:
    for _ in range(5):
        step()
        prof.step()
    
    torch.cuda.synchronize()

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

