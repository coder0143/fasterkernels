# Triton profiler for lower representation code

import os
import torch
from fskernels.triton.triton_fa2 import flash_attention_2_fwd
import tritonparse.parse.utils
import tritonparse.structured_logging

device = "cuda"
dtype = torch.bfloat16

logs_path = "./profile/logs/triton"
tritonparse.structured_logging.init(logs_path, enable_trace_launch=True)

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

B, H, S, D = 4, 16, 4096, 128

# Standard deterministic seeds for verification
torch.manual_seed(42)
q = torch.randn((B, H, S, D), dtype=dtype, device=device)
k = torch.randn((B, H, S, D), dtype=dtype, device=device)
v = torch.randn((B, H, S, D), dtype=dtype, device=device)

def warmup():
    for _ in range(3):
        flash_attention_2_fwd(q, k, v)

warmup()

tritonparse.parse.utils.unified_parse(
    source=logs_path, out="./profile/logs/triton_fa2", overwrite=True,
)

# Pytorch profiler for execution table

def step():
    with torch.profiler.record_function("flash_attention_2_fwd"):
        return flash_attention_2_fwd(q, k, v)
    
    for _ in range(3):
        step()
    
    torch.cuda.synchronize()

trace_dir = "./profile/logs/triton_fa2"
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
    for _ in range(3):
        step()
        prof.step()
    
    torch.cuda.synchronize()

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

