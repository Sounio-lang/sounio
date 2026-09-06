#!/usr/bin/env python3
"""Job-bound two-rank GPU/NCCL smoke. NCCL_NET=IB must be set by launcher."""
import datetime
import json
import os
import socket
import torch
import torch.distributed as dist

assert os.environ.get("SLURM_JOB_ID"), "Slurm allocation required"
assert os.environ.get("NCCL_NET") == "IB", "IB qualification cannot silently fall back"
rank = int(os.environ["PIREUS_RANK"])
assert int(os.environ["SLURM_NTASKS"]) == 2
assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
print(json.dumps({"stage": "inventory", "rank": rank, "host": socket.gethostname(),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0),
                  "capability": torch.cuda.get_device_capability(0),
                  "job": os.environ["SLURM_JOB_ID"]}), flush=True)
dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=2,
                        timeout=datetime.timedelta(seconds=90))
try:
    for size in (1, 1024, 1024 * 1024):
        tensor = torch.full((size,), rank + 1.0, device="cuda", dtype=torch.float32)
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        assert bool(torch.all(tensor == 3.0)), "collective data mismatch"
    dist.barrier()
    print(json.dumps({"stage": "TP2_IB_PASS", "rank": rank,
                      "job": os.environ["SLURM_JOB_ID"]}), flush=True)
finally:
    dist.destroy_process_group()
