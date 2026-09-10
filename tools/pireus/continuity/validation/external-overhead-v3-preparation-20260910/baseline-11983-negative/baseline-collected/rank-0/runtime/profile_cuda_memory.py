#!/usr/bin/env python3
"""Diagnostic-only actual TP2 constructor accounting; no checkpoint tensors or serving."""
import json
import os
import sys
from pathlib import Path
import torch
import sglang.srt.model_loader.loader as loader

original_initialize = loader._initialize_model

def observation(stage, **fields):
    memory = {l.split(":")[0]: int(l.split()[1])*1024
              for l in Path("/proc/meminfo").read_text().splitlines()
              if l.startswith(("MemAvailable:", "Cached:", "SReclaimable:", "Shmem:"))}
    print(json.dumps(dict(stage=stage, job=os.environ["SLURM_JOB_ID"],
         rank=os.environ["PIREUS_RANK"], host_bytes=memory,
         cuda_allocated_bytes=torch.cuda.memory_allocated(),
         cuda_reserved_bytes=torch.cuda.memory_reserved(), **fields)), flush=True)

def profile_initialize(model_config, load_config, quant_config=None):
    register = torch.nn.Module.register_parameter
    count = 0
    parameter_bytes = 0
    def tracked(module, name, param):
        nonlocal count, parameter_bytes
        result = register(module, name, param)
        if param is not None:
            size = param.numel()*param.element_size()
            count += 1
            parameter_bytes += size
            if size >= 8*1024**2 or count % 50 == 0:
                observation("MODEL_CUDA_PARAMETER", index=count, module=type(module).__name__,
                            parameter=name, shape=list(param.shape), dtype=str(param.dtype),
                            parameter_bytes=size, registered_total_bytes=parameter_bytes)
        return result
    observation("MODEL_CUDA_CONSTRUCTOR_BEGIN")
    torch.nn.Module.register_parameter = tracked
    try:
        model = original_initialize(model_config, load_config, quant_config)
    finally:
        torch.nn.Module.register_parameter = register
    observation("MODEL_CUDA_CONSTRUCTOR_COMPLETE", parameter_count=count,
                actual_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                checkpoint_tensors_loaded=False, serving_accepted=False)
    raise SystemExit(74)

loader._initialize_model = profile_initialize

if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("PIREUS_CUDA_PROBE") == "1"
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
