#!/usr/bin/env python3
"""Diagnostic-only TP2 meta construction. It never loads checkpoint tensors or serves."""
import collections
import ctypes
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
import sglang.srt.model_loader.loader as loader

original_initialize = loader._initialize_model

def profile_initialize(model_config, load_config, quant_config=None):
    def host_available():
        return next(int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
    pre_trim=host_available()
    gc.collect()
    libc=ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes=[ctypes.c_size_t]
    libc.malloc_trim.restype=ctypes.c_int
    trim_result=libc.malloc_trim(0)
    post_trim=host_available()
    before = torch.cuda.memory_allocated()
    with torch.device("meta"):
        model = original_initialize(model_config, load_config, quant_config)
    tensors=[]
    groups=collections.Counter()
    dtypes=collections.Counter()
    for name,param in model.named_parameters():
        if param.device.type != "meta":
            raise RuntimeError("Meta diagnostic constructed a real model parameter: " + name)
        size=param.numel()*param.element_size()
        tensors.append(dict(name=name,shape=list(param.shape),dtype=str(param.dtype),bytes=size))
        parts=name.split(".")
        group=".".join(parts[:3]) if len(parts)>2 and parts[1]=="layers" else ".".join(parts[:2])
        groups[group]+=size
        dtypes[str(param.dtype)]+=size
    memory={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:","MemAvailable:")):
            memory[line.split(":")[0]]=int(line.split()[1])*1024
    report=dict(stage="MODEL_META_PROFILE",job=os.environ["SLURM_JOB_ID"],
        rank=os.environ["PIREUS_RANK"],parameter_count=len(tensors),
        pre_trim_available_bytes=pre_trim,post_trim_available_bytes=post_trim,malloc_trim_result=trim_result,
        tokenizer_skipped=os.environ.get("PIREUS_META_SKIP_TOKENIZER")=="1",
        parameter_storage_bytes=sum(t["bytes"] for t in tensors),
        groups_bytes=dict(groups),dtype_bytes=dict(dtypes),
        host_bytes=memory,cuda_allocated_delta_bytes=torch.cuda.memory_allocated()-before,
        checkpoint_tensors_loaded=False,serving_accepted=False,
        meta_parameter_devices_only=True,tensors=tensors)
    out=Path("/scratch/pireus/receipts")/("meta-model-"+report["job"]+"-"+report["rank"]+".json")
    with out.open("x") as stream:
        json.dump(report,stream,indent=2)
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps({k:v for k,v in report.items() if k!="tensors"}),flush=True)
    from sglang.srt.distributed import get_world_group
    dist.barrier(group=get_world_group().cpu_group)
    # Intentional diagnostic exit, not a serving success code.
    raise SystemExit(74)

loader._initialize_model = profile_initialize

if __name__ == "__main__":
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("PIREUS_META_PROBE")=="1"
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(),include_parent=False)
