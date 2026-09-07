#!/usr/bin/env python3
"""Diagnostic: real Inkling loader for one real checkpoint tensor; other params meta."""
import json
import os
from pathlib import Path
import sys
import threading
import time
import torch
import safetensors
import sglang.srt.model_loader.loader as loader
import sglang.srt.layers.moe.fused_moe_triton.layer as fused
original_initialize = loader._initialize_model
def available():
    return next(int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
def emit(stage, **kw):
    print(json.dumps(dict(stage=stage,job=os.environ["SLURM_JOB_ID"],rank=os.environ["PIREUS_RANK"],
        available_bytes=available(),cuda_allocated_bytes=torch.cuda.memory_allocated(),**kw)),flush=True)
def initialize(model_config, load_config, quant_config=None):
    with torch.device("meta"):
        model = original_initialize(model_config,load_config,quant_config)
    vocab_probe = os.environ.get("PIREUS_VOCAB_PATH_PROBE") == "1"
    bf16_probe = os.environ.get("PIREUS_BF16_PATH_PROBE") == "1"
    name = "llm.lm_head.weight" if vocab_probe else ("llm.layers.2.mlp.experts.w2_weight"
            if bf16_probe else "llm.layers.23.mlp.experts.w13_weight")
    checkpoint_name = "model.llm.unembed.weight" if vocab_probe else "model."+name
    module_path, parameter_name = name.rsplit(".",1)
    module = model.get_submodule(module_path)
    old = module.get_parameter(parameter_name)
    from sglang.srt.layers.parameter import BasevLLMParameter
    if isinstance(old,BasevLLMParameter):
        real = type(old)(data=torch.empty(old.shape,dtype=old.dtype,device="cuda"),
                         input_dim=old.input_dim,output_dim=old.output_dim,weight_loader=old.weight_loader)
    else:
        real = type(old)(data=torch.empty(old.shape,dtype=old.dtype,device="cuda"),requires_grad=False)
        real.__dict__.update(old.__dict__)
    module.register_parameter(parameter_name,real)
    import sglang.srt.layers.vocab_parallel_embedding as vocab
    copy_module = vocab if vocab_probe else fused
    copy_name = "_pireus_vocab_copy" if vocab_probe else "_pireus_checkpoint_copy"
    original_copy = getattr(copy_module,copy_name)
    def traced_copy(destination, source):
        emit("CHECKPOINT_COPY_PATH_ENTER",destination_device=str(destination.device),
             source_device=str(source.device),destination_shape=list(destination.shape),
             source_shape=list(source.shape),backend=str(fused.get_moe_runner_backend()))
        result=original_copy(destination,source)
        emit("CHECKPOINT_COPY_PATH_EXIT")
        return result
    setattr(copy_module,copy_name,traced_copy)
    index=json.loads((Path(model_config.model_path)/"model.safetensors.index.json").read_bytes())
    source_file=Path(model_config.model_path)/index["weight_map"][checkpoint_name]
    emit("CHECKPOINT_PATH_META_READY",parameter=name,shape=list(real.shape),
         weight_loader=real.weight_loader.__qualname__,
         config_interleaved=model.text_config.inference_moe_w13_interleaved,
         expert_interleaved=getattr(module,"inference_moe_w13_interleaved",None))
    with safetensors.safe_open(str(source_file),framework="pt",device="cpu") as f:
        weight=f.get_tensor(checkpoint_name)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        initial=torch.cuda.memory_allocated();minimum=[available()]
        stop=threading.Event()
        def watch():
            while not stop.wait(.01):minimum[0]=min(minimum[0],available())
        thread=threading.Thread(target=watch,daemon=True);thread.start()
        try:
            loaded=model.load_weights(iter([(checkpoint_name,weight)]))
            torch.cuda.synchronize()
        finally:
            stop.set();thread.join()
        emit("CHECKPOINT_PATH_LOADED",loaded=sorted(loaded),minimum_available_bytes=minimum[0],
             peak_extra_cuda_bytes=torch.cuda.max_memory_allocated()-initial)
        rank=int(os.environ["PIREUS_RANK"])
        dim = 0 if vocab_probe else (2 if bf16_probe else 1)
        expected=weight.narrow(dim,rank*real.shape[dim],real.shape[dim])
        block = 512 if vocab_probe else 1
        for i in range(0,real.shape[0],block):
            assert torch.equal(real[i:i+block].cpu(),expected[i:i+block])
    emit("CHECKPOINT_PATH_PASS",exact_tp_slice=True,full_model_loaded=False)
    from sglang.srt.distributed import get_world_group
    torch.distributed.barrier(group=get_world_group().cpu_group)
    raise SystemExit(74)
loader._initialize_model=initialize
if __name__=="__main__":
    assert os.environ.get("PIREUS_CHECKPOINT_PATH_PROBE")=="1"
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.plugins import load_plugins
    from sglang.srt.utils import kill_process_tree
    load_plugins()
    try:run_server(prepare_server_args(sys.argv[1:]))
    finally:kill_process_tree(os.getpid(),include_parent=False)
