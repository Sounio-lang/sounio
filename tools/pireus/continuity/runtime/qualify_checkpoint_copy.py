#!/usr/bin/env python3
"""Actual GB10 checkpoint-copy controls against stock TP loading."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import sglang.srt.layers.moe.fused_moe_triton.layer as fused
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from patch_checkpoint_copy import patched_source

assert os.environ.get("SLURM_JOB_ID")
assert torch.cuda.get_device_capability() == (12, 1)
raw = Path(fused.__file__).read_bytes()
changed = patched_source(raw)
tree = ast.parse(changed)
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_load_w13", "_load_w2")]
nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pireus_checkpoint_copy"]
namespace = dict(vars(fused))
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])), "<bounded-checkpoint-copy>", "exec"), namespace)
layer = SimpleNamespace(
    quant_config=SimpleNamespace(get_name=lambda:"modelopt_fp4"),
    quant_method=SimpleNamespace(load_up_proj_weight_first=False),
    moe_runner_config=SimpleNamespace(is_gated=True), use_padded_loading=False,
    use_presharded_weights=False, use_triton_kernels=False)
reports = []
with patch.object(fused, "get_moe_runner_backend", return_value=MoeRunnerBackend.MARLIN):
    namespace["get_moe_runner_backend"] = fused.get_moe_runner_backend
    for experts, rows, cols, dtype in [(8,64,32,torch.uint8), (256,4096,1024,torch.uint8),
                                        (8,64,32,torch.bfloat16), (256,2048,4096,torch.bfloat16)]:
        for shard_id, dim in [("w13",1),("w2",2)]:
            shape = [experts,rows,cols]
            source_shape = list(shape);source_shape[dim] *= 2
            torch.manual_seed(20260907)
            source = torch.randint(0,127,source_shape,dtype=torch.uint8,device="cpu").to(dtype)
            for rank in (0,1):
                expected = torch.empty(shape,dtype=dtype,device="cuda")
                stock = getattr(fused.FusedMoE, "_load_"+shard_id)
                if shard_id == "w13":
                    stock(layer, expert_data=expected, shard_dim=dim, shard_id=shard_id,
                          loaded_weight=source,tp_rank=rank)
                else:
                    # Stock fused 3D ModelOpt w2 is rejected before copying.
                    # Its existing per-expert 2D route is the reference.
                    try:
                        stock(layer, expert_data=expected, shard_dim=dim, shard_id=shard_id,
                              loaded_weight=source,tp_rank=rank)
                    except ValueError as e:
                        assert "Expected 2D tensors" in str(e)
                    else:
                        raise AssertionError("Expected pinned fused-w2 dimensionality defect")
                    for index in range(experts):
                        stock(layer,expert_data=expected[index],shard_dim=dim-1,
                              shard_id=shard_id,loaded_weight=source[index],tp_rank=rank)
                candidate = torch.empty_like(expected)
                pointer = candidate.data_ptr()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                namespace["_load_"+shard_id](layer,expert_data=candidate,shard_dim=dim,
                        shard_id=shard_id,loaded_weight=source,tp_rank=rank)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()-allocated
                assert candidate.data_ptr() == pointer
                assert torch.equal(candidate,expected)
                # Also compare independently with the literal TP source slice.
                narrow = source.narrow(dim,rank*shape[dim],shape[dim])
                for index in range(experts):
                    assert torch.equal(candidate[index].cpu(),narrow[index])
                candidate.view(torch.uint8)[0,0,0] ^= 1
                assert not torch.equal(candidate,expected)
                reports.append(dict(experts=experts,shape=shape,shard=shard_id,
                    tp_rank=rank,dtype=str(dtype),exact=True,negative_control=True,
                    original_storage_retained=True,peak_extra_cuda_bytes=peak))
                del expected,candidate
                torch.cuda.empty_cache()
            del source
    # Wrong packed shape must refuse before an expert copy.
    a=torch.empty((2,4,8),dtype=torch.uint8,device="cuda")
    b=torch.empty((2,4,7),dtype=torch.uint8)
    try:namespace["_pireus_checkpoint_copy"](a,b)
    except AssertionError:pass
    else:raise AssertionError("Mismatched shape accepted")
report=dict(stage="CHECKPOINT_COPY_GPU_PASS",job=os.environ["SLURM_JOB_ID"],
            rank=os.environ["PIREUS_RANK"],source_sha256=hashlib.sha256(raw).hexdigest(),
            patched_sha256=hashlib.sha256(changed).hexdigest(),controls=reports,
            full_model_loaded=False,production_kernels_changed=False)
print(json.dumps(report),flush=True)
