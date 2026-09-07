#!/usr/bin/env python3
"""Actual GB10 baseline/challenger allocation, repack and MoE output controls."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import sglang.srt.layers.quantization.modelopt_quant as quant
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from patch_marlin_placeholders import patched_source

assert os.environ.get("SLURM_JOB_ID")
assert torch.cuda.get_device_capability() == (12, 1)
source = Path(quant.__file__).read_bytes()
changed = patched_source(source)
tree = ast.parse(changed)
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ModelOptNvFp4FusedMoEMethod")
method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="create_weights")
module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
namespace = {}
exec(compile(module,"<custody-bound-marlin-create-weights>","exec"),vars(quant),namespace)
challenger = namespace["create_weights"]
original = quant.ModelOptNvFp4FusedMoEMethod.create_weights
config = SimpleNamespace(is_checkpoint_nvfp4_serialized=True, group_size=16)
runner = SimpleNamespace(is_gated=True, activation="silu")
q = object.__new__(quant.ModelOptNvFp4FusedMoEMethod)
q.quant_config = config
q.enable_flashinfer_trtllm_moe = False
q._moe_runner_backend = MoeRunnerBackend.MARLIN
q.moe_runner_config = runner

def make(create, experts, hidden, intermediate):
    layer = torch.nn.Module()
    layer.num_experts = layer.num_local_experts = experts
    layer.moe_runner_config = runner
    with torch.device("cuda"):
        create(q,layer,experts,hidden,intermediate,torch.bfloat16,weight_loader=lambda *a,**k:None)
    return layer

def fill(layer):
    for name,param in layer.named_parameters():
        if "blockscale_swizzled" in name:
            continue
        if param.dtype == torch.uint8:
            param.data.copy_(torch.randint(0,256,param.shape,device="cuda",dtype=torch.uint8))
        elif param.dtype == torch.float8_e4m3fn:
            param.data.copy_((torch.rand(param.shape,device="cuda")*.125+.03125).to(param.dtype))
        else:
            param.data.fill_(0.03125)

def bits(t):
    return t.detach().contiguous().view(torch.uint8)

def evaluate(layer,x,weights,ids,logits):
    return fused_marlin_moe(hidden_states=x,w1=layer.w13_weight,w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,w2_scale=layer.w2_weight_scale,
        gating_output=logits,topk_weights=weights,topk_ids=ids,
        w1_global_scale=layer.w13_weight_scale_2,w2_global_scale=layer.w2_weight_scale_2,
        num_bits=4,is_k_full=True,inplace=False,activation="silu",is_gated=True)

kernel_module=importlib.import_module("sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe")
stock_gemm=kernel_module.moe_wna16_marlin_gemm
def deterministic_gemm(*args,**kwargs):
    kwargs["use_atomic_add"]=False
    return stock_gemm(*args,**kwargs)

reports=[]
with patch.object(quant,"get_moe_runner_backend",return_value=MoeRunnerBackend.MARLIN):
    for experts,hidden,intermediate in [(8,256,128),(4,4096,1024)]:
        torch.manual_seed(20260907)
        baseline=make(original,experts,hidden,intermediate)
        fill(baseline)
        candidate=make(challenger,experts,hidden,intermediate)
        assert candidate.w13_blockscale_swizzled is None and candidate.w2_blockscale_swizzled is None
        omitted=sum(p.numel()*p.element_size() for n,p in baseline.named_parameters()
                    if "blockscale_swizzled" in n)
        assert omitted > 0
        for name,param in candidate.named_parameters():
            param.data.copy_(baseline.get_parameter(name).data)
        quant.ModelOptNvFp4FusedMoEMethod.process_weights_after_loading(q,baseline)
        quant.ModelOptNvFp4FusedMoEMethod.process_weights_after_loading(q,candidate)
        for name,param in candidate.named_parameters():
            assert torch.equal(bits(param),bits(baseline.get_parameter(name))),name
        compared=0
        stock_repeat_differences=[]
        for tokens in [1,17,64]:
            x=torch.randn(tokens,hidden,device="cuda",dtype=torch.bfloat16)*.1
            logits=torch.randn(tokens,experts,device="cuda",dtype=torch.float32)
            weights,ids=torch.topk(torch.softmax(logits,dim=-1),2,dim=-1)
            ids=ids.to(torch.int32)
            stock_a=evaluate(baseline,x,weights,ids,logits)
            stock_repeat=evaluate(baseline,x,weights,ids,logits)
            stock_b=evaluate(candidate,x,weights,ids,logits)
            torch.cuda.synchronize()
            stock_repeat_differences.append(dict(tokens=tokens,
                baseline_repeat_different_components=int(torch.count_nonzero(stock_a!=stock_repeat)),
                candidate_different_components=int(torch.count_nonzero(stock_a!=stock_b)),
                baseline_repeat_max_abs=float((stock_a.float()-stock_repeat.float()).abs().max()),
                candidate_max_abs=float((stock_a.float()-stock_b.float()).abs().max())))
            # The stock SM121 path uses atomic reduction. Pin only the diagnostic
            # reduction to non-atomic for an exact baseline/challenger comparison.
            # This override is not installed by the production source overlay.
            with patch.object(kernel_module,"moe_wna16_marlin_gemm",deterministic_gemm):
                a=evaluate(baseline,x,weights,ids,logits)
                b=evaluate(candidate,x,weights,ids,logits)
            torch.cuda.synchronize()
            assert torch.isfinite(a).all() and torch.count_nonzero(a)>0
            assert torch.equal(bits(a),bits(b)),"Non-atomic MoE output bits changed"
            compared+=a.numel()
        # A real consumed-scale perturbation must be detected by the same comparison.
        candidate.w2_weight_scale_2.data.mul_(2)
        with patch.object(kernel_module,"moe_wna16_marlin_gemm",deterministic_gemm):
            poisoned=evaluate(candidate,x,weights,ids,logits)
        torch.cuda.synchronize()
        assert not torch.equal(bits(a),bits(poisoned)),"Negative control missed consumed-scale corruption"
        reports.append(dict(experts=experts,hidden=hidden,intermediate=intermediate,
                            omitted_bytes=omitted,output_components_compared=compared,
                            exact_non_atomic_output_pass=True,consumed_scale_negative_pass=True,
                            stock_atomic_repeat_observations=stock_repeat_differences,
                            production_atomic_bitwise_determinism_claimed=False))
        del baseline,candidate
        torch.cuda.empty_cache()
print(json.dumps(dict(stage="MARLIN_PLACEHOLDER_GPU_PASS",job=os.environ["SLURM_JOB_ID"],
                      rank=os.environ["PIREUS_RANK"],source_sha256=hashlib.sha256(source).hexdigest(),
                      patch_sha256=hashlib.sha256(changed).hexdigest(),reports=reports,
                      full_model_serving_accepted=False)),flush=True)
