#!/usr/bin/env python3
"""Diagnostic only: actual BF16 LM-head stock projection versus GPU row tiles."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from safetensors import safe_open
import offline_generate as offline
import qualify_offline_preparation as preparation
import offload_embedding
import offload_lm_head

def main():
    assert os.environ.get("PIREUS_LM_HEAD_TILING_PROBE") == "1"
    rank = int(os.environ["PIREUS_RANK"])
    import torch._inductor.config as inductor_config
    if inductor_config.compile_threads != 1:
        raise ValueError("bounded offline compiler worker profile")
    print(json.dumps(dict(stage="INDUCTOR_COMPILE_PROFILE", job=os.environ["SLURM_JOB_ID"],
                         rank=str(rank), compile_threads=inductor_config.compile_threads)), flush=True)
    args = offline.prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    bench = offline.bench
    bench._set_envs_and_config(args)
    bench.initialize_moe_config(args)
    bench.initialize_fp8_gemm_config(args)
    bench.initialize_fp4_gemm_config(args)
    offline.initialize_bf16_gemm_config(args)
    offline.initialize_mamba_selective_state_update_backend(args)
    preparation.loader.ModelOptModelLoader.load_model = preparation.meta_load
    preparation.loader.DefaultModelLoader.load_model = preparation.meta_load
    runner, cache = offline.load_worker_and_cache(args, rank)
    model = runner.torch_runner
    embed = model.model.llm.lm_head
    old = embed.weight
    real = torch.nn.Parameter(torch.empty(old.shape, dtype=old.dtype, device="cuda"), requires_grad=False)
    real.__dict__.update(old.__dict__)
    embed.weight = real
    del old, real
    root = Path(args.model_path)
    key = "model.llm.unembed.weight"
    index = json.loads((root/"model.safetensors.index.json").read_bytes())["weight_map"]
    with safe_open(root/index[key], framework="pt", device="cpu") as f:
        source = f.get_tensor(key)
        embed.weight_loader(embed.weight, source)
        del source

    processor = model.model.llm.logits_processor
    assert not processor.use_fp32_lm_head and processor.rl_on_policy_target is None
    assert type(embed).__name__ == "ParallelLMHead"
    assert type(embed.quant_method).__name__ == "UnquantizedEmbeddingMethod"
    shape = tuple(embed.weight.shape)
    assert shape == (100512, 4096) and embed.weight.dtype == torch.bfloat16
    # Stock processor, actual checkpoint shard; synthetic hidden states only.
    generator = torch.Generator().manual_seed(20260907)
    probes = [torch.zeros((1,4096)), torch.ones((1,4096)),
              (torch.arange(4096).remainder(2)*2-1).reshape(1,-1).float()]
    for scale in [0.001, 0.1, 1.0, 10.0, 100.0]:
        probes.extend([torch.randn((1,4096), generator=generator)*scale for _ in range(3)])
    for column in [0, 2048, 4095]:
        value = torch.zeros((1,4096))
        value[0,column] = 1
        probes.append(value)
    inputs = [value.to(device="cuda", dtype=torch.bfloat16) for value in probes]
    with torch.no_grad():
        expected = [processor._compute_lm_head(value, embed).cpu() for value in inputs]
    source_sha = hashlib.sha256()
    for start in range(0,shape[0],512):
        cpu = embed.weight[start:start+512].cpu()
        source_sha.update(memoryview(cpu.view(torch.uint8).numpy()))
        del cpu
    results = []
    for tile_rows in [4096, 8192]:
        projected = [torch.empty_like(value) for value in expected]
        with torch.no_grad():
            for start in range(0,shape[0],tile_rows):
                # Owned CPU staging, identical BF16 bytes, no source-mmap CUDA copy.
                cpu = embed.weight[start:start+tile_rows].cpu()
                tile = cpu.to("cuda", non_blocking=False)
                for index, value in enumerate(inputs):
                    projected[index][:,start:start+len(cpu)] = torch.matmul(value, tile.T).cpu()
                del tile, cpu
        mismatches = [int((actual.view(torch.uint8) != ref.view(torch.uint8)).sum())
                      for actual,ref in zip(projected,expected)]
        errors = [float((actual.float()-ref.float()).abs().max()) for actual,ref in zip(projected,expected)]
        results.append(dict(tile_rows=tile_rows, byte_mismatches=mismatches,
                            max_abs_errors=errors, bitwise_equal=not any(mismatches)))
    # Negative control targets a selected basis coordinate and must alter logits.
    with torch.no_grad():
        tile = embed.weight[:4096].clone()
        tile[0,0] += 1
        changed = torch.matmul(inputs[-3], tile.T).cpu()
    corruption_detected = not torch.equal(changed.view(torch.uint8), expected[-3][:,:4096].contiguous().view(torch.uint8))
    if not corruption_detected:
        raise ValueError("LM-head corruption control insensitive")

    storage = offload_lm_head.offload(embed, processor, "/scratch/pireus/cache/lm-head-offload")
    with torch.no_grad():
        for value, reference in zip(inputs, expected):
            actual = processor._compute_lm_head(value, embed).cpu()
            if not torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)):
                raise ValueError("file-backed LM-head disagrees with stock GPU reference")
        embed.weight[0,0] += 1
        changed = processor._compute_lm_head(inputs[-3], embed).cpu()
        if torch.equal(changed.view(torch.uint8), expected[-3].view(torch.uint8)):
            raise ValueError("file-backed LM-head corruption insensitive")
        recovered = processor._compute_lm_head(inputs[-3], embed).cpu()
        if not torch.equal(recovered.view(torch.uint8), expected[-3].view(torch.uint8)):
            raise ValueError("LM-head private corruption persisted")
        try:
            processor._compute_lm_head(inputs[0].expand(2,-1), embed)
        except ValueError:
            pass
        else:
            raise ValueError("LM-head accepted unsupported multirow profile")
    storage.update(helper_sha256=hashlib.sha256(Path(offload_lm_head.__file__).read_bytes()).hexdigest(),
                   file_path_byte_exact_on_controls=True, corruption_detected=True,
                   private_corruption_reverted=True, unsupported_multirow_refused=True)
    receipt = dict(schema=1, stage="LM_HEAD_TILING_CONTROL_COMPLETE",
        job=os.environ["SLURM_JOB_ID"], rank=str(rank), checkpoint_tensor=key,
        checkpoint_revision="b6a99534467840620d411e4cd4ad5819b2610d9c",
        shape=list(shape), dtype=str(embed.weight.dtype), source_gpu_sha256=source_sha.hexdigest(),
        probe_count=len(inputs), hidden_rows=1, results=results, storage=storage,
        corruption_detected=corruption_detected, transformer_layers_executed=False,
        checkpoint_precision_changed=False, inference_accepted=False,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    offline.write(Path("/scratch/pireus/receipts")/("lm-head-tiling-control-"+receipt["job"]+"-"+str(rank)+".json"), receipt)
    print(json.dumps(receipt), flush=True)
    dist.barrier(group=model.tp_group.cpu_group)

if __name__ == "__main__":
    main()
