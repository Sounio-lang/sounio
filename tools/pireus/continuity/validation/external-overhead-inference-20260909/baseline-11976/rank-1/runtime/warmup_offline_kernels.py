#!/usr/bin/env python3
"""Compilation-only full-graph warmup with explicitly synthetic aliased weights.
No checkpoint tensors are read. Discard every output; never emit cycle receipts.
"""
from array import array
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist
import sglang.srt.model_loader.loader as loader
import offline_generate as offline
import offload_embedding

def synthetic_load(self, *, model_config, device_config):
    quant = loader._get_quantization_config(model_config, self.load_config)
    with loader.set_default_torch_dtype(model_config.dtype), torch.device("meta"):
        model = loader._initialize_model(model_config, self.load_config, quant)
    allocations = {}
    for module_name, module in model.named_modules():
        for name, old in list(module._parameters.items()):
            if old is None:
                continue
            full_name = module_name + "." + name
            assert old.device.type == "meta", full_name
            one = "scale" in name or ("norm" in module_name and name == "weight")
            # Input embedding must be unaliased so its GPU storage can be released.
            owner = full_name if full_name == "llm.embed_tokens.weight" else "shared"
            key = (tuple(old.shape), old.dtype, bool(one), owner)
            if key not in allocations:
                allocations[key] = torch.full(old.shape, int(one), dtype=old.dtype, device="cuda")
            param = torch.nn.Parameter(allocations[key], requires_grad=False)
            param.__dict__.update(old.__dict__)
            module._parameters[name] = param
        for name, old in list(module._buffers.items()):
            if old is not None and old.device.type == "meta":
                module._buffers[name] = torch.zeros(old.shape, dtype=old.dtype, device="cuda")
    offline.inference_memory("SYNTHETIC_KERNEL_WEIGHTS", distinct_allocations=len(allocations),
        synthetic_weights=True, checkpoint_tensors_loaded=False, inference_accepted=False)
    # Original post-load operations act on valid zero/one synthetic operands.
    loader.DefaultModelLoader.load_weights_and_postprocess(model, iter(()), torch.device("cuda"))
    self.counter_after_loading_weights = time.perf_counter()
    return model.eval()

def main():
    assert os.environ.get("PIREUS_KERNEL_WARMUP") == "1"
    rank = int(os.environ["PIREUS_RANK"])
    args = offline.prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    bench = offline.bench
    bench._set_envs_and_config(args)
    bench.initialize_moe_config(args)
    bench.initialize_fp8_gemm_config(args)
    bench.initialize_fp4_gemm_config(args)
    offline.initialize_bf16_gemm_config(args)
    offline.initialize_mamba_selective_state_update_backend(args)
    loader.ModelOptModelLoader.load_model = synthetic_load
    loader.DefaultModelLoader.load_model = synthetic_load
    runner, cache = offline.load_worker_and_cache(args, rank)
    model = runner.torch_runner
    placement = offload_embedding.offload(model.model.llm.embed_tokens,
        "/scratch/pireus/cache/synthetic-embedding-warmup")
    assert placement["source_gpu_sha256"] not in {
        "47851b3bdbe199db30fdfdb05818a84c59b5d5045cf0847e211e990859f7dc49",
        "f1b27d9350024731af4d8ffd05f732aefdcb070b980ef0cf5391842db1e8dc8c"}
    bench.TreeCacheNamespace = lambda **kwargs: cache
    for name, module in model.model.named_modules():
        if name.startswith("llm.layers.") and name.count(".") == 2:
            def before(module, args, name=name):
                offline.inference_memory("SYNTHETIC_LAYER_BEGIN", layer=name)
            def after(module, args, output, name=name):
                offline.inference_memory("SYNTHETIC_LAYER_END", layer=name)
            module.register_forward_pre_hook(before)
            module.register_forward_hook(after)
    item = json.loads(Path(os.environ["PIREUS_OFFLINE_INPUT"]).read_bytes())["items"][0]
    params = offline.SamplingParams(temperature=item["temperature"], max_new_tokens=2,
        sampling_seed=item["seed"], stop_token_ids=set(item["stop_token_ids"]))
    params.normalize(None)
    params.verify(model.model_config.vocab_size)
    cache.reset()
    runner.clear()
    req = offline.Req(rid="synthetic-warmup", origin_input_text="",
        origin_input_ids=array("q", item["input_ids"]), sampling_params=params)
    req.init_next_round_input(cache)
    req.logprob_start_len = -1
    req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
    comm = model.tp_group.pynccl_comm
    assert comm is not None and comm.available
    with comm.change_state(enable=True):
        next_ids, logits, batch = runner.extend([req])
        comm.broadcast(next_ids, src=0)
        req.output_ids.append(int(next_ids.item()))
        next_ids, logits = runner.decode(next_ids, batch)
        comm.broadcast(next_ids, src=0)
    torch.cuda.synchronize()
    offline.inference_memory("SYNTHETIC_KERNEL_WARMUP_PASS", input_tokens=len(item["input_ids"]),
        extend_passes=1, decode_passes=1, synthetic_weights=True, outputs_discarded=True,
        checkpoint_tensors_loaded=False, inference_accepted=False)
    dist.barrier(group=model.tp_group.cpu_group)

if __name__ == "__main__":
    main()
