#!/usr/bin/env python3
"""Diagnostic: real hybrid request preparation with meta-only model parameters.
Never reads checkpoint tensors, runs a model layer, samples, or emits proposals.
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

class StopBeforeModel(Exception):
    pass

def meta_load(self, *, model_config, device_config):
    quant = loader._get_quantization_config(model_config, self.load_config)
    with loader.set_default_torch_dtype(model_config.dtype), torch.device("meta"):
        model = loader._initialize_model(model_config, self.load_config, quant)
    if any(p.device.type != "meta" for p in model.parameters()):
        raise RuntimeError("diagnostic allocated real model parameter")
    self.counter_after_loading_weights = time.perf_counter()
    return model.eval()

def main():
    assert os.environ.get("PIREUS_PREPARATION_PROBE") == "1"
    rank = int(os.environ["PIREUS_RANK"])
    bundle = json.loads(Path(os.environ["PIREUS_OFFLINE_INPUT"]).read_bytes())
    args = offline.prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    bench = offline.bench
    bench._set_envs_and_config(args)
    bench.initialize_moe_config(args)
    bench.initialize_fp8_gemm_config(args)
    bench.initialize_fp4_gemm_config(args)
    offline.initialize_bf16_gemm_config(args)
    offline.initialize_mamba_selective_state_update_backend(args)
    loader.ModelOptModelLoader.load_model = meta_load
    loader.DefaultModelLoader.load_model = meta_load
    runner, cache = offline.load_worker_and_cache(args, rank)
    model = runner.torch_runner
    assert all(p.device.type == "meta" for p in model.model.parameters())
    bench.TreeCacheNamespace = lambda **kwargs: cache
    def stop_before_model(*args, **kwargs):
        offline.inference_memory("META_PREPARATION_REACHED_MODEL_BOUNDARY")
        raise StopBeforeModel()
    model.forward = stop_before_model
    item = bundle["items"][0]
    params = offline.SamplingParams(temperature=item["temperature"],
        max_new_tokens=item["max_new_tokens"], sampling_seed=item["seed"],
        stop_token_ids=set(item["stop_token_ids"]))
    params.normalize(None)
    params.verify(model.model_config.vocab_size)
    for iteration in range(2):
        cache.reset()
        runner.clear()
        req = offline.Req(rid=str(iteration), origin_input_text="",
            origin_input_ids=array("q", item["input_ids"]), sampling_params=params)
        req.init_next_round_input(cache)
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        offline.inference_memory("META_PREPARATION_BEGIN", iteration=iteration)
        try:
            runner.extend([req])
        except StopBeforeModel:
            offline.inference_memory("META_PREPARATION_PASS", iteration=iteration,
                checkpoint_tensors_loaded=False, model_forward_executed=False,
                inference_accepted=False, input_tokens=len(item["input_ids"]))
        else:
            raise RuntimeError("preparation control escaped model boundary")
    dist.barrier()

if __name__ == "__main__":
    main()
