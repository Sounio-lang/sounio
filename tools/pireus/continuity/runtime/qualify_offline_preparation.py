#!/usr/bin/env python3
"""Diagnostic: hybrid preparation and synthetic embedding with remaining weights meta.
Never reads checkpoint tensors, runs transformer layers, samples, or emits proposals.
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
    original_forward = model.model.forward
    model.model.forward = stop_before_model
    item = bundle["items"][0]
    params = offline.SamplingParams(temperature=item["temperature"],
        max_new_tokens=item["max_new_tokens"], sampling_seed=item["seed"],
        stop_token_ids=set(item["stop_token_ids"]))
    params.normalize(None)
    params.verify(model.model_config.vocab_size)
    for iteration in range(3):
        scope = "model-entry" if iteration == 0 else "synthetic-embedding-only"
        if iteration == 1:
            model.model.forward = original_forward
            for module, fill in [(model.model.llm.embed_tokens, 0),
                                 (model.model.llm.embed_norm, 1)]:
                old = module.weight
                module.weight = torch.nn.Parameter(torch.full(old.shape, fill,
                    dtype=old.dtype, device="cuda"), requires_grad=False)
            model.model.llm.layers[0].forward = stop_before_model
        cache.reset()
        runner.clear()
        req = offline.Req(rid=str(iteration), origin_input_text="",
            origin_input_ids=array("q", item["input_ids"]), sampling_params=params)
        req.init_next_round_input(cache)
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        offline.inference_memory("META_PREPARATION_BEGIN", iteration=iteration, scope=scope)
        try:
            comm = model.tp_group.pynccl_comm
            assert comm is not None and comm.available
            with comm.change_state(enable=True):
                control = torch.tensor([rank+1.0], device="cuda")
                comm.all_reduce(control)
                assert control.item() == 3.0
                control.fill_(17.0 if rank == 0 else -1.0)
                comm.broadcast(control, src=0)
                assert control.item() == 17.0
                runner.extend([req])
        except StopBeforeModel:
            offline.inference_memory("META_PREPARATION_PASS", iteration=iteration, scope=scope,
                checkpoint_tensors_loaded=False, transformer_layer_executed=False,
                inference_accepted=False, input_tokens=len(item["input_ids"]))
        else:
            raise RuntimeError("preparation control escaped model boundary")
    dist.barrier(group=model.tp_group.cpu_group)

if __name__ == "__main__":
    main()
