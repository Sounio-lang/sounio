#!/usr/bin/env python3
"""Single-process-per-rank SGLang inference using its pinned one_batch engine path."""
from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch.distributed as dist
import sglang.benchmark.one_batch as bench
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.kv_cache_builder import build_kv_cache
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.kernels.ops.mamba.triton_ops import initialize_mamba_selective_state_update_backend
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs
from sglang.srt.server_args import prepare_server_args

REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def emit(stage, **fields):
    available = next(int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines()
                     if l.startswith("MemAvailable:"))
    print(json.dumps(dict(stage=stage, job=os.environ["SLURM_JOB_ID"],
                          rank=os.environ["PIREUS_RANK"], available_bytes=available, **fields)), flush=True)

def write(path, value):
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return digest(raw)

def load_worker_and_cache(server_args, rank):
    # Same parallel layout as pinned one_batch.load_model; use the real worker
    # so the scheduler's hybrid cache builder receives its complete interface.
    at_rank, at_size, ad_rank, ad_size = bench.compute_dp_attention_world_info(
        server_args.enable_dp_attention, rank, server_args.tp_size,
        server_args.dp_size, server_args.attn_cp_size)
    ps = bench.ParallelState(
        tp_rank=rank, tp_size=2, pp_rank=0, pp_size=1, dp_rank=None,
        dp_size=server_args.dp_size, attn_tp_rank=at_rank, attn_tp_size=at_size,
        attn_cp_rank=0, attn_cp_size=server_args.attn_cp_size,
        attn_dp_rank=ad_rank, attn_dp_size=ad_size,
        moe_ep_rank=rank // (2 // server_args.ep_size), moe_ep_size=server_args.ep_size,
        moe_dp_rank=None, moe_dp_size=server_args.moe_dp_size,
        dcp_size=server_args.dcp_size, gpu_id=0)
    worker = TpModelWorker(server_args=server_args, gpu_id=0, ps=ps,
                           nccl_port=PortArgs.init_new(server_args).nccl_port)
    assert worker.tokenizer is None
    worker.alloc_memory_pool()
    worker.init_attention_backends()
    worker.init_cuda_graphs()
    model = worker.model_runner
    result = build_kv_cache(
        server_args=server_args, model_config=model.model_config, tp_worker=worker,
        page_size=server_args.page_size, spec_algorithm=bench.SpeculativeAlgorithm.NONE,
        attn_tp_cpu_group=model.tp_group.cpu_group, tp_cpu_group=model.tp_group.cpu_group,
        attn_cp_cpu_group=None, enable_metrics=False, enable_kv_cache_events=False,
        ps=ps, tp_group=model.tp_group, pp_group=worker.pp_group,
        enable_hierarchical_cache=False)
    if not result.tree_cache.supports_mamba() or not result.tree_cache.supports_swa():
        raise ValueError("Inkling requires its real hybrid cache")
    return bench._TorchBenchRunner(model), result.tree_cache

def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("PIREUS_OFFLINE_MODE") == "generate"
    job, rank = os.environ["SLURM_JOB_ID"], int(os.environ["PIREUS_RANK"])
    path = Path(os.environ["PIREUS_OFFLINE_INPUT"])
    raw = path.read_bytes()
    bundle = json.loads(raw)
    if bundle["revision"] != REVISION or bundle["mode"] != "offline-generate" or len(bundle["items"]) != 8:
        raise ValueError("unrecognized frozen eight-proposal bundle")
    if digest(Path(bench.__file__).read_bytes()) != "f831549fc4c8163aa878c3dac1dff6f4f853227d607942befae8971bf6908f35":
        raise ValueError("unrecognized pinned one_batch source")
    server_args = prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    if (server_args.tp_size != 2 or server_args.nnodes != 2 or server_args.node_rank != rank
        or not server_args.skip_tokenizer_init or server_args.context_length != 16384
        or server_args.max_running_requests != 1 or server_args.load_format == "dummy"
        or server_args.disable_radix_cache):
        raise ValueError("offline profile boundary")
    bench._set_envs_and_config(server_args)
    bench.initialize_moe_config(server_args)
    bench.initialize_fp8_gemm_config(server_args)
    bench.initialize_fp4_gemm_config(server_args)
    initialize_bf16_gemm_config(server_args)
    initialize_mamba_selective_state_update_backend(server_args)
    bench.configure_logger(server_args, prefix=" TP" + str(rank))
    emit("OFFLINE_MODEL_LOAD_BEGIN", one_batch_sha256=digest(Path(bench.__file__).read_bytes()),
         input_sha256=digest(raw), checkpoint_tensors_loaded=False)
    if os.environ.get("PIREUS_OFFLINE_INTERFACE") == "1":
        emit("OFFLINE_INTERFACE_PASS", model_loaded=False, random_seed=server_args.random_seed)
        return
    runner, tree_cache = load_worker_and_cache(server_args, rank)
    model = runner.torch_runner
    bench.TreeCacheNamespace = lambda **kwargs: tree_cache
    emit("OFFLINE_MODEL_READY", max_total_num_tokens=model.max_total_num_tokens,
         checkpoint_tensors_loaded=True, http_serving=False)
    results = []
    for item in bundle["items"]:
        tree_cache.reset()
        runner.clear()
        ids = item["input_ids"]
        if not ids or len(ids) + item["max_new_tokens"] > 16384:
            raise ValueError("request exceeds frozen context")
        params = SamplingParams(temperature=item["temperature"], max_new_tokens=item["max_new_tokens"],
                                sampling_seed=item["seed"], stop_token_ids=set(item["stop_token_ids"]))
        params.normalize(None)
        params.verify(model.model_config.vocab_size)
        torch.manual_seed(item["seed"])
        req = Req(rid=str(item["index"]), origin_input_text="", origin_input_ids=array("q", ids),
                  sampling_params=params)
        req.init_next_round_input(tree_cache)
        req.logprob_start_len = -1
        req.set_extend_range(len(req.prefix_indices), len(req.origin_input_ids))
        started = time.monotonic()
        next_ids, logits, batch = runner.extend([req])
        output = []
        finish = "length"
        for step in range(item["max_new_tokens"]):
            # The TP model has one sampling authority: rank zero. Both ranks
            # consume the same chosen token on the next decode iteration.
            dist.broadcast(next_ids, src=0, group=model.tp_group.device_group)
            token = int(next_ids.item())
            output.append(token)
            req.output_ids.append(token)
            if token in item["stop_token_ids"]:
                finish = "stop"
                break
            if step + 1 < item["max_new_tokens"]:
                next_ids, logits = runner.decode(next_ids, batch)
        torch.cuda.synchronize()
        response = dict(schema=1, transport="sglang-offline-token-ids", index=item["index"],
                        output_ids=output, finish_reason=finish,
                        prompt_tokens=len(ids), completion_tokens=len(output),
                        sampling_authority_rank=0, input_sha256=digest(raw), job=job, revision=REVISION)
        out = Path("/scratch/pireus/receipts") / f"offline-{job}-{rank}-{item['index']:03d}.json"
        response_sha = write(out, response)
        results.append(dict(index=item["index"], response_sha256=response_sha, output_tokens=len(output)))
        emit("OFFLINE_PROPOSAL_SAVED", index=item["index"], output_tokens=len(output),
             seconds=time.monotonic()-started, finish_reason=finish, response_sha256=response_sha)
        runner.cleanup(batch)
        del batch, req, logits, next_ids
    dist.barrier()
    receipt = dict(schema=1, stage="OFFLINE_CYCLE_COMPLETE", job=job, rank=str(rank),
                   revision=REVISION, input_sha256=digest(raw),
                   helper_sha256=digest(Path(__file__).read_bytes()),
                   model_loaded=True, http_serving=False, results=results)
    out = Path("/scratch/pireus/receipts") / f"offline-{job}-{rank}-complete.json"
    write(out, receipt)
    emit("OFFLINE_CYCLE_COMPLETE", count=len(results), http_serving=False)

if __name__ == "__main__":
    main()
