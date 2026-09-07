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
        or server_args.max_running_requests != 1 or server_args.load_format == "dummy"):
        raise ValueError("offline profile boundary")
    bench._set_envs_and_config(server_args)
    bench.initialize_moe_config(server_args)
    bench.initialize_fp8_gemm_config(server_args)
    bench.initialize_fp4_gemm_config(server_args)
    bench.configure_logger(server_args, prefix=" TP" + str(rank))
    # Tokenization has already run in the same pinned image; this local adapter
    # suppresses only one_batch.load_model's trailing tokenizer construction.
    bench.get_tokenizer = lambda *a, **kw: None
    emit("OFFLINE_MODEL_LOAD_BEGIN", one_batch_sha256=digest(Path(bench.__file__).read_bytes()),
         input_sha256=digest(raw), checkpoint_tensors_loaded=False)
    if os.environ.get("PIREUS_OFFLINE_INTERFACE") == "1":
        emit("OFFLINE_INTERFACE_PASS", model_loaded=False, random_seed=server_args.random_seed)
        return
    runner, tokenizer = bench.load_model(server_args, PortArgs.init_new(server_args), 0, rank)
    assert tokenizer is None
    model = runner.torch_runner
    emit("OFFLINE_MODEL_READY", max_total_num_tokens=model.max_total_num_tokens,
         checkpoint_tensors_loaded=True, http_serving=False)
    results = []
    for item in bundle["items"]:
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
        req.full_untruncated_fill_ids = req.origin_input_ids
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
