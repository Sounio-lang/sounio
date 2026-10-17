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

_lifecycle_namespace = {}
exec(compile('"""Diagnostic observations only; independent of the memory guardian."""\nimport json\nimport os\nfrom pathlib import Path\nimport threading\nimport time\n\nHOST_KEYS = ("MemAvailable", "Cached", "SReclaimable", "Shmem", "Slab", "Unevictable")\n\ndef fields(path, names):\n    try:\n        lines = Path(path).read_text().splitlines()\n        parsed = {p[0].rstrip(":"): int(p[1])*1024 for line in lines\n                  if len(p := line.split()) == 3 and p[2] == "kB"}\n        return {name: parsed.get(name) for name in names}, None\n    except (OSError, ValueError) as exc:\n        return {name: None for name in names}, type(exc).__name__\n\nclass Observer:\n    def __init__(self, path, job, rank, cuda, interval=1.0):\n        if not 0.1 <= interval <= 10:\n            raise ValueError("observer interval outside declared bounds")\n        self.out = Path(path).open("x")\n        self.job, self.rank, self.cuda = str(job), str(rank), cuda\n        self.interval = interval\n        self.lock = threading.Lock()\n        self.context = (None, None)\n        self.stop = threading.Event()\n        self.thread = None\n        self.record("OBSERVER_START", extra={"interval_seconds": interval,\n                    "cgroup_memory": None, "device_memory": None,\n                    "unavailable": ["cgroup_memory", "device_memory"],\n                    "peaks_scope": "process lifetime; no counter reset"})\n    def record(self, stage, index=None, token_index=None, process=False, extra=None):\n        started = time.monotonic_ns()\n        row = dict(schema="pireus-lifecycle-observation-v1", stage=stage,\n                   job=self.job, rank=self.rank, pid=os.getpid(),\n                   monotonic_ns=started, index=index, token_index=token_index)\n        row["host"], row["host_error"] = fields("/proc/meminfo", HOST_KEYS)\n        if process:\n            row["process"], row["process_error"] = fields("/proc/self/smaps_rollup",\n                                                        ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem"))\n            try:\n                pending, seen, total = [os.getpid()], set(), 0\n                while pending:\n                    parent = pending.pop()\n                    children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()\n                    for child in children:\n                        if child in seen:\n                            continue\n                        seen.add(child)\n                        pending.append(int(child))\n                        data, error = fields(f"/proc/{child}/status", ("VmRSS",))\n                        if error or data["VmRSS"] is None:\n                            raise OSError("descendant accounting unavailable")\n                        total += data["VmRSS"]\n                row["owned_child_rss_bytes"] = total\n                row["owned_child_error"] = None\n            except OSError as exc:\n                row["owned_child_rss_bytes"] = None\n                row["owned_child_error"] = type(exc).__name__\n            row["cuda"] = {}\n            for key, method in (("allocated", "memory_allocated"), ("reserved", "memory_reserved"),\n                                ("peak_allocated", "max_memory_allocated"),\n                                ("peak_reserved", "max_memory_reserved")):\n                try:\n                    row["cuda"][key] = getattr(self.cuda, method)()\n                except Exception as exc:\n                    row["cuda"][key] = None\n                    row["cuda"][key+"_error"] = type(exc).__name__\n        row.update(extra or {})\n        with self.lock:\n            row["observation_duration_ns"] = time.monotonic_ns()-started\n            self.out.write(json.dumps(row, sort_keys=True)+"\\n")\n            self.out.flush()\n    def mark(self, stage, index, token_index=None):\n        with self.lock:\n            self.context = (index, token_index)\n        self.record(stage, index, token_index, process=True)\n    def _sample(self):\n        previous = time.monotonic_ns()\n        while not self.stop.wait(self.interval):\n            now = time.monotonic_ns()\n            with self.lock:\n                index, token = self.context\n            self.record("HOST_SAMPLE", index, token,\n                        extra={"sample_gap_ns": now-previous,\n                               "context_scope": "last runtime observation"})\n            previous = now\n    def start(self):\n        if self.thread is not None:\n            raise ValueError("observer already started")\n        self.thread = threading.Thread(target=self._sample, daemon=True)\n        self.thread.start()\n    def close(self):\n        self.stop.set()\n        if self.thread is not None:\n            self.thread.join(timeout=2*self.interval+1)\n            if self.thread.is_alive():\n                raise RuntimeError("observer did not stop")\n        self.record("OBSERVER_END")\n        self.out.close()\n', '<lifecycle-observer>', 'exec'), _lifecycle_namespace)
lifecycle = _lifecycle_namespace['Observer'](
    Path('/scratch/pireus/receipts') / ('lifecycle-' + os.environ['SLURM_JOB_ID'] + '-' + os.environ['PIREUS_RANK'] + '.jsonl'),
    os.environ['SLURM_JOB_ID'], os.environ['PIREUS_RANK'], torch.cuda)
lifecycle.start()
import atexit
atexit.register(lifecycle.close)

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

def release_loading_temporaries(model, label):
    import ctypes
    import gc
    def state():
        return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()))
    emit("OFFLINE_RELEASE_BEGIN", phase=label, **state())
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    libc=ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes=[ctypes.c_size_t]
    libc.malloc_trim.restype=ctypes.c_int
    result=libc.malloc_trim(0)
    emit("OFFLINE_RELEASE_END", phase=label, malloc_trim_result=result, **state())

def inference_memory(stage, **fields):
    # Read only process accounting; never dump mappings, environment or tensors.
    accounting = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            accounting[parts[0].rstrip(":")] = int(parts[1]) * 1024
    pending = [os.getpid()]
    seen = set()
    child_rss = 0
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except FileNotFoundError:
            continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            pending.append(int(child))
            try:
                status = Path(f"/proc/{child}/status").read_text().splitlines()
                child_rss += next((int(l.split()[1])*1024 for l in status if l.startswith("VmRSS:")), 0)
            except FileNotFoundError:
                pass
    fields.update(owned_descendant_count=len(seen), owned_descendant_rss_bytes=child_rss)
    emit(stage, cuda_allocated_bytes=torch.cuda.memory_allocated(),
         cuda_reserved_bytes=torch.cuda.memory_reserved(), process_memory=accounting, **fields)

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
    release_loading_temporaries(worker.model_runner.model,"after_weight_load")
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
    release_loading_temporaries(model.model,"before_first_inference")
    return bench._TorchBenchRunner(model), result.tree_cache

def main():
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("PIREUS_OFFLINE_MODE") == "generate"
    job, rank = os.environ["SLURM_JOB_ID"], int(os.environ["PIREUS_RANK"])
    required_env = {"SGLANG_OPT_LINEARIZED_SHARED_SINK": "0",
                    "NCCL_MAX_NCHANNELS": "2", "NCCL_BUFFSIZE": "262144",
                    "TORCHINDUCTOR_COMPILE_THREADS": "1"}
    if any(os.environ.get(k) != v for k, v in required_env.items()):
        raise ValueError("offline memory profile environment mismatch")
    import torch._inductor.config as inductor_config
    if inductor_config.compile_threads != 1:
        raise ValueError("offline Inductor compiler concurrency mismatch")
    emit("OFFLINE_PROFILE_ENV_VERIFIED", environment=required_env)

    path = Path(os.environ["PIREUS_OFFLINE_INPUT"])
    raw = path.read_bytes()
    bundle = json.loads(raw)
    if bundle["revision"] != REVISION or bundle["mode"] != "offline-generate" or len(bundle["items"]) not in (8, 32):
        raise ValueError("unrecognized frozen 8- or 32-proposal bundle")
    if [item["index"] for item in bundle["items"]] != list(range(len(bundle["items"]))):
        raise ValueError("frozen offline request indices differ")
    if digest(Path(bench.__file__).read_bytes()) != "f831549fc4c8163aa878c3dac1dff6f4f853227d607942befae8971bf6908f35":
        raise ValueError("unrecognized pinned one_batch source")
    server_args = prepare_server_args(sys.argv[1:] + ["--random-seed", "20260907"])
    if (server_args.tp_size != 2 or server_args.nnodes != 2 or server_args.node_rank != rank
        or not server_args.skip_tokenizer_init or server_args.context_length != 16384
        or server_args.max_running_requests != 1 or server_args.load_format == "dummy"
        or server_args.disable_radix_cache or server_args.max_total_tokens != 6144
        or server_args.swa_full_tokens_ratio != 0.15):
        raise ValueError("offline profile boundary")
    for item in bundle["items"]:
        if not item["input_ids"] or len(item["input_ids"]) + item["max_new_tokens"] > 6144:
            raise ValueError("frozen request exceeds the 6144-token offline cache budget")
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
        emit("OFFLINE_INTERFACE_PASS", model_loaded=False, random_seed=server_args.random_seed,
             max_total_tokens=server_args.max_total_tokens, context_length=server_args.context_length)
        return
    from sglang.srt.model_loader.loader import DefaultModelLoader
    original_iterator = DefaultModelLoader._get_all_weights
    def traced_weights(loader, model_config, loaded_model):
        emit("OFFLINE_PARAMETERS_CONSTRUCTED",
             parameter_bytes=sum(p.numel()*p.element_size() for p in loaded_model.parameters()),
             checkpoint_tensors_loaded=False)
        for name, weight in original_iterator(loader, model_config, loaded_model):
            size = weight.numel()*weight.element_size()
            if size >= 64*1024**2:
                emit("OFFLINE_CHECKPOINT_TENSOR_BEGIN", name=name, shape=list(weight.shape),
                     dtype=str(weight.dtype), bytes=size)
            yield name, weight
            if size >= 64*1024**2:
                emit("OFFLINE_CHECKPOINT_TENSOR_END", name=name)
        emit("OFFLINE_CHECKPOINT_ITERATOR_COMPLETE")
    DefaultModelLoader._get_all_weights = traced_weights
    try:
        runner, tree_cache = load_worker_and_cache(server_args, rank)
    finally:
        DefaultModelLoader._get_all_weights = original_iterator
    model = runner.torch_runner
    bench.TreeCacheNamespace = lambda **kwargs: tree_cache
    import offload_embedding
    lock = json.loads(Path(__file__).with_name("embedding-offload-lock.json").read_bytes())
    if lock["revision"] != REVISION:
        raise ValueError("embedding lock revision mismatch")
    if model.model.llm.embed_tokens.weight.data_ptr() == model.model.llm.lm_head.weight.data_ptr():
        raise ValueError("tied embedding storage requires separate qualification")
    inference_memory("OFFLINE_EMBEDDING_OFFLOAD_BEGIN")
    embedding_storage = offload_embedding.offload(model.model.llm.embed_tokens,
        "/scratch/pireus/cache/embedding-offload")
    embedding_storage["helper_sha256"] = digest(Path(offload_embedding.__file__).read_bytes())
    if (embedding_storage["source_gpu_sha256"] != lock["rank_sha256"][str(rank)]
        or embedding_storage["file_sha256"] != lock["rank_sha256"][str(rank)]
        or embedding_storage["bytes"] != lock["bytes"]):
        raise ValueError("embedding bytes differ from qualified pinned TP shard")
    inference_memory("OFFLINE_EMBEDDING_OFFLOAD_END", embedding_storage=embedding_storage)

    import offload_lm_head
    lm_lock = json.loads(Path(__file__).with_name("lm-head-offload-lock.json").read_bytes())
    if lm_lock["revision"] != REVISION:
        raise ValueError("LM-head lock revision mismatch")
    inference_memory("OFFLINE_LM_HEAD_OFFLOAD_BEGIN")
    lm_head_storage = offload_lm_head.offload(model.model.llm.lm_head,
        model.model.llm.logits_processor, "/scratch/pireus/cache/lm-head-offload")
    lm_head_storage["helper_sha256"] = digest(Path(offload_lm_head.__file__).read_bytes())
    if (lm_head_storage["source_gpu_sha256"] != lm_lock["rank_sha256"][str(rank)]
        or lm_head_storage["file_sha256"] != lm_lock["rank_sha256"][str(rank)]
        or lm_head_storage["bytes"] != lm_lock["bytes"]
        or lm_head_storage["tile_rows"] != lm_lock["tile_rows"]):
        raise ValueError("LM-head bytes/profile differ from qualified pinned TP shard")
    inference_memory("OFFLINE_LM_HEAD_OFFLOAD_END", lm_head_storage=lm_head_storage)
    emit("OFFLINE_MODEL_READY", max_total_num_tokens=model.max_total_num_tokens,
         checkpoint_tensors_loaded=True, http_serving=False)
    profile = dict(schema=1, scope=("frozen-offline-pilot-batch" if len(bundle["items"]) == 32 else "frozen-offline-canary"), transport="sglang-offline-token-ids",
                   tp_size=2, jit_cache_storage="local-ssd", inductor_compile_threads=1, embedding_placement="file-backed-cpu", lm_head_placement="file-backed-gpu-tiles", lm_head_tile_rows=4096, lm_head_hidden_rows=1, lm_head_numerical_scope="qualified-controls-only", collective_backend="existing-pynccl", context_length=server_args.context_length,
                   max_total_tokens=server_args.max_total_tokens,
                   actual_full_tokens=model.full_max_total_num_tokens,
                   actual_swa_tokens=model.swa_max_total_num_tokens,
                   swa_full_tokens_ratio=server_args.swa_full_tokens_ratio,
                   page_size=server_args.page_size, max_running_requests=1,
                   max_new_tokens=4096, native_host_floor_gib=32, early_stop_gib=33,
                   http_serving=False, general_16k_inference_accepted=False)
    emit("OFFLINE_EXECUTION_PROFILE", execution_profile=profile)
    hooks = []
    def trace_before(name):
        def hook(module, args):
            inference_memory("OFFLINE_FIRST_FORWARD_LAYER_BEGIN", layer=name)
        return hook
    def trace_after(name):
        def hook(module, args, output):
            inference_memory("OFFLINE_FIRST_FORWARD_LAYER_END", layer=name)
        return hook
    for name, module in model.model.named_modules():
        if (name.startswith("llm.layers.") and name.count(".") == 2
            or name.startswith("llm.layers.0.") and name.count(".") <= 5):
            hooks.extend([module.register_forward_pre_hook(trace_before(name)),
                          module.register_forward_hook(trace_after(name))])
    comm = model.tp_group.pynccl_comm
    if comm is None or not comm.available:
        raise ValueError("offline TP2 requires its initialized PyNCCL communicator")
    results = []
    for item in bundle["items"]:
        inference_memory("OFFLINE_REQUEST_BEGIN", index=item["index"])
        tree_cache.reset()
        runner.clear()
        ids = item["input_ids"]
        if not ids or len(ids) + item["max_new_tokens"] > min(16384,model.max_total_num_tokens):
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
        lifecycle.mark("EXTEND_ENTRY", item["index"])
        inference_memory("OFFLINE_EXTEND_BEGIN", index=item["index"])
        with comm.change_state(enable=True):
            next_ids, logits, batch = runner.extend([req])
        for hook in hooks:
            hook.remove()
        hooks.clear()
        inference_memory("OFFLINE_EXTEND_END", index=item["index"])
        output = []
        finish = "length"
        lifecycle.mark("DECODE_ENTRY", item["index"], 0)
        for step in range(item["max_new_tokens"]):
            if step % 16 == 0:
                lifecycle.mark("DECODE_SAMPLE", item["index"], step)
            # The TP model has one sampling authority: rank zero. Both ranks
            # consume the same chosen token on the next decode iteration.
            with comm.change_state(enable=True):
                comm.broadcast(next_ids, src=0)
            token = int(next_ids.item())
            if step == 0:
                emit("OFFLINE_FIRST_TOKEN", index=item["index"], token_id=token)
            output.append(token)
            req.output_ids.append(token)
            if token in item["stop_token_ids"]:
                finish = "stop"
                break
            if step + 1 < item["max_new_tokens"]:
                with comm.change_state(enable=True):
                    next_ids, logits = runner.decode(next_ids, batch)
        torch.cuda.synchronize()
        lifecycle.mark("DECODE_EXIT", item["index"], len(output))
        response = dict(schema=1, transport="sglang-offline-token-ids", index=item["index"],
                        output_ids=output, finish_reason=finish,
                        prompt_tokens=len(ids), completion_tokens=len(output),
                        sampling_authority_rank=0, execution_profile=profile, input_sha256=digest(raw), job=job, revision=REVISION)
        out = Path("/scratch/pireus/receipts") / f"offline-{job}-{rank}-{item['index']:03d}.json"
        response_sha = write(out, response)
        results.append(dict(index=item["index"], response_sha256=response_sha, output_tokens=len(output)))
        emit("OFFLINE_PROPOSAL_SAVED", index=item["index"], output_tokens=len(output),
             seconds=time.monotonic()-started, finish_reason=finish, response_sha256=response_sha)
        lifecycle.mark("PROPOSAL_SAVED", item["index"], len(output))
        lifecycle.mark("CLEANUP_BEFORE", item["index"], len(output))
        runner.cleanup(batch)
        lifecycle.mark("CLEANUP_AFTER", item["index"], len(output))
        del batch, req, logits, next_ids
        lifecycle.mark("REFERENCES_RELEASED", item["index"], len(output))
    dist.barrier(group=model.tp_group.cpu_group)
    receipt = dict(schema=1, stage="OFFLINE_CYCLE_COMPLETE", job=job, rank=str(rank),
                   revision=REVISION, input_sha256=digest(raw),
                   helper_sha256=digest(Path(__file__).read_bytes()),
                   model_loaded=True, http_serving=False, execution_profile=profile,
                   embedding_storage=embedding_storage, lm_head_storage=lm_head_storage, results=results)
    out = Path("/scratch/pireus/receipts") / f"offline-{job}-{rank}-complete.json"
    write(out, receipt)
    emit("OFFLINE_CYCLE_COMPLETE", count=len(results), http_serving=False)

if __name__ == "__main__":
    main()
