# Inkling TP2 memory qualification

The immutable image identifies SGLang commit
a74222ef6e690f851e2e4ff1c0be7dc1357be313. Its KV configurator reserves
pre-model-load free GPU memory times (1 - mem_fraction_static), then
subtracts that slack from post-load available memory. The option is not a
hard host MemAvailable reservation. The user token cap can further reduce
the allocated pool.

Source:
https://github.com/sgl-project/sglang/blob/a74222ef6e690f851e2e4ff1c0be7dc1357be313/python/sglang/srt/mem_cache/kv_cache_configurator.py#L1460

The checkpoint header inventory records serialized tensor bytes:
- model.llm: 166130594960
- model.mtp: 4463824912
- model.visual: 128160768
- model.audio: 10493952

See ../validation/inkling-tensor-storage-inventory.json for per-file header
hashes. Serialized bytes are not measured runtime allocations or a proof of
equal TP distribution. Runtime repacking, replicated tensors, loader peaks,
CUDA/communication allocations and cache pools need separate measurement.

Both hosts reported MemTotal=127600748 kB on 2026-09-06. A read-only fenced
snapshot had MemAvailable=115361536 kB on 3c59 and 120081732 kB on 8e54.
These observations expire when workers and processes restart.

In earlier job 11864, SGLang logged distributed initialization memory usage
2.05 GB / 2.00 GB and pre-weight available memory 104.32 GB / 110.46 GB.
The job was interrupted before completed weight loading; the first fence
trigger remains unproven. These are historical log labels, not converted
claims about host MemAvailable.

Before repeating the load, capture fresh host and CUDA memory after worker/
runtime qualification, account for the unchanged 32768 MiB host floor,
inspect the pinned loader's transient allocations, and cap cache tokens for
the 16384-context/concurrency-one smoke. A successful paper budget or the
upstream two-Spark recipe is not local serving acceptance.

The next serving canary explicitly caps max-total-tokens at 16384, matching
the context-length and single-request smoke. This cap limits cache sizing;
it does not reserve host memory or prove the model fits. TP2 qualification
now emits post-collective CUDA free/total and allocator counters alongside
host MemTotal/MemAvailable, all in bytes and bound to the Slurm job and rank.

## Protected-database migration and bounded serving canary

The production database moved off Spark3c59 on 2026-09-07, with final
endpoint/scheduler activation in 559.823 seconds and all migration gates
passing. Fresh Slurm job11877 passed TP2/InfiniBand on both ranks with
NCCL_MAX_NCHANNELS=8 and NCCL_BUFFSIZE=1048576. Post-collective host
MemAvailable was121450905600 bytes on3c59 and121386692608 bytes on8e54.
CUDA free memory differed (90431643648 and84970627072 bytes respectively);
host and CUDA availability are distinct observations.

Serving now uses a rank-local memory guardian outside Apptainer. It samples
host MemAvailable every50ms and kills only its owned child process group
below36GiB, causing srun's existing kill-on-bad-exit policy to stop the pair.
Missing observations or interruption also kill the owned group. Four tests
cover refusal, child exit status, low-memory cancellation without touching
an unrelated process, and failed observation cleanup. This is an additional
early-stop mechanism, not an atomic reservation or proof that the immutable
32GiB native host floor can never be crossed between samples.

The pinned Inkling implementation builds audio/vision towers only when
enable_multimodal is true; the current text-only launch omits that option.
Source: https://github.com/sgl-project/sglang/blob/a74222ef6e690f851e2e4ff1c0be7dc1357be313/python/sglang/srt/models/inkling.py#L1036
