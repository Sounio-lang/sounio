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
