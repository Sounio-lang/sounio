# Single-process Inkling cycle

HTTP serving attempts11892 and constructor diagnostics11893/11895/11896 did
not finish within the33GiB early guard. Native host floor32GiB remained intact
at the recorded stops. Metadata-only11894 identified640MiB per rank in
linearized shared-expert copies; disabling the stock optimization removes those
copies, but the server process structure still leaves too little margin.

The offline profile runs the pinned SGLang TpModelWorker and one_batch extend/decode engine
in one process per Slurm rank. It bypasses the benchmark CLI's multiprocessing
launcher. The real TpModelWorker honors skip_tokenizer_init directly. It uses the
real checkpoint loader, ModelRunner, ScheduleBatch, attention backend, and
sampler. It does not use dummy weights or benchmark synthetic input. Exact
one_batch source SHA is frozen in runtime-lock.json.

The eight prompts and sampling settings are frozen before launch. Rank zero
is the single sampling authority for this TP model and broadcasts every chosen
token to both ranks. Each rank persists all raw output IDs with input-bundle
identity, model revision, job, token counts, and finish reason. Matching worker
receipts and response hashes are required before the workspace imports them.
These are explicitly offline responses, not fabricated HTTP envelopes.

The profile retains NVFP4, TP2, context16384, max_total_tokens16384,
concurrency1, max output4096, and native32GiB host reserve. It uses the same
33GiB sampled guardian, graph/overlap disabling, prefill128 and Mamba cache8.
The canonical hybrid prefix cache remains enabled; Inkling requires it.
The cache is reset between independent requests. Custom all-reduce is disabled. The stock
SGLANG_OPT_LINEARIZED_SHARED_SINK=0 configuration avoids the640MiB duplicate
buffers. NCCL uses two channels and262144-byte buffers. CUDA memory caching
stays enabled; the no-cache diagnostic11896 did not resolve capacity and its
allocator counters were zero, so no memory saving is claimed for it.
The optional own-checkpoint POSIX_FADV_DONTNEED helper is experimental and
disabled in this profile; it never changes file contents or global caches.

The ModelOpt overlay now combines placeholder omission with expert-local
deinterleave. The Marlin overlay reuses expert storage during repack. GPU
job11897 tests the actual gate/up interleaved layout on both nodes, all consumed
transformed parameter bytes, exact test-only non-atomic output, storage
identity, and the consumed-scale negative control. Peak temporary allocation
remains12584960 bytes at256experts,H4096,I1024. Production atomic output
determinism remains unclaimed.

Operational sequence, in remote tmux:

1. Prepare a fresh cycle.py manifest with --transport sglang-offline-token-ids.
2. tokenized_cycle.py pack-encode; obtain matching pinned tokenizer receipts.
   Existing receipts can be reused only if the exact bundle and helper hashes
   match; no model request may be replayed under that reuse.
3. tokenized_cycle.py accept-encode, then pack-offline.
4. SGLANG_OPT_LINEARIZED_SHARED_SINK=0 NCCL_MAX_NCHANNELS=2
   NCCL_BUFFSIZE=262144 runtime/launch_pair.py offline-generate --minutes90
   --input-bundle RUN/offline-bundle.json.
5. After the exclusive generation job exits, collect both workers' eight
   response files and completion receipts into a private worker directory.
6. tokenized_cycle.py accept-offline --run RUN --worker-dir DIRECTORY.
7. pack-decode, then a separate pinned-SIF tokenizer Slurm job on both nodes.
8. finalize uses exact decoded text without repair; native validate/materialize
   and pair parity/timing follow using the frozen Sounio executables.

A successful offline cycle establishes real model execution and proposal
custody for this batch. It does not establish HTTP service readiness,
continuous production serving, a performance gain, or formal V13/V14 closure.

Pinned engine source:
https://github.com/sgl-project/sglang/blob/a74222ef6e690f851e2e4ff1c0be7dc1357be313/python/sglang/benchmark/one_batch.py

Attempt11900 stopped before parameters because Inkling rejects disable_radix_cache. The corrected adapter uses TpModelWorker plus the scheduler build_kv_cache path, initializes BF16/Mamba backends, and uses the real hybrid cache in one_batch.extend. Interface11901 passes on both nodes; it is not a model-load result.
