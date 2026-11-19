# Token-ID serving profile

The pinned Inkling NVFP4 checkpoint remains TP2 across the two owned Spark
workers, context16384, concurrency1, output budget4096. The native host floor
remains32GiB. The optional token-ID profile stops its own rank process group
when the 50ms MemAvailable sample falls below33GiB; the default guard remains
36GiB. This is a sampled early-stop mechanism, not an atomic reservation.

This profile disables tokenizer initialization, CUDA graphs, and overlap
scheduling, sets chunked prefill128 and max_mamba_cache_size8, and uses the two
hash-bound Marlin source overlays. MALLOC_ARENA_MAX=2 and
MALLOC_TRIM_THRESHOLD_=131072 bound allocator behavior. Full serving acceptance
must be established separately; metadata-only profiles never qualify as loading.

The tokenizer runs in the pinned SIF under separate exclusive Slurm allocations:
encode all eight frozen messages before loading weights; generate with /generate
input_ids and preserve raw responses; stop serving; decode output_ids on both
nodes. Both tokenizer receipts must agree and bind the exact input bundle,
helper, revision, and tokenizer files. The pinned chat template receives
reasoning_effort=none. All output token IDs, special-token-inclusive decoding,
ordinary decoding, and request/response hashes are retained. The exact ordinary
decoded text is passed to native Sounio without JSON repair or extraction.

Commands, from a remote tmux session:

1. cycle.py prepare --condition inkling-ontology --transport sglang-token-ids
   with the actual native context engine, admission SHA, and frozen evidence.
2. tokenized_cycle.py pack-encode --run RUN.
3. runtime/launch_pair.py tokenize --tokenizer-input RUN/encode-bundle.json.
4. Copy the two job-bound tokenizer receipts from the workers into RUN, then
   tokenized_cycle.py accept-encode --run RUN --receipts RANK0 RANK1.
5. runtime/launch_pair.py serve-token-ids --minutes 60.
6. After readiness, tokenized_cycle.py generate --run RUN --serving-job JOB
   --endpoint http://CURRENT_RANK0_WORKER_IP:30000.
7. tokenized_cycle.py pack-decode --run RUN; stop only the owned serving job.
8. runtime/launch_pair.py tokenize --tokenizer-input RUN/decode-bundle.json.
9. Copy both decoder receipts; tokenized_cycle.py finalize --run RUN
   --receipts RANK0 RANK1.
10. Existing cycle.py validate, materialize, benchmark, and report use the
    frozen native engines unchanged.

An interrupted HTTP call leaves its immutable request behind and is never
automatically replayed. An invalid model response is preserved and reaches
native refusal. The ontology context is declared research knowledge, not
observed hardware facts. The deterministic control remains separately labeled.

Pinned SGLang sampling source:
https://github.com/sgl-project/sglang/blob/a74222ef6e690f851e2e4ff1c0be7dc1357be313/python/sglang/srt/sampling/sampling_params.py
