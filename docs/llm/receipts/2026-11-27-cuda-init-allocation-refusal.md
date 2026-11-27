<!-- docs:meta
topic_id: repo.docs.llm.receipts.2026-11-27-cuda-init-allocation-refusal
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.llm.receipts.2026-11-27-cuda-init-allocation-refusal
-->

# CUDA runner allocation refusal (2026-11-27)

**OBSERVED:** R740 diagnostic jobs 29653/29654 returned CUDA Driver API
`CUresult=2` on weight-shard allocations. The previous initializer retained
`init_ok=1`, continued allocating later shards, work and KV/RoPE resources,
and only later returned runner false (`shards=0`). The old production
allocator discarded the specific error code. This is an independent
reporting and fail-closed control-flow defect, not a numerical correction.

The new `cuda_init_alloc_result` obtains the exact driver result, reports
`CUDA_DEVICE_ALLOC_REFUSED stage=… bytes=… CUresult=…`, and clears a sticky
initialization-admission flag. All later initialization allocations and
shard uploads honor that flag, including nonlinear, work, K and V cache
and RoPE paths. Already-owned successful allocations remain in the failed
runner so the existing `qwen3_cuda_runner_free` releases them. A `-1`
CUresult denotes a local precondition/host scratch failure (no driver call).
The checked path is limited to runner initialization. No change to the
arithmetic contract or the pinned Granite numerical audit.

## Discriminating CPU-only regression

`tests/llm/cuda_init_alloc_refusal.sio` constructs a tiny three-shard mapped
fixture and invokes the real runner. The C++23 fake CUDA DSO is test only,
provides the session/module symbols, makes the second `cuMemAlloc_v2`
return 2, and counts allocation/free calls. The test does not allocate
GPU memory or read actual model weights. It requires exactly two calls,
the named refusal with the exact 64-byte request and CUresult 2, runner
false, and cleanup of the single successful allocation.

Same test files in both valid runs; source-built Madaros SHA256
`5df8c456cdb5db06bf48485b4ebfdbae518e39be4d50efa798b5d00ebdbd3659`.
All allocations were CPU-only on `dl380-dl380-proxmox`, neither dl380p nor
5860, Slurm comment `lane:llm-granite`.

| Job | Source | State, elapsed, batch MaxRSS | Result |
|---|---|---|---|
| 29667 | `ae078834a4e8a3204200509dfd179aab9b2d0124` (before, same tests overlaid) | COMPLETED 0:0, 00:02:24, 3590304K | Baseline reproduced: 7 calls, 6 frees; second returns 2 but no named refusal; init=1. |
| 29668 | `b5cd0ee6347131ed86999f203c6e205acc1e6f55` (diagnostic branch after) | COMPLETED 0:0, 00:02:07, 3333372K | `CUDA_INIT_ALLOC_REFUSAL_PASS`; named 64-byte refusal, exactly 2 calls, 1 free, init=0, shard/work/K/V/RoPE flags all false. |

The baseline batch is COMPLETED only because it asserts the expected old
failure mechanism. The corrected diagnostic branch prints the exact
refusal before skipping all subsequent allocations. The fake driver does
not implement `cuMemGetInfo_v2` (the optional diagnostic query reports -1);
this test concerns allocation refusal, not VRAM measurement. Evidence logs
and hashes are under `/orangefs/training/llm-granite-n10/alloc-refusal-29667/`
and `alloc-refusal-29668/`.

## Direct target-branch regression

**OBSERVED.** On the first PR revision, jobs 29663 (00:01:27, 2079148K)
and 29664 (00:01:28, 2072304K) stopped at E137 before the probe could run.
The v2 base was missing a declaration of `ptr_write_u8` in
`bf16_cstr_append`. Two intermediate fixture runs, 29665 and 29666,
stopped at E039 until the fixture used a single model disposal path.

Commit `12eed46a3dbfcdc3144b19a4996cbb3c7cc7ec52` supplies the ported
native raw-memory accessors, POSIX mappings and effect declarations used
by the runner on the v2 base. This commit contains the old allocator and
serves as the control. Job 29735 overlaid only the two fixture source
files from `5be76e3a3307936266d033896c9edac578f51342`. Job 29736 built
that revision of the PR directly. Both jobs used the same C++23 mock and
source-built Madaros with the hash recorded above.

| Job | Slurm result | Wall | Batch MaxRSS | Observation |
|---|---|---:|---:|---|
| 29735 · before | COMPLETED 0:0 | 00:02:29 | 3528412K | Second allocation returned 2, then the runner made five further allocation calls. Seven allocations attempted, six freed, no named refusal. |
| 29736 · after | COMPLETED 0:0 | 00:02:10 | 3272432K | `CUDA_DEVICE_ALLOC_REFUSED stage=shard bytes=64 CUresult=2` at the second call. Two allocations attempted, one freed, probe exit 0 and `CUDA_INIT_ALLOC_REFUSAL_PASS`. |
| 29737 · verification | COMPLETED 0:0 | 00:00:32 | 2041980K | Verified both artifact manifests, checked all eleven affected Sounio library modules, built and ran the raw memory and mapping contract tests. Both printed PASS. |

All three jobs ran CPU-only on `dl380-dl380-proxmox`. Executables, driver
doubles, run logs and SHA256SUMS are under
`/orangefs/training/llm-granite-n10/alloc-refusal-{29735,29736}/`.
The fixture exercises the real runner with an injected driver return code.
**NOT ESTABLISHED.** This patched source has no real-GPU measurement in
this receipt. The three-GPU numerical campaign used its original audit pin.
