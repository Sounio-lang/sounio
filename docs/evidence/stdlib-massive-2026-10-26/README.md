# Stdlib implementation campaign — 2026-10-26

Branch: `devpass/stdlib-massive-20261026`. Base: `5977c066f7d5f0ca04751318e487d8ab670716e0`.

This campaign combines a whole-tree source inventory and executable test selection with checked native storage/platform APIs and previously prepared scientific-library work. The inventory describes source presence and import-witness candidates. It does not certify every API or every module at runtime.

## Implemented foundations

- Python-free source inventory: 1,767 active Sounio files at the first integrated inventory, 179 active module entrypoints. Every source record has a raw-byte SHA-256, public-declaration index, and explicitly labelled import-witness candidates.
- Independent E2E producers with raw diagnostic capture, source/log identity checks, exact selection reconciliation and retained failures. A failed hyper producer no longer prevents the E2E producer from running.
- A strict C++23 hyper-metric oracle. It rejects missing, duplicate, unknown, malformed and nonfinite measurements. The existing quaternion difference and squared-norm outputs are now also in the pinned golden, with an exact C++ reference.
- Owned ByteBuffer, DynamicDeque and DynamicMap with explicit payload budgets and separate status/value results. These exceed the old 4,096-element limits without changing the legacy fixed-container APIs.
- Checked binary integer reading and an overflow-safe pattern bound check.
- A small Linux x86-64 C++23 platform ABI providing seq-cst i64 atomics, error-checking pthread mutexes, i64 function-pointer threads, monotonic time, OS entropy, IPv4 TCP and UDP.
- The callback adapter preserves SysV callee-saved registers across Sounio callbacks. The same original callback failed with SIGSEGV in job 17809 and passed after the adapter in job 17812.
- Canonical sync::atomic and sync::mutex use native operations; time::instant acquires a real monotonic clock. The profiler reports clock, state, capacity and overflow errors rather than silently ignoring them.
- Six intended math::pure exports and the regex u8 branch are repaired. The regex test imports production code instead of a duplicate implementation.

- The disabled VCF module now validates core records and GT over dynamic byte buffers. [Its explicit contract](../../../stdlib/genomics/io/VCF_CORE.md) leaves header/INFO/schema and symbolic-allele validation separate. Independent C++23 and native Sounio fixtures pass in job 17874.

## Scientific consolidation

The selected stdlib, C++ reference, fixture and gate changes come from PR #3200 at `f3aabdb50431009de8be5b7f408db4ab449815f4`. This includes checked distributions, power, Wilcoxon, survival, FFT, numerical differentiation, FEM/Poisson, BDF/DP54, physical covariance/phonons, genomics counts/CSR/batch/shrinkage, and visualization palettes/contours. Original licensed palette source data and historical receipts retain their provenance.

NN ownership/rate dimensions come from PR #3205 at `f0b7026d2d`. The Romberg budget/error-estimate repair comes from PR #3201 at `3f39a3def8`. Their original fixtures and numerical tolerances are retained. The original negative variance/phonon/DP54 witnesses required by their gates are included.

Historical receipt archives imported with those changes are historical evidence. Only jobs explicitly recorded below are executions of this integration campaign.

## Native API migration

Open one Platform with `ffi::platform::platform_open()`, check `platform_status`, close all resources, then consume it with `platform_close`. The loader resolves `libstdlib_platform.so` through the dynamic loader; install a trusted library and set its search path when running native executables.

- Atomics now take `&Platform`; construction returns a linear AtomicI64, loads/add/sub/CAS return PlatformValue, and store/close return errno. CAS returns value 1 for success and 0 for an ordinary mismatch; status distinguishes errors. The old non-atomic implementation is removed.
- Mutex operations take `&Platform`. `mutex_destroy(&platform,&!mutex)` preserves ownership on EBUSY; after successful destruction, `mutex_close(mutex)` consumes the invalidated owner. Recursive locking returns EDEADLK; try-lock returns EBUSY.
- Thread callbacks currently support plain `fn(i64)->i64`. Join once, then close. This is not a closure, floating-point or general foreign-function callback ABI.
- TCP operations may transfer fewer bytes than requested and distinguish EOF from errors. UDP loopback send is explicitly named. Protocol mismatches are rejected before I/O. Truncated receive consumes the datagram, returns EMSGSIZE and the copied count, and leaves the ByteBuffer logical length unchanged.
- Local ByteBuffer operations use their documented storage status codes. PlatformValue consistently uses Linux errno, including ENOBUFS for a payload budget and ENOMEM for allocation failure.
- Collection budgets bound retained payload, not allocator metadata or the transient old-plus-new allocation during a deque/map reallocation.
- `instant_now(&platform)` returns an opaque status-bearing instant. Elapsed operations reject reversed instants.
- Profiler begin/end now take `&Platform` and return errno. Repeated begin is EALREADY, ending an inactive slot is EINVAL, unknown slot is ENOENT, and the explicit 64-slot limit returns ENOSPC.
- `crypto::random` remains a deterministic, non-cryptographic PRNG. `crypto::system_random` uses OS entropy with no PRNG fallback.

Updated consumers include the atomic example and the stdlib sync/profile tests. Legacy fixed containers and the old dynamic-slice/SNN implementation remain separate unresolved work.

## Reproduction

Run compiler builds and native gates on an allocated Slurm Linux x86-64 CPU worker. Node handles operational metadata; all new numerical reference comparisons use C++23.

```bash
export MADAROS_RAW_BIN=/absolute/path/to/source-built/madaros
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_REQUIRE_COMMITTED_MADAROS=0
export STDLIB_MASSIVE_OUT=/absolute/fresh/output/native
bash scripts/stdlib/massive_sprint_gate.sh

export STDLIB_DEEP_OUT=/absolute/fresh/output/science
export STDLIB_DEEP_JOBS=1
bash scripts/stdlib/deep_science_gate.sh

STDLIB_FULL_CAPTURE=1 node scripts/stdlib/run_full_suite.mjs \
  --out-dir /absolute/fresh/output/full --jobs 2
```

The source compiler used here was built in Slurm job 17771 from the unchanged compiler sources at the base checkout. Its raw ELF SHA-256 is `4e3dc94373f03cbfb06cf9b34c1106a17a060e94d2b3fc8021bdadd68b7e2464`. Compiler implementation, canonical seed and CI workflow changes are outside this patch.

## Evidence and remaining gates

- Job 17874: expanded focused gate COMPLETED, exit 0:0, 44 seconds, including VCF and TCP/UDP protocol mismatch checks. See [the receipt](focal-17874.json).
- Job 17864: focused gate COMPLETED, exit 0:0, 41 seconds. See [the machine-readable receipt](focal-17864.json). It contains the exact overlay/source-catalog hashes, per-phase results and artifact hashes. Durable native logs and manifests live under the directory recorded there.
- Job 17800: earlier complete stdlib selection, before scientific consolidation and subsequent repairs: 320 pass, 74 fail, 194 skip, 588 total; zero selection errors. This is a diagnostic baseline, not the final tip.
- Job 17854: integrated full/scientific/hyper validation of snapshot 14 was still running when this document was initially written. Snapshot 15 adds the subsequently passing effects, time/profiler and quaternion-golden changes. Follow-up receipts must retain that distinction.
- Snapshot 14 exposed the legacy SNN native-emission failure (rc=12); its original fixture is preserved. Do not replace it with a smaller passing control or label the hyper gate green.
- Full slow lane, whole-corpus compiler validation, lifetime/fixed-point/transport gates, Lean validation and exact-tip GitHub CI have not been closed by the focused pass.

No merge, auto-merge, production activation or image publication is part of this campaign. Review and merge remain human decisions.
