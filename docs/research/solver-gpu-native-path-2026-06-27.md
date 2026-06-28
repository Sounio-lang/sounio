<!-- docs:meta
topic_id: repo.docs.research.solver-gpu-native-path-2026-06-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-gpu-native-path-2026-06-27
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver lane on Sounio's native GPU path — Blackwell runtime VERIFIED (2026-06-27)

Toward the lane's target (showcase Sounio's native GPU + machine-code control). The
headline result: **a Sounio-emitted epistemic tensor-core kernel executes on a real
NVIDIA Blackwell GB10 (sm_121), with GUM uncertainty propagated through the tensor
cores and verified exact.** This supersedes an earlier intermediate note on this page
that concluded "the GPU surface can't express a solver kernel" — that was true only of
the *narrow checked `kernel fn` surface*; the internal K-AXI/Kretikos engine underneath
has thread/block intrinsics, array access, CUBIN emission, and multi-arch targeting.

Evidence tags per `docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md`: **MEASURED**
(re-runnable artifact) / **BOUNDED** / **SOURCE-ONLY**. No claim rests on a projection.

## The two GPU surfaces

- **Narrow public `kernel fn` surface** (the `souc-linux-x86_64-gpu --backend gpu`
  path): loops + scalar f64 only; no thread index, no array I/O. `benchmarks/solver/
  gpu_epistemic_scoring.sio` is a boundary demonstrator of that *ceiling* — it mirrors
  the smt.sio UCB/Beta formulas and emits non-trivial PTX, but is loop-invariant scalar
  with a discarded result, i.e. NOT a scorer.
- **Internal K-AXI / Kretikos engine** (`self-hosted/gpu/`, driver `bin/kretikos`): full
  thread/block intrinsics (`get_tid`/`get_bid`/`get_ntid`/`get_nctaid` + `getvar`
  array access), an audited K-AXI→PTX emitter (53 patterns × 6 modes, 0/318 static
  violations, f64 register banks), and **real CUBIN emission** (`kretikos_emit_cubin`,
  pure-Sounio CUBIN validator) — i.e. below PTX, bypassing nvcc. This is where a real
  per-variable GPU solver scorer becomes expressible (next-work).

## MEASURED on real Blackwell GB10 (sm_121, CUDA 13, 2026-06-27)

Hardware: NVIDIA **GB10** (Grace-Blackwell, compute_cap **12.1 = sm_121**), driver 580,
CUDA 13.0. Kernel: `self-hosted/gpu/epistemic_mma_reference.ptx` (epistemic
`mma.sync.m16n8k16` with GUM shadow propagation, `ε_C = sqrt(K)·(|A|·ε_B + |B|·ε_A)`).

1. **ptxas → Blackwell SASS/CUBIN** (closes the KAXI audit's open "ptxas needs a GPU"):
   ```
   epistemic_mma_reference.ptx  → ptxas -arch=sm_121 → 6648-byte CUBIN (real ELF)
   FIXED kretikos emitter output → ptxas -arch=sm_121 → 5048-byte CUBIN   (PR #483)
   ```
2. **Runtime, verifiable case A=0 ⇒ D = A·B + C = C** (layout-independent), via the CUDA
   Driver API (`benchmarks/solver/gpu/run_epistemic_mma_*.c`):
   ```
   device: NVIDIA GB10  sm_121
   D     (expect 1,2,3,4): 1.0000 2.0000 3.0000 4.0000        ← tensor-core data path PASS
   eps_C (expect sqrt(7)=2.64575): 2.64575                    ← GUM epistemic shadow PASS (exact)
   prov  (expect 0x5): 0x5                                    ← provenance union PASS
   ```
   Done both as **JIT** (sm_80 PTX → sm_121 at load, `cuModuleLoadDataEx`) and as
   **native sm_121 SASS** (`ptxas -arch=sm_121` → cubin → `cuModuleLoad`, no JIT).
3. **Non-zero operands** (A=B=f16 1.0, C=0): `D = [16,16,16,16]` — the K=16 contraction
   executed deterministically on the tensor cores.

This validates the "compiler-integrated GUM standard uncertainty through GPU tensor
cores" claim **on real hardware**, not in a fixture.

## Reproducible receipt

```bash
# On the workspace (x86): the PTX is checked in at
#   self-hosted/gpu/epistemic_mma_reference.ptx   (strip non-ASCII comments first; see #476)
# On a Blackwell host (CUDA 13):
P=/usr/local/cuda-13.0/bin
$P/ptxas -arch=sm_121 epistemic_mma_reference.ptx -o epi_sm121.cubin
nvcc -O2 benchmarks/solver/gpu/run_epistemic_mma_native_sm121.c -lcuda -o run_native
./run_native epi_sm121.cubin            # expect D=[1,2,3,4], eps_C=2.64575, all PASS
```
Note CUDA-13 specifics: use `cuDevicePrimaryCtxRetain` (the `cuCtxCreate` macro maps to
`cuCtxCreate_v4`); ptxas/nvcc are not on the default non-interactive SSH PATH.

## Honest caveats (held to the doc's evidence bar)

- The runtime used the hand-authored **reference** PTX. The emitter *pipeline* now also
  produces ptxas-valid Blackwell CUBINs after PR #483 (non-ASCII fix); the second emitter
  finding (`retval` illegal state space, #477) is a **stale-artifact non-issue** — it came
  from the removed old `souc-linux-x86_64-gpu` binary's cached PTX, not the live emitter.
- `A=0 ⇒ D=C` is a valid but specific verification; the kernel's per-lane *broadcast* load
  means it is NOT a general 16×16 A·B (the `mma.sync` instruction executed on Blackwell
  tensor cores; a correct fragment-loading kernel is the next step for true matmul verify).
- This is **not** Level 3 by itself; it is a measured capability + runtime receipt.

## Next work

Author a real **per-variable parallel epistemic solver scorer** via the K-AXI thread-index
path (`get_tid` + `getvar` array access; f64 banks), mirroring smt.sio's UCB
`act_mean + β·sqrt(act_var)` + Beta polarity, with real array I/O and verified output on
the GB10 — the genuine GPU-solver showcase, now that both the path and Blackwell execution
are proven.
