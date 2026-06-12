<!-- docs:meta
topic_id: repo.docs.audit.gpu-pipeline-sota-assessment-2026-05-30
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gpu-pipeline-sota-assessment-2026-05-30
-->

# GPU pipeline — ground-truth SOTA assessment (2026-05-30)

Goal of this pass: *investigate the entire GPU pipeline in search of surpassing the SOTA.*
This document is the auditable synthesis. Every claim below is tagged with its
evidence status. **No claim rests on a projection.** Where a number is projected it
says so, and projected numbers are never used to assert a SOTA result.

Method: full-repo map (Explore fan-out) + empirical spot-checks run this session.
Calibration discipline per `CLAUDE.md` §6 (measure before claiming; auditability over speed).

---

## 0. Evidence tags

| Tag | Meaning |
|---|---|
| **MEASURED** | A real artifact / reproduced command backs it. Re-runnable. |
| **PROJECTED** | A planning estimate. Never measured. Carries zero claim weight. |
| **SOURCE-ONLY** | `.sio` exists and type-checks, but is not wired into any *working binary* and has no runtime proof. (`CLAUDE.md` §6.2/§6.3: a stub/model is not a gap, and is not a working feature either.) |
| **STUB / BOUNDED** | Intentionally narrow surface; works only inside a named fixture set. |

---

## 1. What is MEASURED (re-runnable, real)

### 1.1 End-to-end CLI path: `kernel fn` → PTX — **MEASURED, reproduced this session**
- Command (reproduced 2026-05-30):
  ```
  ./artifacts/omega/souc-bin/souc-linux-x86_64-gpu build \
      examples/kernel_vec_add.sio --backend gpu -o /tmp/vec_add_gpubin.ptx
  ```
  → `Wrote PTX to /tmp/vec_add_gpubin.ptx` (61 lines, 2 valid `.visible .entry`,
  `.version 6.4 / .target sm_75`). The emitted `vec_add` body is a structurally-valid
  but **trivial shell** — `ld.param.u64 r64_0, [param_0]; ret;` — because the example
  source kernel body is literally `{ }`. (Note: the richer `mad.lo.u32 / %tid.x`
  template tokens visible in a raw `cat bin/souc` dump are the compiler's *internal
  PTX template strings*, NOT this emission — do not conflate the two.) The point stands:
  the e2e path emits valid PTX skeletons, so "no e2e path from CLI" is refuted.
- **This refutes `CLAUDE.md` §13 "GPU: PTX codegen exists but no end-to-end path
  from CLI."** The path exists and is reproducible. (Doc line corrected in the same
  pass that produced this file.)
- Caveat (binary provenance — see §3): the working GPU CLI is the **dedicated
  GPU-profile binary** `souc-linux-x86_64-gpu` (beta.4, dated **2026-03-08**). The
  general live `bin/souc` (2026-05-29) does **not** emit PTX through
  `build --backend gpu` (it returns lexer/IR debug, no `.ptx`). So the end-to-end
  path is real but lives in a *separate, older* binary, not the current general one.

### 1.2 Baseline epistemic GEMM on real L4 — **MEASURED (2026-02-26)**
- `artifacts/omega/l4_perf_pass_report.v2.txt`: `epistemic_gemm 4096³`,
  **gflops=5270.7**, ms_per_iter=26.08, gpu=NVIDIA L4, status=pass, real dispatch
  via `scripts/cuda_gemm_dispatch.py` on host `10.100.100.215`.
- vs `OPTIMIZATION_REPORT.md` cuBLAS baseline 14,280.2 GFLOPS ⇒ **≈ 2.71× overhead**
  for the *value-only baseline PTX*. This 2.71× is the one honest perf number in the
  whole stack.

### 1.3 Multi-target codegen surface exists — **MEASURED (source compiles + golden refs)**
- PTX, SPIR-V, Metal, K-AXI emitters all present and exercised by
  `tests/golden/kaxi_ptx/` (≈360 golden PTX references across 6 configs) and the
  `kaxi_ptx_golden_gate.sh` regression gate.
- K-AXI runtime-backed source profiles: **13** (vec_add/sub/mul/div × f32/f64, fma
  × f32/f64, epistemic_elementwise_f32, epistemic_dual_output_f32, store_u32_const) —
  these are the only patterns with an *owned CUBIN rung* proven on L4.

---

## 2. What is PROJECTED ONLY (never measured — zero claim weight)

L4 became **unreachable on/around 2026-03-03**: `l4_native_shadow_gate.v1.json`
shows `status:not_run, reason:ssh_unreachable`, and the host `10.100.100.215` does
not respond to ping as of 2026-05-30. Therefore everything below the 2.71× baseline
in `OPTIMIZATION_REPORT.md` was **never measured**:

| Optimization | Projected gain | Status |
|---|---|---|
| FP16 WMMA (f16 in / f32 accum) | 1.8–2.0× | **PROJECTED** |
| `cp.async` double-buffer pipeline | 1.2–1.3× | **PROJECTED** |
| Software pipeline (provenance/WMMA overlap) | 1.1–1.15× | **PROJECTED** |
| Auto-tune tile config | 1.05× | **PROJECTED** |
| **Combined "<1.1× overhead / beats cuBLAS"** | — | **PROJECTED — DO NOT CLAIM** |

`OPTIMIZATION_REPORT.md` itself states the correct policy: *"projected gains are
planning inputs only until the L4 JSON report records measured GFLOPS."* That report
(`l4_optimized_no_rust_report.latest.v2.json`) does **not** exist in `artifacts/`.

---

## 3. What is SOURCE-ONLY (compiles, not in any working binary)

The current `self-hosted/gpu/` tree is ≈82k LOC; `stdlib/gpu/` adds tens of k more.
The only *proven-working* GPU CLI binary is **2026-03-08**. So all gpu-source work
committed after that date is SOURCE-ONLY until rebuilt into a working binary and
runtime-validated:

- `self-hosted/gpu/epistemic_tensor_core_optimized.sio` — the FP16/async model.
  Type-checks (its embedded analysis gate prints `gate_pass=1396`), but is a
  *check-safe model*, not a kernel wired into the working GPU binary. (`run` reports
  `no main` — expected for a library/model file.)
- The advanced PTX / tensor-core / `cp.async` lowering in `ptx_advanced.sio`,
  `opt/async_pipeline.sio`, `epistemic_tensor_core_optimized.sio` — present in source,
  **not proven to round-trip through a working binary to runnable PTX.**

This is the single biggest gap between the *map* and the *reality*: the impressive
LOC count is current source; the impressive PTX demo is an older binary; the two have
not been shown to be the same artifact.

### 3.1 The capability gap, measured at the instruction level (this session)

The SOTA claim (§4) is "GUM uncertainty through GPU tensor cores." Instruction census
of the two relevant PTX files pins down exactly how far the *compiler* carries it:

| PTX artifact | `mma.sync` | shadow ε (`%fe`) | `sqrt` (GUM) | `and.pred` (validity) | `or.b64` (provenance) |
|---|---:|---:|---:|---:|---:|
| `self-hosted/gpu/epistemic_mma_reference.ptx` (**hand-written**) | 5 | 12 | 8 | 2 | 2 |
| `examples/kernel_epistemic_wmma_matmul.ptx` (**compiler-emitted**, `epi_wmma_mm16`) | **0** | **0** | 1 | 2 | 2 |

Reading: the compiler **does** emit the epistemic 4-tuple *memory layout* — value@+0,
epsilon@+8, validity@+12 (via `and.pred`), provenance@+16 (via `or.b64`) stored to
global memory — so the `Knowledge<T>` *data model* is real in emission. But it emits
**no tensor-core `mma.sync` and no shadow-ε propagation registers**. The GUM shadow
path `ε_C = sqrt(K)·(|A|·ε_B + |B|·ε_A)` through `mma.sync` lives **only in the
hand-authored reference**, not in compiler output.

Two further accuracy notes:
- `epistemic_mma_reference.ptx`'s own header comment calls it *"World-first:
  compiler-integrated GUM … shadow registers."* Per the census above that is
  **aspirational** — the file is hand-written and its mma.sync/shadow path is NOT
  compiler-emitted. Do not cite the reference as evidence of compiler integration.
- The emitted 4-tuple stores **placeholder values**: validity is constant-true
  (`setp.eq.u32 p0, 1, 1`) and provenance is zero (`mov.u64 r64_3, 0`). The *layout*
  is real; the *stored values* are not derived from real input combinations yet.

**Honest capability statement:** the uncertainty *type and its 4-field GPU layout* are
compiler-emitted (MEASURED, with placeholder values); the *uncertainty propagation
through tensor-core matmul* is a hand-written reference (SOURCE-ONLY / not-yet-emitted). The SOTA framing in §4 is
the right target, and the data-model half is real — but a paper must not state the
compiler emits GUM-propagating tensor-core kernels until `mma.sync` + `%fe` appear in
*emitted* PTX. **This is the single highest-leverage codegen gap to close** (and it is
locally verifiable, no L4 needed — see §5.3).

---

## 4. The honest SOTA position (the actual answer to the goal)

**Sounio does not, and should not claim to, beat the GFLOPS SOTA.** That belongs to
cuBLAS / CUTLASS / Triton / TensorRT. Sounio's baseline is 2.71× *slower* than cuBLAS
on plain GEMM, and the path to closing that is projected, not measured.

**The defensible, unclaimed frontier is a capability, not a speed:**

> GUM/JCGM-100-certified measurement-uncertainty propagation, carried as a
> *compile-time type* (`Knowledge<T>` via a 4-shadow-register encoding), **through GPU
> tensor-core kernels** — with a confidence gate that admits/blocks at type-check time.

No surveyed GPU compiler (Triton, CUTLASS, TensorRT, cuBLAS) produces certified
uncertainty bounds **at any price** — they have no uncertainty dimension at all. Two
prior adversarial deep-research runs (`SOTA_RESEARCH_2026-05-31.md`: `wf_17f1dbf1`,
`wf_366fc261`, 3-vote verification) found the *combination* — metrological-uncertainty
-as-a-compile-time-type in a self-hosted compiler — genuinely unclaimed (high
confidence on per-source exclusions; medium-high on the global absence claim).

So the GPU thread's SOTA contribution is the **extension of that novelty onto the GPU
tensor-core datapath**: the 2.71× baseline is not "2.71× too slow", it is *the price
of full GUM uncertainty bounds that the SOTA cannot produce at all.* That is the honest
framing — capability, status-tagged, never a benchmark claim.

### Residual reviewer risks (pre-empt, do not bury)
1. "Compile-time uncertainty = interval analysis / abstract interpretation." Must
   differentiate: GUM first-order law-of-propagation + coverage factor + the
   confidence GATE as a type-checker admit/block, vs interval AI.
2. The shadowed (full-provenance) PTX path was **7.94× cuBLAS** measured — the
   uncertainty isn't free; the value-only 2.71× does not carry the shadow registers.
   Any capability claim must state which variant produced the bounds.
3. Arbitrary-source GPU lowering is **not** claimed — only 13 runtime-backed profiles.

---

## 5. Recommended next advances (ranked by leverage × honesty)

1. **(blocked) Measure the optimized path.** The only way to convert §2 PROJECTED →
   MEASURED is L4 access. It is down. Until reachable, no perf SOTA statement is admissible.
2. **Rebuild + runtime-validate current gpu source into one working binary**, closing
   the §3 source-vs-binary gap. Highest-value engineering move that does *not* need L4
   for the codegen half (PTX text round-trip is checkable locally; only runtime needs L4).
3. **Close the §3.1 emission gap (no L4 needed, locally verifiable).** Today the
   compiler emits the epistemic 4-tuple *layout* but not the tensor-core shadow path.
   Make the WMMA lowering emit `mma.sync` + the `%fe` shadow-ε propagation that the
   hand-written `epistemic_mma_reference.ptx` already specifies, then gate it: a local
   command that diffs *emitted* `epi_wmma_mm16` PTX against the reference's instruction
   census (`mma.sync ≥ 1`, `%fe ≥ 1`). This converts the §4 capability from
   "reference-only" to "compiler-emitted" — the concrete proof, independent of perf.
4. **Lean4 soundness of the GPU epistemic lowering** (shadow-register propagation
   preserves GUM semantics) — moves the capability from "implemented" to "certified",
   which is the real moat vs any future copycat.

---

## 5b. Turing (sm_75 / RTX 8000) reachability — NEW FINDINGS (2026-05-30, session 2)

The user has an **RTX 8000 (Quadro Turing TU102 = sm_75)**.

**First, correct an emission-vs-source confusion (same class of error as §1.1's
`mad.lo.u32`).** The *working* GPU binary is Mar-8, built from *older* source. Direct
evidence from §1.1 + §3.1: the emitted kernels are already `.target sm_75` and contain
**`mma.sync: 0`** — i.e. the working binary emits **no tensor-core instructions at all**
for the tested kernels, and already targets Turing. So the items below are **SOURCE-ONLY
in current `self-hosted/gpu/`** — present in emitter source, *not* proven to be the
reached emission path, binary-correspondence unknown. They are the shapes the *current
source would* emit if/when a kernel lowers to a tensor-core op — which none of the
tested kernels do.

| Item (SOURCE-ONLY; not confirmed emitted) | Location | Turing concern if it WERE reached |
|---|---|---|
| `mma.sync.aligned.m16n8k16` (no arch guard) | `lower_to_ptx.sio:934`, `kernel_ir.sio:4486` | **Ampere-only (k=16)**; `ptxas -arch=sm_75` would reject |
| `wmma…m16n16k16.f32.f32` (TF32 inputs) | `ptx_emitter.sio:136` | **TF32 is Ampere+**; not Turing |
| Target profile `gpu_target_profile_cuda_sm80()` in this source file | `lower_to_ptx.sio:31` | sm_80 default in *this path* (NB: tested emission was sm_75 — different/older path) |
| Shadow `ε_C = sqrt(K)·(|A|ε_B+|B|ε_A)` (code computes `sqrt(4·comb)`) | `epistemic_mma_reference.ptx:85-98` (hand-written) | **Not GUM, not derivable** (CLAUDE.md §6) — tensor-core K-reduction destroys the per-element products the formula needs |

**The actual measured gap is larger and simpler than "wrong shape":** the compiler emits
**zero tensor-core ops** for the tested kernels (toy `while`/`{}` bodies never lower to
`GpuWmma`). Before any emitter edit, the real next step is to find/write a kernel that
*does* lower to a tensor-core op, then observe what the binary actually emits.

**Corrected, derivable law (the real novelty):** GUM uncertainty through a matmul is
**the same contraction in variance space** —
`U2 = VA·(B⊙B) + (A⊙A)·VB`, `u(D)=sqrt(U2)` — i.e. 2 extra matmuls, runnable on the
**Turing-native** `wmma m16n16k16 f16→f32` (sm_70+). Full derivation +
codegen-target table: `docs/design/EPISTEMIC_TENSOR_CORE_GUM_TURING.md`.

**Runnable artifact (validates novelty on real silicon, no L4 needed):**
`scripts/gpu/epistemic_wmma_sm75_reference.cu` — `nvcuda::wmma`, fragment-correct by
construction, GPU-vs-CPU-GUM-oracle check. `nvcc -arch=sm_75` + run on the RTX 8000.

**Order (per verification reality — this session cannot reach the RTX 8000):**
validate the `.cu` on hardware FIRST, then wire the emitter to emit equivalent PTX. The
emitter edit is deliberately **deferred** (unverifiable from here; type-check is not
enough for register layouts + GUM constants).

## 6. One-line verdict

The GPU pipeline is large, real, and partly stale-documented; its baseline perf is
honestly behind cuBLAS; its genuine SOTA is the **uncertainty-typed GPU datapath**, a
capability no competitor has — and the correct posture is to ship that capability with
measured/projected tags intact, not to chase a GFLOPS crown it will not win.

---

*Companion to `docs/audit/SOTA_RESEARCH_2026-05-31.md` (compiler/bootstrap thread).
Empirical checks in this file are re-runnable from repo root with the GPU-profile
binary and the `artifacts/omega/l4_*` reports.*
