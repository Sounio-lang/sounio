<!-- docs:meta
topic_id: repo.docs.design.epistemic-tensor-core-gum-turing
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.design.epistemic-tensor-core-gum-turing
-->

# GUM-correct uncertainty propagation through GPU tensor cores (Turing sm_75)

Status: **design + derivation (verifiable math) + reference PTX pending on-hardware
`ptxas -arch=sm_75` validation on the RTX 8000.** Companion:
`docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md`.

This document fixes two things at once:
1. The existing `self-hosted/gpu/epistemic_mma_reference.ptx` uses a **non-derivable**
   uncertainty formula AND an **Ampere-only** instruction — so it neither propagates
   GUM correctly nor assembles on Turing.
2. It states the **derivable** propagation law and the **Turing-reachable** codegen
   shapes, which is what "our emits must reach sm_75" requires.

---

## 1. Why the current reference is wrong (do not port it)

`epistemic_mma_reference.ptx` line 13 / 85–98:

```
SHADOW PATH: ε_C = sqrt(K) * (|A_norm| * ε_B + |B_norm| * ε_A)
```
and the code actually computes `sqrt(4·combined)` (a doubling chain then one `sqrt`),
which equals neither `sqrt(K)·combined` nor any GUM quantity. Two independent defects:

- **Not derivable (violates CLAUDE.md §6).** `sqrt(K)·(|A|ε_B+|B|ε_A)` is a hand-tuned
  heuristic. It is not the GUM law of propagation for a matrix product, and the code
  and comment disagree on the constant.
- **Information-destroying.** A tensor-core `mma` collapses the K-reduction
  `Σ_k A[i,k]·B[k,j]` into the accumulator. After the instruction the per-element
  products `A[i,k]·B[k,j]` are **gone**, so no post-hoc scalar formula on the *output*
  norms can recover the correct per-term uncertainty. The propagation must happen
  **in the same contraction**, not afterwards.

---

> **Novelty scope (read before citing).** The propagation law in §2 (variance =
> squared-operand matmul) is **not itself new** — it is the textbook first-order delta
> method / assumed-density filtering, already used through linear layers in probabilistic
> NNs and Kalman-style filters. Do **not** frame the law as the contribution. The
> defensible novelty (per `SOTA_RESEARCH_2026-05-31.md`) is the **synthesis**:
> compiler-*emitted* GUM/JCGM-100 uncertainty as a first-class *type*, propagated on GPU
> *tensor cores*, with a *confidence gate* at type-check time, in a *self-hosted* compiler.
> §2 is the correct mechanism; the novelty is that the compiler emits and gates it.

## 2. The derivable law — GUM first-order through a matmul

For `D[i,j] = Σ_k A[i,k]·B[k,j] (+ C[i,j])` with **uncorrelated** inputs, the GUM
(JCGM 100:2008 §5.1.2) first-order law of propagation of uncertainty gives, with
sensitivity coefficients `∂D/∂A[i,k] = B[k,j]` and `∂D/∂B[k,j] = A[i,k]`:

```
u²(D[i,j]) = Σ_k [ B[k,j]² · u²(A[i,k]) + A[i,k]² · u²(B[k,j]) ] + u²(C[i,j])
```

The crucial observation: **that sum over k is itself two matrix products** in
*variance space*. Writing `VA = u²(A)`, `VB = u²(B)` (elementwise variances) and
`⊙` for elementwise (Hadamard) square:

```
U2 = VA · (B ⊙ B)  +  (A ⊙ A) · VB  +  VC          (exact first-order GUM)
u(D) = sqrt(U2)                                      (standard uncertainty of D)
```

Both terms are *the identical contraction* as the value matmul — so they run on the
**same tensor cores**. The 4-shadow `Knowledge<T>` channel carries `U2` (variance),
not a heuristic ε; validity/provenance merge as before (`and`/`or`).

### Cost model (honest)
- Value matmul: 1× `mma`.
- Uncertainty: 2 extra `mma` (the two variance terms) + 2 elementwise squares (cheap,
  fuseable into the load) + 1 `sqrt` at store.
- ⇒ ≈ **3× the tensor-core work** (the `mma` ops *only* — the elementwise squares and
  the final `sqrt` are separate ALU cycles, not tensor-core, so total kernel cost is
  somewhat higher than 3×) for *full, exact first-order GUM bounds*. This is the
  honest source of the measured 2.71× value-only / 7.94× full-shadow regime — and it is
  the price of a guarantee cuBLAS cannot provide at any cost. (Correlated inputs need
  the covariance cross-term `+2·Σ B[k,j]A[i,k]·cov(...)`; v1 assumes uncorrelated and
  must say so — that is itself a GUM-admissible, stated modelling choice.)

### Why this is not interval arithmetic (pre-empts reviewer risk #1)
Interval/abstract-interpretation bounds propagate `[lo,hi]` and blow up super-linearly.
This propagates **variance** by the exact first-order GUM law with sensitivity
coefficients — the same object a metrologist computes by hand — and the confidence
*gate* is a type-checker admit/block on `k·u(D)` (coverage factor), not a runtime check.

---

## 3. Turing (sm_75) reachability — the codegen constraints

RTX 8000 = Quadro Turing TU102 = **sm_75**. `ptxas -arch=sm_75` **rejects** (does not
JIT-downgrade) the following, all currently emitted somewhere in `self-hosted/gpu/`:

| Currently emitted | Where | Why it fails on sm_75 | Turing-correct form |
|---|---|---|---|
| `mma.sync.aligned.m16n8k16…` | `lower_to_ptx.sio:934`, `kernel_ir.sio:4486` | `m16n8k16` (k=16) is **Ampere sm_80+** | `mma.sync.aligned.m16n8k8` (k=8) — A:2×b32, B:1×b32, C/D:4×f32 |
| `wmma…m16n16k16.f32.f32` | `ptx_emitter.sio:136` | f32/**TF32 inputs** are Ampere+ | `wmma…m16n16k16.f32.f16.f16.f32` (f16 in, f32 accum) — sm_70+ ✓ |
| `.target sm_80` (hardcoded) | `lower_to_ptx.sio:31` `gpu_target_profile_cuda_sm80()` | locks output to Ampere | thread target arch; `.target sm_75` for Turing |

**Lowest-risk Turing path = WMMA `m16n16k16` with f16 inputs / f32 accumulate**
(`sm_70+`, so valid on Turing). Prefer it over `mma.sync.m16n8k8` fragment surgery for
v1. Apply §2's variance-space algorithm: 3 wmma ops (value + 2 variance) sharing the
same fragment loads, squared operands materialised at load.

---

## 4. Implementation order (reference-first, per the verification reality)

This session cannot reach the RTX 8000 (no host IP/creds, no local `ptxas`). So the
**user is the verification loop** and the order is:

1. **(this doc + reference harness)** State the derivable law; provide a
   fragment-correct-by-construction validation harness
   `scripts/gpu/epistemic_wmma_sm75_reference.cu` (CUDA C++ `nvcuda::wmma`, so nvcc
   encodes the sm_75 fragment layout — no hand-rolled register vectors to get wrong).
   It implements the §2 variance-space algorithm (3 wmma: value + 2 variance) and
   checks GPU output against a CPU GUM oracle.
2. **(user, on the RTX 8000)**
   ```
   nvcc -arch=sm_75 -o /tmp/epi_wmma scripts/gpu/epistemic_wmma_sm75_reference.cu
   /tmp/epi_wmma         # expect: sm_75 detected, PASS, small f16 rel error
   ```
   A clean compile alone proves Turing-legality; PASS proves the GUM math on silicon.
3. **(then, not before)** Make the *Sounio emitter* emit equivalent PTX: thread target
   arch into `gpu_lower_op`, branch the tensor-core shape by arch (WMMA m16n16k16
   f16→f32 for sm_75), and replace the heuristic shadow with the variance-space
   double-matmul. Gate with a golden diff + the same oracle.

Note on language policy (CLAUDE.md §4): the science path stays in Sounio. This `.cu` is
a *hardware-validation harness* for the codegen target, consistent with the existing C
runners in `scripts/gpu/` — not science, and explicitly a stepping-stone to Sounio-emitted PTX.

Emitting toward an unvalidated target is building on sand; do not reorder.

---

## 5. One-line thesis

Uncertainty through a tensor core is not a scalar fix-up on the output — it is **the same
contraction run in variance space**. Get that law emitted, Turing-legal, and
hardware-checked, and Sounio has a GPU epistemic datapath that is *derivable* (not
retrofitted) and *runnable on real Turing silicon* — the defensible novelty, not a
GFLOPS claim.
