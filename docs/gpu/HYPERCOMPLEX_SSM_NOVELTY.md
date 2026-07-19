<!-- docs:meta
topic_id: repo.docs.gpu.hypercomplex-ssm-novelty
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.hypercomplex-ssm-novelty
-->

# Novelty & related work — non-associative hypercomplex SSM on tensor cores

Spine for a systems/PL preprint. Grounded in an 11-agent literature deep-research swarm
(9 survey angles + synthesis + adversarial critique, 131 web searches, 2026-07-19) and an on-hardware
SASS verification. **Framing rule: this is a systems/PL artifact first, with a proof-of-capability ML
result attached — NOT an ML-SOTA claim.**

## 1. The contribution (the defensible triad)
No single piece is the contribution; the *combination* is. Across eight literature angles, **no work
compiles a non-associative octonion/sedenion product or the associator to GPU tensor cores, none treats
the associator as a first-class trainable operation, and none formulates a non-associative structured
SSM with BPTT / exact-f64 gradients.**

1. **Compiler / PL:** a self-hosted language that makes the non-associative Cayley-Dickson product
   (octonion dim-8, sedenion dim-16) **and the associator** first-class, differentiable operations,
   lowering the batched left-multiplication `L(a)·H`, the associator `[a,b,c]`, and their reverse-mode
   VJP to real Blackwell tensor-core tiles — f16 `m16n16k16` and **exact-f64 `m8n8k4`**.
2. **The exact associator `[a,b,c] = (a·b)·c − a·(b·c)` as a first-class operation** of a non-associative
   division/zero-divisor algebra. It is *zero-by-construction* in every Clifford/geometric-algebra and
   quaternion construction, so it cannot even exist as a feature there.
3. **A non-associative (octonion/sedenion) structured SSM** `h_t = σ(A⊗h_{t-1} + B·x_t)`, readout
   `y = Re(C⊗h)`, with `L(a)·H` as a single tensor-core tile and reverse-mode BPTT. The
   octonion/sedenion × SSM intersection is **empty** in the literature.

Why the SSM angle is genuinely different, not a re-skin: the efficiency machinery of modern SSMs
(parallel/associative scans in S5, Mamba/S6; linear-attention reassociation) **depends on associativity
of state-transition composition** — exactly the property octonions (non-associative) and sedenions
(zero divisors) lack. So the tensor-core `L(a)` tiling is a *necessary and distinct* mechanism, and the
apparent weakness (no associative scan) is the motivation.

## 2. Explicit non-claims (primacy already taken — must be disclaimed up front)
- **NOT** "the first octonion/sedenion sequence model." Octonion RNN / Bidirectional Octonion LSTM
  (~2021) and **Numerion** (2025, up to sedenion) precede us. Position them as (a) suppressing rather
  than exploiting non-associativity, and (b) running on generic real-valued emulation — which Numerion
  itself reports (App. F) as a bottleneck, corroborating our systems gap.
- **NOT** "the first to run a hypercomplex product on tensor cores by splitting into real tiles." That
  recipe is incremental against complex-GEMM-on-tensor-cores (Abdelfattah/Dongarra 2020) and CUTLASS
  quaternion GEMM (which, notably, targets CUDA cores — not tensor cores). Our distinctiveness is the
  *non-associative* regime + native exact-f64 + the associator + trainable VJP/BPTT, not the tiling trick.
- **NOT** "training a hypercomplex model is new" — deep octonion/sedenion nets exist via real embeddings.
  Our end-to-end-training novelty is only as (SSM + associator-feature + exact tensor-core codegen).

## 3. Closest neighbors and what each does NOT do
| Work | Closest because | What it does not do |
|---|---|---|
| Octonion LSTM / Octonion RNN (~2021) | octonion, non-associative, recurrent | classic gates, not an SSM; associator never measured; no tensor-core lowering; no VJP-on-TC; no sedenion |
| Numerion (2025) | sedenion sequence model, faithful CD product | MLP not a recurrence; no state operator/BPTT; simulated in generic PyTorch (no native datatype — its own complaint); no associator |
| Clifford Neural Layers / Clifford Group Equivariant / GATr | nearest algebra-structured DL | every Clifford algebra is **associative**, matrix-representable, no zero divisors → associator ≡ 0; octonions/sedenions are provably not Clifford. (GATr's 16-dim is projective GA of 3D — coincidental to sedenion dim-16; disambiguate explicitly.) |
| S4 / S4D / S5 / Mamba | the recurrence backbone; Mamba's HW-aware SSM kernel | associative, ≤ complex scalars; associator trivially zero; scan efficiency depends on associativity (does not transfer); Mamba targets SRAM/scan on CUDA cores, not tensor-core wmma |
| PHM / PHNN (2021-22) | "any nD hypercomplex" learned multiplication | learns a *soft* real matrix — not an exact division-algebra product; no guaranteed non-associativity, no zero divisors, no isolated associator; a linear-layer reparameterization |
| complex-GEMM-on-TC (Abdelfattah 2020) / CUTLASS quaternion | hypercomplex-as-GPU-matmul | associative (dim ≤ 4); CUTLASS quaternion uses **CUDA cores**; no non-associativity, no associator, no autodiff, no SSM, no compiler lowering of a multiplication table |
| HARDBOILED (CGO 2026) / ACT (2025) | compiler "beyond matmul" to tensor cores | image/stencil pipelines, conventional associative operator semantics; no new value type, no multiplication table, no exact-f64, no autodiff |

## 4. Hardware validity (the load-bearing claim — verified)
The adversarial critique's single biggest threat: GB10 is *consumer* Blackwell (compute capability 12.1,
~1:64 FP64:FP32), and native FP64 tensor cores (DMMA / the `m8n8k4` double path) are widely described as
a datacenter-only feature — so our f64 `wmma.mma...m8n8k4.f64` might silently run on CUDA-core FP64.

**Resolved on hardware:** `cuobjdump --dump-sass` on the compiled cubin shows the f64 tile lowers to
**4× `DMMA`** (double-precision matrix multiply-accumulate — the FP64 **tensor-core** instruction), and
the f16 tile to `HMMA`. So the GB10 does execute native DMMA and our f64 path is genuinely tensor-core,
producing exact results.

**Honest caveats to state in the paper:**
- Throughput on this consumer part is modest (few FP64-TC units); do **not** claim a speed win without a
  fair comparison vs real-valued FP64 emulation, ideally on a datacenter part (A100/H100/B200 have full
  DMMA) — that is where the performance ceiling and the "systems win" would be substantiated.
- We target the `wmma`/DMMA tensor-core path; Blackwell's newest ISA is `tcgen05.mma` (a higher-throughput
  surface we do not yet emit). State the lowering target precisely to avoid an easy rebuttal.
- Characterize the f16 path's numerics for a *non-associative* product: reassociation to fit tiles
  changes results, and f16 error through the Cayley-Dickson recursion needs reporting.

## 5. Reviewer traps to pre-empt (one line each)
1. "Why not just use Clifford/geometric algebra?" — Clifford algebras are associative and
   matrix-representable; octonions/sedenions are provably not; the associator is identically zero in GA.
2. "Doesn't PHM already do arbitrary hypercomplex?" — PHM learns a soft real matrix, not an exact
   non-associative division-algebra product (no zero divisors, no guaranteed non-associativity).
3. "Isn't the associator just associativity regularization?" — narrow the claim: we expose the **exact**
   Cayley-Dickson associator as a positive first-class op; existing associativity-regularization
   (arXiv:2605.26035) penalizes a triple-product deviation, a different object and goal.
4. "Why not associative scans for efficiency?" — they require associativity, which fails here; that is
   precisely why the tensor-core `L(a)` mechanism is needed.

## 6. Empirical-payoff status (candid)
The training results to date are **teacher-target identity tests** (recover reachable parameters, loss→0)
— they validate the *machinery* (exact gradients, convergent BPTT), **not** model utility. The
associator-as-feature and zero-divisor claims are **demonstrated-in-principle**. The ML-result framing
becomes justified only once a task shows non-associativity or the associator yielding a **measurable,
ablatable** benefit. That real-capacity, real-task step is tracked in `HYPER_MATVEC_DESIGN.md`.

## 7. Must-cite
Octonion LSTM (2021); Numerion (2025); Deep Octonion Networks (Wu et al. 2019); Deep Sedenion Networks
(Bojesomo et al. 2020); Octonion-Valued NN (Popa 2016); PHM (2021) + PHNN (2022); Quaternion RNN
(Parcollet et al. 2018/19); S4 (2021), S4D (2022), S5 (2022), Mamba/S6 (2023); Clifford Neural Layers
(Brandstetter et al. 2022), Clifford Group Equivariant NNs (Ruhe et al. 2023), GATr (Brehmer et al. 2023);
complex-GEMM on tensor cores (Abdelfattah/Dongarra 2020), CUTLASS quaternion GEMM (2020); HARDBOILED
(CGO 2026), ACT (2025); Triton (2019), Exo (2022), TVM (2018); Ozaki-scheme FP64 emulation (2025);
hypercomplex-NN survey (2025); octonion associators (2015), geometry of sedenion zero divisors (2024).

## 8. Venues
`arXiv cs.PL + cs.LG` first (immediate) → **CGO / PLDI / OOPSLA / CC / ASPLOS / MLSys**
(MLSys is the strongest cross-over). **NeurIPS / ICML only with a task where non-associativity/the
associator measurably helps** — otherwise high rejection risk as method-without-payoff.
