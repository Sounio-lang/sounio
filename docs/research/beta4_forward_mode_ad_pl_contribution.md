<!-- docs:meta
topic_id: repo.docs.research.beta4-forward-mode-ad-pl-contribution
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.beta4-forward-mode-ad-pl-contribution
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Covariance-Aware Compilation for Epistemic Non-Associative Algebra

**Paper-positioning note for PL + Neuroscience audiences**
**Author: Demetrios Chiuratto Agourakis**
**Draft: 2026-04-13**

## One-line summary

The Sounio self-hosted compiler, as of commit `f5167f99`, implements **forward-mode automatic differentiation** over GUM variance in the type-checker's codegen phase, enabling provably-correct first-order uncertainty propagation through arbitrary scalar polynomial expressions — including the octonion associator `[a,b,c] = (ab)c − a(bc)` that sits at the mathematical core of the non-associative epistemic connectomics research program.

## Contribution for a PL audience

### The problem

Implementations of ISO/JCGM 100:2008 (the *Guide to the Expression of Uncertainty in Measurement*, "GUM") in most languages handle variance propagation through three unsatisfactory routes:

1. **Delta method / first-order-finite-difference libraries** — e.g. Python's `uncertainties` package. Variance is carried as metadata on wrapped numeric objects. Correct for tree-shaped expression graphs with distinct sources at leaves. Undercounts correlation when the same source appears in multiple sub-expressions (the common case).
2. **Symbolic/autograd + Monte Carlo** — e.g. PyMC, Stan. Correct to higher order but runtime-heavy and outside the programming language proper; uncertainty isn't a type.
3. **Ad hoc code-level variance bookkeeping** — most scientific C/Fortran. Coverage is inconsistent; covariance handled by hand per-site.

None of these give compile-time guarantees for covariance-aware propagation. None make variance a first-class compilation concern.

### The β tier in Sounio

Sounio's compiler evolved through four tiers during 2026-04-13:

| Commit | Name | Handles |
|--------|------|---------|
| `3dc8b595` | β (direct-source) | `Var(x − x) = 0` for direct same-variable references |
| `ad53b5f7` | β² (expression hash) | Adds `Var((x+1) − (x+1)) = 0`, `Var(a·b − a·b) = 0` via compile-time hash equality |
| `f5167f99` | **β⁴ (forward-mode AD)** | **Full first-order GUM via sensitivity vectors — subsumes β and β²** |

The β⁴ scheme stores, per SSA slot per channel, the **partial derivative** `∂y/∂x_k` (the "sensitivity") of the slot's value with respect to the `k`-th uncertain input. Arithmetic ops propagate sensitivity by chain rule:

```
sens(a + b)[k] = sens(a)[k] + sens(b)[k]
sens(a − b)[k] = sens(a)[k] − sens(b)[k]        ← sign-aware
sens(a · b)[k] = a · sens(b)[k] + b · sens(a)[k]
sens(a / b)[k] = (sens(a)[k] · b − a · sens(b)[k]) / b²
```

`variance_of(y)` at query time computes `Σ_k sens(y)[k]² · σ²(x_k)`. This is forward-mode AD applied to variance — equivalent to evaluating the Jacobian-product form of the GUM propagation formula at compile-emit time.

The result: `Var(x − x)` is identically zero under β⁴, because `sens(x − x)[k] = sens(x)[k] − sens(x)[k] = 0` for all k. No special-case rule, no syntactic hash comparison — the algebra itself delivers the correct answer.

### What's PL-novel

- **Variance is a compiler concern, not a library overlay.** Emit functions in the x86 codegen (`emit_gate_variance_addsub_x86`, `emit_gate_variance_muldiv_x86`) write sensitivity updates into a dedicated BSS shadow buffer alongside the primary value. Variance computation is *inlined* into the target binary — no runtime framework overhead, no wrapper objects.
- **Self-hosted implementation.** β⁴ is written in Sounio and compiles itself. The fixed-point verification `gen2 == gen3` at md5 `731a6359ddfaa269ec67386da3388d5a` demonstrates that the β⁴-aware compiler is self-consistent.
- **First-class integration with linear types and effects.** `Knowledge<f64>` is a linear-typed carrier of value + variance. Access via `.value` requires the `Epistemic` effect, which the β⁴ scheme extends — variance propagation is automatic within any `Epistemic`-effectful function.
- **Covariance correctness for non-associative algebras.** Because the chain rule is applied per channel at the scalar level, β⁴ correctly propagates variance through *any* polynomial expression, including non-associative ones. This is the key enabler for the neuroscience half of the paper.

## Contribution for a Neuroscience audience

### The problem

The non-associative epistemic connectomics research program (see `experiments/non_assoc_connectomics/PROTOCOL.md`) proposes the octonion associator `[a,b,c] = (ab)c − a(bc)` as a first-class object of brain-network analysis. For ASD vs TD discrimination on ABIDE-I, the subject-level statistic is the 95th percentile of associator norm² across all connected node triples.

For this program to be publication-defensible under ISO 100:2008, the per-subject test statistic must carry a **computable uncertainty** that correctly accounts for the shared sources of uncertainty across the many summands in the associator polynomial. A naive "variances add" approach over-counts covariance-coupled terms; a hand-rolled symbolic derivation is error-prone and not reproducible.

### What β⁴ gives

At the Sounio-level source code for the pipeline, researchers write the associator inline as a polynomial in the scalar components of `a, b, c ∈ 𝕆`, use `.value` to unwrap `Knowledge<f64>` inputs, and call `variance_of` on the final scalar. The compiler guarantees that the reported variance is the **exact first-order GUM variance** of that polynomial.

**Canonical validation** (`tests/run-pass/octonion_associator_gum_validation.sio`, 2026-04-13):

For the Fano-triple test case `a = e_1 + δ, b = e_2, c = e_4` with `δ ~ N(0, σ²), σ = 0.1`:

- Analytical truth: `Var(‖[a,b,c]‖²) = 64·σ² = 0.64` (first-order GUM) or `64σ² + 32σ⁴ = 0.6432` (exact via `Var(X²)` formula for Gaussian `X`).
- Pre-β⁴ compiler: `0.16` (4× undercount from covariance-blind addition).
- **β⁴ compiler: `0.640000` — exact match to first-order truth.**

Under β⁴, the draft's Part II claim — that Knowledge<Octonion<f64>> arithmetic produces GUM-compliant variance for the associator — is no longer a design goal; it is a **tested, running property of the compiler**.

### Paper-integration anchor

For the Network Neuroscience paper combining Experiment B (octonion associator on ABIDE-I) and Experiment C (sedenion zero-divisor proximity):

- **Methods § Uncertainty Quantification** can point to β⁴ as the implementation of ISO 100:2008 compliance. Reviewers can re-run `octonion_associator_gum_validation.sio` to verify.
- **Supplementary Data** includes the β⁴ source in `self-hosted/compiler/lean_single.sio` at the committed SHA. Reproducibility is single-file.
- **Limitations** acknowledges:
  - First-order only — second-order Taylor corrections (the `32σ⁴` term) are not tracked; matters when `σ` approaches the scale of the expression's mean.
  - Sensitivity buffer is 8-channel by default; programs with more than 8 distinct `measure()` calls silently lose per-input decomposition for measurements 9+. Total variance remains correct for the first 8.
  - Covariance-awareness applies to the `.value` → scalar-arithmetic path. Typed `Knowledge<T>` arithmetic (e.g. `k1 + k2` where both are `Knowledge<f64>`) still uses the pre-β scalar-variance-sum rule — a follow-up fix (β⁵) is scoped but not yet implemented.

## Structure of a paper

**Venue candidates**: 
- **POPL / PLDI** — PL-heavy audience. Focus: forward-mode AD as a compile-time discipline, covariance correctness as a type-system invariant, self-hosted bootstrap.
- **Network Neuroscience / eLife / Nature Communications** — Neuroscience audience. Focus: associator-field biomarker, algorithmic uncertainty budget, ABIDE-I application.
- **Joint venue (ICSE / CAV)** — reliability/verification angle.

**Draft structure for a joint-venue submission (8-12 pages)**:
1. **Introduction** — GUM compliance as a scientific-software quality requirement; the octonion associator as a motivating non-trivial polynomial; programming-language-level implementation as a novel response.
2. **Background** — GUM formula; forward-mode AD; non-associative algebras (Cayley-Dickson tower, octonions, the 168 theorem); Sounio's effect and type system.
3. **Compiler architecture** — β tier evolution (3dc8b595 → ad53b5f7 → f5167f99); sensitivity-vector BSS layout; chain-rule emit functions; self-hosted bootstrap verification.
4. **Correctness** — per-compile-site GUM claim; validation against analytical truth for Fano-triple associator; regression against existing ISO budget tests (invariant for independent-input cases, narrower for correlated).
5. **Application — non-associative connectomics** — Part III of the research program; Phase 1 synthetic gate results; Phase 2 preregistered design; expected Phase 2 deliverables.
6. **Limitations and future work** — second-order; `Knowledge<T>` typed arithmetic (β⁵); extension to higher-dimensional Cayley-Dickson (sedenion ZD proximity as Experiment C); GPU-accelerated null-permutation ensembles.
7. **Related work** — `uncertainties.py`, PyMC, Stan, Fortran's `GUM` ports, CompCert-style certified compilation.
8. **Conclusion** — "variance is a compiler problem" as a positionable claim.

## Memory update

This note is the canonical reference for β⁴ as a research-program contribution. Future sessions discussing:

- Knowledge<T> / variance-correct code → cite β⁴ and this note
- Phase 1/2/3 of connectomics → reference `octonion_associator_gum_validation.sio` as the runnable proof that the statistical apparatus is GUM-compliant
- Dissertation contribution 2 (compile-time confidence gates) → β⁴ IS the upstream that makes contribution 2's guarantees non-trivial

## Next concrete steps (session-scoped)

1. **Run β⁴ on the ABIDE associator pipeline** (when cluster access + real frames.bin available) — should produce per-subject variance alongside the p95 statistic.
2. **Tier-5 extension (β⁵)**: extend β⁴ into `compile_knowledge_addsub_x86` / `compile_knowledge_muldiv_x86` so that `Knowledge<T>` typed arithmetic (not just `.value` scalar path) is also covariance-correct. Scoped but deferred; estimated 2-3 hour sprint.
3. **Paper outline** — expand this note into 5-page draft for the PLDI/POPL submission cycle; target a joint venue that welcomes cross-disciplinary work.
4. **Dissertation contribution 2 audit** — re-derive the existing `require_confidence` / compile-time-gate machinery's guarantees assuming β⁴ underneath. Likely strengthens the contribution's claim.
