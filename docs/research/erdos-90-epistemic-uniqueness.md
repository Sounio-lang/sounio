<!-- docs:meta
topic_id: repo.docs.research.erdos-90-epistemic-uniqueness
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-90-epistemic-uniqueness
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Erdős [90], reframed: an epistemic unit-distance graph (a Sounio capability study)

*Companion to `stdlib/research/erdos90_epistemic.sio` and `…/erdos90_search_epistemic.sio`.
This is a **capability demonstration** — a different *kind* of unit-distance reasoning,
under measurement uncertainty. It is **not** a new bound on `u(n)`, and it does not beat the
Erdős grid. All numbers below are reproduced by `souc run` (see Reproduce).*

## 1. Motivation
We established (3 nodes × 3 seeds, exact integer search) that lattice subset search cannot
beat the Erdős grid at relevant scales — a *classical* wall. The configurations that could
go past it are non-lattice, with irrational coordinates, where "is this distance exactly 1?"
is an **epistemic** question with an error budget. Classical tools collapse it to a boolean.
Sounio can instead carry a GUM-propagated answer with one genuine static safety guarantee.

## 2. What is static vs runtime (read before any "compile-time" claim)
- **Static (compiler-enforced):** extracting `Knowledge<f64>.value` **requires** the
  `with Epistemic` effect — otherwise error **E170**. This *unwrap-safety* is the **only**
  static guarantee we rely on. Witness: `tests/compile-fail/erdos90_unwrap_unsafe.sio`
  fails with E170 at the `.value` access.
- **Runtime:** the per-edge conformance probability, the `p_c ≥ p*` decision, the counts,
  and the count standard deviation are ordinary float computations.
- **Not relied upon (audited, found immature):** the compiler also has
  `Knowledge<f64, epsilon < τ>` subsumption and `with Epistemic(N)` completeness. We tried
  to use them as input-precision / effect-completeness gates and found them unreliable in the
  current compiler — e.g. the repo's own `tests/compile-fail/epistemic_epsilon_mismatch.sio`
  is *intended* to fail on an ε mismatch but actually fails on E170 first, and a
  straightforward typed-ε pass-case tripped a spurious "ε boundary violation". We therefore
  make **no** static claim beyond E170. Maturing these gates is a roadmap item.

**Earlier drafts of this note said "compile-time gated" for the certified count. That was
wrong: the count decision is a runtime `p_c ≥ p*` comparison. Corrected here.**

## 3. The metric (probability of conformance, with explicit assumptions)
Per edge, with GUM combined uncertainty `σ_d` on the distance `d`, conformance to the unit
specification `[1−τ, 1+τ]` has probability
`p_c = Φ((1+τ−d)/σ_d) − Φ((1−τ−d)/σ_d)`, `Φ(z)=½(1+erf(z/√2))` — the standard JCGM 106:2012
conformance probability. This *replaces* the earlier ad-hoc `margin/(margin+1)` heuristic.
It is principled in **form**, but rests on **modeling choices, not derivations**:
(i) Gaussian / large-effective-DOF model of the combined uncertainty; (ii) `τ` is a *chosen*
tolerance; (iii) `σ_d = σ_{d²}/(2d)` is first-order GUM linearization, valid for `σ_{d²}≪d²`,
undefined at `d→0`; (iv) independence assumed in propagation. `erf` is Abramowitz–Stegun
7.1.26 (verified `erf(1)=0.842701`, `erf(2)=0.995322`).

*Caveat we observed:* the runtime GUM shadow (`variance_of`) and the analytic linearization
`σ_d=√2·σ_coord` (used in the search pilot) agree only approximately; the artifact uses the
runtime shadow, the pilot uses the analytic form. Reconciling them is a roadmap item.

## 4. Demonstration (`u(4)=5` rhombus, measured) — actual `souc run` output
| σ_coord | τ | classical boolean | conformance-certified (p_c≥0.9) | flagged | expected count Σp_c ± σ |
|--------|----|------|------|------|------|
| 0.005 | 0.10 | 5 | **5** | 0 | 5.000 ± 0.000 |
| 0.08  | 0.10 | 5 | **0** | 5 | 3.944 ± 0.913 |
The √3 diagonal stays `p_c ≈ 0` via real GUM propagation. The boolean count is **5 in both
rows**; only the conformance count distinguishes a precise from a noisy realization. The
count uncertainty `σ = √(Σ p_c(1−p_c))` is the **independence** approximation — corrected
after math-review (an earlier draft mislabeled a `√(Σσ_d²)` geometric aggregate as the count
grade). Because the edges share coordinate uncertainty (positively correlated), this `σ`
**under-estimates** the true spread; the correlated variance is a roadmap item.

τ-sensitivity at σ=0.02 (τ is a modeling choice, shown not asserted):
| τ | p_c per unit edge | certified | Σp_c |
|----|------|------|------|
| 0.02 | 0.683 | 0 | 3.41 |
| 0.05 | 0.988 | 5 | 4.94 |
| 0.10 | ≈1.000 | 5 | ≈5.00 |

## 5. Continuous-search pilot (`erdos90_search_epistemic.sio`)
Maximizes the **smooth** expected conformance-count over 6 continuous point coordinates by
gradient ascent with a backtracking line search, **reusing** `stdlib/optimize/epistemic_bfgs`:
its `EpistemicParams`/`EpistemicGradient` types and its **uncertainty-aware step rule**
`ebfgs_compute_step_size` (which shrinks the step when the gradient's propagated uncertainty
is high). From a perturbed start the expected count rises **4.082 → 6.997** (monotone).
*Honest scope:* a smooth surrogate of the count over continuous coordinates; it does **not**
beat the grid and is **not** a `u(n)` result. NB the full BFGS driver
`epistemic_bfgs_quadratic` hardcodes its objective, so we reuse its primitives + step rule
and run ascent; parameterizing it over a fn-ref objective is a roadmap item.

## 6. Honest limitations and scope
- 4 points / 6 pairs in the static demo; 6 points in the pilot. Small, known/perturbed
  configs. **Not** a search result over the open problem, **not** a bound.
- The conformance metric's `τ` and Gaussian model are choices; the `σ_d` linearization is
  invalid for large relative uncertainty and at `d→0`.
- The **only** static guarantee is E170 unwrap-safety; the richer ε/`Epistemic(N)` gates are
  present in the language but were unreliable in our hands (§2).
- Runtime-shadow vs analytic σ_d disagree slightly (§3 caveat).
- The reported count uncertainty `σ=√(Σ p_c(1−p_c))` assumes **independent** edges; the
  edges share coordinate uncertainty, so it under-estimates the true spread (math-review).
- `graded_effects.sio` is spec-level (no run-pass); the pilot inlines its quadrature pattern.

## 7. Comparison with prior art (honest; each capability exists somewhere)
| approach | rigorous uncertainty | PL-static guarantee | exact geometry | one compiled language |
|----------|----|----|----|----|
| interval / affine arithmetic | yes (bounds) | no | partial | no |
| probabilistic programming (Stan/PyMC) | yes (distributions) | no | no | no |
| refinement types (Liquid Haskell / F\*) | no UQ | yes | no | partial |
| exact certifier (our Lean `countUnit`) | n/a (exact) | machine proof | yes | no (separate tool) |
| **Sounio (this work)** | GUM / JCGM-106 | E170 unwrap-safety only | composes w/ Lean | **yes** |
The contribution is the **combination in one compiled language**, with the static guarantee
honestly scoped to unwrap-safety. We do **not** claim the strongest cell in any single
column; each individual capability is stronger elsewhere.

## 8. Composition with the exact certifier
The epistemic layer proposes under uncertainty; the exact engine
(`SounioErdos90PlanarLowerBound.lean`, `countUnit`/`native_decide`) certifies any **rational**
candidate exactly. Epistemic search ⇒ rational candidate ⇒ exact proof.

## 9. Measurable roadmap
1. Mature the static gates: make `Knowledge<f64, epsilon<τ>` and `Epistemic(N)` reliably
   reject under-precise inputs (add the missing compile-fail tests) — then the static claim
   can extend beyond E170.
2. Parameterize `epistemic_bfgs` over a fn-ref objective; re-run the pilot under true BFGS.
3. Reconcile the runtime GUM shadow with the analytic `σ_d` (a known-answer test).
4. Larger configs (n≈7–9) where exact `u(n)` is known: compare epistemic-certified vs exact.
5. τ-sensitivity as a published table (started in §4).

## Reproduce
```
cd /workspace/sounio-erdos90
/workspace/sounio/bin/souc run   stdlib/research/erdos90_epistemic.sio          # §4 tables
/workspace/sounio/bin/souc run   stdlib/research/erdos90_search_epistemic.sio   # §5 pilot 4.08→7.0
/workspace/sounio/bin/souc check tests/compile-fail/erdos90_unwrap_unsafe.sio   # MUST fail E170 (§2)
```
