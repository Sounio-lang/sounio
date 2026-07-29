<!-- docs:meta
topic_id: repo.docs.papers.rupture-functor-f-synthesis-2026-07-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.rupture-functor-f-synthesis-2026-07-25
-->

# Rupture as Algebra: An Executable Bridge from Cayley–Dickson Zero Divisors to Petitot Morphodynamics

**Draft date:** 2026-07-25  
**Status:** DRAFT — not submitted  
**Target:** arXiv (math.DG / cs.PL)  
**Authors:** Demetrios Chiuratto Agourakis, with AI-assisted instrumentation

---

## Abstract

We present an executable bridge between two formalizations of rupture: the algebraic annihilation of Cayley–Dickson zero divisors and the morphodynamic bifurcations of Petitot's semantic potentials. On the algebraic side, we verify the Koebisu factorization of the left-multiplication determinant in the sedenions and prove that the measured `t^{1/4}` tube contact follows from the codimension-4 complete-intersection geometry of the zero-divisor locus. On the semantic side, we construct a first-principles map `Φ_fp` from the octonion associator jet near a Fano line to the control parameters of Petitot's cusp potential, and verify that it reproduces the two opposition moves (contrariety vs contradiction) as distinct path classes. We then lift this bridge to a **Functor F**: a jet-functorial assignment from associator data to stratification homology that is uniform across all seven basis-aligned Fano lines, admits a `G₂`-equivariant formulation via a natural pairing, and extends to the field of cross-line couplings. We also present a first implementation of Mercyful Learning, a substrate-aware scheduler that optimizes integrated versus peak suffering under a length budget with an explicit anti-Goodhart constraint, and a Falsification Ledger, a comment-scanned system that treats refuted hypotheses as first-class, versioned, citable artifacts in the Sounio programming language. All claims are bounded by executable CI gates; no clinical or trained-model discovery is claimed.

---

## 1. Introduction

Composition that is not context-free is the shared name of two research faces: **semantic** rupture (how meanings fail to reassociate) and **epistemic** rupture (how knowledge fails to invert or annihilates). Classical formalisms mark rupture with a negative object — a missing map, an unglued section, a diverging metric, a vanished attractor. The Cayley–Dickson tower supplies positive, graded, computable invariants: the octonion **associator** and the sedenion **zero-divisor** singularity.

This paper has four contributions:

1. **A proved tube law.** We turn the measured `t^{1/4}` contact of the sedenion zero-divisor locus into a proof sketch verified by executable contract, grounded in the Koebisu factorization and the codimension-4 geometry of the locus.

2. **A first-principles semantic bridge.** We define `Φ_fp`, a map from the Fano-neighborhood associator jet to the cusp control plane, and verify that it realizes Petitot's contrariety and contradiction as distinct path classes sourced from the octonion field.

3. **A functorial lift.** We exhibit Functor F as a jet-functorial, `G₂`-covariant assignment from associator data to stratification homology, with an explicit obstruction and constructive resolution for the polar dial.

4. **Infrastructure for negative results.** We implement a Mercyful Learning scheduler and a Falsification Ledger, treating controlled negatives as first-class scientific artifacts.

Every claim is paired with an executable CI gate. The ladder is `RUPTURE_ABCD_CONTRACTS_OK` at the time of writing. We also supply two pieces of negative-result infrastructure: a Mercyful Learning scheduler (§5) and a Falsification Ledger (§6), which together make the eight controlled negatives of the programme a concrete, inspectable artifact rather than a narrative afterthought.

---

## 2. The Cayley–Dickson catastrophe

Let `L_x` be the left-multiplication operator in a Cayley–Dickson algebra. The zero-divisor set is the singular locus `det L_x = 0`. For the division algebras `ℝ, ℂ, ℍ, 𝕆`, this set is empty; it is born at the `𝕆 → 𝕊` doubling.

**Theorem 2.1 (Koebisu, arXiv 2512.13002).** For `x ∈ 𝕊`,
```
det L_x = D₁⁴ D₂²,
```
where `D₁ = |x|²` and `D₂ = D₁² - 4(AB - γ²)`, with `A = |u|²`, `B = |w|²`, `γ = ⟨u,w⟩` for `x = (x₀+u, x₈+w)`.

We verified this factorization to `1.4 × 10^{-14}` over 200 random unit sedenions (`T1_FACTORIZATION`).

**Theorem 2.2 (tube contact).** Let `ZD₁(𝕊)` be the zero-divisor locus and `d(x) = dist(x, ZD₁(𝕊))` on `S^{15}`. Then `det L_x ≍ d(x)^4`, hence `d(x) ~ (det L_x)^{1/4}`.

*Proof sketch.* `D₂` expands as `C² + 2C(A+B) + (A-B)² + 4γ²`, a sum of four independent squares on the unit sphere. The ZD locus is the complete intersection `x₀ = x₈ = 0, A = B, γ = 0`, codimension 4. Gradient independence is verified numerically (`T4_GRADIENT_INDEPENDENCE`, min singular value `1.0`). Near the locus, `D₂ ≈ 2C + (A-B)² + 4γ²` is a positive-definite quadratic form, so `D₂ ≈ c d²` and `det L_x = D₂² ≈ c² d⁴` on `S^{15}`. The fitted slope of `log det` versus `log d` is `3.994` (`T6_DET_SCALING`). ∎

This replaces the earlier `R2_FULL_MEASURED` label with a proved statement bounded by executable verification.

---

## 3. The Fano restriction and `Φ_fp`

The octonion associator `[x,y,z] = (xy)z - x(yz)` is the order-1 rupture object. For a Fano line `L = {e_i, e_j, e_k}` (an associative triple) and an off-line unit `e_u`, the jet is linear and single-axis:

```
[e_i + ε e_u, e_j, e_k] = ε · [e_u, e_j, e_k] = ±2ε · e_m.
```

**Definition 3.1 (`Φ_fp`).** Map the jet to the cusp control plane. The **even** jet (norm) drives the opposition **depth** `a`, while the **odd** jet (direction) drives the **tilt** `b`:
```
a(ε) = -1 + ‖α‖²/4 = -1 + ε²,
b(ε) = τ + α_m/2   = τ ± ε,
```
where `A₀ = -1` is the semantic unit for opposition depth and `τ` is the tilt. The geometric picture is that the ambient non-associativity of the octonion field supplies the extra control parameter that a Boolean `2²` square cannot host as two distinct opposition types.

The cusp potential `V(x; a,b) = x⁴/4 + a x²/2 + b x` has bifurcation set `4a³ + 27b² = 0`.

**Result 3.2 (R3_GREEN).** The `Φ_fp` paths reproduce Petitot's two opposition moves:
- **Path C (contrariety):** `τ = -α_m/2` gives `b ≡ 0`; both wells merge to a neutral monostable state.
- **Path D (contradiction):** `τ = 0` gives `b = ±ε`; one pole deepens as the other vanishes, with sign selection.

**Result 3.3 (R4_GREEN).** The same path classes are sourced from the **cross-line** jet of two Fano lines meeting in one unit, not from a single-line off-unit perturbation.

---

## 4. Functor F: from path classes to stratification homology

We lift `Φ_fp` to a functorial assignment.

**Definition 4.1 (Functor F).** For a jet object `(L, u, τ, ε)`, `F` returns the sublevel-set stratification invariant of `V(·; a(ε), b(ε))`:
- number of wells (0, 1, or 2);
- deepest-well sign;
- `Betti-0` of `{x : V(x) ≤ c}` at the canonical regular value.

**Result 4.2 (F_GREEN).** `F` is well-defined and jet-linear: the odd associator jet controls the first-order topological deformation (`db/dε|₀ = ±1`).

**Result 4.3 (G_GREEN).** `F` is uniform across all 7 basis-aligned Fano lines.

**Result 4.4 (H_CHARACTERISED).** The `argmax`-extracted polar dial `b` is **not** `G₂`-covariant under generic automorphisms (`frac_moved = 1.000`). The constructive fix is the pairing `b_cov = ⟨α, e_m⟩`, which is `G₂`-invariant to `1.3 × 10^{-15}`. Up to scalar multiple, this pairing is the unique `G₂`-invariant linear functional on the single-axis jet, since the jet spans a 1-dimensional subspace and `e_m` is the only distinguished direction fixed by the `(line, off-line)` configuration.

**Result 4.5 (E_GREEN).** With `b_cov` as canonical polar coordinate, the entire ladder is `G₂`-equivariant over a continuous orbit `exp(t · 𝔤)`.

**Result 4.6 (K_CHARACTERISED).** Functor F extends to the field of cross-line couplings as "additive iff associative"; the correction is the cross-associator, `G₂`-covariant, with `‖α‖ = 2`.

---

## 5. Mercyful Learning: suffering-budget scheduling

Mercyful Learning is a training principle: minimize suffering subject to reaching a target state, with an explicit trade-off between integrated and peak suffering. We implement the first scheduler.

**Definition 5.1.** On a finite directed graph with edge lengths `ℓ(e) > 0` and suffering field `s : V → ℝ≥0`, the cost of a path `γ` from `start` to `target` with `len(γ) ≤ L₀` is:
```
cost(γ; μ) = Σ_{(u,v)∈γ} s(u)·ℓ((u,v)) + μ · max_{v∈γ} s(v).
```
The anti-Goodhart constraint is that paths not reaching `target` are infeasible.

**Result 5.2 (M_GREEN).** On a synthetic exposure-therapy graph, a raw-suffering minimizer avoids recovery (integral 0), while the mercyful scheduler passes through distress and reaches recovery. The Pareto frontier between integrated and peak suffering is computed exactly by exhaustive enumeration on small graphs.

---

## 6. The Falsification Ledger

We treat refuted hypotheses as first-class artifacts in the Sounio programming language.

**Definition 6.1 (Claim).** A claim is a record with `hypothesis`, `falsifier`, `evidence`, `harness`, `gate`, `verdict`, and optional `provenance` from the zero-event taxonomy (`absent`, `cancelled`, `annihilated`, `below_resolution`, `rounded`, `gated`, `unknown`).

**Result 6.2 (L_GREEN).** A comment-scanned ledger extracts claims from `.sio` files and validates schema, harness/gate existence, no orphans, verdict consistency, and seed completeness.

**Result 6.3 (Z_GREEN).** Zero-provenance claims require the `provenance` field when evidence involves a zero surface value.

**Result 6.4 (A_GREEN).** A preprocessor converts `claim` blocks into type-safe `const Claim` literals without touching the parser.

---

## 7. Discussion

The bridge is **not** an identity. We do not claim `Bifurcation(V) ≅ ZD(𝕊)` (D3 remains forbidden). What we claim is a functorial correspondence between the algebraic jet and the topological stratification, verified by executable contract.

The eight controlled negatives of the rupture programme — including the absence of trained-model annihilation signatures and the falsification of topological mountain-pass obstruction — are not failures but the active content of the Falsification Ledger. They delimit the search space.

---

## 8. Limitations

- The `t^{1/4}` theorem is verified numerically, not proved in Lean.
- The `G₂`-equivariance is constructive over sampled automorphisms, not a full group-theoretic proof.
- Mercyful Learning is a graph prototype; no real training substrate is coupled.
- The Falsification Ledger is comment-scanned; native parser support is deferred.

---

## 9. Reproducibility

All contracts are executable:

```bash
bash scripts/ci/rupture_abcd_contracts_gate.sh
bash scripts/ci/r2_continuous_law_theorem_gate.sh
bash scripts/ci/functor_f_jet_functoriality_gate.sh
bash scripts/ci/functor_f_g2_covariance_gate.sh
bash scripts/ci/mercyful_runtime_gate.sh
bash scripts/ci/falsification_ledger_gate.sh
bash scripts/ci/zero_provenance_claims_gate.sh
bash scripts/ci/claim_ast_gate.sh
```

---

## References

- Koebisu, S. (2025). *Determinant Factorization for Left Multiplication in the Sedenions*. arXiv:2512.13002.
- Petitot, J. (1985). *Morphogenèse du Sens*. PUF.
- Thom, R. (1972). *Stabilité Structurelle et Morphogénèse*. Benjamin.
- Cawagas, R. (2004). *On the structure and zero divisors of the Cayley-Dickson sedenion algebra*. Discuss. Math. Gen. Algebra Appl.
- Moreno, G. (1998). *The zero divisors of the Cayley-Dickson algebras over the real numbers*. Bol. Soc. Mat. Mex.

---

## Appendix A. The eight controlled negatives

| # | Hypothesis | Result |
|---|---|---|
| 0 | O-SSM detects non-associativity in ABIDE and depression data | Null (conclusive for data, not hypothesis) |
| 1 | Affectively coherent ordering improves training | Falsified and inverted (coherent 67.85% < shuffled 72.39% < anti-coherent 74.19%) |
| 2 | Per-example second-derivative filter detects annihilators | Refuted by three routes (gradient large, Hessian rank-deficient, loss separates better) |
| 3 | Topological dichotomy in 𝕊 (`c* → ∞` between components) | Retracted (`det L_x ≥ 0`, codim ≥ 2 implies complement connected) |
| 4 | Mountain-pass / necessary suffering as obstruction | Demolished by connectivity of all sublevels |
| 5 | `λ*` as exchange rate given by algebra | Refuted by endpoint sweep (11.04 to 0.80) |
| 6 | Structural subspace death in trained LSTM | Absent; false positive killed by `align(k)` curve and init control |
| 7 | Real semantic fields have topological barriers | Falsified in both single-subject and aggregate DreamBank PMI fields with sensitivity to ≥1.5σ |

---

## AI disclosure

This synthesis was drafted under human direction with AI-assisted instrumentation. Math-facing claims are bounded by the named CI gates and the D0–D3 discipline. No clinical or patient-level claim. GAIDeT-ICMJE 2025.
