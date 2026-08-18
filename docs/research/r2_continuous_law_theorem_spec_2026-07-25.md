<!-- docs:meta
topic_id: repo.docs.research.r2-continuous-law-theorem-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.r2-continuous-law-theorem-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# R2 continuous tube law as theorem — from measured t^{1/4} to proved contact order

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parents:** `docs/research/rupture-r2-full-tubular_2026-07-25.md` (R2_FULL_MEASURED), `docs/research/PROGRAM-REGISTRY-mercyful-learning.md` §1.1–1.3  
**Harness:** `scripts/research/r2_continuous_law_theorem_contract.py`  
**Gate:** `scripts/ci/r2_continuous_law_theorem_gate.sh`

---

## 1. What this is

R2_FULL_MEASURED established numerically that the singular-value distance to the sedenion zero-divisor locus scales as `d_sing ~ t^{1/4}` and that the determinant vanishes to order 4 on the locus. This document turns the measured exponent into a **proof sketch with executable verification**: the `t^{1/4}` law is a consequence of the determinant factorization and the codimension-4 geometry of the ZD locus.

---

## 2. Theorem (proof sketch)

**Claim.** Let `ZD₁(𝕊)` be the zero-divisor locus in the sedenions and let `d(x) = dist(x, ZD₁(𝕊))` on the unit sphere `S^{15}`. Then there exist constants `c₁, c₂ > 0` such that, for `x` near `ZD₁(𝕊)`,

```
c₁ · d(x)^4  ≤  det L_x  ≤  c₂ · d(x)^4.
```

Consequently, `d(x) ~ (det L_x)^{1/4}` as `x → ZD₁(𝕊)`.

### Proof steps

1. **Factorization.** By Koebisu (arXiv 2512.13002), verified numerically to `10^{-14}` in `catastrophe_cd.py`:
   ```
   det L_x = D₁⁴ D₂²,
   ```
   where `D₁ = |x|²` and `D₂ = D₁² - 4(AB - γ²)` with `A = |u|²`, `B = |w|²`, `γ = ⟨u,w⟩` for `x = (x₀+u, x₈+w)`.

2. **ZD locus as complete intersection.** The zero-divisor condition is `D₂ = 0` on `D₁ > 0`. Expanding:
   ```
   D₂ = C² + 2C(A+B) + (A-B)² + 4γ²,
   ```
   with `C = x₀² + x₈²`. This is a sum of four squares. It vanishes iff:
   ```
   x₀ = x₈ = 0,   A = B,   γ = 0.
   ```
   These are four independent real conditions, so `ZD₁(𝕊)` is a codimension-4 complete intersection (dimension 11 in `S^{15}`).

3. **Quadratic contact.** On the unit sphere, `A+B = 1-C`, so near the locus (`C` small):
   ```
   D₂ = C² + 2C(A+B) + (A-B)² + 4γ² ≈ 2C + (A-B)² + 4γ²,
   ```
   which is a positive-definite quadratic form in the four defining functions `x₀, x₈, A-B, γ`. Because the gradients of those functions are linearly independent on the locus (checked numerically), there exists `c > 0` such that:
   ```
   D₂(x) ≈ c · d(x, ZD₁(𝕊))²
   ```
   for `x` near the locus.

4. **Determinant scaling.** On the unit sphere, `D₁ = 1`, so:
   ```
   det L_x = D₂(x)² ≈ c² · d(x)^4.
   ```

5. **Conclusion.** `d(x) ≈ (det L_x / c²)^{1/4}`.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **T1_FACTORIZATION** | `det L_x = D₁⁴ D₂²` holds to `1e-12` for random `x`. | Max relative error `< 1e-12`. |
| **T2_D2_SUM_OF_SQUARES** | `D₂ = C² + 2C(A+B) + (A-B)² + 4γ²` holds exactly. | Max absolute error `< 1e-12`. |
| **T3_ZD_CONDITIONS** | `D₂ = 0` iff `x₀ = x₈ = 0, A = B, γ = 0`. | Verified on the 84 canonical ZD pairs. |
| **T4_GRADIENT_INDEPENDENCE** | The four defining functions have linearly independent gradients on the ZD locus. | Min singular value of Jacobian `> 1e-9` on sampled ZD points. |
| **T5_QUADRATIC_CONTACT** | `D₂(x) / d(x, ZD)²` is bounded above and below on a neighborhood of the locus. | Ratio in `[0.1, 10]` on sampled points. |
| **T6_DET_SCALING** | `det L_x / d(x)^4` is bounded above and below. | Ratio in `[0.01, 100]` on sampled points. |

---

## 4. What this is NOT

- **Not a new theorem.** The factorization is Koebisu's; the contact order is standard singularity theory.
- **Not a Lean proof.** The verification is numerical + proof sketch.
- **Not a clinical claim.**

---

## 5. Reproduce

```bash
python3 scripts/research/r2_continuous_law_theorem_contract.py
# expect: T1..T6 PASS, R2_THEOREM_VERDICT T_GREEN

bash scripts/ci/r2_continuous_law_theorem_gate.sh
# expect: R2_CONTINUOUS_LAW_THEOREM_GATE_OK
```

---

## 6. AI disclosure

Proof sketch and harness drafted under human direction (2026-07-25). Factorization attributed to Koebisu (arXiv 2512.13002). No clinical content. GAIDeT-ICMJE 2025.
