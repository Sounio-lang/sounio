<!-- docs:meta
topic_id: repo.docs.research.r2-continuous-law-theorem-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.r2-continuous-law-theorem-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# R2 continuous law theorem — falsifiers and stop rules

**Companion to:** `docs/research/r2_continuous_law_theorem_spec_2026-07-25.md`  
**Harness:** `scripts/research/r2_continuous_law_theorem_contract.py`

---

## Clause-level falsifiers

### T1_FACTORIZATION

**Falsifier:** `det L_x ≠ D₁⁴ D₂²` beyond `1e-12` for any random sedenion.

**Stop rule:** The proof has no foundation; do not proceed.

---

### T2_D2_SUM_OF_SQUARES

**Falsifier:** The expansion of `D₂` as a sum of squares fails.

**Stop rule:** The codimension-4 geometry argument collapses.

---

### T3_ZD_CONDITIONS

**Falsifier:** A canonical zero divisor does not satisfy the four conditions, or a non-ZD satisfies them.

**Stop rule:** The locus is misidentified.

---

### T4_GRADIENT_INDEPENDENCE

**Falsifier:** The Jacobian of the four defining functions is rank-deficient at a ZD point.

**Stop rule:** The complete-intersection claim fails; quadratic contact cannot be proved this way.

---

### T5_QUADRATIC_CONTACT

**Falsifier:** `D₂ / d²` is unbounded or tends to zero near the ZD locus.

**Stop rule:** The contact order is not quadratic; the exponent is not 1/4.

---

### T6_DET_SCALING

**Falsifier:** `det L_x / d⁴` is unbounded or tends to zero.

**Stop rule:** The determinant does not vanish to order 4; the measured law was numerical artifact.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| T1 or T2 fails | `T_RED` | Foundation broken; do not proceed. |
| T4 fails | `T_RED` | Complete intersection fails; theorem wrong. |
| T5 or T6 fails | `T_RED` | Exponent is not 1/4; retract the law. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
