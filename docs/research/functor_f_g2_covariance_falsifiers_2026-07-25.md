<!-- docs:meta
topic_id: repo.docs.research.functor-f-g2-covariance-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-g2-covariance-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F G₂-covariance — falsifiers and stop rules

**Companion to:** `docs/research/functor_f_g2_covariance_spec_2026-07-25.md`  
**Harness:** `scripts/research/functor_f_g2_covariance_contract.py`

---

## Clause-level falsifiers

### G1_UNIFORM_JET

**Falsifier:** Some Fano line / off-line pair produces an associator jet with `\|α(1)\| ≠ 2.0` or with support on more than one imaginary axis.

**Stop rule:** If the jet is not uniform, the whole tower of normalisations in Φ_fp collapses.

---

### G2_UNIFORM_PHI

**Falsifier:** For some Fano line, the Φ_fp parameters at `ε = 1, τ = 0` differ from `a = 0, b = ±1` by more than `1e-9`.

**Stop rule:** Line-dependence at the level of Φ_fp means the functor is not even uniform, let alone G₂-covariant.

---

### G3_PATH_C_UNIFORM / G4_PATH_D_UNIFORM

**Falsifier:** At least one Fano line fails to reproduce the neutral (Path C) or polar (Path D) end-state.

**Stop rule:** If the worked line was special, the R3 result is not portable.

---

### G5_BETTI_UNIFORM

**Falsifier:** At least one Fano line fails the Betti-0 drop (2 → 1).

**Stop rule:** The stratification homology is line-dependent.

---

### G6_CROSS_LINE_CONSISTENT

**Falsifier:** The cross-line jet for the worked pair `(1,2,3)` and `(1,4,5)` does not reproduce the uniform path classes.

**Stop rule:** Field-level coupling breaks the uniformity found at the single-line level.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| G1 or G2 fails | `G_RED` | Φ_fp normalisation is line-dependent; stop. |
| Any of G3–G5 fails on one line | `G_AMBER` | That line is anomalous; investigate basis/sign convention. |
| G6 fails | `G_RED` | Multi-line obstruction is not captured by single-line uniformity. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
