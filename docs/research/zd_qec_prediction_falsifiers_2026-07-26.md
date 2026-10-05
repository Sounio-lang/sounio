<!-- docs:meta
topic_id: repo.docs.research.zd-qec-prediction-falsifiers-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zd-qec-prediction-falsifiers-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sedenion ZD crown-graph code — falsifiers and stop rules

**Companion to:** `docs/research/zd_qec_prediction_spec_2026-07-26.md`
**Harness:** `scripts/research/zd_qec_prediction_contract.py`

---

## Clause-level falsifiers

### Q1_ZD_DESIGNS

**Falsifier:** The canonical ZD index-pair censuses at levels 4/5/6 deviate from 42 / 294 / 1518.

**Stop rule:** The rupture census (parent contracts C1) regressed or the 2-cycle criterion harness broke. Re-audit `cds()` against `routon_zd_contract.py` before touching anything downstream; every later clause is void until Q1 is green.

---

### Q2_CROWN_THEOREM

**Falsifier:** The level-4 ZD graph is not `H₇ ⊔ K₁`: any within-side edge, any non-isolated vertex 8, any vertex of degree ≠ 6, or a fiber that is not its predicted near-perfect matching.

**Stop rule:** The fiber-label law or the defect diagonal (chingon C3/C7) is wrong. A single deviating edge is a novel object — isolate it and re-derive §2.2 of the spec. All code claims (Q4–Q8) are void.

---

### Q3_CYCLE_CENSUS

**Falsifier:** Triangle counts ≠ {0, 1092, 19236} or 4-cycle counts ≠ {210, 17136, 703752} at levels 4/5/6; or the witness triangle `(2, 11, 26)` (label identity `9 ⊕ 17 = 24`) absent at level 5 or 6.

**Stop rule:** Girth/distance claims change. If L4 triangles > 0, the classical code distance is 3 and every exponent-4 prediction (P1/P2 detection mode) is falsified in place. If the L5/L6 witness is absent, the no-threshold theorem (P3) loses its mechanism.

---

### Q4_CLASSICAL_CODE

**Falsifier:** Cycle-code parameters ≠ `[42, 29, 4]`; a weight-4 codeword that is not a 4-cycle; a 4-cycle vector inside the cut space; cut-space minimum ≠ 6 or weight distribution ≠ the tabulated 16-class distribution.

**Stop rule:** The classical physical prediction (P1: `210 p⁴`, `840 p³`) is void. Recount before proceeding to the quantum clauses.

---

### Q5_HGP_CODE

**Falsifier:** CSS commutation fails; `k ≠ 842`; any of the 17640 explicit weight-4 operators fails to be a nontrivial logical; any two single-error syndromes collide (weight-2 stabiliser exists).

**Stop rule:** The `[[1960, 842, 4]]` claim fails. If only the single-error distinctness fails, the correction-mode coefficient (70560) is void but detection-mode predictions survive.

---

### Q6_LOGICAL_SPECTRUM

**Falsifier:** The complete pair-syndrome enumeration finds weight-4 centraliser elements beyond the 8820 + 8820 product-family logicals (e.g. mixed LL/RR support), or finds weight-4 stabilisers (would require min stabiliser weight < 6, contradicting the Q4 cut enumeration).

**Stop rule:** The detection-mode coefficient changes from 17640 to the recount; the spec's "exact coefficient" claim is demoted to the recomputed value and the discrepancy is reported as a finding, not patched over.

---

### Q7_FAMILY_COLLAPSE

**Falsifier:** Family parameters ≠ `[[1960,842,4]]`, `[[87336,70226,3]]`, `[[2308168,2122850,3]]`; or girth 4 re-emerging at level ≥ 5 (triangles vanishing).

**Stop rule:** The no-threshold theorem (P3) fails. Note: a level-7+ computation showing girth ≥ 4 after level 6 would *not* contradict the verified clause but would falsify the spec's extension remark; report against the witness-embedding argument (`(2,11,26)` has all indices < 32, so it embeds at every b ≥ 5 — a girth-4 level ≥ 5 therefore also falsifies the tower embedding C5 of the parent contracts).

---

### Q8_PHYSICAL_COEFFICIENTS

**Falsifier:** Assembly inconsistency — the tabulated coefficients do not follow from the Q4/Q6 counts (210 → `210 p⁴`/`840 p³`; 8820/17640 → the quantum coefficients).

**Stop rule:** Bookkeeping bug; recompute from the spectra.

---

## Experiment-level falsifiers (physical world)

- **F1 (exponent).** In Test 1/2/3 of the spec protocol, the measured leading exponent of the undetected/logical error rate differs from 4 (detection mode) or 3 (correction mode) at statistical significance, under the stated noise model and decoder. *Meaning:* the code's distance is not 4 — the crown/girth theorem is physically falsified.
- **F2 (coefficient).** The measured leading coefficient excludes 210 / 840 / 8820 / 17640 / 35280 / 70560 (as applicable) at 95 % confidence. *Meaning:* the low-weight logical spectrum differs from the exact enumeration — a new weight-4 logical class exists (would have shown in Q6 if computational) or the hardware/noise deviates from the model (then the model assumption, not the rupture claim, is at fault — check the assumptions register first).
- **F3 (small undetected event).** Any observed undetected logical error of weight ≤ 3 on the quantum code. *Meaning:* d < 4; the crown theorem is false.
- **F4 (threshold).** Demonstrated threshold behaviour when scaling within the ZD-code family. *Meaning:* distances grow somewhere — the triangle-forcing mechanism (Q3) fails at some level.
- **F5 (wrong design).** An independent recomputation of the level-4 design (e.g. by the SVD scan of the L4 parent contract) producing a graph other than `H₇ ⊔ K₁`. *Meaning:* the 2-cycle criterion and the SVD reference disagree — a census-level rupture-programme bug, escalate to the parent contracts.

## Non-falsifiers (do not report as failures)

- Departure of the *pseudothreshold crossing* from the formal leading-order estimates (`5.3×10⁻³`, `2.0×10⁻²`): higher-order terms are expected to move it; only exponents and leading coefficients are exact claims.
- Circuit-level (gate-error) noise changing the measured coefficients: the stated coefficients are phenomenological-noise predictions.
- The code being outperformed by other codes: no optimality is claimed.
