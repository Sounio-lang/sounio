<!-- docs:meta
topic_id: repo.docs.research.lean-falsification-ledger-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.lean-falsification-ledger-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lean formalization of Falsification Ledger claim logic

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parent:** `docs/research/falsification_ledger_spec_2026-07-25.md` (L_GREEN), `docs/research/zero_provenance_claims_spec_2026-07-25.md` (Z_GREEN)  
**Lean file:** `formal/lean4/SounioFalsificationLedger.lean`  
**Gate:** `scripts/ci/lean_falsification_ledger_gate.sh`

---

## 1. What this is

The Falsification Ledger currently lives in Python scanners and Sounio comment blocks. This rung formalizes the claim schema in Lean 4, proving that the evidence order is a total order, the zero-provenance taxonomy is exhaustive and mutually exclusive, and verdict transitions preserve consistency.

---

## 2. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **F1_LEAN_COMPILES** | `formal/lean4/SounioFalsificationLedger.lean` compiles with `lake build`. | `lake build` succeeds. |
| **F2_EVIDENCE_TOTAL_ORDER** | Evidence levels are reflexive, transitive, antisymmetric. | Theorems proved without `sorry`. |
| **F3_PROVENANCE_TAXONOMY** | Zero-provenance categories are exhaustive and mutually exclusive. | Theorems proved. |
| **F4_CONSISTENCY_PRESERVED** | Verdict transitions to dormant/refuted preserve claim consistency. | Theorems proved. |
| **F5_NO_CLINICAL_CLAIM** | The formalization contains no clinical interpretation. | No clinical terms in Lean file. |

---

## 3. What this is NOT

- **Not a parser change.** The Lean file is a specification, not compiler code.
- **Not a proof of the rupture mathematics.** It formalizes the ledger logic, not the sedenion geometry.
- **Not a clinical claim.**

---

## 4. Reproduce

```bash
cd formal/lean4 && lake build SounioFalsificationLedger
bash scripts/ci/lean_falsification_ledger_gate.sh
# expect: LEAN_FALSIFICATION_LEDGER_GATE_OK
```

---

## 5. AI disclosure

Spec and Lean file drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
