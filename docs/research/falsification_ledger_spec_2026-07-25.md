<!-- docs:meta
topic_id: repo.docs.research.falsification-ledger-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.falsification-ledger-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Falsification Ledger — compiler-integrated scientific claims in Sounio

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parent:** Original plan Approach A (Falsification Ledger)  
**Harness:** `scripts/research/falsification_ledger_contract.py`  
**Gate:** `scripts/ci/falsification_ledger_gate.sh`  
**Seed content:** `stdlib/epistemic/rupture_claims.sio`

---

## 1. What this is

No mainstream programming language treats a *refuted hypothesis* as a first-class, versioned, citable, executable artifact. This document specifies the **Falsification Ledger**: a source-level vocabulary for scientific claims, a scanner that builds a ledger from the codebase, and a CI gate that enforces claim hygiene.

The first implementation is **comment-scanned** (no parser change) so it can land without touching Madaros while the compiler surface is still moving. A future rung can lift the same schema into native AST.

---

## 2. Claim syntax (comment form)

Claims are declared in `.sio` files using a block of `// @key value` lines:

```sounio
// @claim rupture_lstm_no_annihilation
// @hypothesis trained LSTM on adding problem exhibits composed annihilation via principal-angle alignment
// @falsifier orientation_scramble produces equivalent gap_dominance; align(k) curve is plateau, not shoulder
// @evidence instrument_controlled
// @harness scripts/research/rupture_ord2_trained_lstm_probe.py
// @gate scripts/ci/rupture_abcd_contracts_gate.sh
// @verdict negative
```

### Required fields

| Field | Meaning | Example |
|---|---|---|
| `claim` | Unique identifier | `rupture_lstm_no_annihilation` |
| `hypothesis` | What is being claimed | plain text |
| `falsifier` | What would refute it | plain text |
| `evidence` | Evidence level | `conceived`, `implemented`, `type_check`, `compiles`, `executes`, `gate_green`, `instrument_controlled`, `claim_ready` |
| `harness` | Path to executable script | `scripts/research/...` |
| `gate` | Path to CI gate | `scripts/ci/...` |
| `verdict` | Current status | `alive`, `negative`, `dormant`, `refuted` |

### Optional fields

| Field | Meaning |
|---|---|
| `note` | Free-form note |
| `archive_reason` | Why archived |

---

## 3. Ledger schema

The scanner emits `.sounio/claims/<branch>.jsonl`:

```json
{
  "claim": "rupture_lstm_no_annihilation",
  "file": "stdlib/epistemic/rupture_claims.sio",
  "line": 2,
  "sha": "862ab5a76c7a",
  "hypothesis": "trained LSTM ...",
  "falsifier": "orientation_scramble ...",
  "evidence": "instrument_controlled",
  "harness": "scripts/research/rupture_ord2_trained_lstm_probe.py",
  "gate": "scripts/ci/rupture_abcd_contracts_gate.sh",
  "verdict": "negative",
  "note": null,
  "archive_reason": null
}
```

---

## 4. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **L1_SCHEMA** | Every claim has all required fields and valid enum values. | Schema validation passes. |
| **L2_HARNESS_EXISTS** | Every `harness` path exists and is executable (or at least readable). | File exists. |
| **L3_GATE_EXISTS** | Every `gate` path exists and is executable. | File exists and is executable. |
| **L4_NO_ORPHANS** | Every claim's source file exists. | No dangling references. |
| **L5_VERDICT_CONSISTENT** | `verdict=negative` claims have a `falsifier` and a `gate`. | All negative claims are falsifier-backed. |
| **L6_SEED_RUPTURE** | The 8 rupture negatives are encoded as claims in `stdlib/epistemic/rupture_claims.sio`. | Scanner finds ≥ 8 rupture claims. |

---

## 5. What this is NOT

- **Not a proof system.** The ledger records what is claimed and where the evidence lives; it does not verify the math.
- **Not a compiler pass (yet).** The first implementation is a Python scanner over `.sio` comments.
- **Not a clinical claim.** Clinical claims are out of scope.
- **Not a replacement for docs.** The ledger complements, not replaces, human-readable research notes.

---

## 6. Future rungs

1. **AST-native claims** — parser support for `claim ... { ... }` blocks in Sounio.
2. **LSP hover** — show claim metadata on hover.
3. **Gate cross-check** — run the named gate and verify the verdict matches.
4. **Archival** — moving a claim to `dormant` requires a reason.

---

## 7. Reproduce

```bash
python3 scripts/research/falsification_ledger_contract.py
# expect: L1..L6 PASS, FALSIFICATION_LEDGER_VERDICT L_GREEN

bash scripts/ci/falsification_ledger_gate.sh
# expect: FALSIFICATION_LEDGER_GATE_OK
```

Pure Python.

---

## 8. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
