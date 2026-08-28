<!-- docs:meta
topic_id: repo.docs.research.falsification-ledger-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.falsification-ledger-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Falsification Ledger — falsifiers and stop rules

**Companion to:** `docs/research/falsification_ledger_spec_2026-07-25.md`  
**Harness:** `scripts/research/falsification_ledger_contract.py`

---

## Clause-level falsifiers

### L1_SCHEMA

**Falsifier:** A claim block is missing a required field or uses an invalid evidence/verdict value.

**Stop rule:** Fix the block or reject the claim.

---

### L2_HARNESS_EXISTS / L3_GATE_EXISTS

**Falsifier:** A harness or gate path does not exist in the repo.

**Stop rule:** The claim is orphaned; either create the artifact or archive the claim.

---

### L4_NO_ORPHANS

**Falsifier:** A claim block exists in a file that was deleted or moved.

**Stop rule:** Clean up the claim or restore the file.

---

### L5_VERDICT_CONSISTENT

**Falsifier:** A `verdict=negative` claim has no `falsifier` or no `gate`.

**Stop rule:** A negative claim without falsifier is not a falsifiable negative; it is opinion.

---

### L6_SEED_RUPTURE

**Falsifier:** The scanner finds fewer than 8 rupture-negative claims in the seed file.

**Stop rule:** The seed is incomplete; encode all 8 negatives.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| Any claim is orphaned | `L_RED` | Do not commit until fixed. |
| A negative claim lacks falsifier | `L_RED` | Add falsifier or demote verdict. |
| Scanner crashes on a valid repo | `L_RED` | Fix scanner before proceeding. |
| Fewer than 6 clauses pass | `L_AMBER` | Narrow the claim set and re-run. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
