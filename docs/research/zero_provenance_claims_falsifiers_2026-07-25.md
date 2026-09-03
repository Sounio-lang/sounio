<!-- docs:meta
topic_id: repo.docs.research.zero-provenance-claims-falsifiers-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zero-provenance-claims-falsifiers-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Zero-provenance claims — falsifiers and stop rules

**Companion to:** `docs/research/zero_provenance_claims_spec_2026-07-25.md`  
**Harness:** `scripts/research/zero_provenance_claims_contract.py`

---

## Clause-level falsifiers

### Z1_PROVENANCE_ENUM

**Falsifier:** A `@provenance` value is not one of the seven zero-event categories.

**Stop rule:** Reject the claim or map the value to `unknown`.

---

### Z2_ZERO_CLAIMS_REQUIRE_PROVENANCE

**Falsifier:** A claim whose evidence or falsifier mentions a zero result lacks `@provenance`.

**Stop rule:** The ledger collapses distinct zeros; fix the claim.

---

### Z3_NONZERO_CLAIMS_OPTIONAL

**Falsifier:** A non-zero claim fails because `@provenance` is missing.

**Stop rule:** The contract is over-strict; make provenance optional for non-zero claims.

---

### Z4_SEED_TAXONOMY

**Falsifier:** The seed file has fewer than 7 zero-event categories.

**Stop rule:** Complete the seed.

---

### Z5_LEDGER_INCLUDES_PROVENANCE

**Falsifier:** The emitted ledger omits the provenance field.

**Stop rule:** Fix the scanner/serializer.

---

### Z6_DISCHARGE_NOT_CLAIMED

**Falsifier:** A claim field contains causal or clinical interpretation of provenance.

**Stop rule:** Quarantine the claim; provenance is algebraic, not causal.

---

## Global stop rules

| Trigger | Verdict | Action |
|---|---|---|
| Z1 or Z2 fails | `Z_RED` | Ledger is losing provenance; stop. |
| Z6 fails | `Z_RED` | Causal leakage; remove clinical text. |
| Two or more clauses fail | `Z_AMBER` | Narrow the claim set. |

---

## AI disclosure

Falsifiers drafted under human direction (2026-07-25). GAIDeT-ICMJE 2025.
