<!-- docs:meta
topic_id: repo.docs.research.zero-provenance-claims-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zero-provenance-claims-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Zero-provenance claims — zero-event taxonomy inside the Falsification Ledger

**Date:** 2026-07-25  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (target)  
**Parents:** `docs/research/falsification_ledger_spec_2026-07-25.md` (L_GREEN), `docs/internal/garden/seeds/2026-07-11-the-zero-of-encounter.md`, `stdlib/epistemic/zero_event.sio`  
**Harness:** `scripts/research/zero_provenance_claims_contract.py`  
**Gate:** `scripts/ci/zero_provenance_claims_gate.sh`  
**Seed:** `stdlib/epistemic/zero_provenance_claims.sio`

---

## 1. What this is

The Falsification Ledger records claims with hypothesis, falsifier, evidence, harness, gate, and verdict. This extension adds **provenance** for claims whose evidence involves a zero surface value. A zero is not one thing; it can be absent, cancelled, annihilated, below resolution, rounded, gated, or unknown. The ledger must not silently collapse these.

The provenance field is optional for ordinary claims but **required** when the claim's evidence or falsifier mentions a zero result.

---

## 2. Syntax extension

```sounio
// @claim zero_annihilation_sedenion
// @hypothesis the sedenions contain nonzero zero-divisor pairs (a,b) with ab = 0 exactly
// @falsifier every pair of nonzero sedenions has nonzero product
// @evidence instrument_controlled
// @harness stdlib/epistemic/zero_event.sio
// @gate scripts/ci/zero_event_gate.sh
// @verdict alive
// @provenance annihilated
```

### Provenance vocabulary

| Value | Meaning | Source flag |
|---|---|---|
| `absent` | No effect existed | `ze_flag_absent()` |
| `cancelled` | Sum of opposite effects | `ze_flag_cancelled()` |
| `annihilated` | Nonzero factors multiplied to zero | `ze_flag_annihilated()` |
| `below_resolution` | Latent value below measurement resolution | `ze_flag_below_resolution()` |
| `rounded` | Nonzero correction trail whose f64 lane rounds to zero | `ze_flag_rounded()` |
| `gated` | Original nonzero, suppressed by a gate | `ze_flag_gated()` |
| `unknown` | Provenance not determined | `ze_flag_unknown()` |

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **Z1_PROVENANCE_ENUM** | Every `@provenance` value is in the zero-event taxonomy. | Schema validation passes. |
| **Z2_ZERO_CLAIMS_REQUIRE_PROVENANCE** | Claims whose evidence or falsifier mentions a zero result have a `@provenance` field. | All zero-claims carry provenance. |
| **Z3_NONZERO_CLAIMS_OPTIONAL** | Non-zero claims may omit `@provenance`. | Omission allowed without failure. |
| **Z4_SEED_TAXONOMY** | The seed file contains at least one claim for each zero-event category. | 7 categories present. |
| **Z5_LEDGER_INCLUDES_PROVENANCE** | The emitted `.sounio/claims/*.jsonl` records the provenance field. | Ledger contains provenance keys. |
| **Z6_DISCHARGE_NOT_CLAIMED** | The contract does not claim causal or clinical interpretation of provenance. | No clinical text in claim fields. |

---

## 4. What this is NOT

- **Not a causal inference engine.** Provenance records what the computation path says, not what the world means.
- **Not a clinical interpretation.** `annihilated` is an algebraic category, not a diagnosis.
- **Not a parser change.** Provenance is still comment-scanned.
- **Not a replacement for `zero_event.sio`.** The stdlib module remains the executable witness; the ledger records the claim.

---

## 5. Reproduce

```bash
python3 scripts/research/zero_provenance_claims_contract.py
# expect: Z1..Z6 PASS, ZERO_PROVENANCE_CLAIMS_VERDICT Z_GREEN

bash scripts/ci/zero_provenance_claims_gate.sh
# expect: ZERO_PROVENANCE_CLAIMS_GATE_OK
```

---

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-25). No clinical content. GAIDeT-ICMJE 2025.
