<!-- docs:meta
topic_id: repo.docs.research.particle-nunwa-lean-2026-08-06
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-nunwa-lean-2026-08-06
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Lean leaf — NonUnitary × NWA composition (2026-08-06)

**Source:** `formal/lean4/SounioNonUnitaryNWA.lean`  
**Gate:** `scripts/ci/nunwa_lean_gate.sh` → `SOUNIO_NUNWA_LEAN_GATE_OK`  
**Mirrors:** `stdlib/particle_physics/nonunitary*.sio`, `approx_effects.sio`  
**Pattern:** `SounioApproxCausalKnowledge.lean` (handlers + Nat shadow)

## Claims (proved, no sorry)

1. `handler_commutativity` — discharging NonUnitary then NWA equals the reverse.
2. `nwaPeakDen_pos` / `nwaPeak_eq_div` — scaled peak `C·Γ_in·Γ_out/(M²·Γ_tot²)` is a well-defined Nat quotient when denominators are positive.
3. `continuum_ne_peak_possible` — continuum and NWA peak need not coincide (EXP honesty non-claim, formalized).
4. `dischargeToy_preserves_peak` — clearing tags does not alter Nat observables.

## Math-review

`bin/llm-offload -t math-review -p xai` (2026-08-06):

- Item 1 (NWA Nat peak shadow) **[OK]**
- Item 3 (continuum ≠ peak possibility) **[OK]**
- Item 2 flagged missing handler-interaction axioms — initially deferred to
  `SounioEffects.lean` §19 + this file §4 (2026-08-06 follow-up).

## Follow-up (2026-08-06): handler × payload interaction

- `SounioEffects.lean` §19 — `Effectful α`, `handle_preserves_value`,
  `handle_comm`, `discharge_pair_interaction`.
- This file §4 — `discharge_eq_mask_fold`, `discharge_interaction`
  (tags ↔ TagRow masks; observables preserved).
- Gate: `scripts/ci/effects_handler_interaction_lean_gate.sh`.
- Math-review (xai): items 1+3 **[OK]**; item 2 **[WARN]** on disjointness —
  **disagreement**: `mask_comm` holds for all pairs including `e₁=e₂`;
  item 4 **[WARN]** informal preservation — **disagreement**:
  `discharge_interaction` is the explicit five-conjunct lemma.

## Non-claims

- Not optical theorem / unitarity.
- Not Madaros GUM Var preserve (that is C1).
- Not Float Breit-Wigner; Nat is a discrete shadow.

## Reproduce

```bash
bash scripts/ci/nunwa_lean_gate.sh
# expect: SOUNIO_NUNWA_LEAN_GATE_OK
```
