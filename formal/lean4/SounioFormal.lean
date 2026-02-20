-- formal/lean4/SounioFormal.lean
import SounioLinear
import SounioEffects

/-!
# Sounio Formal Verification — Phase 8

Lean 4 formalization of key Sounio type-system components.
No sorry. No Mathlib dependency. `lake build` completes in < 1 minute.

## Scope

### `SounioLinear` — Four-modality linear type system (Girard 1987)

**Proven:**
- Modality lattice laws: commutativity, associativity, absorption, idempotence
- Linear is bottom; Unrestricted is top; Affine ∥ Relevant (incomparable)
- Structural rule properties: weakening/no-weakening/contraction per modality
- `weakening_monotone`: weakening is monotone in the lattice order
- `usage_add_*`: usage count arithmetic forms a commutative monoid
- `wellUsed` predicate with decidable instances
- `affine_zero_well_used` / `linear_zero_not_well_used` / etc.
- `weakening_lemma`: Affine/Unrestricted entries can be dropped
- `no_weakening_lemma`: Linear/Relevant entries force non-zero usage
- `promotion_exhausted_any`: all-Unrestricted contexts are always exhausted (Bang-I)

### `SounioEffects` — Algebraic effect rows (characteristic function model)

**Proven:**
- `mask_idempotent`: double-handling = single-handling
- `mask_comm`: distinct handlers commute
- `mask_absent_noop`: masking absent effect is identity
- `mask_removes` / `mask_preserves_other`: membership after masking
- `rowUnion_*`: union is commutative, associative, idempotent; pure row is identity
- `effectSubrow_*`: reflexivity, transitivity, antisymmetry (partial order)
- `handler_reduces_effects`: `mask r e ⊆ r`
- `handler_idempotent` / `handler_order_independence`
- `single_mask_pure`: masking sole effect yields pure row
- `effectSubrow_union_left/right/lub`: union is least upper bound
- `mask_union_right`: masking distributes over union when effect absent from left

## Not in scope for Phase 8

- Full typing judgment as an inductive relation (typing derivations)
- Operational semantics and progress/preservation theorem
- Epistemic type `Knowledge[T,ε]` (requires real-number analysis)
- Causal type system formalization
-/
