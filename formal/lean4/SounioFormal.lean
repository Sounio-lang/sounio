-- formal/lean4/SounioFormal.lean
import SounioLinear
import SounioEffects
import SounioTyping
import SounioUnits
import SounioRowPoly

/-!
# Sounio Formal Verification — Phase 8–10

Lean 4 formalization of key Sounio type-system components.
No sorry. No Mathlib dependency. `lake build` completes in < 1 minute.

## Scope

### `SounioLinear` — Four-modality linear type system (Girard 1987)

**Proven (40 theorems):**
- Modality lattice laws: commutativity, associativity, absorption, idempotence,
  distributivity
- Linear is bottom; Unrestricted is top; Affine ∥ Relevant (incomparable)
- Structural rule properties: weakening/no-weakening/contraction per modality
- Monotonicity: weakening/contraction/mustUse preserved under join/meet
- `usage_add_*`: usage count arithmetic (absorb, iff characterizations)
- `wellUsed` predicate: all four modalities × all three usage counts
- `weakening_lemma` / `no_weakening_lemma`: structural rule admissibility
- `promotion_exhausted_any`: all-Unrestricted contexts always exhausted (Bang-I)

### `SounioEffects` — Algebraic effect rows (characteristic function model)

**Proven (50 theorems):**
- `mask_*`: idempotent, commutative, membership-preserving, distributive
- `rowUnion_*`: complete Boolean union algebra (comm, assoc, distrib, absorb, lub)
- `rowInter_*`: complete Boolean intersection algebra (+ GLB proof)
- `rowComplement_*`: De Morgan laws, involution, anti-monotonicity
- `allEffectsRow`: top element; identity laws for ∪ and ∩
- `rowDisjoint_*`: symmetry, pure cases, union monotonicity
- `maskAll_pure`: handling all 10 named effects yields pureRow
- `effectSubrow_*`: partial order (refl, trans, antisymm) + monotonicity

### `SounioTyping` — Linear Effect Typing Judgment (Phase 9–10)

**Defines:**
- `Ty`: types (`Base`, `→[m]`, `⊗`, `!`)
- `Expr`: expressions (λ, app, pair, let-pair, box, let-box)
- `Typing Γ e τ ρ`: inductive judgment `Γ ⊢ e : τ ! ρ`
  with nine constructors (Var, Lam, App, Pair, LetP, Box, LetB, Weak, Sub)

**Proven (38 theorems):**
- `typing_var_pure` / `typing_box_pure` / `typing_lam_effects` / `typing_app_union`
- `typing_sub` / `typing_pure_sub` / `typing_sub_trans`: effect subsumption
- `typing_app_effects_comm`: application effects commute up to Sub
- `typing_dereliction` / `typing_app_sub`: bang and composition
- `typing_weaken_one/affine/unrestricted`: single-entry weakening
- `typing_weaken_list` / `typing_weaken_two` / `typing_weaken_unrestricted_list`
- `typing_box_letB` / `typing_box_letB_pure` / `typing_box_letB_id`
- `typing_bang_weaken` / `typing_pair_bang`
- `typing_app_pure` / `typing_pair_pure` / `typing_pair_same_row`
- `typing_letP_pure_scrut` / `typing_letP_same_row`
- `typing_mask_absent` / `typing_pure_mask`: effect masking in typing
- `typing_double_app` / `typing_nested_letB`: compound expressions
- `typing_app_sub_both`: effect subsumption composes with application
- `typing_identity` / `typing_affine_identity` / `typing_relevant_identity`
- `typing_const_fun`: discarding an argument requires weakenable modality

### `SounioUnits` — Dimensional Analysis (Phase 10)

**Defines:**
- `Dim`: 7-tuple of SI base quantity exponents (M, L, T, I, Θ, N, J)
- Operations: `mul`, `div`, `recip`, `pow`
- Named dimensions: 7 base + 11 derived (velocity, force, energy, ...)
- `compatible`: dimensional equality predicate

**Proven (38 theorems):**
- Abelian group laws: `dim_mul_comm/assoc`, `dimless_mul_left/right`,
  `dim_mul_recip`, `dim_recip_mul`
- Division: `dim_div_self`, `dim_mul_div_cancel`, `dim_div_mul_cancel`,
  `dim_div_eq_mul_recip`
- Reciprocal: `dim_recip_involution`, `dim_recip_dimless`,
  `dim_recip_mul_distrib`
- Powers: `dim_pow_zero/one/add/neg/mul`
- Derived dimension verifications: velocity, acceleration, force, energy,
  power, pressure, frequency, concentration, clearance, area, volume
- Compatibility: reflexive, symmetric, transitive (equivalence relation)
- Safety: `mul_div_roundtrip`, `div_mul_roundtrip`, `mul_recip_dimless`,
  `sq_recip_sq_dimless`

### `SounioRowPoly` — Row-Polymorphic Effect Typing (Phase 10)

**Defines:**
- `RowScheme`: effect row expressions with variables (concrete/var/union/masked)
- `RowSubst`: substitution mapping row variables to concrete rows
- `schemeSubrow`: universally-quantified subrow relation
- `handlerTransform`: handler as row masking
- `effectPoly`: effect-polymorphic property quantifier
- `singleSubst`: pointwise single-variable substitution

**Proven (25 theorems):**
- Substitution: `apply_concrete/var/union/masked` (compositional)
- Identity: `apply_id_concrete/var`
- Subrow lifting: `schemeSubrow_refl/trans/pure`, `schemeSubrow_union_left/right`
- Handler schemes: `handler_reduces_scheme`, `handler_comm_scheme`,
  `handler_idempotent_scheme`
- Polymorphism: `effectPoly_of_forall`, `effectPoly_pure`
- Instantiation: `singleSubst_hit/miss`, `instantiate_var`
- Scheme union: `scheme_union_comm/assoc`, `scheme_union_pure_left`
- Masking interaction: `mask_apply_comm`, `mask_concrete_single`,
  `handler_strips_added_effect`

## Out of scope

- Operational semantics and progress/preservation
- Epistemic type `Knowledge[T,ε]` (requires real-number analysis)
- Causal type system formalization
- Type uniqueness / inversion (requires determinism of Sub+Weak)
- Unit inference / unification (requires substitution model over Dim)
-/
