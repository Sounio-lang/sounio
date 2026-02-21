-- formal/lean4/SounioFormal.lean
import SounioLinear
import SounioEffects
import SounioTyping

/-!
# Sounio Formal Verification — Phase 8 + 9

Lean 4 formalization of key Sounio type-system components.
No sorry. No Mathlib dependency. `lake build` completes in < 1 minute.

## Scope

### `SounioLinear` — Four-modality linear type system (Girard 1987)

**Proven:**
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

**Proven:**
- `mask_*`: idempotent, commutative, membership-preserving, distributive
- `rowUnion_*`: complete Boolean union algebra (comm, assoc, distrib, absorb, lub)
- `rowInter_*`: complete Boolean intersection algebra (+ GLB proof)
- `rowComplement_*`: De Morgan laws, involution, anti-monotonicity
- `allEffectsRow`: top element; identity laws for ∪ and ∩
- `rowDisjoint_*`: symmetry, pure cases, union monotonicity
- `maskAll_pure`: handling all 10 named effects yields pureRow
- `effectSubrow_*`: partial order (refl, trans, antisymm) + monotonicity

### `SounioTyping` — Linear Effect Typing Judgment (Phase 9)

**Defines:**
- `Ty`: types (`Base`, `→[m]`, `⊗`, `!`)
- `Expr`: expressions (λ, app, pair, let-pair, box, let-box)
- `Typing Γ e τ ρ`: inductive judgment `Γ ⊢ e : τ ! ρ`
  with nine constructors (Var, Lam, App, Pair, LetP, Box, LetB, Weak, Sub)

**Proven:**
- `typing_var_pure` / `typing_box_pure` / `typing_lam_effects` / `typing_app_union`
- `typing_sub` / `typing_pure_sub` / `typing_sub_trans`: effect subsumption
- `typing_app_effects_comm`: application effects commute up to Sub
- `typing_dereliction` / `typing_app_sub`: bang and composition
- `typing_weaken_one/affine/unrestricted`: single-entry weakening
- `typing_weaken_list`: list weakening (induction on Δ)
- `typing_box_letB` / `typing_box_letB_pure`: promotion + dereliction
- `typing_bang_weaken`: bang values are weakenable
- `typing_app_pure` / `typing_pair_pure` / `typing_pair_same_row`
- `typing_letP_pure_scrut`: tensor elimination with pure scrutinee
- `typing_unrestricted_app` / `typing_linear_app`
- `typing_identity`: `λ(x:τ)[m]. x` is typable at every type and modality
- `typing_const_fun`: discarding an argument requires weakenable modality

## Out of scope

- Operational semantics and progress/preservation
- Epistemic type `Knowledge[T,ε]` (requires real-number analysis)
- Causal type system formalization
- Type uniqueness / inversion (requires determinism of Sub+Weak)
-/
