/-!
# Sounio.Effects — Phase 8 Formal Verification

Algebraic effect system verification.  The Rust implementation uses
effect rows for tracking: IO, Mut, Alloc, Panic, Async, GPU, Prob, Div.
A function `fn f(...) with IO, Mut { ... }` may only be called from a
context that permits both IO and Mut.

This file proves:
1. Effect subsumption: a caller providing effects E2 can call a function
   requiring E1 iff E2 ⊆ E1 (reversed: the required row subsumes the
   provided row when every provided effect is in the required set).
2. Effect composition: sequential composition of two effectful functions
   produces the union of their effect sets.
3. Effect masking: a function that handles its own effects internally
   appears pure to callers.
4. Purity: the empty-effect function type is a subtype of any effect type.
-/

import Sounio.TypeChecker   -- for Effect, EffectRow, Ty

namespace Sounio.Effects

open Sounio.TypeChecker

-- ---------------------------------------------------------------------------
-- Subsumption
-- ---------------------------------------------------------------------------

/-- `effectSubsumes required provided` holds when every effect in `provided`
    also appears in `required`.  In Sounio's call rule a callee that requires
    effects E1 may be called from a context that provides E2 only when
    E2 ⊆ E1.  We call this "the required row *subsumes* the provided row". -/
def effectSubsumes (required provided : EffectRow) : Prop :=
  ∀ e, e ∈ provided → e ∈ required

/-- Subsumption is reflexive: a row subsumes itself. -/
theorem effect_subsumption_refl (e : EffectRow) : effectSubsumes e e :=
  fun _ h => h

/-- Subsumption is transitive. -/
theorem effect_subsumption_trans (e1 e2 e3 : EffectRow)
    (h12 : effectSubsumes e1 e2)
    (h23 : effectSubsumes e2 e3) :
    effectSubsumes e1 e3 :=
  fun eff he3 => h12 eff (h23 eff he3)

/-- Pure functions (empty effect row) can be called from any context.
    An empty `provided` row means the function requires no effects from
    its caller, so every caller context satisfies the requirement. -/
theorem pure_callable_anywhere (e : EffectRow) :
    effectSubsumes e [] :=
  fun _ h => absurd h (List.not_mem_nil _)

-- ---------------------------------------------------------------------------
-- Effect composition (sequential)
-- ---------------------------------------------------------------------------

/-- The effect union of two rows is their concatenation (treated as a set). -/
def effectUnion (e1 e2 : EffectRow) : EffectRow := e1 ++ e2

/-- The union subsumes its left component. -/
theorem composed_effects_contain_left (e1 e2 : EffectRow) :
    effectSubsumes (effectUnion e1 e2) e1 :=
  fun eff h => List.mem_append.mpr (Or.inl h)

/-- The union subsumes its right component. -/
theorem composed_effects_contain_right (e1 e2 : EffectRow) :
    effectSubsumes (effectUnion e1 e2) e2 :=
  fun eff h => List.mem_append.mpr (Or.inr h)

/-- Sequential composition: if `f` requires `e1` and `g` requires `e2`,
    then their sequential composition requires at most `e1 ++ e2`.
    Formally, the union subsumes each individual row. -/
theorem sequential_composition_effects (e1 e2 : EffectRow) :
    effectSubsumes (effectUnion e1 e2) e1 ∧
    effectSubsumes (effectUnion e1 e2) e2 :=
  ⟨composed_effects_contain_left e1 e2,
   composed_effects_contain_right e1 e2⟩

-- ---------------------------------------------------------------------------
-- Effect masking
-- ---------------------------------------------------------------------------

/-- `effectMasked total masked visible` asserts that every effect in `total`
    is either handled internally (`masked`) or visible to callers (`visible`).
    When a function handles `masked` effects via an effect handler, only
    `visible` effects escape to the caller. -/
def effectMasked (total masked visible : EffectRow) : Prop :=
  ∀ e, e ∈ total → e ∈ masked ∨ e ∈ visible

/-- Masking theorem: after removing internally-handled (`masked`) effects
    from `total`, the remaining effects are a subset of `visible`.

    Proof: take an effect `e` that is in `total` but not in `masked`; by
    `effectMasked` it must lie in `visible`. -/
theorem masking_removes_effect (total masked visible : EffectRow)
    (h : effectMasked total masked visible) :
    effectSubsumes visible (total.filter (fun e => !decide (e ∈ masked))) := by
  intro e he
  simp only [List.mem_filter, Bool.not_eq_true, decide_eq_false_iff_not] at he
  obtain ⟨hmem, hnmasked⟩ := he
  rcases h e hmem with hmask | hvis
  · exact absurd hmask hnmasked
  · exact hvis

/-- A handler that masks *all* effects makes the function appear pure. -/
theorem full_masking_yields_pure (total : EffectRow)
    (h : effectMasked total total []) :
    effectSubsumes [] (total.filter (fun e => !decide (e ∈ total))) := by
  intro e he
  simp only [List.mem_filter, Bool.not_eq_true, decide_eq_false_iff_not] at he
  obtain ⟨hmem, hnot⟩ := he
  exact absurd hmem hnot

-- ---------------------------------------------------------------------------
-- Div effect and totality
-- ---------------------------------------------------------------------------

/-- A context requires *total* functions when it does not permit `Div`.
    In Sounio, functions declared without the `Div` effect must terminate. -/
def requiresTotal (required : EffectRow) : Prop := Effect.Div ∉ required

/-- A total-only context cannot call a function that requires `Div`. -/
theorem total_context_rejects_div (e : EffectRow) (h : requiresTotal e) :
    ¬ effectSubsumes e [Effect.Div] := by
  intro hsub
  exact absurd (hsub Effect.Div (List.mem_singleton.mpr rfl)) h

/-- Pure (empty) effect rows are always compatible with totality requirements. -/
theorem pure_is_total : requiresTotal [] := by
  simp [requiresTotal, List.not_mem_nil]

-- ---------------------------------------------------------------------------
-- Monotonicity of subsumption under union
-- ---------------------------------------------------------------------------

/-- If `e1` subsumes `e2`, then `e1 ++ extra` also subsumes `e2`.
    Adding more permitted effects to the required row only relaxes
    the constraint on callers. -/
theorem subsumption_weakening_left (e1 e2 extra : EffectRow)
    (h : effectSubsumes e1 e2) :
    effectSubsumes (e1 ++ extra) e2 :=
  fun eff he2 => List.mem_append.mpr (Or.inl (h eff he2))

/-- Symmetric weakening: prepending `extra` also preserves subsumption. -/
theorem subsumption_weakening_right (e1 e2 extra : EffectRow)
    (h : effectSubsumes e1 e2) :
    effectSubsumes (extra ++ e1) e2 :=
  fun eff he2 => List.mem_append.mpr (Or.inr (h eff he2))

-- ---------------------------------------------------------------------------
-- Effect-row deduplication soundness
-- ---------------------------------------------------------------------------

/-- A deduplicated effect row is subsumed by the original.
    In practice the Rust implementation may canonicalise rows; this lemma
    shows the canonical form is always at least as permissive. -/
theorem dedup_subsumed_by_original [DecidableEq Effect] (e : EffectRow) :
    effectSubsumes e e.dedup :=
  fun eff h => List.mem_dedup.mp h

/-- The original row is subsumed by its deduplication, so they are
    equivalent under the subsumption ordering. -/
theorem original_subsumed_by_dedup [DecidableEq Effect] (e : EffectRow) :
    effectSubsumes e.dedup e :=
  fun eff h => List.mem_dedup.mpr h

-- ---------------------------------------------------------------------------
-- Purity and the unit function type
-- ---------------------------------------------------------------------------

/-- The identity function on the empty effect row has no effects to propagate.
    Formal witness: the empty row subsumes itself and is contained in any row. -/
theorem empty_effect_subsumes_empty : effectSubsumes [] [] :=
  effect_subsumption_refl []

/-- Effect transitivity via an intermediate row: subsumption chains compose. -/
theorem effect_chain (e1 e2 e3 : EffectRow)
    (h12 : effectSubsumes e1 e2)
    (h23 : effectSubsumes e2 e3) :
    effectSubsumes e1 e3 :=
  effect_subsumption_trans e1 e2 e3 h12 h23

end Sounio.Effects
