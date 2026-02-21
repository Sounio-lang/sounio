-- formal/lean4/SounioRowPoly.lean
import SounioEffects
/-!
# Sounio Row Polymorphism — Lean 4 Formalization

Formalizes row-polymorphic effect typing: effect row schemes with variables,
substitution, handler typing, and effect abstraction.

Builds on `SounioEffects` (concrete effect rows as `Effect → Bool`).

References:
- Leijen, D. (2014). "Koka: Programming with Row Polymorphic Effect Types." HOPE.
- Hillerström, D. and Lindley, S. (2016). "Liberating Effects with Rows and Handlers." TyDe.
- Plotkin, G. and Pretnar, M. (2009). "Handlers of Algebraic Effects." ESOP.

## Design

Row schemes extend concrete effect rows with row variables, enabling
universal quantification over unknown tails (`∀ ρ. τ →[IO ∪ ρ] σ`).
Substitution maps row variables to concrete rows, and all key properties
(subrow, handler correctness) are preserved under substitution.

No sorry. No Mathlib.
-/

open Sounio.Effects

namespace Sounio.RowPoly

-- ================================================================
-- §1. Row Variables and Schemes
-- ================================================================

/-- A row variable identifier. -/
abbrev RowVar := Nat

/-- A row scheme: an effect row expression that may contain variables.
    Extends `EffectRow` with universal quantification support. -/
inductive RowScheme where
  | concrete : EffectRow → RowScheme
  | var      : RowVar → RowScheme
  | union    : RowScheme → RowScheme → RowScheme
  | masked   : RowScheme → Effect → RowScheme

-- ================================================================
-- §2. Row Substitution
-- ================================================================

/-- A substitution maps row variables to concrete effect rows. -/
abbrev RowSubst := RowVar → EffectRow

/-- Apply a substitution to a row scheme, producing a concrete effect row. -/
def RowScheme.apply (σ : RowSubst) : RowScheme → EffectRow
  | .concrete r => r
  | .var v      => σ v
  | .union s₁ s₂ => rowUnion (s₁.apply σ) (s₂.apply σ)
  | .masked s e  => mask (s.apply σ) e

/-- The identity substitution maps every variable to the pure row. -/
def idSubst : RowSubst := fun _ => pureRow

-- ================================================================
-- §3. Substitution Properties
-- ================================================================

/-- Applying any substitution to a concrete row returns that row unchanged. -/
theorem apply_concrete (σ : RowSubst) (r : EffectRow) :
    (RowScheme.concrete r).apply σ = r := rfl

/-- Applying a substitution to a variable returns the substitution's value. -/
theorem apply_var (σ : RowSubst) (v : RowVar) :
    (RowScheme.var v).apply σ = σ v := rfl

/-- Applying a substitution distributes over union. -/
theorem apply_union (σ : RowSubst) (s₁ s₂ : RowScheme) :
    (RowScheme.union s₁ s₂).apply σ = rowUnion (s₁.apply σ) (s₂.apply σ) := rfl

/-- Applying a substitution distributes over masking. -/
theorem apply_masked (σ : RowSubst) (s : RowScheme) (e : Effect) :
    (RowScheme.masked s e).apply σ = mask (s.apply σ) e := rfl

-- ================================================================
-- §4. Identity Substitution
-- ================================================================

/-- The identity substitution sends variables to the pure row;
    a concrete scheme is unchanged. -/
theorem apply_id_concrete (r : EffectRow) :
    (RowScheme.concrete r).apply idSubst = r := rfl

/-- A variable under idSubst yields pureRow. -/
theorem apply_id_var (v : RowVar) :
    (RowScheme.var v).apply idSubst = pureRow := rfl

-- ================================================================
-- §5. Subrow Lifting
-- ================================================================

/-- If scheme s₁ ⊆ s₂ under all substitutions, we call it a scheme subrow. -/
def schemeSubrow (s₁ s₂ : RowScheme) : Prop :=
  ∀ σ : RowSubst, effectSubrow (s₁.apply σ) (s₂.apply σ)

/-- Scheme subrow is reflexive. -/
theorem schemeSubrow_refl (s : RowScheme) : schemeSubrow s s :=
  fun σ => effectSubrow_refl (s.apply σ)

/-- Scheme subrow is transitive. -/
theorem schemeSubrow_trans (s₁ s₂ s₃ : RowScheme)
    (h₁ : schemeSubrow s₁ s₂) (h₂ : schemeSubrow s₂ s₃) :
    schemeSubrow s₁ s₃ :=
  fun σ => effectSubrow_trans _ _ _ (h₁ σ) (h₂ σ)

/-- A concrete pure scheme is a subrow of anything. -/
theorem schemeSubrow_pure (s : RowScheme) :
    schemeSubrow (.concrete pureRow) s :=
  fun σ => pure_is_subrow (s.apply σ)

/-- Union is an upper bound on the left scheme. -/
theorem schemeSubrow_union_left (s₁ s₂ : RowScheme) :
    schemeSubrow s₁ (.union s₁ s₂) :=
  fun σ => effectSubrow_union_left (s₁.apply σ) (s₂.apply σ)

/-- Union is an upper bound on the right scheme. -/
theorem schemeSubrow_union_right (s₁ s₂ : RowScheme) :
    schemeSubrow s₂ (.union s₁ s₂) :=
  fun σ => effectSubrow_union_right (s₁.apply σ) (s₂.apply σ)

-- ================================================================
-- §6. Handler Typing as Row Transformation
-- ================================================================

/-- A handler for effect `e` transforms a row scheme by masking `e`.
    This models: `handle e with { ... } in body`. -/
def handlerTransform (s : RowScheme) (e : Effect) : RowScheme :=
  .masked s e

/-- Handler transformation always reduces effects (for any substitution). -/
theorem handler_reduces_scheme (s : RowScheme) (e : Effect) :
    schemeSubrow (handlerTransform s e) s :=
  fun σ => handler_reduces_effects (s.apply σ) e

/-- Two handlers for different effects commute at the scheme level. -/
theorem handler_comm_scheme (s : RowScheme) (e₁ e₂ : Effect) :
    ∀ σ : RowSubst,
      (handlerTransform (handlerTransform s e₁) e₂).apply σ =
      (handlerTransform (handlerTransform s e₂) e₁).apply σ := by
  intro σ
  simp only [handlerTransform, RowScheme.apply]
  exact mask_comm (s.apply σ) e₁ e₂

/-- Applying a handler twice for the same effect is idempotent. -/
theorem handler_idempotent_scheme (s : RowScheme) (e : Effect) :
    ∀ σ : RowSubst,
      (handlerTransform (handlerTransform s e) e).apply σ =
      (handlerTransform s e).apply σ := by
  intro σ
  simp only [handlerTransform, RowScheme.apply]
  exact mask_idempotent (s.apply σ) e

-- ================================================================
-- §7. Effect Polymorphism
-- ================================================================

/-- An effect-polymorphic property holds for all instantiations of a row variable. -/
def effectPoly (P : EffectRow → Prop) (s : RowScheme) : Prop :=
  ∀ σ : RowSubst, P (s.apply σ)

/-- If P holds for all rows, it holds for any scheme. -/
theorem effectPoly_of_forall (P : EffectRow → Prop)
    (h : ∀ r : EffectRow, P r) (s : RowScheme) :
    effectPoly P s :=
  fun σ => h (s.apply σ)

/-- Purity is preserved: a concrete pure scheme is pure under any substitution. -/
theorem effectPoly_pure :
    effectPoly (fun r => r = pureRow) (.concrete pureRow) :=
  fun _ => rfl

-- ================================================================
-- §8. Row Scheme Instantiation
-- ================================================================

/-- Pointwise substitution: replace variable v with row r. -/
def singleSubst (v : RowVar) (r : EffectRow) : RowSubst :=
  fun w => if w = v then r else pureRow

/-- Single substitution for the target variable yields the given row. -/
theorem singleSubst_hit (v : RowVar) (r : EffectRow) :
    singleSubst v r v = r := by
  simp [singleSubst]

/-- Single substitution for a different variable yields pureRow. -/
theorem singleSubst_miss (v w : RowVar) (r : EffectRow) (hne : w ≠ v) :
    singleSubst v r w = pureRow := by
  simp [singleSubst, hne]

/-- Instantiating a variable scheme with singleSubst returns the substituted row. -/
theorem instantiate_var (v : RowVar) (r : EffectRow) :
    (RowScheme.var v).apply (singleSubst v r) = r := by
  simp [RowScheme.apply, singleSubst]

-- ================================================================
-- §9. Scheme Union Properties
-- ================================================================

/-- Scheme union commutes under all substitutions. -/
theorem scheme_union_comm (s₁ s₂ : RowScheme) :
    ∀ σ, (RowScheme.union s₁ s₂).apply σ = (RowScheme.union s₂ s₁).apply σ := by
  intro σ
  simp only [RowScheme.apply]
  exact rowUnion_comm (s₁.apply σ) (s₂.apply σ)

/-- Scheme union associates under all substitutions. -/
theorem scheme_union_assoc (s₁ s₂ s₃ : RowScheme) :
    ∀ σ, (RowScheme.union (RowScheme.union s₁ s₂) s₃).apply σ =
         (RowScheme.union s₁ (RowScheme.union s₂ s₃)).apply σ := by
  intro σ
  simp only [RowScheme.apply]
  exact rowUnion_assoc (s₁.apply σ) (s₂.apply σ) (s₃.apply σ)

/-- Union with concrete pureRow on the left is identity under any substitution. -/
theorem scheme_union_pure_left (s : RowScheme) :
    ∀ σ, (RowScheme.union (.concrete pureRow) s).apply σ = s.apply σ := by
  intro σ
  simp only [RowScheme.apply]
  exact rowUnion_pure_left (s.apply σ)

-- ================================================================
-- §10. Masking and Instantiation Interaction
-- ================================================================

/-- Masking commutes with substitution (it's a concrete operation). -/
theorem mask_apply_comm (s : RowScheme) (e : Effect) (σ : RowSubst) :
    (RowScheme.masked s e).apply σ = mask (s.apply σ) e := rfl

/-- Masking a concrete singleton for its own effect yields pure. -/
theorem mask_concrete_single (e : Effect) :
    ∀ σ, (RowScheme.masked (.concrete (singleRow e)) e).apply σ = pureRow := by
  intro σ
  simp only [RowScheme.apply]
  exact single_mask_pure e

/-- Union of a variable with a singleton, then masking, yields just the variable. -/
theorem handler_strips_added_effect (v : RowVar) (e : Effect) :
    ∀ σ, ¬(e ∈ᵣ σ v) →
      (RowScheme.masked (RowScheme.union (.var v) (.concrete (singleRow e))) e).apply σ = σ v := by
  intro σ habs
  simp only [RowScheme.apply]
  rw [mask_union_right (σ v) (singleRow e) e (by
    cases h : σ v e
    · rfl
    · exact absurd h habs)]
  rw [single_mask_pure e]
  exact rowUnion_pure_right (σ v)

end Sounio.RowPoly
