-- formal/lean4/SounioSemantics.lean
import SounioLinear
import SounioEffects
import SounioTyping
/-!
# Sounio Operational Semantics — Lean 4 Formalization

Call-by-value small-step operational semantics for the linear effect calculus.

References:
- Wright, A. and Felleisen, M. (1994). "A Syntactic Approach to Type Soundness."
- Pierce, B. (2002). "Types and Programming Languages." MIT Press.

No sorry. No Mathlib.
-/

open Sounio.Linear Sounio.Effects Sounio.Typing

namespace Sounio.Semantics

-- ================================================================
-- §1. Value Predicate
-- ================================================================

/-- CBV values: lambda abstractions, pairs of values, boxed values. -/
inductive IsValue : Expr → Prop where
  | lam : IsValue (.lam x m τ e)
  | pair : IsValue e₁ → IsValue e₂ → IsValue (.pair e₁ e₂)
  | box : IsValue e → IsValue (.box e)

-- ================================================================
-- §2. Capture-Avoiding Substitution
-- ================================================================

/-- e[z := v]: replace free occurrences of z with v. -/
def subst (z : String) (v : Expr) : Expr → Expr
  | .var y       => if y = z then v else .var y
  | .lam y m τ e => if y = z then .lam y m τ e
                     else .lam y m τ (subst z v e)
  | .app f a     => .app (subst z v f) (subst z v a)
  | .pair e₁ e₂  => .pair (subst z v e₁) (subst z v e₂)
  | .letP x y e body =>
      .letP x y (subst z v e)
        (if x = z ∨ y = z then body else subst z v body)
  | .box e       => .box (subst z v e)
  | .letB x e body =>
      .letB x (subst z v e)
        (if x = z then body else subst z v body)

-- §2.1 Substitution properties

theorem subst_var_hit (z : String) (v : Expr) :
    subst z v (.var z) = v := by
  simp [subst]

theorem subst_var_miss (z y : String) (v : Expr) (hne : y ≠ z) :
    subst z v (.var y) = .var y := by
  simp [subst, hne]

theorem subst_lam_shadow (z : String) (v : Expr) (m : Modality) (τ : Ty) (e : Expr) :
    subst z v (.lam z m τ e) = .lam z m τ e := by
  simp [subst]

theorem subst_app (z : String) (v f a : Expr) :
    subst z v (.app f a) = .app (subst z v f) (subst z v a) :=
  rfl

theorem subst_pair (z : String) (v e₁ e₂ : Expr) :
    subst z v (.pair e₁ e₂) = .pair (subst z v e₁) (subst z v e₂) :=
  rfl

theorem subst_box (z : String) (v e : Expr) :
    subst z v (.box e) = .box (subst z v e) :=
  rfl

-- ================================================================
-- §3. Small-Step CBV Reduction
-- ================================================================

/-- Call-by-value small-step reduction.
    Left-to-right evaluation; beta only with values. -/
inductive Step : Expr → Expr → Prop where
  | beta : IsValue v →
      Step (.app (.lam x m τ e) v) (subst x v e)
  | letP_beta : IsValue v₁ → IsValue v₂ →
      Step (.letP x y (.pair v₁ v₂) body) (subst y v₂ (subst x v₁ body))
  | letB_beta : IsValue v →
      Step (.letB x (.box v) body) (subst x v body)
  | app_left : Step f f' →
      Step (.app f a) (.app f' a)
  | app_right : IsValue f → Step a a' →
      Step (.app f a) (.app f a')
  | pair_left : Step e₁ e₁' →
      Step (.pair e₁ e₂) (.pair e₁' e₂)
  | pair_right : IsValue v₁ → Step e₂ e₂' →
      Step (.pair v₁ e₂) (.pair v₁ e₂')
  | box_step : Step e e' →
      Step (.box e) (.box e')
  | letP_step : Step e e' →
      Step (.letP x y e body) (.letP x y e' body)
  | letB_step : Step e e' →
      Step (.letB x e body) (.letB x e' body)

-- ================================================================
-- §4. Value Stability
-- ================================================================

/-- Values are irreducible: no step is possible. -/
theorem value_irreducible {e : Expr} (hv : IsValue e) :
    ∀ e', ¬Step e e' := by
  induction hv with
  | lam => intro e' hs; cases hs
  | pair _ _ ih₁ ih₂ =>
    intro e' hs
    cases hs with
    | pair_left hs => exact ih₁ _ hs
    | pair_right _ hs => exact ih₂ _ hs
  | box _ ih =>
    intro e' hs
    cases hs with
    | box_step hs => exact ih _ hs

-- ================================================================
-- §5. Normal Forms
-- ================================================================

/-- An expression is in normal form if it cannot step. -/
def NormalForm (e : Expr) : Prop := ∀ e', ¬Step e e'

/-- Every value is a normal form. -/
theorem value_is_normal {e : Expr} (hv : IsValue e) : NormalForm e :=
  value_irreducible hv

/-- Lambda abstractions are normal forms. -/
theorem lam_normal (x : String) (m : Modality) (τ : Ty) (e : Expr) :
    NormalForm (.lam x m τ e) :=
  value_is_normal .lam

-- ================================================================
-- §6. Determinism
-- ================================================================

/-- Small-step CBV is deterministic: each term has at most one reduct. -/
theorem step_deterministic {e e₁ e₂ : Expr}
    (h₁ : Step e e₁) (h₂ : Step e e₂) : e₁ = e₂ := by
  induction h₁ generalizing e₂ with
  | beta hv =>
    cases h₂ with
    | beta _ => rfl
    | app_left hs => exact absurd hs (value_irreducible .lam _)
    | app_right _ hs => exact absurd hs (value_irreducible hv _)
  | letP_beta hv₁ hv₂ =>
    cases h₂ with
    | letP_beta _ _ => rfl
    | letP_step hs => exact absurd hs (value_irreducible (.pair hv₁ hv₂) _)
  | letB_beta hv =>
    cases h₂ with
    | letB_beta _ => rfl
    | letB_step hs => exact absurd hs (value_irreducible (.box hv) _)
  | app_left hs₁ ih =>
    cases h₂ with
    | beta hv => exact absurd hs₁ (value_irreducible .lam _)
    | app_left hs₂ => congr 1; exact ih hs₂
    | app_right hvf hs₂ => exact absurd hs₁ (value_irreducible hvf _)
  | app_right hvf hs₁ ih =>
    cases h₂ with
    | beta hv => exact absurd hs₁ (value_irreducible hv _)
    | app_left hs₂ => exact absurd hs₂ (value_irreducible hvf _)
    | app_right _ hs₂ => congr 1; exact ih hs₂
  | pair_left hs₁ ih =>
    cases h₂ with
    | pair_left hs₂ => congr 1; exact ih hs₂
    | pair_right hv₁ hs₂ => exact absurd hs₁ (value_irreducible hv₁ _)
  | pair_right hv₁ hs₁ ih =>
    cases h₂ with
    | pair_left hs₂ => exact absurd hs₂ (value_irreducible hv₁ _)
    | pair_right _ hs₂ => congr 1; exact ih hs₂
  | box_step hs₁ ih =>
    cases h₂ with
    | box_step hs₂ => congr 1; exact ih hs₂
  | letP_step hs₁ ih =>
    cases h₂ with
    | letP_beta hv₁ hv₂ => exact absurd hs₁ (value_irreducible (.pair hv₁ hv₂) _)
    | letP_step hs₂ => congr 1; exact ih hs₂
  | letB_step hs₁ ih =>
    cases h₂ with
    | letB_beta hv => exact absurd hs₁ (value_irreducible (.box hv) _)
    | letB_step hs₂ => congr 1; exact ih hs₂

-- ================================================================
-- §7. Multi-Step Reduction
-- ================================================================

/-- Reflexive-transitive closure of Step. -/
inductive MultiStep : Expr → Expr → Prop where
  | refl : MultiStep e e
  | step : Step e e' → MultiStep e' e'' → MultiStep e e''

/-- A single step lifts to multi-step. -/
theorem multistep_one {e e' : Expr} (h : Step e e') : MultiStep e e' :=
  .step h .refl

/-- Multi-step is transitive. -/
theorem multistep_trans {e₁ e₂ e₃ : Expr}
    (h₁ : MultiStep e₁ e₂) (h₂ : MultiStep e₂ e₃) : MultiStep e₁ e₃ := by
  induction h₁ with
  | refl => exact h₂
  | step hs _ ih => exact .step hs (ih h₂)

-- ================================================================
-- §8. Multi-Step Congruence
-- ================================================================

/-- Multi-step in function position of application. -/
theorem multistep_app_left {f f' : Expr} (a : Expr)
    (h : MultiStep f f') : MultiStep (.app f a) (.app f' a) := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.app_left hs) ih

/-- Multi-step in argument position (function is a value). -/
theorem multistep_app_right {a a' : Expr} (f : Expr) (hvf : IsValue f)
    (h : MultiStep a a') : MultiStep (.app f a) (.app f a') := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.app_right hvf hs) ih

/-- Multi-step in left component of a pair. -/
theorem multistep_pair_left {e₁ e₁' : Expr} (e₂ : Expr)
    (h : MultiStep e₁ e₁') : MultiStep (.pair e₁ e₂) (.pair e₁' e₂) := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.pair_left hs) ih

/-- Multi-step in right component of a pair (left is a value). -/
theorem multistep_pair_right {e₂ e₂' : Expr} (v₁ : Expr) (hv₁ : IsValue v₁)
    (h : MultiStep e₂ e₂') : MultiStep (.pair v₁ e₂) (.pair v₁ e₂') := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.pair_right hv₁ hs) ih

/-- Multi-step inside a box. -/
theorem multistep_box {e e' : Expr}
    (h : MultiStep e e') : MultiStep (.box e) (.box e') := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.box_step hs) ih

-- ================================================================
-- §9. Evaluation Sequences
-- ================================================================

/-- Full beta reduction: evaluate function, argument, then beta. -/
theorem eval_app_full {f a a' : Expr} {x : String} {m : Modality}
    {τ : Ty} {body : Expr}
    (hf : MultiStep f (.lam x m τ body))
    (ha : MultiStep a a') (hva : IsValue a') :
    MultiStep (.app f a) (subst x a' body) :=
  multistep_trans
    (multistep_app_left a hf)
    (multistep_trans
      (multistep_app_right _ .lam ha)
      (multistep_one (.beta hva)))

/-- Multi-step in scrutinee of letP. -/
theorem multistep_letP {e e' : Expr} (x y : String) (body : Expr)
    (h : MultiStep e e') : MultiStep (.letP x y e body) (.letP x y e' body) := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.letP_step hs) ih

/-- Full let-pair evaluation. -/
theorem eval_letP_full {e v₁ v₂ : Expr} {x y : String} {body : Expr}
    (he : MultiStep e (.pair v₁ v₂))
    (hv₁ : IsValue v₁) (hv₂ : IsValue v₂) :
    MultiStep (.letP x y e body) (subst y v₂ (subst x v₁ body)) :=
  multistep_trans (multistep_letP x y body he) (multistep_one (.letP_beta hv₁ hv₂))

/-- Multi-step in scrutinee of letB. -/
theorem multistep_letB {e e' : Expr} (x : String) (body : Expr)
    (h : MultiStep e e') : MultiStep (.letB x e body) (.letB x e' body) := by
  induction h with
  | refl => exact .refl
  | step hs _ ih => exact .step (.letB_step hs) ih

/-- Full let-box evaluation. -/
theorem eval_letB_full {e v : Expr} {x : String} {body : Expr}
    (he : MultiStep e (.box v))
    (hv : IsValue v) :
    MultiStep (.letB x e body) (subst x v body) :=
  multistep_trans (multistep_letB x body he) (multistep_one (.letB_beta hv))

end Sounio.Semantics
