-- formal/lean4/SounioPreservation.lean
import SounioLinear
import SounioEffects
import SounioTyping
import SounioSemantics
/-!
# Sounio Preservation & Type Safety — Lean 4 Formalization

Safety algebra for the Sounio linear effect calculus:
- Stuck / Safe predicates and their algebra
- Deterministic evaluation (EvaluatesTo) and uniqueness
- Counted multi-step reduction (MultiStepN)
- Preservation framework (Preserves combinator)
- Parametric type safety (assuming single-step preservation and progress)
- Value stability under reduction
- Effect-preservation composition theorems

The full subject-reduction (preservation) theorem requires a substitution
lemma that is being developed in `SounioSubstitution.lean` concurrently.
This file provides the safety infrastructure that composes with that lemma
once available, and proves all properties that are self-contained.

References:
- Wright, A. and Felleisen, M. (1994). "A Syntactic Approach to Type Soundness."
- Pierce, B. (2002). "Types and Programming Languages." MIT Press, Ch. 8.

No sorry. No Mathlib.
-/

open Sounio.Linear Sounio.Effects Sounio.Typing Sounio.Semantics

namespace Sounio.Preservation

-- ================================================================
-- §1. Stuck and Safe Predicates
-- ================================================================

/-- An expression is stuck if it is not a value and cannot step. -/
def Stuck (e : Expr) : Prop := ¬IsValue e ∧ NormalForm e

/-- An expression is safe if it is not stuck. -/
def Safe (e : Expr) : Prop := ¬Stuck e

/-- Every value is safe. -/
theorem value_safe {e : Expr} (hv : IsValue e) : Safe e :=
  fun ⟨hnv, _⟩ => hnv hv

/-- Every expression that can step is safe. -/
theorem stepping_safe {e e' : Expr} (hs : Step e e') : Safe e :=
  fun ⟨_, hnf⟩ => hnf e' hs

/-- Safe expressions satisfy the progress property: value or steppable. -/
theorem safe_dichotomy {e : Expr} (h : Safe e) : IsValue e ∨ ∃ e', Step e e' :=
  Classical.byContradiction fun hc =>
    h ⟨fun hv => hc (Or.inl hv), fun e' hs => hc (Or.inr ⟨e', hs⟩)⟩

/-- An open variable is stuck. -/
theorem var_stuck (x : String) : Stuck (.var x) := by
  constructor
  · intro hv; cases hv
  · intro e' hs; cases hs

-- ================================================================
-- §2. Deterministic Evaluation
-- ================================================================

/-- Deterministic evaluation to a value. -/
def EvaluatesTo (e v : Expr) : Prop := MultiStep e v ∧ IsValue v

/-- Every value evaluates to itself. -/
theorem value_evaluatesTo_self {e : Expr} (hv : IsValue e) : EvaluatesTo e e :=
  ⟨.refl, hv⟩

/-- If a value multi-steps, it does not change. -/
theorem value_multistep_id {e e' : Expr} (hv : IsValue e) (hms : MultiStep e e') :
    e = e' := by
  induction hms with
  | refl => rfl
  | step hs _ _ => exact absurd hs (value_irreducible hv _)

/-- Evaluation is deterministic: if e evaluates to both v1 and v2, they are equal. -/
theorem evaluatesTo_unique {e v₁ v₂ : Expr}
    (h₁ : EvaluatesTo e v₁) (h₂ : EvaluatesTo e v₂) : v₁ = v₂ := by
  obtain ⟨hms₁, hv₁⟩ := h₁
  obtain ⟨hms₂, hv₂⟩ := h₂
  induction hms₁ with
  | refl =>
    cases hms₂ with
    | refl => rfl
    | step hs _ => exact absurd hs (value_irreducible hv₁ _)
  | step hs₁ _ ih =>
    cases hms₂ with
    | refl => exact absurd hs₁ (value_irreducible hv₂ _)
    | step hs₂ hms₂' =>
      have heq := step_deterministic hs₁ hs₂
      subst heq
      exact ih hv₁ hms₂'

/-- If e evaluates to v, and e also multi-steps to e', then e' multi-steps to v. -/
theorem evaluatesTo_multistep_diamond {e v e' : Expr}
    (hev : EvaluatesTo e v) (hms : MultiStep e e') :
    MultiStep e' v := by
  obtain ⟨hmsv, hv⟩ := hev
  induction hms with
  | refl => exact hmsv
  | step hs _ ih =>
    cases hmsv with
    | refl => exact absurd hs (value_irreducible hv _)
    | step hs' hmsv' =>
      have heq := step_deterministic hs hs'
      subst heq
      exact ih hmsv'

-- ================================================================
-- §3. Counted Multi-Step Reduction (MultiStepN)
-- ================================================================

/-- Multi-step reduction with an explicit step count. -/
inductive MultiStepN : Nat → Expr → Expr → Prop where
  | zero : MultiStepN 0 e e
  | succ : Step e e' → MultiStepN n e' e'' → MultiStepN (n + 1) e e''

/-- A counted multi-step lifts to the uncounted version. -/
theorem multiStepN_to_multiStep {n : Nat} {e e' : Expr}
    (h : MultiStepN n e e') : MultiStep e e' := by
  induction h with
  | zero => exact .refl
  | succ hs _ ih => exact .step hs ih

/-- Zero steps means identity. -/
theorem multiStepN_zero_eq {e e' : Expr} (h : MultiStepN 0 e e') : e = e' := by
  cases h; rfl

/-- Counted multi-step is transitive with step-count addition. -/
theorem multiStepN_trans {n m : Nat} {e₁ e₂ e₃ : Expr}
    (h₁ : MultiStepN n e₁ e₂) (h₂ : MultiStepN m e₂ e₃) :
    MultiStepN (n + m) e₁ e₃ := by
  induction h₁ with
  | zero =>
    simp only [Nat.zero_add]
    exact h₂
  | succ hs _ ih =>
    simp only [Nat.add_succ, Nat.succ_add]
    exact .succ hs (ih h₂)

/-- A value cannot take any positive number of steps. -/
theorem multiStepN_value_zero {n : Nat} {e e' : Expr}
    (h : MultiStepN n e e') (hv : IsValue e) : n = 0 ∧ e = e' := by
  cases h with
  | zero => exact ⟨rfl, rfl⟩
  | succ hs _ => exact absurd hs (value_irreducible hv _)

/-- One counted step. -/
theorem multiStepN_one {e e' : Expr} (hs : Step e e') : MultiStepN 1 e e' :=
  .succ hs .zero

/-- Determinism for counted multi-step: same count, same start, same result. -/
theorem multiStepN_deterministic {n : Nat} {e e₁ e₂ : Expr}
    (h₁ : MultiStepN n e e₁) (h₂ : MultiStepN n e e₂) : e₁ = e₂ := by
  induction h₁ generalizing e₂ with
  | zero => cases h₂; rfl
  | succ hs₁ _ ih =>
    cases h₂ with
    | succ hs₂ hms₂ =>
      have heq := step_deterministic hs₁ hs₂
      subst heq
      exact ih hms₂

-- ================================================================
-- §4. Preserves Combinator
-- ================================================================

/-- A property P is preserved under single-step reduction. -/
def Preserves (P : Expr → Prop) : Prop :=
  ∀ e e', P e → Step e e' → P e'

/-- Any preserved property extends to multi-step reduction. -/
theorem preserves_multistep {P : Expr → Prop} (hp : Preserves P)
    {e e' : Expr} (he : P e) (hms : MultiStep e e') : P e' := by
  induction hms with
  | refl => exact he
  | step hs _ ih => exact ih (hp _ _ he hs)

/-- Any preserved property extends to counted multi-step reduction. -/
theorem preserves_multiStepN {P : Expr → Prop} (hp : Preserves P)
    {n : Nat} {e e' : Expr} (he : P e) (hms : MultiStepN n e e') : P e' := by
  induction hms with
  | zero => exact he
  | succ hs _ ih => exact ih (hp _ _ he hs)

/-- The conjunction of two preserved properties is preserved. -/
theorem preserves_and {P Q : Expr → Prop} (hp : Preserves P) (hq : Preserves Q) :
    Preserves (fun e => P e ∧ Q e) := by
  intro e e' ⟨hpe, hqe⟩ hs
  exact ⟨hp e e' hpe hs, hq e e' hqe hs⟩

/-- A preserved property holds at the end of evaluation. -/
theorem preserves_evaluatesTo {P : Expr → Prop} (hp : Preserves P)
    {e v : Expr} (he : P e) (hev : EvaluatesTo e v) : P v :=
  preserves_multistep hp he hev.1

/-- A universally-true property is trivially preserved. -/
theorem preserves_true : Preserves (fun _ => True) := by
  intro _ _ _ _; trivial

/-- The false property is vacuously preserved. -/
theorem preserves_false : Preserves (fun _ => False) := by
  intro _ _ h _; exact h

/-- IsValue is preserved under stepping (vacuously: values don't step). -/
theorem preserves_isValue : Preserves IsValue := by
  intro e _ hv hs
  exact absurd hs (value_irreducible hv _)

-- ================================================================
-- §5. Single-Step Preservation (Parametric)
-- ================================================================

/-- Single-step preservation: the typing judgment is preserved under reduction.
    This is the key hypothesis for type safety. -/
def SingleStepPres : Prop :=
  ∀ {Γ : TyCtx} {e e' : Expr} {τ : Ty} {ρ : EffectRow},
    Typing Γ e τ ρ → Step e e' → Typing Γ e' τ ρ

/-- If single-step preservation holds, typing is preserved under multi-step. -/
theorem multistep_preservation (hpres : SingleStepPres)
    {Γ : TyCtx} {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ ρ) (hms : MultiStep e e') :
    Typing Γ e' τ ρ := by
  induction hms with
  | refl => exact ht
  | step hs _ ih => exact ih (hpres ht hs)

/-- If single-step preservation holds, typing is preserved under counted multi-step. -/
theorem multiStepN_preservation (hpres : SingleStepPres)
    {n : Nat} {Γ : TyCtx} {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ ρ) (hms : MultiStepN n e e') :
    Typing Γ e' τ ρ := by
  induction hms with
  | zero => exact ht
  | succ hs _ ih => exact ih (hpres ht hs)

/-- If single-step preservation holds, evaluation produces a well-typed value. -/
theorem evaluation_preservation (hpres : SingleStepPres)
    {Γ : TyCtx} {e v : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ ρ) (hev : EvaluatesTo e v) :
    Typing Γ v τ ρ :=
  multistep_preservation hpres ht hev.1

-- ================================================================
-- §6. Parametric Type Safety
-- ================================================================

/-- The progress property for a typing context Gamma. -/
def ProgressFor (Γ : TyCtx) : Prop :=
  ∀ {e : Expr} {τ : Ty} {ρ : EffectRow},
    Typing Γ e τ ρ → IsValue e ∨ ∃ e', Step e e'

/-- Type safety: if preservation and progress hold for the empty context,
    closed well-typed terms never get stuck during evaluation. -/
theorem type_safety (hpres : SingleStepPres) (hprog : ProgressFor [])
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStep e e') :
    ¬Stuck e' := by
  have ht' := multistep_preservation hpres ht hms
  intro ⟨hnv, hnf⟩
  rcases hprog ht' with hv | ⟨e'', hs⟩
  · exact hnv hv
  · exact hnf e'' hs

/-- Counted variant of type safety. -/
theorem type_safety_n (hpres : SingleStepPres) (hprog : ProgressFor [])
    {n : Nat} {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStepN n e e') :
    ¬Stuck e' :=
  type_safety hpres hprog ht (multiStepN_to_multiStep hms)

/-- If preservation and progress hold, every reduct of a closed well-typed
    term is safe. -/
theorem reduct_safe (hpres : SingleStepPres) (hprog : ProgressFor [])
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStep e e') :
    Safe e' :=
  type_safety hpres hprog ht hms

/-- If preservation and progress hold, a closed well-typed normal form is
    a value. -/
theorem typed_normal_is_value (hpres : SingleStepPres) (hprog : ProgressFor [])
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStep e e') (hnf : NormalForm e') :
    IsValue e' := by
  have ht' := multistep_preservation hpres ht hms
  rcases hprog ht' with hv | ⟨e'', hs⟩
  · exact hv
  · exact absurd hs (hnf e'')

-- ================================================================
-- §7. Safety Under Multi-Step
-- ================================================================

/-- If safety is preserved under single steps, it extends to multi-step. -/
theorem multistep_safe_preservation
    (h : ∀ e e', Safe e → Step e e' → Safe e')
    {e e' : Expr} (hs : Safe e) (hms : MultiStep e e') : Safe e' := by
  induction hms with
  | refl => exact hs
  | step hstep _ ih => exact ih (h _ _ hs hstep)

/-- Value safety is trivially preserved: values don't step. -/
theorem value_safety_preserved : ∀ e e', IsValue e → Step e e' → IsValue e' := by
  intro e e' hv hs
  exact absurd hs (value_irreducible hv _)

-- ================================================================
-- §8. Congruence Step Lemmas
-- ================================================================

/-- App left congruence. -/
theorem step_app_left_cong {f f' a : Expr} (hs : Step f f') :
    Step (.app f a) (.app f' a) :=
  .app_left hs

/-- App right congruence. -/
theorem step_app_right_cong {f a a' : Expr} (hvf : IsValue f) (hs : Step a a') :
    Step (.app f a) (.app f a') :=
  .app_right hvf hs

/-- Pair left congruence. -/
theorem step_pair_left_cong {e₁ e₁' e₂ : Expr} (hs : Step e₁ e₁') :
    Step (.pair e₁ e₂) (.pair e₁' e₂) :=
  .pair_left hs

/-- Pair right congruence. -/
theorem step_pair_right_cong {e₁ e₂ e₂' : Expr} (hv : IsValue e₁) (hs : Step e₂ e₂') :
    Step (.pair e₁ e₂) (.pair e₁ e₂') :=
  .pair_right hv hs

/-- Box congruence. -/
theorem step_box_cong {e e' : Expr} (hs : Step e e') :
    Step (.box e) (.box e') :=
  .box_step hs

/-- LetP scrutinee congruence. -/
theorem step_letP_cong {e e' : Expr} {x y : String} {body : Expr} (hs : Step e e') :
    Step (.letP x y e body) (.letP x y e' body) :=
  .letP_step hs

/-- LetB scrutinee congruence. -/
theorem step_letB_cong {e e' : Expr} {x : String} {body : Expr} (hs : Step e e') :
    Step (.letB x e body) (.letB x e' body) :=
  .letB_step hs

-- ================================================================
-- §9. Multi-Step Evaluation Sequences
-- ================================================================

/-- Full beta: evaluate function to lambda, argument to value, then beta. -/
theorem eval_app_sequence {f a v_a : Expr} {x : String} {m : Modality}
    {τ : Ty} {body : Expr}
    (hf : MultiStep f (.lam x m τ body))
    (ha : MultiStep a v_a) (hva : IsValue v_a) :
    MultiStep (.app f a) (subst x v_a body) :=
  eval_app_full hf ha hva

/-- Full letP: evaluate scrutinee to pair, then letP beta. -/
theorem eval_letP_sequence {e v₁ v₂ : Expr} {x y : String} {body : Expr}
    (he : MultiStep e (.pair v₁ v₂))
    (hv₁ : IsValue v₁) (hv₂ : IsValue v₂) :
    MultiStep (.letP x y e body) (subst y v₂ (subst x v₁ body)) :=
  eval_letP_full he hv₁ hv₂

/-- Full letB: evaluate scrutinee to box, then letB beta. -/
theorem eval_letB_sequence {e v : Expr} {x : String} {body : Expr}
    (he : MultiStep e (.box v)) (hv : IsValue v) :
    MultiStep (.letB x e body) (subst x v body) :=
  eval_letB_full he hv

-- ================================================================
-- §10. Effect Subsumption Preservation
-- ================================================================

/-- Effect subsumption is always admissible via the Sub typing rule. -/
theorem effect_sub_preservation {Γ : TyCtx} {e : Expr} {τ : Ty}
    {ρ ρ' : EffectRow}
    (ht : Typing Γ e τ ρ) (hsub : effectSubrow ρ ρ') :
    Typing Γ e τ ρ' :=
  Typing.Sub ht hsub

/-- Any well-typed term can be given the all-effects row. -/
theorem widen_to_all_effects {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ ρ) :
    Typing Γ e τ allEffectsRow :=
  Typing.Sub ht (effectSubrow_allEffects ρ)

/-- If preservation holds for row rho, it also holds for any superset rho'. -/
theorem preservation_effect_monotone (hpres : SingleStepPres)
    {Γ : TyCtx} {e e' : Expr} {τ : Ty} {ρ ρ' : EffectRow}
    (ht : Typing Γ e τ ρ) (hsub : effectSubrow ρ ρ')
    (hs : Step e e') :
    Typing Γ e' τ ρ' :=
  Typing.Sub (hpres ht hs) hsub

-- ================================================================
-- §11. Weakening and Preservation
-- ================================================================

/-- Weakening is always sound for weakenable modalities. -/
theorem weaken_preserves_typing {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    {x : String} {σ : Ty} {m : Modality}
    (hw : m.allowsWeakening = true)
    (ht : Typing Γ e τ ρ) :
    Typing ((x, σ, m) :: Γ) e τ ρ :=
  Typing.Weak ht hw

/-- Multi-weakening: add multiple weakenable variables. -/
theorem multi_weaken_preserves {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Δ : TyCtx)
    (hw : ∀ entry ∈ Δ, entry.2.2.allowsWeakening = true)
    (ht : Typing Γ e τ ρ) :
    Typing (Δ ++ Γ) e τ ρ :=
  typing_weaken_list Δ hw ht

-- ================================================================
-- §12. Evaluation Determinism (Extended)
-- ================================================================

/-- A value evaluates to itself and nothing else. -/
theorem value_evaluation_unique {v v' : Expr} (hv : IsValue v)
    (hev : EvaluatesTo v v') : v = v' :=
  value_multistep_id hv hev.1

/-- Multi-step from a value is reflexive. -/
theorem value_multistep_refl {v : Expr} (_hv : IsValue v) : MultiStep v v :=
  .refl

/-- If e evaluates to v, and e also multi-steps to v' which is a value,
    then v = v'. -/
theorem evaluation_value_unique {e v v' : Expr}
    (hev : EvaluatesTo e v) (hms : MultiStep e v') (hv' : IsValue v') :
    v = v' :=
  evaluatesTo_unique hev ⟨hms, hv'⟩

/-- If e steps to e', and e evaluates to v, then e' also evaluates to v. -/
theorem step_evaluation {e e' v : Expr}
    (hs : Step e e') (hev : EvaluatesTo e v) :
    EvaluatesTo e' v := by
  obtain ⟨hms, hv⟩ := hev
  cases hms with
  | refl => exact absurd hs (value_irreducible hv _)
  | step hs' hms' =>
    have heq := step_deterministic hs hs'
    subst heq
    exact ⟨hms', hv⟩

/-- Evaluation is stable under prepending a step. -/
theorem evaluation_prepend {e e' v : Expr}
    (hs : Step e e') (hev : EvaluatesTo e' v) :
    EvaluatesTo e v :=
  ⟨.step hs hev.1, hev.2⟩

-- ================================================================
-- §13. Preservation and Canonical Forms (Parametric)
-- ================================================================

/-- Canonical form for function types: a value of function type is a lambda.
    (Self-contained re-derivation, no dependency on SounioProgress.) -/
theorem canonical_fun {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ_full ρ) (hv : IsValue e)
    (heq : ∃ m τ σ, τ_full = .Fun m τ σ) :
    ∃ x m' τ' body, e = .lam x m' τ' body := by
  induction ht with
  | Var _ => cases hv
  | Lam _ => exact ⟨_, _, _, _, rfl⟩
  | App _ _ => cases hv
  | Pair _ _ => obtain ⟨_, _, _, h⟩ := heq; exact absurd h Ty.noConfusion
  | LetP _ _ => cases hv
  | Box _ => obtain ⟨_, _, _, h⟩ := heq; exact absurd h Ty.noConfusion
  | LetB _ _ => cases hv
  | Weak _ _ ih => exact ih hv heq
  | Sub _ _ ih => exact ih hv heq

/-- Canonical form for product types: a value of product type is a pair. -/
theorem canonical_prod {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ_full ρ) (hv : IsValue e)
    (heq : ∃ τ₁ τ₂, τ_full = .Prod τ₁ τ₂) :
    ∃ e₁ e₂, e = .pair e₁ e₂ := by
  induction ht with
  | Var _ => cases hv
  | Lam _ => obtain ⟨_, _, h⟩ := heq; exact absurd h Ty.noConfusion
  | App _ _ => cases hv
  | Pair _ _ => exact ⟨_, _, rfl⟩
  | LetP _ _ => cases hv
  | Box _ => obtain ⟨_, _, h⟩ := heq; exact absurd h Ty.noConfusion
  | LetB _ _ => cases hv
  | Weak _ _ ih => exact ih hv heq
  | Sub _ _ ih => exact ih hv heq

/-- Canonical form for bang types: a value of !tau is a box. -/
theorem canonical_bang {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
    (ht : Typing Γ e τ_full ρ) (hv : IsValue e)
    (heq : ∃ τ, τ_full = .Bang τ) :
    ∃ e₁, e = .box e₁ := by
  induction ht with
  | Var _ => cases hv
  | Lam _ => obtain ⟨_, h⟩ := heq; exact absurd h Ty.noConfusion
  | App _ _ => cases hv
  | Pair _ _ => obtain ⟨_, h⟩ := heq; exact absurd h Ty.noConfusion
  | LetP _ _ => cases hv
  | Box _ => exact ⟨_, rfl⟩
  | LetB _ _ => cases hv
  | Weak _ _ ih => exact ih hv heq
  | Sub _ _ ih => exact ih hv heq

/-- If preservation holds, a closed function-typed term evaluating to a value
    evaluates to a lambda. -/
theorem eval_fun_to_lam (hpres : SingleStepPres)
    {e v : Expr} {m : Modality} {τ σ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Fun m τ σ) ρ) (hev : EvaluatesTo e v) :
    ∃ x m' τ' body, v = .lam x m' τ' body :=
  canonical_fun (evaluation_preservation hpres ht hev) hev.2 ⟨_, _, _, rfl⟩

/-- If preservation holds, a closed product-typed term evaluating to a value
    evaluates to a pair. -/
theorem eval_prod_to_pair (hpres : SingleStepPres)
    {e v : Expr} {τ₁ τ₂ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Prod τ₁ τ₂) ρ) (hev : EvaluatesTo e v) :
    ∃ e₁ e₂, v = .pair e₁ e₂ :=
  canonical_prod (evaluation_preservation hpres ht hev) hev.2 ⟨_, _, rfl⟩

/-- If preservation holds, a closed bang-typed term evaluating to a value
    evaluates to a box. -/
theorem eval_bang_to_box (hpres : SingleStepPres)
    {e v : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Bang τ) ρ) (hev : EvaluatesTo e v) :
    ∃ e₁, v = .box e₁ :=
  canonical_bang (evaluation_preservation hpres ht hev) hev.2 ⟨_, rfl⟩

-- ================================================================
-- §14. Well-Typedness as a Preserved Property
-- ================================================================

/-- The property "well-typed at (Gamma, tau, rho)" is a Preserves instance
    exactly when SingleStepPres holds. -/
theorem wellTyped_is_preserves (hpres : SingleStepPres)
    (Γ : TyCtx) (τ : Ty) (ρ : EffectRow) :
    Preserves (fun e => Typing Γ e τ ρ) := by
  intro e e' ht hs
  exact hpres ht hs

/-- Safety of closed well-typed terms is preserved under stepping
    (when preservation and progress hold). -/
theorem closed_safety_preserved (hpres : SingleStepPres) (hprog : ProgressFor [])
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hs : Step e e') :
    Safe e' := by
  have ht' := hpres ht hs
  intro ⟨hnv, hnf⟩
  rcases hprog ht' with hv | ⟨e'', hstep⟩
  · exact hnv hv
  · exact hnf e'' hstep

-- ================================================================
-- §15. Multi-Step Safety Chain
-- ================================================================

/-- Every term in a multi-step chain from a closed well-typed term is safe. -/
theorem multistep_chain_safe (hpres : SingleStepPres) (hprog : ProgressFor [])
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStep e e') :
    Safe e' :=
  type_safety hpres hprog ht hms

/-- Every term in a counted chain from a closed well-typed term is well-typed. -/
theorem multiStepN_chain_typed (hpres : SingleStepPres)
    {n : Nat} {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStepN n e e') :
    Typing [] e' τ ρ :=
  multiStepN_preservation hpres ht hms

-- ================================================================
-- §16. Backward Safety
-- ================================================================

/-- If e steps to e', then e was safe (regardless of e'). -/
theorem step_backward_safe {e e' : Expr} (hs : Step e e') :
    Safe e :=
  stepping_safe hs

/-- An expression that can step is safe. -/
theorem steppable_safe {e : Expr} (h : ∃ e', Step e e') : Safe e := by
  obtain ⟨e', hs⟩ := h
  exact stepping_safe hs

-- ================================================================
-- §17. Type Safety for Specific Type Forms
-- ================================================================

/-- Function-typed closed terms satisfy the progress property. -/
theorem fun_typed_progress (hprog : ProgressFor [])
    {e : Expr} {m : Modality} {τ σ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Fun m τ σ) ρ) :
    IsValue e ∨ ∃ e', Step e e' :=
  hprog ht

/-- Product-typed closed terms satisfy the progress property. -/
theorem prod_typed_progress (hprog : ProgressFor [])
    {e : Expr} {τ₁ τ₂ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Prod τ₁ τ₂) ρ) :
    IsValue e ∨ ∃ e', Step e e' :=
  hprog ht

/-- Bang-typed closed terms satisfy the progress property. -/
theorem bang_typed_progress (hprog : ProgressFor [])
    {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Bang τ) ρ) :
    IsValue e ∨ ∃ e', Step e e' :=
  hprog ht

-- ================================================================
-- §18. Preservation Algebra
-- ================================================================

/-- Conjunction of preserved properties is preserved (restated for API). -/
theorem preserves_pair {P Q : Expr → Prop}
    (hp : Preserves P) (hq : Preserves Q) :
    Preserves (fun e => P e ∧ Q e) :=
  preserves_and hp hq

/-- A disjunction where both disjuncts are preserved is preserved. -/
theorem preserves_or {P Q : Expr → Prop}
    (hp : Preserves P) (hq : Preserves Q) :
    Preserves (fun e => P e ∨ Q e) := by
  intro e e' hpq hs
  rcases hpq with hp' | hq'
  · exact Or.inl (hp e e' hp' hs)
  · exact Or.inr (hq e e' hq' hs)

-- ================================================================
-- §19. Pure Row Composition
-- ================================================================

/-- A pure expression is well-typed at any effect row. -/
theorem pure_to_any_effects {Γ : TyCtx} {e : Expr} {τ : Ty} (ρ : EffectRow)
    (ht : Typing Γ e τ pureRow) : Typing Γ e τ ρ :=
  typing_pure_sub ρ ht

/-- The pure row is a subrow of any row (restated for this namespace). -/
theorem pure_sub_any (ρ : EffectRow) : effectSubrow pureRow ρ :=
  pure_is_subrow ρ

-- ================================================================
-- §20. Evaluation Composition
-- ================================================================

/-- If e evaluates to v and v is a lambda, then a subsequent application
    can beta-reduce. -/
theorem evaluation_then_apply {e : Expr} {x : String} {m : Modality}
    {τ : Ty} {body a : Expr}
    (hev : EvaluatesTo e (.lam x m τ body))
    (hva : IsValue a) :
    MultiStep (.app e a) (subst x a body) :=
  eval_app_full hev.1 .refl hva

/-- If e evaluates to a pair (v1, v2), then a subsequent letP can reduce. -/
theorem evaluation_then_letP {e v₁ v₂ : Expr} {x y : String} {body : Expr}
    (hev : EvaluatesTo e (.pair v₁ v₂))
    (hv₁ : IsValue v₁) (hv₂ : IsValue v₂) :
    MultiStep (.letP x y e body) (subst y v₂ (subst x v₁ body)) :=
  eval_letP_full hev.1 hv₁ hv₂

/-- If e evaluates to box v, then a subsequent letB can reduce. -/
theorem evaluation_then_letB {e v : Expr} {x : String} {body : Expr}
    (hev : EvaluatesTo e (.box v))
    (hv : IsValue v) :
    MultiStep (.letB x e body) (subst x v body) :=
  eval_letB_full hev.1 hv

end Sounio.Preservation
