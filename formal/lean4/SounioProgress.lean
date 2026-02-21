-- formal/lean4/SounioProgress.lean
import SounioLinear
import SounioEffects
import SounioTyping
import SounioSemantics
/-!
# Sounio Progress Theorem -- Lean 4 Formalization

Canonical forms lemmas and the Progress theorem for the Sounio
linear effect calculus.

**Progress**: A closed, well-typed expression is either a value or can
take a step.

Together with Preservation (proved separately), this yields type safety
by the Wright-Felleisen syntactic method.

References:
- Wright, A. and Felleisen, M. (1994). "A Syntactic Approach to Type Soundness."
  Information and Computation 115(1):38-94.
- Pierce, B. (2002). "Types and Programming Languages." MIT Press, Ch. 8.
- Girard, J.-Y. (1987). "Linear Logic." TCS 50(1):1-102.

No sorry. No Mathlib.
-/

open Sounio.Linear Sounio.Effects Sounio.Typing Sounio.Semantics

namespace Sounio.Progress

-- ================================================================
-- S1. Value Shape Lemmas (from IsValue alone, no typing needed)
-- ================================================================

/-- Every value is a lambda, pair, or box. -/
theorem value_shape {e : Expr} (hv : IsValue e) :
    (∃ x m τ body, e = .lam x m τ body) ∨
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨
    (∃ e₁, e = .box e₁) := by
  cases hv with
  | lam => exact Or.inl ⟨_, _, _, _, rfl⟩
  | pair _ _ => exact Or.inr (Or.inl ⟨_, _, rfl⟩)
  | box _ => exact Or.inr (Or.inr ⟨_, rfl⟩)

/-- A variable is not a value. -/
theorem value_not_var (x : String) : ¬IsValue (.var x) := by
  intro hv; cases hv

/-- An application is not a value. -/
theorem value_not_app (f a : Expr) : ¬IsValue (.app f a) := by
  intro hv; cases hv

/-- A let-pair elimination is not a value. -/
theorem value_not_letP (x y : String) (e body : Expr) :
    ¬IsValue (.letP x y e body) := by
  intro hv; cases hv

/-- A let-bang elimination is not a value. -/
theorem value_not_letB (x : String) (e body : Expr) :
    ¬IsValue (.letB x e body) := by
  intro hv; cases hv

-- ================================================================
-- S2. Typing-Value Shape
-- ================================================================

/-- A well-typed value in any context is a lambda, pair, or box. -/
theorem typing_value_shape {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (_ht : Typing Γ e τ ρ) (hv : IsValue e) :
    (∃ x m τ₁ body, e = .lam x m τ₁ body) ∨
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨
    (∃ e₁, e = .box e₁) :=
  value_shape hv

-- ================================================================
-- S3. Canonical Forms (Key Lemmas)
-- ================================================================

/-- Canonical form for function types:
    A well-typed value of function type must be a lambda abstraction. -/
theorem canonical_fun_form {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
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

/-- Canonical form for product types:
    A well-typed value of product type must be a pair. -/
theorem canonical_prod_form {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
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

/-- Canonical form for bang types:
    A well-typed value of !tau must be a boxed expression. -/
theorem canonical_bang_form {Γ : TyCtx} {e : Expr} {τ_full : Ty} {ρ : EffectRow}
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

/-- Canonical form for base types:
    A well-typed value of base type has the generic value shape. -/
theorem canonical_base_shape {Γ : TyCtx} {e : Expr} {s : String} {ρ : EffectRow}
    (_ht : Typing Γ e (.Base s) ρ) (hv : IsValue e) :
    (∃ x m τ body, e = .lam x m τ body) ∨
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨
    (∃ e₁, e = .box e₁) :=
  value_shape hv

-- ================================================================
-- S4. Value Sub-Structure Lemmas
-- ================================================================

/-- A lambda is always a value. -/
theorem value_lam_intro {x : String} {m : Modality} {τ : Ty} {body : Expr} :
    IsValue (.lam x m τ body) := .lam

/-- A pair value has sub-values. -/
theorem value_pair_subvalues {e₁ e₂ : Expr} (hv : IsValue (.pair e₁ e₂)) :
    IsValue e₁ ∧ IsValue e₂ := by
  cases hv with
  | pair h₁ h₂ => exact ⟨h₁, h₂⟩

/-- A box value has a sub-value. -/
theorem value_box_subvalue {e : Expr} (hv : IsValue (.box e)) :
    IsValue e := by
  cases hv with
  | box h => exact h

/-- A pair of values is a value. -/
theorem value_pair_intro {e₁ e₂ : Expr}
    (hv₁ : IsValue e₁) (hv₂ : IsValue e₂) :
    IsValue (.pair e₁ e₂) := .pair hv₁ hv₂

/-- A boxed value is a value. -/
theorem value_box_intro {e : Expr} (hv : IsValue e) :
    IsValue (.box e) := .box hv

-- ================================================================
-- S5. Empty Context Lemmas
-- ================================================================

/-- No variable is a member of the empty context. -/
theorem empty_ctx_no_var {x : String} {τ : Ty} {m : Modality} :
    (x, τ, m) ∉ ([] : TyCtx) :=
  List.not_mem_nil

/-- A cons context is never empty. -/
theorem cons_ctx_ne_nil {x : String} {τ : Ty} {m : Modality} {Γ : TyCtx} :
    (x, τ, m) :: Γ ≠ [] :=
  List.cons_ne_nil _ _

-- ================================================================
-- S6. The Progress Theorem
-- ================================================================

/-- **Progress**: Every closed, well-typed expression is either a value
    or can take a step.

    This is the fundamental safety property: well-typed programs don't
    get stuck. Combined with preservation, it yields type safety.

    Proof is by induction on the typing derivation, with a generalized
    context parameter to handle the Weak constructor. -/
theorem progress {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) :
    IsValue e ∨ ∃ e', Step e e' := by
  generalize hΓ : ([] : TyCtx) = Γ at ht
  induction ht with
  | Var hmem =>
    subst hΓ; exact absurd hmem List.not_mem_nil
  | Lam _ =>
    exact Or.inl .lam
  | App hf _ha ihf iha =>
    subst hΓ
    rcases ihf rfl with hvf | ⟨f', hsf⟩
    · rcases iha rfl with hva | ⟨a', hsa⟩
      · obtain ⟨x, m', τ', body, rfl⟩ :=
          canonical_fun_form hf hvf ⟨_, _, _, rfl⟩
        exact Or.inr ⟨_, .beta hva⟩
      · exact Or.inr ⟨_, .app_right hvf hsa⟩
    · exact Or.inr ⟨_, .app_left hsf⟩
  | Pair _h₁ _h₂ ih₁ ih₂ =>
    subst hΓ
    rcases ih₁ rfl with hv₁ | ⟨e₁', hs₁⟩
    · rcases ih₂ rfl with hv₂ | ⟨e₂', hs₂⟩
      · exact Or.inl (.pair hv₁ hv₂)
      · exact Or.inr ⟨_, .pair_right hv₁ hs₂⟩
    · exact Or.inr ⟨_, .pair_left hs₁⟩
  | LetP he _hb ihe _ihb =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', hs⟩
    · obtain ⟨e₁, e₂, rfl⟩ :=
        canonical_prod_form he hv ⟨_, _, rfl⟩
      obtain ⟨hv₁, hv₂⟩ := value_pair_subvalues hv
      exact Or.inr ⟨_, .letP_beta hv₁ hv₂⟩
    · exact Or.inr ⟨_, .letP_step hs⟩
  | Box _h ih =>
    subst hΓ
    rcases ih rfl with hv | ⟨e', hs⟩
    · exact Or.inl (.box hv)
    · exact Or.inr ⟨_, .box_step hs⟩
  | LetB he _hb ihe _ihb =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', hs⟩
    · obtain ⟨e₁, rfl⟩ :=
        canonical_bang_form he hv ⟨_, rfl⟩
      exact Or.inr ⟨_, .letB_beta (value_box_subvalue hv)⟩
    · exact Or.inr ⟨_, .letB_step hs⟩
  | Weak _ _ _ =>
    exact absurd hΓ.symm (List.cons_ne_nil _ _)
  | Sub _ _ ih =>
    exact ih hΓ

-- ================================================================
-- S7. Stuck Terms and Safety Properties
-- ================================================================

/-- An expression is stuck if it is neither a value nor can it step. -/
def Stuck (e : Expr) : Prop := ¬IsValue e ∧ ∀ e', ¬Step e e'

/-- A value is not stuck. -/
theorem value_not_stuck {e : Expr} (hv : IsValue e) : ¬Stuck e := by
  intro ⟨hnv, _⟩; exact hnv hv

/-- An expression that can step is not stuck. -/
theorem stepping_not_stuck {e e' : Expr} (hs : Step e e') : ¬Stuck e := by
  intro ⟨_, hnstep⟩; exact hnstep e' hs

/-- Closed well-typed terms are not stuck (immediate corollary of progress). -/
theorem progress_not_stuck {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) : ¬Stuck e := by
  rcases progress ht with hv | ⟨e', hs⟩
  · exact value_not_stuck hv
  · exact stepping_not_stuck hs

/-- A stuck expression cannot be well-typed in the empty context. -/
theorem stuck_not_well_typed {e : Expr} (hstuck : Stuck e) :
    ∀ τ ρ, ¬Typing [] e τ ρ := by
  intro τ ρ ht; exact progress_not_stuck ht hstuck

/-- A closed well-typed stuck term leads to a contradiction. -/
theorem no_closed_typed_stuck {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hstuck : Stuck e) : False :=
  progress_not_stuck ht hstuck

-- ================================================================
-- S8. Progress Corollaries for Specific Types
-- ================================================================

/-- A closed term of function type is either a lambda or can step. -/
theorem progress_fun {e : Expr} {m : Modality} {τ σ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Fun m τ σ) ρ) :
    (∃ x m' τ' body, e = .lam x m' τ' body) ∨ (∃ e', Step e e') := by
  rcases progress ht with hv | hs
  · exact Or.inl (canonical_fun_form ht hv ⟨_, _, _, rfl⟩)
  · exact Or.inr hs

/-- A closed term of product type is either a pair or can step. -/
theorem progress_prod {e : Expr} {τ₁ τ₂ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Prod τ₁ τ₂) ρ) :
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨ (∃ e', Step e e') := by
  rcases progress ht with hv | hs
  · exact Or.inl (canonical_prod_form ht hv ⟨_, _, rfl⟩)
  · exact Or.inr hs

/-- A closed term of bang type is either a box or can step. -/
theorem progress_bang {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Bang τ) ρ) :
    (∃ e₁, e = .box e₁) ∨ (∃ e', Step e e') := by
  rcases progress ht with hv | hs
  · exact Or.inl (canonical_bang_form ht hv ⟨_, rfl⟩)
  · exact Or.inr hs

-- ================================================================
-- S9. Normal Forms and Values
-- ================================================================

/-- A closed, well-typed normal form must be a value. -/
theorem closed_normal_is_value {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hnf : NormalForm e) :
    IsValue e := by
  rcases progress ht with hv | ⟨e', hs⟩
  · exact hv
  · exact absurd hs (hnf e')

/-- A closed, well-typed normal form of function type is a lambda. -/
theorem closed_normal_fun_is_lam {e : Expr} {m : Modality} {τ σ : Ty}
    {ρ : EffectRow}
    (ht : Typing [] e (.Fun m τ σ) ρ) (hnf : NormalForm e) :
    ∃ x m' τ' body, e = .lam x m' τ' body :=
  canonical_fun_form ht (closed_normal_is_value ht hnf) ⟨_, _, _, rfl⟩

/-- A closed, well-typed normal form of product type is a pair. -/
theorem closed_normal_prod_is_pair {e : Expr} {τ₁ τ₂ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Prod τ₁ τ₂) ρ) (hnf : NormalForm e) :
    ∃ e₁ e₂, e = .pair e₁ e₂ :=
  canonical_prod_form ht (closed_normal_is_value ht hnf) ⟨_, _, rfl⟩

/-- A closed, well-typed normal form of bang type is a box. -/
theorem closed_normal_bang_is_box {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Bang τ) ρ) (hnf : NormalForm e) :
    ∃ e₁, e = .box e₁ :=
  canonical_bang_form ht (closed_normal_is_value ht hnf) ⟨_, rfl⟩

-- ================================================================
-- S10. Closed Value Canonical Forms
-- ================================================================

/-- A closed value of function type is a lambda. -/
theorem closed_value_fun_is_lam {e : Expr} {m : Modality} {τ σ : Ty}
    {ρ : EffectRow}
    (ht : Typing [] e (.Fun m τ σ) ρ) (hv : IsValue e) :
    ∃ x m' τ' body, e = .lam x m' τ' body :=
  canonical_fun_form ht hv ⟨_, _, _, rfl⟩

/-- A closed value of product type is a pair. -/
theorem closed_value_prod_is_pair {e : Expr} {τ₁ τ₂ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Prod τ₁ τ₂) ρ) (hv : IsValue e) :
    ∃ e₁ e₂, e = .pair e₁ e₂ :=
  canonical_prod_form ht hv ⟨_, _, rfl⟩

/-- A closed value of bang type is a box. -/
theorem closed_value_bang_is_box {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e (.Bang τ) ρ) (hv : IsValue e) :
    ∃ e₁, e = .box e₁ :=
  canonical_bang_form ht hv ⟨_, rfl⟩

-- ================================================================
-- S11. Progress under Effect Subsumption
-- ================================================================

/-- Progress is preserved under effect subsumption. -/
theorem progress_effect_sub {e : Expr} {τ : Ty} {ρ ρ' : EffectRow}
    (ht : Typing [] e τ ρ) (_hsub : effectSubrow ρ ρ') :
    IsValue e ∨ ∃ e', Step e e' :=
  progress ht

/-- Progress for pure computations. -/
theorem progress_pure {e : Expr} {τ : Ty}
    (ht : Typing [] e τ pureRow) :
    IsValue e ∨ ∃ e', Step e e' :=
  progress ht

-- ================================================================
-- S12. Expression Shape Enumeration
-- ================================================================

/-- Every expression has one of the seven syntactic forms. -/
theorem expr_cases (e : Expr) :
    (∃ x, e = .var x) ∨
    (∃ x m τ body, e = .lam x m τ body) ∨
    (∃ f a, e = .app f a) ∨
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨
    (∃ x y scrut body, e = .letP x y scrut body) ∨
    (∃ e₁, e = .box e₁) ∨
    (∃ x scrut body, e = .letB x scrut body) := by
  cases e with
  | var x => exact Or.inl ⟨x, rfl⟩
  | lam x m τ body => exact Or.inr (Or.inl ⟨x, m, τ, body, rfl⟩)
  | app f a => exact Or.inr (Or.inr (Or.inl ⟨f, a, rfl⟩))
  | pair e₁ e₂ => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨e₁, e₂, rfl⟩)))
  | letP x y e body =>
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨x, y, e, body, rfl⟩))))
  | box e₁ =>
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨e₁, rfl⟩)))))
  | letB x e body =>
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨x, e, body, rfl⟩)))))

-- ================================================================
-- S13. Progress Trichotomy (Value XOR Step)
-- ================================================================

/-- Values and stepping are mutually exclusive. -/
theorem value_step_exclusive {e : Expr} (hv : IsValue e) :
    ∀ e', ¬Step e e' :=
  value_irreducible hv

/-- A closed well-typed expression is a value xor can step:
    exactly one of these holds, never both. -/
theorem progress_xor {e : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) :
    (IsValue e ∧ ∀ e', ¬Step e e') ∨
    (¬IsValue e ∧ ∃ e', Step e e') := by
  rcases progress ht with hv | ⟨e', hs⟩
  · exact Or.inl ⟨hv, value_irreducible hv⟩
  · exact Or.inr ⟨fun hv => value_irreducible hv e' hs, e', hs⟩

/-- Both branches of progress_xor are mutually exclusive. -/
theorem progress_xor_exclusive (e : Expr) :
    ¬((IsValue e ∧ ∀ e', ¬Step e e') ∧
       (¬IsValue e ∧ ∃ e', Step e e')) := by
  intro ⟨⟨hv, _⟩, ⟨hnv, _⟩⟩
  exact hnv hv

-- ================================================================
-- S14. Elimination Forms Always Step (Closed, Well-Typed)
-- ================================================================

/-- An application of a closed well-typed term can always step. -/
theorem app_always_steps {f a : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] (.app f a) τ ρ) :
    ∃ e', Step (.app f a) e' := by
  rcases progress ht with hv | hs
  · cases hv
  · exact hs

/-- A let-pair of a closed well-typed term can always step. -/
theorem letP_always_steps {x y : String} {e body : Expr} {τ : Ty}
    {ρ : EffectRow}
    (ht : Typing [] (.letP x y e body) τ ρ) :
    ∃ e', Step (.letP x y e body) e' := by
  rcases progress ht with hv | hs
  · cases hv
  · exact hs

/-- A let-bang of a closed well-typed term can always step. -/
theorem letB_always_steps {x : String} {e body : Expr} {τ : Ty}
    {ρ : EffectRow}
    (ht : Typing [] (.letB x e body) τ ρ) :
    ∃ e', Step (.letB x e body) e' := by
  rcases progress ht with hv | hs
  · cases hv
  · exact hs

-- ================================================================
-- S15. Stuck Term Characterization
-- ================================================================

/-- An open variable is stuck (not a value, cannot step). -/
theorem var_is_stuck (x : String) : Stuck (.var x) := by
  constructor
  · intro hv; cases hv
  · intro e' hs; cases hs

/-- Stuck terms include all expression forms except lambdas. -/
theorem stuck_shape {e : Expr} (hstuck : Stuck e) :
    (∃ x, e = .var x) ∨
    (∃ f a, e = .app f a) ∨
    (∃ e₁ e₂, e = .pair e₁ e₂) ∨
    (∃ x y scrut body, e = .letP x y scrut body) ∨
    (∃ e₁, e = .box e₁) ∨
    (∃ x scrut body, e = .letB x scrut body) := by
  obtain ⟨hnv, _⟩ := hstuck
  cases e with
  | var x => exact Or.inl ⟨x, rfl⟩
  | lam x m τ body => exact absurd IsValue.lam hnv
  | app f a => exact Or.inr (Or.inl ⟨f, a, rfl⟩)
  | pair e₁ e₂ => exact Or.inr (Or.inr (Or.inl ⟨e₁, e₂, rfl⟩))
  | letP x y scrut body =>
    exact Or.inr (Or.inr (Or.inr (Or.inl ⟨x, y, scrut, body, rfl⟩)))
  | box e₁ =>
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨e₁, rfl⟩))))
  | letB x scrut body =>
    exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨x, scrut, body, rfl⟩))))

-- ================================================================
-- S16. Canonical Forms under Weakening
-- ================================================================

/-- Canonical form for functions is stable under weakening. -/
theorem canonical_fun_weakening {Γ : TyCtx} {e : Expr} {m : Modality}
    {τ σ : Ty} {ρ : EffectRow} {x : String} {σ' : Ty} {m' : Modality}
    (ht : Typing ((x, σ', m') :: Γ) e (.Fun m τ σ) ρ) (hv : IsValue e) :
    ∃ x' m'' τ' body, e = .lam x' m'' τ' body :=
  canonical_fun_form ht hv ⟨_, _, _, rfl⟩

/-- Canonical form for products is stable under weakening. -/
theorem canonical_prod_weakening {Γ : TyCtx} {e : Expr}
    {τ₁ τ₂ : Ty} {ρ : EffectRow} {x : String} {σ' : Ty} {m' : Modality}
    (ht : Typing ((x, σ', m') :: Γ) e (.Prod τ₁ τ₂) ρ) (hv : IsValue e) :
    ∃ e₁ e₂, e = .pair e₁ e₂ :=
  canonical_prod_form ht hv ⟨_, _, rfl⟩

/-- Canonical form for bang is stable under weakening. -/
theorem canonical_bang_weakening {Γ : TyCtx} {e : Expr}
    {τ : Ty} {ρ : EffectRow} {x : String} {σ' : Ty} {m' : Modality}
    (ht : Typing ((x, σ', m') :: Γ) e (.Bang τ) ρ) (hv : IsValue e) :
    ∃ e₁, e = .box e₁ :=
  canonical_bang_form ht hv ⟨_, rfl⟩

-- ================================================================
-- S17. Multi-Step Safety
-- ================================================================

/-- A value multi-steps only to itself. -/
theorem value_multistep_self {e e' : Expr}
    (hv : IsValue e) (hms : MultiStep e e') : e = e' := by
  cases hms with
  | refl => rfl
  | step hs _ => exact absurd hs (value_irreducible hv _)

/-- Assuming preservation holds, a multi-step reduct of a closed
    well-typed term that reaches a normal form is a value.
    (We state this with preservation as an explicit hypothesis,
    since preservation is proved in a separate file.) -/
theorem multistep_normal_is_value
    (preservation : ∀ {e e' : Expr} {τ : Ty} {ρ : EffectRow},
      Typing [] e τ ρ → Step e e' → Typing [] e' τ ρ)
    {e e' : Expr} {τ : Ty} {ρ : EffectRow}
    (ht : Typing [] e τ ρ) (hms : MultiStep e e') (hnf : NormalForm e') :
    IsValue e' := by
  induction hms with
  | refl => exact closed_normal_is_value ht hnf
  | step hs _ ih =>
    exact ih (preservation ht hs) hnf

-- ================================================================
-- S18. Progress for Sub-Expressions
-- ================================================================

/-- If an application is well-typed with closed context, the function
    sub-expression either is a value or steps. -/
theorem progress_app_fun {f a : Expr} {τ σ : Ty} {m : Modality}
    {ρ₁ ρ₂ : EffectRow}
    (hf : Typing [] f (.Fun m τ σ) ρ₁) (_ha : Typing [] a τ ρ₂) :
    IsValue f ∨ ∃ f', Step f f' :=
  progress hf

/-- If an application is well-typed with closed context, the argument
    sub-expression either is a value or steps. -/
theorem progress_app_arg {f a : Expr} {τ σ : Ty} {m : Modality}
    {ρ₁ ρ₂ : EffectRow}
    (_hf : Typing [] f (.Fun m τ σ) ρ₁) (ha : Typing [] a τ ρ₂) :
    IsValue a ∨ ∃ a', Step a a' :=
  progress ha

end Sounio.Progress
