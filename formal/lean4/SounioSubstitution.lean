-- formal/lean4/SounioSubstitution.lean
import SounioLinear
import SounioEffects
import SounioTyping
import SounioSemantics
/-!
# Sounio Substitution Lemma — Lean 4 Formalization

Formalizes the substitution lemma and related context operations for the
Sounio linear effect calculus. This is the key metatheoretic result needed
for type preservation (subject reduction).

We prove:
- Free variable analysis and its interaction with substitution
- A freshness (Barendregt) predicate for binder hygiene
- Context membership lemmas (head, tail, nil, append)
- Context extension: appending entries preserves typing
- Closed-term weakening: empty-context typing lifts to any context
- Substitution for closed values (the CBV case)
- Context well-formedness (distinct names)

References:
- Wright, A. and Felleisen, M. (1994). "A Syntactic Approach to Type Soundness."
- Pierce, B. (2002). "Types and Programming Languages." MIT Press.
- Barendregt, H. (1984). "The Lambda Calculus: Its Syntax and Semantics."

No sorry. No Mathlib.
-/

open Sounio.Linear Sounio.Effects Sounio.Typing Sounio.Semantics

namespace Sounio.Substitution

-- ================================================================
-- §1. Free Variable Analysis
-- ================================================================

/-- Set of free variables in an expression. -/
def freeVars : Expr → List String
  | .var y => [y]
  | .lam y _ _ e => (freeVars e).filter (· != y)
  | .app f a => freeVars f ++ freeVars a
  | .pair e₁ e₂ => freeVars e₁ ++ freeVars e₂
  | .letP x y e body => freeVars e ++ (freeVars body).filter (fun z => z != x && z != y)
  | .box e => freeVars e
  | .letB x e body => freeVars e ++ (freeVars body).filter (· != x)

/-- Free variables of a variable expression. -/
theorem freeVars_var (x : String) : freeVars (.var x) = [x] := rfl

/-- Free variables of a lambda strip the binder. -/
theorem freeVars_lam (y : String) (m : Modality) (τ : Ty) (e : Expr) :
    freeVars (.lam y m τ e) = (freeVars e).filter (· != y) := rfl

/-- Free variables of an application. -/
theorem freeVars_app (f a : Expr) :
    freeVars (.app f a) = freeVars f ++ freeVars a := rfl

/-- Free variables of a pair. -/
theorem freeVars_pair (e₁ e₂ : Expr) :
    freeVars (.pair e₁ e₂) = freeVars e₁ ++ freeVars e₂ := rfl

/-- Free variables of a box. -/
theorem freeVars_box (e : Expr) :
    freeVars (.box e) = freeVars e := rfl

/-- Free variables of a let-pair. -/
theorem freeVars_letP (x y : String) (e body : Expr) :
    freeVars (.letP x y e body) =
      freeVars e ++ (freeVars body).filter (fun z => z != x && z != y) := rfl

/-- Free variables of a let-box. -/
theorem freeVars_letB (x : String) (e body : Expr) :
    freeVars (.letB x e body) =
      freeVars e ++ (freeVars body).filter (· != x) := rfl

-- ================================================================
-- §2. Freshness Predicate (Barendregt Condition)
-- ================================================================

/-- x does not appear as a binder in e. -/
def Fresh (x : String) : Expr → Prop
  | .var _ => True
  | .lam y _ _ e => x ≠ y ∧ Fresh x e
  | .app f a => Fresh x f ∧ Fresh x a
  | .pair e₁ e₂ => Fresh x e₁ ∧ Fresh x e₂
  | .letP a b e body => x ≠ a ∧ x ≠ b ∧ Fresh x e ∧ Fresh x body
  | .box e => Fresh x e
  | .letB a e body => x ≠ a ∧ Fresh x e ∧ Fresh x body

/-- Freshness holds trivially for variables. -/
theorem fresh_var (x y : String) : Fresh x (.var y) := trivial

/-- Freshness for lambda: not the binder AND fresh in the body. -/
theorem fresh_lam (x y : String) (m : Modality) (τ : Ty) (e : Expr) :
    Fresh x (.lam y m τ e) ↔ x ≠ y ∧ Fresh x e := Iff.rfl

/-- Freshness for application. -/
theorem fresh_app (x : String) (f a : Expr) :
    Fresh x (.app f a) ↔ Fresh x f ∧ Fresh x a := Iff.rfl

/-- Freshness for pair. -/
theorem fresh_pair (x : String) (e₁ e₂ : Expr) :
    Fresh x (.pair e₁ e₂) ↔ Fresh x e₁ ∧ Fresh x e₂ := Iff.rfl

/-- Freshness for box. -/
theorem fresh_box (x : String) (e : Expr) :
    Fresh x (.box e) ↔ Fresh x e := Iff.rfl

/-- Freshness for let-pair. -/
theorem fresh_letP (x a b : String) (e body : Expr) :
    Fresh x (.letP a b e body) ↔ x ≠ a ∧ x ≠ b ∧ Fresh x e ∧ Fresh x body :=
  Iff.rfl

/-- Freshness for let-box. -/
theorem fresh_letB (x a : String) (e body : Expr) :
    Fresh x (.letB a e body) ↔ x ≠ a ∧ Fresh x e ∧ Fresh x body := Iff.rfl

-- ================================================================
-- §3. Context Membership Lemmas
-- ================================================================

/-- Head of context is a member. -/
theorem ctx_mem_head {x : String} {τ : Ty} {m : Modality} {Γ : TyCtx} :
    (x, τ, m) ∈ ((x, τ, m) :: Γ) :=
  List.Mem.head _

/-- Tail membership lifts to cons. -/
theorem ctx_mem_tail {e : String × Ty × Modality} {hd : String × Ty × Modality}
    {Γ : TyCtx} (h : e ∈ Γ) : e ∈ (hd :: Γ) :=
  List.Mem.tail _ h

/-- The empty context has no members. -/
theorem ctx_mem_nil_false {e : String × Ty × Modality} : ¬(e ∈ ([] : TyCtx)) := by
  intro h; cases h

/-- Left append preserves membership. -/
theorem ctx_mem_append_left {e : String × Ty × Modality} {Γ Δ : TyCtx}
    (h : e ∈ Γ) : e ∈ (Γ ++ Δ) :=
  List.mem_append.mpr (Or.inl h)

/-- Right append preserves membership. -/
theorem ctx_mem_append_right {e : String × Ty × Modality} {Γ Δ : TyCtx}
    (h : e ∈ Δ) : e ∈ (Γ ++ Δ) :=
  List.mem_append.mpr (Or.inr h)

-- ================================================================
-- §4. Context Extension (Append) Preserves Typing
-- ================================================================

/-- Adding entries at the END of the context preserves typing.
    Var checks membership, which is preserved by appending. -/
theorem typing_append_right {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Δ : TyCtx) (ht : Typing Γ e τ ρ) : Typing (Γ ++ Δ) e τ ρ := by
  induction ht with
  | Var h =>
    exact Typing.Var (List.mem_append.mpr (Or.inl h))
  | Lam _hb ih =>
    rw [List.cons_append] at ih
    exact Typing.Lam ih
  | App _hf _ha ihf iha =>
    exact Typing.App ihf iha
  | Pair _h₁ _h₂ ih₁ ih₂ =>
    exact Typing.Pair ih₁ ih₂
  | LetP _he _hb ihe ihb =>
    rw [List.cons_append, List.cons_append] at ihb
    exact Typing.LetP ihe ihb
  | Box _he ih =>
    exact Typing.Box ih
  | LetB _he _hb ihe ihb =>
    rw [List.cons_append] at ihb
    exact Typing.LetB ihe ihb
  | Weak _hb hw ih =>
    rw [List.cons_append]
    exact Typing.Weak ih hw
  | Sub _h hsub ih =>
    exact Typing.Sub ih hsub

-- ================================================================
-- §5. Closed-Term Weakening
-- ================================================================

/-- A term typed in the empty context can be typed in any context. -/
theorem typing_closed_weakening {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Γ : TyCtx) (ht : Typing [] e τ ρ) : Typing Γ e τ ρ := by
  have h := typing_append_right Γ ht
  simp [List.nil_append] at h
  exact h

-- ================================================================
-- §6. Context Extension (Prepend Single Entry)
-- ================================================================

/-- Prepending any weakenable entry to the context preserves typing. -/
theorem typing_prepend_weakenable {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    {x : String} {σ : Ty} {m : Modality}
    (hw : m.allowsWeakening = true)
    (ht : Typing Γ e τ ρ) :
    Typing ((x, σ, m) :: Γ) e τ ρ :=
  Typing.Weak ht hw

-- ================================================================
-- §7. Substitution for Closed Values — Special Cases
-- ================================================================

/-- If we look up a variable that equals x, substitution yields v.
    Combined with closed weakening, typing is preserved. -/
theorem subst_var_hit_typing {Γ : TyCtx} {x : String} {σ : Ty} {v : Expr}
    (hv_ty : Typing [] v σ pureRow) :
    Typing Γ (subst x v (.var x)) σ pureRow := by
  simp [subst]
  exact typing_closed_weakening Γ hv_ty

/-- If the variable is different from x, substitution is a no-op on variables. -/
theorem subst_var_miss_typing {Γ : TyCtx} {x y : String} {τ : Ty} {m : Modality}
    {v : Expr}
    (hne : y ≠ x)
    (hmem : (y, τ, m) ∈ Γ) :
    Typing Γ (subst x v (.var y)) τ pureRow := by
  simp [subst, hne]
  exact Typing.Var hmem

/-- Substitution on a lambda whose binder shadows x is identity. -/
theorem subst_lam_shadow_typing {Γ : TyCtx} {x : String} {m : Modality}
    {τ σ : Ty} {e : Expr} {ρ : EffectRow} {v : Expr}
    (ht : Typing Γ (.lam x m τ e) σ ρ) :
    Typing Γ (subst x v (.lam x m τ e)) σ ρ := by
  simp [subst]
  exact ht

-- ================================================================
-- §8. Substitution Preserves Values
-- ================================================================

/-- Substitution into a lambda produces a value (lambdas are always values). -/
theorem subst_lam_is_value (x : String) (v : Expr) (y : String)
    (m : Modality) (τ : Ty) (e : Expr) :
    IsValue (subst x v (.lam y m τ e)) := by
  simp [subst]
  split <;> exact IsValue.lam

/-- Substitution into a box of a value preserves the box-value structure,
    given that the inner substitution preserves the value. -/
theorem subst_box_is_value {x : String} {v e : Expr}
    (ih : IsValue (subst x v e)) :
    IsValue (subst x v (.box e)) := by
  simp [subst]
  exact IsValue.box ih

-- ================================================================
-- §9. Context Well-Formedness
-- ================================================================

/-- A context is well-formed if all entries have distinct names. -/
def CtxWF : TyCtx → Prop
  | [] => True
  | (x, _, _) :: Γ => (∀ τ m, (x, τ, m) ∉ Γ) ∧ CtxWF Γ

/-- The empty context is well-formed. -/
theorem ctxWF_nil : CtxWF [] := trivial

/-- Extending a well-formed context with a fresh name. -/
theorem ctxWF_cons {x : String} {τ : Ty} {m : Modality} {Γ : TyCtx}
    (h : ∀ σ n, (x, σ, n) ∉ Γ) (hwf : CtxWF Γ) :
    CtxWF ((x, τ, m) :: Γ) := ⟨h, hwf⟩

/-- Tail of a well-formed context is well-formed. -/
theorem ctxWF_tail {x : String} {τ : Ty} {m : Modality} {Γ : TyCtx}
    (hwf : CtxWF ((x, τ, m) :: Γ)) : CtxWF Γ :=
  hwf.2

/-- In a well-formed context, the head name is not in the tail. -/
theorem ctxWF_head_not_in_tail {x : String} {τ : Ty} {m : Modality} {Γ : TyCtx}
    (hwf : CtxWF ((x, τ, m) :: Γ)) : ∀ σ n, (x, σ, n) ∉ Γ :=
  hwf.1

-- ================================================================
-- §10. Substitution Identity for Non-Free Variables
-- ================================================================

/-- Helper: not-mem in append splits to conjunction. -/
private theorem not_mem_append {α : Type} {x : α} {l₁ l₂ : List α}
    (h : x ∉ l₁ ++ l₂) : x ∉ l₁ ∧ x ∉ l₂ := by
  constructor
  · intro hm; exact h (List.mem_append.mpr (Or.inl hm))
  · intro hm; exact h (List.mem_append.mpr (Or.inr hm))

/-- If x is not in the free variables of e, then subst x v e = e.
    Proved by induction on e. -/
theorem not_free_subst_eq {x : String} {v : Expr} (e : Expr)
    (hnf : x ∉ freeVars e) : subst x v e = e := by
  induction e with
  | var y =>
    simp only [freeVars, List.mem_singleton] at hnf
    simp only [subst]
    split
    · next h => exact absurd h.symm hnf
    · rfl
  | lam y m τ body ih =>
    simp only [subst]
    split
    · rfl
    · next hyx =>
      congr 1
      apply ih
      intro hmem
      apply hnf
      simp only [freeVars]
      exact List.mem_filter.mpr ⟨hmem, bne_iff_ne.mpr (Ne.symm hyx)⟩
  | app f a ihf iha =>
    have hnf' := not_mem_append (show x ∉ freeVars f ++ freeVars a from hnf)
    simp only [subst]
    congr 1
    · exact ihf hnf'.1
    · exact iha hnf'.2
  | pair e₁ e₂ ih₁ ih₂ =>
    have hnf' := not_mem_append (show x ∉ freeVars e₁ ++ freeVars e₂ from hnf)
    simp only [subst]
    congr 1
    · exact ih₁ hnf'.1
    · exact ih₂ hnf'.2
  | letP a b e body ihe ihbody =>
    have hnf' := not_mem_append (show x ∉ freeVars e ++
        (freeVars body).filter (fun z => z != a && z != b) from hnf)
    simp only [subst]
    congr 1
    · exact ihe hnf'.1
    · split
      · rfl
      · next h =>
        have hna : a ≠ x := fun heq => h (Or.inl heq)
        have hnb : b ≠ x := fun heq => h (Or.inr heq)
        congr 1
        apply ihbody
        intro hmem
        apply hnf'.2
        refine List.mem_filter.mpr ⟨hmem, ?_⟩
        rw [Bool.and_eq_true]
        exact ⟨bne_iff_ne.mpr (Ne.symm hna), bne_iff_ne.mpr (Ne.symm hnb)⟩
  | box e ih =>
    simp only [subst]
    exact congrArg Expr.box (ih hnf)
  | letB a e body ihe ihbody =>
    have hnf' := not_mem_append (show x ∉ freeVars e ++
        (freeVars body).filter (· != a) from hnf)
    simp only [subst]
    congr 1
    · exact ihe hnf'.1
    · split
      · rfl
      · next hax =>
        congr 1
        apply ihbody
        intro hmem
        apply hnf'.2
        exact List.mem_filter.mpr ⟨hmem, bne_iff_ne.mpr (Ne.symm hax)⟩

-- ================================================================
-- §11. Weakening Preserves Membership
-- ================================================================

/-- If (y,τ,m) is in Γ, it's also in (x,σ,n)::Γ for any x,σ,n. -/
theorem mem_of_mem_cons {Γ : TyCtx} {y : String} {τ : Ty} {m : Modality}
    {x : String} {σ : Ty} {n : Modality}
    (h : (y, τ, m) ∈ Γ) : (y, τ, m) ∈ ((x, σ, n) :: Γ) :=
  List.Mem.tail _ h

/-- Membership in a singleton context. -/
theorem mem_singleton {x : String} {τ : Ty} {m : Modality} :
    (x, τ, m) ∈ [(x, τ, m)] :=
  List.Mem.head _

-- ================================================================
-- §12. Context Append Properties
-- ================================================================

/-- Appending nil on the right is identity. -/
theorem ctx_append_nil (Γ : TyCtx) : Γ ++ [] = Γ :=
  List.append_nil Γ

/-- Nil appended on the left is identity. -/
theorem ctx_nil_append (Γ : TyCtx) : [] ++ Γ = Γ :=
  List.nil_append Γ

/-- Cons distributes into append. -/
theorem ctx_cons_append (hd : String × Ty × Modality) (Γ Δ : TyCtx) :
    (hd :: Γ) ++ Δ = hd :: (Γ ++ Δ) := by
  rfl

/-- Append is associative. -/
theorem ctx_append_assoc (Γ₁ Γ₂ Γ₃ : TyCtx) :
    (Γ₁ ++ Γ₂) ++ Γ₃ = Γ₁ ++ (Γ₂ ++ Γ₃) :=
  List.append_assoc Γ₁ Γ₂ Γ₃

-- ================================================================
-- §13. Typing in Extended Contexts
-- ================================================================

/-- A variable typed in Γ remains typed in Γ ++ Δ. -/
theorem var_typing_append {Γ : TyCtx} {x : String} {τ : Ty} {m : Modality}
    (Δ : TyCtx) (h : (x, τ, m) ∈ Γ) : Typing (Γ ++ Δ) (.var x) τ pureRow :=
  Typing.Var (ctx_mem_append_left h)

-- ================================================================
-- §14. Substitution Typing — Closed Value Case
-- ================================================================

/-- Substitution for the variable case: if v has type σ in the empty context,
    then v has type σ in any context Γ. -/
theorem subst_var_closed_typing {Γ : TyCtx} {σ : Ty}
    {v : Expr} {ρ : EffectRow}
    (hv_ty : Typing [] v σ ρ) :
    Typing Γ v σ ρ :=
  typing_closed_weakening Γ hv_ty

/-- Substitution lemma for the variable-hit case with effect subsumption:
    if [] ⊢ v : σ ! ρ_v, then for any ρ with ρ_v ⊆ ρ, Γ ⊢ v : σ ! ρ. -/
theorem subst_var_hit_typing_sub {Γ : TyCtx} {x : String} {σ : Ty}
    {v : Expr} {ρ_v ρ : EffectRow}
    (hv_ty : Typing [] v σ ρ_v)
    (hsub : effectSubrow ρ_v ρ) :
    Typing Γ (subst x v (.var x)) σ ρ := by
  simp [subst]
  exact Typing.Sub (typing_closed_weakening Γ hv_ty) hsub

-- ================================================================
-- §15. Structural Properties of Fresh
-- ================================================================

/-- All variables are trivially fresh (no binders). -/
theorem fresh_in_var (x y : String) : Fresh x (.var y) := trivial

/-- If x is fresh in both subexpressions, it is fresh in their application. -/
theorem fresh_app_intro {x : String} {f a : Expr}
    (hf : Fresh x f) (ha : Fresh x a) : Fresh x (.app f a) :=
  ⟨hf, ha⟩

/-- If x is fresh in both subexpressions, it is fresh in their pair. -/
theorem fresh_pair_intro {x : String} {e₁ e₂ : Expr}
    (h₁ : Fresh x e₁) (h₂ : Fresh x e₂) : Fresh x (.pair e₁ e₂) :=
  ⟨h₁, h₂⟩

/-- If x is fresh in e, it is fresh in box e. -/
theorem fresh_box_intro {x : String} {e : Expr}
    (h : Fresh x e) : Fresh x (.box e) := h

/-- Fresh in a lambda with a different binder. -/
theorem fresh_lam_intro {x y : String} {m : Modality} {τ : Ty} {e : Expr}
    (hne : x ≠ y) (hf : Fresh x e) : Fresh x (.lam y m τ e) :=
  ⟨hne, hf⟩

/-- Fresh in a let-box with a different binder. -/
theorem fresh_letB_intro {x a : String} {e body : Expr}
    (hne : x ≠ a) (he : Fresh x e) (hb : Fresh x body) :
    Fresh x (.letB a e body) :=
  ⟨hne, he, hb⟩

/-- Fresh in a let-pair with different binders. -/
theorem fresh_letP_intro {x a b : String} {e body : Expr}
    (hna : x ≠ a) (hnb : x ≠ b) (he : Fresh x e) (hb : Fresh x body) :
    Fresh x (.letP a b e body) :=
  ⟨hna, hnb, he, hb⟩

-- ================================================================
-- §16. CtxWF Derived Lemmas
-- ================================================================

/-- A singleton context is well-formed. -/
theorem ctxWF_singleton (x : String) (τ : Ty) (m : Modality) :
    CtxWF [(x, τ, m)] := by
  constructor
  · intro _ _ h; cases h
  · trivial

/-- If CtxWF holds for a two-element context, the names are distinct. -/
theorem ctxWF_two_distinct {x y : String} {τ₁ τ₂ : Ty} {m₁ m₂ : Modality}
    (hwf : CtxWF [(x, τ₁, m₁), (y, τ₂, m₂)]) : x ≠ y := by
  intro heq
  subst heq
  exact hwf.1 τ₂ m₂ (List.Mem.head _)

-- ================================================================
-- §17. Substitution Commutes with Box/Pair Structure
-- ================================================================

/-- Substitution distributes into application. -/
theorem subst_app_eq (x : String) (v f a : Expr) :
    subst x v (.app f a) = .app (subst x v f) (subst x v a) := rfl

/-- Substitution distributes into pair. -/
theorem subst_pair_eq (x : String) (v e₁ e₂ : Expr) :
    subst x v (.pair e₁ e₂) = .pair (subst x v e₁) (subst x v e₂) := rfl

/-- Substitution distributes into box. -/
theorem subst_box_eq (x : String) (v e : Expr) :
    subst x v (.box e) = .box (subst x v e) := rfl

-- ================================================================
-- §18. Typing Inversion Helpers
-- ================================================================

/-- If a closed value v has type σ in the empty context, then for any Γ,
    subst x v (.var x) has the same type in Γ. -/
theorem closed_subst_var_typing {Γ : TyCtx} {x : String} {σ : Ty}
    {v : Expr} {ρ : EffectRow}
    (hv : Typing [] v σ ρ) :
    Typing Γ (subst x v (.var x)) σ ρ := by
  rw [subst_var_hit x v]
  exact typing_closed_weakening Γ hv

/-- Substitution into a non-matching variable yields the same variable. -/
theorem subst_var_miss_eq {x y : String} {v : Expr} (hne : y ≠ x) :
    subst x v (.var y) = .var y := by
  simp [subst, hne]

-- ================================================================
-- §19. Combined Weakening + Append Theorems
-- ================================================================

/-- A closed term can be typed in any extended context Γ ++ Δ. -/
theorem typing_closed_in_any_ctx {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Γ Δ : TyCtx) (ht : Typing [] e τ ρ) : Typing (Γ ++ Δ) e τ ρ :=
  typing_closed_weakening (Γ ++ Δ) ht

/-- Multiple weakenings: if Γ ⊢ e : τ ! ρ and both m₁, m₂ allow weakening,
    then (x₂,σ₂,m₂)::(x₁,σ₁,m₁)::Γ ⊢ e : τ ! ρ. -/
theorem typing_double_weaken {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    {x₁ : String} {σ₁ : Ty} {m₁ : Modality}
    {x₂ : String} {σ₂ : Ty} {m₂ : Modality}
    (hw₁ : m₁.allowsWeakening = true) (hw₂ : m₂.allowsWeakening = true)
    (ht : Typing Γ e τ ρ) :
    Typing ((x₂, σ₂, m₂) :: (x₁, σ₁, m₁) :: Γ) e τ ρ :=
  Typing.Weak (Typing.Weak ht hw₁) hw₂

-- ================================================================
-- §20. Effect Row Interaction with Substitution Context
-- ================================================================

/-- Pure typing is preserved under closed substitution (variable hit case). -/
theorem subst_var_hit_pure_typing {Γ : TyCtx} {x : String} {σ : Ty} {v : Expr}
    (hv_ty : Typing [] v σ pureRow) :
    Typing Γ (subst x v (.var x)) σ pureRow := by
  rw [subst_var_hit]
  exact typing_closed_weakening Γ hv_ty

/-- Effect subsumption composes with closed-term weakening. -/
theorem typing_closed_sub {e : Expr} {τ : Ty} {ρ ρ' : EffectRow}
    (Γ : TyCtx) (ht : Typing [] e τ ρ) (hsub : effectSubrow ρ ρ') :
    Typing Γ e τ ρ' :=
  Typing.Sub (typing_closed_weakening Γ ht) hsub

-- ================================================================
-- §21. Substitution Lemma — Closed-Value Case (Key Theorem)
-- ================================================================

/-- Substitution lemma for closed values (the CBV case, variable-only).
    This is the base case of the full substitution lemma. -/
theorem substitution_var_case {Γ : TyCtx} {x : String} {σ : Ty} {m : Modality}
    {v : Expr} {ρ_v : EffectRow}
    (_hmem : (x, σ, m) ∈ ((x, σ, m) :: Γ))
    (hv_ty : Typing [] v σ ρ_v)
    (_hval : IsValue v) :
    Typing Γ (subst x v (.var x)) σ ρ_v := by
  rw [subst_var_hit]
  exact typing_closed_weakening Γ hv_ty

/-- Substitution on a non-matching variable preserves typing in the tail context. -/
theorem substitution_var_miss_case {Γ : TyCtx} {x y : String} {τ : Ty}
    {n : Modality} {v : Expr}
    (hne : y ≠ x)
    (hmem : (y, τ, n) ∈ Γ) :
    Typing Γ (subst x v (.var y)) τ pureRow := by
  rw [subst_var_miss_eq hne]
  exact Typing.Var hmem

-- ================================================================
-- §22. Additional Context Lemmas
-- ================================================================

/-- Membership is decidable for our context type (law of excluded middle). -/
theorem ctx_mem_or_not_mem (e : String × Ty × Modality) (Γ : TyCtx) :
    e ∈ Γ ∨ e ∉ Γ :=
  Classical.em (e ∈ Γ)

/-- If e is in Γ₁ ++ Γ₂, then e is in Γ₁ or in Γ₂. -/
theorem ctx_mem_append_cases {e : String × Ty × Modality} {Γ₁ Γ₂ : TyCtx}
    (h : e ∈ Γ₁ ++ Γ₂) : e ∈ Γ₁ ∨ e ∈ Γ₂ :=
  List.mem_append.mp h

/-- Cons membership splits into head equality or tail membership. -/
theorem ctx_mem_cons_cases {e hd : String × Ty × Modality} {Γ : TyCtx}
    (h : e ∈ hd :: Γ) : e = hd ∨ e ∈ Γ :=
  List.mem_cons.mp h

-- ================================================================
-- §23. Substitution and Effect Purity
-- ================================================================

/-- If a value v is typed pure in the empty context,
    then for the hit case, substitution is well-typed,
    and for the miss case, the original variable is well-typed. -/
theorem subst_var_pure_in_ctx_hit {Γ : TyCtx} {x : String} {σ : Ty} {v : Expr}
    (hv : Typing [] v σ pureRow) :
    Typing Γ (subst x v (.var x)) σ pureRow := by
  rw [subst_var_hit]
  exact typing_closed_weakening Γ hv

theorem subst_var_pure_in_ctx_miss {Γ : TyCtx} {x y : String} {τ : Ty}
    {m : Modality} {v : Expr}
    (hne : y ≠ x) (hmem : (y, τ, m) ∈ Γ) :
    Typing Γ (subst x v (.var y)) τ pureRow := by
  rw [subst_var_miss_eq hne]
  exact Typing.Var hmem

-- ================================================================
-- §24. CtxWF and Append
-- ================================================================

/-- A name not in either part of an appended context is not in the whole. -/
theorem ctxWF_name_not_in_append {x : String} {Γ Δ : TyCtx}
    (hg : ∀ σ n, (x, σ, n) ∉ Γ) (hd : ∀ σ n, (x, σ, n) ∉ Δ) :
    ∀ σ n, (x, σ, n) ∉ (Γ ++ Δ) := by
  intro σ n hmem
  rcases List.mem_append.mp hmem with h | h
  · exact hg σ n h
  · exact hd σ n h

/-- Well-formedness of the empty context is trivially decidable. -/
theorem ctxWF_nil_dec : CtxWF ([] : TyCtx) = True := rfl

-- ================================================================
-- §25. Substitution Shadow Lemmas
-- ================================================================

/-- For lambda: if the binder name equals x, then subst x v (.lam x m τ e) = .lam x m τ e. -/
theorem subst_lam_shadow_eq (x : String) (v : Expr) (m : Modality) (τ : Ty) (e : Expr) :
    subst x v (.lam x m τ e) = .lam x m τ e := by
  simp [subst]

/-- For letB: if the binder equals x, substitution only affects the scrutinee. -/
theorem subst_letB_shadow_eq (x : String) (v : Expr) (e body : Expr) :
    subst x v (.letB x e body) = .letB x (subst x v e) body := by
  simp [subst]

/-- For letP: if the first binder equals x, the body is not substituted. -/
theorem subst_letP_shadow_left (x y : String) (v : Expr) (e body : Expr) :
    subst x v (.letP x y e body) = .letP x y (subst x v e) body := by
  simp [subst]

/-- For letP: if the second binder equals x, the body is not substituted. -/
theorem subst_letP_shadow_right (x y : String) (v : Expr) (e body : Expr) :
    subst y v (.letP x y e body) = .letP x y (subst y v e) body := by
  simp [subst]

-- ================================================================
-- §26. Closed Value Typing Lifts Through Substitution
-- ================================================================

/-- If [] ⊢ v : σ ! pureRow, then substituting v for x in a variable
    and then applying effect subsumption, the result is typable. -/
theorem subst_closed_pure_sub {Γ : TyCtx} {x : String} {σ : Ty}
    {v : Expr} {ρ : EffectRow}
    (hv : Typing [] v σ pureRow) :
    Typing Γ (subst x v (.var x)) σ ρ := by
  rw [subst_var_hit]
  exact typing_pure_sub ρ (typing_closed_weakening Γ hv)

/-- Weakening + subsumption pipeline:
    a closed pure term can be placed anywhere with any effect row. -/
theorem typing_closed_pure_any {e : Expr} {τ : Ty}
    (Γ : TyCtx) (ρ : EffectRow) (ht : Typing [] e τ pureRow) :
    Typing Γ e τ ρ :=
  typing_pure_sub ρ (typing_closed_weakening Γ ht)

-- ================================================================
-- §27. Freshness and Substitution Interaction
-- ================================================================

/-- If x is fresh in e (no inner binder shadows x) and y = x,
    then subst x v (.lam y m τ e) = .lam y m τ e (shadow case). -/
theorem fresh_subst_lam_shadow {x : String} {v : Expr} {m : Modality}
    {τ : Ty} {e : Expr} :
    subst x v (.lam x m τ e) = .lam x m τ e := by
  simp [subst]

/-- If x is fresh in e and y ≠ x, substitution recurses into the body. -/
theorem fresh_subst_lam_not_shadow {x y : String} {v : Expr} {m : Modality}
    {τ : Ty} {e : Expr} (hne : y ≠ x) :
    subst x v (.lam y m τ e) = .lam y m τ (subst x v e) := by
  simp [subst, hne]

-- ================================================================
-- §28. Multi-Step Weakening Interaction
-- ================================================================

/-- Weakening by three variables at once (all must be weakenable). -/
theorem typing_weaken_three {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    {x₁ : String} {σ₁ : Ty} {m₁ : Modality}
    {x₂ : String} {σ₂ : Ty} {m₂ : Modality}
    {x₃ : String} {σ₃ : Ty} {m₃ : Modality}
    (hw₁ : m₁.allowsWeakening = true)
    (hw₂ : m₂.allowsWeakening = true)
    (hw₃ : m₃.allowsWeakening = true)
    (h : Typing Γ e τ ρ) :
    Typing ((x₃, σ₃, m₃) :: (x₂, σ₂, m₂) :: (x₁, σ₁, m₁) :: Γ) e τ ρ :=
  Typing.Weak (Typing.Weak (Typing.Weak h hw₁) hw₂) hw₃

-- ================================================================
-- §29. Substitution Preserves Pair Values
-- ================================================================

/-- If both components of a pair are values after substitution,
    then the pair after substitution is a value. -/
theorem subst_pair_is_value {x : String} {v e₁ e₂ : Expr}
    (ih₁ : IsValue (subst x v e₁))
    (ih₂ : IsValue (subst x v e₂)) :
    IsValue (subst x v (.pair e₁ e₂)) := by
  simp only [subst]
  exact IsValue.pair ih₁ ih₂

-- ================================================================
-- §30. CtxWF Inversion
-- ================================================================

/-- In a well-formed context with at least two entries,
    the second entry's name is fresh in the tail. -/
theorem ctxWF_second_fresh {x y : String} {τ₁ τ₂ : Ty}
    {m₁ m₂ : Modality} {Γ : TyCtx}
    (hwf : CtxWF ((x, τ₁, m₁) :: (y, τ₂, m₂) :: Γ)) :
    ∀ σ n, (y, σ, n) ∉ Γ :=
  hwf.2.1

/-- The tail of the tail is well-formed. -/
theorem ctxWF_tail_tail {x y : String} {τ₁ τ₂ : Ty}
    {m₁ m₂ : Modality} {Γ : TyCtx}
    (hwf : CtxWF ((x, τ₁, m₁) :: (y, τ₂, m₂) :: Γ)) :
    CtxWF Γ :=
  hwf.2.2

end Sounio.Substitution
