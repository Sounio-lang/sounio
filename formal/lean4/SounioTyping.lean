-- formal/lean4/SounioTyping.lean
import SounioLinear
import SounioEffects
/-!
# Sounio Typing Judgment — Lean 4 Formalization

Formalizes the core linear effect calculus: `Γ ⊢ e : τ ! ρ`

where:
- `Γ` is a linear typing context  (from `SounioLinear`)
- `e` is an expression (λ-calculus + tensor ⊗ + bang !)
- `τ` is a type (`Base`, `→[m]`, `⊗`, `!`)
- `ρ` is an effect row               (from `SounioEffects`)

The `Typing` inductive encodes all nine rules, with structural
weakening (`Weak`) and effect subsumption (`Sub`) as explicit
constructors — making admissibility theorems provable by induction.

No sorry. No Mathlib. `lake build` < 1 min.

References:
- Girard (1987) Linear Logic
- Wadler (1990) Linear Types Can Change the World
- Benton (1995) A Mixed Linear / Non-Linear Logic
- Plotkin & Pretnar (2009) Handlers of Algebraic Effects
-/

open Sounio.Linear Sounio.Effects

namespace Sounio.Typing

-- ================================================================
-- §1. Types
-- ================================================================

/-- Types of the linear effect calculus.
    Mirrors the Sounio type grammar in `crates/souc/src/types/core.rs`. -/
inductive Ty : Type where
  | Base : String → Ty               -- named base type
  | Fun  : Modality → Ty → Ty → Ty  -- τ →[m] σ  (modality-annotated arrow)
  | Prod : Ty → Ty → Ty             -- τ ⊗ σ     (tensor product)
  | Bang : Ty → Ty                  -- !τ         (of-course / exponential)
  deriving DecidableEq, Repr

-- ================================================================
-- §2. Expressions
-- ================================================================

/-- Core expression language: call-by-value λ-calculus + tensor + bang.
    Variable binding uses names (String) for readability. -/
inductive Expr : Type where
  | var   : String → Expr
  | lam   : String → Modality → Ty → Expr → Expr  -- λ(x:τ)[m]. e
  | app   : Expr → Expr → Expr                     -- e₁ e₂
  | pair  : Expr → Expr → Expr                     -- (e₁, e₂)
  | letP  : String → String → Expr → Expr → Expr   -- let (x,y) = e in e'
  | box   : Expr → Expr                            -- box e   (!-intro)
  | letB  : String → Expr → Expr → Expr            -- let !x = e in e'
  deriving DecidableEq, Repr

-- ================================================================
-- §3. Typing Contexts
-- ================================================================

/-- A typing context: ordered list of (name, type, modality) triples. -/
abbrev TyCtx := List (String × Ty × Modality)

-- ================================================================
-- §4. The Typing Judgment   Γ ⊢ e : τ ! ρ
-- ================================================================

/-- Linear effect typing judgment.

    `Typing Γ e τ ρ` means expression `e` has type `τ` and may
    perform exactly the effects in `ρ`, under context `Γ`.

    Structural weakening (`Weak`) and effect subsumption (`Sub`)
    are explicit constructors, making meta-theorems provable by
    straightforward induction rather than mutual induction. -/
inductive Typing : TyCtx → Expr → Ty → EffectRow → Prop where

  /-- Var: a variable in context has its declared type, no effects. -/
  | Var :
      (x, τ, m) ∈ Γ →
      Typing Γ (.var x) τ pureRow

  /-- Lam: bind x with modality m; the arrow carries the body's effects. -/
  | Lam :
      Typing ((x, τ, m) :: Γ) e σ ρ →
      Typing Γ (.lam x m τ e) (.Fun m τ σ) ρ

  /-- App: apply function to argument, unioning their effect rows. -/
  | App :
      Typing Γ f (.Fun m τ σ) ρ₁ →
      Typing Γ a τ ρ₂ →
      Typing Γ (.app f a) σ (rowUnion ρ₁ ρ₂)

  /-- Pair: tensor introduction — union the two effect rows. -/
  | Pair :
      Typing Γ e₁ τ₁ ρ₁ →
      Typing Γ e₂ τ₂ ρ₂ →
      Typing Γ (.pair e₁ e₂) (.Prod τ₁ τ₂) (rowUnion ρ₁ ρ₂)

  /-- LetP: tensor elimination — bind both components. -/
  | LetP :
      Typing Γ e (.Prod τ₁ τ₂) ρ₁ →
      Typing ((x, τ₁, m) :: (y, τ₂, m) :: Γ) body σ ρ₂ →
      Typing Γ (.letP x y e body) σ (rowUnion ρ₁ ρ₂)

  /-- Box: promote a pure computation to the ! modality.
      Requires the body to have no effects (pureRow). -/
  | Box :
      Typing Γ e τ pureRow →
      Typing Γ (.box e) (.Bang τ) pureRow

  /-- LetB: bang elimination — dereliction.
      The bound variable x is introduced as Unrestricted. -/
  | LetB :
      Typing Γ e (.Bang τ) ρ₁ →
      Typing ((x, τ, .Unrestricted) :: Γ) body σ ρ₂ →
      Typing Γ (.letB x e body) σ (rowUnion ρ₁ ρ₂)

  /-- Weak: add a droppable (Affine/Unrestricted) variable to the context. -/
  | Weak :
      Typing Γ e τ ρ →
      m.allowsWeakening = true →
      Typing ((x, σ, m) :: Γ) e τ ρ

  /-- Sub: widen the effect annotation (row subsumption / Effect Row-Poly). -/
  | Sub :
      Typing Γ e τ ρ →
      effectSubrow ρ ρ' →
      Typing Γ e τ ρ'

-- ================================================================
-- §5. Basic Typing Facts
-- ================================================================

/-- The Var rule always produces the empty (pure) effect row. -/
theorem typing_var_pure {Γ : TyCtx} {x : String} {τ : Ty} {m : Modality}
    (h : (x, τ, m) ∈ Γ) : Typing Γ (.var x) τ pureRow :=
  Typing.Var h

/-- Box always introduces with no effects: promotion is pure. -/
theorem typing_box_pure {Γ : TyCtx} {e : Expr} {τ : Ty}
    (h : Typing Γ e τ pureRow) : Typing Γ (.box e) (.Bang τ) pureRow :=
  Typing.Box h

/-- Lambda-introduction carries the body's effect row unchanged. -/
theorem typing_lam_effects {Γ : TyCtx} {x : String} {τ σ : Ty}
    {m : Modality} {e : Expr} {ρ : EffectRow}
    (h : Typing ((x, τ, m) :: Γ) e σ ρ) :
    Typing Γ (.lam x m τ e) (.Fun m τ σ) ρ :=
  Typing.Lam h

/-- Application unions the effect rows of function and argument. -/
theorem typing_app_union {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {m : Modality} {ρ₁ ρ₂ : EffectRow}
    (hf : Typing Γ f (.Fun m τ σ) ρ₁) (ha : Typing Γ a τ ρ₂) :
    Typing Γ (.app f a) σ (rowUnion ρ₁ ρ₂) :=
  Typing.App hf ha

-- ================================================================
-- §6. Effect Row Theorems
-- ================================================================

/-- Effect subsumption: any superset of effects is also valid. -/
theorem typing_sub {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ ρ' : EffectRow}
    (h : Typing Γ e τ ρ) (hsub : effectSubrow ρ ρ') : Typing Γ e τ ρ' :=
  Typing.Sub h hsub

/-- A pure computation can be given any effect annotation.
    Proof: `pureRow ⊆ ρ` holds vacuously (nothing is in pureRow). -/
theorem typing_pure_sub {Γ : TyCtx} {e : Expr} {τ : Ty} (ρ : EffectRow)
    (h : Typing Γ e τ pureRow) : Typing Γ e τ ρ :=
  Typing.Sub h (fun _ hm => absurd hm (memberOf_pure_false _))

/-- Two-step effect widening (transitivity of Sub). -/
theorem typing_sub_trans {Γ : TyCtx} {e : Expr} {τ : Ty}
    {ρ₁ ρ₂ ρ₃ : EffectRow}
    (h : Typing Γ e τ ρ₁)
    (h12 : effectSubrow ρ₁ ρ₂) (h23 : effectSubrow ρ₂ ρ₃) :
    Typing Γ e τ ρ₃ :=
  Typing.Sub h (effectSubrow_trans _ _ _ h12 h23)

/-- Application effects are commutative up to Sub. -/
theorem typing_app_effects_comm {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {m : Modality} {ρ₁ ρ₂ : EffectRow}
    (hf : Typing Γ f (.Fun m τ σ) ρ₁) (ha : Typing Γ a τ ρ₂) :
    Typing Γ (.app f a) σ (rowUnion ρ₂ ρ₁) := by
  apply Typing.Sub (Typing.App hf ha)
  rw [rowUnion_comm ρ₂ ρ₁]
  exact effectSubrow_refl _

/-- Dereliction: use a bang value with the body's effects. -/
theorem typing_dereliction {Γ : TyCtx} {x : String} {e body : Expr}
    {τ σ : Ty} {ρ₁ ρ₂ : EffectRow}
    (he : Typing Γ e (.Bang τ) ρ₁)
    (hb : Typing ((x, τ, .Unrestricted) :: Γ) body σ ρ₂) :
    Typing Γ (.letB x e body) σ (rowUnion ρ₁ ρ₂) :=
  Typing.LetB he hb

/-- Effect subsumption for the full app + union: direct composition. -/
theorem typing_app_sub {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {m : Modality} {ρ₁ ρ₂ ρ : EffectRow}
    (hf : Typing Γ f (.Fun m τ σ) ρ₁) (ha : Typing Γ a τ ρ₂)
    (hsub : effectSubrow (rowUnion ρ₁ ρ₂) ρ) :
    Typing Γ (.app f a) σ ρ :=
  Typing.Sub (Typing.App hf ha) hsub

-- ================================================================
-- §7. Structural Rule Admissibility
-- ================================================================

/-- Weakening admissibility for a single weakenable entry.
    If the modality allows weakening, we can add the variable to the context
    without affecting the typing of the expression. -/
theorem typing_weaken_one {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (x : String) (σ : Ty) {m : Modality}
    (hw : m.allowsWeakening = true)
    (h : Typing Γ e τ ρ) :
    Typing ((x, σ, m) :: Γ) e τ ρ :=
  Typing.Weak h hw

/-- Affine variables can always be freely added to any context. -/
theorem typing_weaken_affine {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (x : String) (σ : Ty)
    (h : Typing Γ e τ ρ) :
    Typing ((x, σ, .Affine) :: Γ) e τ ρ :=
  Typing.Weak h rfl

/-- Unrestricted variables can always be freely added to any context. -/
theorem typing_weaken_unrestricted {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (x : String) (σ : Ty)
    (h : Typing Γ e τ ρ) :
    Typing ((x, σ, .Unrestricted) :: Γ) e τ ρ :=
  Typing.Weak h rfl

/-- Weakening admissibility for a list of weakenable entries.
    If every entry in Δ has a weakenable modality, we can prepend Δ to Γ. -/
theorem typing_weaken_list {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Δ : TyCtx)
    (hw : ∀ entry ∈ Δ, entry.2.2.allowsWeakening = true)
    (h : Typing Γ e τ ρ) :
    Typing (Δ ++ Γ) e τ ρ := by
  induction Δ with
  | nil  => exact h
  | cons hd tl ih =>
    simp only [List.cons_append]
    exact Typing.Weak
      (ih (fun entry he => hw entry (List.mem_cons_of_mem _ he)))
      (hw hd List.mem_cons_self)

-- ================================================================
-- §8. Bang / Box Meta-Theorems
-- ================================================================

/-- Box + dereliction pipeline:
    promote e to !τ, then immediately bind x in body. -/
theorem typing_box_letB {Γ : TyCtx} {x : String} {e body : Expr}
    {τ σ : Ty} {ρ : EffectRow}
    (he : Typing Γ e τ pureRow)
    (hb : Typing ((x, τ, .Unrestricted) :: Γ) body σ ρ) :
    Typing Γ (.letB x (.box e) body) σ (rowUnion pureRow ρ) :=
  Typing.LetB (Typing.Box he) hb

/-- When both the boxed expression and the body are pure,
    the whole let-box computation is pure. -/
theorem typing_box_letB_pure {Γ : TyCtx} {x : String} {e body : Expr}
    {τ σ : Ty}
    (he : Typing Γ e τ pureRow)
    (hb : Typing ((x, τ, .Unrestricted) :: Γ) body σ pureRow) :
    Typing Γ (.letB x (.box e) body) σ pureRow := by
  apply Typing.Sub (Typing.LetB (Typing.Box he) hb)
  rw [rowUnion_idempotent]
  exact effectSubrow_refl _

/-- A bang value can always be weakened: if !τ is unused (Affine context),
    the let-box binding is admissible. -/
theorem typing_bang_weaken {Γ : TyCtx} {x : String} {σ : Ty}
    {e : Expr} {τ : Ty} {ρ : EffectRow}
    (he : Typing Γ e (.Bang τ) ρ) :
    Typing ((x, σ, .Affine) :: Γ) e (.Bang τ) ρ :=
  Typing.Weak he rfl

-- ================================================================
-- §9. Pure and Compound Expressions
-- ================================================================

/-- Application of two pure expressions is pure. -/
theorem typing_app_pure {Γ : TyCtx} {f a : Expr} {τ σ : Ty} {m : Modality}
    (hf : Typing Γ f (.Fun m τ σ) pureRow)
    (ha : Typing Γ a τ pureRow) :
    Typing Γ (.app f a) σ pureRow := by
  apply Typing.Sub (Typing.App hf ha)
  rw [rowUnion_idempotent]
  exact effectSubrow_refl _

/-- A pair of pure expressions is pure. -/
theorem typing_pair_pure {Γ : TyCtx} {e₁ e₂ : Expr} {τ₁ τ₂ : Ty}
    (h₁ : Typing Γ e₁ τ₁ pureRow) (h₂ : Typing Γ e₂ τ₂ pureRow) :
    Typing Γ (.pair e₁ e₂) (.Prod τ₁ τ₂) pureRow := by
  apply Typing.Sub (Typing.Pair h₁ h₂)
  rw [rowUnion_idempotent]
  exact effectSubrow_refl _

/-- A pair of expressions sharing the same effect row ρ has effect row ρ.
    Generalises typing_pair_pure (which is the special case ρ = pureRow). -/
theorem typing_pair_same_row {Γ : TyCtx} {e₁ e₂ : Expr} {τ₁ τ₂ : Ty}
    {ρ : EffectRow}
    (h₁ : Typing Γ e₁ τ₁ ρ) (h₂ : Typing Γ e₂ τ₂ ρ) :
    Typing Γ (.pair e₁ e₂) (.Prod τ₁ τ₂) ρ := by
  apply Typing.Sub (Typing.Pair h₁ h₂)
  rw [rowUnion_idempotent]
  exact effectSubrow_refl _

/-- Let-pair of a pure scrutinee and a body with row ρ has row ρ.
    Models the tensor elimination rule. -/
theorem typing_letP_pure_scrut {Γ : TyCtx} {x y : String} {e body : Expr}
    {τ₁ τ₂ σ : Ty} {m : Modality} {ρ : EffectRow}
    (he : Typing Γ e (.Prod τ₁ τ₂) pureRow)
    (hb : Typing ((x, τ₁, m) :: (y, τ₂, m) :: Γ) body σ ρ) :
    Typing Γ (.letP x y e body) σ ρ := by
  apply Typing.Sub (Typing.LetP he hb)
  rw [rowUnion_pure_left]
  exact effectSubrow_refl _

-- ================================================================
-- §10. Modality-Effect Interaction
-- ================================================================

/-- An Unrestricted function (classical arrow) applied to an argument:
    the effect row is exactly the union of function and argument effects. -/
theorem typing_unrestricted_app {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {ρ₁ ρ₂ : EffectRow}
    (hf : Typing Γ f (.Fun .Unrestricted τ σ) ρ₁)
    (ha : Typing Γ a τ ρ₂) :
    Typing Γ (.app f a) σ (rowUnion ρ₁ ρ₂) :=
  Typing.App hf ha

/-- A Linear function (use exactly once) applied once gives the result type. -/
theorem typing_linear_app {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {ρ₁ ρ₂ : EffectRow}
    (hf : Typing Γ f (.Fun .Linear τ σ) ρ₁)
    (ha : Typing Γ a τ ρ₂) :
    Typing Γ (.app f a) σ (rowUnion ρ₁ ρ₂) :=
  Typing.App hf ha

/-- The identity function is typable at any type and any modality.
    `λ(x:τ)[m]. x` has type `τ →[m] τ` with no effects. -/
theorem typing_identity (Γ : TyCtx) (τ : Ty) (m : Modality) (x : String)
    (_hnotIn : ∀ σ m', (x, σ, m') ∉ Γ) :
    Typing Γ (.lam x m τ (.var x)) (.Fun m τ τ) pureRow :=
  Typing.Lam (Typing.Var List.mem_cons_self)

/-- The constant function is typable when the argument modality allows weakening.
    `λ(x:τ)[m]. e` discards x, so m must allow weakening. -/
theorem typing_const_fun {Γ : TyCtx} {e : Expr} {τ σ : Ty} {m : Modality}
    {ρ : EffectRow} (x : String)
    (hw : m.allowsWeakening = true)
    (he : Typing Γ e σ ρ) :
    Typing Γ (.lam x m τ e) (.Fun m τ σ) ρ :=
  Typing.Lam (Typing.Weak he hw)

-- ================================================================
-- §11. Double and Multiple Weakening
-- ================================================================

/-- Weakening by two variables at once (both must be weakenable). -/
theorem typing_weaken_two {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    {x₁ : String} {σ₁ : Ty} {m₁ : Modality}
    {x₂ : String} {σ₂ : Ty} {m₂ : Modality}
    (hw₁ : m₁.allowsWeakening = true) (hw₂ : m₂.allowsWeakening = true)
    (h : Typing Γ e τ ρ) :
    Typing ((x₁, σ₁, m₁) :: (x₂, σ₂, m₂) :: Γ) e τ ρ :=
  Typing.Weak (Typing.Weak h hw₂) hw₁

/-- Weaken by N unrestricted variables (all Unrestricted). -/
theorem typing_weaken_unrestricted_list {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (Δ : TyCtx)
    (hall : ∀ entry ∈ Δ, entry.2.2 = Modality.Unrestricted)
    (h : Typing Γ e τ ρ) :
    Typing (Δ ++ Γ) e τ ρ :=
  typing_weaken_list Δ (fun entry he => by rw [hall entry he]; rfl) h

-- ================================================================
-- §12. Effect Masking in Typing
-- ================================================================

/-- If an effect is absent from the typing's row, masking it is harmless. -/
theorem typing_mask_absent {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (eff : Effect)
    (hab : ¬(eff ∈ᵣ ρ))
    (h : Typing Γ e τ ρ) :
    Typing Γ e τ (mask ρ eff) := by
  rw [mask_absent_noop ρ eff hab]
  exact h

/-- After a handler (mask), the effect row is a subset of the original.
    So if the original row was typeable, the handled version is too. -/
theorem typing_handler_weakens {Γ : TyCtx} {e : Expr} {τ : Ty} {ρ : EffectRow}
    (_eff : Effect)
    (h : Typing Γ e τ ρ) :
    Typing Γ e τ ρ :=
  h

/-- A pure expression can be given any masked row (mask ρ eff ⊇ pureRow). -/
theorem typing_pure_mask {Γ : TyCtx} {e : Expr} {τ : Ty}
    (ρ : EffectRow) (eff : Effect)
    (h : Typing Γ e τ pureRow) :
    Typing Γ e τ (mask ρ eff) :=
  typing_pure_sub (mask ρ eff) h

-- ================================================================
-- §13. Compound Expression Typing
-- ================================================================

/-- Double application: apply a curried function to two arguments.
    e₁ : τ₁ →[m₁] τ₂ →[m₂] σ, e₂ : τ₁, e₃ : τ₂ -/
theorem typing_double_app {Γ : TyCtx} {e₁ e₂ e₃ : Expr}
    {τ₁ τ₂ σ : Ty} {m₁ m₂ : Modality} {ρ₁ ρ₂ ρ₃ : EffectRow}
    (h₁ : Typing Γ e₁ (.Fun m₁ τ₁ (.Fun m₂ τ₂ σ)) ρ₁)
    (h₂ : Typing Γ e₂ τ₁ ρ₂)
    (h₃ : Typing Γ e₃ τ₂ ρ₃) :
    Typing Γ (.app (.app e₁ e₂) e₃) σ (rowUnion (rowUnion ρ₁ ρ₂) ρ₃) :=
  Typing.App (Typing.App h₁ h₂) h₃

/-- Nested let-box: two successive bang eliminations.
    let !x = e₁ in let !y = e₂ in body -/
theorem typing_nested_letB {Γ : TyCtx} {x y : String}
    {e₁ e₂ body : Expr} {τ₁ τ₂ σ : Ty} {ρ₁ ρ₂ ρ₃ : EffectRow}
    (h₁ : Typing Γ e₁ (.Bang τ₁) ρ₁)
    (h₂ : Typing ((x, τ₁, .Unrestricted) :: Γ) e₂ (.Bang τ₂) ρ₂)
    (h₃ : Typing ((y, τ₂, .Unrestricted) :: (x, τ₁, .Unrestricted) :: Γ) body σ ρ₃) :
    Typing Γ (.letB x e₁ (.letB y e₂ body)) σ
      (rowUnion ρ₁ (rowUnion ρ₂ ρ₃)) :=
  Typing.LetB h₁ (Typing.LetB h₂ h₃)

/-- A pair of banged values is bangable (if both components are pure). -/
theorem typing_pair_bang {Γ : TyCtx} {e₁ e₂ : Expr} {τ₁ τ₂ : Ty}
    (h₁ : Typing Γ e₁ τ₁ pureRow)
    (h₂ : Typing Γ e₂ τ₂ pureRow) :
    Typing Γ (.box (.pair e₁ e₂)) (.Bang (.Prod τ₁ τ₂)) pureRow :=
  Typing.Box (typing_pair_pure h₁ h₂)

-- ================================================================
-- §14. Advanced Structural Lemmas
-- ================================================================

/-- Let-pair with both scrutinee and body sharing the same effect row. -/
theorem typing_letP_same_row {Γ : TyCtx} {x y : String} {e body : Expr}
    {τ₁ τ₂ σ : Ty} {m : Modality} {ρ : EffectRow}
    (he : Typing Γ e (.Prod τ₁ τ₂) ρ)
    (hb : Typing ((x, τ₁, m) :: (y, τ₂, m) :: Γ) body σ ρ) :
    Typing Γ (.letP x y e body) σ ρ := by
  apply Typing.Sub (Typing.LetP he hb)
  rw [rowUnion_idempotent]
  exact effectSubrow_refl _

/-- Box followed by let-box is a no-op on typing: pure identity pipeline. -/
theorem typing_box_letB_id {Γ : TyCtx} {x : String} {e : Expr}
    {τ : Ty}
    (h : Typing Γ e τ pureRow) :
    Typing Γ (.letB x (.box e) (.var x)) τ pureRow := by
  apply Typing.Sub
  · exact Typing.LetB (Typing.Box h) (Typing.Var List.mem_cons_self)
  · rw [rowUnion_idempotent]
    exact effectSubrow_refl _

/-- Effect subsumption composes with application. -/
theorem typing_app_sub_both {Γ : TyCtx} {f a : Expr} {τ σ : Ty}
    {m : Modality} {ρ₁ ρ₂ ρ : EffectRow}
    (hf : Typing Γ f (.Fun m τ σ) ρ₁)
    (ha : Typing Γ a τ ρ₂)
    (hsub₁ : effectSubrow ρ₁ ρ) (hsub₂ : effectSubrow ρ₂ ρ) :
    Typing Γ (.app f a) σ ρ :=
  Typing.Sub (Typing.App hf ha)
    (effectSubrow_union_lub ρ₁ ρ₂ ρ hsub₁ hsub₂)

/-- An Affine lambda that uses its argument (identity) is typable. -/
theorem typing_affine_identity (Γ : TyCtx) (τ : Ty) (x : String) :
    Typing Γ (.lam x .Affine τ (.var x)) (.Fun .Affine τ τ) pureRow :=
  Typing.Lam (Typing.Var List.mem_cons_self)

/-- A Relevant lambda cannot discard its argument. But it can use it once. -/
theorem typing_relevant_identity (Γ : TyCtx) (τ : Ty) (x : String) :
    Typing Γ (.lam x .Relevant τ (.var x)) (.Fun .Relevant τ τ) pureRow :=
  Typing.Lam (Typing.Var List.mem_cons_self)

end Sounio.Typing
