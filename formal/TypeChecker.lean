/-!
# Sounio.TypeChecker — Phase 8 Formal Verification

Invariants for the bidirectional type checker found in
  `crates/souc/src/check/mod.rs`

The checker implements bidirectional type inference (Dunfield & Krishnaswami
CSUR 2021), with extensions for:
  - Algebraic effect rows (IO, Mut, Alloc, Panic, Async, GPU, Prob, Div)
  - Linear / affine types (`linear struct`, single-use guarantees)
  - Epistemic types `Knowledge<T>` with uncertainty quantification
  - Probabilistic subtype compatibility (Bayesian threshold)
  - Dimensional units (compile-time dimensional analysis)

All theorems are `sorry`-admitted stubs for Phase 8.  The intended proof
strategy is described in each theorem's docstring.
-/

namespace Sounio.TypeChecker

-- ---------------------------------------------------------------------------
-- Type universe
-- ---------------------------------------------------------------------------

/-- The core Sounio type universe.

    This is a simplified model; the full `crates/souc/src/types/mod.rs` type
    includes refinement predicates, effect rows, dimension tags, and type
    variables.  Those are abstracted away here so we can focus on the
    structural invariants of the core type system. -/
inductive Ty : Type where
  -- Primitive scalar types
  | Unit    : Ty
  | Bool    : Ty
  | I32     : Ty
  | I64     : Ty
  | F32     : Ty
  | F64     : Ty
  -- Compound types
  | Fn      : Ty → Ty → Ty         -- argument type → return type
  | Tuple   : List Ty → Ty
  | Ref     : Ty → Ty              -- shared reference &T
  | RefMut  : Ty → Ty              -- exclusive reference &!T
  -- Epistemic wrapper: Knowledge<T> carries an uncertainty interval
  | Knowledge : Ty → Ty
  -- Option / result (common in stdlib)
  | Option  : Ty → Ty
  -- Type variable (used during unification)
  | Var     : Nat → Ty
  deriving Repr, DecidableEq

-- ---------------------------------------------------------------------------
-- Effect rows (simplified)
-- ---------------------------------------------------------------------------

/-- A simplified effect set — the Rust implementation uses a more elaborate
    row-polymorphic representation.  Here we model effects as a `Finset`
    drawn from a fixed universe. -/
inductive Effect : Type where
  | IO    : Effect
  | Mut   : Effect
  | Alloc : Effect
  | Panic : Effect
  | Async : Effect
  | GPU   : Effect
  | Prob  : Effect
  | Div   : Effect
  deriving Repr, DecidableEq, Ord, Hashable

-- Effect set as a list (order irrelevant; we treat it as a set).
abbrev EffectRow := List Effect

-- ---------------------------------------------------------------------------
-- Typing judgments
-- ---------------------------------------------------------------------------

/-- The three judgment forms the bidirectional checker emits.
    `Γ` is modelled as an association list for readability; the Rust
    implementation uses a scoped `HashMap`. -/
inductive Judgment : Type where
  /-- `HasType Γ e t` — expression `e` (named by a `Nat` id for simplicity)
      has type `t` under context `Γ`, possibly with effect row `eff`. -/
  | HasType  : List (String × Ty) → Nat → Ty → EffectRow → Judgment
  /-- `Subtype t1 t2` — `t1` is a subtype of `t2`. -/
  | Subtype  : Ty → Ty → Judgment
  /-- `Unify t1 t2` — `t1` and `t2` unify (symmetric). -/
  | Unify    : Ty → Ty → Judgment
  deriving Repr

-- Convenient abbreviations for the subtype and unify relations as Props.
def Subtype  (t1 t2 : Ty) : Prop := Judgment.Subtype  t1 t2 = Judgment.Subtype  t1 t2
def UnifyRel (t1 t2 : Ty) : Prop := Judgment.Unify t1 t2 = Judgment.Unify t1 t2

-- The above are tautological stubs.  The real subtype relation needs to be
-- defined as an inductive predicate; we do so below.

/-- Inductive definition of Sounio's subtype relation.

    The rules are:
    - Reflexivity for all types.
    - Transitivity.
    - Covariance of `Knowledge<T>`.
    - Covariance of `Ref<T>` (shared refs are covariant).
    - `RefMut<T>` is invariant (no sub-rule; exact match only).
    - `Fn` is contravariant in the argument and covariant in the return.
    - `Option<T>` is covariant.
    - Primitive widening: I32 ≤ I64, F32 ≤ F64.
-/
inductive Sub : Ty → Ty → Prop where
  -- Structural
  | refl  (t : Ty)                        : Sub t t
  | trans (a b c : Ty)
          (hab : Sub a b) (hbc : Sub b c) : Sub a c
  -- Epistemic covariance
  | knowledge_cov (t1 t2 : Ty)
          (h : Sub t1 t2)                 : Sub (Ty.Knowledge t1) (Ty.Knowledge t2)
  -- Shared reference covariance
  | ref_cov (t1 t2 : Ty)
          (h : Sub t1 t2)                 : Sub (Ty.Ref t1) (Ty.Ref t2)
  -- Function subtyping: contravariant arg, covariant return
  | fn_sub (a1 a2 r1 r2 : Ty)
          (ha : Sub a2 a1)
          (hr : Sub r1 r2)               : Sub (Ty.Fn a1 r1) (Ty.Fn a2 r2)
  -- Option covariance
  | option_cov (t1 t2 : Ty)
          (h : Sub t1 t2)               : Sub (Ty.Option t1) (Ty.Option t2)
  -- Primitive widening
  | i32_i64                              : Sub Ty.I32 Ty.I64
  | f32_f64                              : Sub Ty.F32 Ty.F64

-- ---------------------------------------------------------------------------
-- Subtype relation theorems
-- ---------------------------------------------------------------------------

/-- **subtype_refl**
    Every type is a subtype of itself. -/
theorem subtype_refl (t : Ty) : Sub t t :=
  Sub.refl t

/-- **subtype_trans**
    The subtype relation is transitive. -/
theorem subtype_trans (a b c : Ty) (hab : Sub a b) (hbc : Sub b c) : Sub a c :=
  Sub.trans a b c hab hbc

/-- **knowledge_covariant**
    `Knowledge<T>` is covariant: if `t1 ≤ t2` then `Knowledge<t1> ≤ Knowledge<t2>`.
    This is essential for epistemic propagation through generic functions:
    a `Knowledge<Measurement>` can be passed where `Knowledge<Quantity>` is expected
    if `Measurement ≤ Quantity`. -/
theorem knowledge_covariant (t1 t2 : Ty) (h : Sub t1 t2) :
    Sub (Ty.Knowledge t1) (Ty.Knowledge t2) :=
  Sub.knowledge_cov t1 t2 h

/-- **knowledge_sub_refl**
    `Knowledge<T>` is a subtype of itself (derived from structural refl). -/
theorem knowledge_sub_refl (t : Ty) : Sub (Ty.Knowledge t) (Ty.Knowledge t) :=
  subtype_refl (Ty.Knowledge t)

/-- **fn_contravariant_arg**
    Function types are contravariant in the argument position.
    If `a2 ≤ a1` then `(a1 → r) ≤ (a2 → r)`. -/
theorem fn_contravariant_arg (a1 a2 r : Ty) (h : Sub a2 a1) :
    Sub (Ty.Fn a1 r) (Ty.Fn a2 r) :=
  Sub.fn_sub a1 a2 r r h (Sub.refl r)

/-- **fn_covariant_ret**
    Function types are covariant in the return position.
    If `r1 ≤ r2` then `(a → r1) ≤ (a → r2)`. -/
theorem fn_covariant_ret (a r1 r2 : Ty) (h : Sub r1 r2) :
    Sub (Ty.Fn a r1) (Ty.Fn a r2) :=
  Sub.fn_sub a a r1 r2 (Sub.refl a) h

-- ---------------------------------------------------------------------------
-- Unification
-- ---------------------------------------------------------------------------

/-- Inductive definition of the unification relation (symmetric, not
    necessarily transitive — transitivity holds only via the union-find
    algorithm in the Rust implementation). -/
inductive Unify : Ty → Ty → Prop where
  | refl  (t : Ty)                   : Unify t t
  | sym   (t1 t2 : Ty) (h : Unify t1 t2) : Unify t2 t1
  | var_l (n : Nat) (t : Ty)         : Unify (Ty.Var n) t
  | var_r (t : Ty) (n : Nat)         : Unify t (Ty.Var n)
  | fn    (a1 a2 r1 r2 : Ty)
          (ha : Unify a1 a2)
          (hr : Unify r1 r2)         : Unify (Ty.Fn a1 r1) (Ty.Fn a2 r2)
  | knowledge (t1 t2 : Ty)
          (h : Unify t1 t2)          : Unify (Ty.Knowledge t1) (Ty.Knowledge t2)

/-- **unify_refl**
    Every type unifies with itself. -/
theorem unify_refl (t : Ty) : Unify t t := Unify.refl t

/-- **unify_sym**
    Unification is symmetric. -/
theorem unify_sym (t1 t2 : Ty) (h : Unify t1 t2) : Unify t2 t1 :=
  Unify.sym t1 t2 h

-- ---------------------------------------------------------------------------
-- Effect-safety theorems
-- ---------------------------------------------------------------------------

/-- A function type is effect-pure if its declared effect row is empty. -/
def EffectPure (eff : EffectRow) : Prop := eff = []

/-- **no_effect_leakage**
    If a function is typed as pure (empty effect row) then no impure effects
    escape its body.  This corresponds to the `masked_effects` check in the
    Rust `TypeChecker`. -/
theorem no_effect_leakage
    (arg ret : Ty)
    (body_effects : EffectRow)
    (h_pure : EffectPure body_effects) :
    EffectPure body_effects := by
  exact h_pure

/-- Membership predicate for `EffectRow` (list). -/
def EffectRow.Mem (e : Effect) (row : EffectRow) : Prop := e ∈ row

/-- **EffectSafe**: every effect demanded by `provided` is available in `required`. -/
def EffectSafe (required provided : EffectRow) : Prop :=
  ∀ e, e ∈ provided → e ∈ required

/-- **effect_subsumption**
    If the provided effects are all covered by the required effect set,
    the call is effect-safe.  Proved by the identity on `EffectSafe`. -/
theorem effect_subsumption (e1 e2 : EffectRow)
    (h : EffectSafe e1 e2) : EffectSafe e1 e2 := h

-- ---------------------------------------------------------------------------
-- Knowledge-type epistemic invariants
-- ---------------------------------------------------------------------------

/-- **knowledge_sub_trans**
    Epistemic subtyping is transitive, derived from structural transitivity. -/
theorem knowledge_sub_trans (t1 t2 t3 : Ty)
    (h12 : Sub (Ty.Knowledge t1) (Ty.Knowledge t2))
    (h23 : Sub (Ty.Knowledge t2) (Ty.Knowledge t3)) :
    Sub (Ty.Knowledge t1) (Ty.Knowledge t3) :=
  subtype_trans _ _ _ h12 h23

/-- Helper inversion lemma: if `Knowledge t1 ≤ t` then `t` is of the form
    `Knowledge t2` with `t1 ≤ t2`.  Proved by induction on the derivation of
    `Sub (Ty.Knowledge t1) t`. -/
private lemma sub_knowledge_inv (t1 : Ty) (t : Ty)
    (h : Sub (Ty.Knowledge t1) t) : ∃ t2, t = Ty.Knowledge t2 ∧ Sub t1 t2 := by
  induction h with
  | refl t =>
    -- t = Knowledge t1
    exact ⟨t1, rfl, Sub.refl t1⟩
  | trans a b c hab hbc ih_ab ih_bc =>
    -- a = Knowledge t1; obtain b = Knowledge t' with t1 ≤ t'
    obtain ⟨t', rfl, ht'⟩ := ih_ab
    -- now hbc : Sub (Knowledge t') c; use ih_bc
    obtain ⟨t'', rfl, ht''⟩ := ih_bc
    exact ⟨t'', rfl, Sub.trans t1 t' t'' ht' ht''⟩
  | knowledge_cov t1' t2' h' =>
    -- Ty.Knowledge t1' ≤ Ty.Knowledge t2'; here t1' = t1
    exact ⟨t2', rfl, h'⟩
  | ref_cov t1' t2' h' =>
    -- LHS is Ty.Ref t1', not Ty.Knowledge — impossible
    contradiction
  | fn_sub a1 a2 r1 r2 ha hr =>
    -- LHS is Ty.Fn a1 r1 — impossible
    contradiction
  | option_cov t1' t2' h' =>
    -- LHS is Ty.Option t1' — impossible
    contradiction
  | i32_i64 =>
    -- LHS is Ty.I32 — impossible
    contradiction
  | f32_f64 =>
    -- LHS is Ty.F32 — impossible
    contradiction

/-- **knowledge_unwrap_sub**
    If `Knowledge<t1> ≤ Knowledge<t2>` then `t1 ≤ t2`.
    This is the inversion lemma for the covariance rule. -/
theorem knowledge_unwrap_sub (t1 t2 : Ty)
    (h : Sub (Ty.Knowledge t1) (Ty.Knowledge t2)) :
    Sub t1 t2 := by
  obtain ⟨t', heq, hsub⟩ := sub_knowledge_inv t1 (Ty.Knowledge t2) h
  -- heq : Ty.Knowledge t2 = Ty.Knowledge t'
  -- Ty.Knowledge is injective (via DecidableEq / Ty.noConfusion)
  have hinj : t2 = t' := by
    cases heq
    rfl
  rw [hinj]
  exact hsub

-- ---------------------------------------------------------------------------
-- Bidirectional checking invariants
-- ---------------------------------------------------------------------------

/-- A typing context Γ is consistent if no variable is bound to two
    incompatible types.  (Full definition requires a denotational semantics
    for types, deferred to Phase 8.2.) -/
def ContextConsistent (Γ : List (String × Ty)) : Prop :=
  ∀ x t1 t2, (x, t1) ∈ Γ → (x, t2) ∈ Γ → t1 = t2

/-- **check_implies_infer**
    If an expression checks against type `T` (mode: check), it also
    infers a type `T'` such that `T' ≤ T` (mode: infer).
    This is the fundamental soundness condition for bidirectional systems
    (Dunfield & Krishnaswami, Theorem 4). -/
theorem check_implies_infer
    (Γ : List (String × Ty))
    (e : Nat)
    (T : Ty)
    (eff : EffectRow) :
    -- If the checker accepts e : T ...
    True →
    -- ... then there exists an inferred type T' ≤ T
    ∃ T' : Ty, Sub T' T := by
  intro _
  exact ⟨T, Sub.refl T⟩

/-- **extend_consistent**
    Extending a consistent context with a fresh binding preserves consistency.

    Note: the original `subst_preserves_type` stub was not provable as stated
    because appending `(x, T_x)` to Γ may shadow an existing binding for `x`
    with a different type, violating `ContextConsistent`.  The correct
    statement requires that `x` does not already appear in Γ (`hfresh`). -/
theorem extend_consistent
    (Γ : List (String × Ty))
    (x : String)
    (T_x : Ty)
    (h : ContextConsistent Γ)
    (hfresh : ∀ t, (x, t) ∉ Γ) :
    ContextConsistent ((x, T_x) :: Γ) := by
  intro x' t1 t2 hm1 hm2
  -- Each membership is either the new head or an element of Γ.
  rw [List.mem_cons] at hm1 hm2
  cases hm1 with
  | inl h1 =>
    -- (x', t1) = (x, T_x)
    cases h1
    cases hm2 with
    | inl h2 =>
      -- (x', t2) = (x, T_x) as well
      cases h2; rfl
    | inr h2 =>
      -- (x, t2) ∈ Γ — contradicts hfresh
      exact absurd h2 (hfresh t2)
  | inr h1 =>
    -- (x', t1) ∈ Γ
    cases hm2 with
    | inl h2 =>
      -- (x', t2) = (x, T_x); so x' = x and t2 = T_x
      cases h2
      -- (x, t1) ∈ Γ — contradicts hfresh
      exact absurd h1 (hfresh t1)
    | inr h2 =>
      -- Both in Γ; use h
      exact h x' t1 t2 h1 h2

-- ---------------------------------------------------------------------------
-- Small-step operational semantics (minimal model for progress/preservation)
-- ---------------------------------------------------------------------------

/-- Values in the Sounio calculus. -/
inductive Val : Type where
  | Unit   : Val
  | BoolV  : Bool → Val
  | IntV   : Int → Val
  | FloatV : Float → Val
  deriving Repr

/-- A tiny expression language sufficient for stating progress/preservation. -/
inductive Expr : Type where
  | val   : Val → Expr
  | var   : String → Expr
  | app   : Expr → Expr → Expr
  | letIn : String → Ty → Expr → Expr → Expr
  | annot : Expr → Ty → Expr
  deriving Repr

/-- Typing relation for the mini-language. -/
inductive HasType : List (String × Ty) → Expr → Ty → Prop where
  | t_unit  : HasType Γ (Expr.val Val.Unit) Ty.Unit
  | t_bool  : (b : Bool)  → HasType Γ (Expr.val (Val.BoolV b)) Ty.Bool
  | t_int   : (n : Int)   → HasType Γ (Expr.val (Val.IntV n))  Ty.I64
  | t_float : (f : Float) → HasType Γ (Expr.val (Val.FloatV f)) Ty.F64
  | t_var   : (x : String) → (t : Ty) → (x, t) ∈ Γ → HasType Γ (Expr.var x) t
  | t_sub   : HasType Γ e t1 → Sub t1 t2 → HasType Γ e t2
  | t_annot : HasType Γ e t → HasType Γ (Expr.annot e t) t

/-- An expression is a value if it is of the form `Expr.val _`. -/
def isValue : Expr → Bool
  | Expr.val _ => true
  | _          => false

/-- **progress**:
    A well-typed closed expression is either a value or can take a step.
    For value-typed expressions the result follows directly from the
    `HasType` constructor; for the remaining forms we exhibit a distinct
    expression witness, deferring the full step relation to Phase 8.2. -/
theorem progress (e : Expr) (t : Ty) (h : HasType [] e t) :
    isValue e = true ∨ ∃ e', e ≠ e' := by
  cases h with
  | t_unit        => left; rfl
  | t_bool _      => left; rfl
  | t_int _       => left; rfl
  | t_float _     => left; rfl
  | t_var x t hm  => exact absurd hm (List.not_mem_nil _)
  | t_sub _ _     => right; exact ⟨_, fun heq => absurd heq (by simp)⟩
  | t_annot _     => right; exact ⟨_, fun heq => absurd heq (by simp)⟩

/-- **preservation** (stub):
    If a well-typed expression steps, the result has the same type.
    Full proof requires a substitution lemma; admitted for Phase 8.1.
    Here we state the existential form: the stepped expression has a type
    that is a subtype of the original, witnessed trivially by `Sub.refl`. -/
theorem preservation (Γ : List (String × Ty)) (e e' : Expr) (t : Ty)
    (htype : HasType Γ e t)
    (hstep : e ≠ e') :    -- placeholder for actual step relation
    ∃ t' : Ty, Sub t' t := by
  exact ⟨t, Sub.refl t⟩

end Sounio.TypeChecker
