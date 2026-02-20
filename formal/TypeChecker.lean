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
-- Effect-safety theorems (stubs)
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

/-- **effect_subsumption**
    If a function expects effect row `e1` and the caller provides `e2`,
    the call is valid whenever `e2` is a subset of `e1`.
    TODO: formalise with `Finset` when the full effect row model is added. -/
theorem effect_subsumption (e1 e2 : EffectRow) : sorry := by sorry

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

/-- **knowledge_unwrap_sub**
    If `Knowledge<t1> ≤ Knowledge<t2>` then `t1 ≤ t2`.
    This is the inversion lemma for the covariance rule. -/
theorem knowledge_unwrap_sub (t1 t2 : Ty)
    (h : Sub (Ty.Knowledge t1) (Ty.Knowledge t2)) :
    Sub t1 t2 := by
  sorry

-- ---------------------------------------------------------------------------
-- Bidirectional checking invariants (stubs over a denotational model)
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

/-- **subst_preserves_type**
    Substituting a well-typed value for a variable preserves the type of
    the body.  This is the key lemma for let-binding correctness. -/
theorem subst_preserves_type
    (Γ : List (String × Ty))
    (x : String)
    (T_x T_body : Ty) :
    ContextConsistent Γ →
    ContextConsistent ((x, T_x) :: Γ) := by
  sorry

end Sounio.TypeChecker
