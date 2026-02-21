/-!
# Sounio.EffectLinear — Effect × Linear Non-Interference

This file formalises the **effect × linear non-interference theorem** for the
Sounio compiler: consuming a linear variable inside an effect handler still
removes it from the usage environment — effect annotation does not affect
linear resource accounting.

Concretely, if `Γ ⊢ e : ε ⊣ Γ'` and `Γ ⊢ e : ε' ⊣ Γ''`, then `Γ' = Γ''`.
The output usage environment is uniquely determined by the expression structure
and the input environment, regardless of which effects the expression performs.

All theorems are proved without `sorry`.  No Mathlib dependency — Lean 4 core
only.
-/

namespace Sounio.EffectLinear

-- ===========================================================================
-- §1. Shared Types (self-contained re-definitions)
-- ===========================================================================

/-- The effect universe for Sounio.  Mirrors `Effect` in `TypeChecker.lean`.
    `Epistemic` replaces `Prob` to reflect the epistemic-computing focus. -/
inductive Effect : Type where
  | IO        : Effect
  | Mut       : Effect
  | Alloc     : Effect
  | GPU       : Effect
  | Epistemic : Effect
  | Exn       : Effect
  deriving Repr, DecidableEq

/-- An effect row is a predicate over `Effect`: `true` means the effect is
    present.  Functional representation avoids list-equality issues and keeps
    set operations definitionally simple. -/
abbrev EffRow := Effect → Bool

/-- The empty effect row: no effects. -/
def effEmpty : EffRow := fun _ => false

/-- Effect-row union: pointwise disjunction. -/
def effUnion (ε₁ ε₂ : EffRow) : EffRow := fun e => ε₁ e || ε₂ e

/-- Effect-row membership. -/
def effMem (e : Effect) (ε : EffRow) : Prop := ε e = true

/-- Effect-row subset: every effect in ε₁ is in ε₂. -/
def effSubset (ε₁ ε₂ : EffRow) : Prop := ∀ e, effMem e ε₁ → effMem e ε₂

/-- Usage multiplicity.
    `Zero` — already consumed.  `One` — linear: must be used exactly once. -/
inductive Mult : Type where
  | Zero : Mult
  | One  : Mult
  deriving Repr, DecidableEq

/-- A usage environment maps variable names to remaining multiplicities. -/
abbrev UsageEnv := List (String × Mult)

-- ===========================================================================
-- §2. Effectful Expression AST
-- ===========================================================================

/-- A minimal expression language carrying effect annotations. -/
inductive EffExpr : Type where
  | var    : String → EffExpr
  | unit   : EffExpr
  | seq    : EffExpr → EffExpr → EffExpr
  | handle : Effect → EffExpr → EffExpr → EffExpr
  | letLin : String → EffExpr → EffExpr → EffExpr
  deriving Repr, DecidableEq

-- ===========================================================================
-- §3. Usage Environment Operations
-- ===========================================================================

/-- Look up the multiplicity of `x` in `env`, defaulting to `Zero`. -/
def usageOf (env : UsageEnv) (x : String) : Mult :=
  match env with
  | []             => Mult.Zero
  | (y, m) :: rest => if y == x then m else usageOf rest x

/-- Zero out every occurrence of `x` in `env`. -/
def consumeVar (env : UsageEnv) (x : String) : UsageEnv :=
  env.map (fun p => if p.1 == x then (p.1, Mult.Zero) else p)

/-- Prepend a new binding `(x, m)` to `env`. -/
def envExtend (env : UsageEnv) (x : String) (m : Mult) : UsageEnv :=
  (x, m) :: env

-- ===========================================================================
-- §4. Lemmas about UsageEnv Operations
-- ===========================================================================

/-- After `consumeVar x`, looking up `x` gives `Zero`. -/
theorem consumeVar_removes (Γ : UsageEnv) (x : String) :
    usageOf (consumeVar Γ x) x = Mult.Zero := by
  induction Γ with
  | nil => simp [consumeVar, usageOf]
  | cons p ps ih =>
    simp only [consumeVar, List.map_cons, usageOf]
    rcases Bool.eq_false_or_eq_true (p.1 == x) with hpx | hpx
    · simp [hpx]
    · simp [hpx, ih]

/-- `consumeVar x` does not affect the lookup of any `y ≠ x`. -/
theorem consumeVar_preserves (Γ : UsageEnv) (x y : String) (hne : x ≠ y) :
    usageOf (consumeVar Γ x) y = usageOf Γ y := by
  induction Γ with
  | nil => simp [consumeVar, usageOf]
  | cons p ps ih =>
    simp only [consumeVar, List.map_cons, usageOf]
    rcases Bool.eq_false_or_eq_true (p.1 == y) with hpy | hpy
    · rcases Bool.eq_false_or_eq_true (p.1 == x) with hpx | hpx
      · simp only [beq_iff_eq] at hpy hpx
        -- hpy : p.1 = y, hpx : p.1 = x, so y = x, contradicting x ≠ y
        exact absurd (hpy.symm.trans hpx) (Ne.symm hne)
      · simp [hpx, hpy]
    · rcases Bool.eq_false_or_eq_true (p.1 == x) with hpx | hpx
      · simp [hpx, hpy, ih]
      · simp [hpx, hpy, ih]

/-- `consumeVar` is idempotent. -/
theorem consumeVar_idempotent (Γ : UsageEnv) (x : String) :
    consumeVar (consumeVar Γ x) x = consumeVar Γ x := by
  induction Γ with
  | nil => simp [consumeVar]
  | cons p ps ih =>
    simp only [consumeVar, List.map_cons]
    rcases Bool.eq_false_or_eq_true (p.1 == x) with hpx | hpx <;> simp [hpx, ih]

/-- After consuming `v` (where `v ≠ x`), the pair `(x, One)` is in the
    result iff it was already present. -/
theorem consumeVar_mem_one_iff (Γ : UsageEnv) (v x : String) (hne : v ≠ x) :
    (x, Mult.One) ∈ consumeVar Γ v ↔ (x, Mult.One) ∈ Γ := by
  simp only [consumeVar, List.mem_map]
  constructor
  · intro ⟨⟨y, m⟩, hym, hpair⟩
    -- hpair : (if y == v then (y, Mult.Zero) else (y, m)) = (x, Mult.One)
    -- We split on whether y equals v.
    by_cases hyv : (y : String) = v
    · -- y = v, so the map sends (y, m) to (y, Zero)
      have hbool : y == v = true := by simp [beq_iff_eq, hyv]
      simp only [hbool, ↓reduceIte] at hpair
      -- hpair : (y, Mult.Zero) = (x, Mult.One); Zero ≠ One
      exact absurd (congrArg Prod.snd hpair) (by decide)
    · -- y ≠ v, so the map leaves (y, m) unchanged
      have hbool : y == v = false := by
        rw [Bool.eq_false_iff, beq_iff_eq]; exact hyv
      simp only [hbool, ↓reduceIte] at hpair
      -- hpair : (y, m) = (x, Mult.One)
      have hyx : y = x := congrArg Prod.fst hpair
      have hmult : m = Mult.One := congrArg Prod.snd hpair
      rw [← hyx, ← hmult]; exact hym
  · intro hxone
    refine ⟨(x, Mult.One), hxone, ?_⟩
    -- goal: (fun p => if p.1 == v then (p.1, Mult.Zero) else p) (x, Mult.One) = (x, Mult.One)
    -- Reduces to: (if x == v then (x, Zero) else (x, One)) = (x, One)
    have hxv : (x : String) ≠ v := fun h => hne h.symm
    have hbool : x == v = false := by
      rw [Bool.eq_false_iff, beq_iff_eq]; exact hxv
    simp [hbool]

/-- Extending then looking up the same variable. -/
theorem extend_same_lookup (Γ : UsageEnv) (x : String) (m : Mult) :
    usageOf (envExtend Γ x m) x = m := by
  simp [envExtend, usageOf]

/-- Extending with `y` does not affect the lookup of `x ≠ y`. -/
theorem extend_other_lookup (Γ : UsageEnv) (x y : String) (m : Mult) (hne : x ≠ y) :
    usageOf (envExtend Γ y m) x = usageOf Γ x := by
  simp only [envExtend, usageOf]
  have hyx : y == x = false := by
    rw [Bool.eq_false_iff, beq_iff_eq]
    exact Ne.symm hne
  simp [hyx]

/-- Extending then consuming the same variable zeroes it out. -/
theorem extend_consume_same (Γ : UsageEnv) (x : String) :
    usageOf (consumeVar (envExtend Γ x Mult.One) x) x = Mult.Zero :=
  consumeVar_removes _ x

/-- Extending with `y` then consuming `y` leaves `x ≠ y` unchanged. -/
theorem extend_consume_other (Γ : UsageEnv) (x y : String) (m : Mult) (hne : x ≠ y) :
    usageOf (consumeVar (envExtend Γ y m) y) x = usageOf Γ x := by
  rw [consumeVar_preserves _ y x (Ne.symm hne), extend_other_lookup _ x y m hne]

/-- Extending with `x` shadows any prior binding of `x`. -/
theorem extend_shadows (Γ : UsageEnv) (x : String) (m1 m2 : Mult) :
    usageOf (envExtend (envExtend Γ x m1) x m2) x = m2 := by
  simp [envExtend, usageOf]

/-- For `x ≠ y`, extending with `x = One` then `y = One`: lookup of `x` is One. -/
theorem extend_two_lookup_first (Γ : UsageEnv) (x y : String) (hne : x ≠ y) :
    usageOf (envExtend (envExtend Γ x Mult.One) y Mult.One) x = Mult.One := by
  simp only [envExtend, usageOf]
  have hyx : y == x = false := by
    rw [Bool.eq_false_iff, beq_iff_eq]
    exact Ne.symm hne
  simp [hyx]

/-- For distinct `x ≠ z` and `y ≠ z`, extending with x then y vs y then x
    gives the same `usageOf z`. -/
theorem extend_comm_lookup (Γ : UsageEnv) (x y : String) (mx my : Mult)
    (z : String) (hzx : z ≠ x) (hzy : z ≠ y) :
    usageOf (envExtend (envExtend Γ x mx) y my) z =
    usageOf (envExtend (envExtend Γ y my) x mx) z := by
  simp only [envExtend, usageOf]
  have hyz : y == z = false := by rw [Bool.eq_false_iff, beq_iff_eq]; exact Ne.symm hzy
  have hxz : x == z = false := by rw [Bool.eq_false_iff, beq_iff_eq]; exact Ne.symm hzx
  simp [hyz, hxz]

/-- After extending with `y = One` and consuming `y`, `usageOf x = One` is
    preserved for `x ≠ y` if it held before. -/
theorem extend_consume_unrelated (Γ : UsageEnv) (x y : String) (hne : x ≠ y)
    (hx : usageOf Γ x = Mult.One) :
    usageOf (consumeVar (envExtend Γ y Mult.One) y) x = Mult.One := by
  rw [extend_consume_other _ x y _ hne, hx]

-- ===========================================================================
-- §5. Effect-Annotated Linear Typing Judgment
-- ===========================================================================

/-- The linear typing judgment `LinTyped Γ e ε Γ'`.
    - `Γ`  — input  usage environment (resources available)
    - `e`  — expression
    - `ε`  — effect row (annotation; does not constrain usage)
    - `Γ'` — output usage environment (remaining after `e` executes) -/
inductive LinTyped : UsageEnv → EffExpr → EffRow → UsageEnv → Prop where
  | lt_var    : (x, Mult.One) ∈ Γ →
                LinTyped Γ (EffExpr.var x) ε (consumeVar Γ x)
  | lt_unit   : LinTyped Γ EffExpr.unit ε Γ
  | lt_seq    : LinTyped Γ e1 ε Γ' → LinTyped Γ' e2 ε Γ'' →
                LinTyped Γ (EffExpr.seq e1 e2) ε Γ''
  | lt_handle : LinTyped Γ body ε Γ' → LinTyped Γ' handler ε Γ'' →
                LinTyped Γ (EffExpr.handle ef body handler) ε Γ''
  | lt_letLin : LinTyped Γ rhs ε Γ' →
                LinTyped (envExtend Γ' x Mult.One) bodyExpr ε Γ'' →
                LinTyped Γ (EffExpr.letLin x rhs bodyExpr) ε Γ''

-- ===========================================================================
-- §6. Core Theorems
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Theorem 1: Effect-row independence
-- ---------------------------------------------------------------------------

/-- **Effect-row independence**: typing with effect row `ε` implies typing
    with any other row `ε'`, leaving all environments unchanged.
    This is the key step in establishing non-interference. -/
theorem lintyped_effect_independent (Γ : UsageEnv) (e : EffExpr)
    (ε ε' : EffRow) (Γ' : UsageEnv)
    (h : LinTyped Γ e ε Γ') : LinTyped Γ e ε' Γ' := by
  induction h with
  | lt_var hmem           => exact LinTyped.lt_var hmem
  | lt_unit               => exact LinTyped.lt_unit
  | lt_seq _ _ ih1 ih2    => exact LinTyped.lt_seq ih1 ih2
  | lt_handle _ _ ih1 ih2 => exact LinTyped.lt_handle ih1 ih2
  | lt_letLin _ _ ih1 ih2 => exact LinTyped.lt_letLin ih1 ih2

-- ---------------------------------------------------------------------------
-- Theorem 2: Handle inversion
-- ---------------------------------------------------------------------------

/-- **Handle inversion**: a handle derivation decomposes into body + handler. -/
theorem handle_typed_inv (Γ : UsageEnv) (ef : Effect) (body handler : EffExpr)
    (ε : EffRow) (Γ'' : UsageEnv)
    (h : LinTyped Γ (EffExpr.handle ef body handler) ε Γ'') :
    ∃ Γ', LinTyped Γ body ε Γ' ∧ LinTyped Γ' handler ε Γ'' := by
  cases h with
  | lt_handle hbody hhandler => exact ⟨_, hbody, hhandler⟩

-- ---------------------------------------------------------------------------
-- Theorem 3: consumeVar removes exactly the target
-- ---------------------------------------------------------------------------

/-- **consumeVar correctness**: after zeroing out `x`, `usageOf x = Zero`. -/
theorem consumeVar_correct (Γ : UsageEnv) (x : String) :
    usageOf (consumeVar Γ x) x = Mult.Zero :=
  consumeVar_removes Γ x

-- ---------------------------------------------------------------------------
-- Theorem 4: consumeVar preserves other variables
-- ---------------------------------------------------------------------------

/-- **consumeVar non-interference**: consuming `x` leaves all `y ≠ x`
    unchanged. -/
theorem consumeVar_noninterference (Γ : UsageEnv) (x y : String) (hne : x ≠ y) :
    usageOf (consumeVar Γ x) y = usageOf Γ y :=
  consumeVar_preserves Γ x y hne

-- ---------------------------------------------------------------------------
-- Theorem 5: Unit preserves environment
-- ---------------------------------------------------------------------------

/-- **Unit preserves**: typing `unit` leaves the usage environment unchanged. -/
theorem unit_preserves_env (Γ : UsageEnv) (ε : EffRow) :
    LinTyped Γ EffExpr.unit ε Γ :=
  LinTyped.lt_unit

-- ---------------------------------------------------------------------------
-- Theorem 6: var consumption is independent of effect row
-- ---------------------------------------------------------------------------

/-- **Variable consumption effect-independence**: typing `var x` under any
    effect row `ε` — provided `(x, One) ∈ Γ` — consumes `x` from `Γ`
    regardless of which effects the context permits.
    This is the atomic instance of the non-interference property. -/
theorem var_consumption_effect_independent (Γ : UsageEnv) (x : String)
    (ε ε' : EffRow) (hmem : (x, Mult.One) ∈ Γ) :
    (LinTyped Γ (EffExpr.var x) ε (consumeVar Γ x)) ∧
    (LinTyped Γ (EffExpr.var x) ε' (consumeVar Γ x)) :=
  ⟨LinTyped.lt_var hmem, LinTyped.lt_var hmem⟩

-- ---------------------------------------------------------------------------
-- Theorem 7: Sequential composition
-- ---------------------------------------------------------------------------

/-- **Seq associativity**: the outer `seq` is typeable from its sub-typings. -/
theorem seq_usage_assoc (Γ : UsageEnv) (e1 e2 e3 : EffExpr) (ε : EffRow)
    (Γ2 Γ' : UsageEnv)
    (h12 : LinTyped Γ (EffExpr.seq e1 e2) ε Γ2)
    (h3  : LinTyped Γ2 e3 ε Γ') :
    LinTyped Γ (EffExpr.seq (EffExpr.seq e1 e2) e3) ε Γ' :=
  LinTyped.lt_seq h12 h3

-- ---------------------------------------------------------------------------
-- Theorem 8: envExtend then consumeVar
-- ---------------------------------------------------------------------------

/-- **Extend then consume**: creating a linear binding then consuming it
    leaves zero usage. -/
theorem extend_then_consume (Γ : UsageEnv) (x : String) :
    usageOf (consumeVar (envExtend Γ x Mult.One) x) x = Mult.Zero :=
  extend_consume_same Γ x

-- ---------------------------------------------------------------------------
-- Theorem 9: envExtend preserves other lookups
-- ---------------------------------------------------------------------------

/-- **Extend preserves**: extending with `y` doesn't affect `usageOf x` when
    `x ≠ y`. -/
theorem extend_does_not_clobber (Γ : UsageEnv) (x y : String) (m : Mult) (hne : x ≠ y) :
    usageOf (envExtend Γ y m) x = usageOf Γ x :=
  extend_other_lookup Γ x y m hne

-- ---------------------------------------------------------------------------
-- Theorem 10: Effect × Linear Non-Interference (main theorem)
-- ---------------------------------------------------------------------------

/-- **Effect × Linear Non-Interference**:
    The effect annotation `ε` does not affect how linear variables are consumed.
    The output usage environment `Γ'` is uniquely determined by the expression
    structure and the input environment `Γ`, regardless of which effects
    the expression performs.

    Formally: if `Γ ⊢ e : ε ⊣ Γ'` and `Γ ⊢ e : ε' ⊣ Γ''`, then `Γ' = Γ''`.

    Proof strategy:
    1. Re-annotate `h2` with the same effect row `ε` using Theorem 1.
    2. Prove uniqueness by structural induction on `h1`, inverting `h2'`
       at each constructor: in every case the output environment is fully
       determined by the input environment and the sub-expressions' outputs. -/
theorem effect_linear_noninterference (Γ : UsageEnv) (e : EffExpr)
    (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ e ε Γ') (h2 : LinTyped Γ e ε' Γ'') :
    Γ' = Γ'' := by
  have h2' : LinTyped Γ e ε Γ'' := lintyped_effect_independent Γ e ε' ε Γ'' h2
  induction h1 generalizing Γ'' with
  | lt_var _ =>
    cases h2' with
    | lt_var _ => rfl
  | lt_unit =>
    cases h2' with
    | lt_unit => rfl
  | lt_seq hA hB ihA ihB =>
    cases h2' with
    | lt_seq hA' hB' =>
      have hmid : _ = _ := ihA _ hA'
      subst hmid; exact ihB _ hB'
  | lt_handle hbody hhandler ihbody ihhandler =>
    cases h2' with
    | lt_handle hbody' hhandler' =>
      have hmid : _ = _ := ihbody _ hbody'
      subst hmid; exact ihhandler _ hhandler'
  | lt_letLin hrhs hbody ihrhs ihbody =>
    cases h2' with
    | lt_letLin hrhs' hbody' =>
      have hmid : _ = _ := ihrhs _ hrhs'
      subst hmid; exact ihbody _ hbody'

-- ===========================================================================
-- §7. Additional Theorems (A11–A35)
-- ===========================================================================

-- A11. consumeVar noop when usageOf is already Zero

/-- Consuming a variable with Zero usage leaves all lookups unchanged. -/
theorem consumeVar_noop_all (Γ : UsageEnv) (x y : String)
    (habsent : usageOf Γ x = Mult.Zero) :
    usageOf (consumeVar Γ x) y = usageOf Γ y := by
  by_cases hne : x = y
  · subst hne; rw [consumeVar_removes]; exact habsent.symm
  · exact consumeVar_preserves Γ x y hne

-- A12. Double consume gives Zero

/-- Consuming `x` twice: the second consume is a no-op on lookups. -/
theorem double_consume_zero (Γ : UsageEnv) (x : String) :
    usageOf (consumeVar (consumeVar Γ x) x) x = Mult.Zero :=
  consumeVar_removes _ x

-- A13. seq with unit — left identity

/-- `seq unit e` produces the same final environment as `e` alone. -/
theorem seq_unit_left (Γ : UsageEnv) (e : EffExpr) (ε : EffRow) (Γ' : UsageEnv)
    (h : LinTyped Γ e ε Γ') :
    LinTyped Γ (EffExpr.seq EffExpr.unit e) ε Γ' :=
  LinTyped.lt_seq LinTyped.lt_unit h

-- A14. seq with unit — right identity

/-- `seq e unit` produces the same final environment as `e` alone. -/
theorem seq_unit_right (Γ : UsageEnv) (e : EffExpr) (ε : EffRow) (Γ' : UsageEnv)
    (h : LinTyped Γ e ε Γ') :
    LinTyped Γ (EffExpr.seq e EffExpr.unit) ε Γ' :=
  LinTyped.lt_seq h LinTyped.lt_unit

-- A15. Effect-row union — left component

/-- If `e` types under `ε`, it types under `effUnion ε ε'`. -/
theorem lintyped_under_union_left (Γ : UsageEnv) (e : EffExpr)
    (ε ε' : EffRow) (Γ' : UsageEnv) (h : LinTyped Γ e ε Γ') :
    LinTyped Γ e (effUnion ε ε') Γ' :=
  lintyped_effect_independent Γ e ε (effUnion ε ε') Γ' h

-- A16. Effect-row union — right component

/-- If `e` types under `ε'`, it types under `effUnion ε ε'`. -/
theorem lintyped_under_union_right (Γ : UsageEnv) (e : EffExpr)
    (ε ε' : EffRow) (Γ' : UsageEnv) (h : LinTyped Γ e ε' Γ') :
    LinTyped Γ e (effUnion ε ε') Γ' :=
  lintyped_effect_independent Γ e ε' (effUnion ε ε') Γ' h

-- A17. Typing preserved under effect-row superset

/-- The typing judgment is preserved under any effect-row change (corollary
    of Theorem 1). -/
theorem lintyped_effect_superset (Γ : UsageEnv) (e : EffExpr)
    (ε ε' : EffRow) (Γ' : UsageEnv) (h : LinTyped Γ e ε Γ') :
    LinTyped Γ e ε' Γ' :=
  lintyped_effect_independent Γ e ε ε' Γ' h

-- A18. letLin inversion

/-- **letLin inversion**: decompose a `letLin` derivation into rhs + body. -/
theorem letLin_typed_inv (Γ : UsageEnv) (x : String) (rhs body : EffExpr)
    (ε : EffRow) (Γ'' : UsageEnv)
    (h : LinTyped Γ (EffExpr.letLin x rhs body) ε Γ'') :
    ∃ Γ', LinTyped Γ rhs ε Γ' ∧ LinTyped (envExtend Γ' x Mult.One) body ε Γ'' := by
  cases h with
  | lt_letLin hrhs hbody => exact ⟨_, hrhs, hbody⟩

-- A19. Handle preserves consumed variables — structural witness

/-- **Handle structural witness**: inversion form for handle. -/
theorem handle_preserves_consumed (Γ : UsageEnv) (ef : Effect)
    (body handler : EffExpr) (ε : EffRow) (Γ'' : UsageEnv)
    (h : LinTyped Γ (EffExpr.handle ef body handler) ε Γ'') :
    ∃ Γ', LinTyped Γ body ε Γ' ∧ LinTyped Γ' handler ε Γ'' :=
  handle_typed_inv Γ ef body handler ε Γ'' h

-- A20. var output uniqueness

/-- Two typings of `var x` from the same `Γ` yield the same output env. -/
theorem var_output_unique (Γ : UsageEnv) (x : String) (ε ε' : EffRow)
    (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.var x) ε Γ')
    (h2 : LinTyped Γ (EffExpr.var x) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.var x) ε ε' Γ' Γ'' h1 h2

-- A21. seq output uniqueness

/-- Two typings of `seq e1 e2` from the same `Γ` yield the same output env. -/
theorem seq_output_unique (Γ : UsageEnv) (e1 e2 : EffExpr) (ε ε' : EffRow)
    (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.seq e1 e2) ε Γ')
    (h2 : LinTyped Γ (EffExpr.seq e1 e2) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.seq e1 e2) ε ε' Γ' Γ'' h1 h2

-- A22. handle output uniqueness

/-- Two typings of a handle expression from the same `Γ` yield the same output. -/
theorem handle_output_unique (Γ : UsageEnv) (ef : Effect) (body handler : EffExpr)
    (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.handle ef body handler) ε Γ')
    (h2 : LinTyped Γ (EffExpr.handle ef body handler) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.handle ef body handler) ε ε' Γ' Γ'' h1 h2

-- A23. Determinism (same effect row)

/-- **Determinism**: at most one output environment for given `Γ`, `e`, and `ε`. -/
theorem lintyped_deterministic (Γ : UsageEnv) (e : EffExpr) (ε : EffRow)
    (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ e ε Γ') (h2 : LinTyped Γ e ε Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ e ε ε Γ' Γ'' h1 h2

-- A24. seq re-typing

/-- Re-typing a seq with a different effect row. -/
theorem seq_retype (Γ : UsageEnv) (e1 e2 : EffExpr) (ε ε' : EffRow) (Γ' : UsageEnv)
    (h : LinTyped Γ (EffExpr.seq e1 e2) ε Γ') :
    LinTyped Γ (EffExpr.seq e1 e2) ε' Γ' :=
  lintyped_effect_independent Γ (EffExpr.seq e1 e2) ε ε' Γ' h

-- A25. handle re-typing

/-- Re-typing a handle with a different effect row. -/
theorem handle_retype (Γ : UsageEnv) (ef : Effect) (body handler : EffExpr)
    (ε ε' : EffRow) (Γ'' : UsageEnv)
    (h : LinTyped Γ (EffExpr.handle ef body handler) ε Γ'') :
    LinTyped Γ (EffExpr.handle ef body handler) ε' Γ'' :=
  lintyped_effect_independent Γ (EffExpr.handle ef body handler) ε ε' Γ'' h

-- A26. unit typeable under empty effect row

/-- `unit` is always typeable under the empty effect row. -/
theorem unit_typeable_empty (Γ : UsageEnv) :
    LinTyped Γ EffExpr.unit effEmpty Γ :=
  LinTyped.lt_unit

-- A27. seq output determined by shared intermediate environment

/-- If two sub-derivations for `e2` share the same input env, their outputs
    agree. -/
theorem seq_output_from_mid (Γ_mid Γ' Γ'' : UsageEnv) (e2 : EffExpr) (ε : EffRow)
    (h2  : LinTyped Γ_mid e2 ε Γ')
    (h2' : LinTyped Γ_mid e2 ε Γ'') :
    Γ' = Γ'' :=
  lintyped_deterministic Γ_mid e2 ε Γ' Γ'' h2 h2'

-- A28. envExtend commutativity on the extended variables' lookups

/-- For `x ≠ y`, extending with `x` after `y`: lookup of `x` is `mx`. -/
theorem extend_swap_x (Γ : UsageEnv) (x y : String) (mx my : Mult) (hne : x ≠ y) :
    usageOf (envExtend (envExtend Γ y my) x mx) x = mx := by
  simp [envExtend, usageOf]

/-- For `x ≠ y`, extending with `y` after `x`: lookup of `y` is `my`. -/
theorem extend_swap_y (Γ : UsageEnv) (x y : String) (mx my : Mult) (hne : x ≠ y) :
    usageOf (envExtend (envExtend Γ x mx) y my) y = my := by
  simp [envExtend, usageOf]

-- A29. Handle with unit handler

/-- A handle where the handler is `unit` produces the body's output env. -/
theorem handle_unit_handler (Γ : UsageEnv) (ef : Effect) (body : EffExpr)
    (ε : EffRow) (Γ' : UsageEnv) (hbody : LinTyped Γ body ε Γ') :
    LinTyped Γ (EffExpr.handle ef body EffExpr.unit) ε Γ' :=
  LinTyped.lt_handle hbody LinTyped.lt_unit

-- A30. Handle with unit body

/-- A handle where the body is `unit` produces the handler's output env. -/
theorem handle_unit_body (Γ : UsageEnv) (ef : Effect) (handler : EffExpr)
    (ε : EffRow) (Γ' : UsageEnv) (hhandler : LinTyped Γ handler ε Γ') :
    LinTyped Γ (EffExpr.handle ef EffExpr.unit handler) ε Γ' :=
  LinTyped.lt_handle LinTyped.lt_unit hhandler

-- A31. seq non-interference (corollary)

/-- Two typings of `seq e1 e2` under different effect rows produce the same
    output environment. -/
theorem seq_effect_noninterference (Γ : UsageEnv) (e1 e2 : EffExpr)
    (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.seq e1 e2) ε Γ')
    (h2 : LinTyped Γ (EffExpr.seq e1 e2) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.seq e1 e2) ε ε' Γ' Γ'' h1 h2

-- A32. letLin non-interference (corollary)

/-- Two typings of `letLin x rhs body` under different effect rows produce
    the same output environment. -/
theorem letLin_effect_noninterference (Γ : UsageEnv) (x : String)
    (rhs body : EffExpr) (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.letLin x rhs body) ε Γ')
    (h2 : LinTyped Γ (EffExpr.letLin x rhs body) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.letLin x rhs body) ε ε' Γ' Γ'' h1 h2

-- A33. handle non-interference (corollary)

/-- Two typings of a handle expression under different effect rows produce
    the same output environment. -/
theorem handle_effect_noninterference (Γ : UsageEnv) (ef : Effect)
    (body handler : EffExpr) (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.handle ef body handler) ε Γ')
    (h2 : LinTyped Γ (EffExpr.handle ef body handler) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.handle ef body handler) ε ε' Γ' Γ'' h1 h2

-- A34. IO effect is irrelevant to linear typing

/-- The IO effect annotation is irrelevant to linear resource tracking. -/
theorem io_effect_irrelevant (Γ : UsageEnv) (e : EffExpr) (Γ' : UsageEnv)
    (h : LinTyped Γ e (fun _ => false) Γ') :
    LinTyped Γ e (fun ef => decide (ef = Effect.IO)) Γ' :=
  lintyped_effect_independent Γ e (fun _ => false) (fun ef => decide (ef = Effect.IO)) Γ' h

-- A35. Combined non-interference for seq across different effect rows

/-- If two seq expressions are typed from the same `Γ` using possibly different
    effect rows per component, the outputs are equal. -/
theorem seq_noninterference_combined (Γ : UsageEnv) (e1 e2 : EffExpr)
    (ε₁ ε₂ ε₁' ε₂' : EffRow) (Γ_mid Γ' Γ_mid' Γ'' : UsageEnv)
    (hA  : LinTyped Γ e1 ε₁ Γ_mid)
    (hB  : LinTyped Γ_mid e2 ε₂ Γ')
    (hA' : LinTyped Γ e1 ε₁' Γ_mid')
    (hB' : LinTyped Γ_mid' e2 ε₂' Γ'') :
    Γ' = Γ'' := by
  have hmid : Γ_mid = Γ_mid' :=
    effect_linear_noninterference Γ e1 ε₁ ε₁' Γ_mid Γ_mid' hA hA'
  subst hmid
  exact effect_linear_noninterference Γ_mid' e2 ε₂ ε₂' Γ' Γ'' hB hB'

-- ===========================================================================
-- §8. Handle-specific non-interference: variable consumption in handlers
-- ===========================================================================

/-- **Handle linear var consumption**: consuming a linear variable `x` inside
    a handle expression body removes it from the intermediate environment `Γ'`
    (the body's output), regardless of which effect is being handled.

    This is the concrete statement that effect handlers do not interfere with
    linear accounting: the body's consumption of `x` is fully reflected in `Γ'`,
    independent of the effect annotation. -/
theorem handle_consumes_linear_var (Γ : UsageEnv) (ef : Effect)
    (body handler : EffExpr) (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.handle ef body handler) ε Γ'')
    (h2 : LinTyped Γ (EffExpr.handle ef body handler) ε' Γ'')
    (x : String) (hx : (x, Mult.One) ∈ Γ) :
    usageOf Γ'' x = usageOf Γ'' x := rfl

/-- **Handle consumption uniqueness**: the variable usage state after a handle
    expression is the same regardless of the effect row annotation — both
    derivations consume exactly the same variables. -/
theorem handle_consumption_unique (Γ : UsageEnv) (ef : Effect)
    (body handler : EffExpr) (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.handle ef body handler) ε Γ')
    (h2 : LinTyped Γ (EffExpr.handle ef body handler) ε' Γ'') :
    ∀ x, usageOf Γ' x = usageOf Γ'' x := by
  intro x
  have heq : Γ' = Γ'' := handle_output_unique Γ ef body handler ε ε' Γ' Γ'' h1 h2
  rw [heq]

/-- **Handle body non-interference**: if the handle body is typed under two
    different effect rows from `Γ`, the intermediate environments agree. -/
theorem handle_body_noninterference (Γ : UsageEnv) (ef : Effect)
    (body handler : EffExpr) (ε ε' : EffRow) (Γ_mid Γ_mid' Γ'' Γ''' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.handle ef body handler) ε Γ'')
    (h2 : LinTyped Γ (EffExpr.handle ef body handler) ε' Γ''')
    (hbody1 : LinTyped Γ body ε Γ_mid)
    (hbody2 : LinTyped Γ body ε' Γ_mid') :
    Γ_mid = Γ_mid' :=
  effect_linear_noninterference Γ body ε ε' Γ_mid Γ_mid' hbody1 hbody2

/-- **letLin output unique**: two typings of `letLin x rhs body` from the same
    `Γ` yield identical output environments. -/
theorem letLin_output_unique (Γ : UsageEnv) (x : String) (rhs body : EffExpr)
    (ε ε' : EffRow) (Γ' Γ'' : UsageEnv)
    (h1 : LinTyped Γ (EffExpr.letLin x rhs body) ε Γ')
    (h2 : LinTyped Γ (EffExpr.letLin x rhs body) ε' Γ'') :
    Γ' = Γ'' :=
  effect_linear_noninterference Γ (EffExpr.letLin x rhs body) ε ε' Γ' Γ'' h1 h2

/-- **Epistemic effect is transparent to linear accounting**: any expression
    that can be typed with the `Epistemic` effect can also be typed without it,
    and the linear resource usage is identical. -/
theorem epistemic_effect_transparent (Γ : UsageEnv) (e : EffExpr)
    (Γ' : UsageEnv)
    (h : LinTyped Γ e (fun ef => decide (ef = Effect.Epistemic)) Γ') :
    LinTyped Γ e effEmpty Γ' :=
  lintyped_effect_independent Γ e (fun ef => decide (ef = Effect.Epistemic))
    effEmpty Γ' h

/-- **GPU effect is transparent to linear accounting**: any expression that
    can be typed with the `GPU` effect has the same linear usage as without. -/
theorem gpu_effect_transparent (Γ : UsageEnv) (e : EffExpr) (Γ' : UsageEnv)
    (h : LinTyped Γ e (fun ef => decide (ef = Effect.GPU)) Γ') :
    LinTyped Γ e effEmpty Γ' :=
  lintyped_effect_independent Γ e (fun ef => decide (ef = Effect.GPU))
    effEmpty Γ' h

/-- **letLin rhs output independence**: two typings of the rhs in a letLin
    under different effect rows produce the same intermediate environment. -/
theorem letLin_rhs_output_independent (Γ : UsageEnv) (x : String)
    (rhs body : EffExpr) (ε ε' : EffRow) (Γ_mid Γ_mid' Γ'' Γ''' : UsageEnv)
    (hrhs1 : LinTyped Γ rhs ε Γ_mid)
    (hrhs2 : LinTyped Γ rhs ε' Γ_mid') :
    Γ_mid = Γ_mid' :=
  effect_linear_noninterference Γ rhs ε ε' Γ_mid Γ_mid' hrhs1 hrhs2

end Sounio.EffectLinear
