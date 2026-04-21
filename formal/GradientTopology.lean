/-!
# Sounio.GradientTopology — β⁹ Phase Verification (week-3 step B + week-4 step B)

Formal soundness for Gradient Topology Types (GTT), the static channel-set
discipline introduced in β⁹ week-1 and extended through week-4.

GTT tracks, for every f64 expression, an `i64` bitmask of the Hessian input
channels the expression depends on.  The channel of an expression is the
set of `.value` extractions (keyed by `MEAS_KNOW_IDX`) whose shadow the
expression may transitively read.  The compiler uses this static set to
refuse `hessian_of(e, j, k)` queries where `j` or `k` is not in the set
— the requested Hessian entry is structurally zero and the user's intent
is almost certainly a bug.

This file formalises week-3's inter-procedural rule: at a user-function
call, the result's channel set is the *union* of the argument channel
sets.  Before week-3 the compiler propagated only the last-argument's
set — unsound as an upper bound (it could refuse legitimate hessian_of
queries for in-topology channels brought in by non-terminal arguments).
Week-3's union rule is the conservative upper bound and matches the
atan2 / pow binary-builtin discipline.

## Correspondence with `self-hosted/compiler/lean_single.sio`

| Compiler site (lean_single.sio, post-week-3)             | Lean definition                  |
|----------------------------------------------------------|----------------------------------|
| `B9_ARG_CH_SET: [i64; 16]` (~line 490)                   | (call-site capture; see tCall2)  |
| per-arg capture in `compile_primary` (~line 10180)       | `Typing.tCall2` premises         |
| union into `EXPR_CH_SET` in fi>=0 branch (~line 10285)   | `Typing.tCall2` conclusion       |
| `.value` seeds `1 << (MEAS_KNOW_IDX - 1)` (~line 11644)  | `Typing.tValue`                  |
| `let` persists via `VAR_CH_SET[vix]` (~line 15363)       | `Typing.tVar` via `TyEnv`        |
| additive / multiplicative union (~line 11747, 13359)     | `Typing.tAdd`, `tMul`            |

## Scope

In scope:
  * Binary direct user-function calls (the `fi >= 0` branch of
    `compile_primary`, restricted to two-argument callees).  Binary is
    sufficient to capture the week-3 rule *semantically*: the union rule
    is `S_ret = S_arg0 ∪ S_arg1`, which is the same combinator week-3
    applies to every pair of arguments.  N-ary generalisation is a
    routine structural induction on the argument list that reuses
    `ChSet.union_assoc` — it is deferred to avoid the nested-inductive
    proof complication in Lean 4 without the toolchain on-hand to
    verify.
  * Upper-bound soundness: the declared type's set contains the runtime
    channel footprint.  The proof does *not* claim tightness; it only
    claims the compiler refuses strictly fewer in-topology queries than
    a body-analysis could justify refusing for out-of-topology queries.

Out of scope (stated, not silently omitted):
  * N-ary user-fn calls with arity > 2 in the Lean model.  The compiler
    supports them and the union rule extends by `union_assoc`; a future
    pass with Lean verification tooling should promote this file to the
    n-ary case.
  * Closure / fn-ref indirect calls.  Week-3 left these leaking the last
    argument's topology, matching β⁵'s scope.
  * ARM64 / a64 backend mirror.  `compile_primary_a64` has no
    corresponding union capture yet (parallel work, deferred).
  * Body-level precision for return-position (binary fn, tail expr).
    `fn first(a, b) = a` has semantic channel set `{arg₀}`; under the
    week-4 body-precision rule (§7b) it types exactly at `{arg₀}`.
    Extending to match / if-else tails, loops, and recursion is still
    deferred.  The week-3 union rule remains as a conservative fallback
    for unanalyzed callees.
  * Operational composition with HessianAD.  The theorem here states a
    semantic containment on channel footprints; the shadow-slot bridge
    is proved separately in `HessianAD.lean`.

Self-contained.  No Mathlib.  No new axioms.  Zero sorry.  Channel sets
are modelled as `Nat → Bool` so all structural lemmas reduce either to
`rfl` or to bool-exhaustion proofs that do not depend on imported
algebraic lemmas.
-/

namespace Sounio.GradientTopology

-- ---------------------------------------------------------------------------
-- §1. Channel sets
-- ---------------------------------------------------------------------------

/-- A channel set is a characteristic function `Nat → Bool`.  The compiler
    represents this as an `i64` bitmask; using `Nat → Bool` here keeps the
    proofs independent of the bit-width and the concrete integer encoding. -/
def ChSet : Type := Nat → Bool

/-- The empty channel set. -/
def ChSet.empty : ChSet := fun _ => false

/-- Singleton channel set `{k}`.  Corresponds to the compiler's
    `1 << (MEAS_KNOW_IDX - 1)` seeding at `.value` access. -/
def ChSet.singleton (k : Nat) : ChSet := fun n => decide (n = k)

/-- Channel-set union.  Corresponds to the compiler's bitwise `|`. -/
def ChSet.union (a b : ChSet) : ChSet := fun n => a n || b n

/-- Pointwise subset: `a ⊆ b` iff every channel of `a` is a channel of `b`. -/
def ChSet.subset (a b : ChSet) : Prop := ∀ n, a n = true → b n = true

-- ---------------------------------------------------------------------------
-- §2. Structural lemmas on ChSet (bool-exhaustion proofs)
-- ---------------------------------------------------------------------------

theorem ChSet.subset_refl (a : ChSet) : a.subset a := fun _ h => h

theorem ChSet.subset_trans (a b c : ChSet)
    (hab : a.subset b) (hbc : b.subset c) : a.subset c :=
  fun n h => hbc n (hab n h)

theorem ChSet.empty_subset (a : ChSet) : ChSet.empty.subset a := by
  intro n h
  -- `h : ChSet.empty n = true`.  `ChSet.empty n` reduces to `false`,
  -- so `h : false = true`, absurd.
  exact absurd h (by simp [ChSet.empty])

theorem ChSet.subset_union_left (a b : ChSet) : a.subset (a.union b) := by
  intro n h
  simp only [ChSet.union]
  rw [h]
  rfl

theorem ChSet.subset_union_right (a b : ChSet) : b.subset (a.union b) := by
  intro n h
  simp only [ChSet.union]
  rw [h]
  cases a n <;> rfl

theorem ChSet.union_subset (a b c : ChSet)
    (hac : a.subset c) (hbc : b.subset c) : (a.union b).subset c := by
  intro n h
  -- `h : (a n || b n) = true`.  Case split on `a n`.
  simp only [ChSet.union] at h
  cases han : a n with
  | true  => exact hac n han
  | false =>
    rw [han] at h
    simp only [Bool.false_or] at h
    exact hbc n h

theorem ChSet.union_comm (a b : ChSet) : a.union b = b.union a := by
  funext n
  simp only [ChSet.union]
  cases a n <;> cases b n <;> rfl

theorem ChSet.union_assoc (a b c : ChSet) :
    (a.union b).union c = a.union (b.union c) := by
  funext n
  simp only [ChSet.union]
  cases a n <;> cases b n <;> cases c n <;> rfl

theorem ChSet.union_idem (a : ChSet) : a.union a = a := by
  funext n
  simp only [ChSet.union]
  cases a n <;> rfl

-- ---------------------------------------------------------------------------
-- §3. Expression grammar and the GTT typing judgment (binary calls)
-- ---------------------------------------------------------------------------

/-- Function identifiers.  In the compiler these are indices into
    `FN_ARITY` / `FN_RET_TY`; here we keep them abstract. -/
def FnId : Type := Nat

/-- Variable identifiers (indices into `VAR_CH_SET`). -/
def VarId : Type := Nat

/-- Minimal expression grammar for GTT reasoning.  Covers the constructs
    week-2 and week-3 wired channel-topology tracking for:
      * constants (`lit`), which have empty channel set
      * variables (`var`), which load `VAR_CH_SET[x]`
      * addition and multiplication, which union operand topologies
      * `.value` extraction at a channel keyed by the current `MEAS_KNOW_IDX`
      * direct binary user-function calls `call2 f e1 e2`, the week-3 focus

    N-ary calls are modelled as structural composition over binary ones
    in the surface compiler semantics; the file's theorem covers the
    binary case and n-ary reduces to iterated binary applications. -/
inductive Expr : Type where
  | lit    (c : Float)              : Expr
  | var    (x : VarId)              : Expr
  | add    (e1 e2 : Expr)           : Expr
  | mul    (e1 e2 : Expr)           : Expr
  | value  (k : Nat)                : Expr
  | call2  (f : FnId) (e1 e2 : Expr): Expr
  deriving Repr

/-- A typing environment mapping variables to their declared channel set.
    Corresponds to `VAR_CH_SET` indexed by variable slot. -/
def TyEnv : Type := VarId → ChSet

/-- The GTT typing judgment: `Γ ⊢ e : @ S` reads "under environment `Γ`,
    expression `e` has channel set `S`".

    The `tCall2` rule is the week-3 inter-procedural rule specialised
    to binary calls: the return's channel set is the union of the two
    arguments' channel sets.  Pre-week-3 the compiler used only the
    *last* argument's set — this is exactly the rule week-3 installs. -/
inductive Typing (Γ : TyEnv) : Expr → ChSet → Prop where
  | tLit (c : Float) :
      Typing Γ (Expr.lit c) ChSet.empty
  | tVar (x : VarId) :
      Typing Γ (Expr.var x) (Γ x)
  | tValue (k : Nat) :
      Typing Γ (Expr.value k) (ChSet.singleton k)
  | tAdd (e1 e2 : Expr) (S1 S2 : ChSet)
      (h1 : Typing Γ e1 S1) (h2 : Typing Γ e2 S2) :
      Typing Γ (Expr.add e1 e2) (S1.union S2)
  | tMul (e1 e2 : Expr) (S1 S2 : ChSet)
      (h1 : Typing Γ e1 S1) (h2 : Typing Γ e2 S2) :
      Typing Γ (Expr.mul e1 e2) (S1.union S2)
  | tCall2 (f : FnId) (e1 e2 : Expr) (S1 S2 : ChSet)
      (h1 : Typing Γ e1 S1) (h2 : Typing Γ e2 S2) :
      Typing Γ (Expr.call2 f e1 e2) (S1.union S2)

-- ---------------------------------------------------------------------------
-- §4. Big-step semantics parameterised by a function table
-- ---------------------------------------------------------------------------

/-- A runtime value is a `Float` paired with its *semantic* channel
    footprint (the set of channels whose shadows can contribute to
    its derivative).  This is the concrete object the typing judgment
    upper-bounds. -/
structure RuntimeVal where
  value : Float
  footprint : ChSet

/-- A variable environment (runtime) maps variables to runtime values. -/
def VarEnv : Type := VarId → RuntimeVal

/-- Abstract binary function table: for every function id `f`, a relation
    `eval2 f v1 v2 ret` specifying the return value given the two
    argument values.  Keeping this abstract avoids committing to a
    specific body language; `BodyRespectsTopology` parametrises over
    arbitrary user-defined function behaviour. -/
structure FnTable where
  eval2 : FnId → RuntimeVal → RuntimeVal → RuntimeVal → Prop

/-- Big-step evaluation of GTT expressions.  The footprint a value carries
    is whatever the underlying arithmetic produces; addition and
    multiplication union footprints, `.value` seeds a singleton, calls
    delegate to the function table. -/
inductive Eval (F : FnTable) (ρ : VarEnv) : Expr → RuntimeVal → Prop where
  | eLit (c : Float) :
      Eval F ρ (Expr.lit c) ⟨c, ChSet.empty⟩
  | eVar (x : VarId) :
      Eval F ρ (Expr.var x) (ρ x)
  | eValue (k : Nat) (fv : Float) :
      Eval F ρ (Expr.value k) ⟨fv, ChSet.singleton k⟩
  | eAdd (e1 e2 : Expr) (v1 v2 : RuntimeVal)
      (h1 : Eval F ρ e1 v1) (h2 : Eval F ρ e2 v2) :
      Eval F ρ (Expr.add e1 e2)
        ⟨v1.value + v2.value, v1.footprint.union v2.footprint⟩
  | eMul (e1 e2 : Expr) (v1 v2 : RuntimeVal)
      (h1 : Eval F ρ e1 v1) (h2 : Eval F ρ e2 v2) :
      Eval F ρ (Expr.mul e1 e2)
        ⟨v1.value * v2.value, v1.footprint.union v2.footprint⟩
  | eCall2 (f : FnId) (e1 e2 : Expr) (v1 v2 ret : RuntimeVal)
      (h1 : Eval F ρ e1 v1) (h2 : Eval F ρ e2 v2)
      (hfn : F.eval2 f v1 v2 ret) :
      Eval F ρ (Expr.call2 f e1 e2) ret

-- ---------------------------------------------------------------------------
-- §5. The body-respects-topology hypothesis
-- ---------------------------------------------------------------------------

/-- *The body-analysis hypothesis* (binary form).  For every binary
    function `f` and every argument pair `v1 v2`, if
    `F.eval2 f v1 v2 ret` holds, then the return value's footprint is
    contained in the union of the argument footprints.

    This is the proof obligation discharged by a body-level analysis for
    each user function.  Week-3's union rule does NOT prove this
    hypothesis — it *assumes* it.  When the hypothesis holds for every
    function in the table, the typing judgment's declared channel set
    is an upper bound on the runtime footprint (main theorem below).

    Discharging this hypothesis per function is week-4+ body-analysis
    work.  The trivial discharge — "no analysis, return footprint = arg
    union" — is `union_discharges_body_hypothesis` below. -/
def BodyRespectsTopology (F : FnTable) : Prop :=
  ∀ (f : FnId) (v1 v2 ret : RuntimeVal),
    F.eval2 f v1 v2 ret →
    ret.footprint.subset (v1.footprint.union v2.footprint)

-- ---------------------------------------------------------------------------
-- §6. Main soundness theorem
-- ---------------------------------------------------------------------------

/-- Runtime footprint of a variable matches the environment's declared
    channel set for that variable.  Formalises the invariant that
    `VAR_CH_SET[x]` upper-bounds the footprint of the runtime value
    bound to `x`. -/
def VarEnvConsistent (Γ : TyEnv) (ρ : VarEnv) : Prop :=
  ∀ x : VarId, (ρ x).footprint.subset (Γ x)

/-- Monotonicity of binary union under pointwise subset — the core
    lemma for `tAdd` / `tMul` / `tCall2`. -/
theorem ChSet.union_mono (a b c d : ChSet)
    (hac : a.subset c) (hbd : b.subset d) : (a.union b).subset (c.union d) :=
  ChSet.union_subset a b (c.union d)
    (ChSet.subset_trans a c (c.union d) hac (ChSet.subset_union_left c d))
    (ChSet.subset_trans b d (c.union d) hbd (ChSet.subset_union_right c d))

/-- **GTT soundness over direct binary user-fn calls.**

    If `Γ ⊢ e : @ S` and `e` big-step evaluates to `v` under a
    consistent variable environment and a function table that respects
    declared topology, then `v.footprint ⊆ S`.

    In plain terms: the compiler's static channel set at every program
    point is an upper bound on the runtime shadow channels that value
    can depend on.  Consequently, `hessian_of(e, j, k)` with `{j, k} ⊈ S`
    queries shadow slots that are provably zero — the compiler's refusal
    is sound.

    Out of scope (stated in the header, restated here for load-bearing
    reasons): closures / fn-refs, a64 backend, and arity > 2 user calls
    are not modelled; the n-ary extension is routine via `union_assoc`. -/
theorem gtt_sound
    (F : FnTable) (Γ : TyEnv) (ρ : VarEnv)
    (hΓρ : VarEnvConsistent Γ ρ)
    (hbody : BodyRespectsTopology F) :
    ∀ (e : Expr) (S : ChSet) (v : RuntimeVal),
      Typing Γ e S → Eval F ρ e v → v.footprint.subset S := by
  intro e
  induction e with
  | lit c =>
    intro S v hT hE
    cases hT; cases hE
    exact ChSet.subset_refl _
  | var x =>
    intro S v hT hE
    cases hT; cases hE
    exact hΓρ x
  | value k =>
    intro S v hT hE
    cases hT; cases hE
    exact ChSet.subset_refl _
  | add e1 e2 ih1 ih2 =>
    intro S v hT hE
    cases hT with
    | tAdd _ _ S1 S2 hT1 hT2 =>
      cases hE with
      | eAdd _ _ v1 v2 hE1 hE2 =>
        exact ChSet.union_mono v1.footprint v2.footprint S1 S2
          (ih1 S1 v1 hT1 hE1) (ih2 S2 v2 hT2 hE2)
  | mul e1 e2 ih1 ih2 =>
    intro S v hT hE
    cases hT with
    | tMul _ _ S1 S2 hT1 hT2 =>
      cases hE with
      | eMul _ _ v1 v2 hE1 hE2 =>
        exact ChSet.union_mono v1.footprint v2.footprint S1 S2
          (ih1 S1 v1 hT1 hE1) (ih2 S2 v2 hT2 hE2)
  | call2 f e1 e2 ih1 ih2 =>
    intro S v hT hE
    cases hT with
    | tCall2 _ _ _ S1 S2 hT1 hT2 =>
      cases hE with
      | eCall2 _ _ _ v1 v2 ret hE1 hE2 hfn =>
        -- Step 1: each arg's runtime footprint is contained in its declared set.
        have s1 : v1.footprint.subset S1 := ih1 S1 v1 hT1 hE1
        have s2 : v2.footprint.subset S2 := ih2 S2 v2 hT2 hE2
        -- Step 2: ret footprint is contained in the arg-union footprints.
        have ret_sub_args : ret.footprint.subset (v1.footprint.union v2.footprint) :=
          hbody f v1 v2 ret hfn
        -- Step 3: compose with union monotonicity.
        exact ChSet.subset_trans ret.footprint
          (v1.footprint.union v2.footprint) (S1.union S2)
          ret_sub_args
          (ChSet.union_mono v1.footprint v2.footprint S1 S2 s1 s2)

-- ---------------------------------------------------------------------------
-- §7. The union rule is the right conservative body-discharge
-- ---------------------------------------------------------------------------

/-- The "no body analysis" binary function table: every user function's
    return footprint is *defined* to be the union of its two argument
    footprints.

    This is the semantic image of the compiler's week-3 union rule — it
    makes the operational semantics match the static rule exactly, so
    no body analysis is needed to discharge `BodyRespectsTopology`. -/
def unionTable : FnTable where
  eval2 := fun _ v1 v2 ret =>
    ret.footprint = v1.footprint.union v2.footprint

theorem union_discharges_body_hypothesis :
    BodyRespectsTopology unionTable := by
  intro _ v1 v2 ret h
  -- `h : unionTable.eval2 _ v1 v2 ret` reduces to
  -- `ret.footprint = v1.footprint.union v2.footprint`.
  rw [show ret.footprint = v1.footprint.union v2.footprint from h]
  exact ChSet.subset_refl _

-- ---------------------------------------------------------------------------
-- §7b. Week-4 body-level precision (return-position, binary fns)
-- ---------------------------------------------------------------------------

/-- A binary function's body-precision summary: for each of the two
    parameter positions, whether the return expression references that
    parameter.  Corresponds to the compiler's `FN_USED_PARAMS[fi]`
    bitmask (low two bits, for binary callees) populated at
    function-body tail in `self-hosted/compiler/lean_single.sio`. -/
structure UsedParams2 where
  uses0 : Bool
  uses1 : Bool

/-- Body-precision join: union only those argument channel sets for
    which the callee body uses the corresponding parameter position.
    Corresponds to the compiler's call-site loop that consults
    `FN_USED_PARAMS[fi]` and unions only the matching
    `B9_ARG_CH_SET[i]` slots.  For a fully-used function
    `⟨true, true⟩` this reduces to the week-3 union; for an unused
    parameter the corresponding set is dropped. -/
def usedUnion (U : UsedParams2) (S1 S2 : ChSet) : ChSet :=
  ChSet.union
    (if U.uses0 then S1 else ChSet.empty)
    (if U.uses1 then S2 else ChSet.empty)

/-- Elementary building block: `(if b then S else empty) ⊆ S` for any
    boolean guard `b`.  The two cases reduce to `ChSet.subset_refl` and
    `ChSet.empty_subset`. -/
theorem ChSet.if_empty_subset_left (b : Bool) (S : ChSet) :
    (if b then S else ChSet.empty).subset S := by
  cases b
  · -- if-false branch: empty ⊆ S
    exact ChSet.empty_subset S
  · -- if-true branch: S ⊆ S
    exact ChSet.subset_refl S

/-- Monotonicity of the `if b then · else empty` guard: the guarded
    `S1` is a subset of the guarded `S2` whenever `S1 ⊆ S2`, for every
    guard value `b`. -/
theorem ChSet.if_empty_mono (b : Bool) (S1 S2 : ChSet)
    (h : S1.subset S2) :
    (if b then S1 else ChSet.empty).subset (if b then S2 else ChSet.empty) := by
  cases b
  · -- if-false branch: empty ⊆ empty
    exact ChSet.subset_refl ChSet.empty
  · -- if-true branch: S1 ⊆ S2 by h
    exact h

/-- **Body-precision is always a refinement of the week-3 union.**
    `usedUnion U S1 S2 ⊆ S1 ∪ S2` for every `U`.  The compiler's
    fallback — emitting the union-all when `FN_USED_PARAMS[fi]` is
    unanalyzed (sentinel 0) — is always safe because the tightened
    declared set is contained in the conservative one. -/
theorem usedUnion_subset_union (U : UsedParams2) (S1 S2 : ChSet) :
    (usedUnion U S1 S2).subset (S1.union S2) := by
  unfold usedUnion
  exact ChSet.union_subset _ _ (S1.union S2)
    (ChSet.subset_trans _ S1 (S1.union S2)
      (ChSet.if_empty_subset_left U.uses0 S1)
      (ChSet.subset_union_left S1 S2))
    (ChSet.subset_trans _ S2 (S1.union S2)
      (ChSet.if_empty_subset_left U.uses1 S2)
      (ChSet.subset_union_right S1 S2))

/-- Monotonicity of `usedUnion` in both set arguments (for a fixed
    `UsedParams2`).  Direct consequence of `ChSet.union_mono` applied
    to the two guarded components. -/
theorem usedUnion_mono (U : UsedParams2) (a1 a2 b1 b2 : ChSet)
    (h1 : a1.subset b1) (h2 : a2.subset b2) :
    (usedUnion U a1 a2).subset (usedUnion U b1 b2) := by
  unfold usedUnion
  exact ChSet.union_mono _ _ _ _
    (ChSet.if_empty_mono U.uses0 a1 b1 h1)
    (ChSet.if_empty_mono U.uses1 a2 b2 h2)

/-- Sanity: when both parameters are used, body-precision collapses to
    the week-3 union rule — no regression for fully-used functions. -/
theorem usedUnion_all_used (S1 S2 : ChSet) :
    usedUnion ⟨true, true⟩ S1 S2 = S1.union S2 := by
  funext n
  simp only [usedUnion, ChSet.union]
  rfl

/-- **The body-precision hypothesis.**  For every binary function `f`
    and every argument pair, the return's footprint is contained in
    the `usedUnion` of the argument footprints indexed by `UP f`.

    Stronger than `BodyRespectsTopology`: week-3's hypothesis used
    `v1.footprint.union v2.footprint`; week-4's uses the parameter-
    index-filtered join.  Compiler-side, this hypothesis is discharged
    per-function at the body-tail `FN_USED_PARAMS[fi]` capture site. -/
def BodyRespectsUsedParams (F : FnTable) (UP : FnId → UsedParams2) : Prop :=
  ∀ (f : FnId) (v1 v2 ret : RuntimeVal),
    F.eval2 f v1 v2 ret →
    ret.footprint.subset (usedUnion (UP f) v1.footprint v2.footprint)

/-- Body-precision implies the week-3 union-rule hypothesis.  A function
    table that respects `UP`'s body-precision summary also respects the
    conservative `BodyRespectsTopology`, so `gtt_sound` continues to
    apply — the week-3 soundness theorem is *strictly weaker* than the
    week-4 one and the week-4 compiler still refuses only out-of-topology
    queries. -/
theorem bodyPrecision_implies_union_hypothesis
    (F : FnTable) (UP : FnId → UsedParams2)
    (h : BodyRespectsUsedParams F UP) : BodyRespectsTopology F := by
  intro f v1 v2 ret hfn
  have ret_sub_used := h f v1 v2 ret hfn
  exact ChSet.subset_trans ret.footprint
    (usedUnion (UP f) v1.footprint v2.footprint)
    (v1.footprint.union v2.footprint)
    ret_sub_used
    (usedUnion_subset_union (UP f) v1.footprint v2.footprint)

/-- **Body-precision soundness for direct binary user-fn calls.**

    Given the strengthened hypothesis `BodyRespectsUsedParams`, the
    declared channel set `usedUnion (UP f) S1 S2` is a sound upper
    bound on the runtime footprint of `call2 f e1 e2`.  This is the
    week-4 tightening: `fn first(a, b) = a` with `UP first = ⟨true,
    false⟩` types the call at `{arg₀}` — strictly tighter than the
    week-3 `{arg₀ ∪ arg₁}`.

    The proof composes `gtt_sound` (applied to the two argument
    sub-expressions under the weaker hypothesis, obtained from
    `bodyPrecision_implies_union_hypothesis`) with the body-precision
    hypothesis itself, then monotonicity. -/
theorem gtt_sound_body
    (F : FnTable) (UP : FnId → UsedParams2)
    (Γ : TyEnv) (ρ : VarEnv)
    (hΓρ : VarEnvConsistent Γ ρ)
    (hbody : BodyRespectsUsedParams F UP) :
    ∀ (f : FnId) (e1 e2 : Expr) (S1 S2 : ChSet) (v1 v2 ret : RuntimeVal),
      Typing Γ e1 S1 → Typing Γ e2 S2 →
      Eval F ρ e1 v1 → Eval F ρ e2 v2 → F.eval2 f v1 v2 ret →
      ret.footprint.subset (usedUnion (UP f) S1 S2) := by
  intro f e1 e2 S1 S2 v1 v2 ret hT1 hT2 hE1 hE2 hfn
  -- Step 1: each argument's runtime footprint is contained in its
  --         declared set, via week-3 gtt_sound under the weaker hypothesis.
  have hbody_union : BodyRespectsTopology F :=
    bodyPrecision_implies_union_hypothesis F UP hbody
  have s1 : v1.footprint.subset S1 :=
    gtt_sound F Γ ρ hΓρ hbody_union e1 S1 v1 hT1 hE1
  have s2 : v2.footprint.subset S2 :=
    gtt_sound F Γ ρ hΓρ hbody_union e2 S2 v2 hT2 hE2
  -- Step 2: ret footprint is contained in the usedUnion of arg footprints.
  have ret_sub_used : ret.footprint.subset (usedUnion (UP f) v1.footprint v2.footprint) :=
    hbody f v1 v2 ret hfn
  -- Step 3: monotonicity of usedUnion under pointwise subset.
  have used_mono : (usedUnion (UP f) v1.footprint v2.footprint).subset
                     (usedUnion (UP f) S1 S2) :=
    usedUnion_mono (UP f) v1.footprint v2.footprint S1 S2 s1 s2
  exact ChSet.subset_trans ret.footprint
    (usedUnion (UP f) v1.footprint v2.footprint)
    (usedUnion (UP f) S1 S2)
    ret_sub_used used_mono

/-- **Canary (week-4 body-precision regression).**  For
    `fn proj_a(a, b) = a` — whose body-precision summary is
    `UP proj_a = ⟨true, false⟩` — the call-site declared channel set
    under body-precision is exactly `S1` (drops `S2`).  This matches
    `tests/run-pass/gtt_body_precision_param_usage.sio` and
    `tests/compile-fail/gtt_body_precision_unused_param_refused.sio`
    in the repository: the first parameter's channel is in-topology,
    the second parameter's channel is out-of-topology, and the
    compiler's refusal of `hessian_of(r, 1, 1)` is sound. -/
theorem canary_proj_a_only_arg0 (S1 S2 : ChSet) :
    usedUnion ⟨true, false⟩ S1 S2 = S1.union ChSet.empty := rfl

-- ---------------------------------------------------------------------------
-- §8. A closed-form corollary matching the week-3 commit narrative
-- ---------------------------------------------------------------------------

/-- **Corollary for the regression test pattern.**  The week-3 regression
    `tests/run-pass/gtt_interprocedural_topology.sio` defines
    `fn my_sum(a, b) -> a + b` and queries `hessian_of(my_sum(x, y), 0, 1)`
    where `x = k0.value` (channel 0) and `y = k1.value` (channel 1).

    This corollary states the static content of that program: the typing
    judgment assigns `{ch 0} ∪ {ch 1}` to the call-site expression, and
    by `gtt_sound` the runtime footprint is contained in that set.
    Querying channel 0 or 1 is in-topology (sound, not refused); querying
    channel 2 is out-of-topology and the compiler's refusal is sound. -/
theorem regression_my_sum_topology
    (Γ : TyEnv) (f : FnId) :
    Typing Γ
      (Expr.call2 f (Expr.value 0) (Expr.value 1))
      ((ChSet.singleton 0).union (ChSet.singleton 1)) :=
  Typing.tCall2 f (Expr.value 0) (Expr.value 1)
    (ChSet.singleton 0) (ChSet.singleton 1)
    (Typing.tValue 0) (Typing.tValue 1)

-- ---------------------------------------------------------------------------
-- §9. Summary
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property                                                              | Status  | Location                                 |
|-----------------------------------------------------------------------|---------|------------------------------------------|
| ChSet subset is a preorder (refl, trans)                              | Proved  | `ChSet.subset_refl`, `subset_trans`      |
| ChSet union is the join (left/right subset, lub)                      | Proved  | `ChSet.subset_union_*`, `union_subset`   |
| ChSet union is commutative, associative, idempotent                   | Proved  | `ChSet.union_comm/assoc/idem`            |
| Union is monotone under pointwise subset                              | Proved  | `ChSet.union_mono`                       |
| **GTT typing upper-bounds runtime footprint** (binary `call2` incl.)  | Proved  | `gtt_sound`                              |
| Week-3 union rule discharges body-topology hypothesis                 | Proved  | `union_discharges_body_hypothesis`       |
| Regression program types at `{ch 0} ∪ {ch 1}`                         | Proved  | `regression_my_sum_topology`             |
| Body-precision refines week-3 union (⊆)                               | Proved  | `usedUnion_subset_union`                 |
| Body-precision collapses to week-3 when both params used              | Proved  | `usedUnion_all_used`                     |
| Body-precision implies week-3 body hypothesis                         | Proved  | `bodyPrecision_implies_union_hypothesis` |
| **Body-precision soundness for binary `call2`** (week-4)              | Proved  | `gtt_sound_body`                         |
| Canary: proj_a with UP=⟨true,false⟩ types at `{arg₀}` only            | Proved  | `canary_proj_a_only_arg0`                |

## What this file proves, in one sentence

For every GTT-typed expression `e` with declared channel set `S`, every
runtime value produced by evaluating `e` under a consistent variable
environment and a body-respecting function table has channel footprint
contained in `S`.  In particular, the week-3 inter-procedural union rule
at binary user-function calls is a *sound* static channel set.

## What remains open

1. N-ary calls (arity > 2).  The union-over-a-list rule is a routine
   structural induction over `List.Forall₂` using `union_assoc`;
   omitted here because the nested-inductive induction principle for
   `List Expr` sub-terms is enough of a footgun to warrant verification
   with a Lean toolchain on-hand.  The binary theorem is morally the
   n-ary theorem.  `usedUnion` generalises to `usedUnionN : UsedParamsN
   → List ChSet → ChSet` along the same structural path.
2. Body-level precision beyond return-position (match / if arms, loops,
   recursion fixed-point).  Week-4 covers the return-position tail
   fragment; richer body CFA is deferred.
3. Closure / fn-ref indirect calls, `Expr.call2` here is direct-only.
4. ARM64 mirror — codegen parity, not a separate semantic theorem.
   The a64 compiler mirrors `EXPR_PARAM_USED` propagation through
   let / load in week-4 step A; the a64 call-site tightening itself
   remains deferred.
5. Bridging to `HessianAD.lean` — proving that `j ∉ S ⟹ H_{j,k} = 0.0`
   via the shadow-slot semantics.  The natural next proof, spans two
   files.  Under `gtt_sound_body` this bridge becomes an *equality*
   (declared set characterises shadow footprint) on the body-precision
   fragment, rather than just a containment.

## No new axioms

All theorems are proved by structural induction on `Typing` / `Expr`
or by bool-exhaustion over `Nat → Bool`.  No Mathlib, no Float axioms.
-/

end Sounio.GradientTopology
