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
abbrev ChSet : Type := Nat → Bool

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

/-- Monotonicity of binary union under pointwise subset — the core
    lemma for `tAdd` / `tMul` / `tCall2`.  Moved up from §6 so §2b's
    `chsetUnionFin_mono` can reuse it. -/
theorem ChSet.union_mono (a b c d : ChSet)
    (hac : a.subset c) (hbd : b.subset d) : (a.union b).subset (c.union d) :=
  ChSet.union_subset a b (c.union d)
    (ChSet.subset_trans a c (c.union d) hac (ChSet.subset_union_left c d))
    (ChSet.subset_trans b d (c.union d) hbd (ChSet.subset_union_right c d))

-- ---------------------------------------------------------------------------
-- §2b. Fin-indexed channel-set union (n-ary support)
-- ---------------------------------------------------------------------------

/-- `Fin n`-indexed channel-set union.  The Fin-based analogue of
    `ChSet.unionL` (§7c) — structurally simpler for `Expr.callN`
    reasoning because the argument type `Fin n → Expr` exposes indices
    directly to induction.  Defined by recursion on `n`. -/
def chsetUnionFin : {n : Nat} → (Fin n → ChSet) → ChSet
  | 0,   _  => ChSet.empty
  | n+1, Ss => (Ss 0).union (chsetUnionFin (fun i : Fin n => Ss i.succ))

theorem chsetUnionFin_zero (Ss : Fin 0 → ChSet) :
    chsetUnionFin Ss = ChSet.empty := rfl

theorem chsetUnionFin_succ {n : Nat} (Ss : Fin (n+1) → ChSet) :
    chsetUnionFin Ss = (Ss 0).union (chsetUnionFin (fun i : Fin n => Ss i.succ)) := rfl

/-- **Monotonicity of `chsetUnionFin`** under pointwise subset.  Fin
    analogue of `ChSet.unionL_mono` / `ChSet.union_mono`.  Proved by
    `n`-induction. -/
theorem chsetUnionFin_mono : {n : Nat} → (Ss Ts : Fin n → ChSet)
    → (∀ i : Fin n, (Ss i).subset (Ts i))
    → (chsetUnionFin Ss).subset (chsetUnionFin Ts)
  | 0,   _,  _,  _  => by
      rw [chsetUnionFin_zero, chsetUnionFin_zero]
      exact ChSet.subset_refl _
  | n+1, Ss, Ts, h => by
      rw [chsetUnionFin_succ, chsetUnionFin_succ]
      exact ChSet.union_mono _ _ _ _
        (h 0)
        (chsetUnionFin_mono _ _ (fun i : Fin n => h i.succ))

/-- Membership in the Fin-indexed union: every `Ss i` is a subset of
    `chsetUnionFin Ss`. -/
theorem subset_chsetUnionFin : {n : Nat} → (Ss : Fin n → ChSet) → (i : Fin n)
    → (Ss i).subset (chsetUnionFin Ss)
  | 0,   _,  i => absurd i.isLt (by simp)
  | n+1, Ss, i => by
      rw [chsetUnionFin_succ]
      -- Case on whether i = 0 or i = j.succ for some j : Fin n
      match h : i with
      | ⟨0,     _⟩ =>
          -- i = 0 → Ss 0 ⊆ Ss 0 ∪ rest
          exact ChSet.subset_union_left (Ss 0) _
      | ⟨k+1, hk⟩ =>
          -- i = ⟨k+1, hk⟩ = Fin.succ ⟨k, _⟩
          have hk' : k < n := Nat.lt_of_succ_lt_succ hk
          have : Ss ⟨k+1, hk⟩ = (fun j : Fin n => Ss j.succ) ⟨k, hk'⟩ := rfl
          rw [this]
          exact ChSet.subset_trans _ _ _
            (subset_chsetUnionFin (fun j : Fin n => Ss j.succ) ⟨k, hk'⟩)
            (ChSet.subset_union_right (Ss 0) _)

-- ---------------------------------------------------------------------------
-- §3. Expression grammar and the GTT typing judgment (binary + n-ary calls)
-- ---------------------------------------------------------------------------

/-- Function identifiers.  In the compiler these are indices into
    `FN_ARITY` / `FN_RET_TY`; here we keep them abstract. -/
abbrev FnId : Type := Nat

/-- Variable identifiers (indices into `VAR_CH_SET`). -/
abbrev VarId : Type := Nat

/-- Minimal expression grammar for GTT reasoning.  Covers the constructs
    week-2 and week-3 wired channel-topology tracking for:
      * constants (`lit`), which have empty channel set
      * variables (`var`), which load `VAR_CH_SET[x]`
      * addition and multiplication, which union operand topologies
      * `.value` extraction at a channel keyed by the current `MEAS_KNOW_IDX`
      * direct binary user-function calls `call2 f e1 e2`, the week-3 focus
      * direct n-ary user-function calls `callN f args` — week-5 surface
        extension, parameterised by `{n : Nat}` and `args : Fin n → Expr`
        to stay strictly positive (the Lean 4 `induction` tactic rejects
        nested inductives like `args : List Expr`, per the β⁹ week-3
        "nested-inductive induction principle ... footgun" scope note). -/
inductive Expr : Type where
  | lit    (c : Float)              : Expr
  | var    (x : VarId)              : Expr
  | add    (e1 e2 : Expr)           : Expr
  | mul    (e1 e2 : Expr)           : Expr
  | value  (k : Nat)                : Expr
  | call2  (f : FnId) (e1 e2 : Expr): Expr
  | callN  (f : FnId) {n : Nat} (args : Fin n → Expr) : Expr
  | ifE    (cond e1 e2 : Expr)      : Expr

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
  | tCallN (f : FnId) {n : Nat} (args : Fin n → Expr) (Ss : Fin n → ChSet)
      (h : ∀ i : Fin n, Typing Γ (args i) (Ss i)) :
      Typing Γ (Expr.callN f args) (chsetUnionFin Ss)
  | tIf (cond e1 e2 : Expr) (Sc S1 S2 : ChSet)
      (hc : Typing Γ cond Sc) (h1 : Typing Γ e1 S1) (h2 : Typing Γ e2 S2) :
      Typing Γ (Expr.ifE cond e1 e2) (Sc.union (S1.union S2))

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
  /-- N-ary abstract function table (week-5).  Returns the relation
      between an `n`-tuple of argument values and the return value for
      function `f`.  The arity `n` is part of the relation so the same
      `FnTable` can host callees of different arities simultaneously
      — matching the compiler's `FN_ARITY` table at
      `self-hosted/compiler/lean_single.sio:~485`. -/
  evalN : FnId → (n : Nat) → (Fin n → RuntimeVal) → RuntimeVal → Prop

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
  | eCallN (f : FnId) {n : Nat} (args : Fin n → Expr)
           (vs : Fin n → RuntimeVal) (ret : RuntimeVal)
      (h : ∀ i : Fin n, Eval F ρ (args i) (vs i))
      (hfn : F.evalN f n vs ret) :
      Eval F ρ (Expr.callN f args) ret
  /-- `if` big-step semantics — then-arm selected.  The result value's
      `.value` is the then-arm's, and the footprint is the union of the
      cond's footprint and the then-arm's footprint (the else-arm didn't
      run, so its footprint doesn't contribute).  Both arms must admit
      an evaluation to preserve well-formedness — matches the compiler's
      "both arms compiled, one executed" contract. -/
  | eIfThen (cond e1 e2 : Expr) (vc v1 : RuntimeVal)
      (hc : Eval F ρ cond vc)
      (h1 : Eval F ρ e1 v1) :
      Eval F ρ (Expr.ifE cond e1 e2)
        ⟨v1.value, vc.footprint.union v1.footprint⟩
  /-- `if` big-step semantics — else-arm selected.  Mirror of `eIfThen`. -/
  | eIfElse (cond e1 e2 : Expr) (vc v2 : RuntimeVal)
      (hc : Eval F ρ cond vc)
      (h2 : Eval F ρ e2 v2) :
      Eval F ρ (Expr.ifE cond e1 e2)
        ⟨v2.value, vc.footprint.union v2.footprint⟩

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
def BodyRespectsTopology2 (F : FnTable) : Prop :=
  ∀ (f : FnId) (v1 v2 ret : RuntimeVal),
    F.eval2 f v1 v2 ret →
    ret.footprint.subset (v1.footprint.union v2.footprint)

/-- N-ary analogue of `BodyRespectsTopology2`: for every `n`-ary callee,
    the return's footprint is contained in the n-ary union of the
    argument footprints.  Week-5. -/
def BodyRespectsTopologyN (F : FnTable) : Prop :=
  ∀ (f : FnId) (n : Nat) (vs : Fin n → RuntimeVal) (ret : RuntimeVal),
    F.evalN f n vs ret →
    ret.footprint.subset (chsetUnionFin (fun i => (vs i).footprint))

/-- Combined body-respects hypothesis: both the binary and n-ary call
    rules are sound.  `gtt_sound` needs both when the expression tree
    contains a mixture of `call2` and `callN` nodes. -/
def BodyRespectsTopology (F : FnTable) : Prop :=
  BodyRespectsTopology2 F ∧ BodyRespectsTopologyN F

-- ---------------------------------------------------------------------------
-- §6. Main soundness theorem
-- ---------------------------------------------------------------------------

/-- Runtime footprint of a variable matches the environment's declared
    channel set for that variable.  Formalises the invariant that
    `VAR_CH_SET[x]` upper-bounds the footprint of the runtime value
    bound to `x`. -/
def VarEnvConsistent (Γ : TyEnv) (ρ : VarEnv) : Prop :=
  ∀ x : VarId, (ρ x).footprint.subset (Γ x)

-- `ChSet.union_mono` lives at §2 (moved up so §2b's `chsetUnionFin_mono`
-- can reuse it).  The §6 original location is kept as this comment
-- pointer so the §6 proof chain (`union_mono` → `gtt_sound`) still reads
-- linearly.

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
    reasons): closures / fn-refs, a64 backend.  Arity > 2 user calls
    are covered via the `tCallN`/`eCallN` constructors (week-5).  The
    closure/fn-ref indirect call case remains deferred. -/
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
      | @eCall2 _ _ _ v1 v2 _ hE1 hE2 hfn =>
        -- Step 1: each arg's runtime footprint is contained in its declared set.
        have s1 : v1.footprint.subset S1 := ih1 S1 v1 hT1 hE1
        have s2 : v2.footprint.subset S2 := ih2 S2 v2 hT2 hE2
        -- Step 2: ret footprint (here: `v`, the outer intro'd return) is
        --         contained in the arg-union footprints.
        have ret_sub_args : v.footprint.subset (v1.footprint.union v2.footprint) :=
          hbody.1 f v1 v2 v hfn
        -- Step 3: compose with union monotonicity.
        exact ChSet.subset_trans v.footprint
          (v1.footprint.union v2.footprint) (S1.union S2)
          ret_sub_args
          (ChSet.union_mono v1.footprint v2.footprint S1 S2 s1 s2)
  | callN f args ih =>
    -- ih : ∀ i : Fin n, ∀ S v, Typing Γ (args i) S → Eval F ρ (args i) v
    --                    → v.footprint.subset S
    intro S v hT hE
    cases hT with
    | tCallN _ _ Ss hTi =>
      cases hE with
      | eCallN _ _ vs _ hEi hfn =>
        -- Step 1: each arg i's runtime footprint is contained in Ss i.
        have si : ∀ i, (vs i).footprint.subset (Ss i) :=
          fun i => ih i (Ss i) (vs i) (hTi i) (hEi i)
        -- Step 2: v.footprint ⊆ chsetUnionFin (fun i => (vs i).footprint)
        have ret_sub_args :
            v.footprint.subset (chsetUnionFin (fun i => (vs i).footprint)) :=
          hbody.2 f _ vs v hfn
        -- Step 3: monotonicity.
        exact ChSet.subset_trans v.footprint
          (chsetUnionFin (fun i => (vs i).footprint))
          (chsetUnionFin Ss)
          ret_sub_args
          (chsetUnionFin_mono _ _ si)
  | ifE cond e1 e2 ihc ih1 ih2 =>
    intro S v hT hE
    cases hT with
    | tIf _ _ _ Sc S1 S2 hTc hT1 hT2 =>
      -- Case on which arm was selected at runtime.
      cases hE with
      | eIfThen _ _ _ vc v1 hEc hE1 =>
        -- v = ⟨v1.value, vc.footprint ∪ v1.footprint⟩
        -- Goal: (vc.footprint ∪ v1.footprint).subset (Sc ∪ (S1 ∪ S2))
        have sc : vc.footprint.subset Sc := ihc Sc vc hTc hEc
        have s1 : v1.footprint.subset S1 := ih1 S1 v1 hT1 hE1
        -- v1 ⊆ S1 ⊆ S1 ∪ S2
        have s1' : v1.footprint.subset (S1.union S2) :=
          ChSet.subset_trans _ S1 _ s1 (ChSet.subset_union_left S1 S2)
        exact ChSet.union_mono _ _ _ _ sc s1'
      | eIfElse _ _ _ vc v2 hEc hE2 =>
        have sc : vc.footprint.subset Sc := ihc Sc vc hTc hEc
        have s2 : v2.footprint.subset S2 := ih2 S2 v2 hT2 hE2
        have s2' : v2.footprint.subset (S1.union S2) :=
          ChSet.subset_trans _ S2 _ s2 (ChSet.subset_union_right S1 S2)
        exact ChSet.union_mono _ _ _ _ sc s2'

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
  evalN := fun _ _ vs ret =>
    ret.footprint = chsetUnionFin (fun i => (vs i).footprint)

theorem union_discharges_body_hypothesis :
    BodyRespectsTopology unionTable := by
  refine ⟨?_, ?_⟩
  · intro _ v1 v2 ret h
    -- `h : unionTable.eval2 _ v1 v2 ret` reduces to
    -- `ret.footprint = v1.footprint.union v2.footprint`.
    rw [show ret.footprint = v1.footprint.union v2.footprint from h]
    exact ChSet.subset_refl _
  · intro _ _ vs ret h
    rw [show ret.footprint = chsetUnionFin (fun i => (vs i).footprint) from h]
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
    (h : BodyRespectsUsedParams F UP) : BodyRespectsTopology2 F := by
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
    (hbody : BodyRespectsUsedParams F UP)
    (hbodyN : BodyRespectsTopologyN F) :
    ∀ (f : FnId) (e1 e2 : Expr) (S1 S2 : ChSet) (v1 v2 ret : RuntimeVal),
      Typing Γ e1 S1 → Typing Γ e2 S2 →
      Eval F ρ e1 v1 → Eval F ρ e2 v2 → F.eval2 f v1 v2 ret →
      ret.footprint.subset (usedUnion (UP f) S1 S2) := by
  intro f e1 e2 S1 S2 v1 v2 ret hT1 hT2 hE1 hE2 hfn
  -- Step 1: each argument's runtime footprint is contained in its
  --         declared set, via week-3 gtt_sound under the weaker hypothesis.
  have hbody_union : BodyRespectsTopology F :=
    ⟨bodyPrecision_implies_union_hypothesis F UP hbody, hbodyN⟩
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
-- §7c. N-ary body-level precision (ChSet list machinery)
-- ---------------------------------------------------------------------------

/-!
The compiler supports n-ary user-fn calls at arity up to 16, per the
`B9_ARG_CH_SET[16]` / `B9_ARG_PARAM_USED[16]` / `FN_USED_PARAMS[fi]`
bitmask layout in `self-hosted/compiler/lean_single.sio:~490` and the
call-site iteration at `compile_primary:~10300`.  §7b proves the
soundness theorem for the binary specialisation; this section lifts the
underlying set-theoretic union machinery to arbitrary arity and states
the abstract n-ary soundness theorem over a list of per-argument
runtime footprints and declared channel sets.

**Scope note (week-5).**  §2b introduced the Fin-based `chsetUnionFin`
and extended `Expr` with `callN` + `gtt_sound` with a `tCallN` case,
supplanting the week-5 scope-note's "ChSet-level only" framing.  This
§7c section's `List ChSet` machinery remains useful as an alternative
statement shape that matches the compiler's `List`-of-16 layout at the
call site, and `gtt_sound_bodyN` still provides the pure set-theoretic
step-3 composition.  §7d pairs the Fin-indexed machinery with
body-precision (`UsedParamsFin` / `usedUnionFin`) and proves
`gtt_sound_body_callN`, the n-ary Expr-surface analogue of
`gtt_sound_body`.
-/

/-- List-indexed channel-set join.  Folds `ChSet.union` right-
    associatively over a list of channel sets; the result is the
    smallest set containing every element of `Ss`.  N-ary analogue of
    `ChSet.union`. -/
def ChSet.unionL : List ChSet → ChSet
  | []       => ChSet.empty
  | S :: Ss  => S.union (ChSet.unionL Ss)

theorem ChSet.unionL_nil : ChSet.unionL [] = ChSet.empty := rfl

theorem ChSet.unionL_cons (S : ChSet) (Ss : List ChSet) :
    ChSet.unionL (S :: Ss) = S.union (ChSet.unionL Ss) := rfl

/-- Pointwise-subset relation between two lists of channel sets.  The
    two lists must have the same length (enforced by the constructor
    structure), and each corresponding pair must satisfy
    `ChSet.subset`.  Used to state `unionL_mono` without `List.get`
    indexing. -/
inductive ChSet.subsetL : List ChSet → List ChSet → Prop where
  | nil  : ChSet.subsetL [] []
  | cons {S T : ChSet} {Ss Ts : List ChSet} :
      S.subset T → ChSet.subsetL Ss Ts →
      ChSet.subsetL (S :: Ss) (T :: Ts)

theorem ChSet.subsetL_refl : ∀ (Ss : List ChSet), ChSet.subsetL Ss Ss
  | []       => ChSet.subsetL.nil
  | S :: Ss  => ChSet.subsetL.cons (ChSet.subset_refl S)
                                    (ChSet.subsetL_refl Ss)

/-- Every list element is a subset of the list-union.  N-ary analogue
    of `ChSet.subset_union_left` / `ChSet.subset_union_right`. -/
theorem ChSet.subset_unionL : ∀ (Ss : List ChSet) (S : ChSet),
    S ∈ Ss → S.subset (ChSet.unionL Ss)
  | [], _, h => absurd h (by simp)
  | T :: Ss, S, h => by
      rw [ChSet.unionL_cons]
      rcases List.mem_cons.mp h with rfl | hmem
      · exact ChSet.subset_union_left S (ChSet.unionL Ss)
      · exact ChSet.subset_trans S (ChSet.unionL Ss) _
          (ChSet.subset_unionL Ss S hmem)
          (ChSet.subset_union_right T (ChSet.unionL Ss))

/-- **Monotonicity of list-union** under pointwise subset.  N-ary
    analogue of `ChSet.union_mono`. -/
theorem ChSet.unionL_mono : ∀ {Ss Ts : List ChSet},
    ChSet.subsetL Ss Ts →
    (ChSet.unionL Ss).subset (ChSet.unionL Ts)
  | [],      [],      _ => ChSet.subset_refl ChSet.empty
  | S :: Ss, T :: Ts, .cons hhd htail => by
      rw [ChSet.unionL_cons, ChSet.unionL_cons]
      exact ChSet.union_mono S (ChSet.unionL Ss) T (ChSet.unionL Ts)
        hhd (ChSet.unionL_mono htail)

/-- N-ary analogue of `UsedParams2`: a `Bool` per parameter position.
    Corresponds to the low bits of the compiler's `FN_USED_PARAMS[fi]`
    bitmask, read one bit per `i` at `compile_primary:~10317`. -/
abbrev UsedParamsL : Type := List Bool

/-- N-ary body-precision join: zip a per-parameter boolean list with a
    list of channel sets, and union only those sets whose boolean is
    `true`.  Mirrors the compiler's loop `if (FN_USED_PARAMS[fi] & (1 <<
    i)) != 0 then union B9_ARG_CH_SET[i] else skip`.  When either list
    is empty the result is `empty`; length mismatches are treated as
    truncation.  N-ary analogue of `usedUnion`. -/
def usedUnionL : UsedParamsL → List ChSet → ChSet
  | [],      _        => ChSet.empty
  | _,       []       => ChSet.empty
  | u :: U, S :: Ss =>
      ChSet.union
        (if u then S else ChSet.empty)
        (usedUnionL U Ss)

theorem usedUnionL_nil_left (Ss : List ChSet) :
    usedUnionL [] Ss = ChSet.empty := rfl

theorem usedUnionL_nil_right : ∀ (U : UsedParamsL),
    usedUnionL U [] = ChSet.empty
  | []      => rfl
  | _ :: _  => rfl

theorem usedUnionL_cons (u : Bool) (S : ChSet)
    (U : UsedParamsL) (Ss : List ChSet) :
    usedUnionL (u :: U) (S :: Ss) =
      ChSet.union (if u then S else ChSet.empty) (usedUnionL U Ss) := rfl

/-- **N-ary body-precision is always a refinement of the n-ary
    union.**  The compiler's fallback — an unanalyzed callee collapses
    to the conservative list-union — is always safe because the
    body-precision set is contained in the conservative one.  N-ary
    analogue of `usedUnion_subset_union`. -/
theorem usedUnionL_subset_unionL :
    ∀ (U : UsedParamsL) (Ss : List ChSet),
      (usedUnionL U Ss).subset (ChSet.unionL Ss)
  | [],      Ss      => by
      rw [usedUnionL_nil_left]; exact ChSet.empty_subset _
  | _ :: _, []       => by
      rw [usedUnionL_nil_right]; exact ChSet.empty_subset _
  | u :: U, S :: Ss  => by
      rw [usedUnionL_cons, ChSet.unionL_cons]
      exact ChSet.union_subset _ _ _
        (ChSet.subset_trans _ S _
          (ChSet.if_empty_subset_left u S)
          (ChSet.subset_union_left S (ChSet.unionL Ss)))
        (ChSet.subset_trans _ (ChSet.unionL Ss) _
          (usedUnionL_subset_unionL U Ss)
          (ChSet.subset_union_right S (ChSet.unionL Ss)))

/-- Monotonicity of `usedUnionL` in the channel-set list (for a fixed
    per-parameter bit list).  N-ary analogue of `usedUnion_mono`. -/
theorem usedUnionL_mono :
    ∀ (U : UsedParamsL) {Ss Ts : List ChSet},
      ChSet.subsetL Ss Ts →
      (usedUnionL U Ss).subset (usedUnionL U Ts)
  | [],      _,        _,        _ => ChSet.subset_refl _
  | _ :: _, [],        [],       _ => ChSet.subset_refl _
  | u :: U, S :: Ss,   T :: Ts,  .cons hhd htail => by
      rw [usedUnionL_cons, usedUnionL_cons]
      exact ChSet.union_mono _ _ _ _
        (ChSet.if_empty_mono u S T hhd)
        (usedUnionL_mono U htail)

/-- Sanity: when every parameter is used, `usedUnionL` collapses to
    the conservative n-ary list-union.  N-ary analogue of
    `usedUnion_all_used`; no regression for fully-used functions. -/
theorem usedUnionL_all_used :
    ∀ (Ss : List ChSet),
      usedUnionL (List.replicate Ss.length true) Ss = ChSet.unionL Ss
  | []      => rfl
  | S :: Ss => by
      show usedUnionL (true :: List.replicate Ss.length true) (S :: Ss)
             = ChSet.unionL (S :: Ss)
      rw [usedUnionL_cons, ChSet.unionL_cons, usedUnionL_all_used Ss]
      show ChSet.union (if true then S else ChSet.empty) (ChSet.unionL Ss)
             = S.union (ChSet.unionL Ss)
      rfl

/-- **Abstract n-ary body-precision soundness.**

    Given
      * `args_fps` — per-argument runtime footprints
      * `S_args`   — per-argument declared channel sets
      * `U`        — body-precision summary (one bit per parameter)
      * `ret_fp`   — return-value runtime footprint
      * `hArgs`    — each argument's fp ⊆ its declared set (pointwise)
      * `hBody`    — return fp ⊆ `usedUnionL U args_fps`
                     (the n-ary body-precision hypothesis, discharged
                      compiler-side at `FN_USED_PARAMS[fi]` capture)

    conclude: return fp ⊆ `usedUnionL U S_args` — the *declared* n-ary
    body-precision set.

    N-ary analogue of `gtt_sound_body`, discharging step 3
    (monotonicity) of that theorem's chain — steps 1 and 2
    (per-argument `gtt_sound` recursion and per-call body-precision)
    become the explicit `hArgs` / `hBody` hypotheses here, awaiting a
    future `Expr.callN (args : List Expr)` inductive to discharge them
    by a `gtt_sound`-style structural recursion.  The statement is
    set-theoretic, so it is independent of the specific
    `Expr` / `Eval` / `FnTable` formulation and applies to any n-ary
    call semantics that itself discharges `hBody`. -/
theorem gtt_sound_bodyN
    (U : UsedParamsL)
    {args_fps S_args : List ChSet}
    {ret_fp : ChSet}
    (hArgs : ChSet.subsetL args_fps S_args)
    (hBody : ret_fp.subset (usedUnionL U args_fps)) :
    ret_fp.subset (usedUnionL U S_args) :=
  ChSet.subset_trans ret_fp (usedUnionL U args_fps) (usedUnionL U S_args)
    hBody
    (usedUnionL_mono U hArgs)

/-- **N-ary fallback soundness.**  When a callee's body-precision is
    unanalyzed (compiler fallback: union-all), the return fp is bounded
    by the n-ary list-union of declared sets.  Composes `ChSet.subset_trans`
    + `ChSet.unionL_mono`.  N-ary analogue of `gtt_sound` at the
    ChSet level. -/
theorem gtt_sound_fallbackN
    {args_fps S_args : List ChSet}
    {ret_fp : ChSet}
    (hArgs : ChSet.subsetL args_fps S_args)
    (hBody : ret_fp.subset (ChSet.unionL args_fps)) :
    ret_fp.subset (ChSet.unionL S_args) :=
  ChSet.subset_trans ret_fp (ChSet.unionL args_fps) (ChSet.unionL S_args)
    hBody
    (ChSet.unionL_mono hArgs)

/-- **Canary (week-5 n-ary body-precision).**  Three-argument call with
    `UP = [true, false, false]` (first parameter used, others dropped):
    the `usedUnionL`-computed set reduces definitionally to a union of
    `S1` with empty tails.  Mirrors `canary_proj_a_only_arg0` lifted to
    arity 3; witnesses that `usedUnionL` correctly drops
    unused-parameter channel sets. -/
theorem canary_nary_proj_a_drops_args (S1 S2 S3 : ChSet) :
    usedUnionL [true, false, false] [S1, S2, S3]
      = ChSet.union S1 (ChSet.union ChSet.empty
          (ChSet.union ChSet.empty ChSet.empty)) := rfl

/-- **Canary — pointwise collapse.**  Every channel of the n-ary
    projection result equals the corresponding channel of `S1` — the
    unused-parameter channel sets `S2` and `S3` contribute nothing. -/
theorem canary_nary_proj_a_pointwise (S1 S2 S3 : ChSet) (n : Nat) :
    (usedUnionL [true, false, false] [S1, S2, S3]) n = S1 n := by
  rw [canary_nary_proj_a_drops_args]
  show (S1 n || (false || (false || false))) = S1 n
  cases S1 n <;> rfl

-- ---------------------------------------------------------------------------
-- §7d. N-ary body-level precision (Fin-indexed surface machinery)
-- ---------------------------------------------------------------------------

/-!
Week-5 surface extension.  The §2b `chsetUnionFin` combinator pairs
naturally with a Fin-indexed per-parameter usage summary; together
these give a direct analogue of §7b's binary body-precision
(`UsedParams2` / `usedUnion` / `gtt_sound_body`) for `Expr.callN`.

The "per-call" arity `n` is existentially quantified at the `tCallN`
introduction and propagates through the `BodyRespectsUsedParamsFin`
hypothesis as an explicit universal.  A single `FnId` can be called at
different arities across the program (enforced in reality by
`FN_ARITY` in the compiler); the Lean model keeps the arity abstract,
and the body-precision summary is keyed by *both* `FnId` and `n`.
-/

/-- Per-parameter body-precision summary for an `n`-ary callee: a
    `Fin n`-indexed family of booleans, one bit per parameter position.
    Fin analogue of `UsedParams2`.  Corresponds to bits 0..(n-1) of
    `FN_USED_PARAMS[fi]` for an `n`-ary callee, read at the call site
    (`compile_primary:~10318`). -/
abbrev UsedParamsFin (n : Nat) : Type := Fin n → Bool

/-- N-ary body-precision join: union only those `Ss i` sets whose
    `U i` bit is `true`.  Fin analogue of `usedUnion` / `usedUnionL`. -/
def usedUnionFin : {n : Nat} → UsedParamsFin n → (Fin n → ChSet) → ChSet
  | 0,   _, _  => ChSet.empty
  | n+1, U, Ss =>
      ChSet.union
        (if U 0 then Ss 0 else ChSet.empty)
        (usedUnionFin (fun i : Fin n => U i.succ)
                      (fun i : Fin n => Ss i.succ))

theorem usedUnionFin_zero (U : UsedParamsFin 0) (Ss : Fin 0 → ChSet) :
    usedUnionFin U Ss = ChSet.empty := rfl

theorem usedUnionFin_succ {n : Nat} (U : UsedParamsFin (n+1))
    (Ss : Fin (n+1) → ChSet) :
    usedUnionFin U Ss =
      ChSet.union
        (if U 0 then Ss 0 else ChSet.empty)
        (usedUnionFin (fun i : Fin n => U i.succ)
                      (fun i : Fin n => Ss i.succ)) := rfl

/-- **N-ary body-precision is always a refinement of `chsetUnionFin`.**
    Fin analogue of `usedUnion_subset_union`. -/
theorem usedUnionFin_subset_chsetUnionFin :
    {n : Nat} → (U : UsedParamsFin n) → (Ss : Fin n → ChSet)
    → (usedUnionFin U Ss).subset (chsetUnionFin Ss)
  | 0,   _, _  => by
      rw [usedUnionFin_zero, chsetUnionFin_zero]
      exact ChSet.subset_refl _
  | n+1, U, Ss => by
      rw [usedUnionFin_succ, chsetUnionFin_succ]
      exact ChSet.union_mono _ _ _ _
        (ChSet.if_empty_subset_left (U 0) (Ss 0))
        (usedUnionFin_subset_chsetUnionFin _ _)

/-- Monotonicity of `usedUnionFin` in the channel-set argument (for a
    fixed `U`).  Fin analogue of `usedUnion_mono` / `usedUnionL_mono`. -/
theorem usedUnionFin_mono :
    {n : Nat} → (U : UsedParamsFin n) → (Ss Ts : Fin n → ChSet)
    → (∀ i : Fin n, (Ss i).subset (Ts i))
    → (usedUnionFin U Ss).subset (usedUnionFin U Ts)
  | 0,   _, _,  _,  _ => by
      rw [usedUnionFin_zero, usedUnionFin_zero]
      exact ChSet.subset_refl _
  | n+1, U, Ss, Ts, h => by
      rw [usedUnionFin_succ, usedUnionFin_succ]
      exact ChSet.union_mono _ _ _ _
        (ChSet.if_empty_mono (U 0) (Ss 0) (Ts 0) (h 0))
        (usedUnionFin_mono _ _ _ (fun i : Fin n => h i.succ))

/-- Sanity: when every parameter is used, `usedUnionFin` collapses to
    `chsetUnionFin`.  Fin analogue of `usedUnion_all_used`. -/
theorem usedUnionFin_all_used :
    {n : Nat} → (Ss : Fin n → ChSet)
    → usedUnionFin (fun _ => true) Ss = chsetUnionFin Ss
  | 0,   _  => rfl
  | n+1, Ss => by
      rw [usedUnionFin_succ, chsetUnionFin_succ,
          usedUnionFin_all_used (fun i : Fin n => Ss i.succ)]
      show ChSet.union (if true then Ss 0 else ChSet.empty) _
             = (Ss 0).union _
      rfl

/-- **The n-ary body-precision hypothesis.**  For every `n`-ary callee
    `f` at arity `n` and every argument tuple `vs`, the return's
    footprint is contained in the `usedUnionFin`-indexed union of the
    argument footprints.  Fin analogue of `BodyRespectsUsedParams`.

    The summary `UP : (f : FnId) → (n : Nat) → UsedParamsFin n` is
    keyed by both `FnId` and arity.  This matches the compiler reality:
    `FN_USED_PARAMS[fi]` is a 16-bit mask and the `n` used bits depend
    on `FN_ARITY[fi]`. -/
def BodyRespectsUsedParamsFin (F : FnTable)
    (UP : (f : FnId) → (n : Nat) → UsedParamsFin n) : Prop :=
  ∀ (f : FnId) (n : Nat) (vs : Fin n → RuntimeVal) (ret : RuntimeVal),
    F.evalN f n vs ret →
    ret.footprint.subset (usedUnionFin (UP f n) (fun i => (vs i).footprint))

/-- Fin-body-precision implies the n-ary union-rule hypothesis.  Fin
    analogue of `bodyPrecision_implies_union_hypothesis`. -/
theorem bodyPrecisionFin_implies_union_hypothesis
    (F : FnTable)
    (UP : (f : FnId) → (n : Nat) → UsedParamsFin n)
    (h : BodyRespectsUsedParamsFin F UP) : BodyRespectsTopologyN F := by
  intro f n vs ret hfn
  exact ChSet.subset_trans ret.footprint
    (usedUnionFin (UP f n) (fun i => (vs i).footprint))
    (chsetUnionFin (fun i => (vs i).footprint))
    (h f n vs ret hfn)
    (usedUnionFin_subset_chsetUnionFin (UP f n) (fun i => (vs i).footprint))

/-- **Body-precision soundness for direct n-ary user-fn calls.**

    Given the strengthened hypothesis `BodyRespectsUsedParamsFin`, the
    declared channel set `usedUnionFin (UP f n) Ss` is a sound upper
    bound on the runtime footprint of `callN f args`.  N-ary analogue
    of `gtt_sound_body`.

    Like `gtt_sound_body`, the proof composes `gtt_sound` on each
    sub-argument (for argument-footprint containment) with the
    body-precision hypothesis, then monotonicity of `usedUnionFin`. -/
theorem gtt_sound_body_callN
    (F : FnTable)
    (UP2 : FnId → UsedParams2)
    (UP : (f : FnId) → (n : Nat) → UsedParamsFin n)
    (Γ : TyEnv) (ρ : VarEnv)
    (hΓρ : VarEnvConsistent Γ ρ)
    (hbody2 : BodyRespectsUsedParams F UP2)
    (hbodyN : BodyRespectsUsedParamsFin F UP) :
    ∀ (f : FnId) {n : Nat} (args : Fin n → Expr) (Ss : Fin n → ChSet)
      (vs : Fin n → RuntimeVal) (ret : RuntimeVal),
      (∀ i, Typing Γ (args i) (Ss i)) →
      (∀ i, Eval F ρ (args i) (vs i)) → F.evalN f n vs ret →
      ret.footprint.subset (usedUnionFin (UP f n) Ss) := by
  intro f n args Ss vs ret hTi hEi hfn
  -- Step 1: each argument i's runtime footprint is contained in its declared set.
  have hbody_union : BodyRespectsTopology F :=
    ⟨bodyPrecision_implies_union_hypothesis F UP2 hbody2,
     bodyPrecisionFin_implies_union_hypothesis F UP hbodyN⟩
  have si : ∀ i, (vs i).footprint.subset (Ss i) :=
    fun i => gtt_sound F Γ ρ hΓρ hbody_union (args i) (Ss i) (vs i)
             (hTi i) (hEi i)
  -- Step 2: ret footprint is contained in `usedUnionFin (UP f n) (fun i => (vs i).fp)`.
  have ret_sub_used :
      ret.footprint.subset
        (usedUnionFin (UP f n) (fun i => (vs i).footprint)) :=
    hbodyN f n vs ret hfn
  -- Step 3: monotonicity of usedUnionFin under pointwise subset.
  exact ChSet.subset_trans ret.footprint
    (usedUnionFin (UP f n) (fun i => (vs i).footprint))
    (usedUnionFin (UP f n) Ss)
    ret_sub_used
    (usedUnionFin_mono _ _ _ si)

/-- **Canary — `usedUnionFin` at arity 3 drops unused-parameter sets.**
    The Fin analogue of `canary_proj_a_only_arg0`.  Three booleans
    (`b0, b1, b2`) controlling whether each of three channel sets
    contributes.  When only bit 0 is set, the result is `S0 ∪ empty ∪
    (empty ∪ empty) = S0 ∪ _`.  Witnesses that `usedUnionFin` correctly
    gates each slot by its boolean. -/
theorem canary_callN_proj_a_fin (S0 S1 S2 : ChSet) :
    usedUnionFin
        (fun i : Fin 3 => match i with
                          | ⟨0, _⟩ => true
                          | _      => false)
        (fun i : Fin 3 => match i with
                          | ⟨0, _⟩ => S0
                          | ⟨1, _⟩ => S1
                          | ⟨2, _⟩ => S2)
      = ChSet.union S0 (ChSet.union ChSet.empty
            (ChSet.union ChSet.empty ChSet.empty)) := rfl

/-- **Canary — pointwise collapse at Fin surface.**  The `usedUnionFin`
    canary evaluates to `S0 n` pointwise; unused-parameter sets `S1`,
    `S2` contribute nothing.  Fin analogue of `canary_nary_proj_a_pointwise`. -/
theorem canary_callN_proj_a_fin_pointwise (S0 S1 S2 : ChSet) (n : Nat) :
    (usedUnionFin
        (fun i : Fin 3 => match i with
                          | ⟨0, _⟩ => true
                          | _      => false)
        (fun i : Fin 3 => match i with
                          | ⟨0, _⟩ => S0
                          | ⟨1, _⟩ => S1
                          | ⟨2, _⟩ => S2)) n
      = S0 n := by
  rw [canary_callN_proj_a_fin]
  show (S0 n || (false || (false || false))) = S0 n
  cases S0 n <;> rfl

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
| **Fin-indexed channel-set union / monotonicity / membership**         | Proved  | `chsetUnionFin_*` / `subset_chsetUnionFin` |
| **`gtt_sound` at `Expr.callN` (week-5 surface)**                      | Proved  | `gtt_sound` (`callN` case)               |
| **`gtt_sound` at `Expr.ifE` (week-6 surface, both-arms-union)**       | Proved  | `gtt_sound` (`ifE` case, 2 sub-cases)    |
| **Body-precision soundness for direct n-ary `callN`** (week-5)        | Proved  | `gtt_sound_body_callN`                   |
| Fin body-precision refines n-ary union / monotone / all-used          | Proved  | `usedUnionFin_subset_chsetUnionFin` etc. |
| Canary: `usedUnionFin` at arity 3 collapses to `S0`                   | Proved  | `canary_callN_proj_a_fin*`               |
| **N-ary list-union / pointwise-subset / monotonicity** (ChSet level)  | Proved  | `ChSet.unionL_*` / `ChSet.subsetL_*`     |
| **N-ary body-precision refines n-ary union** (`usedUnionL`)           | Proved  | `usedUnionL_subset_unionL`               |
| N-ary body-precision collapses to union when all params used          | Proved  | `usedUnionL_all_used`                    |
| N-ary body-precision is monotone in argument channel sets             | Proved  | `usedUnionL_mono`                        |
| **Abstract n-ary body-precision soundness** (week-5 ChSet form)       | Proved  | `gtt_sound_bodyN`                        |
| N-ary union-rule fallback soundness (week-3 generalised)              | Proved  | `gtt_sound_fallbackN`                    |
| Canary: 3-ary proj_a with UP=[true,false,false] ⇒ ret ⊆ `S1`          | Proved  | `canary_nary_proj_a_drops_args`          |

## What this file proves, in one sentence

For every GTT-typed expression `e` with declared channel set `S`, every
runtime value produced by evaluating `e` under a consistent variable
environment and a body-respecting function table has channel footprint
contained in `S`.  In particular, the week-3 inter-procedural union rule
at binary user-function calls is a *sound* static channel set, and the
week-5 abstract n-ary body-precision soundness theorem
(`gtt_sound_bodyN`) lifts the week-4 binary soundness to arbitrary
arity at the set-theoretic level.

## What remains open

1. **`Expr.ifE` path-sensitive tightening + compiler-side arm-union
   emission.**  Week-6 Stage 1 (current) adds `Expr.ifE` + `Typing.tIf`
   at the both-arms-union form (`Sc ∪ (S1 ∪ S2)`).  Stage 2 would
   tighten to a path-sensitive form tracking which arm ran at runtime,
   AND fix the compiler soundness hole: `compile_primary` (line ~15889)
   currently compiles both arms but doesn't propagate EXPR_CH_SET
   through them, leaking the last-compiled arm's topology — which this
   section's theorem *would* correctly identify as sound if fixed.
   Body-level precision for match arms, loops, recursion fixed-point
   remain deferred beyond that.
2. Closure / fn-ref indirect calls, `Expr.call2` and `Expr.callN` here
   are both direct-only.  The compiler week-3 scope leaves
   closure-indirect calls leaking the last-arg topology (§7c docstring).
3. ARM64 mirror — codegen parity, not a separate semantic theorem.
   The a64 compiler mirrors `EXPR_PARAM_USED` propagation through
   let / load in week-4 step A; the a64 call-site tightening itself
   remains deferred.
4. Bridging to `HessianAD.lean` — proving that `j ∉ S ⟹ H_{j,k} = 0.0`
   via the shadow-slot semantics.  The natural next proof, spans two
   files.  Under `gtt_sound_body` this bridge becomes an *equality*
   (declared set characterises shadow footprint) on the body-precision
   fragment, rather than just a containment.
5. **Compiler-side n-ary surface wiring.**  `Expr.callN` is now in the
   Lean model and `gtt_sound_body_callN` proves its soundness, but the
   compiler's `compile_primary` (line ~10205) currently emits only the
   binary `call2`-shaped path; extending emission to a true
   variadic call site with per-arg `B9_ARG_CH_SET[16]` iteration is
   a week-6+ compiler task (not a Lean proof task).

## No new axioms

All theorems are proved by structural induction on `Typing` / `Expr`
or by bool-exhaustion over `Nat → Bool`.  No Mathlib, no Float axioms.
-/

end Sounio.GradientTopology
