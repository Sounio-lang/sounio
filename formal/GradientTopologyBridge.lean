import GradientTopology
import HessianAD

/-!
# Sounio.GradientTopologyBridge — Full inductive GTT-HessianAD bridge theorem

Proves: for every expression `e` in the body-precision fragment, if the GTT
typing `⊢ e : S` holds and `D = eval(e)`, then

  `j ∉ S ⟹ D.grad j = 0`        (GradFootprint)
  `j ∉ S ⟹ ∀ k, D.hess j k = 0` (HessFootprint)

This connects two companion files:

  - `GradientTopology.lean` (GTT typing, `usedUnion`, body-precision §7b)
  - `HessianAD.lean`        (Dual2 arithmetic, footprint preservation §9b)

Both companions are imported; `ChSet`, `UsedParams2`, `usedUnion`,
`Dual2`, `dual2Add/Mul/...`, and `GradFootprint`/`HessFootprint` come
from those files.  This file defines only the combined `Footprint` (grad
∧ hess), the binary-expression fragment `BExpr`/`BTyping`/`evalB`, the
inductive bridge theorem, and the Phase-1 canary tightness witness.

Zero sorry.  No Mathlib.  No new axioms beyond `Dual2Field`.
-/

namespace Sounio.GradientTopologyBridge

open Sounio.GradientTopology
open Sounio.HessianAD

-- ===========================================================================
-- §1-§3.  Channel sets, body-precision summary, Dual2 arithmetic.
-- All moved to their companion files; imported above.  See:
--   - `GradientTopology.lean`: `ChSet`, `UsedParams2`, `usedUnion`, `usedUnion_subset_union`
--   - `HessianAD.lean`:        `Dual2Field`, `Dual2`, `dual2Const/Seed/Add/Mul/ApplyUnary`
-- ===========================================================================

variable {α : Type} [F : Dual2Field α]

local notation "𝟎" => @Dual2Field.zero α F
local notation "𝟏" => @Dual2Field.one α F

-- ===========================================================================
-- §4. Combined footprint predicate  (grad ∧ hess)
-- ===========================================================================

-- `GradFootprint` / `HessFootprint` are imported from `HessianAD` (§9b).
-- This file adds the conjunction so §5-§11 can state lemmas once instead
-- of pair-wise.

def Footprint (S : ChSet) (D : Dual2 α) : Prop :=
  GradFootprint S D ∧ HessFootprint S D

-- Helper: channel j absent from a union means absent from both components.
private theorem not_union_iff (Sa Sb : ChSet) (j : Fin 8)
    (h : ¬ (Sa.union Sb) j.val) : ¬ Sa j.val ∧ ¬ Sb j.val := by
  simp only [ChSet.union] at h
  constructor
  · intro ha; exact h (by simp [ha])
  · intro hb; exact h (by simp [hb])

-- Helper: channel j absent from usedUnion implies absent from whichever
-- component the union includes.
private theorem not_usedUnion_left (U : UsedParams2) (Sa Sb : ChSet) (j : Fin 8)
    (h : ¬ (usedUnion U Sa Sb) j.val) (hu : U.uses0 = true) : ¬ Sa j.val := by
  intro ha
  apply h
  simp only [usedUnion, ChSet.union, hu, if_true]
  simp [ha]

private theorem not_usedUnion_right (U : UsedParams2) (Sa Sb : ChSet) (j : Fin 8)
    (h : ¬ (usedUnion U Sa Sb) j.val) (hu : U.uses1 = true) : ¬ Sb j.val := by
  intro hb
  apply h
  simp only [usedUnion, ChSet.union, hu, if_true]
  cases hval : U.uses0 with
  | true  => simp [hb]
  | false => simp [hb]

-- ===========================================================================
-- §5. Footprint preservation — same-set versions  (mirrors HessianAD §9b)
-- ===========================================================================

-- These handle the `unary` case where input and output have the same set.

theorem fp_const (S : ChSet) (v : α) : Footprint S (dual2Const v) :=
  ⟨fun _ _ => rfl, fun _ _ _ => rfl⟩

theorem fp_seed (c : Fin 8) (v : α) :
    Footprint (ChSet.singleton c.val) (dual2Seed c v) := by
  constructor
  · intro j hj
    simp only [dual2Seed]
    have hne : j ≠ c := by
      intro h; apply hj; simp only [ChSet.singleton, h, decide_true]
    rw [if_neg hne]
  · intro _ _ _; rfl

theorem fp_same_add (S : ChSet) (A B : Dual2 α)
    (hA : Footprint S A) (hB : Footprint S B) :
    Footprint S (dual2Add A B) := by
  exact ⟨fun j hj => by simp only [dual2Add, hA.1 j hj, hB.1 j hj, F.zero_add],
         fun j k hj => by simp only [dual2Add, hA.2 j k hj, hB.2 j k hj, F.zero_add]⟩

theorem fp_apply_unary (S : ChSet) (fv fp fpp : α) (G : Dual2 α)
    (hG : Footprint S G) :
    Footprint S (dual2ApplyUnary fv fp fpp G) :=
  ⟨fun j hj => by simp only [dual2ApplyUnary, hG.1 j hj, F.mul_zero],
   fun j k hj => by simp only [dual2ApplyUnary, hG.1 j hj, hG.2 j k hj,
                                F.mul_zero, F.zero_mul, F.add_zero]⟩

-- ===========================================================================
-- §6. Footprint preservation — union versions  (for add / mul typing rules)
-- ===========================================================================

-- The typing rules for add and mul give the result type `Sa.union Sb`
-- when the two operands have types `Sa` and `Sb` respectively.  Proving
-- footprint requires extracting absence from both components.

theorem fp_add_union (Sa Sb : ChSet) (A B : Dual2 α)
    (hA : Footprint Sa A) (hB : Footprint Sb B) :
    Footprint (Sa.union Sb) (dual2Add A B) := by
  constructor
  · intro j hj
    obtain ⟨hsa, hsb⟩ := not_union_iff Sa Sb j hj
    simp only [dual2Add, hA.1 j hsa, hB.1 j hsb, F.zero_add]
  · intro j k hj
    obtain ⟨hsa, hsb⟩ := not_union_iff Sa Sb j hj
    simp only [dual2Add, hA.2 j k hsa, hB.2 j k hsb, F.zero_add]

theorem fp_mul_union (Sa Sb : ChSet) (A B : Dual2 α)
    (hA : Footprint Sa A) (hB : Footprint Sb B) :
    Footprint (Sa.union Sb) (dual2Mul A B) := by
  constructor
  · intro j hj
    obtain ⟨hsa, hsb⟩ := not_union_iff Sa Sb j hj
    simp only [dual2Mul, hA.1 j hsa, hB.1 j hsb, F.mul_zero, F.zero_mul, F.add_zero]
  · intro j k hj
    obtain ⟨hsa, hsb⟩ := not_union_iff Sa Sb j hj
    simp only [dual2Mul, hA.2 j k hsa, hB.2 j k hsb, hA.1 j hsa, hB.1 j hsb,
               F.zero_mul, F.mul_zero, F.zero_add, F.add_zero]

-- ===========================================================================
-- §7. Bridge expression language
-- ===========================================================================

/-- The body-precision expression fragment.  Binary user-fn calls carry
    (a) their `UsedParams2` body-precision summary and (b) the callee as
    an abstract `Dual2 → Dual2 → Dual2` function, so the evaluator is
    concrete without needing to know callee source code. -/
inductive BExpr (α : Type) [Dual2Field α] : Type where
  | const  : α → BExpr α
  | var    : Fin 8 → BExpr α
  | add    : BExpr α → BExpr α → BExpr α
  | mul    : BExpr α → BExpr α → BExpr α
  | unary  : α → α → BExpr α → BExpr α            -- fp, fpp, arg
  | call2  : UsedParams2
           → (Dual2 α → Dual2 α → Dual2 α)         -- callee body (abstract)
           → BExpr α → BExpr α → BExpr α

/-- Environment: maps each variable `k : Fin 8` to its `Dual2` denotation
    and its declared GTT channel set. -/
structure BEnv (α : Type) [Dual2Field α] where
  denotation : Fin 8 → Dual2 α
  chset      : Fin 8 → ChSet

/-- Environment well-typedness: each variable's denotation has footprint
    matching its declared channel set. -/
def EnvWT (env : BEnv α) : Prop :=
  ∀ k : Fin 8, Footprint (env.chset k) (env.denotation k)

/-- A callee body `f` satisfies body-precision for usage summary `U` if,
    whenever the two inputs have footprint `Sa` and `Sb`, the result has
    footprint `usedUnion U Sa Sb`.  This is the formal counterpart of
    `gtt_sound_body` discharging `BodyRespectsUsedParams` for direct
    analysed callees. -/
def CalleeRespects (U : UsedParams2) (f : Dual2 α → Dual2 α → Dual2 α) : Prop :=
  ∀ (Sa Sb : ChSet) (Da Db : Dual2 α),
    Footprint Sa Da → Footprint Sb Db →
    Footprint (usedUnion U Sa Sb) (f Da Db)

-- ===========================================================================
-- §8. Evaluator
-- ===========================================================================

def evalB (env : BEnv α) : BExpr α → Dual2 α
  | .const v         => dual2Const v
  | .var k           => env.denotation k
  | .add a b         => dual2Add  (evalB env a) (evalB env b)
  | .mul a b         => dual2Mul  (evalB env a) (evalB env b)
  | .unary fp fpp g  => dual2ApplyUnary (evalB env g).val fp fpp (evalB env g)
  | .call2 _ f a b   => f (evalB env a) (evalB env b)

-- ===========================================================================
-- §9. GTT typing relation for BExpr
-- ===========================================================================

/-- GTT typing derivation.  The `var` rule uses `env.chset k` — the
    environment declares the channel set of each variable, so `EnvWT env`
    ensures the denotation matches.  The `call2` rule embeds `CalleeRespects`
    directly so the bridge theorem needs no extra global hypothesis. -/
inductive BTyping (env : BEnv α) : BExpr α → ChSet → Prop where
  | const  : ∀ v,
      BTyping env (.const v) ChSet.empty
  | var    : ∀ k,
      BTyping env (.var k) (env.chset k)
  | add    : ∀ a b Sa Sb,
      BTyping env a Sa → BTyping env b Sb →
      BTyping env (.add a b) (Sa.union Sb)
  | mul    : ∀ a b Sa Sb,
      BTyping env a Sa → BTyping env b Sb →
      BTyping env (.mul a b) (Sa.union Sb)
  | unary  : ∀ fp fpp g Sg,
      BTyping env g Sg →
      BTyping env (.unary fp fpp g) Sg
  | call2  : ∀ (U : UsedParams2) (f : Dual2 α → Dual2 α → Dual2 α) a b Sa Sb,
      CalleeRespects U f →
      BTyping env a Sa → BTyping env b Sb →
      BTyping env (.call2 U f a b) (usedUnion U Sa Sb)

-- ===========================================================================
-- §10. Bridge theorem
-- ===========================================================================

/-- **GTT-HessianAD Bridge Theorem.**
    For every expression `e` in the body-precision fragment, if the
    environment is well-typed (`EnvWT env`) and the GTT typing derivation
    `BTyping env e S` holds, then the evaluator denotation `evalB env e`
    has footprint `S`:
      * `j ∉ S ⟹ (evalB env e).grad j = 0`
      * `j ∉ S ⟹ ∀ k, (evalB env e).hess j k = 0`

    Proof: structural induction on the `BTyping` derivation.  Each case
    uses the corresponding preservation lemma from §5-§6. -/
theorem bridge (env : BEnv α) (hwf : EnvWT env) :
    ∀ (e : BExpr α) (S : ChSet),
    BTyping env e S →
    Footprint S (evalB env e) := by
  intro e S ht
  induction ht with
  | const v =>
      simp only [evalB]; exact fp_const _ _
  | var k =>
      simp only [evalB]; exact hwf k
  | add a b Sa Sb _ha _hb ihA ihB =>
      simp only [evalB]; exact fp_add_union Sa Sb _ _ ihA ihB
  | mul a b Sa Sb _ha _hb ihA ihB =>
      simp only [evalB]; exact fp_mul_union Sa Sb _ _ ihA ihB
  | unary fp fpp g Sg _hg ihG =>
      simp only [evalB]; exact fp_apply_unary Sg _ fp fpp _ ihG
  | call2 U f a b Sa Sb hcr _ha _hb ihA ihB =>
      simp only [evalB]; exact hcr Sa Sb _ _ ihA ihB

-- ===========================================================================
-- §11. Corollaries
-- ===========================================================================

/-- **Hessian sparsity corollary.**  Channel `j ∉ S` implies the full
    `j`-row of the output Hessian is zero.  This is the compiler-level claim:
    `EXPR_HSHADOW_jk = 0` for all `k` when `j ∉ GTT(e)`. -/
theorem hessian_sparsity (env : BEnv α) (hwf : EnvWT env)
    (e : BExpr α) (S : ChSet) (ht : BTyping env e S)
    (j k : Fin 8) (hj : ¬ S j.val) :
    (evalB env e).hess j k = 𝟎 :=
  (bridge env hwf e S ht).2 j k hj

/-- **Gradient sparsity corollary.**  Channel `j ∉ S` implies the first-order
    shadow slot `EXPR_SSHADOW_j = 0`. -/
theorem gradient_sparsity (env : BEnv α) (hwf : EnvWT env)
    (e : BExpr α) (S : ChSet) (ht : BTyping env e S)
    (j : Fin 8) (hj : ¬ S j.val) :
    (evalB env e).grad j = 𝟎 :=
  (bridge env hwf e S ht).1 j hj

/-- **Body-precision safety**: `usedUnion U Sa Sb ⊆ Sa.union Sb`.
    The body-precision channel set is contained in the week-3 union, so
    week-3 conservatism is preserved even after tightening. -/
theorem body_precision_safety (U : UsedParams2) (Sa Sb : ChSet) :
    (usedUnion U Sa Sb).subset (Sa.union Sb) :=
  usedUnion_subset_union U Sa Sb

/-- **Seed environment example.**  A seed environment assigns each variable `k`
    channel set `{k}` and denotation `dual2Seed k v_k`.  `fp_seed` witnesses
    that this environment is well-typed, providing a concrete base-case for
    the bridge theorem. -/
theorem seed_env_is_wt (vals : Fin 8 → α) :
    EnvWT (⟨fun k => dual2Seed k (vals k), fun k => ChSet.singleton k.val⟩ : BEnv α) :=
  fun k => fp_seed k (vals k)

-- ===========================================================================
-- §11b. Tightness witness — body-precision canary over `Nat`
-- ===========================================================================

/-!
The `bridge` theorem above establishes the *upper bound*: the declared GTT
channel set contains the nonzero-grad support.  Body-precision (week-4
`BTyping.call2` + `CalleeRespects`) narrows the declared set below the
week-3 union — but that narrowing is only meaningful if the tightened set
is *achievable*, i.e. the bound is not slack.

This section exhibits an exact witness for the Phase-1 canary
`proj_a(k1.value, k2.value)` under `UP = ⟨true, false⟩`:

  * The week-4 declared set is
      `usedUnion ⟨true,false⟩ (singleton 0) (singleton 1)`
    which reduces pointwise to `ChSet.singleton 0` (channel 1 is dropped
    because `proj_a` ignores its second argument).
  * The returned `Dual2 Nat` has `grad ⟨0, _⟩ = 1` (channel 0 active) and
    `grad ⟨1, _⟩ = 0` (channel 1 absent).

Combined with `bridge`, the declared set is witnessed *exactly* — not a
slack upper bound.  The same-shape claim about the static set is proved
as `canary_proj_a_only_arg0` in `GradientTopology.lean:561`.
-/

/-- Concrete `Dual2Field` instance on `Nat`.  Used as the ground algebra
    for the tightness witness, where `𝟎 = 0` and `𝟏 = 1` are decidably
    distinct. -/
instance instDual2FieldNat : Dual2Field Nat where
  zero := 0
  one  := 1
  zero_mul := Nat.zero_mul
  mul_zero := Nat.mul_zero
  zero_add := Nat.zero_add
  add_zero := Nat.add_zero
  add_comm := Nat.add_comm
  mul_comm := Nat.mul_comm
  mul_assoc := Nat.mul_assoc

/-- The `proj_a` callee as an abstract Dual2 operator: returns its first
    argument, discards the second.  Models `fn proj_a(a, b) = a` which is
    the week-4 Phase-1 compiler canary (body uses only parameter 0). -/
def projA_body (a _b : Dual2 Nat) : Dual2 Nat := a

/-- `proj_a` respects body-precision at `UP = ⟨true, false⟩`: the result's
    footprint lies in the first argument's footprint and is independent of
    the second.  Direct from `usedUnion ⟨true,false⟩ Sa Sb ≡ Sa` pointwise. -/
theorem projA_respects_body_precision :
    CalleeRespects ⟨true, false⟩ projA_body := by
  intro Sa _Sb Da _Db hA _hB
  refine ⟨?_, ?_⟩
  · intro j hj
    apply hA.1 j
    intro ha
    apply hj
    simp only [usedUnion, ChSet.union, if_true, if_false]
    simp [ha]
  · intro j k hj
    apply hA.2 j k
    intro ha
    apply hj
    simp only [usedUnion, ChSet.union, if_true, if_false]
    simp [ha]

/-- The canary expression: `proj_a(var 0, var 1)` under `UP = ⟨true, false⟩`. -/
def canaryE : BExpr Nat :=
  BExpr.call2 ⟨true, false⟩ projA_body
    (BExpr.var ⟨0, by decide⟩)
    (BExpr.var ⟨1, by decide⟩)

/-- Canary environment.  Slot 0: seed on channel 0 with value 0.
    Slot 1: seed on channel 1 with value 0.  Slots 2-7: constant 0
    (empty footprint).  Matches the static `canary_proj_a_only_arg0`
    set-up in `GradientTopology.lean:561`. -/
def canaryEnv : BEnv Nat where
  denotation k :=
    if k.val = 0 then dual2Seed ⟨0, by decide⟩ 0
    else if k.val = 1 then dual2Seed ⟨1, by decide⟩ 0
    else dual2Const 0
  chset k :=
    if k.val = 0 then ChSet.singleton 0
    else if k.val = 1 then ChSet.singleton 1
    else ChSet.empty

/-- The canary environment is well-typed: each slot's denotation has
    footprint matching its declared channel set.  Case analysis on
    `k.val = 0` / `k.val = 1`, discharged by `fp_seed` / `fp_const`. -/
theorem canaryEnv_wellTyped : EnvWT canaryEnv := by
  intro k
  by_cases h0 : k.val = 0
  · show Footprint (canaryEnv.chset k) (canaryEnv.denotation k)
    simp only [canaryEnv, h0, ↓reduceIte]
    -- fp_seed ⟨0, _⟩ 0 : Footprint (ChSet.singleton (⟨0,_⟩ : Fin 8).val)
    --                              (dual2Seed ⟨0,_⟩ 0)
    -- and (⟨0,_⟩ : Fin 8).val = 0 by rfl
    exact fp_seed ⟨0, by decide⟩ 0
  · by_cases h1 : k.val = 1
    · show Footprint (canaryEnv.chset k) (canaryEnv.denotation k)
      simp only [canaryEnv, h0, h1, ↓reduceIte]
      exact fp_seed ⟨1, by decide⟩ 0
    · show Footprint (canaryEnv.chset k) (canaryEnv.denotation k)
      simp only [canaryEnv, h0, h1, ↓reduceIte]
      exact fp_const ChSet.empty 0

/-- The canary expression types at `usedUnion ⟨true,false⟩ {0} {1}` under
    the canary environment.  Applies `BTyping.call2` with the body-precision
    `CalleeRespects` witness and `BTyping.var` on each argument. -/
theorem canaryE_types :
    BTyping canaryEnv canaryE
      (usedUnion ⟨true, false⟩
        (ChSet.singleton 0) (ChSet.singleton 1)) := by
  -- The call2 rule concludes `usedUnion U (canaryEnv.chset ⟨0,_⟩)
  -- (canaryEnv.chset ⟨1,_⟩)`.  Each of those chset reads reduces by
  -- computation to the corresponding singleton (the `if k.val = 0` /
  -- `if k.val = 1` branches fire).
  have harg0 : canaryEnv.chset ⟨0, by decide⟩ = ChSet.singleton 0 := by
    simp [canaryEnv]
  have harg1 : canaryEnv.chset ⟨1, by decide⟩ = ChSet.singleton 1 := by
    simp [canaryEnv]
  have h0 : BTyping canaryEnv (BExpr.var ⟨0, by decide⟩) (ChSet.singleton 0) :=
    harg0 ▸ BTyping.var ⟨0, by decide⟩
  have h1 : BTyping canaryEnv (BExpr.var ⟨1, by decide⟩) (ChSet.singleton 1) :=
    harg1 ▸ BTyping.var ⟨1, by decide⟩
  exact BTyping.call2 ⟨true, false⟩ projA_body _ _ _ _
    projA_respects_body_precision h0 h1

/-- **Tightness witness** for the body-precision canary.

    The canary `proj_a(var 0, var 1)` under `UP = ⟨true, false⟩` evaluates
    to `dual2Seed ⟨0, _⟩ 0` (the first argument is projected through).
    We read off two facts plus a nonzero discriminator:

      * `grad ⟨0, _⟩ = 1` — slot 0 is active, matching the declared set.
      * `grad ⟨1, _⟩ = 0` — slot 1 is absent, matching the body-precision
        narrowing from week-3's conservative union.
      * `1 ≠ 0` at `Nat` — the two values are decidably distinct.

    Together these three facts exhibit the declared set `{0}` *exactly*:
    the active slot is precisely slot 0. -/
theorem gtt_hessian_bridge_tight :
    (evalB canaryEnv canaryE).grad ⟨0, by decide⟩ = (1 : Nat) ∧
    (evalB canaryEnv canaryE).grad ⟨1, by decide⟩ = (0 : Nat) ∧
    ((1 : Nat) ≠ (0 : Nat)) := by
  refine ⟨?_, ?_, ?_⟩
  · -- evalB canary = projA_body (canaryEnv.denotation ⟨0,_⟩)
    --                           (canaryEnv.denotation ⟨1,_⟩)
    --              = canaryEnv.denotation ⟨0,_⟩ = dual2Seed ⟨0,_⟩ 0
    -- so grad ⟨0,_⟩ = (if ⟨0,_⟩ = ⟨0,_⟩ then 𝟏 else 𝟎) = 1
    show (projA_body (canaryEnv.denotation ⟨0, by decide⟩)
                     (canaryEnv.denotation ⟨1, by decide⟩)).grad
         ⟨0, by decide⟩ = 1
    simp only [projA_body, canaryEnv, ↓reduceIte, dual2Seed]
    rfl
  · show (projA_body (canaryEnv.denotation ⟨0, by decide⟩)
                     (canaryEnv.denotation ⟨1, by decide⟩)).grad
         ⟨1, by decide⟩ = 0
    simp only [projA_body, canaryEnv, ↓reduceIte, dual2Seed]
    -- (if (⟨1,_⟩ : Fin 8) = ⟨0,_⟩ then 𝟏 else 𝟎) = 0
    decide
  · decide

/-- **Bridge exactness for the canary.**  Combine the `bridge` upper bound
    (established above) with `gtt_hessian_bridge_tight`:

      * Upper bound ⟹ `grad ⟨1,_⟩ = 0` because channel 1 ∉ declared set.
      * Tightness   ⟹ `grad ⟨0,_⟩ = 1 ≠ 0` because channel 0 is realised.

    The declared set is thus the exact nonzero-grad support on this
    environment, proving the body-precision narrowing is not conservative. -/
theorem canary_bridge_exact :
    (evalB canaryEnv canaryE).grad ⟨0, by decide⟩ ≠ (0 : Nat) ∧
    (evalB canaryEnv canaryE).grad ⟨1, by decide⟩ = (0 : Nat) := by
  refine ⟨?_, ?_⟩
  · rw [gtt_hessian_bridge_tight.1]; exact gtt_hessian_bridge_tight.2.2
  · -- Read grad ⟨1,_⟩ = 0 off the bridge theorem.  Channel 1 is absent
    -- from `usedUnion ⟨true,false⟩ (singleton 0) (singleton 1)` because
    -- the second `if` branch degenerates to `empty`.
    have hfp :=
      (bridge canaryEnv canaryEnv_wellTyped canaryE _ canaryE_types).1
    exact hfp ⟨1, by decide⟩ (by decide)

-- ===========================================================================
-- §12. Cross-world correspondence (Stage 1: reification interface)
-- ===========================================================================

/-!
This section bridges the two typing worlds formalised across
`GradientTopology.lean` and this file:

  * GT world: `Expr` / `Typing Γ e S` / `Eval F ρ e v` with scalar
    `RuntimeVal { value : Float; footprint : ChSet }` and a relational
    `FnTable`.  Soundness theorem: `gtt_sound` / `gtt_sound_body`.
  * B  world: `BExpr α` / `BTyping env e S` / `evalB env e` with
    second-order `Dual2 α` and functional callees.  Soundness theorem:
    `bridge` (this file, §10).

The two worlds speak about the same declared channel set `S : ChSet` but
track different runtime values.  §12 gives the reification interface —
`ExprWellFormed`, `Compat`, `lower` — that connects them.  §13 proves
lowering preserves (tightens) typing; §14 composes that with `bridge` to
get the cross-world soundness theorem `gtt_corresponds`.

Design:

  * Abstract ground algebra `α` with `[Dual2Field α]` — the cross-world
    theorem is polymorphic, instantiable at Nat (decidable canary, §14b),
    Float (compiler runtime under the axiomatised HessianAD §11 instance,
    §14c), or any other Dual2Field instance.
  * `Compat α` bundles the two environments plus the compatibility
    witnesses linking the GT `TyEnv` / `FnTable` to the B `BEnv` /
    call-lifter.
  * `Expr.value k` on the GT side emits a channel-`k` singleton; the B
    side encodes channels as `Fin 8` slots.  The reification requires
    `k < 8` (compiler only emits k ∈ {0..7}, there are only 8 shadow
    channels).  `ExprWellFormed` captures this.
-/

/-- Well-formedness predicate for expressions.  The only non-trivial
    constraint: `Expr.value k` requires `k < 8`, because the B side
    encodes channels as `Fin 8` slots and the compiler only emits
    shadow channels in that range. -/
def ExprWellFormed : Expr → Prop
  | .lit _          => True
  | .var _          => True
  | .value k        => k < 8
  | .add e1 e2      => ExprWellFormed e1 ∧ ExprWellFormed e2
  | .mul e1 e2      => ExprWellFormed e1 ∧ ExprWellFormed e2
  | .call2 _ e1 e2  => ExprWellFormed e1 ∧ ExprWellFormed e2

/-- Cross-world compatibility bundle.

    Links the two typing environments (GT `TyEnv`, B `BEnv`) and provides
    the B-side functional callee together with the body-precision
    summary.  The compatibility hypotheses below discipline the maps so
    the lowering preserves channel-set typing. -/
structure Compat (α : Type) [Dual2Field α] where
  /-- GT side: variable-to-channel-set assignment. -/
  Γ         : TyEnv
  /-- B side: environment mapping each Fin 8 slot to its Dual2 denotation
      and declared channel set. -/
  env       : BEnv α
  /-- Lift Expr literals (`Float`) into the B ground algebra. -/
  liftLit   : Float → α
  /-- Each GT variable `x : VarId (= Nat)` maps to a B slot `Fin 8`. -/
  varSlot   : VarId → Fin 8
  /-- Each channel index (from `Expr.value k`) maps to a B slot.  Only
      constrained for `k < 8` via `chanComp`. -/
  chanSlot  : Nat → Fin 8
  /-- GT function id `f` → body-precision summary (mirrors `FN_USED_PARAMS`
      on the compiler side).  Matches the `UP` parameter of
      `gtt_sound_body`. -/
  UP        : FnId → UsedParams2
  /-- GT function id `f` → B-side functional callee.  The B side is
      functional (deterministic), so the user provides one callee per
      FnId; `cliftResp` below ensures each respects body-precision. -/
  clift     : FnId → (Dual2 α → Dual2 α → Dual2 α)
  /-- Variable compatibility: the B slot declared for `x` carries the
      channel set `Γ x`. -/
  varComp   : ∀ x : VarId, env.chset (varSlot x) = Γ x
  /-- Channel compatibility: for valid channels (`k < 8`), the B slot
      for channel `k` declares the singleton channel set `{k}`. -/
  chanComp  : ∀ k : Nat, k < 8 → env.chset (chanSlot k) = ChSet.singleton k
  /-- Callee-lifting respects body-precision: each `clift f` has B-side
      footprint footprint contained in `usedUnion (UP f) _ _`.  This is the
      B-side counterpart of `BodyRespectsUsedParams F UP` on the GT side. -/
  cliftResp : ∀ f : FnId, CalleeRespects (UP f) (clift f)
  /-- The B environment is well-typed: each slot's denotation has
      footprint matching its declared channel set. -/
  envWT     : EnvWT env

/-- Reify a GT `Expr` into a B `BExpr α` using a `Compat` bundle.

    * Literals: `Expr.lit c` → `BExpr.const (liftLit c)`.
    * Variables: `Expr.var x` → `BExpr.var (varSlot x)`.
    * Channel seeds: `Expr.value k` → `BExpr.var (chanSlot k)`.  (For
      well-formed Expr, `k < 8`; otherwise the slot is arbitrary and
      `lowering_types` requires `ExprWellFormed` to discharge the case.)
    * Binary arithmetic: structural.
    * User-function calls: `Expr.call2 f e1 e2` →
      `BExpr.call2 (UP f) (clift f) (lower e1) (lower e2)`. -/
def lower (C : Compat α) : Expr → BExpr α
  | .lit c          => BExpr.const (C.liftLit c)
  | .var x          => BExpr.var (C.varSlot x)
  | .value k        => BExpr.var (C.chanSlot k)
  | .add e1 e2      => BExpr.add (lower C e1) (lower C e2)
  | .mul e1 e2      => BExpr.mul (lower C e1) (lower C e2)
  | .call2 f e1 e2  => BExpr.call2 (C.UP f) (C.clift f) (lower C e1) (lower C e2)

-- ===========================================================================
-- §13. Cross-world correspondence (Stage 2: lowering preserves typing)
-- ===========================================================================

/-- **Lowering preserves (tightens) GT typing.**

    Given a `Compat` bundle and a well-formed GT expression `e` with
    `Typing C.Γ e S`, the reified B expression `lower C e` has
    `BTyping C.env (lower C e) S'` for some `S' ⊆ S`.

    The `S' = S` refinement holds for the non-call cases (tLit, tVar,
    tValue, tAdd, tMul produce matching sets).  The `S' ⊊ S` case arises
    only at `tCall2`: GT's week-3 rule gives `S1.union S2` while the B
    side's body-precision rule gives `usedUnion (UP f) S1' S2'`, which
    can be strictly smaller when `UP f` drops an unused parameter.  The
    `∃ S'` shape accommodates this tightening — strictly better than
    `=`, and exactly what composition with the existing `bridge` theorem
    needs.

    Proof: structural induction on the GT typing derivation.  Each case
    picks a witness `S'` and discharges the subset obligation using the
    imported `ChSet.union_mono` / `usedUnion_mono` /
    `usedUnion_subset_union` from `GradientTopology`. -/
theorem lowering_types (C : Compat α) (e : Expr) (S : ChSet)
    (hT : Typing C.Γ e S) :
    ExprWellFormed e →
    ∃ S', BTyping C.env (lower C e) S' ∧ S'.subset S := by
  induction hT with
  | tLit c =>
      intro _
      refine ⟨ChSet.empty, ?_, ChSet.subset_refl _⟩
      show BTyping C.env (BExpr.const (C.liftLit c)) ChSet.empty
      exact BTyping.const _
  | tVar x =>
      intro _
      refine ⟨C.Γ x, ?_, ChSet.subset_refl _⟩
      show BTyping C.env (BExpr.var (C.varSlot x)) (C.Γ x)
      rw [← C.varComp x]
      exact BTyping.var _
  | tValue k =>
      intro hwf
      simp only [ExprWellFormed] at hwf
      refine ⟨ChSet.singleton k, ?_, ChSet.subset_refl _⟩
      show BTyping C.env (BExpr.var (C.chanSlot k)) (ChSet.singleton k)
      rw [← C.chanComp k hwf]
      exact BTyping.var _
  | tAdd e1 e2 S1 S2 _h1 _h2 ih1 ih2 =>
      intro hwf
      simp only [ExprWellFormed] at hwf
      obtain ⟨hwf1, hwf2⟩ := hwf
      obtain ⟨S1', hBT1, hsub1⟩ := ih1 hwf1
      obtain ⟨S2', hBT2, hsub2⟩ := ih2 hwf2
      refine ⟨S1'.union S2', ?_, ChSet.union_mono _ _ _ _ hsub1 hsub2⟩
      show BTyping C.env (BExpr.add (lower C e1) (lower C e2)) (S1'.union S2')
      exact BTyping.add _ _ _ _ hBT1 hBT2
  | tMul e1 e2 S1 S2 _h1 _h2 ih1 ih2 =>
      intro hwf
      simp only [ExprWellFormed] at hwf
      obtain ⟨hwf1, hwf2⟩ := hwf
      obtain ⟨S1', hBT1, hsub1⟩ := ih1 hwf1
      obtain ⟨S2', hBT2, hsub2⟩ := ih2 hwf2
      refine ⟨S1'.union S2', ?_, ChSet.union_mono _ _ _ _ hsub1 hsub2⟩
      show BTyping C.env (BExpr.mul (lower C e1) (lower C e2)) (S1'.union S2')
      exact BTyping.mul _ _ _ _ hBT1 hBT2
  | tCall2 f e1 e2 S1 S2 _h1 _h2 ih1 ih2 =>
      intro hwf
      simp only [ExprWellFormed] at hwf
      obtain ⟨hwf1, hwf2⟩ := hwf
      obtain ⟨S1', hBT1, hsub1⟩ := ih1 hwf1
      obtain ⟨S2', hBT2, hsub2⟩ := ih2 hwf2
      refine ⟨usedUnion (C.UP f) S1' S2', ?_, ?_⟩
      · show BTyping C.env
              (BExpr.call2 (C.UP f) (C.clift f) (lower C e1) (lower C e2))
              (usedUnion (C.UP f) S1' S2')
        exact BTyping.call2 (C.UP f) (C.clift f) _ _ _ _
                (C.cliftResp f) hBT1 hBT2
      · -- Chain: usedUnion (UP f) S1' S2' ⊆ usedUnion (UP f) S1 S2 ⊆ S1.union S2
        exact ChSet.subset_trans _ _ _
          (usedUnion_mono (C.UP f) S1' S2' S1 S2 hsub1 hsub2)
          (usedUnion_subset_union (C.UP f) S1 S2)

-- ===========================================================================
-- §14. Cross-world soundness (Stage 3: gtt_corresponds)
-- ===========================================================================

/-- **Footprint is monotone in the set argument.**

    If `S' ⊆ S` and `Footprint S' D`, then `Footprint S D`.  Intuitively:
    a looser declared set only adds channels at which the value *may* be
    nonzero; channels already declared absent in `S'` remain absent in
    `S`.  Contrapositive reasoning: `j ∉ S ⟹ j ∉ S'` (by subset), so
    `hfp`'s zero conclusion transports. -/
theorem Footprint_mono (S' S : ChSet) (hsub : S'.subset S)
    (D : Dual2 α) (hfp : Footprint S' D) : Footprint S D := by
  refine ⟨?_, ?_⟩
  · intro j hj
    apply hfp.1 j
    intro h
    exact hj (hsub j.val h)
  · intro j k hj
    apply hfp.2 j k
    intro h
    exact hj (hsub j.val h)

/-- **Cross-world soundness theorem** (β⁹ week-4+ coherence claim).

    For every well-formed GT-typed expression `e` with `Typing C.Γ e S`,
    the second-order AD evaluation `evalB C.env (lower C e)` has
    footprint bounded by `S`:

      * `j ∉ S ⟹ (evalB env (lower e)).grad j = 𝟎`
      * `j ∉ S ⟹ ∀ k, (evalB env (lower e)).hess j k = 𝟎`

    This is the formal counterpart of the compiler's claim
    "`hessian_of(e, j, k) = 0` whenever `j ∉ GTT(e)`": the declared
    channel set from the GT typing judgment soundly bounds the actual
    first- and second-order derivative support of the Dual2-evaluated
    expression.

    Proof: compose `lowering_types` (GT → B typing) with the existing
    inductive `bridge` theorem (B typing → Footprint) using
    `Footprint_mono` to loosen the tighter B-side set back to the GT
    declared set.  The existing `bridge` and its corollaries remain
    intact — this theorem is *additive*, not a replacement. -/
theorem gtt_corresponds (C : Compat α) (e : Expr) (S : ChSet)
    (hwf : ExprWellFormed e) (hT : Typing C.Γ e S) :
    Footprint S (evalB C.env (lower C e)) := by
  obtain ⟨S', hBT, hsub⟩ := lowering_types C e S hT hwf
  have fp_tight : Footprint S' (evalB C.env (lower C e)) :=
    bridge C.env C.envWT (lower C e) S' hBT
  exact Footprint_mono S' S hsub _ fp_tight

/-- **Cross-world Hessian sparsity corollary.**  Channel `j ∉ S` on the
    GT side implies the full `j`-row of the output Hessian is zero —
    compiler-level: `EXPR_HSHADOW_jk = 0` for all `k` when `j` is
    outside the GT-declared channel set of `e`. -/
theorem gtt_hessian_sparsity (C : Compat α) (e : Expr) (S : ChSet)
    (hwf : ExprWellFormed e) (hT : Typing C.Γ e S)
    (j k : Fin 8) (hj : ¬ S j.val) :
    (evalB C.env (lower C e)).hess j k = 𝟎 :=
  (gtt_corresponds C e S hwf hT).2 j k hj

/-- **Cross-world gradient sparsity corollary.**  Channel `j ∉ S` on the
    GT side implies `EXPR_SSHADOW_j = 0`. -/
theorem gtt_gradient_sparsity (C : Compat α) (e : Expr) (S : ChSet)
    (hwf : ExprWellFormed e) (hT : Typing C.Γ e S)
    (j : Fin 8) (hj : ¬ S j.val) :
    (evalB C.env (lower C e)).grad j = 𝟎 :=
  (gtt_corresponds C e S hwf hT).1 j hj

-- ===========================================================================
-- §14b. Cross-world canary — end-to-end witness at `α = Nat`
-- ===========================================================================

/-!
`§14`'s `gtt_corresponds` has four compatibility fields beyond the base
data (`varComp`, `chanComp`, `cliftResp`, `envWT`).  This section
exhibits a concrete `Compat Nat` that discharges all four, plus a
concrete GT expression + typing derivation, and then applies
`gtt_corresponds` to derive a non-vacuous Footprint.

The canary Expr is `add (value 0) (value 1)` — GT-typed at
`singleton 0 ∪ singleton 1`, lowered to `add (var ⟨0,_⟩) (var ⟨1,_⟩)`
under a full seed environment.  The derived Footprint checks that
gradient slots 2-7 are zero and Hessian rows 2-7 are zero for the
evaluated Dual2 value.
-/

/-- Full-seed environment on `Nat`: each `Fin 8` slot carries the
    corresponding seed and the singleton channel set matching its
    numeric identity.  This is the canonical `EnvWT` instance that
    supports all 8 channels. -/
def seedEnvNat : BEnv Nat where
  denotation k := dual2Seed k 0
  chset      k := ChSet.singleton k.val

/-- The seed environment is well-typed: each slot's denotation footprint
    matches its declared channel set. -/
theorem seedEnvNat_wellTyped : EnvWT seedEnvNat :=
  fun k => fp_seed k 0

/-- Compatibility bundle instantiating `Compat Nat` against `seedEnvNat`.

    * `Γ`, `varSlot` are consistent by construction (`varComp := rfl`).
    * `chanSlot k = ⟨k, hk⟩` for `k < 8` yields `seedEnvNat.chset (chanSlot k)
      = ChSet.singleton k` (chanComp).
    * `UP = ⟨true, true⟩` + `clift = dual2Add` gives `CalleeRespects` via
      `usedUnion_all_used` + `fp_add_union`.
    * `envWT` = `seedEnvNat_wellTyped`.

    Exhibiting this witness is the inhabitability guarantee for
    `gtt_corresponds`: the cross-world theorem is not vacuously true. -/
def seedCompatNat : Compat Nat where
  Γ         := fun _ => ChSet.singleton 0
  env       := seedEnvNat
  liftLit   := fun _ => 0
  varSlot   := fun _ => ⟨0, by decide⟩
  chanSlot  := fun k => if h : k < 8 then ⟨k, h⟩ else ⟨0, by decide⟩
  UP        := fun _ => ⟨true, true⟩
  clift     := fun _ => dual2Add
  varComp   := fun _ => rfl
  chanComp  := fun k hk => by
    show seedEnvNat.chset (if h : k < 8 then ⟨k, h⟩ else ⟨0, by decide⟩)
         = ChSet.singleton k
    simp [hk, seedEnvNat]
  cliftResp := fun _ Sa Sb Da Db hA hB => by
    show Footprint (usedUnion ⟨true, true⟩ Sa Sb) (dual2Add Da Db)
    rw [usedUnion_all_used]
    exact fp_add_union Sa Sb Da Db hA hB
  envWT     := seedEnvNat_wellTyped

/-- Canary GT expression: `add (value 0) (value 1)`.  Well-formed (both
    `value` channels are in `{0..7}`) and types at
    `(singleton 0).union (singleton 1)` under any `TyEnv`. -/
def gtCanaryExpr : Expr := Expr.add (Expr.value 0) (Expr.value 1)

/-- `gtCanaryExpr` is well-formed: both channel indices (0 and 1) are < 8. -/
theorem gtCanaryExpr_wellFormed : ExprWellFormed gtCanaryExpr := by
  -- Unfold both defs so the goal reduces to `(0 < 8) ∧ (1 < 8)`.
  simp only [gtCanaryExpr, ExprWellFormed]
  exact ⟨by decide, by decide⟩

/-- `gtCanaryExpr` GT-types at `(ChSet.singleton 0).union (ChSet.singleton 1)`
    under any typing environment.  Proof: `tAdd` over two `tValue`. -/
theorem gtCanaryExpr_types (Γ : TyEnv) :
    Typing Γ gtCanaryExpr ((ChSet.singleton 0).union (ChSet.singleton 1)) :=
  Typing.tAdd (Expr.value 0) (Expr.value 1)
    (ChSet.singleton 0) (ChSet.singleton 1)
    (Typing.tValue 0) (Typing.tValue 1)

/-- **End-to-end cross-world canary.**

    `gtt_corresponds` applied to `gtCanaryExpr` under `seedCompatNat`
    yields `Footprint ((singleton 0).union (singleton 1))
    (evalB seedEnvNat (lower seedCompatNat gtCanaryExpr))`.  In
    particular: every `Fin 8` slot with `j.val ∉ {0, 1}` has zero
    gradient and zero Hessian row on the evaluated Dual2 value. -/
theorem canary_gtt_corresponds :
    Footprint ((ChSet.singleton 0).union (ChSet.singleton 1))
      (evalB seedEnvNat (lower seedCompatNat gtCanaryExpr)) :=
  gtt_corresponds seedCompatNat gtCanaryExpr _
    gtCanaryExpr_wellFormed (gtCanaryExpr_types seedCompatNat.Γ)

/-- Concrete corollary of the cross-world canary: slot `⟨2, _⟩` has
    zero gradient on the evaluated Dual2 value — slot 2 is outside both
    singleton channels, so the cross-world theorem rules out any
    first-order contribution from that slot. -/
theorem canary_gtt_slot2_gradient_zero :
    (evalB seedEnvNat (lower seedCompatNat gtCanaryExpr)).grad ⟨2, by decide⟩
      = (0 : Nat) := by
  apply canary_gtt_corresponds.1 ⟨2, by decide⟩
  -- ¬ ((ChSet.singleton 0).union (ChSet.singleton 1)) 2
  decide

-- ===========================================================================
-- §14c. Cross-world canary at `α = Float` — compiler-runtime inhabitant
-- ===========================================================================

/-!
`§14b` exhibits a concrete `Compat Nat`, proving the cross-world theorem
is not vacuously true at the decidable ground algebra.  But the Sounio
compiler's runtime shadow channels hold `Float`, not `Nat`.  This
section exhibits a concrete `Compat Float` under the `Dual2Field Float`
instance from `HessianAD §11`, closing the inhabitability gap at the
algebra actually used by the compiler.

The canary expression is *pure add* (`add (value 0) (value 1)`).
Semantically the witness construction uses only the three IEEE-754
axioms that are bit-exact for finite non-NaN values (`float_zero_add_ax`,
`float_add_zero_ax`, `float_add_comm_ax`) — `fp_add_union` invokes
`F.zero_add`, and the reified Dual2 evaluation threads through
`dual2Add` only.

However, `#print axioms canary_gtt_corresponds_float` reports the full
seven-axiom set because Lean's dependency tracker traces every field
of the `Dual2Field Float` instance bundle, including the four mul-laws
that are never exercised along this witness.  This is a Lean
bookkeeping artifact, not a semantic reliance: the `float_mul_assoc_ax`
trust boundary is never applied inside the proof steps, only sitting
unused in the instance record.  A typeclass refactor that splits
`Dual2AddField` out of `Dual2Field` would tighten the reported
footprint to match the semantic one; deferred.
-/

/-- Full-seed environment on `Float`.  Structural twin of `seedEnvNat`;
    each `Fin 8` slot carries the corresponding channel-seed and the
    singleton channel set matching its numeric identity. -/
noncomputable def seedEnvFloat : BEnv Float where
  denotation k := dual2Seed k 0.0
  chset      k := ChSet.singleton k.val

/-- The Float seed environment is well-typed. -/
theorem seedEnvFloat_wellTyped : EnvWT seedEnvFloat :=
  fun k => fp_seed k 0.0

/-- Compatibility bundle instantiating `Compat Float` against `seedEnvFloat`.
    Mirrors `seedCompatNat` exactly, swapping `Nat` for `Float` and using
    the `Dual2Field Float` instance from `HessianAD §11`.  The `cliftResp`
    proof goes through because `fp_add_union` is polymorphic over
    `[Dual2Field α]` and invokes only `F.zero_add` — the IEEE-754-exact
    `float_zero_add_ax`. -/
noncomputable def seedCompatFloat : Compat Float where
  Γ         := fun _ => ChSet.singleton 0
  env       := seedEnvFloat
  liftLit   := fun c => c
  varSlot   := fun _ => ⟨0, by decide⟩
  chanSlot  := fun k => if h : k < 8 then ⟨k, h⟩ else ⟨0, by decide⟩
  UP        := fun _ => ⟨true, true⟩
  clift     := fun _ => dual2Add
  varComp   := fun _ => rfl
  chanComp  := fun k hk => by
    show seedEnvFloat.chset (if h : k < 8 then ⟨k, h⟩ else ⟨0, by decide⟩)
         = ChSet.singleton k
    simp [hk, seedEnvFloat]
  cliftResp := fun _ Sa Sb Da Db hA hB => by
    show Footprint (usedUnion ⟨true, true⟩ Sa Sb) (dual2Add Da Db)
    rw [usedUnion_all_used]
    exact fp_add_union Sa Sb Da Db hA hB
  envWT     := seedEnvFloat_wellTyped

/-- **End-to-end cross-world canary at `α = Float`.**

    `gtt_corresponds` applied to `gtCanaryExpr` under `seedCompatFloat`
    yields `Footprint ((singleton 0).union (singleton 1))
    (evalB seedEnvFloat (lower seedCompatFloat gtCanaryExpr))`.
    In particular: every `Fin 8` slot with `j.val ∉ {0, 1}` has zero
    gradient and zero Hessian row on the evaluated `Dual2 Float` value. -/
theorem canary_gtt_corresponds_float :
    Footprint ((ChSet.singleton 0).union (ChSet.singleton 1))
      (evalB seedEnvFloat (lower seedCompatFloat gtCanaryExpr)) :=
  gtt_corresponds seedCompatFloat gtCanaryExpr _
    gtCanaryExpr_wellFormed (gtCanaryExpr_types seedCompatFloat.Γ)

/-- Float counterpart of `canary_gtt_slot2_gradient_zero`: slot `⟨2,_⟩`
    has zero gradient on the evaluated `Dual2 Float` value — slot 2 is
    outside both singleton channels, so the cross-world theorem rules
    out any first-order contribution from that slot. -/
theorem canary_gtt_slot2_gradient_zero_float :
    (evalB seedEnvFloat (lower seedCompatFloat gtCanaryExpr)).grad
        ⟨2, by decide⟩
      = (0.0 : Float) := by
  apply canary_gtt_corresponds_float.1 ⟨2, by decide⟩
  -- ¬ ((ChSet.singleton 0).union (ChSet.singleton 1)) 2
  decide

-- ===========================================================================
-- §15. Summary
-- ===========================================================================

/-!
## Verified Properties

| Property                                               | Status  | Location                   |
|--------------------------------------------------------|---------|----------------------------|
| ChSet / usedUnion / usedUnion_subset_union (imported)  | proved  | GradientTopology §2        |
| Dual2 arithmetic (const/seed/add/mul/unary, imported)  | def     | HessianAD §1-§4            |
| GradFootprint / HessFootprint (imported)               | def     | HessianAD §9b              |
| Footprint (grad ∧ hess conjunction)                    | def     | §4                         |
| not_union_iff / not_usedUnion_{left,right}             | proved  | §4 helpers                 |
| fp_const / fp_seed                                     | proved  | §5                         |
| fp_same_add / fp_apply_unary  (same-set)               | proved  | §5                         |
| fp_add_union / fp_mul_union   (union of two sets)      | proved  | §6                         |
| BExpr / BEnv / EnvWT / CalleeRespects / evalB          | def     | §7, §8                     |
| BTyping inductive relation                             | induct. | §9                         |
| **bridge** (EnvWT + BTyping ⟹ Footprint)              | proved  | §10                        |
| hessian_sparsity / gradient_sparsity corollaries       | proved  | §11                        |
| body_precision_safety (usedUnion ⊆ union)              | proved  | §11                        |
| seed_env_is_wt (concrete EnvWT witness)                | proved  | §11                        |
| instDual2FieldNat (concrete ground algebra)            | def     | §11b                       |
| projA_respects_body_precision (CalleeRespects witness) | proved  | §11b                       |
| canaryE / canaryEnv / canaryEnv_wellTyped              | def+prv | §11b                       |
| canaryE_types (BTyping at usedUnion ⟨true,false⟩)      | proved  | §11b                       |
| **gtt_hessian_bridge_tight** (tightness witness)       | proved  | §11b                       |
| **canary_bridge_exact** (upper-bound × tightness)      | proved  | §11b                       |
| ExprWellFormed / Compat / lower                        | def     | §12                        |
| **lowering_types** (GT Typing → B BTyping, tightened)  | proved  | §13                        |
| Footprint_mono (subset → Footprint monotone)           | proved  | §14                        |
| **gtt_corresponds** (cross-world soundness)            | proved  | §14                        |
| gtt_hessian_sparsity / gtt_gradient_sparsity           | proved  | §14 (cross-world corollaries) |
| seedEnvNat / seedEnvNat_wellTyped                      | def+prv | §14b                       |
| seedCompatNat (concrete Compat Nat witness)            | def     | §14b                       |
| gtCanaryExpr / gtCanaryExpr_{wellFormed,types}         | def+prv | §14b                       |
| **canary_gtt_corresponds** (end-to-end inhabitant)     | proved  | §14b                       |
| canary_gtt_slot2_gradient_zero (concrete slot corollary)| proved | §14b                       |
| seedEnvFloat / seedEnvFloat_wellTyped                  | def+prv | §14c                       |
| seedCompatFloat (concrete Compat Float witness)        | def     | §14c                       |
| **canary_gtt_corresponds_float** (Float inhabitant)    | proved  | §14c                       |
| canary_gtt_slot2_gradient_zero_float                   | proved  | §14c                       |

## Key design decisions

- **CalleeRespects in BTyping.call2**: the callee contract is part of the
  typing evidence, so `bridge` needs only `EnvWT env` as a side hypothesis.
- **Union-of-two-sets preservation** (`fp_add_union`, `fp_mul_union`): the
  key insight is `not_union_iff` — absence from a union implies absence from
  both components — so we can extract both `¬ Sa j` and `¬ Sb j` and apply
  `hA` and `hB` independently.
- **var types at env.chset k**: the typing rule for variables defers to the
  environment's declared channel set, matching standard type-theoretic
  practice.  `seed_env_is_wt` shows the canonical instance.
- **Tightness via concrete Nat instance**: `gtt_hessian_bridge_tight` uses
  `instDual2FieldNat` so that `𝟎 = 0 : Nat` and `𝟏 = 1 : Nat` are
  decidably distinct (`by decide`).  Float would need noncomputable
  decidability; Nat avoids that without losing generality of the
  upper-bound theorem, which remains polymorphic over `[Dual2Field α]`.
- **Tightness is existential, not universal**: we exhibit a specific env
  in which the body-precision declared set is exactly the nonzero-grad
  support.  A general statement "for every `UP` and every typing
  derivation some env witnesses equality" is stronger and deferred.
- **Cross-world correspondence is additive, not a replacement**: the
  inductive `bridge` theorem (§10) remains the minimal API that
  `hessian_sparsity`/`gradient_sparsity`/`canary_bridge_exact` consume.
  §14's `gtt_corresponds` links the GT and B typing worlds *without*
  replacing the clean B-side proof; it lifts the typing judgment from
  GT's scalar/relational world (`Expr`/`Typing`/`Eval`/`FnTable`) into
  the B world via `lower : Expr → BExpr α`, then composes with `bridge`.
- **Existential shape of `lowering_types`**: the result is
  `∃ S', BTyping (lower e) S' ∧ S'.subset S` rather than
  `BTyping (lower e) S`.  At `tCall2`, GT's week-3 rule gives
  `S1.union S2` but B's rule gives `usedUnion (UP f) S1' S2'` — strictly
  tighter when the callee's body-precision summary drops parameters.
  `Footprint_mono` reconciles the tightening at the final composition.
- **Reification parameterised on a `Compat` bundle**: the user provides
  the variable-slot map, channel-slot map, literal lifter, body-precision
  summary, callee lifter, and the compatibility/respects witnesses.
  Keeping all of these in one bundle avoids threading ~7 hypotheses
  through every sub-theorem and matches the shape of §11b's `canaryEnv`
  / `canaryE_types` setup.

## Remaining gaps (same as HessianAD §10)

- N-ary calls (arity > 2): extend `BExpr.call2` to `BExpr.callN` with a
  `List (Bool × BExpr α)` argument list and a `Finset (Fin 8) → ChSet`
  usage summary.  The proof is structural induction on the argument list.
- Closure / fn-ref indirect calls.
- ARM64 evaluator mirror.
- Connecting `CalleeRespects` to concrete compiler emission: proving that
  the x86-64 code emitted by `lean_single.sio` for a body-precision-analysed
  callee satisfies `CalleeRespects`.  This requires a full compiler
  correctness theorem (future work).

## Axioms used

None beyond `Dual2Field` hypotheses.  All proofs use only structural
induction, `simp` with the field axioms, and `Bool` lemmas (`Bool.or_eq_true`,
`Bool.not_eq_true`).
-/

end Sounio.GradientTopologyBridge
