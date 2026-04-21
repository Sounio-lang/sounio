/-!
# Sounio.GradientTopologyBridge — Full inductive GTT-HessianAD bridge theorem

Proves: for every expression `e` in the body-precision fragment, if the GTT
typing `⊢ e : S` holds and `D = eval(e)`, then

  `j ∉ S ⟹ D.grad j = 0`        (GradFootprint)
  `j ∉ S ⟹ ∀ k, D.hess j k = 0` (HessFootprint)

This connects two companion files:

  - `GradientTopology.lean` (GTT typing, `usedUnion`, body-precision §7b)
  - `HessianAD.lean`        (Dual2 arithmetic, footprint preservation §9b)

Self-contained: definitions repeat those in the companions verbatim so the
file compiles independently (matching the project's no-import convention).
Zero sorry.  No Mathlib.  No new axioms beyond `Dual2Field`.
-/

namespace Sounio.GradientTopologyBridge

-- ===========================================================================
-- §1. Channel sets  (mirrors GradientTopology §2)
-- ===========================================================================

abbrev ChSet := Nat → Bool

namespace ChSet
  def union  (S T : ChSet) : ChSet := fun j => S j || T j
  def empty  : ChSet := fun _ => false
  def singleton (j : Nat) : ChSet := fun i => decide (i = j)
  def subset (S T : ChSet) : Prop := ∀ n, S n → T n
end ChSet

-- ===========================================================================
-- §2. Body-precision summary  (mirrors GradientTopology §7b)
-- ===========================================================================

structure UsedParams2 where
  uses0 : Bool
  uses1 : Bool

def usedUnion (U : UsedParams2) (S1 S2 : ChSet) : ChSet :=
  ChSet.union
    (if U.uses0 then S1 else ChSet.empty)
    (if U.uses1 then S2 else ChSet.empty)

theorem usedUnion_subset_union (U : UsedParams2) (S1 S2 : ChSet) :
    (usedUnion U S1 S2).subset (S1.union S2) := by
  intro n h
  simp only [usedUnion, ChSet.union] at h
  cases hu0 : U.uses0 with
  | true =>
    cases hu1 : U.uses1 with
    | true  => rw [hu0, hu1] at h; simp only [if_true] at h; simp only [ChSet.union]; exact h
    | false =>
      rw [hu0, hu1] at h; simp only [if_true, if_false] at h
      simp only [ChSet.union]
      cases hs1 : S1 n with
      | true  => simp [hs1]
      | false => rw [hs1] at h; simp [ChSet.empty] at h
  | false =>
    cases hu1 : U.uses1 with
    | true  =>
      rw [hu0, hu1] at h; simp only [if_false, if_true] at h
      simp only [ChSet.union]
      cases hs2 : S2 n with
      | true  => simp [hs2]
      | false => rw [hs2] at h; simp [ChSet.empty] at h
    | false => rw [hu0, hu1] at h; simp [ChSet.empty] at h

-- ===========================================================================
-- §3. Dual2 arithmetic  (mirrors HessianAD §1-§4)
-- ===========================================================================

class Dual2Field (α : Type) extends Add α, Mul α where
  zero      : α
  one       : α
  zero_mul  : ∀ a : α, zero * a = zero
  mul_zero  : ∀ a : α, a * zero = zero
  zero_add  : ∀ a : α, zero + a = a
  add_zero  : ∀ a : α, a + zero = a
  add_comm  : ∀ a b : α, a + b = b + a
  mul_comm  : ∀ a b : α, a * b = b * a
  mul_assoc : ∀ a b c : α, (a * b) * c = a * (b * c)

variable {α : Type} [F : Dual2Field α]

local notation "𝟎" => @Dual2Field.zero α F
local notation "𝟏" => @Dual2Field.one α F

structure Dual2 (α : Type) [Dual2Field α] where
  val  : α
  grad : Fin 8 → α
  hess : Fin 8 → Fin 8 → α

def dual2Const (v : α) : Dual2 α :=
  ⟨v, fun _ => 𝟎, fun _ _ => 𝟎⟩

def dual2Seed (k : Fin 8) (v : α) : Dual2 α :=
  ⟨v, fun i => if i = k then 𝟏 else 𝟎, fun _ _ => 𝟎⟩

def dual2Add (a b : Dual2 α) : Dual2 α :=
  ⟨a.val + b.val,
   fun k => a.grad k + b.grad k,
   fun j k => a.hess j k + b.hess j k⟩

def dual2Mul (a b : Dual2 α) : Dual2 α :=
  ⟨a.val * b.val,
   fun k => a.val * b.grad k + b.val * a.grad k,
   fun j k =>
     a.hess j k * b.val +
     (a.grad j * b.grad k + a.grad k * b.grad j) +
     a.val * b.hess j k⟩

def dual2ApplyUnary (fv fp fpp : α) (g : Dual2 α) : Dual2 α :=
  ⟨fv,
   fun k => fp * g.grad k,
   fun j k => fpp * g.grad j * g.grad k + fp * g.hess j k⟩

-- ===========================================================================
-- §4. Footprint predicates  (mirrors HessianAD §9b)
-- ===========================================================================

-- We write `¬ S j.val` throughout; Lean 4 coerces `Bool → Prop` via `= true`,
-- so `¬ S j.val` means `S j.val = false` (channel j not in set S).

def GradFootprint (S : ChSet) (D : Dual2 α) : Prop :=
  ∀ j : Fin 8, ¬ S j.val → D.grad j = 𝟎

def HessFootprint (S : ChSet) (D : Dual2 α) : Prop :=
  ∀ j k : Fin 8, ¬ S j.val → D.hess j k = 𝟎

def Footprint (S : ChSet) (D : Dual2 α) : Prop :=
  GradFootprint S D ∧ HessFootprint S D

-- Helper: channel j absent from a union means absent from both components.
private lemma not_union_iff (Sa Sb : ChSet) (j : Fin 8)
    (h : ¬ (Sa.union Sb) j.val) : ¬ Sa j.val ∧ ¬ Sb j.val := by
  simp only [ChSet.union] at h
  constructor
  · intro ha; exact h (by simp [ha])
  · intro hb; exact h (by simp [hb])

-- Helper: channel j absent from usedUnion implies absent from whichever
-- component the union includes.
private lemma not_usedUnion_left (U : UsedParams2) (Sa Sb : ChSet) (j : Fin 8)
    (h : ¬ (usedUnion U Sa Sb) j.val) (hu : U.uses0 = true) : ¬ Sa j.val := by
  intro ha
  apply h
  simp only [usedUnion, ChSet.union, hu, if_true]
  simp [ha]

private lemma not_usedUnion_right (U : UsedParams2) (Sa Sb : ChSet) (j : Fin 8)
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
-- §12. Summary
-- ===========================================================================

/-!
## Verified Properties

| Property                                               | Status  | Location                   |
|--------------------------------------------------------|---------|----------------------------|
| ChSet / usedUnion / usedUnion_subset_union             | proved  | §1, §2                     |
| Dual2 arithmetic (const/seed/add/mul/unary)            | def     | §3                         |
| GradFootprint / HessFootprint / Footprint              | def     | §4                         |
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
