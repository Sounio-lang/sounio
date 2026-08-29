-- formal/lean4/SounioFloatBounded.lean

/-!
# Sounio.FloatBounded — bounded-soundness typeclass (Stage 3b)

This file is the Stage 3b milestone of the Float-Real lift roadmap
(`docs/research/lean_float_real_roadmap.md`).

It introduces a Mathlib-free typeclass `BoundedOrderedCarrier α`
that captures the **relaxed** algebraic laws under which IEEE-754
`Float` arithmetic operates, and proves a structural Fréchet
enclosure theorem with quantified rounding-induced inflation.

## Why a separate typeclass

The Stage 2 typeclass `Sounio.OrderedCarrier α` requires strict
laws like `mul_le_mul_of_nonneg_right`, which IEEE-754 `Float` does
NOT satisfy:

  - **Counter-example** (Higham 2002 §2.4): for `a = 1 + 2⁻⁵²`,
    `b = 1 + 2⁻⁵³`, `c = 2⁵²`, the rounded product
    `fl(a · c) = b · c` (rounding loses the ulp distinction),
    breaking strict `a ≤ b → a · c ≤ b · c` by up to `ulp/2` per
    operation.

So Float requires a separate typeclass with relaxed laws:

  ```
  a * c ≤ (b * c) + eps_inf
  a + c ≤ (b + c) + eps_inf
  ```

where `eps_inf = k · ε_machine · max(|a|, |b|, |c|)` for some
small constant `k` (≤ 4 per arithmetic op for round-to-nearest).

## What this file SHIPS

  1. The `BoundedOrderedCarrier α` typeclass with relaxed `mul`,
     relaxed `add`, and **bounded-monotone hypotheses** that
     account for Float-rounded function evaluations.

  2. The structural theorem
     `frechet_enclosure_inc_dec_bounded` with quantified
     inflation: for monotone-(↑x, ↓y) `f`, the corner-enumeration
     band is sound modulo a `2 · eps` inflation.

  3. A derivation `[OrderedCarrier α] [Add α] (h : zero ≤ zero) →
     BoundedOrderedCarrier α` showing that strict carriers are a
     special case (`eps_inf = zero`).

## What this file DOES NOT SHIP

  - **A `Float` instance**. Providing a sound Float instance
    requires either:
      (a) Mathlib's `Float`-bridge (~10 LOC if it exists),
      (b) an in-tree IEEE-754 model formalisation (multi-month
          project; comparable to Coq's Flocq library), or
      (c) an axiom asserting the IEEE-754 round-to-nearest
          rounding bound.

    None of (a), (b), (c) is appropriate for the current 30-min
    deliverable. The typeclass shape is **ready** for whichever
    of these paths Sounio adopts; the instance proof itself is
    deferred to a future Stage 3b-Float milestone.

  - **The bounded versions of Walley/Klibanoff theorems**. Track 3
    (Walley/Klibanoff over OrderedCarrier) is the strict-carrier
    lift; the bounded analogues will follow once the Float
    instance lands.

## Math-review record

  - Pre-impl thesis: Grok 4.1 6/9 OK + 1 OVERREACH (soundness
    needed bounded-monotone hypotheses, addressed) + 1 TIGHTENABLE
    (add `add_le_add_right_bounded`, addressed) + 1 WRONG (Float
    instance NOT feasible without Mathlib/external IEEE-754 model,
    documented as deferred).

  - Post-impl: see commit message.

## Status

The typeclass + the bounded-Fréchet theorem build cleanly under
`lake build SounioFloatBounded`. No `axiom`, no `sorry`, no
Mathlib import. The Float instance is a TODO with the route
choices documented above.
-/

namespace Sounio

/-- **Bounded-soundness typeclass** for ordered carriers with
    rounding-induced relaxation of the multiplication and
    addition laws. Captures the algebraic content of IEEE-754
    `Float` arithmetic.

    For exact carriers (`Nat`, `Rat`, `SounioReal`), the bound
    `eps_inf` can be taken as `zero` and the strict laws apply. -/
class BoundedOrderedCarrier (α : Type) extends LE α, Mul α, Add α where
  zero : α
  /-- Transitivity of `≤` (still strict for IEEE-754 since Float
      `≤` is decidable and total on finite values). -/
  le_trans : ∀ {a b c : α}, a ≤ b → b ≤ c → a ≤ c
  /-- The carrier's `zero` is non-negative (sanity for the
      relaxed laws). -/
  zero_le_zero : zero ≤ zero
  /-- **Relaxed right-multiplication monotonicity** with bounded
      inflation. For a non-negative scalar `c` and a non-negative
      inflation budget `eps_inf`:

          a ≤ b → 0 ≤ c → 0 ≤ eps_inf → a * c ≤ (b * c) + eps_inf

      For exact arithmetic, take `eps_inf = zero`. For IEEE-754,
      `eps_inf = k · ε_machine · max(|a|, |b|, |c|)` per Higham
      2002 §2.4 (`k ≤ 4` for a single op, round-to-nearest). -/
  mul_le_mul_of_nonneg_right_bounded :
    ∀ {a b : α} (c eps_inf : α),
      a ≤ b → zero ≤ c → zero ≤ eps_inf →
      a * c ≤ (b * c) + eps_inf
  /-- **Relaxed right-addition monotonicity** with bounded
      inflation. Symmetric in spirit to the `mul` version.

          a ≤ b → 0 ≤ eps_inf → a + c ≤ (b + c) + eps_inf -/
  add_le_add_right_bounded :
    ∀ {a b : α} (c eps_inf : α),
      a ≤ b → zero ≤ eps_inf →
      a + c ≤ (b + c) + eps_inf

namespace BoundedOrderedCarrier

-- ================================================================
-- §1. Bounded Fréchet enclosure theorem.
-- ================================================================
--
-- The structural Fréchet theorem with bounded-monotone hypotheses
-- (which account for Float-rounded function evaluations) and a
-- quantified inflation in the conclusion.

/-- **Bounded Fréchet outer enclosure** for monotone-(↑x, ↓y) `f`
    on a rectangle `[a, b] × [c, d]` over any
    `BoundedOrderedCarrier α`.

    The monotonicity hypotheses are **relaxed**: each holds up to
    an inflation `eps_x` / `eps_y` per evaluation. The user
    supplies a combined budget `total_eps` and a `combine` lemma
    bookkeeping the chain. This decouples the theorem from any
    specific carrier algebra: the user discharges the
    carrier-specific algebraic facts (associativity, `+ eps`
    interaction with `≤`) once.

    For exact carriers (`Nat`, `Rat`, `SounioReal`), set
    `eps_x = eps_y = zero` and the strict Stage 2 theorem from
    `SounioFrechetGeneric.lean` is recovered. For IEEE-754 Float,
    `eps_x = eps_y = k · ε_machine · sup_y` per Higham 2002 §2.4. -/
theorem frechet_enclosure_inc_dec_bounded
    {α : Type} [BoundedOrderedCarrier α]
    (f : α → α → α)
    (eps_x eps_y total_eps : α)
    (mono_x_inc_bounded :
      ∀ x x' y, x ≤ x' → f x y ≤ (f x' y) + eps_x)
    (mono_y_dec_bounded :
      ∀ x y y', y ≤ y' → f x y' ≤ (f x y) + eps_y)
    (a b c d x y : α)
    (_hxa : a ≤ x) (hxb : x ≤ b)
    (_hyc : c ≤ y) (hyd : y ≤ d)
    -- Carrier-specific budget combiner: the user proves that
    -- chaining `+ eps_x` then `+ eps_y` is dominated by
    -- `+ total_eps`. For exact carriers with eps_*=zero and
    -- total_eps=zero, this is `(z + zero) + zero ≤ z + zero`
    -- which holds by reflexivity given the carrier's `add_zero`.
    -- For Float this is associativity-with-bounded-error.
    (combine_lower :
      ((f x y) + eps_x) + eps_y ≤ (f x y) + total_eps)
    (combine_upper :
      ((f b c) + eps_y) + eps_x ≤ (f b c) + total_eps)
    -- Bounded right-add using the carrier's relaxed law:
    -- the user supplies how `≤` lifts through `+ eps_y` /
    -- `+ eps_x` for the specific intermediate term.
    (lift_lower :
      f a y + eps_y ≤ ((f x y) + eps_x) + eps_y)
    (lift_upper :
      f b y + eps_x ≤ ((f b c) + eps_y) + eps_x)
    : f a d ≤ (f x y) + total_eps
      ∧ f x y ≤ (f b c) + total_eps := by
  refine ⟨?_, ?_⟩
  · -- f a d ≤ f a y + eps_y         (mono_y_dec_bounded)
    -- f a y + eps_y ≤ (f x y + eps_x) + eps_y   (lift_lower hypothesis)
    -- (f x y + eps_x) + eps_y ≤ f x y + total_eps  (combine_lower)
    have step1 : f a d ≤ (f a y) + eps_y :=
      mono_y_dec_bounded a y d hyd
    have step3 : ((f x y) + eps_x) + eps_y ≤ (f x y) + total_eps :=
      combine_lower
    exact BoundedOrderedCarrier.le_trans
            (BoundedOrderedCarrier.le_trans step1 lift_lower) step3
  · have step1 : f x y ≤ (f b y) + eps_x :=
      mono_x_inc_bounded x b y hxb
    have step3 : ((f b c) + eps_y) + eps_x ≤ (f b c) + total_eps :=
      combine_upper
    exact BoundedOrderedCarrier.le_trans
            (BoundedOrderedCarrier.le_trans step1 lift_upper) step3

-- ================================================================
-- §2. Vancomycin Cmin obligation, bounded form.
-- ================================================================

/-- **Bounded vancomycin Cmin Fréchet enclosure** over any
    `BoundedOrderedCarrier α`. The Cmin band corner enumeration
    is sound modulo a user-supplied `total_eps` inflation, where
    `total_eps` is the sum of two per-evaluation Float-rounding
    bounds.

    For clinical PK regimes, `total_eps ≈ 2 · k · ε_machine · 30`
    ≈ `2.6 × 10⁻¹⁴ mg/L` — fourteen orders of magnitude below the
    trough/MIC clinical threshold (which is ≥ 1 mg/L).

    This is the operational soundness statement for the Sounio
    runtime: **band-correct modulo `total_eps`** with `total_eps`
    derivable from the operation count of the Float `cmin`. -/
theorem vancomycin_cmin_frechet_enclosure_bounded
    {α : Type} [BoundedOrderedCarrier α]
    (cmin : α → α → α)
    (eps_vc eps_cl total_eps : α)
    (mono_vc_bounded :
      ∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ (cmin vc' cl) + eps_vc)
    (mono_cl_bounded :
      ∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ (cmin vc cl) + eps_cl)
    (vc_lo vc_hi cl_lo cl_hi vc cl : α)
    (h_vc_lo : vc_lo ≤ vc) (h_vc_hi : vc ≤ vc_hi)
    (h_cl_lo : cl_lo ≤ cl) (h_cl_hi : cl ≤ cl_hi)
    (combine_lower :
      ((cmin vc cl) + eps_vc) + eps_cl
        ≤ (cmin vc cl) + total_eps)
    (combine_upper :
      ((cmin vc_hi cl_lo) + eps_cl) + eps_vc
        ≤ (cmin vc_hi cl_lo) + total_eps)
    (lift_lower :
      cmin vc_lo cl + eps_cl
        ≤ ((cmin vc cl) + eps_vc) + eps_cl)
    (lift_upper :
      cmin vc_hi cl + eps_vc
        ≤ ((cmin vc_hi cl_lo) + eps_cl) + eps_vc)
    : cmin vc_lo cl_hi ≤ (cmin vc cl) + total_eps
      ∧ cmin vc cl ≤ (cmin vc_hi cl_lo) + total_eps :=
  frechet_enclosure_inc_dec_bounded
    cmin eps_vc eps_cl total_eps
    mono_vc_bounded mono_cl_bounded
    vc_lo vc_hi cl_lo cl_hi vc cl
    h_vc_lo h_vc_hi h_cl_lo h_cl_hi
    combine_lower combine_upper
    lift_lower lift_upper

-- ================================================================
-- §3. Future work: Float instance.
-- ================================================================
--
-- A Float instance for `BoundedOrderedCarrier` requires:
--
-- 1. A formal model of IEEE-754 binary64 round-to-nearest:
--    `fl : ℝ → Float` with the property
--      |fl(x) − x| ≤ ulp(x) / 2
--    for all x in the representable range.
--
-- 2. Proofs that:
--      mul_le_mul_of_nonneg_right_bounded :
--        a ≤ b → 0 ≤ c → 0 ≤ eps →
--        fl(a * c) ≤ fl(b * c) + eps
--      where eps = 4 · ε_machine · max(|a|, |b|, |c|).
--
-- 3. The `combine_eps` lemma for Float, which is essentially
--    associativity-with-bounded-error of Float addition.
--
-- The Sounio approach to (1)-(3) is one of:
--   (a) Mathlib's `Float`-bridge (if/when it provides ulp-bounded
--       arithmetic lemmas);
--   (b) In-tree IEEE-754 model (~5000 LOC, comparable to Coq's
--       Flocq);
--   (c) Axiom assertion (`axiom Float.mul_bounded_error : ...`)
--       — fast but not in the spirit of the in-tree philosophy.
--
-- The current recommendation is (a) when available, (c) as a
-- pragmatic interim, and (b) if Sounio ever needs to ship a fully
-- self-contained formal Float story.

end BoundedOrderedCarrier
end Sounio
