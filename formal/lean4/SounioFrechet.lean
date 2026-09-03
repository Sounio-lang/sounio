-- formal/lean4/SounioFrechet.lean
/-!
# Fréchet Outer Enclosure for Monotone-in-Each-Arg Functions (M2.5)

This file mirrors the Sounio module
`stdlib/epistemic/knightian.sio` (the `pb_apply2_monotone_*` family)
and provides a Mathlib-free Lean discharge of the central Fréchet
enclosure theorem that justifies the vancomycin Cmin Knightian-band
implementation in `stdlib/clinical/vancomycin_pbpk.sio`.

## The claim

For any function `f : α → α → α` on a preorder, if `f` is monotone
in each argument with KNOWN signs (e.g., increasing in `x`, decreasing
in `y`), then for every point `(x, y)` of a closed rectangle
`[a, b] × [c, d]`, the value `f x y` lies between the two
sign-determined corners:

    f a d ≤ f x y ≤ f b c    (when f ↑ x, ↓ y)

This is the Fréchet (point-mass) outer enclosure: the interval
`[f a d, f b c]` contains every realisation of `f X Y` regardless of
the joint distribution of `(X, Y)` whose marginals are supported on
`[a, b]` and `[c, d]`. **No assumption about correlation, copula, or
joint dependence is required** — the bound is uniform over the
Fréchet class of all joints with the given marginals.

## Why this matters for vancomycin

`Cmin(Vc, CL)` is strictly increasing in `Vc` and strictly decreasing
in `CL` for all physiological positive parameters (math-review-validated
2026-04-30; see `stdlib/clinical/vancomycin_pbpk.sio` header). The
corner enumeration in `predict_cmin_knightian`,

    cmin_low  = Cmin(Vc_lo, CL_hi)
    cmin_high = Cmin(Vc_hi, CL_lo)

is therefore the Fréchet outer enclosure for any joint distribution of
`(Vc, CL)` supported on the marginal Knightian rectangle. This
addresses the consensus pushback on M0/M2 recorded in
`docs/research/knightian_operator_consensus_2026-04-30.md` (DeepSeek
+ Grok 4.1 flagged "joint (Vc, CL) dependence is the killer omission";
the response is that for monotone-in-each-arg PBPK functions the
omission *does not bite* — the enclosure holds copula-free).

## References

- Moore (1966) *Interval Analysis*, §3 — interval extension over rectangles
- Williamson & Downs (1990) — p-box arithmetic under unknown copula
- Ferson, Joslyn, Helton (2003) "Constructing Probability Boxes and
  Dempster-Shafer Structures", Sandia SAND2002-4015, §3.2
- Walley (1991) *Statistical Reasoning with Imprecise Probabilities* —
  natural extension for monotone functions

## Status

The two main theorems below
(`frechet_enclosure_monotone_inc_dec_nat` and the symmetric
`frechet_enclosure_monotone_inc_inc_nat`) are **fully proven** in core
Lean 4 over `Nat`, no `axiom`, no `sorry`, no Mathlib. The proofs use
only `Nat.le_trans` and the supplied monotonicity hypotheses.

The `Nat`-typed statement captures the structural content of the
theorem. Lifting to `Float` (the Sounio implementation type) requires
either a Mathlib import (`Mathlib.Order.Lattice` + `Float`-Real
correspondence) or an extension of the in-tree
`formal/Epistemic.lean` with a Mathlib-free Float order theory.
Either is a deferred milestone of its own (4-6 weeks); the
mathematical content is captured at the Nat level.
-/

namespace Sounio.Frechet

-- ================================================================
-- §1. Generic abstract version (preorder via Nat instance)
-- ================================================================

/-- Fréchet outer enclosure for a function strictly monotone with
    sign pattern (↑x, ↓y) on a rectangle.

    Given `f : Nat → Nat → Nat` monotone-increasing in the first
    argument and monotone-decreasing in the second, and any
    `(x, y) ∈ [a, b] × [c, d]`, the value `f x y` is sandwiched by
    the two diagonally opposite corners:

        f a d ≤ f x y ≤ f b c.

    This is the central claim that justifies the corner-enumeration
    Knightian band in vancomycin_pbpk.sio. The proof is two
    applications of monotonicity + transitivity, no Mathlib needed.
-/
theorem frechet_enclosure_monotone_inc_dec_nat
    (f : Nat → Nat → Nat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Nat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a d ≤ f x y ∧ f x y ≤ f b c := by
  refine ⟨?_, ?_⟩
  · exact Nat.le_trans (mono_y_dec a y d hyd) (mono_x_inc a x y hxa)
  · exact Nat.le_trans (mono_x_inc x b y hxb) (mono_y_dec b c y hyc)

/-- Symmetric version: both arguments increasing. Sharp corners are
    `(a, c)` for the lower bound and `(b, d)` for the upper bound. -/
theorem frechet_enclosure_monotone_inc_inc_nat
    (f : Nat → Nat → Nat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_inc : ∀ x y y', y ≤ y' → f x y ≤ f x y')
    (a b c d x y : Nat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a c ≤ f x y ∧ f x y ≤ f b d := by
  refine ⟨?_, ?_⟩
  · exact Nat.le_trans (mono_y_inc a c y hyc) (mono_x_inc a x y hxa)
  · exact Nat.le_trans (mono_x_inc x b y hxb) (mono_y_inc b y d hyd)

/-- Symmetric version: both arguments decreasing. Sharp corners are
    `(b, d)` for the lower bound and `(a, c)` for the upper bound. -/
theorem frechet_enclosure_monotone_dec_dec_nat
    (f : Nat → Nat → Nat)
    (mono_x_dec : ∀ x x' y, x ≤ x' → f x' y ≤ f x y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Nat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f b d ≤ f x y ∧ f x y ≤ f a c := by
  refine ⟨?_, ?_⟩
  · exact Nat.le_trans (mono_y_dec b y d hyd) (mono_x_dec x b y hxb)
  · exact Nat.le_trans (mono_x_dec a x y hxa) (mono_y_dec a c y hyc)

-- ================================================================
-- §2. Vancomycin specialisation (interface only — Float-Real
--     monotonicity proofs deferred to the future Float-Real lift).
-- ================================================================

/-- Statement of the vancomycin Cmin Fréchet enclosure as a
    *concrete instantiation* of `frechet_enclosure_monotone_inc_dec_nat`.

    The actual `Cmin : Float → Float → Float` function in the Sounio
    implementation is monotone with signs (↑Vc, ↓CL) on the
    physiological rectangle (math-review 2026-04-30, Q2). When the
    Mathlib-free Float order theory is added, this signature can be
    discharged by direct instantiation of the abstract theorem above
    with `Cmin` and the Float-version monotonicity hypotheses.

    For the present milestone, the theorem statement encodes the
    intent and serves as the canonical proof obligation. -/
def vancomycin_cmin_frechet_enclosure_obligation : Prop :=
  ∀ (cmin : Nat → Nat → Nat),
    (∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ cmin vc' cl) →
    (∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ cmin vc cl) →
    ∀ (vc_lo vc_hi cl_lo cl_hi vc cl : Nat),
    vc_lo ≤ vc → vc ≤ vc_hi →
    cl_lo ≤ cl → cl ≤ cl_hi →
    cmin vc_lo cl_hi ≤ cmin vc cl ∧ cmin vc cl ≤ cmin vc_hi cl_lo

/-- The vancomycin Cmin Fréchet enclosure obligation is satisfied —
    proven by direct instantiation of the abstract theorem. -/
theorem vancomycin_cmin_frechet_enclosure :
    vancomycin_cmin_frechet_enclosure_obligation := by
  unfold vancomycin_cmin_frechet_enclosure_obligation
  intros cmin mono_vc mono_cl vc_lo vc_hi cl_lo cl_hi vc cl
         h_vc_lo h_vc_hi h_cl_lo h_cl_hi
  exact frechet_enclosure_monotone_inc_dec_nat
          cmin mono_vc mono_cl
          vc_lo vc_hi cl_lo cl_hi vc cl
          h_vc_lo h_vc_hi h_cl_lo h_cl_hi

end Sounio.Frechet
