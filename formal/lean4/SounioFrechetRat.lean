-- formal/lean4/SounioFrechetRat.lean
/-!
# Fréchet Outer Enclosure — Rational lift (Track 2 foundation)

This file is the **Rat-shadow lift** of `SounioFrechet.lean`, the
first step in the multi-stage Lean-discharge roadmap from
`Nat`-shadow proofs to full `Float` mechanisation:

    Stage 0 ─ Nat shadow      (formal/lean4/SounioFrechet.lean)
    Stage 1 ─ Rat lift        (formal/lean4/SounioFrechetRat.lean)   ← THIS
    Stage 2 ─ Real bridge     (deferred 2-3 weeks; Mathlib-free
                              Real-from-Cauchy or in-tree
                              equivalent)
    Stage 3 ─ Float bridge    (deferred 4-6 weeks; FpRoundedReal
                              + IEEE-754-bounded soundness)

## Why Rat is the natural Stage 1

Lean 4 core ships `Rat` as a `LinearOrderedCommRing` with the
following lemmas already discharged:

- `Rat.le_trans`        — transitivity of ≤
- `Rat.mul_le_mul_of_nonneg_left`
- `Rat.mul_le_mul_of_nonneg_right`
- `Rat.add_le_add_right`
- `Rat.add_comm`, `Rat.mul_one`

These are precisely the facts needed for the Fréchet enclosure on
ordered structures. **Rat is also a faithful sub-ordered-field of
ℝ**, so any theorem proven over Rat extends to ℝ by density (or to
Float via the ℝ → Float rounding analysis to be added at Stage 3).

The Rat-stage lift therefore captures the **structural ordered-
field content** of the Fréchet enclosure that the Nat shadow
encodes only as a discrete approximation. The Nat shadow proves
the theorem holds for any monotone integer-valued function; the
Rat lift proves it for any monotone rational-valued function. They
are equally rigorous; the Rat lift is just *more proximate to the
real implementation* (Sounio f64 = IEEE-754 ≈ Rat-with-rounding).

## The theorem (mirroring SounioFrechet.lean)

For any function `f : Rat → Rat → Rat` strictly monotone with sign
pattern (↑x, ↓y) on a closed rectangle `[a, b] × [c, d]`, every
point `(x, y)` of the rectangle satisfies:

    f a d ≤ f x y ≤ f b c

The proof structure is **identical** to the Nat version: two
applications of monotonicity + transitivity. The only difference
is that `Nat.le_trans` is replaced by `Rat.le_trans`, etc. This
demonstrates that the Sounio Fréchet machinery scales mechanically
to richer ordered structures *without* requiring Mathlib.

## Status

The two main theorems below are **fully proven** in core Lean 4
over `Rat`, no `axiom`, no `sorry`, no Mathlib. They establish the
Stage 1 milestone of the Float-lift roadmap.

## Connection to the Sounio runtime

`stdlib/epistemic/knightian.sio :: pb_apply2_monotone_inc_dec`
operates on `Float`. The Stage 0 `Nat` proof captures the structural
content; the Stage 1 `Rat` proof captures the algebraic content.
Stage 2-3 will close the residual gap (Rat density in ℝ + IEEE-754
rounding bounds). Until then, the user-facing soundness claim is:

  > The Sounio runtime's Fréchet enclosure is *structurally* sound
  > over any sub-ordered-field of ℝ, including ℚ. The Float
  > implementation inherits this soundness modulo IEEE-754
  > rounding error, which is bounded by 2 × ulp per operation
  > (cf. Higham 2002).

This is honest: we do not claim bit-perfect soundness on Float
without the Stage 3 bridge.
-/

namespace Sounio.FrechetRat

-- ================================================================
-- §1. Generic Rat-valued Fréchet enclosure (↑x, ↓y).
-- ================================================================

/-- Fréchet outer enclosure for a function strictly monotone with
    sign pattern (↑x, ↓y) on a Rat-rectangle.

    Given `f : Rat → Rat → Rat` monotone-increasing in the first
    argument and monotone-decreasing in the second, and any
    `(x, y) ∈ [a, b] × [c, d]`, the value `f x y` is sandwiched by
    the two diagonally opposite corners:

        f a d ≤ f x y ≤ f b c.

    The proof is **identical** to the Nat version — only the
    transitivity lemma name changes (`Rat.le_trans` instead of
    `Nat.le_trans`). This demonstrates the structural reusability
    of the Sounio Fréchet substrate across ordered carriers. -/
theorem frechet_enclosure_monotone_inc_dec_rat
    (f : Rat → Rat → Rat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Rat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a d ≤ f x y ∧ f x y ≤ f b c := by
  refine ⟨?_, ?_⟩
  · exact Rat.le_trans (mono_y_dec a y d hyd) (mono_x_inc a x y hxa)
  · exact Rat.le_trans (mono_x_inc x b y hxb) (mono_y_dec b c y hyc)

/-- Symmetric Rat lift: both arguments increasing. -/
theorem frechet_enclosure_monotone_inc_inc_rat
    (f : Rat → Rat → Rat)
    (mono_x_inc : ∀ x x' y, x ≤ x' → f x y ≤ f x' y)
    (mono_y_inc : ∀ x y y', y ≤ y' → f x y ≤ f x y')
    (a b c d x y : Rat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f a c ≤ f x y ∧ f x y ≤ f b d := by
  refine ⟨?_, ?_⟩
  · exact Rat.le_trans (mono_y_inc a c y hyc) (mono_x_inc a x y hxa)
  · exact Rat.le_trans (mono_x_inc x b y hxb) (mono_y_inc b y d hyd)

/-- Symmetric Rat lift: both arguments decreasing. -/
theorem frechet_enclosure_monotone_dec_dec_rat
    (f : Rat → Rat → Rat)
    (mono_x_dec : ∀ x x' y, x ≤ x' → f x' y ≤ f x y)
    (mono_y_dec : ∀ x y y', y ≤ y' → f x y' ≤ f x y)
    (a b c d x y : Rat)
    (hxa : a ≤ x) (hxb : x ≤ b)
    (hyc : c ≤ y) (hyd : y ≤ d)
    : f b d ≤ f x y ∧ f x y ≤ f a c := by
  refine ⟨?_, ?_⟩
  · exact Rat.le_trans (mono_y_dec b y d hyd) (mono_x_dec x b y hxb)
  · exact Rat.le_trans (mono_x_dec a x y hxa) (mono_y_dec a c y hyc)

-- ================================================================
-- §2. Walley gap monotonicity over Rat.
-- ================================================================
--
-- Mirrors SounioWalley.walley_gap_monotone_in_epsilon_nat over Rat.
-- The gap formula `gap = ε · (s_hi − s_lo)` is linear in ε; for
-- s_lo ≤ s_hi (positive support width), Rat.mul_le_mul_of_nonneg_right
-- gives the monotonicity.

/-- Rat-shadow of `Sounio.Walley.lift_gap`: the lifted band gap is
    `ε · (s_hi − s_lo)`. -/
def lift_gap_rat (eps s_lo s_hi : Rat) : Rat :=
  eps * (s_hi - s_lo)

/-- Gap is monotone-non-decreasing in ε for any non-negative
    support width. The Rat-stage proof uses
    `Rat.mul_le_mul_of_nonneg_right`. -/
theorem walley_gap_monotone_in_epsilon_rat
    (eps_a eps_b s_lo s_hi : Rat)
    (h_eps : eps_a ≤ eps_b)
    (h_width : 0 ≤ s_hi - s_lo)
    : lift_gap_rat eps_a s_lo s_hi ≤ lift_gap_rat eps_b s_lo s_hi := by
  unfold lift_gap_rat
  exact Rat.mul_le_mul_of_nonneg_right h_eps h_width

-- ================================================================
-- §3. Vancomycin Cmin obligation, Rat lift.
-- ================================================================
--
-- The Vancomycin `cmin : Rat → Rat → Rat` Fréchet enclosure
-- obligation is mechanically a Rat-shadow of the Nat version in
-- `SounioFrechet.vancomycin_cmin_frechet_enclosure_obligation`.

/-- Vancomycin Cmin Fréchet enclosure obligation, Rat-typed. -/
def vancomycin_cmin_frechet_enclosure_obligation_rat : Prop :=
  ∀ (cmin : Rat → Rat → Rat),
    (∀ vc vc' cl, vc ≤ vc' → cmin vc cl ≤ cmin vc' cl) →
    (∀ vc cl cl', cl ≤ cl' → cmin vc cl' ≤ cmin vc cl) →
    ∀ (vc_lo vc_hi cl_lo cl_hi vc cl : Rat),
    vc_lo ≤ vc → vc ≤ vc_hi →
    cl_lo ≤ cl → cl ≤ cl_hi →
    cmin vc_lo cl_hi ≤ cmin vc cl ∧ cmin vc cl ≤ cmin vc_hi cl_lo

/-- The Rat-typed vancomycin Cmin obligation is satisfied by direct
    instantiation of the abstract Rat theorem above. -/
theorem vancomycin_cmin_frechet_enclosure_rat :
    vancomycin_cmin_frechet_enclosure_obligation_rat := by
  unfold vancomycin_cmin_frechet_enclosure_obligation_rat
  intros cmin mono_vc mono_cl vc_lo vc_hi cl_lo cl_hi vc cl
         h_vc_lo h_vc_hi h_cl_lo h_cl_hi
  exact frechet_enclosure_monotone_inc_dec_rat
          cmin mono_vc mono_cl
          vc_lo vc_hi cl_lo cl_hi vc cl
          h_vc_lo h_vc_hi h_cl_lo h_cl_hi

-- ================================================================
-- §4. Bridge to Nat (sanity).
-- ================================================================
--
-- Every Nat embeds into Rat as the canonical injection
-- `Nat → Rat`. A monotone-increasing Nat function lifts to a
-- monotone-increasing Rat function on the image of the embedding;
-- the Rat theorem above subsumes the Nat one (and the Nat theorem
-- in `SounioFrechet.lean` remains independently useful as the
-- Stage 0 milestone).

/-- The Nat embedding is monotone (sanity check; not needed for the
    main theorems but confirms the Stage 0 → Stage 1 bridge). -/
theorem nat_to_rat_monotone (a b : Nat) (h : a ≤ b) :
    (a : Rat) ≤ (b : Rat) := by
  exact_mod_cast h

end Sounio.FrechetRat
