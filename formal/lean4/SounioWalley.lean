-- formal/lean4/SounioWalley.lean
/-!
# Walley ε-Contamination Credal Set Soundness (M3.5)

This file mirrors the Sounio module
`stdlib/epistemic/walley.sio` and provides a Mathlib-free Lean
discharge of three structural theorems that justify the
`credal_to_pbox` lift used at the Walley elicitation surface.

## The lift

A credal set `M_ε(P0, S) := { (1−ε)·P0 + ε·Q : supp(Q) ⊆ S }`
parametrised by:
- nominal mean μ_0
- contamination radius ε ∈ [0, 1]
- support [s_lo, s_hi]

is lifted to a mean p-box by:

    lo_mean = (1 − ε)·μ_0 + ε·s_lo
    hi_mean = (1 − ε)·μ_0 + ε·s_hi
    gap     = hi_mean − lo_mean = ε·(s_hi − s_lo)

## Theorems mechanised here

1. **walley_collapse_at_zero_nat** — at ε = 0 the lifted lo/hi
   collapse to a single point μ_0. This is the structural guarantee
   that the Walley elicitation surface *strictly extends* the GUM-
   precise `Knowledge<f64>` representation: when there is no
   Knightian ambiguity, the surface degenerates to the precise case.

2. **walley_vacuous_at_one_nat** — at ε = 1 the lifted band is the
   entire support [s_lo, s_hi]. This is total Knightian ignorance
   conditional only on the support — the canonical "novel patient
   never seen" prior.

3. **walley_gap_monotone_in_epsilon_nat** — the lifted band gap is
   monotone-non-decreasing in ε. More information (smaller ε) ⇒
   tighter band; less information (larger ε) ⇒ wider band. This is
   the soundness invariant that makes ε an honest knob: increasing
   ε can never strengthen a posterior.

## Composition with M2.5 Fréchet enclosure

Theorem 3 (`walley_gap_monotone_in_epsilon_nat`) composes with the
Fréchet enclosure in `SounioFrechet.lean` as follows:

    f monotone-in-each-arg on support rect  ⟹
    Fréchet[f] on credal_to_support_pbox  ⟹
    enclosure of f(X, Y) for any joint dependence

Note that this composition operates on the **support pbox**
(`credal_to_support_pbox`), not the **mean pbox** (`credal_to_pbox`).
For nonlinear monotone f, propagating means via Fréchet does *not*
enclose realised values of f(X, Y); see math-review 2026-04-30 Q4
and `stdlib/epistemic/walley.sio` header for the discussion.

## Proof technique

Same `Nat`-shadow strategy as `SounioFrechet.lean`: the structural
mathematical content is captured in core Lean 4 with:

- discrete weights for ε (numerator/denominator pair)
- `Nat`-level arithmetic for monotonicity
- no Mathlib, no `axiom`, no `sorry`

Lifting to `Float` (the Sounio implementation type) requires a
Mathlib-free Float-Real ordered-field theory (deferred milestone).

## Status

All three theorems below are **fully proven** over `Nat`. The
discrete-ε encoding (`eps_num`, `eps_den` with `eps_num ≤ eps_den`)
admits any rational ε ∈ [0, 1] without loss of generality.
-/

namespace Sounio.Walley

-- ================================================================
-- §1. Discrete-weight encoding of the Walley lift.
-- ================================================================
--
-- We encode ε ∈ [0, 1] as a pair (eps_num, eps_den) of `Nat` with
-- eps_num ≤ eps_den. The lifted lo/hi are unnormalised by the
-- common factor eps_den; the gap is *exactly* eps_num · (s_hi − s_lo).
--
-- The unnormalised representation avoids `Nat` division and lets
-- the structural theorems be `Nat.mul_*` arguments.

/-- Unnormalised lower mean: `(eps_den − eps_num)·μ + eps_num·s_lo`. -/
def lift_lo (eps_num eps_den μ s_lo : Nat) : Nat :=
  (eps_den - eps_num) * μ + eps_num * s_lo

/-- Unnormalised upper mean: `(eps_den − eps_num)·μ + eps_num·s_hi`. -/
def lift_hi (eps_num eps_den μ s_hi : Nat) : Nat :=
  (eps_den - eps_num) * μ + eps_num * s_hi

/-- Unnormalised band gap: `lift_hi − lift_lo`. -/
def lift_gap (eps_num s_lo s_hi : Nat) : Nat :=
  eps_num * (s_hi - s_lo)

-- ================================================================
-- §2. Theorem 1: collapse at ε = 0.
-- ================================================================

/-- At `eps_num = 0`, the lifted lower and upper means coincide
    with `eps_den · μ`. The band collapses to a single point. -/
theorem walley_collapse_at_zero_nat
    (eps_den μ s_lo s_hi : Nat)
    : lift_lo 0 eps_den μ s_lo = eps_den * μ
    ∧ lift_hi 0 eps_den μ s_hi = eps_den * μ := by
  refine ⟨?_, ?_⟩
  · unfold lift_lo
    simp [Nat.sub_zero, Nat.zero_mul, Nat.add_zero]
  · unfold lift_hi
    simp [Nat.sub_zero, Nat.zero_mul, Nat.add_zero]

/-- At ε = 0, the lifted gap is identically zero, regardless of
    support width. This is the precise/Knowledge-recovery point. -/
theorem walley_collapse_gap_zero_nat
    (s_lo s_hi : Nat) : lift_gap 0 s_lo s_hi = 0 := by
  unfold lift_gap
  simp [Nat.zero_mul]

-- ================================================================
-- §3. Theorem 2: vacuous at ε = 1.
-- ================================================================
--
-- We encode ε = 1 by `eps_num = eps_den = N` for any N ≥ 1.

/-- At `eps_num = eps_den`, the lifted lower mean equals
    `eps_den · s_lo` (μ has zero weight). -/
theorem walley_vacuous_lo_at_one_nat
    (eps_den μ s_lo : Nat)
    : lift_lo eps_den eps_den μ s_lo = eps_den * s_lo := by
  unfold lift_lo
  simp [Nat.sub_self, Nat.zero_mul, Nat.zero_add]

/-- At `eps_num = eps_den`, the lifted upper mean equals
    `eps_den · s_hi`. -/
theorem walley_vacuous_hi_at_one_nat
    (eps_den μ s_hi : Nat)
    : lift_hi eps_den eps_den μ s_hi = eps_den * s_hi := by
  unfold lift_hi
  simp [Nat.sub_self, Nat.zero_mul, Nat.zero_add]

/-- At ε = 1, the lifted gap is the full support width
    `eps_den · (s_hi − s_lo)`. -/
theorem walley_vacuous_gap_at_one_nat
    (eps_den s_lo s_hi : Nat)
    : lift_gap eps_den s_lo s_hi = eps_den * (s_hi - s_lo) := by
  unfold lift_gap
  rfl

-- ================================================================
-- §4. Theorem 3: gap monotone-non-decreasing in ε.
-- ================================================================

/-- For any two ε-values `eps_num1 ≤ eps_num2`, the lifted gap is
    monotone-non-decreasing.

    More Knightian uncertainty (larger ε) ⇒ wider band. This is the
    soundness invariant of Walley elicitation: increasing ε can
    never tighten the resulting p-box.

    Proof: `lift_gap eps s_lo s_hi = eps · (s_hi − s_lo)` is linear
    in `eps`, and `Nat.mul_le_mul_right` gives the monotonicity. -/
theorem walley_gap_monotone_in_epsilon_nat
    (eps_num1 eps_num2 s_lo s_hi : Nat)
    (h : eps_num1 ≤ eps_num2)
    : lift_gap eps_num1 s_lo s_hi ≤ lift_gap eps_num2 s_lo s_hi := by
  unfold lift_gap
  exact Nat.mul_le_mul_right (s_hi - s_lo) h

-- ================================================================
-- §5. Composition obligation with Fréchet (interface).
-- ================================================================
--
-- The Walley → Fréchet composition is captured as a Prop-level
-- obligation that the support-band lift, fed into the Fréchet
-- enclosure, encloses any value of f(X, Y) for any joint
-- distribution with marginals in M_ε(C_X) × M_ε(C_Y).
--
-- The actual proof is direct: `credal_to_support_pbox` produces
-- the support rectangle, and `SounioFrechet` already closes the
-- bound. We state the obligation here for documentation/audit.

/-- Composition obligation: for monotone-(↑x, ↓y) `f`, the
    Fréchet enclosure on the support rectangle of two credal sets
    encloses `f x y` for every (x, y) with x ∈ supp(C_X) and
    y ∈ supp(C_Y). This is direct instantiation of
    `Sounio.Frechet.frechet_enclosure_monotone_inc_dec_nat` with the
    support endpoints — no Walley-specific argument needed,
    because the support pbox discards μ_0 and ε. -/
def walley_frechet_composition_obligation : Prop :=
  ∀ (f : Nat → Nat → Nat),
    (∀ x x' y, x ≤ x' → f x y ≤ f x' y) →
    (∀ x y y', y ≤ y' → f x y' ≤ f x y) →
    ∀ (s_x_lo s_x_hi s_y_lo s_y_hi x y : Nat),
    s_x_lo ≤ x → x ≤ s_x_hi →
    s_y_lo ≤ y → y ≤ s_y_hi →
    f s_x_lo s_y_hi ≤ f x y ∧ f x y ≤ f s_x_hi s_y_lo

/-- The Walley/Fréchet composition obligation is satisfied. The
    proof reduces to `Sounio.Frechet.frechet_enclosure_monotone_inc_dec_nat`
    with the support endpoints as the rectangle. -/
theorem walley_frechet_composition_holds :
    walley_frechet_composition_obligation := by
  unfold walley_frechet_composition_obligation
  intros f mono_x mono_y s_x_lo s_x_hi s_y_lo s_y_hi x y
         hxa hxb hyc hyd
  refine ⟨?_, ?_⟩
  · exact Nat.le_trans (mono_y s_x_lo y s_y_hi hyd) (mono_x s_x_lo x y hxa)
  · exact Nat.le_trans (mono_x x s_x_hi y hxb) (mono_y s_x_hi s_y_lo y hyc)

end Sounio.Walley
