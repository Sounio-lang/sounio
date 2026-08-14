/-!
# E5 L4 — pure arithmetic: F closed under (T)+(C)

Edge **E5** of the completeness pincer is obligation (iii) of §57.50: the base case of the
unmasked deviation law

    s3(2^m) − s3(1) = 1728 · [m,3]_2      (H = 2^{m+1}).

`docs/research/zd_e5_qbinomial_mechanism_2026-08-10.md` §10 reduces E5 to four CD obligations
plus finished arithmetic. This file is that arithmetic — Lean target **L4**
(`e5_inductive_form`): the candidate

    F(m) := P(2^{m+1}) − 1728 · [m,3]_2
    P(H) := H³ − 12 H² + 28 H − 16

is closed under the transfer

    (T)+(C)  s ↦ 8 s + 24 · (−(H−2)(H−6)) − 176 + 72 H

for every integer half-dimension H (hence for every tower level H = 2^{m+1}).

## Why a 7-cleared form

    1728 · [m,3]_2 = (9/7) (H−2)(H−4)(H−8)

so the integer

    F7(H) := 7 · P(H) − 9 (H−2)(H−4)(H−8)

equals 7 · F whenever H = 2^{m+1} and the product formula for [m,3]_2 is exact
(always, as a Gaussian binomial; computationally for m ≥ 3 via Int division below).
Working with F7 avoids Int division inside the induction step.

## What this file does **not** claim

- (T) itself — the transfer of `s3` at fixed W=1 — is E4b / obligation (i).
- (C) `cp2 = −(H−2)(H−6)` on g=0 is E4a / Tier 109 (already a theorem on the tip).
- (B) `s3(3,1) = −272` is a finite enumeration (recorded here as `F_base` / `F7_base`).
- (S) the maximal-seam polynomial is S0–S4 of the E5 note (open CD content).

Mathlib-free. No `sorry`. No import of the load-bearing tip
(`SounioZDFiberAntisym.lean` — claimed by another lane).

Axioms of the kernel-clean theorems: `[propext, Classical.choice, Quot.sound]`
(Classical.choice via `grind` / `simp`, not a computational trust anchor).
`F_base` / `F7_base` use `native_decide` (finite constants).
-/

namespace SounioZDE5Inductive

/-- Maximal-seam polynomial in H = 2^{m+1}. -/
def seamPoly (H : Int) : Int := H^3 - 12 * H^2 + 28 * H - 16

/-- Gaussian gap cleared of the denominator 7:
    1728 · [m,3]_2 = (9/7)(H−2)(H−4)(H−8), so
    7 · (1728 · [m,3]_2) = 9 (H−2)(H−4)(H−8). -/
def gap7 (H : Int) : Int :=
  9 * (H - 2) * (H - 4) * (H - 8)

/-- 7-cleared candidate for s3(m, W=1):
    F7(H) = 7 · P(H) − 9 (H−2)(H−4)(H−8) = 7 · (P(H) − 1728 · [m,3]_2). -/
def F7 (H : Int) : Int :=
  7 * seamPoly H - gap7 H

/-- Inhomogeneous term of the s3-transfer under (T)+(C). -/
def transferInhom (H : Int) : Int :=
  24 * (-(H - 2) * (H - 6)) - 176 + 72 * H

/-- Transfer on an absolute s3-value under (T)+(C). -/
def transferRhs (s H : Int) : Int :=
  8 * s + transferInhom H

/-- Gaussian binomial [m choose 3]_2 via the product formula
    (Int division; exact for m ≥ 3, where 168 divides the numerator). -/
def qbin3 (m : Nat) : Int :=
  (((2 : Int)^m - 1) * ((2 : Int)^m - 2) * ((2 : Int)^m - 4)) / 168

/-- F(m) := P(2^{m+1}) − 1728 · [m,3]_2. -/
def F (m : Nat) : Int :=
  seamPoly ((2 : Int)^(m + 1)) - 1728 * qbin3 m

/-! ## L4 core — free integer H -/

/-- Poly residual of the induction step. -/
theorem poly_residual (H : Int) :
    8 * seamPoly H - seamPoly (2 * H) + transferInhom H
      = -72 * (H - 2) * (H - 4) := by
  unfold seamPoly transferInhom; grind

/-- Gaussian residual of the induction step, 7-cleared.
    8 · gap7 H − gap7 (2H) = −504 (H−2)(H−4) = 7 · (−72 (H−2)(H−4)). -/
theorem gauss_residual_cleared (H : Int) :
    8 * gap7 H - gap7 (2 * H) = -504 * (H - 2) * (H - 4) := by
  unfold gap7; grind

/-- The two residuals match at the cleared scale. -/
theorem residuals_match (H : Int) :
    7 * (8 * seamPoly H - seamPoly (2 * H) + transferInhom H)
      = 8 * gap7 H - gap7 (2 * H) := by
  rw [poly_residual, gauss_residual_cleared]; grind

/-- **L4 (`e5_inductive_form`).**
    The 7-cleared closed form F7 is closed under the transfer (T)+(C)
    for every integer H — a fortiori for every tower level H = 2^{m+1}.

    Unwinding: `F7(2H) = 8 · F7(H) + 7 · transferInhom(H)`, which is exactly
    `7 · (F(m+1) = 8 · F(m) + transferInhom(H))` once `F7 = 7 · F`. -/
theorem e5_inductive_form (H : Int) :
    F7 (2 * H) = 8 * F7 H + 7 * transferInhom H := by
  unfold F7 gap7 transferInhom seamPoly; grind

/-- H doubles along the tower: H(m+1) = 2 · H(m). -/
theorem H_succ (m : Nat) :
    (2 : Int)^(m + 1 + 1) = 2 * (2 : Int)^(m + 1) := by
  rw [Int.pow_succ, Int.mul_comm]

/-- Tower form of L4: F7 steps correctly from level m to m+1. -/
theorem e5_inductive_form_tower (m : Nat) :
    F7 ((2 : Int)^(m + 2))
      = 8 * F7 ((2 : Int)^(m + 1)) + 7 * transferInhom ((2 : Int)^(m + 1)) := by
  have h := e5_inductive_form ((2 : Int)^(m + 1))
  rwa [← H_succ m] at h

/-! ## Base case -/

/-- At m = 3, H = 16: F(3) = −272, so F7(16) = −1904. -/
theorem F7_base : F7 16 = -1904 := by native_decide

theorem F_base : F 3 = -272 := by native_decide

theorem F7_base_div : F7 16 / 7 = -272 := by native_decide

/-! ## Computational certificates (product-formula F, m ≥ 3) -/

example : F 3 = -272 := F_base
example : F 4 = -4560 := by native_decide
example : F 5 = -53072 := by native_decide
example : transferRhs (F 3) ((2 : Int)^(3 + 1)) = F 4 := by native_decide
example : transferRhs (F 4) ((2 : Int)^(4 + 1)) = F 5 := by native_decide
example : transferRhs (F 5) ((2 : Int)^(5 + 1)) = F 6 := by native_decide
example : transferRhs (F 8) ((2 : Int)^(8 + 1)) = F 9 := by native_decide
example : transferRhs (F 12) ((2 : Int)^(12 + 1)) = F 13 := by native_decide

/-- F7 equals 7 · F at the checked tower levels. -/
example : F7 16 = 7 * F 3 := by native_decide
example : F7 32 = 7 * F 4 := by native_decide
example : F7 64 = 7 * F 5 := by native_decide

/-! ## Axiom audit (kernel-clean theorems) -/

-- #print axioms e5_inductive_form
-- #print axioms e5_inductive_form_tower
-- #print axioms poly_residual
-- #print axioms gauss_residual_cleared
-- #print axioms residuals_match

end SounioZDE5Inductive
