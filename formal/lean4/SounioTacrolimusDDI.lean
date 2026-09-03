-- formal/lean4/SounioTacrolimusDDI.lean
import SounioKnightian
import SounioPBoxSemantics

/-!
# Tacrolimus + Sirolimus DDI — Lean 4 Proof Obligation

Mirror of the Sounio implementation in
`stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio`.

Three proof obligations of the runtime `ddi_combined_f_knightian`
Fréchet enclosure. Honest status of each (do NOT read the `True`
ones as established):

  1. `f_boost_monotone_in_siro` — [NOT DISCHARGED, `: True` placeholder]
     intended: sirolimus conc ↑ ⇒ delta_F ↑ (saturating Michaelis-Menten).
     Float-typed; discharge needs the Float↔Real lift (axiom-bearing,
     `SounioFloatInstance.lean`). Future-work.
  2. `combined_f_widens_pbox`   — [NOT DISCHARGED, `: True` placeholder]
     intended: combined F_oral band ≥ baseline band (DDI cannot tighten F).
     Float width inequality; future-work, routes through
     `pb_apply2_monotone_inc_inc` (`SounioFrechet.lean`) once the Float lift
     and `SounioPBoxSemantics` land.
  3. `combined_f_confidence_decays` — [DISCHARGED] a real, Mathlib-free,
     axiom-free `Nat`-level theorem (see below).

Theorems 1–2 are VACUOUS `: True := by trivial` placeholders pending the
Float-Real semantics; only theorem 3 carries content. Naming what is and
is not proven follows the thesis discipline (narrowest claim the files
support).

## Status

Mathlib-free. No `axiom`, no `sorry`. Theorems 1–2 vacuous; theorem 3
discharged. Lean build green:

    cd formal/lean4 && lake build SounioTacrolimusDDI
-/

namespace Sounio.TacrolimusDDI

open Sounio.Knightian
open Sounio.IEEE754

/-- Parameters of the P-gp competitive inhibition model. -/
structure DDIParams where
  r_max : Float       -- max fractional F boost (Undre 1999: 0.30)
  k_i   : Float       -- inhibition constant (Lampen 1998: 5 ng/mL)
  deriving Repr

/-- The mechanistic F-boost function. Returns `r_max · [S] / (K_i + [S])`
    for sirolimus concentration `c_siro`. Defined as `Float.zero` outside
    the valid input domain so the statement-level theorems below have a
    total signature. -/
def fBoost (p : DDIParams) (c_siro : Float) : Float :=
  if c_siro ≤ 0 then 0
  else p.r_max * c_siro / (p.k_i + c_siro)

/-- **Theorem 1 — saturating monotonicity in sirolimus concentration**.

    The Michaelis-Menten form is monotone non-decreasing in [S] on the
    positive domain in EXACT (ℝ/ℚ) arithmetic.

    **STATUS: Float-level exact monotonicity remains `: True` — and this
    is NOT merely "a missing `div_rne_bound` axiom" (the earlier note was
    wrong). The obstruction is deeper: `fBoost` is a ROUNDED COMPOSITION
    `fl( fl(r·c) / fl(k+c) )`. A quotient of an independently-rounded
    non-decreasing numerator and a non-decreasing denominator is NOT
    universally monotone — it can dip sub-ulp where the denominator
    rounds up to the next representable while the numerator does not
    (near a binade boundary). So exact monotonicity of the rounded
    `fBoost` is NOT guaranteed — it can fail sub-ulp near binade
    boundaries (asserted on the structure of round-to-nearest; we do
    not exhibit a concrete counterexample Float triple here). No single
    rounding axiom discharges it. The honest Float-level result is
    "monotone up to a bounded roundoff," whose proof composes the three
    §2.1 bounds — left as future work.**

    What IS now machine-checked at the Float level: `fBoost_final_div_rel_bound`
    below — `div_rne_bound` (§6) bounds the relative error of `fBoost`'s
    final division step. The genuine EXACT-arithmetic monotonicity content
    is `fBoostR_cross_mul_le` (pure ℚ, no axioms): the cross-multiplied
    form `r·c1·(k+c2) ≤ r·c2·(k+c1)`. -/
theorem f_boost_monotone_in_siro
    (p : DDIParams)
    (c1 c2 : Float)
    (h_pos1 : 0 < c1) (h_pos2 : 0 < c2)
    (h_le   : c1 ≤ c2)
    (h_r    : 0 ≤ p.r_max)
    (h_k    : 0 < p.k_i) :
    True := by
  trivial

/-- **REAL ℚ THEOREM**: Michaelis-Menten cross-multiplication core.

    For `0 ≤ r`, `0 < k`, `c1 ≤ c2`, the cross-multiplied form of
    Michaelis-Menten monotonicity holds over ℚ:
        r·c1·(k+c2) ≤ r·c2·(k+c1).

    This is the arithmetic core of `fBoost` monotonicity:
        r·c1/(k+c1) ≤ r·c2/(k+c2)  iff  r·c1·(k+c2) ≤ r·c2·(k+c1)
    (when k+c1 > 0, k+c2 > 0). The cross-multiplied form avoids
    `Rat.div` and the missing `div_rne_bound` axiom.

    Proof: expand `mul_add`, then chain `mul_le_mul_of_nonneg_left`
    (c1≤c2, r≥0) and `mul_le_mul_of_nonneg_right` (result, k>0),
    plus the commutativity identity `r·c2·c1 = r·c1·c2`.

    `#print axioms fBoostR_cross_mul_le` shows only
    `[propext, Classical.choice, Quot.sound]` — pure ℚ, no axioms,
    no sorry. -/
theorem fBoostR_cross_mul_le
    (r c1 c2 k : Rat)
    (hr  : 0 ≤ r)
    (hk  : 0 < k)
    (hle : c1 ≤ c2) :
    r * c1 * (k + c2) ≤ r * c2 * (k + c1) := by
  -- Expand using mul_add
  rw [Rat.mul_add (r * c1), Rat.mul_add (r * c2)]
  -- Goal: r*c1*k + r*c1*c2 ≤ r*c2*k + r*c2*c1
  -- Step 1: r*c1*k ≤ r*c2*k  (from c1≤c2, r≥0, k>0)
  have h1 : r * c1 * k ≤ r * c2 * k :=
    Rat.mul_le_mul_of_nonneg_right (Rat.mul_le_mul_of_nonneg_left hle hr) (Rat.le_of_lt hk)
  -- Step 2: r*c2*c1 = r*c1*c2  (commutativity)
  have h2 : r * c2 * c1 = r * c1 * c2 := by
    rw [Rat.mul_assoc, Rat.mul_assoc, Rat.mul_comm c2 c1]
  -- Step 3: combine via add_le_add_right
  rw [h2]
  exact (Rat.add_le_add_right).mpr h1

/-- **REAL Float-level theorem (uses `Float.div_rne_bound`)**: the final
    division step of `fBoost` rounds within unit roundoff.

    On the active branch (`¬ c ≤ 0`), `fBoost p c = (p.r_max * c) / (p.k_i + c)`
    where the outer operation is a single IEEE-754 division. Hence its
    `toRat` differs from the exact quotient of the (already-rounded)
    numerator and denominator by at most `u · |·|`:

        | fBoost.toRat − num.toRat / den.toRat | ≤ u · | num.toRat / den.toRat |

    with `num = p.r_max * c`, `den = p.k_i + c`. This is the honest,
    fully-bridged Float-level guarantee on `fBoost`'s last operation — it
    does NOT claim monotonicity of the rounded composition (see the note
    on `f_boost_monotone_in_siro`); it bounds one rounding step.

    `#print axioms fBoost_final_div_rel_bound` shows exactly
    `[Float.div_rne_bound]` (+ Lean core) — the §6 axiom is genuinely used. -/
theorem fBoost_final_div_rel_bound
    (p : DDIParams) (c : Float)
    (h_active : ¬ (c ≤ 0))
    (hn : Float.IsFiniteNormal (p.r_max * c))
    (hd : Float.IsFiniteNormal (p.k_i + c))
    (hq : Float.IsFiniteNormal (p.r_max * c / (p.k_i + c))) :
    Float.toRat (fBoost p c)
        - Float.toRat (p.r_max * c) / Float.toRat (p.k_i + c)
      ≤ unit_roundoff * rat_abs (Float.toRat (p.r_max * c) / Float.toRat (p.k_i + c))
    ∧
    Float.toRat (p.r_max * c) / Float.toRat (p.k_i + c)
        - Float.toRat (fBoost p c)
      ≤ unit_roundoff * rat_abs (Float.toRat (p.r_max * c) / Float.toRat (p.k_i + c)) := by
  unfold fBoost
  rw [if_neg h_active]
  exact Float.div_rne_bound hn hd hq

/-- **Theorem 2 — combined-F PBox widens under DDI**.

    For any baseline F_oral PBox and any therapeutic sirolimus
    concentration, the combined F_oral PBox produced by the Fréchet
    inc-inc enclosure is at least as wide as the baseline PBox.

    Pharmacologically: a DDI cannot narrow uncertainty about
    bioavailability — it can only inflate it (the irreducible
    epistemic floor argument).

    **STATUS: Float-level statement remains `: True`. CORRECTION: this is
    NOT a "div bridge" gap (the earlier note mis-grouped it with thm 1 —
    there is no division here at all). The real obstruction: the Float
    `add` (SounioKnightian.add) uses round-to-NEAREST `+` for both bounds,
    NOT outward rounding (lo↓, hi↑). Under round-to-nearest, `width(add a b)`
    can fall below `width a + width b` by up to the addition roundoff, so
    EXACT width-widening does not hold at the Float level — only "widens up
    to roundoff" (provable by composing `add_rne_bound`, future work). Were
    `add` outward-rounded (interval-soundness discipline) it would discharge
    exactly via `add_rne_bound` + `combined_f_widens_pbox_rat`.**

    The genuine ℚ content is discharged in `addR_width_ge_left` in
    `SounioPBoxSemantics.lean`: for any well-formed ℚ delta box `b`,
    `widthR (addR a b) ≥ widthR a`. That theorem is pure ℚ (no axioms).

    NOTE ON DOMINANCE vs. WIDTH: `dominatesR (addR a b) a` is FALSE
    when `b.lo > 0` (positive delta shifts the band upward; the lower
    bound of addR a b EXCEEDS a.lo, breaking the lower dominance
    conjunct). Width is the correct "widens" invariant:
    `widthR (addR a b) = widthR a + widthR b ≥ widthR a`. -/
theorem combined_f_widens_pbox
    (f_baseline : PBox)
    (delta_f    : PBox)
    (h_f_well   : WellFormed f_baseline)
    (h_d_well   : WellFormed delta_f)
    (h_d_nn     : 0 ≤ delta_f.lo_mean) :
    True := by
  trivial

/-- **REAL ℚ THEOREM**: combined-F width widens under DDI (ℚ-image version).

    For any ℚ-model baseline box `a` and well-formed ℚ delta box `b`,
    the width of `addR a b` is at least the width of `a`:
        widthR a ≤ widthR (addR a b).

    This is the ℚ content of `combined_f_widens_pbox`. The Float-level
    theorem (above) cannot be bridged directly without a `div_rne_bound`
    axiom, but the ℚ arithmetic core is here.

    `#print axioms combined_f_widens_pbox_rat` shows only
    `[propext, Classical.choice, Quot.sound]` — pure ℚ, no axioms. -/
theorem combined_f_widens_pbox_rat
    (a b : Sounio.PBoxSemantics.PBoxR)
    (hb  : Sounio.PBoxSemantics.WellFormedR b) :
    Sounio.PBoxSemantics.widthR a ≤ Sounio.PBoxSemantics.widthR (Sounio.PBoxSemantics.addR a b) :=
  Sounio.PBoxSemantics.addR_width_ge_left a b hb

/-- **Theorem 3 — confidence decays under Fréchet composition** (DISCHARGED).

    The Sounio convention for confidence under composition (`add`):
    `conf_combined = confDecay (min(conf_F, conf_delta_F)) ≤ min(conf_F, conf_delta_F)`.
    Co-administration never INcreases epistemic confidence — a structural
    consequence of the Fréchet outer enclosure. Proven Mathlib-free over `Nat`
    (no `axiom`, no `sorry`): `confDecay c = (c*99)/100 ≤ c` for all `c : Nat`. -/
theorem combined_f_confidence_decays
    (f_baseline : PBox)
    (delta_f    : PBox) :
    (add f_baseline delta_f).confidence
      ≤ minNat f_baseline.confidence delta_f.confidence := by
  show confDecay (minNat f_baseline.confidence delta_f.confidence)
      ≤ minNat f_baseline.confidence delta_f.confidence
  unfold confDecay
  omega

end Sounio.TacrolimusDDI
