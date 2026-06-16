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
    positive domain. Statement-only at the Float level: the Float→ℚ
    bridge covers only `add_rne_bound` and `mul_rne_bound`; there is
    no `div_rne_bound` axiom, so the Float-level monotonicity for
    `p.r_max * c_siro / (p.k_i + c_siro)` cannot be bridged directly.

    **STATUS: Float-level statement remains `: True` (no div bridge axiom).**

    The genuine ℚ content is discharged in the supporting lemma
    `fBoostR_cross_mul_le` below (pure ℚ, no axioms). That lemma proves
    the cross-multiplied form: `r·c1·(k+c2) ≤ r·c2·(k+c1)`, which is
    the algebraic core of Michaelis-Menten monotonicity. The Float
    division form would follow from a future `Float.div_rne_bound` axiom
    (Phase 2 work). -/
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

/-- **Theorem 2 — combined-F PBox widens under DDI**.

    For any baseline F_oral PBox and any therapeutic sirolimus
    concentration, the combined F_oral PBox produced by the Fréchet
    inc-inc enclosure is at least as wide as the baseline PBox.

    Pharmacologically: a DDI cannot narrow uncertainty about
    bioavailability — it can only inflate it (the irreducible
    epistemic floor argument).

    **STATUS: Float-level statement remains `: True` (no Float div bridge).**

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
