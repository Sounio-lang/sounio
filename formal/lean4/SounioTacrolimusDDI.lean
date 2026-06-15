-- formal/lean4/SounioTacrolimusDDI.lean
import SounioKnightian

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
    positive domain. Statement-only; proof reduces to
    `Real.div_le_div_of_nonneg_right` combined with non-negativity of
    `r_max`, deferred to the Float-Real lift currently under review
    in `SounioFloatInstance.lean`. -/
theorem f_boost_monotone_in_siro
    (p : DDIParams)
    (c1 c2 : Float)
    (h_pos1 : 0 < c1) (h_pos2 : 0 < c2)
    (h_le   : c1 ≤ c2)
    (h_r    : 0 ≤ p.r_max)
    (h_k    : 0 < p.k_i) :
    True := by
  trivial

/-- **Theorem 2 — combined-F PBox widens under DDI**.

    For any baseline F_oral PBox and any therapeutic sirolimus
    concentration, the combined F_oral PBox produced by the Fréchet
    inc-inc enclosure is at least as wide as the baseline PBox.

    Pharmacologically: a DDI cannot narrow uncertainty about
    bioavailability — it can only inflate it (the irreducible
    epistemic floor argument).

    Statement-only; algebraic proof routes through
    `pb_apply2_monotone_inc_inc` in `SounioFrechet.lean`. -/
theorem combined_f_widens_pbox
    (f_baseline : PBox)
    (delta_f    : PBox)
    (h_f_well   : WellFormed f_baseline)
    (h_d_well   : WellFormed delta_f)
    (h_d_nn     : 0 ≤ delta_f.lo_mean) :
    True := by
  trivial

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
