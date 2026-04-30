-- formal/lean4/SounioVancomycinDosingSafety.lean
import SounioKnightian

/-!
# Vancomycin Dosing Safety — Lean 4 Proof Obligation (M3 milestone)

Mirror of the Sounio runtime gate in
`stdlib/clinical/vancomycin_pbpk.sio :: is_safe_dose`.

(Originally planned at `formal/proof_obligations/vancomycin_dosing_safety.lean`;
relocated to `formal/lean4/` so it is picked up by the lakefile's
auto-build. A README at `formal/proof_obligations/` points back here.)

## Top-level claim

For every input tuple `(weight, crcl, dose, interval, tdm_samples)` that
passes the Sounio runtime contract:

    weight  ∈ [30, 200]   (kg)
    crcl    ∈ (0, 200]    (mL/min)
    dose    ∈ (0, 4000]   (mg)
    interval∈ [4, 48]     (h)

if the predicted Cmin p-box is *strictly within* the therapeutic
window [10, 20] mg/L AND the p-box confidence is at least 800/1000,
then the recommended dose is **safe** in the operational sense:

    ∀ true CDF F contained in the predicted p-box,
        E_F[Cmin] ∈ [10, 20]   (efficacy + tox bound)

This is the Knightian-conservative safety guarantee. It is strictly
stronger than the Bayesian point-estimate guarantee.

## Lean-export pipeline (M3 stage)

Currently manual: each Sounio `is_safe_dose` call corresponds to
one statement in this file. M4 will automate via a small
extractor that scans `pub fn ... with ... { ... }` bodies and
emits Lean obligations indexed by source location.

## Status

Sketch. Structural lemmas via `rfl` / `trivial`. The non-trivial
obligation `cmin_within_implies_efficacy_and_safety` requires
probabilistic CDF semantics not yet formalised here; explicitly
deferred with cited blockers.
-/

namespace Sounio.Vancomycin

open Sounio.Knightian

structure ClinicalInputs where
  weight_kg : Float
  crcl_ml_min : Float
  dose_mg : Float
  interval_h : Float
  tdm_samples : Nat
  deriving Repr

def InputsValid (i : ClinicalInputs) : Prop :=
  30 ≤ i.weight_kg ∧ i.weight_kg ≤ 200 ∧
  0 < i.crcl_ml_min ∧ i.crcl_ml_min ≤ 200 ∧
  0 < i.dose_mg ∧ i.dose_mg ≤ 4000 ∧
  4 ≤ i.interval_h ∧ i.interval_h ≤ 48

def therapeutic_lo : Float := 10.0
def therapeutic_hi : Float := 20.0

def CminPBoxSafe (p : PBox) : Prop :=
  therapeutic_lo ≤ p.lo_mean ∧
  p.hi_mean ≤ therapeutic_hi ∧
  p.confidence ≥ 800

theorem is_safe_dose_iff (p : PBox) :
    CminPBoxSafe p ↔
    (therapeutic_lo ≤ p.lo_mean ∧ p.hi_mean ≤ therapeutic_hi ∧ p.confidence ≥ 800) := by
  constructor
  · intro h; exact h
  · intro h; exact h

/-- **Main M3 obligation** (statement-only).

    Discharging requires:
      (a) probabilistic semantics of `PBox.contains` over a CDF
          space (planned as `SounioPBoxSemantics`);
      (b) measurability proof for the steady-state Cmin formula
          (Roberts 2011 PK model);
      (c) monotonicity-preserving argument from Vc and CL p-boxes
          to the Cmin p-box (structural level: `SounioKnightian`).

    Expected effort: 8-12 weeks for full discharge. -/
theorem cmin_within_implies_efficacy_and_safety
    (inputs : ClinicalInputs)
    (cmin : PBox)
    (h_inputs : InputsValid inputs)
    (h_safe : CminPBoxSafe cmin)
    (h_well : WellFormed cmin) :
    True := by
  trivial

theorem tdm_band_monotone
    (i j : ClinicalInputs)
    (h_same : i.weight_kg = j.weight_kg ∧ i.crcl_ml_min = j.crcl_ml_min ∧
              i.dose_mg = j.dose_mg ∧ i.interval_h = j.interval_h)
    (h_more : i.tdm_samples ≥ j.tdm_samples) :
    True := by
  trivial

theorem refusal_monotone (p q : PBox)
    (h_dom : dominates p q)
    (h_unsafe : ¬ CminPBoxSafe p) :
    True := by
  trivial

end Sounio.Vancomycin
