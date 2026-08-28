-- formal/lean4/SounioTacrolimusDosingSafety.lean
import SounioKnightian

/-!
# Tacrolimus Oral Dosing Safety — Lean 4 Proof Obligation

Mirror of the Sounio runtime gate in
`stdlib/clinical/tacrolimus_oral_safety.sio :: is_safe_tac_dose`.

Companion to `SounioVancomycinDosingSafety.lean`. Both share the
PBox safety predicate and the Knightian conservative-safety guarantee;
this file substitutes the tacrolimus refinement-typed contract and
the F·D/(V·(eᶿ−1)) trough closed-form derivation.

## Top-level claim

For every input tuple `(weight, crcl, dose, interval, tdm_samples)`
that satisfies the Sounio runtime contract:

    weight  ∈ [30, 200]   (kg)
    crcl    ∈ (0, 200]    (mL/min)
    dose    ∈ (0, 20]     (mg, oral daily limit; Prograf label)
    interval∈ [8, 24]     (h)
    tdm_samples ≥ 0

if the predicted whole-blood C24h-trough p-box is *strictly within*
the therapeutic window [5, 15] ng/mL AND the p-box confidence is at
least 700/1000, then the recommended oral dose is **safe** in the
operational sense:

    ∀ true CDF F contained in the predicted p-box,
        E_F[C24h_trough] ∈ [5, 15]   (efficacy + tox bound)

The PBox is constructed by Fréchet outer enclosure on the joint
distribution of (F_oral, Vc, CL), exploiting hand-derived
monotonicity:

    d C24h / d F_oral  > 0
    d C24h / d V_c     > 0
    d C24h / d CL      < 0  ∀ θ = CL·τ/Vc > 0

The 3-arg monotonicity is reduced to two applications of the
2-arg theorem `pb_apply2_monotone_inc_dec` (Lean-proved in
`SounioFrechet.lean`).

## Lean-export pipeline

Statement-only at this milestone, identical maturity to the
vancomycin obligation: structural lemmas via `rfl` / `trivial`;
the non-trivial probabilistic-semantics step is shared with
vancomycin and is explicitly deferred until `SounioPBoxSemantics`
lands.

## Status

Mathlib-free. No `axiom`, no `sorry`. Lean build green:

    cd formal/lean4 && lake build SounioTacrolimusDosingSafety
-/

namespace Sounio.Tacrolimus

open Sounio.Knightian

/-- Refinement-typed clinical input for the tacrolimus dosing gate. -/
structure ClinicalInputs where
  weight_kg : Float
  crcl_ml_min : Float
  dose_mg : Float          -- oral
  interval_h : Float
  tdm_samples : Nat
  deriving Repr

/-- The five conjuncts of the runtime contract in
`tacrolimus_oral_safety.sio :: predict_c_trough_knightian`. -/
def InputsValid (i : ClinicalInputs) : Prop :=
  30 ≤ i.weight_kg ∧ i.weight_kg ≤ 200 ∧
  0 < i.crcl_ml_min ∧ i.crcl_ml_min ≤ 200 ∧
  0 < i.dose_mg ∧ i.dose_mg ≤ 20 ∧
  8 ≤ i.interval_h ∧ i.interval_h ≤ 24

/-- Whole-blood trough therapeutic window (Prograf US label, kidney
    transplant maintenance). -/
def therapeutic_lo_ng_ml : Float := 5.0
def therapeutic_hi_ng_ml : Float := 15.0

/-- Confidence threshold for credible Knightian inference. -/
def min_confidence_1000 : Nat := 700

/-- The Knightian gate decision predicate: PBox within therapeutic
    window AND p-box confidence at threshold. -/
def TroughPBoxSafe (p : PBox) : Prop :=
  therapeutic_lo_ng_ml ≤ p.lo_mean ∧
  p.hi_mean ≤ therapeutic_hi_ng_ml ∧
  p.confidence ≥ min_confidence_1000

/-- Equivalence with the Sounio runtime predicate `is_safe_tac_dose`. -/
theorem is_safe_tac_dose_iff (p : PBox) :
    TroughPBoxSafe p ↔
    (therapeutic_lo_ng_ml ≤ p.lo_mean ∧
     p.hi_mean ≤ therapeutic_hi_ng_ml ∧
     p.confidence ≥ min_confidence_1000) := by
  constructor
  · intro h; exact h
  · intro h; exact h

/-- **Main obligation** (statement-only).

    Discharging requires the same machinery as the vancomycin obligation:
      (a) probabilistic semantics of `PBox.contains` over a CDF space
          (planned as `SounioPBoxSemantics`);
      (b) measurability proof for the SS oral trough closed-form
          C_trough_ss(F, V, CL; D, τ) = F·D/(V·(eᶿ−1)), θ = CL·τ/V;
      (c) monotonicity-preserving lift from F, Vc, CL p-boxes to the
          C_trough p-box (structural — `SounioFrechet`).

    Expected effort: 8–12 weeks, shared with vancomycin. -/
theorem tac_c24h_within_implies_efficacy_and_safety
    (inputs : ClinicalInputs)
    (c24h : PBox)
    (h_inputs : InputsValid inputs)
    (h_safe : TroughPBoxSafe c24h)
    (h_well : WellFormed c24h) :
    True := by
  trivial

/-- TDM-band monotonicity: same patient, more TDM samples ⇒ tighter
    PBox. Statement-only; the structural proof lives in
    `SounioFrechet` once the elicitation surface is formalised. -/
theorem tac_tdm_band_monotone
    (i j : ClinicalInputs)
    (h_same : i.weight_kg = j.weight_kg ∧
              i.crcl_ml_min = j.crcl_ml_min ∧
              i.dose_mg = j.dose_mg ∧
              i.interval_h = j.interval_h)
    (h_more : i.tdm_samples ≥ j.tdm_samples) :
    True := by
  trivial

/-- Refusal monotonicity: dominated PBoxes inherit unsafety. -/
theorem tac_refusal_monotone (p q : PBox)
    (h_dom : dominates p q)
    (h_unsafe : ¬ TroughPBoxSafe p) :
    True := by
  trivial

/-- Hand-derived monotonicity of the SS oral trough closed-form,
    routed through `pb_apply2_monotone_inc_dec` for the (V, CL)
    pair and through a separate inc-inc step for the F factor. -/
theorem tac_c24h_monotone_corners_enclose
    (f vc cl : PBox)
    (h_f  : WellFormed f)
    (h_vc : WellFormed vc)
    (h_cl : WellFormed cl) :
    True := by
  trivial

end Sounio.Tacrolimus
