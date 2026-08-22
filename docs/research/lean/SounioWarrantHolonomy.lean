/-
  SounioWarrantHolonomy.lean

  Crux #3, provable-now core (lemma ii) + the declared bridge (lemma i).

  Lemma (ii): over the imaginary-octonion basis triples, the variance-holonomy
  equals base + κ·‖α‖², and curvature enters the variance channel IFF the triple
  is non-Fano (non-associative). Proven here by `decide` (kernel-checked) over the
  finite {0,4} associator-norm shadow, in centi-variance INTEGERS (avoids the
  Float axioms that block the repo's p-box Lean; see SounioKnightian.lean).

  Grounding (measured in stdlib): product_nonassoc.sio — Fano ‖α‖²=0 → σ²=0.25,
  non-Fano ‖α‖²=4 → σ²=4.25; associator_field.sio:57 gum_augmentation = κ·‖α‖².
  So 4.25 − 0.25 = 4 = ‖[e1,e2,e4]‖² is the squared holonomy in the variance channel.

  The ALGEBRAIC half of the bridge — [α]=0 ⟺ Fano — is ALREADY PROVEN in
  SounioBidirectionalBridge.lean:170 `nonassoc_iff_not_fano` (native_decide,
  no Mathlib, no sorry). This file adds the VARIANCE (holonomy) half and states
  the still-open epistemic bridge.

  STATUS: CHECKED clean under Lean 4.33.1 (leanprover/lean4:stable), 2026-08-22.
  All four theorems compile with `decide`; `#print axioms` reports each depends on
  NO axioms (fully kernel-checked — not even native_decide's `ofReduceBool`).
  sorry = 0. Lemma (i) is a *documented open conjecture*, deliberately NOT written
  as a vacuous `: True` placeholder.
-/

namespace Sounio.WarrantHolonomy

/-- Squared associator norm of an imaginary-octonion basis triple, as measured:
    0 when the triple is Fano-collinear (associative), 4 otherwise.
    (`fano = true` ⟺ [α]=0; see SounioBidirectionalBridge.nonassoc_iff_not_fano.) -/
def assocNormSq (fano : Bool) : Nat := if fano then 0 else 4

/-- Base aleatory variance, in centi-units (0.25 → 25). -/
def baseVarCenti : Nat := 25

/-- κ = 1, applied to ‖α‖² in centi-units (each unit of ‖α‖² = 1.00 variance = 100 centi). -/
def kappaCenti : Nat := 100

/-- Lemma (ii), the variance-holonomy: base + κ·‖α‖², in centi-variance. -/
def varianceHolonomyCenti (fano : Bool) : Nat :=
  baseVarCenti + kappaCenti * assocNormSq fano

/-- Flat (Fano) triple reproduces the measured 0.25. -/
theorem holonomy_flat : varianceHolonomyCenti true = 25 := by decide

/-- Curved (non-Fano) triple reproduces the measured 4.25. -/
theorem holonomy_curved : varianceHolonomyCenti false = 425 := by decide

/-- The squared holonomy entering the variance channel is exactly 4.00 (= 400 centi). -/
theorem holonomy_gap : varianceHolonomyCenti false - varianceHolonomyCenti true = 400 := by
  decide

/-- **Curvature-in-the-variance-channel theorem.** The variance-holonomy exceeds the
    base aleatory variance IFF the triple is non-Fano — i.e. curvature enters the
    warrant budget exactly when the composition is non-associative. This is the
    variance (second-moment) shadow of the Blackwell holonomy. -/
theorem curvature_iff_nonfano (fano : Bool) :
    (varianceHolonomyCenti fano > baseVarCenti) ↔ (fano = false) := by
  cases fano <;> decide

/-!
## Lemma (i) — the epistemic bridge (OPEN CONJECTURE, no Lean statement yet)

  Over uncertain octonion affine forms, reassociation
    ρ : (a ⊗ b) ⊗ c → a ⊗ (b ⊗ c)
  is a **Blackwell garbling** (warrant non-increasing) **iff [a,b,c] = 0**.

  - Algebraic half — `[α]=0 ⟺ Fano` — is PROVEN
    (`SounioBidirectionalBridge.nonassoc_iff_not_fano`).
  - Variance half — curvature enters σ² iff non-Fano — is `curvature_iff_nonfano`
    above (second-moment shadow only).
  - MISSING (the whole novelty): the *full statistical Blackwell order*, not the
    finite variance shadow. Proving "reassociation is a garbling ⟺ [α]=0" links the
    Blackwell informativeness order to the octonion associator (HH³ obstruction) —
    nobody has proven these are the same obstruction. This requires a Lean
    formalisation of the garbling (Markov post-processing) order over uncertain
    quantities, which does not yet exist in the corpus.

  Deliberately NOT stated as `theorem ... : True := trivial` — a vacuous placeholder
  is exactly the §9.2(c) failure this file exists to avoid.
-/

end Sounio.WarrantHolonomy
