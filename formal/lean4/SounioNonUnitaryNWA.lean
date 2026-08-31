-- formal/lean4/SounioNonUnitaryNWA.lean
/-!
# NonUnitary × NarrowWidthApproximation composition — Lean 4 leaf

Mirrors the *physical approximation* effect pair encoded in Sounio:

- `NonUnitary` (bit 22) — unstable / Breit-Wigner continuum amplitudes
  (`stdlib/particle_physics/nonunitary.sio`, `nonunitary_amp.sio`)
- `NarrowWidthApproximation` (bit 24) — on-shell partial-width peak toys
  (`stdlib/particle_physics/approx_effects.sio`)

and the optional co-presence with `Perturbative` (bit 23), as in
`h_bb_width_nwa_ep` (`with NarrowWidthApproximation, Perturbative`).

## Top-level claim

1. **Handler commutation.** Discharging NonUnitary and NWA in either order
   yields the same residual effect tags (handlers are structural clears).
2. **NWA peak Nat identity.** The scaled peak formula
   `peak = C · Γ_in · Γ_out / (M² · Γ_tot²)` is well-defined when `Γ_tot > 0`
   and `M² > 0`, and the numerator/denominator form is pinned by `rfl` on Nat.
3. **Honesty non-equality.** Continuum and NWA peak are *not* definitionally
   equal — matching EXP14–19 construction-gap non-claims
   (`continuum_ne_peak_possible`).

No Mathlib. No `sorry`. Pattern follows `SounioApproxCausalKnowledge.lean`.

## Non-claims

- Not a proof of the optical theorem.
- Not a proof that Madaros preserves GUM variance (see C1 trust gate).
- Not a floating-point model of Breit-Wigner; Nat is a discrete shadow.
-/

namespace Sounio.NonUnitaryNWA

/-- Tags carried by a physical approximation computation. -/
structure ApproxTags where
  nonUnitary : Bool
  nwa : Bool
  perturbative : Bool
  deriving DecidableEq, Repr

/-- Observables used by amp→σ honesty leaves (Nat-scaled). -/
structure ResonanceToy where
  continuum : Nat
  peak : Nat
  mass2 : Nat
  gammaTot : Nat
  gammaIn : Nat
  gammaOut : Nat
  prefactor : Nat
  tags : ApproxTags
  deriving Repr

-- ================================================================
-- §1. Effect handlers (structural discharge)
-- ================================================================

def handleNonUnitary (t : ApproxTags) : ApproxTags :=
  { t with nonUnitary := false }

def handleNWA (t : ApproxTags) : ApproxTags :=
  { t with nwa := false }

def handlePerturbative (t : ApproxTags) : ApproxTags :=
  { t with perturbative := false }

def dischargeNU_NWA (t : ApproxTags) : ApproxTags :=
  handleNWA (handleNonUnitary t)

def dischargeNWA_NU (t : ApproxTags) : ApproxTags :=
  handleNonUnitary (handleNWA t)

/-- NonUnitary and NWA handlers commute. -/
theorem handler_commutativity (t : ApproxTags) :
    dischargeNU_NWA t = dischargeNWA_NU t := by
  cases t <;> rfl

/-- Triple discharge with Perturbative also independent of NU/NWA order. -/
theorem handler_commutativity_with_pert (t : ApproxTags) :
    handlePerturbative (dischargeNU_NWA t) =
    handlePerturbative (dischargeNWA_NU t) := by
  cases t <;> rfl

/-- After discharging both, neither physical tag remains. -/
theorem discharge_clears_nu_nwa (t : ApproxTags) :
    (dischargeNU_NWA t).nonUnitary = false ∧
    (dischargeNU_NWA t).nwa = false := by
  cases t <;> simp [dischargeNU_NWA, handleNWA, handleNonUnitary]

-- ================================================================
-- §2. NWA peak formula (Nat shadow)
-- ================================================================

/-- Scaled NWA peak numerator: `C · Γ_in · Γ_out`. -/
def nwaPeakNum (C gin gout : Nat) : Nat :=
  C * gin * gout

/-- Scaled NWA peak denominator: `M² · Γ_tot²`. -/
def nwaPeakDen (mass2 gtot : Nat) : Nat :=
  mass2 * gtot * gtot

/-- Peak via Nat division (0 if denominator vanishes). -/
def nwaPeak (C mass2 gin gout gtot : Nat) : Nat :=
  let den := nwaPeakDen mass2 gtot
  if den = 0 then 0 else nwaPeakNum C gin gout / den

theorem nwaPeakNum_unfolds (C gin gout : Nat) :
    nwaPeakNum C gin gout = C * gin * gout := by
  rfl

theorem nwaPeakDen_unfolds (mass2 gtot : Nat) :
    nwaPeakDen mass2 gtot = mass2 * gtot * gtot := by
  rfl

/-- Denominator positive whenever Γ_tot and M² are positive. -/
theorem nwaPeakDen_pos (mass2 gtot : Nat)
    (hm : mass2 > 0) (hg : gtot > 0) :
    nwaPeakDen mass2 gtot > 0 := by
  simp [nwaPeakDen]
  exact Nat.mul_pos (Nat.mul_pos hm hg) hg

/-- When the denominator is positive, `nwaPeak` is the truncated quotient. -/
theorem nwaPeak_eq_div (C mass2 gin gout gtot : Nat)
    (hden : nwaPeakDen mass2 gtot ≠ 0) :
    nwaPeak C mass2 gin gout gtot =
      nwaPeakNum C gin gout / nwaPeakDen mass2 gtot := by
  simp [nwaPeak, hden]

-- ================================================================
-- §3. Resonance toy well-formedness + honesty
-- ================================================================

def WellFormed (r : ResonanceToy) : Prop :=
  r.mass2 > 0 ∧
  r.gammaTot > 0 ∧
  r.prefactor > 0 ∧
  r.peak = nwaPeak r.prefactor r.mass2 r.gammaIn r.gammaOut r.gammaTot

/-- Building a toy from NWA ingredients yields a well-formed peak channel. -/
def mkNWAToy (C mass2 gin gout gtot cont : Nat) (nu : Bool) : ResonanceToy :=
  { continuum := cont
    peak := nwaPeak C mass2 gin gout gtot
    mass2 := mass2
    gammaTot := gtot
    gammaIn := gin
    gammaOut := gout
    prefactor := C
    tags := { nonUnitary := nu, nwa := true, perturbative := false } }

theorem mkNWAToy_peak_wf (C mass2 gin gout gtot cont : Nat) (nu : Bool)
    (hC : C > 0) (hm : mass2 > 0) (hg : gtot > 0) :
    WellFormed (mkNWAToy C mass2 gin gout gtot cont nu) := by
  refine ⟨hm, hg, hC, rfl⟩

/-- Continuum and peak need not coincide (construction-gap honesty). -/
theorem continuum_ne_peak_possible :
    ∃ r : ResonanceToy, WellFormed r ∧ r.continuum ≠ r.peak := by
  -- C=12, M²=1, Γ_in=Γ_out=Γ_tot=1 → peak = 12; continuum = 7
  let r := mkNWAToy 12 1 1 1 1 7 true
  have hw : WellFormed r := mkNWAToy_peak_wf 12 1 1 1 1 7 true (by decide) (by decide) (by decide)
  have hne : r.continuum ≠ r.peak := by
    simp [r, mkNWAToy, nwaPeak, nwaPeakDen, nwaPeakNum]
  exact ⟨r, hw, hne⟩

/-- Discharging tags does not alter Nat observables. -/
def dischargeToy (r : ResonanceToy) : ResonanceToy :=
  { r with tags := dischargeNU_NWA r.tags }

theorem dischargeToy_preserves_peak (r : ResonanceToy) :
    (dischargeToy r).peak = r.peak := by
  rfl

theorem dischargeToy_preserves_continuum (r : ResonanceToy) :
    (dischargeToy r).continuum = r.continuum := by
  rfl

-- ================================================================
-- §4. Handler × observable interaction (math-review follow-up)
-- ================================================================

/-! Encode `ApproxTags` as a 3-bit characteristic row and prove that
    structural discharge is successive masking that (a) clears the handled
    bits, (b) preserves the third bit, and (c) leaves Nat observables alone.
    This answers the SounioEffects-side interaction obligation without claiming
    optical theorem / Float BW content. -/

/-- Characteristic row over the three physical approximation tags. -/
structure TagRow where
  nonUnitary : Bool
  nwa : Bool
  perturbative : Bool
  deriving DecidableEq, Repr

def tagsToRow (t : ApproxTags) : TagRow :=
  { nonUnitary := t.nonUnitary, nwa := t.nwa, perturbative := t.perturbative }

def maskNU (r : TagRow) : TagRow := { r with nonUnitary := false }
def maskNWA (r : TagRow) : TagRow := { r with nwa := false }

/-- Structural handlers on tags coincide with successive TagRow masks. -/
theorem discharge_eq_mask_fold (t : ApproxTags) :
    tagsToRow (dischargeNU_NWA t) = maskNWA (maskNU (tagsToRow t)) := by
  cases t <;> rfl

/-- Commutation lifts from handlers to TagRow masks. -/
theorem tagRow_mask_comm (r : TagRow) :
    maskNWA (maskNU r) = maskNU (maskNWA r) := by
  cases r <;> rfl

/-- After NU+NWA discharge, only the perturbative bit may remain. -/
theorem discharge_clears_only_nu_nwa (t : ApproxTags) :
    (tagsToRow (dischargeNU_NWA t)).nonUnitary = false ∧
    (tagsToRow (dischargeNU_NWA t)).nwa = false ∧
    (tagsToRow (dischargeNU_NWA t)).perturbative = t.perturbative := by
  cases t <;> simp [tagsToRow, dischargeNU_NWA, handleNWA, handleNonUnitary]

/-- Full interaction: discharge preserves continuum+peak and clears NU/NWA. -/
theorem discharge_interaction (r : ResonanceToy) :
    (dischargeToy r).continuum = r.continuum ∧
    (dischargeToy r).peak = r.peak ∧
    (dischargeToy r).tags.nonUnitary = false ∧
    (dischargeToy r).tags.nwa = false ∧
    (dischargeToy r).tags.perturbative = r.tags.perturbative := by
  cases r with
  | mk cont peak mass2 gtot gin gout pref tags =>
    cases tags <;> simp [dischargeToy, dischargeNU_NWA, handleNWA, handleNonUnitary]

end Sounio.NonUnitaryNWA
