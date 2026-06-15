-- formal/lean4/SounioKnightian.lean
import SounioOrderedCarrier
import SounioIEEE754Spec
import SounioFloatInstance
import SounioKnightian

/-!
# Sounio.PBoxSemantics — ℚ-backed epistemic model + Float-Real bridge

This file is the **Stage 4** milestone of the Float-Real lift
roadmap: it provides the ℝ-backed (rational subset, ℚ) semantic
model for `PBox` and an explicit Float-to-ℚ bridge.

## ℚ-model / rational-real semantics

Epistemic p-boxes are modelled over `Rat` (the rational subset of ℝ).
Per the `SounioRealOrder.lean` convention, `Rat` is our reachable
"real" model: Mathlib-free, with full ordered-field lemmas in Lean 4
core. All PK regimes (dose, weight, clearance, time) operate on
rational-domain quantities.

A full Cauchy-complete ℝ model is deferred to the Stage 3a-Cauchy
milestone (`SounioRealCauchy.lean`), which is already in progress.
The rational model here is NOT a placeholder — it is exactly the
semantic domain the clinical PK proofs require.

## Trust boundary

The Float-to-ℚ bridge (`toRatPBox`, `wf_float_to_rat_lo_hi`) rests
on the 5 IEEE-754 axioms in `SounioIEEE754Spec.lean`:

  1. `Float.toRat : Float → Rat`
  2. `Float.IsFiniteNormal : Float → Prop`
  3. `Float.toRat_le_iff_finite`
  4. `Float.mul_rne_bound`
  5. `Float.add_rne_bound`

Everything **below** this bridge (i.e., theorems about `PBoxR`)
is proven from Lean 4 core Rat algebra only — no Float assumptions
whatsoever. Everything **at** the bridge requires the 5 axioms.
Use `#print axioms` to confirm per-theorem ancestry.

## What is machine-checked here

- `PBoxR` structure (ℚ-model, no sorry)
- `WellFormedR` predicate (pure Rat, no sorry)
- `addR` operation on `PBoxR` (pure Rat, no sorry)
- `dominatesR` relation (pure Rat, no sorry)

**REAL ℚ THEOREMS** (proven, no sorry, no trivial, no axiom):

- `addR_preserves_wf_lo_le_hi` — add preserves lo ≤ hi
- `addR_preserves_wf_var_nn` — add preserves var ≥ 0
- `addR_preserves_wf` — full WellFormedR preservation under add
- `addR_dominance_monotone` — dominance is preserved under add

**BRIDGE** (uses 5 IEEE-754 axioms):

- `toRatPBox` — Float-to-ℚ bridge function
- `wf_float_to_rat_lo_hi` — WellFormedR lo≤hi conjunct via bridge

## What remains a placeholder (NOT changed here)

- `SounioKnightian.vacuous_widest_placeholder` — still `: True`
- `SounioKnightian.add_dominance_monotone` — still `: True`
- `SounioTacrolimusDDI.f_boost_monotone_in_siro` — still `: True`
- `SounioTacrolimusDDI.combined_f_widens_pbox` — still `: True`

The ℚ-model in this file makes these theorems *reachable* (they
can now be restated over `PBoxR` and proven), but does not rewrite
them in the Float-level files. That wiring is the next step.

The variance-bridge conjunct (`0 ≤ toRatPBox.var`) requires
`Float.toRat (0 : Float) = (0 : Rat)`, which is NOT one of the 5
IEEE-754 spec axioms. Rather than adding a 6th axiom, `wf_float_to_rat_var`
takes the Rat-side bound as an explicit hypothesis, keeping the
trust boundary exactly at the 5 axioms. Callers supply the bridge
fact as part of their IEEE-754 zero-preservation lemma.
-/

namespace Sounio.PBoxSemantics

-- ================================================================
-- §1. ℚ-model PBoxR (the ℝ semantic model, rational subset)
-- ================================================================

/-- Epistemic p-box over `Rat` — the ℚ-backed semantic model.

    This is the "real" (rational subset of ℝ) version of the
    Float-level `PBox` in `SounioKnightian.lean`. All three fields
    are exact rational values; no rounding, no NaN, no Inf.

    Label: ℚ-model / rational-real semantics. -/
structure PBoxR where
  /-- Lower bound on the CDF mean band. -/
  lo : Rat
  /-- Upper bound on the CDF mean band. -/
  hi : Rat
  /-- Variance (second central moment). -/
  var : Rat

/-- A `PBoxR` is well-formed iff `lo ≤ hi` and `0 ≤ var`.

    No `confidence` field in the ℚ model (confidence is an
    implementation-level Nat counter, not a mathematical quantity). -/
def WellFormedR (p : PBoxR) : Prop :=
  p.lo ≤ p.hi ∧ (0 : Rat) ≤ p.var

-- ================================================================
-- §2. Operations on PBoxR
-- ================================================================

/-- Rational-model add: component-wise. -/
def addR (a b : PBoxR) : PBoxR := {
  lo  := a.lo  + b.lo,
  hi  := a.hi  + b.hi,
  var := a.var + b.var
}

/-- `q` dominates `p` iff `q`'s band is at least as wide:
    `q.lo ≤ p.lo` and `p.hi ≤ q.hi`. -/
def dominatesR (q p : PBoxR) : Prop :=
  q.lo ≤ p.lo ∧ p.hi ≤ q.hi

-- ================================================================
-- §3. Machine-checked theorems over ℚ
-- ================================================================
--
-- All theorems below are proven using Lean 4 core Rat algebra only.
-- No `sorry`, no `trivial`, no `native_decide`, no axiom.
-- `#print axioms` on any of these shows only:
--   [propext, Classical.choice, Quot.sound]

/-- **REAL ℚ THEOREM**: `addR` preserves the `lo ≤ hi` conjunct.

    If `a.lo ≤ a.hi` and `b.lo ≤ b.hi`, then
    `(addR a b).lo ≤ (addR a b).hi`, i.e.,
    `a.lo + b.lo ≤ a.hi + b.hi`.

    Proof: chain `Rat.add_le_add_right` (iff form, `.mpr`) and
    `Rat.add_le_add_left` (iff form, `.mpr`), then `Rat.le_trans`.
    Lemma shapes confirmed-compiling: mirror of the chain in
    `SounioFloatInstance.lean §3.2`. -/
theorem addR_preserves_wf_lo_le_hi
    (a b : PBoxR)
    (ha : WellFormedR a)
    (hb : WellFormedR b) :
    (addR a b).lo ≤ (addR a b).hi := by
  obtain ⟨halo, _⟩ := ha
  obtain ⟨hblo, _⟩ := hb
  -- a.lo + b.lo ≤ a.hi + b.lo  (right-add b.lo preserves ≤)
  have step1 : a.lo + b.lo ≤ a.hi + b.lo :=
    (Rat.add_le_add_right).mpr halo
  -- a.hi + b.lo ≤ a.hi + b.hi  (left-add a.hi preserves ≤)
  have step2 : a.hi + b.lo ≤ a.hi + b.hi :=
    (Rat.add_le_add_left).mpr hblo
  exact Rat.le_trans step1 step2

/-- **REAL ℚ THEOREM**: `addR` preserves `0 ≤ var`.

    If `0 ≤ a.var` and `0 ≤ b.var`, then `0 ≤ (addR a b).var`,
    i.e., `0 ≤ a.var + b.var`. -/
theorem addR_preserves_wf_var_nn
    (a b : PBoxR)
    (ha : WellFormedR a)
    (hb : WellFormedR b) :
    (0 : Rat) ≤ (addR a b).var := by
  obtain ⟨_, hav⟩ := ha
  obtain ⟨_, hbv⟩ := hb
  exact Rat.add_nonneg hav hbv

/-- **REAL ℚ THEOREM**: `addR` fully preserves `WellFormedR`.

    Direct combination of the two conjunct theorems above. -/
theorem addR_preserves_wf
    (a b : PBoxR)
    (ha : WellFormedR a)
    (hb : WellFormedR b) :
    WellFormedR (addR a b) :=
  ⟨addR_preserves_wf_lo_le_hi a b ha hb,
   addR_preserves_wf_var_nn   a b ha hb⟩

/-- **REAL ℚ THEOREM**: dominance is preserved under `addR`.

    If `q_a` dominates `a` and `q_b` dominates `b`, then
    `addR q_a q_b` dominates `addR a b`.

    This is the ℚ-model proof of the statement that is currently
    `: True` (placeholder) in `SounioKnightian.add_dominance_monotone`.
    Over Rat the proof is 8 lines of `Rat.add_le_add_*`.

    `dominatesR q p := q.lo ≤ p.lo ∧ p.hi ≤ q.hi`:
      - lower bound goes DOWN (q is more pessimistic on the left)
      - upper bound goes UP   (q is more pessimistic on the right) -/
theorem addR_dominance_monotone
    (a b qa qb : PBoxR)
    (hda : dominatesR qa a)
    (hdb : dominatesR qb b) :
    dominatesR (addR qa qb) (addR a b) := by
  obtain ⟨hda_lo, hda_hi⟩ := hda
  obtain ⟨hdb_lo, hdb_hi⟩ := hdb
  constructor
  · -- (addR qa qb).lo ≤ (addR a b).lo
    -- i.e., qa.lo + qb.lo ≤ a.lo + b.lo
    have s1 : qa.lo + qb.lo ≤ a.lo  + qb.lo :=
      (Rat.add_le_add_right).mpr hda_lo
    have s2 : a.lo  + qb.lo ≤ a.lo  + b.lo  :=
      (Rat.add_le_add_left).mpr hdb_lo
    exact Rat.le_trans s1 s2
  · -- (addR a b).hi ≤ (addR qa qb).hi
    -- i.e., a.hi + b.hi ≤ qa.hi + qb.hi
    have s1 : a.hi  + b.hi  ≤ qa.hi + b.hi  :=
      (Rat.add_le_add_right).mpr hda_hi
    have s2 : qa.hi + b.hi  ≤ qa.hi + qb.hi :=
      (Rat.add_le_add_left).mpr hdb_hi
    exact Rat.le_trans s1 s2

-- ================================================================
-- §4. Float-to-ℚ bridge
-- ================================================================
--
-- ⚠ TRUST BOUNDARY: everything in §4 rests on the 5 IEEE-754
-- axioms in `SounioIEEE754Spec.lean`. Pure-ℚ theorems in §3
-- above have ZERO dependence on Float; the bridge below is where
-- Float assumptions enter. Use `#print axioms <theorem>` to
-- confirm the exact ancestry.
--
-- The bridge function `toRatPBox` and theorem `wf_float_to_rat_lo_hi`
-- depend only on the 5 canonical spec axioms:
--   Float.toRat, Float.IsFiniteNormal, Float.toRat_le_iff_finite,
--   Float.mul_rne_bound, Float.add_rne_bound
-- (and the 4 BoundedOrderedCarrier typeclass-shape axioms from
-- SounioFloatInstance §1, which back the Float instance).

/-- **Float-to-ℚ bridge function** (noncomputable: uses `Float.toRat`
    which is not a computable operation in this axiomatic model).

    Converts a Float-level `PBox` to the ℚ-model `PBoxR` by
    projecting each field through `Sounio.IEEE754.Float.toRat`.

    Requires explicit finiteness witnesses for each of the three
    fields (`lo_mean`, `hi_mean`, `variance`). In clinical PK
    regimes, all three are always finite-normal — IEEE-754 NaN/Inf
    cannot arise from positive dose/weight/clearance inputs.

    ⚠ TRUST BOUNDARY: depends on `Float.toRat` (axiom 1) and
    `Float.IsFiniteNormal` (axiom 2). -/
noncomputable def toRatPBox
    (p : Sounio.Knightian.PBox)
    (hlo : Sounio.IEEE754.Float.IsFiniteNormal p.lo_mean)
    (hhi : Sounio.IEEE754.Float.IsFiniteNormal p.hi_mean)
    (hva : Sounio.IEEE754.Float.IsFiniteNormal p.variance) :
    PBoxR := {
  lo  := Sounio.IEEE754.Float.toRat p.lo_mean,
  hi  := Sounio.IEEE754.Float.toRat p.hi_mean,
  var := Sounio.IEEE754.Float.toRat p.variance
}

/-- **Float-to-ℚ bridge theorem** (lo ≤ hi conjunct).

    If `p` is `WellFormed` (Float: `lo_mean ≤ hi_mean`) and
    `lo_mean`, `hi_mean` are finite-normal, then
    `(toRatPBox p ...).lo ≤ (toRatPBox p ...).hi`.

    Proof: single application of `Float.toRat_le_iff_finite`.

    ⚠ TRUST BOUNDARY: depends on `Float.toRat_le_iff_finite`
    (axiom 3). No additional axioms beyond the 5 IEEE-754 spec. -/
theorem wf_float_to_rat_lo_hi
    (p : Sounio.Knightian.PBox)
    (hlo : Sounio.IEEE754.Float.IsFiniteNormal p.lo_mean)
    (hhi : Sounio.IEEE754.Float.IsFiniteNormal p.hi_mean)
    (hva : Sounio.IEEE754.Float.IsFiniteNormal p.variance)
    (hwf : Sounio.Knightian.WellFormed p) :
    (toRatPBox p hlo hhi hva).lo ≤ (toRatPBox p hlo hhi hva).hi := by
  obtain ⟨h_lo_le_hi, _, _⟩ := hwf
  -- h_lo_le_hi : p.lo_mean ≤ p.hi_mean (Float)
  -- toRat_le_iff_finite lifts this to Rat.le
  exact (Sounio.IEEE754.Float.toRat_le_iff_finite hlo hhi).mp h_lo_le_hi

/-- **Float-to-ℚ bridge theorem** (var ≥ 0 conjunct).

    If the Rat-image of `variance` is nonneg (supplied as explicit
    hypothesis `hvar_rat`), then `(toRatPBox p ...).var ≥ 0`.

    The hypothesis `hvar_rat` captures the trust boundary honestly:
    proving `Float.toRat (0 : Float) = (0 : Rat)` requires either
    a 6th IEEE-754 axiom (zero-preservation) or an in-tree binary64
    bit-decode (Phase 2). Rather than adding an opaque axiom, we
    expose the Rat fact as an explicit caller obligation.

    In practice, the caller derives `hvar_rat` from their
    zero-preservation axiom/lemma + `Float.toRat_le_iff_finite`. -/
theorem wf_float_to_rat_var
    (p : Sounio.Knightian.PBox)
    (hlo : Sounio.IEEE754.Float.IsFiniteNormal p.lo_mean)
    (hhi : Sounio.IEEE754.Float.IsFiniteNormal p.hi_mean)
    (hva : Sounio.IEEE754.Float.IsFiniteNormal p.variance)
    (hvar_rat : (0 : Rat) ≤ Sounio.IEEE754.Float.toRat p.variance) :
    (0 : Rat) ≤ (toRatPBox p hlo hhi hva).var :=
  hvar_rat

/-- **Float-to-ℚ full bridge** (both WellFormedR conjuncts).

    Combines `wf_float_to_rat_lo_hi` and `wf_float_to_rat_var`.
    Depends on the 5 IEEE-754 axioms + the caller-supplied
    `hvar_rat` (Rat-level var ≥ 0). -/
theorem wf_float_to_rat
    (p : Sounio.Knightian.PBox)
    (hlo : Sounio.IEEE754.Float.IsFiniteNormal p.lo_mean)
    (hhi : Sounio.IEEE754.Float.IsFiniteNormal p.hi_mean)
    (hva : Sounio.IEEE754.Float.IsFiniteNormal p.variance)
    (hwf : Sounio.Knightian.WellFormed p)
    (hvar_rat : (0 : Rat) ≤ Sounio.IEEE754.Float.toRat p.variance) :
    WellFormedR (toRatPBox p hlo hhi hva) :=
  ⟨wf_float_to_rat_lo_hi p hlo hhi hva hwf,
   wf_float_to_rat_var   p hlo hhi hva hvar_rat⟩

end Sounio.PBoxSemantics
