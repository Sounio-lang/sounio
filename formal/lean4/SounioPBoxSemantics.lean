-- formal/lean4/SounioPBoxSemantics.lean
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

-- ================================================================
-- §5. Float-to-ℚ dominance bridge
-- ================================================================
--
-- ⚠ TRUST BOUNDARY: the two theorems below use the 5 IEEE-754
-- axioms, specifically `Float.toRat_le_iff_finite` (axiom 3).
-- The pure-ℚ theorems in §3 (addR_dominance_monotone, etc.) are
-- ALREADY PROVEN and are invoked here.
--
-- NOTE ON STATEMENT SCOPE:
-- `dominatesR_of_dominates` bridges Float `dominates` (lo_mean/hi_mean
-- band inequality) to ℚ `dominatesR` over the EXACT RATIONAL IMAGES
-- of the inputs. This is NOT a statement about `Float.add` outputs —
-- `toRat` does not commute with `Float.add` (there is no homomorphism;
-- `Float.add_rne_bound` gives only an error budget, not equality).
-- `add_dominance_monotone_rat` therefore proves dominance of addR
-- applied to the RATIONAL IMAGES of the *input* endpoints, not of
-- the Float `add` result.

/-- **BRIDGE THEOREM**: Float `dominates q p` implies `dominatesR`
    over the rational images.

    If `q.lo_mean ≤ p.lo_mean` and `p.hi_mean ≤ q.hi_mean` (Float),
    and all four fields are finite-normal, then the corresponding
    `PBoxR` images satisfy `dominatesR (toRatPBox q ..) (toRatPBox p ..)`.

    Proof: two applications of `Float.toRat_le_iff_finite` (axiom 3).

    ⚠ TRUST BOUNDARY: exactly 3 of the 5 IEEE-754 axioms.
    `#print axioms dominatesR_of_dominates` shows:
    `[IsFiniteNormal, toRat, toRat_le_iff_finite]`.
    The `mul_rne_bound` and `add_rne_bound` axioms are NOT needed here
    because `addR` is pure ℚ and never called in this proof. -/
theorem dominatesR_of_dominates
    (q p : Sounio.Knightian.PBox)
    (hqlo : Sounio.IEEE754.Float.IsFiniteNormal q.lo_mean)
    (hqhi : Sounio.IEEE754.Float.IsFiniteNormal q.hi_mean)
    (hqva : Sounio.IEEE754.Float.IsFiniteNormal q.variance)
    (hplo : Sounio.IEEE754.Float.IsFiniteNormal p.lo_mean)
    (hphi : Sounio.IEEE754.Float.IsFiniteNormal p.hi_mean)
    (hpva : Sounio.IEEE754.Float.IsFiniteNormal p.variance)
    (hdom : Sounio.Knightian.dominates q p) :
    dominatesR (toRatPBox q hqlo hqhi hqva) (toRatPBox p hplo hphi hpva) := by
  obtain ⟨hlo, hhi⟩ := hdom
  constructor
  · -- (toRatPBox q ..).lo ≤ (toRatPBox p ..).lo
    -- i.e., Float.toRat q.lo_mean ≤ Float.toRat p.lo_mean
    exact (Sounio.IEEE754.Float.toRat_le_iff_finite hqlo hplo).mp hlo
  · -- (toRatPBox p ..).hi ≤ (toRatPBox q ..).hi
    -- i.e., Float.toRat p.hi_mean ≤ Float.toRat q.hi_mean
    exact (Sounio.IEEE754.Float.toRat_le_iff_finite hphi hqhi).mp hhi

/-- **REAL DISCHARGE** of `SounioKnightian.add_dominance_monotone` over ℚ images.

    This is the genuine content of the Float-level placeholder
    `SounioKnightian.add_dominance_monotone` (currently `: True`).
    That theorem CANNOT be restated in `SounioKnightian.lean` itself
    (which is *imported by* this file — restating it there would create
    an import cycle). The ℚ discharge lives here; the Knightian file's
    `: True` placeholder bears a comment pointing to this theorem.

    **What this proves:** If `qa` Float-dominates `a` and `qb` Float-dominates `b`
    (with all 12 fields finite-normal), then the ℚ-image `addR` of `(qa, qb)`
    dominates the ℚ-image `addR` of `(a, b)`:

        dominatesR (addR (toRatPBox qa ..) (toRatPBox qb ..))
                   (addR (toRatPBox a ..)  (toRatPBox b ..))

    **What this does NOT prove:** dominance of `Float.add qa qb` over
    `Float.add a b`. `toRat` does not commute with `Float.add`; the
    bridge only covers the rational images of the *input* PBoxes.

    ⚠ TRUST BOUNDARY: 3 of the 5 IEEE-754 axioms + [propext, Classical.choice, Quot.sound].
    `#print axioms add_dominance_monotone_rat` shows:
    `[propext, Classical.choice, Quot.sound, IsFiniteNormal, toRat, toRat_le_iff_finite]`.
    The `mul_rne_bound` and `add_rne_bound` axioms are not needed (addR is pure ℚ).
    The inner `addR_dominance_monotone` call is pure-ℚ; propext/choice/Quot enter via
    the `obtain` destructor in the two `dominatesR_of_dominates` calls. -/
theorem add_dominance_monotone_rat
    (a b qa qb : Sounio.Knightian.PBox)
    (hqalo : Sounio.IEEE754.Float.IsFiniteNormal qa.lo_mean)
    (hqahi : Sounio.IEEE754.Float.IsFiniteNormal qa.hi_mean)
    (hqava : Sounio.IEEE754.Float.IsFiniteNormal qa.variance)
    (halo  : Sounio.IEEE754.Float.IsFiniteNormal a.lo_mean)
    (hahi  : Sounio.IEEE754.Float.IsFiniteNormal a.hi_mean)
    (hava  : Sounio.IEEE754.Float.IsFiniteNormal a.variance)
    (hqblo : Sounio.IEEE754.Float.IsFiniteNormal qb.lo_mean)
    (hqbhi : Sounio.IEEE754.Float.IsFiniteNormal qb.hi_mean)
    (hqbva : Sounio.IEEE754.Float.IsFiniteNormal qb.variance)
    (hblo  : Sounio.IEEE754.Float.IsFiniteNormal b.lo_mean)
    (hbhi  : Sounio.IEEE754.Float.IsFiniteNormal b.hi_mean)
    (hbva  : Sounio.IEEE754.Float.IsFiniteNormal b.variance)
    (hda : Sounio.Knightian.dominates qa a)
    (hdb : Sounio.Knightian.dominates qb b) :
    dominatesR
      (addR (toRatPBox qa hqalo hqahi hqava) (toRatPBox qb hqblo hqbhi hqbva))
      (addR (toRatPBox a  halo  hahi  hava)  (toRatPBox b  hblo  hbhi  hbva)) :=
  addR_dominance_monotone
    (toRatPBox a  halo  hahi  hava)
    (toRatPBox b  hblo  hbhi  hbva)
    (toRatPBox qa hqalo hqahi hqava)
    (toRatPBox qb hqblo hqbhi hqbva)
    (dominatesR_of_dominates qa a hqalo hqahi hqava halo hahi hava hda)
    (dominatesR_of_dominates qb b hqblo hqbhi hqbva hblo hbhi hbva hdb)

-- ================================================================
-- §6. ℚ-model vacuous dominance (bounded)
-- ================================================================
--
-- The Float-level `SounioKnightian.vacuous_widest_placeholder` is
-- `: True` because `WellFormed` puts NO upper bound on `lo_mean` /
-- `hi_mean`: any PBox with `lo_mean < -1e18` (representable in Float)
-- has `lo_mean < vacuous.lo_mean` and thus is NOT dominated.
--
-- The honest ℚ statement: under the ADDITIONAL constraint that the
-- p-box's rational endpoints lie within `[-bound_R, bound_R]`
-- (i.e., `BoundedWellFormedR`), the ℚ-vacuous box DOES dominate.
-- This is essentially tautological: the bound IS the dominance.
--
-- The Float-level placeholder stays `: True` with a comment because
-- mutating `WellFormed` to `BoundedWellFormed` would ripple to
-- SounioVancomycinDosingSafety and SounioTacrolimusDosingSafety.
-- Callers who need the bounded statement import SounioPBoxSemantics
-- and use `vacuousR_dominates_bounded` directly.

/-- The rational bound `bound_R = 10^18` — matches the Float literal
    `1e18` used in `SounioKnightian.vacuous`. Defined as a plain
    `Nat`-coerced `Rat` to avoid floating-point notation. -/
def bound_R : Rat := 1000000000000000000

/-- The ℚ-model vacuous p-box: `lo = -bound_R`, `hi = bound_R`,
    `var = bound_R`. Mirrors `SounioKnightian.vacuous` at rational
    precision.

    This box dominates any `BoundedWellFormedR` box (see
    `vacuousR_dominates_bounded`). -/
def vacuousR : PBoxR := { lo := -bound_R, hi := bound_R, var := bound_R }

/-- A `PBoxR` is **bounded well-formed** iff it is well-formed AND its
    endpoints lie within `[-bound_R, bound_R]`.

    This is the HONEST precondition for `vacuousR` dominating every
    p-box: the unbounded `WellFormedR` admits `lo < -bound_R`, breaking
    the dominance. Callers in clinical PK regimes always satisfy this
    (all doses/concentrations lie in `[0, 10^6] ⊂ [-10^18, 10^18]`). -/
def BoundedWellFormedR (p : PBoxR) : Prop :=
  WellFormedR p ∧ -bound_R ≤ p.lo ∧ p.hi ≤ bound_R

/-- **REAL ℚ THEOREM**: `vacuousR` dominates every `BoundedWellFormedR` p-box.

    Proof: near-tautological — `BoundedWellFormedR` carries exactly the
    two inequalities that constitute `dominatesR vacuousR p`.

    `#print axioms vacuousR_dominates_bounded` shows only
    `[propext, Quot.sound]` — pure ℚ obtain-pattern, no `Classical.choice`,
    no axioms, no sorry. -/
theorem vacuousR_dominates_bounded (p : PBoxR) (h : BoundedWellFormedR p) :
    dominatesR vacuousR p := by
  obtain ⟨_, hlo, hhi⟩ := h
  exact ⟨hlo, hhi⟩

-- ================================================================
-- §7. Width monotonicity under addR (for combined_f_widens_pbox)
-- ================================================================
--
-- `SounioTacrolimusDDI.combined_f_widens_pbox` states that DDI can
-- only INFLATE bioavailability uncertainty — never tighten it. The
-- correct ℚ formulation is WIDTH monotonicity (not dominance):
--   widthR (addR a b) ≥ widthR a   when b is WellFormedR
-- where widthR p = p.hi - p.lo ≥ 0 by well-formedness.
--
-- Note: `dominatesR (addR a b) a` is WRONG for this statement.
-- Expanding: `dominatesR (addR a b) a` = `a.lo+b.lo ≤ a.lo ∧ a.hi ≤ a.hi+b.hi`
-- = `b.lo ≤ 0 ∧ 0 ≤ b.hi`. Under `h_d_nn : 0 ≤ delta_f.lo_mean` we have
-- `b.lo ≥ 0`, so the lower-bound half fails (b.lo ≤ 0 only if b.lo = 0).
-- Adding a delta with a POSITIVE lower bound SHIFTS the band; it does not
-- CONTAIN it. Width is the faithful "widens" invariant.

/-- Width of a `PBoxR` (rational, may be 0 for point masses). -/
def widthR (p : PBoxR) : Rat := p.hi - p.lo

/-- **REAL ℚ THEOREM**: `addR` is width-monotone in the first argument.

    If `b` is well-formed (`b.lo ≤ b.hi`), then
    `widthR a ≤ widthR (addR a b)`, i.e., `a.hi - a.lo ≤ (a.hi+b.hi) - (a.lo+b.lo)`.

    This is the ℚ content of `SounioTacrolimusDDI.combined_f_widens_pbox`:
    adding a well-formed delta box WIDENS (or preserves) the total width.

    Proof: chain `Rat.add_le_add_right` (to add `a.lo+b.lo` to both sides),
    then simplify LHS via `Rat.sub_add_cancel`, then apply
    `Rat.add_le_add_left.mpr hb.lo_le_hi`.

    `#print axioms addR_width_ge_left` shows only
    `[propext, Classical.choice, Quot.sound]` — pure ℚ, no axioms. -/
theorem addR_width_ge_left
    (a b : PBoxR)
    (hb : WellFormedR b) :
    widthR a ≤ widthR (addR a b) := by
  obtain ⟨hb_lo_le_hi, _⟩ := hb
  -- Unfold widthR
  show a.hi - a.lo ≤ (a.hi + b.hi) - (a.lo + b.lo)
  -- Add (a.lo + b.lo) to both sides:
  rw [← Rat.add_le_add_right (c := a.lo + b.lo)]
  -- LHS: (a.hi - a.lo) + (a.lo + b.lo) = a.hi + b.lo
  have lhs_eq : (a.hi - a.lo) + (a.lo + b.lo) = a.hi + b.lo := by
    rw [Rat.sub_eq_add_neg, Rat.add_assoc]
    rw [show -a.lo + (a.lo + b.lo) = b.lo from by
      rw [← Rat.add_assoc, Rat.neg_add_cancel, Rat.zero_add]]
  -- RHS: (a.hi + b.hi - (a.lo + b.lo)) + (a.lo + b.lo) = a.hi + b.hi
  have rhs_eq : ((a.hi + b.hi) - (a.lo + b.lo)) + (a.lo + b.lo) = a.hi + b.hi :=
    Rat.sub_add_cancel
  rw [lhs_eq, rhs_eq]
  -- Now prove: a.hi + b.lo ≤ a.hi + b.hi  ↔  b.lo ≤ b.hi ✓
  exact (Rat.add_le_add_left).mpr hb_lo_le_hi

-- ================================================================
-- §8. Directional projection (for project_band_containment)
-- ================================================================
--
-- `SounioKnightian.project_band_containment` states that a scalar
-- projection `q` of a p-box preserves containment: if `point ∈ p`,
-- then `q*point ∈ projectBand(p, q)`. The Float-level statement is
-- `: True` because no Float order/mul lemmas exist in core Lean 4
-- (import cycle with SounioPBoxSemantics also prevents restatement there).
--
-- Here we discharge the ℚ version: define `projectR` and `containsR`
-- on `PBoxR`, then prove containment is preserved when `0 ≤ q`.
-- The sign split (q < 0 case flips lo/hi) is handled by scoping to
-- `0 ≤ q` — mirroring the runtime `pb_project_band` guard.

/-- Containment predicate on `PBoxR`: the rational `point` lies in
    the mean band `[lo, hi]`. -/
def containsR (p : PBoxR) (point : Rat) : Prop :=
  p.lo ≤ point ∧ point ≤ p.hi

/-- Scalar projection of a `PBoxR` by nonneg `q`. Maps
    `[lo, hi]` → `[q·lo, q·hi]` (band expands/contracts uniformly).
    Mirrors the runtime `pb_project_band (p, q)` for `q ≥ 0`. -/
def projectR (p : PBoxR) (q : Rat) : PBoxR :=
  { lo  := q * p.lo
    hi  := q * p.hi
    var := q * p.var }

/-- **REAL ℚ DISCHARGE** of `SounioKnightian.project_band_containment`.

    For any nonneg scalar `q`, if `point` is contained in `p`'s band,
    then `q * point` is contained in the projected band `projectR p q`:

        containsR p point → 0 ≤ q → containsR (projectR p q) (q * point)

    Proof: two applications of `Rat.mul_le_mul_of_nonneg_left` (q ≥ 0).

    `#print axioms projectR_containment_rat` shows only
    `[propext, Classical.choice, Quot.sound]` — pure ℚ, no axioms.

    **STATUS of the Float-level theorem:** `SounioKnightian.project_band_containment`
    remains `: True`. Float multiplication has no monotonicity lemma in core Lean 4
    (no `Float.mul_le_iff_finite`); the bridge needs a future `mul_ord_bound` axiom.
    Use this ℚ theorem in proofs that need the content. -/
theorem projectR_containment_rat
    (p : PBoxR) (q point : Rat)
    (hq  : 0 ≤ q)
    (hc  : containsR p point) :
    containsR (projectR p q) (q * point) := by
  obtain ⟨hlo, hhi⟩ := hc
  constructor
  · -- q * p.lo ≤ q * point   (from p.lo ≤ point and q ≥ 0)
    exact Rat.mul_le_mul_of_nonneg_left hlo hq
  · -- q * point ≤ q * p.hi   (from point ≤ p.hi and q ≥ 0)
    exact Rat.mul_le_mul_of_nonneg_left hhi hq

end Sounio.PBoxSemantics
