/-
  SounioAntiGarblingModel.lean

  Crux #1 (the keystone) + §12 mechanism, as a kernel-checked model.

  A value is an AFFINE FORM over independent unit-variance noise symbols ε1, ε2
  (coefficients c1, c2). The affine coefficients ARE the noise symbols of §12.
  - `trueVar`  = the exact variance (affine model — tracks shared sources).
  - `naiveAddVar` = the scalar-independence variance (`ep_add`: var_a + var_b),
    which forgets source identity — the §11 anti-garbling generator.

  Proven here (concrete witnesses, kernel-checked by `decide`, no Mathlib):
  - the naive scalar add UNDER-states variance on a shared source (anti-garbling);
  - the understatement is exactly 2·⟨a,b⟩ (twice the covariance);
  - the naive add is SOUND iff the sources are disjoint (⟨a,b⟩ = 0) — the DISJ
    side-condition of Crux #1, and exactly the §11 x+x finding.

  General identity (the theorem these witness):
    trueAddVar a b − naiveAddVar a b = 2·⟨a,b⟩,
  hence naiveAddVar = trueAddVar ⟺ ⟨a,b⟩ = 0 (DISJ). The general form is a
  polynomial identity provable by `ring` under Mathlib; this file discharges
  representative Int witnesses without Mathlib, matching the repo's Mathlib-free
  Lean discipline.

  STATUS: CHECKED clean under Lean 4.33.1 (leanprover/lean4:stable), 2026-08-22;
  all theorems `decide`, `#print axioms` = no axioms (kernel-checked). sorry = 0.
-/

namespace Sounio.AntiGarbling

/-- An affine form over two independent unit-variance noise symbols ε1, ε2. -/
structure AffVal where
  c1 : Int
  c2 : Int
deriving DecidableEq

/-- Exact variance: the affine model, which tracks the shared noise symbols. -/
def trueVar (a : AffVal) : Int := a.c1 * a.c1 + a.c2 * a.c2

/-- Affine addition: add coefficients symbol-by-symbol (correlation handled by construction). -/
def addAff (a b : AffVal) : AffVal := ⟨a.c1 + b.c1, a.c2 + b.c2⟩

/-- The true variance of the sum. -/
def trueAddVar (a b : AffVal) : Int := trueVar (addAff a b)

/-- The naive scalar `ep_add` variance: var(a) + var(b), assuming independence. -/
def naiveAddVar (a b : AffVal) : Int := trueVar a + trueVar b

/-- Covariance = the noise-symbol inner product. Zero ⟺ disjoint sources (DISJ). -/
def inner (a b : AffVal) : Int := a.c1 * b.c1 + a.c2 * b.c2

/-- A measured value on source ε1. -/
def x : AffVal := ⟨1, 0⟩
/-- A measured value on the disjoint source ε2. -/
def y : AffVal := ⟨0, 1⟩
/-- A larger value sharing both sources with itself. -/
def z : AffVal := ⟨2, 1⟩

-- ── Anti-garbling: the naive add fabricates precision on a shared source ──

/-- `x + x`: naive scalar add gives variance 2, the true (affine) variance is 4. -/
theorem anti_garbling_x_plus_x : naiveAddVar x x < trueAddVar x x := by decide

/-- The understatement is exactly 2·⟨x,x⟩ (twice the covariance). -/
theorem anti_garbling_gap_x : trueAddVar x x - naiveAddVar x x = 2 * inner x x := by decide

/-- Same law at a larger coefficient: gap = 2·⟨z,z⟩. -/
theorem anti_garbling_gap_z : trueAddVar z z - naiveAddVar z z = 2 * inner z z := by decide

-- ── Soundness restored exactly under DISJ (disjoint noise symbols) ──

/-- Disjoint sources ⇒ zero covariance. -/
theorem disjoint_inner_zero : inner x y = 0 := by decide

/-- Under DISJ the naive scalar add is exact — no anti-garbling. -/
theorem sound_under_disjoint : naiveAddVar x y = trueAddVar x y := by decide

/-- The gap vanishes exactly when the covariance does (DISJ), on the x/y witnesses. -/
theorem gap_zero_iff_disjoint_witness :
    (trueAddVar x y - naiveAddVar x y = 0) ↔ (inner x y = 0) := by decide

end Sounio.AntiGarbling
