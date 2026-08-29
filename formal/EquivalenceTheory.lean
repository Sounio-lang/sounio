/-
  formal/EquivalenceTheory.lean

  Discharged Lean 4 obligations for the Equivalence Theory program ("types carry their
  theory of equivalence"). Each theorem below is the machine-checked counterpart of a
  decidable claim the Sounio compiler relies on; the comment on each names its Sounio
  source. The obligations are MIRRORED by hand from the Sounio definitions (automatic
  emission from the compiler — wiring `self-hosted/compiler/epistemic_lean_export.sio`
  into the build — remains future work; see the audit note).

  Scope: the DECIDABLE cores — the invariance-group lattice + comparability rule
  (Feature B) and the e-graph float-reassociation gate composition (Features A-1 and
  A-layer-2). These are finite-enum / Boolean facts, proved by `decide` / exhaustive
  `cases`, so they need no Mathlib. Feature C-a's geodesic identities are real-analysis
  statements (arccosh/cosh round-trips) verified numerically in
  `tests/run-pass/hyperbolic_geodesic.sio`; their Lean formalisation (Mathlib `Real`)
  is deliberately left as future work.

  Mathlib-free. Lean toolchain: leanprover/lean4:v4.30.0-rc2 (see formal/lean-toolchain).
-/

namespace Sounio.EquivalenceTheory

/-! ## Feature B — invariance-group lattice (mirror `stdlib/check/invariance.sio`) -/

/-- The finite, totally ordered lattice of invariance groups: larger ⇒ stronger
    invariant. Mirrors `enum Group { DiffInf, Diff1, BiLip, Homeo }`. -/
inductive Group : Type where
  | DiffInf : Group
  | Diff1   : Group
  | BiLip   : Group
  | Homeo   : Group
  deriving Repr, DecidableEq

/-- `group_rank` — DiffInf=1 < Diff1=2 < BiLip=3 < Homeo=4. -/
def groupRank : Group → Nat
  | .DiffInf => 1
  | .Diff1   => 2
  | .BiLip   => 3
  | .Homeo   => 4

/-- `group_intersect` — the smaller (more restrictive) group; never empty in a chain. -/
def groupIntersect (a b : Group) : Group :=
  if groupRank a ≤ groupRank b then a else b

/-- `transform_in_group` — a transform of finest class `c` belongs to group `g`
    iff `rank c ≤ rank g` (a Diff1 map is also Homeo, but not conversely). -/
def transformInGroup (c g : Group) : Bool := decide (groupRank c ≤ groupRank g)

/-- `frames_comparable` — observables of groups `ga, gb` whose frames are related by a
    transform of finest class `c` are comparable iff `c ∈ (ga ∩ gb)`. -/
def framesComparable (ga gb c : Group) : Bool :=
  transformInGroup c (groupIntersect ga gb)

/-- `invariant_comparable_default_ranks` — the conservative cross-frame rule: with no
    discharged frame proof the relating transform is taken as `Homeo`. -/
def comparableDefault (ga gb : Group) : Bool := framesComparable ga gb Group.Homeo

-- The lattice is a strict total order: `groupRank` determines identity, so the four
-- groups are pairwise rank-distinct (rank is injective).
theorem groupRank_determines : ∀ a b : Group, (a = b) ↔ (groupRank a = groupRank b) := by
  intro a b; cases a <;> cases b <;> decide

-- `group_intersect` is exactly the rank-minimum: the algebraic content of "intersection".
theorem groupIntersect_isMin :
    ∀ a b : Group, groupRank (groupIntersect a b) = min (groupRank a) (groupRank b) := by
  intro a b; cases a <;> cases b <;> decide

-- `group_intersect` is commutative on ranks.
theorem groupIntersect_comm_rank :
    ∀ a b : Group, groupRank (groupIntersect a b) = groupRank (groupIntersect b a) := by
  intro a b; cases a <;> cases b <;> decide

-- Comparability is correct: it holds iff the transform's rank is within the intersection.
theorem framesComparable_iff :
    ∀ ga gb c : Group,
      framesComparable ga gb c = true ↔ groupRank c ≤ min (groupRank ga) (groupRank gb) := by
  intro ga gb c; cases ga <;> cases gb <;> cases c <;> decide

-- SAME FRAME ⇒ comparable at ANY group (the relating transform is the identity = DiffInf).
-- This is what makes Diff1/BiLip invariants usable (FrameId follow-up).
theorem sameFrame_comparable :
    ∀ ga gb : Group, framesComparable ga gb Group.DiffInf = true := by
  intro ga gb; cases ga <;> cases gb <;> decide

-- §4.4 case 1: two Lyapunov-based (Diff1) depths in frames related only by a Homeo are
-- NOT comparable under the conservative default — the compiler rejects them.
theorem lyapunov_crossHomeo_not_comparable :
    comparableDefault Group.Diff1 Group.Diff1 = false := by decide

-- §4.4 case 2: two entropy-based (Homeo) depths across a Homeo-related frame ARE
-- comparable — the compiler accepts the ratio.
theorem entropy_crossHomeo_comparable :
    comparableDefault Group.Homeo Group.Homeo = true := by decide

-- Under the conservative default only Homeo-group observables are comparable.
theorem comparableDefault_onlyHomeo :
    ∀ ga gb : Group, comparableDefault ga gb = true ↔ (ga = Group.Homeo ∧ gb = Group.Homeo) := by
  intro ga gb; cases ga <;> cases gb <;> decide

/-! ## Features A-1 / A-layer-2 — e-graph float-reassociation gate
    (mirror `eg_small_saturate_float` in `self-hosted/ir/egraph.sio`).

    A float reassociation fires iff ALL THREE orthogonal axes permit it:
      • `valueAlgebraPermits` — Cayley-Dickson / Fano (`reassoc_strategy`),
      • `inexactOptIn`        — A-1 exactness axis (`allow_inexact_reassoc`),
      • `chaotic`             — A-layer-2 chaos axis (`ctx.chaotic`), which BLOCKS. -/

def reassocAllowed (valueAlgebraPermits inexactOptIn chaotic : Bool) : Bool :=
  !chaotic && inexactOptIn && valueAlgebraPermits

-- A-layer-2: a CHAOTIC path refuses reassociation regardless of the other axes — the
-- chaos gate dominates the precision-preserving opt-in. (The theorem behind unit test T143d.)
theorem chaos_dominates :
    ∀ v i : Bool, reassocAllowed v i true = false := by
  intro v i; cases v <;> cases i <;> decide

-- A-1: an inexact carrier (f64) that has NOT opted in is never reassociated — the default.
theorem inexact_default_blocks :
    ∀ v c : Bool, reassocAllowed v false c = false := by
  intro v c; cases v <;> cases c <;> decide

-- The value-algebra axis is independently necessary (e.g. sedenion multiplication blocks).
theorem valueAlgebra_necessary :
    ∀ i c : Bool, reassocAllowed false i c = false := by
  intro i c; cases i <;> cases c <;> decide

-- When all three axes permit (algebra ok, opted in, not chaotic), reassociation fires —
-- so the gate removes only unsound rewrites, not all of them.
theorem permit_when_all_ok : reassocAllowed true true false = true := by decide

-- The full composition: reassociation fires IFF algebra-permits ∧ opted-in ∧ ¬chaotic.
theorem reassoc_composition :
    ∀ v i c : Bool, reassocAllowed v i c = true ↔ (v = true ∧ i = true ∧ c = false) := by
  intro v i c; cases v <;> cases i <;> cases c <;> decide

end Sounio.EquivalenceTheory
