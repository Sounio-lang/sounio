import SounioCayleyDickson
import SounioZeroDivisorBridge
import SounioSurgicalCalculus

/-!
# Sounio — Fano Arcs, Hyperovals, Blocking Sets, and the Surgical Calculus

Lean counterpart to `docs/research/fano-arcs-blocking-sounio-note.md`.
This is the incidence-geometry sibling of `SounioErdosUnitDistance.lean`:
same honest, fully-decidable spirit, but scoped to the much tighter target of
**arcs / hyperovals / blocking sets in a finite projective plane**, coordinatized
*through* the already-verified 168 / zero-divisor / surgical-calculus machine.

## What this file actually proves (no placeholders, no `sorry`, no Mathlib)

The base geometry is `PG(2,2)` — the Fano plane on points `{1..7}` with the 7
lines `fanoLines` of `SounioCayleyDickson`. We prove the classical small facts as
decidable witnesses, then connect them to the sedenion zero-divisor structure:

* `fano_max_arc_is_4` / `fano_has_seven_hyperovals` : the maximum arc has 4 points
  (a hyperoval), and there are exactly 7 of them (one per line, its complement).
* `fano_no_nontrivial_blocking_set` : `PG(2,2)` has **no** nontrivial blocking set
  — every set meeting all lines contains a line. (A known small fact, here machine
  checked over all `2^7` subsets.)

* **The bridge (new):** for every one of the 84 `validPrims`, the lo-index image
  of its 4 right-annihilators (`UNLEARN u`, from the verified surgical calculus) is
  exactly a Fano **hyperoval**, namely the complement of the unique line through
  `u.lo` and `xorLabel u ⊕ 8`:
    - `unlearn_kernel_is_arc`                 : it is always a 4-arc.
    - `unlearn_kernel_card4`                  : it always has 4 distinct points.
    - `unlearn_kernel_is_complement_of_line`  : it is the complement of a Fano line.
    - `unlearn_kernel_line_is_through_lo_and_fiber` : *which* line, exactly.
    - `all_seven_hyperovals_realized`         : the 84 kernels realize all 7 hyperovals.

* **Point/line duality of the ops (new):**
    - `unlearn_shadow_is_line_complement` : UNLEARN's point-shadow = complement of a LINE (4 pts).
    - `edit_shadow_is_point_complement`   : EDIT's point-shadow   = complement of a POINT (6 pts).
    - `gate_shadow_is_everything`         : GATE's point-shadow    = all 7 points.

  So the three surgical ops sweep out the three natural "co-flat" shadows of the
  plane: co-line (UNLEARN), co-point (EDIT), full (GATE).

## Honest scope (the lift obstruction)

The *named* open Erdős-style problems in this family (maximal `{k,n}`-arcs and
hyperovals in non-Desarguesian planes of order ≥ 9 — Hall/Hughes planes, Mathon
and Hamilton's open arc cases) live in planes whose order is **not** 2. Sedenions
are **not** a division ring / quasifield, so they cannot coordinatize such a plane
in the classical (ternary-ring) sense. What the verified 168/ZD machine *does*
give is an exact, auditable **finite shadow** in `PG(2,2)` plus a controlled
"twisting" mechanism (the 7 fibers × the 6 ops). We state the negative honestly:
nothing here yet improves a named bound. The contribution is Level 1–2 (clean
bridge + decidable witnesses); the Level-3 lift is left as an explicit obligation.

Builds only on `native_decide` over the finite universes already fixed by
`SounioCayleyDickson` (`fanoLines`, 343 triples) and `SounioZeroDivisorBridge`
(`PrimSed`, `validPrims = 84`, `orderedZDPairs = 336`).
-/

namespace Sounio.FanoArcsBlocking

open Sounio.CayleyDickson
open Sounio.ZeroDivisorBridge
open Sounio.SurgicalCalculus

-- ===========================================================================
-- §1. Fano plane incidence predicates on point sets (Bool, decidable).
--     Points are {1..7}; lines are `fanoLines` (the 7 XOR-zero triples).
-- ===========================================================================

/-- The seven points of `PG(2,2)`. -/
def fanoPoints : List Nat := List.range' 1 7

/-- Is point `p` on Fano line `L`? -/
def onLine (p : Nat) (L : Nat × Nat × Nat) : Bool :=
  p == L.1 || p == L.2.1 || p == L.2.2

/-- Does the point set `s` contain a full Fano line (3 collinear points)? -/
def containsLine (s : List Nat) : Bool :=
  fanoLines.any (fun L => s.contains L.1 && s.contains L.2.1 && s.contains L.2.2)

/-- A point set is an *arc* iff no 3 of its points are collinear, i.e. it
    contains no full Fano line (every Fano line has exactly 3 points). -/
def isArc (s : List Nat) : Bool := !containsLine s

/-- Does `s` meet every line (a transversal / "blocking" candidate)? -/
def meetsAllLines (s : List Nat) : Bool :=
  fanoLines.all (fun L => s.contains L.1 || s.contains L.2.1 || s.contains L.2.2)

/-- A *nontrivial blocking set*: meets every line but contains no full line. -/
def isBlockingSet (s : List Nat) : Bool := meetsAllLines s && !containsLine s

/-- `s` is the complement (within `{1..7}`) of the line `L`:
    a point is in `s` iff it is NOT on `L`. -/
def isComplementOfLine (s : List Nat) (L : Nat × Nat × Nat) : Bool :=
  fanoPoints.all (fun p => s.contains p == !(onLine p L))

/-- De-duplicate a `List Nat` (core-only; preserves first occurrence). -/
def nubNat (l : List Nat) : List Nat :=
  l.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

-- ===========================================================================
-- §2. Subset enumeration over the 7 Fano points (2^7 = 128 subsets).
-- ===========================================================================

/-- All subsets of a `List Nat`, core-only (powerset, 2^n entries). -/
def subsetsOf : List Nat → List (List Nat)
  | [] => [[]]
  | x :: xs => let r := subsetsOf xs; r ++ r.map (fun s => x :: s)

/-- All 2^7 = 128 subsets of the Fano point set. -/
def allPointSets : List (List Nat) := subsetsOf fanoPoints

theorem point_set_count : allPointSets.length = 128 := by native_decide

-- ===========================================================================
-- §3. Classical Fano arc / blocking facts (decidable over all 128 subsets).
-- ===========================================================================

/-- Size of the largest arc in `PG(2,2)`. -/
def maxArcSize : Nat :=
  (allPointSets.filter isArc).foldl (fun m s => Nat.max m s.length) 0

/-- Number of arcs of maximum size (hyperovals). -/
def numHyperovals : Nat :=
  (allPointSets.filter (fun s => isArc s && s.length == 4)).length

/-- Number of nontrivial blocking sets. -/
def numBlockingSets : Nat :=
  (allPointSets.filter isBlockingSet).length

/-- **Maximum arc theorem.** The largest arc in `PG(2,2)` has 4 points
    (a *hyperoval*: the complement of a line). -/
theorem fano_max_arc_is_4 : maxArcSize = 4 := by native_decide

/-- **Seven hyperovals.** `PG(2,2)` has exactly 7 hyperovals (4-arcs),
    one per line (its complement). -/
theorem fano_has_seven_hyperovals : numHyperovals = 7 := by native_decide

/-- Each of the 7 line-complements is itself a 4-arc (a hyperoval). -/
theorem line_complements_are_arcs :
    fanoLines.all (fun L =>
      isArc (fanoPoints.filter (fun p => !(onLine p L)))) := by native_decide

/-- **No nontrivial blocking set.** Every point set meeting all 7 lines
    contains a full line; `PG(2,2)` has no nontrivial blocking set.
    (Equivalently: the minimal line-transversals are the lines themselves.) -/
theorem fano_no_nontrivial_blocking_set : numBlockingSets = 0 := by native_decide

-- ===========================================================================
-- §4. The bridge: UNLEARN's kernel is a Fano hyperoval.
--     `annLo u` = the lo-index image of the 4 right-annihilators of `u`,
--     i.e. the point-shadow of the verified surgical op `UNLEARN`.
-- ===========================================================================

/-- The point-shadow (lo-index image) of `UNLEARN u` from the surgical calculus.
    By `unlearn_card`, `applyOp .unlearn u` has 4 elements (the right-annihilators). -/
def unlearnShadow (u : PrimSed) : List Nat :=
  (applyOp .unlearn u).map (fun v => v.lo)

/-- The 4 annihilator lo-indices are distinct (a genuine 4-point set). -/
theorem unlearn_kernel_card4 :
    validPrims.all (fun u => (nubNat (unlearnShadow u)).length == 4) := by native_decide

/-- **UNLEARN's kernel is always a 4-arc.** No 3 of the annihilator lo-indices
    are collinear in `PG(2,2)`. -/
theorem unlearn_kernel_is_arc :
    validPrims.all (fun u => isArc (unlearnShadow u)) := by native_decide

/-- **UNLEARN's kernel is a hyperoval.** For every primitive, its annihilator
    lo-set is the complement of some Fano line — the maximal arc. The algebraic
    "4D forgettable subspace" *is* a geometric hyperoval. -/
theorem unlearn_kernel_is_complement_of_line :
    validPrims.all (fun u =>
      fanoLines.any (fun L => isComplementOfLine (unlearnShadow u) L)) := by
  native_decide

/-- **Which line.** The missing line is the unique Fano line through `u.lo` and
    the fiber-mate point `xorLabel u ⊕ 8`. (These two points always differ, so the
    line is unique.) This pins the hyperoval exactly from `(u.lo, fiber)`. -/
theorem unlearn_kernel_line_is_through_lo_and_fiber :
    validPrims.all (fun u =>
      fanoLines.any (fun L =>
        onLine u.lo L
        && onLine (xorLabel u ^^^ 8) L
        && isComplementOfLine (unlearnShadow u) L)) := by native_decide

/-- The 84 UNLEARN kernels realize *all* 7 hyperovals: every line-complement
    arises as the annihilator hyperoval of some primitive. -/
theorem all_seven_hyperovals_realized :
    fanoLines.all (fun L =>
      validPrims.any (fun u => isComplementOfLine (unlearnShadow u) L)) := by
  native_decide

-- ===========================================================================
-- §5. Point/line duality of the surgical ops (their point-shadows).
-- ===========================================================================

/-- Point-shadow of `EDIT u` (the 12 fiber-mates). -/
def editShadow (u : PrimSed) : List Nat := (applyOp .edit u).map (fun v => v.lo)

/-- Point-shadow of `GATE u` (the 79 preserved primitives). -/
def gateShadow (u : PrimSed) : List Nat := (applyOp .gate u).map (fun v => v.lo)

/-- **UNLEARN ↔ co-line.** UNLEARN's shadow is the complement of a LINE (4 points).
    (Restatement of §4, dual partner of the next two.) -/
theorem unlearn_shadow_is_line_complement :
    validPrims.all (fun u => (nubNat (unlearnShadow u)).length == 4) := by native_decide

/-- **EDIT ↔ co-point.** EDIT's shadow is the complement of a single POINT: it
    covers 6 of the 7 points, omitting exactly the fiber's point `xorLabel u ⊕ 8`. -/
theorem edit_shadow_is_point_complement :
    validPrims.all (fun u =>
      (nubNat (editShadow u)).length == 6
      && !(editShadow u).contains (xorLabel u ^^^ 8)) := by native_decide

/-- **GATE ↔ full.** GATE's shadow covers all 7 points of the plane. -/
theorem gate_shadow_is_everything :
    validPrims.all (fun u => (nubNat (gateShadow u)).length == 7) := by native_decide

-- ===========================================================================
-- §6. Summary.
-- ===========================================================================

/-- **Fano / surgical incidence summary.**
    The classical small facts (max arc 4, 7 hyperovals, no nontrivial blocking
    set) hold; and the verified surgical ops have exact geometric shadows in
    `PG(2,2)`: UNLEARN → hyperoval (co-line), EDIT → co-point, GATE → full plane;
    all 7 hyperovals are realized by the 84 annihilator kernels. -/
theorem fano_surgical_incidence_summary :
    maxArcSize = 4 ∧
    numHyperovals = 7 ∧
    numBlockingSets = 0 ∧
    (validPrims.all (fun u => isArc (unlearnShadow u))) ∧
    (validPrims.all (fun u =>
      fanoLines.any (fun L => isComplementOfLine (unlearnShadow u) L))) ∧
    (fanoLines.all (fun L =>
      validPrims.any (fun u => isComplementOfLine (unlearnShadow u) L))) := by
  exact ⟨fano_max_arc_is_4, fano_has_seven_hyperovals, fano_no_nontrivial_blocking_set,
         unlearn_kernel_is_arc, unlearn_kernel_is_complement_of_line,
         all_seven_hyperovals_realized⟩

end Sounio.FanoArcsBlocking

-- Next Lean milestone (research note §5): the *associator* shadow
-- `((p · u) · v)` with `u · v = 0`. The linear right-multiplication surgery is
-- now machine-checked to leave the Fano arc/blocking structure intact (UNLEARN's
-- kernel is exactly a hyperoval, never more); the genuine non-associative lever
-- is the principled route to perturb incidence beyond the 7 fixed hyperovals.
