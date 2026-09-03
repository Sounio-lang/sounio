import SounioCayleyDickson
import SounioZeroDivisorBridge
import SounioSurgicalCalculus

/-!
# Sounio — Sunflowers over the 168 / Zero-Divisor / Surgical structure (Erdős [20])

Lean counterpart to `docs/research/sunflower-168-sounio-note.md`. Incidence-combinatorics
sibling of `SounioErdosUnitDistance.lean` and `SounioFanoArcsBlocking.lean`: same honest,
fully-decidable spirit, scoped to **Erdős problem [20], the sunflower (Δ-system)
conjecture**, read on the finite set system that the verified 168/ZD/Surgical machine
already fixes.

## Sunflower vocabulary

A *sunflower with `p` petals* (a Δ-system) is a family of `p` sets `A₁ … A_p` whose
pairwise intersections all equal one common *core* `Y`: `A_i ∩ A_j = Y` for `i ≠ j`.
The *petals* `A_i \ Y` are then pairwise disjoint. (Empty core ⇔ pairwise-disjoint sets.)

## Two set systems, both already verified

* **Edge system (the 168 ZD classes as 2-element sets).** The annihilation graph of
  `SounioZeroDivisorBridge` is 4-regular (`every_primitive_has_4_annihilators`,
  `zd_degree_4`). In sunflower language, **every primitive `u` is the core of a
  sunflower with exactly 4 petals**: its 4 incident ZD classes `{u, aᵢ}` pairwise meet
  in exactly `{u}`, with the four annihilators `aᵢ` as petals. This is a one-line
  reinterpretation of the verified degree-4 theorem — see `star_is_4petal_sunflower`.

* **Neighborhood system (the 84 annihilator 4-sets `UNLEARN u`).** Here the sunflower
  structure is genuinely constrained by the algebra:
    - `ann_lies_in_one_fiber`        : each `UNLEARN u` sits inside a single xor-fiber.
    - `fibers_pairwise_disjoint`     : the 7 fibers partition the 84 primitives, hence
                                       form the canonical empty-core **7-petal** sunflower.
    - `fiber_ann_3petal_sunflower_free` : inside a fiber, no 3 distinct annihilator
                                       neighborhoods form a sunflower (max 2 petals).
    - `fiber_no_three_disjoint_ann`  : inside a fiber, no 3 annihilator neighborhoods
                                       are pairwise disjoint (≤ 2 per fiber).
    - `fourteen_petal_disjoint_witness` : an explicit 14-primitive family whose
                                       annihilator sets are pairwise disjoint.
  Together with cross-fiber disjointness these pin the **maximum empty-core sunflower of
  the neighborhood system at exactly 14 = 7 fibers × 2** (synthesis in the note §3).

## Honest scope (the lift obstruction — same as the sibling files)

The classical sunflower lemma bounds `f(n, p)` *asymptotically* in the set size `n`.
Our object is a **fixed finite set system** (84 primitives, sets of size 2 or 4), so
nothing here improves the asymptotic constant in Erdős [20]. What it gives, exactly and
auditably, is: a finite set system whose entire sunflower structure is *forced by the
verified algebra* (4-regularity ⇒ uniform 4-petal stars; intra-fiber annihilation ⇒
bounded cross-sunflowers), plus the 6 surgical ops as provable core/petal manipulators.
Level 3 (a new bound or counterexample family for the classical `f(n)`) is **not**
claimed; it is left as the explicit obligation in the note §4.

All theorems `native_decide` over the finite universes already fixed by
`SounioZeroDivisorBridge` (`validPrims = 84`, `orderedZDPairs = 336`) and
`SounioSurgicalCalculus` (`applyOp`). No `sorry`, no Mathlib.
-/

namespace Sounio.Sunflower

open Sounio.CayleyDickson
open Sounio.ZeroDivisorBridge
open Sounio.SurgicalCalculus

-- ===========================================================================
-- §1. Sets-as-lists over PrimSed, and the sunflower predicate.
-- ===========================================================================

/-- The petal/neighborhood set of `u`: the 4 right-annihilators (`UNLEARN u`). -/
def annOf (u : PrimSed) : List PrimSed := applyOp .unlearn u

/-- Set intersection on `List PrimSed` (elements of `a` that are also in `b`). -/
def inter (a b : List PrimSed) : List PrimSed := a.filter (fun x => b.contains x)

/-- Set equality on lists (mutual containment). -/
def seteq (a b : List PrimSed) : Bool :=
  a.all (fun x => b.contains x) && b.all (fun x => a.contains x)

/-- Two list-sets are disjoint. -/
def disjointL (a b : List PrimSed) : Bool := !(a.any (fun x => b.contains x))

/-- Three sets form a sunflower iff all three pairwise intersections coincide. -/
def isSunflower3 (x y z : List PrimSed) : Bool :=
  let cxy := inter x y
  let cxz := inter x z
  let cyz := inter y z
  seteq cxy cxz && seteq cxy cyz

/-- De-duplicate a `List PrimSed`. -/
def nubP (l : List PrimSed) : List PrimSed :=
  l.foldl (fun acc x => if acc.contains x then acc else acc ++ [x]) []

/-- The members of a fiber (the 12 primitives with a given xor-label). -/
def fiberOf (L : Nat) : List PrimSed := validPrims.filter (fun v => xorLabel v == L)

/-- The 7 active fiber labels. -/
def labels7 : List Nat := [9, 10, 11, 12, 13, 14, 15]

-- ===========================================================================
-- §2. Edge system: the 168 ZD classes — every point is a 4-petal sunflower core.
-- ===========================================================================

/-- **Star sunflower.** Every primitive `u` is the core of a sunflower with exactly
    4 petals: its 4 right-annihilators are distinct and none equals `u`, so the 4
    incident ZD classes `{u, aᵢ}` pairwise intersect in exactly `{u}`. This is the
    sunflower reading of `every_primitive_has_4_annihilators` + `zd_degree_4`. -/
theorem star_is_4petal_sunflower :
    validPrims.all (fun u =>
      let petals := annOf u
      petals.length == 4
      && (nubP petals).length == 4          -- 4 distinct petals
      && petals.all (fun a => a != u)) := by -- core point u is not a petal
  native_decide

-- ===========================================================================
-- §3. Neighborhood system: the 84 annihilator 4-sets.
-- ===========================================================================

/-- Each annihilator neighborhood lies inside a single xor-fiber. -/
theorem ann_lies_in_one_fiber :
    validPrims.all (fun u => (annOf u).all (fun a => xorLabel a == xorLabel u)) := by
  native_decide

/-- **Canonical 7-petal sunflower.** The 7 fibers are pairwise disjoint (they partition
    the 84 primitives), hence form an empty-core sunflower with 7 petals — the "parallel
    classes" of the structure. -/
theorem fibers_pairwise_disjoint :
    labels7.all (fun L => labels7.all (fun M =>
      L == M || disjointL (fiberOf L) (fiberOf M))) := by
  native_decide

/-- **No 3-petal sunflower inside a fiber.** Among the annihilator neighborhoods of a
    single fiber, no three *distinct* sets form a sunflower: the neighborhood system is
    3-sunflower-free fiber-by-fiber (maximum 2 petals). -/
theorem fiber_ann_3petal_sunflower_free :
    labels7.all (fun L =>
      let fp := fiberOf L
      fp.all (fun u => fp.all (fun v => fp.all (fun w =>
        -- distinct as SETS ⇒ not a sunflower
        !( !seteq (annOf u) (annOf v)
           && !seteq (annOf u) (annOf w)
           && !seteq (annOf v) (annOf w)
           && isSunflower3 (annOf u) (annOf v) (annOf w) ))))) := by
  native_decide

/-- **At most 2 disjoint neighborhoods per fiber.** No three annihilator neighborhoods
    within one fiber are pairwise disjoint; with cross-fiber disjointness this caps the
    empty-core sunflower of the neighborhood system at `7 × 2 = 14`. -/
theorem fiber_no_three_disjoint_ann :
    labels7.all (fun L =>
      let fp := fiberOf L
      fp.all (fun u => fp.all (fun v => fp.all (fun w =>
        !( disjointL (annOf u) (annOf v)
           && disjointL (annOf u) (annOf w)
           && disjointL (annOf v) (annOf w)
           && !seteq (annOf u) (annOf v)
           && !seteq (annOf u) (annOf w)
           && !seteq (annOf v) (annOf w) ))))) := by
  native_decide

/-- Annihilator neighborhoods in *different* fibers are always disjoint
    (immediate from `ann_lies_in_one_fiber`, stated for the §3 synthesis). -/
theorem ann_cross_fiber_disjoint :
    validPrims.all (fun u => validPrims.all (fun v =>
      xorLabel u == xorLabel v || disjointL (annOf u) (annOf v))) := by
  native_decide

-- ===========================================================================
-- §4. The 14-petal empty-core witness (lower bound for the neighborhood system).
-- ===========================================================================

/-- An explicit family of 14 primitives, two per fiber, whose annihilator
    neighborhoods are pairwise disjoint (extracted by the Python census). -/
def witness14 : List PrimSed := [
  PrimSed.mk 2 11 false, PrimSed.mk 2 11 true,
  PrimSed.mk 1 11 false, PrimSed.mk 1 11 true,
  PrimSed.mk 1 10 false, PrimSed.mk 1 10 true,
  PrimSed.mk 1 13 false, PrimSed.mk 1 13 true,
  PrimSed.mk 1 12 false, PrimSed.mk 1 12 true,
  PrimSed.mk 1 15 false, PrimSed.mk 1 15 true,
  PrimSed.mk 1 14 false, PrimSed.mk 1 14 true ]

/-- All 14 witness elements are valid primitives. -/
theorem witness14_valid : witness14.all isPrimValid := by native_decide

/-- **14-petal empty-core sunflower.** The 14 witness annihilator neighborhoods are
    pairwise disjoint — an empty-core sunflower with 14 petals in the neighborhood
    system. With `fiber_no_three_disjoint_ann` + `ann_cross_fiber_disjoint` this is the
    maximum (note §3). -/
theorem fourteen_petal_disjoint_witness :
    witness14.all (fun u => witness14.all (fun v =>
      u == v || disjointL (annOf u) (annOf v))) := by
  native_decide

/-- The witness genuinely has 14 distinct primitives. -/
theorem witness14_card : (nubP witness14).length = 14 := by native_decide

-- ===========================================================================
-- §5. Surgical-op dictionary (core / petal manipulation), cardinalities reused.
-- ===========================================================================

/-- **Sunflower dictionary, machine-checked.**
    * UNLEARN `u` = the 4 petals of `u`'s star (the forgettable 4-set).
    * EDIT `u`    = the 12 fiber-mates = the parallel class ("bouquet") of `u`.
    * GATE `u`    = the 79 survivors = everything outside `u`'s petals and `u` itself.
    These are the verified cardinalities (`unlearn_card`, `edit_card`, `gate_card`)
    re-read as sunflower operations: UNLEARN extracts petals, EDIT names the parallel
    class, GATE returns the non-petal universe. -/
theorem surgical_sunflower_dictionary :
    validPrims.all (fun u =>
      (applyOp .unlearn u).length == 4
      && (applyOp .edit u).length == 12
      && (applyOp .gate u).length == 79) := by
  native_decide

-- ===========================================================================
-- §6. Summary.
-- ===========================================================================

/-- **Sunflower structure summary.** Every point is a 4-petal star core; annihilator
    neighborhoods are intra-fiber; the 7 fibers form a 7-petal empty-core sunflower;
    the neighborhood system is 3-sunflower-free within fibers, has at most 2 disjoint
    neighborhoods per fiber, and realizes a 14-petal empty-core sunflower. -/
theorem sunflower_structure_summary :
    (validPrims.all (fun u =>
      (annOf u).length == 4 && (nubP (annOf u)).length == 4)) ∧
    (validPrims.all (fun u => (annOf u).all (fun a => xorLabel a == xorLabel u))) ∧
    (labels7.all (fun L => labels7.all (fun M =>
      L == M || disjointL (fiberOf L) (fiberOf M)))) ∧
    (witness14.all (fun u => witness14.all (fun v =>
      u == v || disjointL (annOf u) (annOf v)))) := by
  exact ⟨by native_decide, ann_lies_in_one_fiber, fibers_pairwise_disjoint,
         fourteen_petal_disjoint_witness⟩

end Sounio.Sunflower

-- Next Lean milestone (note §4–§5): close the neighborhood-system maximum at "= 14"
-- *inside* Lean (currently witness ≥14 + per-fiber ≤2 + cross-fiber disjoint give the
-- synthesis in prose), and lift to the associator-generated families `((p·u)·v)` with
-- `u·v = 0` — the only lever that perturbs the fixed 4-regular/intra-fiber structure,
-- mirroring the deferred associator milestone in SounioErdosUnitDistance.lean.
