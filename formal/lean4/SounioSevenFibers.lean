import SounioZeroDivisorBridge
import SounioSurgicalCalculus

/-!
# Seven fibers of twelve — measurement, not a new calculus

claude-1 predicted, from the arithmetic `84 ÷ 12 = 7`, that the 84
primitives split into seven fibers of twelve, a second partition
orthogonal to the annihilator split `{kernel 4, self 1, complement 79}`.

This file does not alter any existing theorem. It names the 12 members
of `fiber9Prims`, names one primitive outside that list, and decides
whether the seven xor-fibers form a partition and whether the
annihilator kernel of a primitive ever leaves its fiber.

No `sorry`. No Mathlib. All `native_decide` over the finite 84.
-/

namespace Sounio.SevenFibers

open Sounio.ZeroDivisorBridge
open Sounio.SurgicalCalculus

/-- The 12 members of fiber 9, in the order `fiberPrims 9` produces
    them (filter of `allPrims`). This is the positive control. -/
def fiber9Explicit : List PrimSed := [
  PrimSed.mk 2 11 false, PrimSed.mk 2 11 true,
  PrimSed.mk 3 10 false, PrimSed.mk 3 10 true,
  PrimSed.mk 4 13 false, PrimSed.mk 4 13 true,
  PrimSed.mk 5 12 false, PrimSed.mk 5 12 true,
  PrimSed.mk 6 15 false, PrimSed.mk 6 15 true,
  PrimSed.mk 7 14 false, PrimSed.mk 7 14 true
]

/-- Positive: `fiber9Prims` is exactly these twelve, not merely of length 12. -/
theorem fiber9_is_these_twelve : fiber9Prims = fiber9Explicit := by
  native_decide

/-- A valid primitive whose xor-label is 11, not 9. Negative control. -/
def outsider : PrimSed := PrimSed.mk 1 10 false

theorem outsider_valid : isPrimValid outsider = true := by native_decide

theorem outsider_label_11 : xorLabel outsider = 11 := by native_decide

theorem outsider_not_in_fiber9 : fiber9Prims.contains outsider = false := by
  native_decide

def labels7 : List Nat := [9, 10, 11, 12, 13, 14, 15]

/-- Each of the seven active labels carries exactly 12 primitives. -/
theorem each_fiber_card_12 :
    labels7.all (fun L => (fiberPrims L).length == 12) := by
  native_decide

/-- Every valid primitive has xor-label in `{9..15}`. -/
theorem every_prim_has_active_label :
    validPrims.all (fun v => labels7.contains (xorLabel v)) := by
  native_decide

/-- The seven fibers are pairwise disjoint. -/
theorem fibers_pairwise_disjoint :
    labels7.all (fun L => labels7.all (fun M =>
      L == M || (fiberPrims L).all (fun v => !(fiberPrims M).contains v))) := by
  native_decide

/-- Cardinality of the disjoint union is 7 × 12 = 84. -/
theorem seven_times_twelve_is_eighty_four :
    labels7.foldl (fun acc L => acc + (fiberPrims L).length) 0 = 84 := by
  native_decide

/-- Each fiber list is duplicate-free, so the 12 is a set cardinality. -/
theorem each_fiber_nodup :
    labels7.all (fun L => (fiberPrims L).Nodup) := by
  native_decide

theorem fiber9_explicit_nodup : fiber9Explicit.Nodup := by native_decide

/-- The annihilator kernel of every primitive sits inside that primitive's
    own xor-fiber. The kernel does not split across fibers. -/
theorem kernel_stays_in_own_fiber :
    validPrims.all (fun u =>
      (applyOp .unlearn u).all (fun a => xorLabel a == xorLabel u)) := by
  native_decide

/-- `edit u` is exactly the xor-fiber of `u`. -/
theorem edit_is_own_fiber :
    validPrims.all (fun u =>
      applyOp .edit u == fiberPrims (xorLabel u)) := by
  native_decide

/-- The kernel is a proper subset of the fiber: every annihilator of `u`
    is an edit-mate of `u`, but the two lists are not the same size. -/
theorem kernel_subset_of_edit :
    validPrims.all (fun u =>
      (applyOp .unlearn u).all (fun a => (applyOp .edit u).contains a)) := by
  native_decide

theorem edit_twelve_unlearn_four :
    validPrims.all (fun u =>
      (applyOp .edit u).length == 12 && (applyOp .unlearn u).length == 4) := by
  native_decide

/-- `u` itself is in `edit u` and is not in `unlearn u`. -/
theorem self_in_edit_not_in_unlearn :
    validPrims.all (fun u =>
      (applyOp .edit u).contains u && !(applyOp .unlearn u).contains u) := by
  native_decide

/-- Summary used by the receipt. Seven disjoint fibers of 12 partition
    the 84. The annihilator kernel never leaves its fiber, so the two
    partitions are nested, not orthogonal. -/
theorem seven_fibers_nested_not_orthogonal :
    fiber9Prims = fiber9Explicit ∧
    fiber9Prims.contains outsider = false ∧
    labels7.all (fun L => (fiberPrims L).length == 12) ∧
    labels7.all (fun L => (fiberPrims L).Nodup) ∧
    labels7.all (fun L => labels7.all (fun M =>
      L == M || (fiberPrims L).all (fun v => !(fiberPrims M).contains v))) ∧
    labels7.foldl (fun acc L => acc + (fiberPrims L).length) 0 = 84 ∧
    validPrims.all (fun v => labels7.contains (xorLabel v)) ∧
    validPrims.all (fun u =>
      (applyOp .unlearn u).all (fun a => xorLabel a == xorLabel u)) ∧
    validPrims.all (fun u =>
      (applyOp .edit u).length == 12 && (applyOp .unlearn u).length == 4) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact fiber9_is_these_twelve
  · exact outsider_not_in_fiber9
  · exact each_fiber_card_12
  · exact each_fiber_nodup
  · exact fibers_pairwise_disjoint
  · exact seven_times_twelve_is_eighty_four
  · exact every_prim_has_active_label
  · exact kernel_stays_in_own_fiber
  · exact edit_twelve_unlearn_four

end Sounio.SevenFibers
