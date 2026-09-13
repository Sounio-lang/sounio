/-
  FORMAL_PARITY two-sided equivalence between the frozen invertible matrix-code
  subtype and the constructive analytic ordered-basis subtype.

  This file is deliberately only the packaging layer.  The scan embedding and
  analytic basis encoder are checked in separate modules so later census proofs
  consume compiled interfaces instead of re-elaborating one monolithic proof.
-/
import SounioPireusGL4AnalyticBasisEncoder

namespace SounioPireusGL4AnalyticScanBijection

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusGL4ActionEnumeration
open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusGL4AnalyticCensus
open SounioPireusGL4AnalyticScanEmbedding
open SounioPireusGL4AnalyticBasisEncoder

structure AnalyticScanBijection where
  toAnalytic : GL4ScanEntry -> AnalyticBasisEntry
  toScan : AnalyticBasisEntry -> GL4ScanEntry
  toScanToAnalytic : ∀ entry, toScan (toAnalytic entry) = entry
  toAnalyticToScan : ∀ entry, toAnalytic (toScan entry) = entry

def analyticScanEquiv : AnalyticScanBijection :=
  { toAnalytic := fun entry =>
      ⟨basisOfScanEntry entry,
        basis_of_scan_entry_mem_analytic_ordered_bases entry⟩
  , toScan := scanEntryOfAnalyticBasis
  , toScanToAnalytic := scan_entry_of_basis_of_scan_entry
  , toAnalyticToScan := by
      intro entry
      apply Subtype.ext
      exact basis_of_scan_entry_of_analytic_basis entry }

theorem analytic_scan_to_basis_injective :
    Function.Injective analyticScanEquiv.toAnalytic := by
  intro left right equal
  rw [← analyticScanEquiv.toScanToAnalytic left,
    ← analyticScanEquiv.toScanToAnalytic right, equal]

theorem analytic_basis_to_scan_injective :
    Function.Injective analyticScanEquiv.toScan := by
  intro left right equal
  rw [← analyticScanEquiv.toAnalyticToScan left,
    ← analyticScanEquiv.toAnalyticToScan right, equal]

theorem eraseMany_nodup
    {Alpha : Type} [BEq Alpha] [LawfulBEq Alpha]
    (pool forbidden : List Alpha) (poolNodup : pool.Nodup) :
    (eraseMany pool forbidden).Nodup := by
  induction forbidden generalizing pool with
  | nil => simpa [eraseMany] using poolNodup
  | cons head tail ih =>
      rw [eraseMany_cons]
      exact ih (pool.erase head) (poolNodup.erase head)

theorem choicesOutside_nodup (span : List Lane) :
    (choicesOutside span).Nodup := by
  exact eraseMany_nodup laneUniverse span (List.nodup_finRange 16)

theorem nodup_flatMap_of_tagged_fibers
    {Tag Value : Type} {tags : List Tag} {fiber : Tag -> List Value}
    (tagsNodup : tags.Nodup)
    (fiberNodup : ∀ tag ∈ tags, (fiber tag).Nodup)
    (tagged : ∀ left ∈ tags, ∀ right ∈ tags, left ≠ right ->
      ∀ value, value ∈ fiber left -> value ∈ fiber right -> False) :
    (tags.flatMap fiber).Nodup := by
  induction tags with
  | nil => simp
  | cons head tail ih =>
      have tagFacts := List.nodup_cons.mp tagsNodup
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨fiberNodup head (by simp), ?_, ?_⟩
      · apply ih tagFacts.2
        · intro tag tagMem
          exact fiberNodup tag (List.mem_cons_of_mem head tagMem)
        · intro left leftMem right rightMem different value leftValue rightValue
          exact tagged left (List.mem_cons_of_mem head leftMem)
            right (List.mem_cons_of_mem head rightMem) different
            value leftValue rightValue
      · intro leftValue leftMem rightValue rightMem equal
        subst rightValue
        obtain ⟨rightTag, rightTagMem, rightValueMem⟩ :=
          List.mem_flatMap.mp rightMem
        have different : head ≠ rightTag := by
          intro same
          subst rightTag
          exact tagFacts.1 rightTagMem
        exact tagged head (by simp) rightTag
          (List.mem_cons_of_mem head rightTagMem) different
          leftValue leftMem rightValueMem

theorem fourth_completions_nodup (first second third : Lane) :
    (fourthCompletions first second third).Nodup := by
  unfold fourthCompletions
  apply nodup_map_of_injective
  · intro left right equal
    exact congrArg basisFourth equal
  · exact choicesOutside_nodup (spanThree first second third)

theorem fourth_completions_tagged
    (first second leftThird rightThird : Lane)
    (different : leftThird ≠ rightThird) (basis : OrderedBasis4)
    (leftMem : basis ∈ fourthCompletions first second leftThird)
    (rightMem : basis ∈ fourthCompletions first second rightThird) : False := by
  obtain ⟨leftFourth, _, leftEqual⟩ := List.mem_map.mp leftMem
  obtain ⟨rightFourth, _, rightEqual⟩ := List.mem_map.mp rightMem
  apply different
  calc
    leftThird = basisThird (first, second, leftThird, leftFourth) := rfl
    _ = basisThird basis := congrArg basisThird leftEqual
    _ = basisThird (first, second, rightThird, rightFourth) :=
      congrArg basisThird rightEqual.symm
    _ = rightThird := rfl

theorem third_completions_nodup (first second : Lane) :
    (thirdCompletions first second).Nodup := by
  unfold thirdCompletions
  apply nodup_flatMap_of_tagged_fibers
  · exact choicesOutside_nodup (spanTwo first second)
  · intro third _
    exact fourth_completions_nodup first second third
  · intro leftThird _ rightThird _ different basis leftMem rightMem
    exact fourth_completions_tagged first second leftThird rightThird
      different basis leftMem rightMem

theorem third_completions_tagged
    (first leftSecond rightSecond : Lane)
    (different : leftSecond ≠ rightSecond) (basis : OrderedBasis4)
    (leftMem : basis ∈ thirdCompletions first leftSecond)
    (rightMem : basis ∈ thirdCompletions first rightSecond) : False := by
  obtain ⟨leftThird, _, leftFourthMem⟩ := List.mem_flatMap.mp leftMem
  obtain ⟨rightThird, _, rightFourthMem⟩ := List.mem_flatMap.mp rightMem
  obtain ⟨leftFourth, _, leftEqual⟩ := List.mem_map.mp leftFourthMem
  obtain ⟨rightFourth, _, rightEqual⟩ := List.mem_map.mp rightFourthMem
  apply different
  calc
    leftSecond = basisSecond (first, leftSecond, leftThird, leftFourth) := rfl
    _ = basisSecond basis := congrArg basisSecond leftEqual
    _ = basisSecond (first, rightSecond, rightThird, rightFourth) :=
      congrArg basisSecond rightEqual.symm
    _ = rightSecond := rfl

theorem second_completions_nodup (first : Lane) :
    (secondCompletions first).Nodup := by
  unfold secondCompletions
  apply nodup_flatMap_of_tagged_fibers
  · exact choicesOutside_nodup (spanOne first)
  · intro second _
    exact third_completions_nodup first second
  · intro leftSecond _ rightSecond _ different basis leftMem rightMem
    exact third_completions_tagged first leftSecond rightSecond
      different basis leftMem rightMem

theorem second_completions_tagged
    (leftFirst rightFirst : Lane) (different : leftFirst ≠ rightFirst)
    (basis : OrderedBasis4)
    (leftMem : basis ∈ secondCompletions leftFirst)
    (rightMem : basis ∈ secondCompletions rightFirst) : False := by
  obtain ⟨leftSecond, _, leftThirdMem⟩ := List.mem_flatMap.mp leftMem
  obtain ⟨rightSecond, _, rightThirdMem⟩ := List.mem_flatMap.mp rightMem
  obtain ⟨leftThird, _, leftFourthMem⟩ := List.mem_flatMap.mp leftThirdMem
  obtain ⟨rightThird, _, rightFourthMem⟩ := List.mem_flatMap.mp rightThirdMem
  obtain ⟨leftFourth, _, leftEqual⟩ := List.mem_map.mp leftFourthMem
  obtain ⟨rightFourth, _, rightEqual⟩ := List.mem_map.mp rightFourthMem
  apply different
  calc
    leftFirst = basisFirst
        (leftFirst, leftSecond, leftThird, leftFourth) := rfl
    _ = basisFirst basis := congrArg basisFirst leftEqual
    _ = basisFirst
        (rightFirst, rightSecond, rightThird, rightFourth) :=
      congrArg basisFirst rightEqual.symm
    _ = rightFirst := rfl

theorem analytic_ordered_bases_nodup : analyticOrderedBases.Nodup := by
  unfold analyticOrderedBases
  apply nodup_flatMap_of_tagged_fibers
  · exact choicesOutside_nodup spanZero
  · intro first _
    exact second_completions_nodup first
  · intro leftFirst _ rightFirst _ different basis leftMem rightMem
    exact second_completions_tagged leftFirst rightFirst different
      basis leftMem rightMem

theorem nodup_of_map_nodup
    {Alpha Beta : Type} (function : Alpha -> Beta) :
    forall {values : List Alpha}, (values.map function).Nodup -> values.Nodup
  | [], _ => by simp
  | head :: tail, mappedNodup => by
      have facts := List.nodup_cons.mp mappedNodup
      apply List.nodup_cons.mpr
      constructor
      · intro headMem
        exact facts.1 (List.mem_map_of_mem headMem)
      · exact nodup_of_map_nodup function facts.2

-- This is the frozen carrier, not a request to evaluate its 65536-code filter.
-- The cardinality proof below uses only structural filter/subtype facts and the
-- analytic bijection; it never normalizes this list or imports its native count.
def frozenScanEntries : List GL4ScanEntry := invertibleMatrixCodes.attach

def analyticBasisEntries : List AnalyticBasisEntry := analyticOrderedBases.attach

theorem frozen_scan_entries_nodup : frozenScanEntries.Nodup := by
  apply nodup_of_map_nodup (fun entry : GL4ScanEntry => entry.val)
  have attachValues :
      invertibleMatrixCodes.attach.map
          (fun entry : GL4ScanEntry => entry.val) =
        invertibleMatrixCodes := by
    have attached :=
      List.attach_map_val (l := invertibleMatrixCodes) (f := id)
    rw [List.map_id] at attached
    exact attached
  rw [frozenScanEntries, attachValues]
  change (List.filter matrixInvertible (List.range matrixCodes)).Nodup
  exact List.filter_sublist.nodup List.nodup_range

theorem analytic_basis_entries_nodup : analyticBasisEntries.Nodup := by
  apply nodup_of_map_nodup (fun entry : AnalyticBasisEntry => entry.val)
  have attachValues :
      analyticOrderedBases.attach.map
          (fun entry : AnalyticBasisEntry => entry.val) =
        analyticOrderedBases := by
    have attached :=
      List.attach_map_val (l := analyticOrderedBases) (f := id)
    rw [List.map_id] at attached
    exact attached
  rw [analyticBasisEntries, attachValues]
  exact analytic_ordered_bases_nodup

theorem frozen_scan_length_eq_analytic_ordered_basis_length :
    frozenScanEntries.length = analyticOrderedBases.length := by
  have forwardNodup :
      (frozenScanEntries.map analyticScanEquiv.toAnalytic).Nodup :=
    nodup_map_of_injective analytic_scan_to_basis_injective
      frozen_scan_entries_nodup
  have forwardSubset :
      frozenScanEntries.map analyticScanEquiv.toAnalytic ⊆
        analyticBasisEntries := by
    intro entry entryMem
    exact List.mem_attach analyticOrderedBases entry
  have forwardLength := forwardNodup.length_le_of_subset forwardSubset
  have reverseNodup :
      (analyticBasisEntries.map analyticScanEquiv.toScan).Nodup :=
    nodup_map_of_injective analytic_basis_to_scan_injective
      analytic_basis_entries_nodup
  have reverseSubset :
      analyticBasisEntries.map analyticScanEquiv.toScan ⊆
        frozenScanEntries := by
    intro entry entryMem
    exact List.mem_attach invertibleMatrixCodes entry
  have reverseLength := reverseNodup.length_le_of_subset reverseSubset
  have forward : frozenScanEntries.length ≤ analyticBasisEntries.length := by
    simpa using forwardLength
  have reverse : analyticBasisEntries.length ≤ frozenScanEntries.length := by
    simpa using reverseLength
  have equal := Nat.le_antisymm forward reverse
  simpa only [analyticBasisEntries, List.length_attach] using equal

theorem frozen_scan_census_is_20160_analytically :
    frozenScanEntries.length = 20160 := by
  rw [frozen_scan_length_eq_analytic_ordered_basis_length,
    analytic_ordered_basis_census_is_20160]

end SounioPireusGL4AnalyticScanBijection
