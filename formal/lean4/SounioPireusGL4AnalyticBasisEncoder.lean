/-
  Structural FORMAL_PARITY encoder from a constructive ordered basis of
  F2^4 to the corresponding bounded 4x4 binary matrix code.

  The four basis lanes become the matrix columns.  The row nibbles are
  assembled analytically, and their one-hot images recover the original
  basis.  The second half of this module derives invertibility from the
  successive outside-span obligations carried by analyticOrderedBases.
-/
import SounioPireusGL4AnalyticScanEmbedding

namespace SounioPireusGL4AnalyticBasisEncoder

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusMatrixCodeXorEquiv
open SounioPireusGL4ActionEnumeration
open SounioPireusGL4AnalyticCensus
open SounioPireusGL4AnalyticScanEmbedding

def basisFirst (basis : OrderedBasis4) : Lane := basis.1
def basisSecond (basis : OrderedBasis4) : Lane := basis.2.1
def basisThird (basis : OrderedBasis4) : Lane := basis.2.2.1
def basisFourth (basis : OrderedBasis4) : Lane := basis.2.2.2

def encodedRow (basis : OrderedBasis4) (coordinate : Nat) : Nat :=
  encodeF2Word4 (rowBit (basisFirst basis) coordinate)
    (rowBit (basisSecond basis) coordinate)
    (rowBit (basisThird basis) coordinate)
    (rowBit (basisFourth basis) coordinate)

theorem encoded_row_lt_sixteen (basis : OrderedBasis4) (coordinate : Nat) :
    encodedRow basis coordinate < 16 := by
  unfold encodedRow encodeF2Word4
  have firstLt := (rowBit (basisFirst basis) coordinate).isLt
  have secondLt := (rowBit (basisSecond basis) coordinate).isLt
  have thirdLt := (rowBit (basisThird basis) coordinate).isLt
  have fourthLt := (rowBit (basisFourth basis) coordinate).isLt
  omega

def encodedRowLane (basis : OrderedBasis4) (coordinate : Nat) : Lane :=
  ⟨encodedRow basis coordinate, encoded_row_lt_sixteen basis coordinate⟩

private theorem encode_f2_word4_bit_zero :
    forall b0 b1 b2 b3 : F2Bit,
      bit (encodeF2Word4 b0 b1 b2 b3) 0 = b0.val := by
  decide

private theorem encode_f2_word4_bit_one :
    forall b0 b1 b2 b3 : F2Bit,
      bit (encodeF2Word4 b0 b1 b2 b3) 1 = b1.val := by
  decide

private theorem encode_f2_word4_bit_two :
    forall b0 b1 b2 b3 : F2Bit,
      bit (encodeF2Word4 b0 b1 b2 b3) 2 = b2.val := by
  decide

private theorem encode_f2_word4_bit_three :
    forall b0 b1 b2 b3 : F2Bit,
      bit (encodeF2Word4 b0 b1 b2 b3) 3 = b3.val := by
  decide

theorem encoded_row_components (basis : OrderedBasis4) (coordinate : Nat) :
    rowBit (encodedRowLane basis coordinate) 0 =
        rowBit (basisFirst basis) coordinate ∧
      rowBit (encodedRowLane basis coordinate) 1 =
        rowBit (basisSecond basis) coordinate ∧
      rowBit (encodedRowLane basis coordinate) 2 =
        rowBit (basisThird basis) coordinate ∧
      rowBit (encodedRowLane basis coordinate) 3 =
        rowBit (basisFourth basis) coordinate := by
  constructor
  · apply Fin.ext
    exact encode_f2_word4_bit_zero _ _ _ _
  constructor
  · apply Fin.ext
    exact encode_f2_word4_bit_one _ _ _ _
  constructor
  · apply Fin.ext
    exact encode_f2_word4_bit_two _ _ _ _
  · apply Fin.ext
    exact encode_f2_word4_bit_three _ _ _ _

def matrixCodeOfBasis (basis : OrderedBasis4) : Nat :=
  encodedRow basis 0 + 16 * encodedRow basis 1 +
    256 * encodedRow basis 2 + 4096 * encodedRow basis 3

theorem matrix_code_of_basis_lt_matrix_codes (basis : OrderedBasis4) :
    matrixCodeOfBasis basis < matrixCodes := by
  have row0Lt := encoded_row_lt_sixteen basis 0
  have row1Lt := encoded_row_lt_sixteen basis 1
  have row2Lt := encoded_row_lt_sixteen basis 2
  have row3Lt := encoded_row_lt_sixteen basis 3
  unfold matrixCodeOfBasis matrixCodes
  omega

def boundedMatrixCodeOfBasis (basis : OrderedBasis4) : Fin matrixCodes :=
  ⟨matrixCodeOfBasis basis, matrix_code_of_basis_lt_matrix_codes basis⟩

theorem matrix_row_code_of_basis_zero (basis : OrderedBasis4) :
    matrixRow (matrixCodeOfBasis basis) 0 = encodedRow basis 0 := by
  have row0Lt := encoded_row_lt_sixteen basis 0
  have row1Lt := encoded_row_lt_sixteen basis 1
  have row2Lt := encoded_row_lt_sixteen basis 2
  have row3Lt := encoded_row_lt_sixteen basis 3
  rw [matrixRow, show 4 * 0 = 0 by decide, Nat.shiftRight_zero,
    show 15 = 2 ^ 4 - 1 by decide, Nat.and_two_pow_sub_one_eq_mod]
  change matrixCodeOfBasis basis % 16 = encodedRow basis 0
  unfold matrixCodeOfBasis
  omega

theorem matrix_row_code_of_basis_one (basis : OrderedBasis4) :
    matrixRow (matrixCodeOfBasis basis) 1 = encodedRow basis 1 := by
  have row0Lt := encoded_row_lt_sixteen basis 0
  have row1Lt := encoded_row_lt_sixteen basis 1
  have row2Lt := encoded_row_lt_sixteen basis 2
  have row3Lt := encoded_row_lt_sixteen basis 3
  rw [matrixRow, show 4 * 1 = 4 by decide, Nat.shiftRight_eq_div_pow,
    show 15 = 2 ^ 4 - 1 by decide, Nat.and_two_pow_sub_one_eq_mod]
  change matrixCodeOfBasis basis / 16 % 16 = encodedRow basis 1
  unfold matrixCodeOfBasis
  omega

theorem matrix_row_code_of_basis_two (basis : OrderedBasis4) :
    matrixRow (matrixCodeOfBasis basis) 2 = encodedRow basis 2 := by
  have row0Lt := encoded_row_lt_sixteen basis 0
  have row1Lt := encoded_row_lt_sixteen basis 1
  have row2Lt := encoded_row_lt_sixteen basis 2
  have row3Lt := encoded_row_lt_sixteen basis 3
  rw [matrixRow, show 4 * 2 = 8 by decide, Nat.shiftRight_eq_div_pow,
    show 15 = 2 ^ 4 - 1 by decide, Nat.and_two_pow_sub_one_eq_mod]
  change matrixCodeOfBasis basis / 256 % 16 = encodedRow basis 2
  unfold matrixCodeOfBasis
  omega

theorem matrix_row_code_of_basis_three (basis : OrderedBasis4) :
    matrixRow (matrixCodeOfBasis basis) 3 = encodedRow basis 3 := by
  have row0Lt := encoded_row_lt_sixteen basis 0
  have row1Lt := encoded_row_lt_sixteen basis 1
  have row2Lt := encoded_row_lt_sixteen basis 2
  have row3Lt := encoded_row_lt_sixteen basis 3
  rw [matrixRow, show 4 * 3 = 12 by decide, Nat.shiftRight_eq_div_pow,
    show 15 = 2 ^ 4 - 1 by decide, Nat.and_two_pow_sub_one_eq_mod]
  change matrixCodeOfBasis basis / 4096 % 16 = encodedRow basis 3
  unfold matrixCodeOfBasis
  omega

private theorem matrix_row_lane_code_of_basis
    (basis : OrderedBasis4) (row : Fin 4) :
    matrixRowLane (matrixCodeOfBasis basis) row.val =
      encodedRowLane basis row.val := by
  rcases row with ⟨row, rowLt⟩
  apply Fin.ext
  have cases4 : row = 0 ∨ row = 1 ∨ row = 2 ∨ row = 3 := by
    omega
  rcases cases4 with equal | equal | equal | equal <;> subst row
  · exact matrix_row_code_of_basis_zero basis
  · exact matrix_row_code_of_basis_one basis
  · exact matrix_row_code_of_basis_two basis
  · exact matrix_row_code_of_basis_three basis

theorem matrix_parity_code_of_basis_lane1
    (basis : OrderedBasis4) (row : Fin 4) :
    matrixParity (matrixCodeOfBasis basis) row.val lane1 =
      rowBit (basisFirst basis) row.val := by
  rw [matrix_parity_lane1, matrix_row_lane_code_of_basis]
  exact (encoded_row_components basis row.val).1

theorem matrix_parity_code_of_basis_lane2
    (basis : OrderedBasis4) (row : Fin 4) :
    matrixParity (matrixCodeOfBasis basis) row.val lane2 =
      rowBit (basisSecond basis) row.val := by
  rw [matrix_parity_lane2, matrix_row_lane_code_of_basis]
  exact (encoded_row_components basis row.val).2.1

theorem matrix_parity_code_of_basis_lane4
    (basis : OrderedBasis4) (row : Fin 4) :
    matrixParity (matrixCodeOfBasis basis) row.val lane4 =
      rowBit (basisThird basis) row.val := by
  rw [matrix_parity_lane4, matrix_row_lane_code_of_basis]
  exact (encoded_row_components basis row.val).2.2.1

theorem matrix_parity_code_of_basis_lane8
    (basis : OrderedBasis4) (row : Fin 4) :
    matrixParity (matrixCodeOfBasis basis) row.val lane8 =
      rowBit (basisFourth basis) row.val := by
  rw [matrix_parity_lane8, matrix_row_lane_code_of_basis]
  exact (encoded_row_components basis row.val).2.2.2

theorem matrix_lane_map_code_of_basis_lane1 (basis : OrderedBasis4) :
    matrixLaneMap (matrixCodeOfBasis basis) lane1 = basisFirst basis := by
  apply Fin.ext
  change matrixApply (matrixCodeOfBasis basis) lane1.val = (basisFirst basis).val
  rw [matrix_apply_eq_encode_f2_word4,
    matrix_parity_code_of_basis_lane1 basis ⟨0, by decide⟩,
    matrix_parity_code_of_basis_lane1 basis ⟨1, by decide⟩,
    matrix_parity_code_of_basis_lane1 basis ⟨2, by decide⟩,
    matrix_parity_code_of_basis_lane1 basis ⟨3, by decide⟩]
  exact encode_row_bits (basisFirst basis)

theorem matrix_lane_map_code_of_basis_lane2 (basis : OrderedBasis4) :
    matrixLaneMap (matrixCodeOfBasis basis) lane2 = basisSecond basis := by
  apply Fin.ext
  change matrixApply (matrixCodeOfBasis basis) lane2.val = (basisSecond basis).val
  rw [matrix_apply_eq_encode_f2_word4,
    matrix_parity_code_of_basis_lane2 basis ⟨0, by decide⟩,
    matrix_parity_code_of_basis_lane2 basis ⟨1, by decide⟩,
    matrix_parity_code_of_basis_lane2 basis ⟨2, by decide⟩,
    matrix_parity_code_of_basis_lane2 basis ⟨3, by decide⟩]
  exact encode_row_bits (basisSecond basis)

theorem matrix_lane_map_code_of_basis_lane4 (basis : OrderedBasis4) :
    matrixLaneMap (matrixCodeOfBasis basis) lane4 = basisThird basis := by
  apply Fin.ext
  change matrixApply (matrixCodeOfBasis basis) lane4.val = (basisThird basis).val
  rw [matrix_apply_eq_encode_f2_word4,
    matrix_parity_code_of_basis_lane4 basis ⟨0, by decide⟩,
    matrix_parity_code_of_basis_lane4 basis ⟨1, by decide⟩,
    matrix_parity_code_of_basis_lane4 basis ⟨2, by decide⟩,
    matrix_parity_code_of_basis_lane4 basis ⟨3, by decide⟩]
  exact encode_row_bits (basisThird basis)

theorem matrix_lane_map_code_of_basis_lane8 (basis : OrderedBasis4) :
    matrixLaneMap (matrixCodeOfBasis basis) lane8 = basisFourth basis := by
  apply Fin.ext
  change matrixApply (matrixCodeOfBasis basis) lane8.val = (basisFourth basis).val
  rw [matrix_apply_eq_encode_f2_word4,
    matrix_parity_code_of_basis_lane8 basis ⟨0, by decide⟩,
    matrix_parity_code_of_basis_lane8 basis ⟨1, by decide⟩,
    matrix_parity_code_of_basis_lane8 basis ⟨2, by decide⟩,
    matrix_parity_code_of_basis_lane8 basis ⟨3, by decide⟩]
  exact encode_row_bits (basisFourth basis)

def spanFour (first second third fourth : Lane) : List Lane :=
  spanThree first second third ++
    (spanThree first second third).map (fun lane => lane ^^^ fourth)

theorem span_extend_closed
    (span : List Lane) (outside left right : Lane)
    (spanClosed : ∀ first ∈ span, ∀ second ∈ span,
      first ^^^ second ∈ span)
    (leftMem : left ∈ span ++ span.map (fun lane => lane ^^^ outside))
    (rightMem : right ∈ span ++ span.map (fun lane => lane ^^^ outside)) :
    left ^^^ right ∈ span ++ span.map (fun lane => lane ^^^ outside) := by
  rcases List.mem_append.mp leftMem with leftBaseMem | leftCosetMem
  · rcases List.mem_append.mp rightMem with rightBaseMem | rightCosetMem
    · exact List.mem_append.mpr (.inl
        (spanClosed left leftBaseMem right rightBaseMem))
    · obtain ⟨rightBase, rightBaseMem, rfl⟩ :=
        List.mem_map.mp rightCosetMem
      apply List.mem_append.mpr (.inr ?_)
      apply List.mem_map.mpr
      exact ⟨left ^^^ rightBase,
        spanClosed left leftBaseMem rightBase rightBaseMem,
        lane_xor_assoc left rightBase outside⟩
  · rcases List.mem_append.mp rightMem with rightBaseMem | rightCosetMem
    · obtain ⟨leftBase, leftBaseMem, rfl⟩ :=
        List.mem_map.mp leftCosetMem
      apply List.mem_append.mpr (.inr ?_)
      apply List.mem_map.mpr
      exact ⟨leftBase ^^^ right,
        spanClosed leftBase leftBaseMem right rightBaseMem,
        (lane_xor_swap_middle leftBase outside right).symm⟩
    · obtain ⟨leftBase, leftBaseMem, rfl⟩ :=
        List.mem_map.mp leftCosetMem
      obtain ⟨rightBase, rightBaseMem, rfl⟩ :=
        List.mem_map.mp rightCosetMem
      apply List.mem_append.mpr (.inl ?_)
      rw [lane_xor_cancel_coset]
      exact spanClosed leftBase leftBaseMem rightBase rightBaseMem

theorem spanThree_closed (first second third left right : Lane)
    (leftMem : left ∈ spanThree first second third)
    (rightMem : right ∈ spanThree first second third) :
    left ^^^ right ∈ spanThree first second third := by
  rw [spanThree] at leftMem rightMem ⊢
  exact span_extend_closed (spanTwo first second) third left right
    (fun spanLeft spanLeftMem spanRight spanRightMem =>
      spanTwo_closed first second spanLeft spanRight
        spanLeftMem spanRightMem)
    leftMem rightMem

theorem spanFour_nodup_of_outside
    {first second third fourth : Lane}
    (spanThreeNodup : (spanThree first second third).Nodup)
    (fourthOutside : fourth ∉ spanThree first second third) :
    (spanFour first second third fourth).Nodup := by
  exact nodup_append_xor_coset (spanThree first second third) fourth
    spanThreeNodup
    (fun left leftMem right rightMem =>
      spanThree_closed first second third left right leftMem rightMem)
    fourthOutside

theorem map_spanFour (code : Nat)
    (first second third fourth : Lane) :
    (spanFour first second third fourth).map (matrixLaneMap code) =
      spanFour (matrixLaneMap code first) (matrixLaneMap code second)
        (matrixLaneMap code third) (matrixLaneMap code fourth) := by
  rw [spanFour, List.map_append, map_spanThree, spanFour]
  congr 1
  rw [List.map_map, ← map_spanThree code first second third, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact matrix_lane_map_xor code lane fourth

theorem standard_span_four_is_lane_universe :
    spanFour lane1 lane2 lane4 lane8 = laneUniverse := by
  decide

theorem analytic_ordered_basis_membership_facts
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    basisFirst basis ∈ firstChoices ∧
      basisSecond basis ∈ secondChoices (basisFirst basis) ∧
      basisThird basis ∈
        thirdChoices (basisFirst basis) (basisSecond basis) ∧
      basisFourth basis ∈ fourthChoices (basisFirst basis)
        (basisSecond basis) (basisThird basis) := by
  rcases basis with ⟨first, second, third, fourth⟩
  simp only [analyticOrderedBases, List.mem_flatMap] at membership
  obtain ⟨firstWitness, firstMem, secondCompletionMem⟩ := membership
  simp only [secondCompletions, List.mem_flatMap] at secondCompletionMem
  obtain ⟨secondWitness, secondMem, thirdCompletionMem⟩ := secondCompletionMem
  simp only [thirdCompletions, List.mem_flatMap] at thirdCompletionMem
  obtain ⟨thirdWitness, thirdMem, fourthCompletionMem⟩ := thirdCompletionMem
  obtain ⟨fourthWitness, fourthMem, tupleEqual⟩ :=
    List.mem_map.mp fourthCompletionMem
  simp only [basisFirst, basisSecond, basisThird, basisFourth]
  have firstEqual : firstWitness = first := congrArg basisFirst tupleEqual
  have secondEqual : secondWitness = second := congrArg basisSecond tupleEqual
  have thirdEqual : thirdWitness = third := congrArg basisThird tupleEqual
  have fourthEqual : fourthWitness = fourth := congrArg basisFourth tupleEqual
  subst firstWitness
  subst secondWitness
  subst thirdWitness
  subst fourthWitness
  exact ⟨firstMem, secondMem, thirdMem, fourthMem⟩

theorem analytic_basis_span_four_nodup
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    (spanFour (basisFirst basis) (basisSecond basis)
      (basisThird basis) (basisFourth basis)).Nodup := by
  have facts := analytic_ordered_basis_membership_facts membership
  have spanThreeNodup := spanThree_nodup_of_third_choice
    facts.1 facts.2.1 facts.2.2.1
  have fourthOutside := mem_choicesOutside_not_mem_span
    (span := spanThree (basisFirst basis) (basisSecond basis)
      (basisThird basis)) facts.2.2.2
  exact spanFour_nodup_of_outside spanThreeNodup fourthOutside

theorem encoded_basis_lane_universe_image_nodup
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    (laneUniverse.map (matrixLaneMap (matrixCodeOfBasis basis))).Nodup := by
  rw [← standard_span_four_is_lane_universe,
    map_spanFour,
    matrix_lane_map_code_of_basis_lane1,
    matrix_lane_map_code_of_basis_lane2,
    matrix_lane_map_code_of_basis_lane4,
    matrix_lane_map_code_of_basis_lane8]
  exact analytic_basis_span_four_nodup membership

theorem injective_of_lane_universe_map_nodup
    {map : Lane -> Lane} (mappedNodup : (laneUniverse.map map).Nodup) :
    Function.Injective map := by
  intro left right imagesEqual
  apply Fin.ext
  have leftBound : left.val < (laneUniverse.map map).length := by
    simp [laneUniverse]
  have rightBound : right.val < (laneUniverse.map map).length := by
    simp [laneUniverse]
  have listValues :
      (laneUniverse.map map)[left.val] =
        (laneUniverse.map map)[right.val] := by
    simpa [laneUniverse] using imagesEqual
  exact (List.getElem_inj (h₀ := leftBound) (h₁ := rightBound)
    mappedNodup).mp listValues

theorem matrix_lane_map_code_of_analytic_basis_injective
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    Function.Injective (matrixLaneMap (matrixCodeOfBasis basis)) := by
  exact injective_of_lane_universe_map_nodup
    (encoded_basis_lane_universe_image_nodup membership)

theorem eraseDups_eq_self_of_nodup_nat :
    forall values : List Nat, values.Nodup -> values.eraseDups = values
  | [], _ => rfl
  | head :: tail, nodup => by
      have facts := List.nodup_cons.mp nodup
      rw [List.eraseDups_cons]
      have filterSelf :
          tail.filter (fun value => !(value == head)) = tail := by
        apply List.filter_eq_self.mpr
        intro value valueMem
        have notEqual : value ≠ head := by
          intro equal
          subst value
          exact facts.1 valueMem
        simp [notEqual]
      rw [filterSelf, eraseDups_eq_self_of_nodup_nat tail facts.2]

theorem matrix_images_code_of_analytic_basis_nodup
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    (matrixImages (matrixCodeOfBasis basis)).Nodup := by
  have laneNodup := encoded_basis_lane_universe_image_nodup membership
  have finValInjective : Function.Injective (fun lane : Lane => lane.val) := by
    intro left right equal
    exact Fin.ext equal
  have valueNodup := nodup_map_of_injective finValInjective laneNodup
  have finRangeValues :
      (List.finRange 16).map (fun lane : Lane => lane.val) =
        List.range 16 := by
    apply List.ext_getElem
    · simp
    · intro index leftBound rightBound
      simp
  have imageListsEqual :
      (laneUniverse.map (matrixLaneMap (matrixCodeOfBasis basis))).map
          (fun lane : Lane => lane.val) =
        matrixImages (matrixCodeOfBasis basis) := by
    rw [laneUniverse, List.map_map]
    change (List.finRange 16).map
        (fun lane : Lane => matrixApply (matrixCodeOfBasis basis) lane.val) =
      (List.range 16).map (matrixApply (matrixCodeOfBasis basis))
    rw [← finRangeValues, List.map_map]
    apply List.map_congr_left
    intro lane _
    rfl
  rw [← imageListsEqual]
  exact valueNodup

theorem matrix_code_of_analytic_basis_invertible
    {basis : OrderedBasis4} (membership : basis ∈ analyticOrderedBases) :
    matrixInvertible (matrixCodeOfBasis basis) = true := by
  unfold matrixInvertible
  rw [eraseDups_eq_self_of_nodup_nat _
    (matrix_images_code_of_analytic_basis_nodup membership)]
  simp [matrixImages, lanes]

abbrev AnalyticBasisEntry := {basis // basis ∈ analyticOrderedBases}

def matrixWitnessOfAnalyticBasis
    (entry : AnalyticBasisEntry) : InvertibleMatrixCode :=
  { code := boundedMatrixCodeOfBasis entry.val
  , invertible := matrix_code_of_analytic_basis_invertible entry.property }

def scanEntryOfAnalyticBasis (entry : AnalyticBasisEntry) : GL4ScanEntry :=
  ⟨matrixCodeOfBasis entry.val,
    every_invertible_4x4_code_is_in_the_scan
      (matrixCodeOfBasis entry.val)
      (matrix_code_of_basis_lt_matrix_codes entry.val)
      (matrix_code_of_analytic_basis_invertible entry.property)⟩

theorem basis_of_scan_entry_of_analytic_basis (entry : AnalyticBasisEntry) :
    basisOfScanEntry (scanEntryOfAnalyticBasis entry) = entry.val := by
  change (matrixLaneMap (matrixCodeOfBasis entry.val) lane1,
      matrixLaneMap (matrixCodeOfBasis entry.val) lane2,
      matrixLaneMap (matrixCodeOfBasis entry.val) lane4,
      matrixLaneMap (matrixCodeOfBasis entry.val) lane8) = entry.val
  rw [matrix_lane_map_code_of_basis_lane1,
    matrix_lane_map_code_of_basis_lane2,
    matrix_lane_map_code_of_basis_lane4,
    matrix_lane_map_code_of_basis_lane8]
  rcases entry.val with ⟨first, second, third, fourth⟩
  rfl

theorem scan_entry_of_basis_of_scan_entry (entry : GL4ScanEntry) :
    scanEntryOfAnalyticBasis
        ⟨basisOfScanEntry entry,
          basis_of_scan_entry_mem_analytic_ordered_bases entry⟩ = entry := by
  apply basis_of_scan_entry_injective
  exact basis_of_scan_entry_of_analytic_basis
    ⟨basisOfScanEntry entry,
      basis_of_scan_entry_mem_analytic_ordered_bases entry⟩

end SounioPireusGL4AnalyticBasisEncoder
