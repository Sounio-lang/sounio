/-
  Structural FORMAL_PARITY embedding of every frozen invertible matrix-code
  scan entry into the analytic ordered-basis census.

  The standard XOR basis (1, 2, 4, 8) is transported through the admitted
  matrix.  Explicit span transport plus injectivity proves that its images
  satisfy the four successive outside-span obligations.  Equality of those
  four images recovers all matrix rows and therefore the bounded 16-bit code.
  No 65536-code scan and no imported native census is consumed.
-/
import SounioPireusGL4AnalyticCensus

namespace SounioPireusGL4AnalyticScanEmbedding

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusMatrixCodeXorEquiv
open SounioPireusGL4ActionEnumeration
open SounioPireusGL4AnalyticCensus

theorem mem_eraseMany_of_mem_of_not_mem
    {Alpha : Type} [BEq Alpha] [LawfulBEq Alpha]
    {pool forbidden : List Alpha} {value : Alpha}
    (poolMem : value ∈ pool) (notForbidden : value ∉ forbidden) :
    value ∈ eraseMany pool forbidden := by
  induction forbidden generalizing pool with
  | nil => simpa [eraseMany] using poolMem
  | cons head tail ih =>
      rw [eraseMany_cons]
      apply ih
      · have valueNe : value ≠ head := by
          intro equal
          subst value
          exact notForbidden (by simp)
        exact (List.mem_erase_of_ne valueNe).2 poolMem
      · intro tailMem
        exact notForbidden (List.mem_cons_of_mem head tailMem)

theorem mem_choicesOutside_of_not_mem_span
    {span : List Lane} {lane : Lane} (outside : lane ∉ span) :
    lane ∈ choicesOutside span := by
  exact mem_eraseMany_of_mem_of_not_mem (List.mem_finRange lane) outside

theorem map_spanZero (code : Nat) :
    spanZero.map (matrixLaneMap code) = spanZero := by
  simp [spanZero, matrix_lane_map_zero]

theorem map_spanOne (code : Nat) (first : Lane) :
    (spanOne first).map (matrixLaneMap code) =
      spanOne (matrixLaneMap code first) := by
  simp [spanOne, matrix_lane_map_zero]

theorem map_spanTwo (code : Nat) (first second : Lane) :
    (spanTwo first second).map (matrixLaneMap code) =
      spanTwo (matrixLaneMap code first) (matrixLaneMap code second) := by
  rw [spanTwo, List.map_append, map_spanOne, spanTwo]
  congr 1
  rw [List.map_map, ← map_spanOne code first, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact matrix_lane_map_xor code lane second

theorem map_spanThree (code : Nat) (first second third : Lane) :
    (spanThree first second third).map (matrixLaneMap code) =
      spanThree (matrixLaneMap code first) (matrixLaneMap code second)
        (matrixLaneMap code third) := by
  rw [spanThree, List.map_append, map_spanTwo, spanThree]
  congr 1
  rw [List.map_map, ← map_spanTwo code first second, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact matrix_lane_map_xor code lane third

theorem image_not_mem_map_of_injective
    {Alpha Beta : Type} {map : Alpha -> Beta} {value : Alpha}
    {values : List Alpha} (injective : Function.Injective map)
    (outside : value ∉ values) : map value ∉ values.map map := by
  intro membership
  obtain ⟨preimage, preimageMem, equal⟩ := List.mem_map.mp membership
  have same : preimage = value := injective equal
  subst preimage
  exact outside preimageMem

theorem standard_first_outside : lane1 ∉ spanZero := by decide
theorem standard_second_outside : lane2 ∉ spanOne lane1 := by decide
theorem standard_third_outside : lane4 ∉ spanTwo lane1 lane2 := by decide
theorem standard_fourth_outside : lane8 ∉ spanThree lane1 lane2 lane4 := by decide

def basisOfScanEntry (entry : GL4ScanEntry) : OrderedBasis4 :=
  (matrixLaneMap entry.val lane1,
    matrixLaneMap entry.val lane2,
    matrixLaneMap entry.val lane4,
    matrixLaneMap entry.val lane8)

theorem basis_of_scan_entry_first_mem (entry : GL4ScanEntry) :
    matrixLaneMap entry.val lane1 ∈ firstChoices := by
  have outsideMapped := image_not_mem_map_of_injective
    (matrix_lane_map_injective (matrixWitnessOfScanEntry entry))
    standard_first_outside
  rw [map_spanZero] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_scan_entry_second_mem (entry : GL4ScanEntry) :
    matrixLaneMap entry.val lane2 ∈
      secondChoices (matrixLaneMap entry.val lane1) := by
  have outsideMapped := image_not_mem_map_of_injective
    (matrix_lane_map_injective (matrixWitnessOfScanEntry entry))
    standard_second_outside
  rw [map_spanOne] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_scan_entry_third_mem (entry : GL4ScanEntry) :
    matrixLaneMap entry.val lane4 ∈
      thirdChoices (matrixLaneMap entry.val lane1)
        (matrixLaneMap entry.val lane2) := by
  have outsideMapped := image_not_mem_map_of_injective
    (matrix_lane_map_injective (matrixWitnessOfScanEntry entry))
    standard_third_outside
  rw [map_spanTwo] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_scan_entry_fourth_mem (entry : GL4ScanEntry) :
    matrixLaneMap entry.val lane8 ∈
      fourthChoices (matrixLaneMap entry.val lane1)
        (matrixLaneMap entry.val lane2) (matrixLaneMap entry.val lane4) := by
  have outsideMapped := image_not_mem_map_of_injective
    (matrix_lane_map_injective (matrixWitnessOfScanEntry entry))
    standard_fourth_outside
  rw [map_spanThree] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_scan_entry_mem_analytic_ordered_bases
    (entry : GL4ScanEntry) : basisOfScanEntry entry ∈ analyticOrderedBases := by
  rw [analyticOrderedBases, List.mem_flatMap]
  refine ⟨matrixLaneMap entry.val lane1,
    basis_of_scan_entry_first_mem entry, ?_⟩
  rw [secondCompletions, List.mem_flatMap]
  refine ⟨matrixLaneMap entry.val lane2,
    basis_of_scan_entry_second_mem entry, ?_⟩
  rw [thirdCompletions, List.mem_flatMap]
  refine ⟨matrixLaneMap entry.val lane4,
    basis_of_scan_entry_third_mem entry, ?_⟩
  rw [fourthCompletions, List.mem_map]
  exact ⟨matrixLaneMap entry.val lane8,
    basis_of_scan_entry_fourth_mem entry, rfl⟩

def rowBit (row : Lane) (coordinate : Nat) : F2Bit :=
  ⟨bit row.val coordinate, Nat.mod_lt _ (by decide)⟩

theorem encode_row_bits (row : Lane) :
    encodeF2Word4 (rowBit row 0) (rowBit row 1)
      (rowBit row 2) (rowBit row 3) = row.val := by
  rcases row with ⟨value, valueLt⟩
  have cases16 : value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨
      value = 4 ∨ value = 5 ∨ value = 6 ∨ value = 7 ∨
      value = 8 ∨ value = 9 ∨ value = 10 ∨ value = 11 ∨
      value = 12 ∨ value = 13 ∨ value = 14 ∨ value = 15 := by
    omega
  rcases cases16 with h | h | h | h | h | h | h | h |
      h | h | h | h | h | h | h | h <;> subst value <;>
    simp [rowBit, encodeF2Word4, bit]

theorem parity4_and_lane1 : forall row : Lane,
    parity4 (row.val &&& lane1.val) = bit row.val 0 := by decide
theorem parity4_and_lane2 : forall row : Lane,
    parity4 (row.val &&& lane2.val) = bit row.val 1 := by decide
theorem parity4_and_lane4 : forall row : Lane,
    parity4 (row.val &&& lane4.val) = bit row.val 2 := by decide
theorem parity4_and_lane8 : forall row : Lane,
    parity4 (row.val &&& lane8.val) = bit row.val 3 := by decide

@[simp] theorem matrix_parity_lane1 (code row : Nat) :
    matrixParity code row lane1 = rowBit (matrixRowLane code row) 0 := by
  apply Fin.ext
  exact parity4_and_lane1 (matrixRowLane code row)

@[simp] theorem matrix_parity_lane2 (code row : Nat) :
    matrixParity code row lane2 = rowBit (matrixRowLane code row) 1 := by
  apply Fin.ext
  exact parity4_and_lane2 (matrixRowLane code row)

@[simp] theorem matrix_parity_lane4 (code row : Nat) :
    matrixParity code row lane4 = rowBit (matrixRowLane code row) 2 := by
  apply Fin.ext
  exact parity4_and_lane4 (matrixRowLane code row)

@[simp] theorem matrix_parity_lane8 (code row : Nat) :
    matrixParity code row lane8 = rowBit (matrixRowLane code row) 3 := by
  apply Fin.ext
  exact parity4_and_lane8 (matrixRowLane code row)

theorem encode_f2_word4_eq_iff :
    forall a0 a1 a2 a3 b0 b1 b2 b3 : F2Bit,
      encodeF2Word4 a0 a1 a2 a3 = encodeF2Word4 b0 b1 b2 b3 ↔
        a0 = b0 ∧ a1 = b1 ∧ a2 = b2 ∧ a3 = b3 := by
  decide

theorem matrix_parities_equal_of_lane_image_equal
    {leftCode rightCode : Nat} {lane : Lane}
    (equal : matrixLaneMap leftCode lane = matrixLaneMap rightCode lane) :
    matrixParity leftCode 0 lane = matrixParity rightCode 0 lane ∧
      matrixParity leftCode 1 lane = matrixParity rightCode 1 lane ∧
      matrixParity leftCode 2 lane = matrixParity rightCode 2 lane ∧
      matrixParity leftCode 3 lane = matrixParity rightCode 3 lane := by
  have valueEqual := congrArg Fin.val equal
  change matrixApply leftCode lane.val = matrixApply rightCode lane.val at valueEqual
  rw [matrix_apply_eq_encode_f2_word4,
    matrix_apply_eq_encode_f2_word4] at valueEqual
  exact (encode_f2_word4_eq_iff _ _ _ _ _ _ _ _).mp valueEqual

theorem matrix_row_eq_of_basis_parities
    {leftCode rightCode row : Nat}
    (lane1Equal : matrixParity leftCode row lane1 =
      matrixParity rightCode row lane1)
    (lane2Equal : matrixParity leftCode row lane2 =
      matrixParity rightCode row lane2)
    (lane4Equal : matrixParity leftCode row lane4 =
      matrixParity rightCode row lane4)
    (lane8Equal : matrixParity leftCode row lane8 =
      matrixParity rightCode row lane8) :
    matrixRow leftCode row = matrixRow rightCode row := by
  have leftEncoded :
      encodeF2Word4 (matrixParity leftCode row lane1)
        (matrixParity leftCode row lane2)
        (matrixParity leftCode row lane4)
        (matrixParity leftCode row lane8) = matrixRow leftCode row := by
    simpa [matrixRowLane] using encode_row_bits (matrixRowLane leftCode row)
  have rightEncoded :
      encodeF2Word4 (matrixParity rightCode row lane1)
        (matrixParity rightCode row lane2)
        (matrixParity rightCode row lane4)
        (matrixParity rightCode row lane8) = matrixRow rightCode row := by
    simpa [matrixRowLane] using encode_row_bits (matrixRowLane rightCode row)
  rw [← leftEncoded, ← rightEncoded, lane1Equal, lane2Equal,
    lane4Equal, lane8Equal]

theorem matrix_rows_determine_bounded_code
    (leftCode rightCode : Fin matrixCodes)
    (row0Equal : matrixRow leftCode.val 0 = matrixRow rightCode.val 0)
    (row1Equal : matrixRow leftCode.val 1 = matrixRow rightCode.val 1)
    (row2Equal : matrixRow leftCode.val 2 = matrixRow rightCode.val 2)
    (row3Equal : matrixRow leftCode.val 3 = matrixRow rightCode.val 3) :
    leftCode = rightCode := by
  apply Fin.ext
  apply Nat.eq_of_testBit_eq
  intro coordinate
  by_cases within : coordinate < 16
  · have cases16 : coordinate = 0 ∨ coordinate = 1 ∨
        coordinate = 2 ∨ coordinate = 3 ∨ coordinate = 4 ∨
        coordinate = 5 ∨ coordinate = 6 ∨ coordinate = 7 ∨
        coordinate = 8 ∨ coordinate = 9 ∨ coordinate = 10 ∨
        coordinate = 11 ∨ coordinate = 12 ∨ coordinate = 13 ∨
        coordinate = 14 ∨ coordinate = 15 := by omega
    rcases cases16 with h | h | h | h | h | h | h | h |
        h | h | h | h | h | h | h | h <;> subst coordinate
    · simpa [matrixRow] using congrArg (fun value => value.testBit 0) row0Equal
    · simpa [matrixRow, show Nat.testBit 15 1 = true by decide] using congrArg (fun value => value.testBit 1) row0Equal
    · simpa [matrixRow, show Nat.testBit 15 2 = true by decide] using congrArg (fun value => value.testBit 2) row0Equal
    · simpa [matrixRow, show Nat.testBit 15 3 = true by decide] using congrArg (fun value => value.testBit 3) row0Equal
    · simpa [matrixRow] using congrArg (fun value => value.testBit 0) row1Equal
    · simpa [matrixRow, show Nat.testBit 15 1 = true by decide] using congrArg (fun value => value.testBit 1) row1Equal
    · simpa [matrixRow, show Nat.testBit 15 2 = true by decide] using congrArg (fun value => value.testBit 2) row1Equal
    · simpa [matrixRow, show Nat.testBit 15 3 = true by decide] using congrArg (fun value => value.testBit 3) row1Equal
    · simpa [matrixRow] using congrArg (fun value => value.testBit 0) row2Equal
    · simpa [matrixRow, show Nat.testBit 15 1 = true by decide] using congrArg (fun value => value.testBit 1) row2Equal
    · simpa [matrixRow, show Nat.testBit 15 2 = true by decide] using congrArg (fun value => value.testBit 2) row2Equal
    · simpa [matrixRow, show Nat.testBit 15 3 = true by decide] using congrArg (fun value => value.testBit 3) row2Equal
    · simpa [matrixRow] using congrArg (fun value => value.testBit 0) row3Equal
    · simpa [matrixRow, show Nat.testBit 15 1 = true by decide] using congrArg (fun value => value.testBit 1) row3Equal
    · simpa [matrixRow, show Nat.testBit 15 2 = true by decide] using congrArg (fun value => value.testBit 2) row3Equal
    · simpa [matrixRow, show Nat.testBit 15 3 = true by decide] using congrArg (fun value => value.testBit 3) row3Equal
  · have sixteenLe : 16 ≤ coordinate := Nat.le_of_not_lt within
    have powers : 2 ^ 16 ≤ 2 ^ coordinate :=
      Nat.pow_le_pow_right (by decide) sixteenLe
    have leftLt : leftCode.val < 2 ^ coordinate :=
      Nat.lt_of_lt_of_le leftCode.isLt powers
    have rightLt : rightCode.val < 2 ^ coordinate :=
      Nat.lt_of_lt_of_le rightCode.isLt powers
    rw [Nat.testBit_lt_two_pow leftLt, Nat.testBit_lt_two_pow rightLt]

theorem basis_of_scan_entry_injective :
    Function.Injective basisOfScanEntry := by
  intro leftEntry rightEntry basisEqual
  have lane1Equal := congrArg (fun basis : OrderedBasis4 => basis.1) basisEqual
  have lane2Equal := congrArg (fun basis : OrderedBasis4 => basis.2.1) basisEqual
  have lane4Equal := congrArg (fun basis : OrderedBasis4 => basis.2.2.1) basisEqual
  have lane8Equal := congrArg (fun basis : OrderedBasis4 => basis.2.2.2) basisEqual
  simp only [basisOfScanEntry] at lane1Equal lane2Equal lane4Equal lane8Equal
  have parities1 := matrix_parities_equal_of_lane_image_equal lane1Equal
  have parities2 := matrix_parities_equal_of_lane_image_equal lane2Equal
  have parities4 := matrix_parities_equal_of_lane_image_equal lane4Equal
  have parities8 := matrix_parities_equal_of_lane_image_equal lane8Equal
  have row0Equal := matrix_row_eq_of_basis_parities
    parities1.1 parities2.1 parities4.1 parities8.1
  have row1Equal := matrix_row_eq_of_basis_parities
    parities1.2.1 parities2.2.1 parities4.2.1 parities8.2.1
  have row2Equal := matrix_row_eq_of_basis_parities
    parities1.2.2.1 parities2.2.2.1 parities4.2.2.1 parities8.2.2.1
  have row3Equal := matrix_row_eq_of_basis_parities
    parities1.2.2.2 parities2.2.2.2 parities4.2.2.2 parities8.2.2.2
  apply Subtype.ext
  exact congrArg Fin.val (matrix_rows_determine_bounded_code
    (matrixWitnessOfScanEntry leftEntry).code
    (matrixWitnessOfScanEntry rightEntry).code
    row0Equal row1Equal row2Equal row3Equal)

end SounioPireusGL4AnalyticScanEmbedding
