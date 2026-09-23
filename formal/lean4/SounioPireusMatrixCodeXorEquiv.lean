/-
  FORMAL_PARITY bridge from the frozen V13 4x4 binary matrix-code predicate
  to the XOR-linear lane equivalences used by the gauge-descent proof.

  The matrix application is decomposed into four F2 row parities. Invertible
  means that its sixteen lane images are distinct; on Fin 16 that gives a
  computable inverse by the index of each target in the image list. Thus every
  admitted matrix-code witness yields an XorLaneEquiv and a LinearSwapAction.

  This file does not yet package the complete 20160-code list, prove that the
  40320-view fold is a minimum, or close V13 Target-03.
-/
import SounioPireusBasisFixedGaugeRebase
import SounioPireusOperatorOrbitCanonicalization

namespace SounioPireusMatrixCodeXorEquiv

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusGaugeCoboundaryAction

abbrev F2Bit := Fin 2

def encodeF2Word4 (b0 b1 b2 b3 : F2Bit) : Nat :=
  b0.val + 2 * b1.val + 4 * b2.val + 8 * b3.val

theorem encode_f2_word4_xor :
    forall a0 a1 a2 a3 b0 b1 b2 b3 : F2Bit,
      encodeF2Word4 (a0 ^^^ b0) (a1 ^^^ b1) (a2 ^^^ b2) (a3 ^^^ b3) =
        (encodeF2Word4 a0 a1 a2 a3 ^^^ encodeF2Word4 b0 b1 b2 b3) := by
  decide

theorem parity4_lt_two (value : Nat) : parity4 value < 2 := by
  exact Nat.mod_lt _ (by decide)

def matrixParity (code row : Nat) (lane : Lane) : F2Bit :=
  ⟨parity4 (matrixRow code row &&& lane.val), parity4_lt_two _⟩

def matrixRowLane (code row : Nat) : Lane :=
  ⟨matrixRow code row,
    Nat.lt_of_le_of_lt Nat.and_le_right (by decide)⟩

theorem parity4_and_xor_on_lanes :
    forall row left right : Lane,
      parity4 (row.val &&& (left.val ^^^ right.val)) =
        (parity4 (row.val &&& left.val) ^^^
          parity4 (row.val &&& right.val)) := by
  decide

theorem matrix_parity_xor
    (code row : Nat) (left right : Lane) :
    matrixParity code row (left ^^^ right) =
      (matrixParity code row left ^^^ matrixParity code row right) := by
  apply Fin.ext
  rw [Fin.xor_val_of_two_pow (w := 1)]
  simpa [matrixParity, matrixRowLane, lane_xor_val] using
    parity4_and_xor_on_lanes (matrixRowLane code row) left right

theorem matrix_apply_eq_encode_f2_word4 (code : Nat) (lane : Lane) :
    matrixApply code lane.val =
      encodeF2Word4 (matrixParity code 0 lane) (matrixParity code 1 lane)
        (matrixParity code 2 lane) (matrixParity code 3 lane) := by
  have range4 : List.range 4 = [0, 1, 2, 3] := by decide
  unfold matrixApply
  rw [show bits = 4 by rfl, range4]
  simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.pow_zero,
    Nat.mul_one, Nat.reducePow]
  simp [encodeF2Word4, matrixParity, Nat.mul_comm, Nat.add_assoc]

theorem matrix_apply_lt_sixteen (code vector : Nat) :
    matrixApply code vector < 16 := by
  have range4 : List.range 4 = [0, 1, 2, 3] := by decide
  unfold matrixApply
  rw [show bits = 4 by rfl, range4]
  simp only [List.foldl_cons, List.foldl_nil, Nat.zero_add, Nat.pow_zero,
    Nat.mul_one, Nat.reducePow]
  have h0 := parity4_lt_two (matrixRow code 0 &&& vector)
  have h1 := parity4_lt_two (matrixRow code 1 &&& vector)
  have h2 := parity4_lt_two (matrixRow code 2 &&& vector)
  have h3 := parity4_lt_two (matrixRow code 3 &&& vector)
  omega

theorem matrix_apply_xor (code : Nat) (left right : Lane) :
    matrixApply code (left ^^^ right).val =
      (matrixApply code left.val ^^^ matrixApply code right.val) := by
  rw [matrix_apply_eq_encode_f2_word4, matrix_apply_eq_encode_f2_word4,
    matrix_apply_eq_encode_f2_word4]
  rw [matrix_parity_xor, matrix_parity_xor, matrix_parity_xor,
    matrix_parity_xor]
  exact encode_f2_word4_xor _ _ _ _ _ _ _ _

def matrixLaneMap (code : Nat) (lane : Lane) : Lane :=
  ⟨matrixApply code lane.val, matrix_apply_lt_sixteen _ _⟩

theorem matrix_lane_map_zero (code : Nat) : matrixLaneMap code 0 = 0 := by
  apply Fin.ext
  change matrixApply code 0 = 0
  have range4 : List.range 4 = [0, 1, 2, 3] := by decide
  unfold matrixApply
  rw [show bits = 4 by rfl, range4]
  simp [parity4, bit]

theorem matrix_lane_map_xor (code : Nat) (left right : Lane) :
    matrixLaneMap code (left ^^^ right) =
      (matrixLaneMap code left ^^^ matrixLaneMap code right) := by
  apply Fin.ext
  rw [Fin.xor_val_of_two_pow (w := 4)]
  exact matrix_apply_xor code left right

theorem eraseDups_length_le_nat :
    forall values : List Nat, values.eraseDups.length <= values.length
  | [] => by simp
  | head :: tail => by
      rw [List.eraseDups_cons]
      simp only [List.length_cons, Nat.succ_le_succ_iff]
      exact Nat.le_trans
        (eraseDups_length_le_nat (tail.filter fun value => !(value == head)))
        (List.length_filter_le _ _)
termination_by values => values.length
decreasing_by
  exact Nat.lt_succ_of_le (List.length_filter_le _ _)

theorem nodup_of_eraseDups_length_eq_nat {values : List Nat}
    (sameLength : values.eraseDups.length = values.length) :
    values.Nodup := by
  induction values with
  | nil => simp
  | cons head tail ih =>
    rw [List.eraseDups_cons] at sameLength
    simp only [List.length_cons, Nat.succ.injEq] at sameLength
    let filtered := tail.filter fun value => !(value == head)
    change filtered.eraseDups.length = tail.length at sameLength
    have erasedLe : filtered.eraseDups.length <= filtered.length :=
      eraseDups_length_le_nat filtered
    have filteredLe : filtered.length <= tail.length :=
      List.length_filter_le _ _
    have filteredLength : filtered.length = tail.length :=
      Nat.le_antisymm filteredLe (by
        rw [← sameLength]
        exact erasedLe)
    have filteredSelf : filtered = tail :=
      List.filter_eq_self.mpr
        (List.length_filter_eq_length_iff.mp filteredLength)
    have tailLength : tail.eraseDups.length = tail.length := by
      rw [filteredSelf] at sameLength
      exact sameLength
    apply List.nodup_cons.mpr
    refine ⟨?_, ih tailLength⟩
    intro headMem
    have headPass :=
      (List.length_filter_eq_length_iff.mp filteredLength) head headMem
    have beqSelf : (head == head) = true := by
      change decide (head = head) = true
      exact decide_eq_true rfl
    rw [beqSelf] at headPass
    exact Bool.noConfusion headPass

theorem matrix_images_nodup_of_invertible {code : Nat}
    (invertible : matrixInvertible code = true) :
    (matrixImages code).Nodup := by
  apply nodup_of_eraseDups_length_eq_nat
  have imageCount : (matrixImages code).eraseDups.length = lanes := by
    simpa [matrixInvertible] using invertible
  simpa [matrixImages, lanes] using imageCount

theorem matrix_images_subset_range (code : Nat) :
    matrixImages code ⊆ List.range lanes := by
  intro image inImages
  rcases List.mem_map.mp inImages with ⟨vector, _, rfl⟩
  simp only [List.mem_range]
  exact matrix_apply_lt_sixteen code vector

theorem matrix_images_get (code index : Nat)
    (bound : index < (matrixImages code).length) :
    (matrixImages code)[index] = matrixApply code index := by
  simp [matrixImages, lanes]

def natMemDecidable (target : Nat) :
    (values : List Nat) -> Decidable (target ∈ values)
  | [] => isFalse (by simp)
  | head :: tail =>
      if equal : target = head then
        isTrue (by simp [equal])
      else
        match natMemDecidable target tail with
        | isTrue present => isTrue (by simp [present])
        | isFalse missing => isFalse (by simp [equal, missing])

theorem every_lane_mem_matrix_images {code : Nat}
    (invertible : matrixInvertible code = true) (target : Lane) :
    target.val ∈ matrixImages code := by
  cases natMemDecidable target.val (matrixImages code) with
  | isTrue present => exact present
  | isFalse missing =>
    have withTargetNodup : (target.val :: matrixImages code).Nodup :=
      List.nodup_cons.mpr
        ⟨missing, matrix_images_nodup_of_invertible invertible⟩
    have withTargetSubset :
        target.val :: matrixImages code ⊆ List.range lanes := by
      intro image membership
      simp only [List.mem_cons] at membership
      rcases membership with rfl | inImages
      · rw [List.mem_range]
        exact target.isLt
      · exact matrix_images_subset_range code inImages
    have impossible :=
      withTargetNodup.length_le_of_subset withTargetSubset
    have seventeenLeSixteen : 17 <= 16 := by
      simpa only [List.length_cons, matrixImages, List.length_map,
        List.length_range, lanes] using impossible
    omega

structure InvertibleMatrixCode where
  code : Fin matrixCodes
  invertible : matrixInvertible code.val = true

theorem matrix_lane_map_injective
    (matrix : InvertibleMatrixCode) :
    Function.Injective (matrixLaneMap matrix.code.val) := by
  intro left right imagesEqual
  apply Fin.ext
  have imageValues :
      matrixApply matrix.code.val left.val =
        matrixApply matrix.code.val right.val :=
    congrArg Fin.val imagesEqual
  have leftBound : left.val < (matrixImages matrix.code.val).length := by
    exact left.isLt
  have rightBound : right.val < (matrixImages matrix.code.val).length := by
    exact right.isLt
  have listValues :
      (matrixImages matrix.code.val)[left.val] =
        (matrixImages matrix.code.val)[right.val] := by
    calc
      _ = matrixApply matrix.code.val left.val :=
        matrix_images_get _ _ leftBound
      _ = matrixApply matrix.code.val right.val := imageValues
      _ = _ := (matrix_images_get _ _ rightBound).symm
  exact (List.getElem_inj (h₀ := leftBound) (h₁ := rightBound)
    (matrix_images_nodup_of_invertible matrix.invertible)).mp listValues

def firstPreimage16 (map : Lane -> Lane) (target : Lane) : Lane :=
  if map lane0 = target then lane0 else
  if map lane1 = target then lane1 else
  if map lane2 = target then lane2 else
  if map lane3 = target then lane3 else
  if map lane4 = target then lane4 else
  if map lane5 = target then lane5 else
  if map lane6 = target then lane6 else
  if map lane7 = target then lane7 else
  if map lane8 = target then lane8 else
  if map lane9 = target then lane9 else
  if map lane10 = target then lane10 else
  if map lane11 = target then lane11 else
  if map lane12 = target then lane12 else
  if map lane13 = target then lane13 else
  if map lane14 = target then lane14 else lane15

theorem firstPreimage16_left_inverse (map : Lane -> Lane)
    (injective : Function.Injective map) (lane : Lane) :
    firstPreimage16 map (map lane) = lane := by
  rcases lane with ⟨value, valueLt⟩
  have cases16 : value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨
      value = 4 ∨ value = 5 ∨ value = 6 ∨ value = 7 ∨
      value = 8 ∨ value = 9 ∨ value = 10 ∨ value = 11 ∨
      value = 12 ∨ value = 13 ∨ value = 14 ∨ value = 15 := by
    omega
  rcases cases16 with h | h | h | h | h | h | h | h |
      h | h | h | h | h | h | h | h <;> subst value <;>
    simp [firstPreimage16, injective.eq_iff, lane0, lane1, lane2, lane3,
      lane4, lane5, lane6, lane7, lane8, lane9, lane10, lane11, lane12,
      lane13, lane14, lane15]

def matrixLaneInverse
    (matrix : InvertibleMatrixCode) (target : Lane) : Lane :=
  firstPreimage16 (matrixLaneMap matrix.code.val) target

theorem matrix_lane_left_inverse
    (matrix : InvertibleMatrixCode) (lane : Lane) :
    matrixLaneInverse matrix (matrixLaneMap matrix.code.val lane) = lane := by
  exact firstPreimage16_left_inverse _ (matrix_lane_map_injective matrix) lane

theorem matrix_lane_right_inverse
    (matrix : InvertibleMatrixCode) (target : Lane) :
    matrixLaneMap matrix.code.val (matrixLaneInverse matrix target) = target := by
  have inImages := every_lane_mem_matrix_images matrix.invertible target
  rcases List.mem_map.mp inImages with
    ⟨preimage, preimageInRange, imageEquals⟩
  have preimageLt : preimage < 16 := by
    exact List.mem_range.mp preimageInRange
  let preimageLane : Lane := ⟨preimage, preimageLt⟩
  have mapped : matrixLaneMap matrix.code.val preimageLane = target := by
    apply Fin.ext
    exact imageEquals
  calc
    _ = matrixLaneMap matrix.code.val
        (matrixLaneInverse matrix (matrixLaneMap matrix.code.val preimageLane)) := by
      rw [mapped]
    _ = matrixLaneMap matrix.code.val preimageLane := by
      rw [matrix_lane_left_inverse]
    _ = target := mapped

theorem matrix_lane_inverse_zero (matrix : InvertibleMatrixCode) :
    matrixLaneInverse matrix 0 = 0 := by
  apply matrix_lane_map_injective matrix
  rw [matrix_lane_right_inverse, matrix_lane_map_zero]

theorem matrix_lane_inverse_xor
    (matrix : InvertibleMatrixCode) (left right : Lane) :
    matrixLaneInverse matrix (left ^^^ right) =
      (matrixLaneInverse matrix left ^^^ matrixLaneInverse matrix right) := by
  apply matrix_lane_map_injective matrix
  rw [matrix_lane_right_inverse, matrix_lane_map_xor,
    matrix_lane_right_inverse, matrix_lane_right_inverse]

def matrixCodeXorEquiv (matrix : InvertibleMatrixCode) : XorLaneEquiv :=
  { toFun := matrixLaneMap matrix.code.val
  , invFun := matrixLaneInverse matrix
  , mapZero := matrix_lane_map_zero matrix.code.val
  , mapXor := matrix_lane_map_xor matrix.code.val
  , invMapZero := matrix_lane_inverse_zero matrix
  , invMapXor := matrix_lane_inverse_xor matrix
  , leftInverse := matrix_lane_left_inverse matrix
  , rightInverse := matrix_lane_right_inverse matrix }

def matrixCodeLinearSwapAction
    (matrix : InvertibleMatrixCode) (swap : Bool) : LinearSwapAction :=
  { linear := matrixCodeXorEquiv matrix
  , swap := swap }

theorem matrix_code_action_transports_basis_fixed_gauge
    (matrix : InvertibleMatrixCode) (swap : Bool)
    (word : GaugeWord) (table : SignTable) :
    rawAct (matrixCodeLinearSwapAction matrix swap) (gaugeAct word table) =
      gaugeAct
        (rebasedGaugeWord (matrixCodeXorEquiv matrix) word)
        (rawAct (matrixCodeLinearSwapAction matrix swap) table) := by
  exact raw_action_transports_basis_fixed_gauge _ _ _

structure MatrixCodeBridgeBoundary where
  parentBasisFixedRebaseProved : Bool
  matrixApplicationBounded : Bool
  matrixApplicationPreservesXor : Bool
  invertibleImageListNodupProved : Bool
  everyLaneHasPreimageProved : Bool
  computableInverseProved : Bool
  matrixCodeToXorEquivBridgeProved : Bool
  perWitnessLinearSwapActionInstantiated : Bool
  concrete20160WitnessListInstantiated : Bool
  outer40320ViewListInstantiated : Bool
  outer40320ViewMinimumProved : Bool
  concreteCanonicalEqualityIffFullDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def matrixCodeBridgeBoundary : MatrixCodeBridgeBoundary :=
  { parentBasisFixedRebaseProved := true
  , matrixApplicationBounded := true
  , matrixApplicationPreservesXor := true
  , invertibleImageListNodupProved := true
  , everyLaneHasPreimageProved := true
  , computableInverseProved := true
  , matrixCodeToXorEquivBridgeProved := true
  , perWitnessLinearSwapActionInstantiated := true
  , concrete20160WitnessListInstantiated := false
  , outer40320ViewListInstantiated := false
  , outer40320ViewMinimumProved := false
  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem matrix_code_bridge_progress_does_not_close_v13_target03 :
    matrixCodeBridgeBoundary.parentBasisFixedRebaseProved &&
      matrixCodeBridgeBoundary.matrixApplicationBounded &&
      matrixCodeBridgeBoundary.matrixApplicationPreservesXor &&
      matrixCodeBridgeBoundary.invertibleImageListNodupProved &&
      matrixCodeBridgeBoundary.everyLaneHasPreimageProved &&
      matrixCodeBridgeBoundary.computableInverseProved &&
      matrixCodeBridgeBoundary.matrixCodeToXorEquivBridgeProved &&
      matrixCodeBridgeBoundary.perWitnessLinearSwapActionInstantiated &&
      !matrixCodeBridgeBoundary.concrete20160WitnessListInstantiated &&
      !matrixCodeBridgeBoundary.outer40320ViewListInstantiated &&
      !matrixCodeBridgeBoundary.outer40320ViewMinimumProved &&
      !matrixCodeBridgeBoundary.concreteCanonicalEqualityIffFullDeclaredOrbitProved &&
      !matrixCodeBridgeBoundary.formalTarget03Closed &&
      !matrixCodeBridgeBoundary.formalParityClosed &&
      !matrixCodeBridgeBoundary.claimReady := by
  decide

end SounioPireusMatrixCodeXorEquiv
