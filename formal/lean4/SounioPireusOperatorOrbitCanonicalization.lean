/-
  FORMAL_PARITY for frozen Sounio Pireus Operator Orbit Canonicalization v13.

  Sounio owns the executable semantics, first result, and expected result. This
  file independently proves the finite GL(4,2) scan, swap-view census, gauge
  codec, 15x15 interior codec, separator arithmetic, microprogram indexing,
  and deliberately narrow claim boundary. The concrete 30 baseline classes
  and 32 admissions remain hash-bound Sounio facts rather than Lean-created
  expected values.
-/
import SounioPireusOperatorMorphogenesis

namespace SounioPireusOperatorOrbitCanonicalization

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae"

def frozenSourceManifestSha256 : String :=
  "022fda14573d31009c3740f0cb374b8ac06b1047fa0a90ae9ac5f44074c3e44d"

def frozenSounioSemanticsSha256 : String :=
  "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"

def frozenParentSemanticsSha256 : String :=
  "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"

def frozenFreezeReceiptSha256 : String :=
  "11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84"

def bits : Nat := 4
def lanes : Nat := 16
def cells : Nat := 256
def nonzeroLanes : Nat := 15
def interiorCells : Nat := 225
def matrixCodes : Nat := 65536
def expectedGL4Order : Nat := 20160
def expectedActionViews : Nat := 40320
def gaugeBits : Nat := 11
def gaugeWords : Nat := 2048

def bit (value coordinate : Nat) : Nat :=
  (value / 2 ^ coordinate) % 2

def parity4 (value : Nat) : Nat :=
  (bit value 0 + bit value 1 + bit value 2 + bit value 3) % 2

def matrixRow (code row : Nat) : Nat :=
  (code >>> (4 * row)) &&& 15

def matrixApply (code vector : Nat) : Nat :=
  (List.range bits).foldl
    (fun image row => image + parity4 (matrixRow code row &&& vector) * 2 ^ row) 0

def matrixImages (code : Nat) : List Nat :=
  (List.range lanes).map (matrixApply code)

def matrixInvertible (code : Nat) : Bool :=
  (matrixImages code).eraseDups.length == lanes

def invertibleMatrixCodes : List Nat :=
  (List.range matrixCodes).filter matrixInvertible

def invertibleMatrixCount : Nat := invertibleMatrixCodes.length

def declaredLinearSwapViews : List (Nat × Bool) :=
  invertibleMatrixCodes.flatMap fun code => [(code, false), (code, true)]

theorem gl4_f2_enumeration_has_exactly_20160_matrices :
    invertibleMatrixCount = expectedGL4Order := by
  native_decide

theorem every_invertible_4x4_code_is_in_the_scan
    (code : Nat) (inRange : code < matrixCodes)
    (invertible : matrixInvertible code = true) :
    code ∈ invertibleMatrixCodes := by
  simp [invertibleMatrixCodes, inRange, invertible]

theorem declared_linear_swap_view_census_is_40320 :
    declaredLinearSwapViews.length = expectedActionViews := by
  native_decide

def encodeGaugeBits (word : Nat) : List Bool :=
  (List.range gaugeBits).map fun rank => bit word rank == 1

def decodeGaugeBits : List Bool -> Nat
  | [] => 0
  | head :: tail => (if head then 1 else 0) + 2 * decodeGaugeBits tail

def gaugeRoundTripCensus : Bool :=
  (List.range gaugeWords).all fun word =>
    decodeGaugeBits (encodeGaugeBits word) == word

def basisFixedGaugeLanes : List Nat := [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

def gaugeLaneCensus : Bool :=
  basisFixedGaugeLanes.length == gaugeBits &&
    basisFixedGaugeLanes.eraseDups.length == gaugeBits &&
    basisFixedGaugeLanes.all fun lane => lane < lanes &&
      lane != 0 && lane != 1 && lane != 2 && lane != 4 && lane != 8

def allGaugeWords : List Nat := List.range gaugeWords

theorem basis_fixed_gauge_codec_roundtrips_all_2048_words :
    gaugeRoundTripCensus && gaugeLaneCensus := by
  native_decide

theorem every_11_bit_gauge_word_is_in_the_scan
    (word : Nat) (inRange : word < gaugeWords) :
    word ∈ allGaugeWords := by
  simpa [allGaugeWords] using inRange

def phaseIndex (left right : Nat) : Nat :=
  (left - 1) * nonzeroLanes + (right - 1)

def phaseLeft (rank : Nat) : Nat := rank / nonzeroLanes + 1
def phaseRight (rank : Nat) : Nat := rank % nonzeroLanes + 1

def phaseCodecCensus : Bool :=
  (List.range interiorCells).all fun rank =>
    phaseLeft rank > 0 && phaseLeft rank < lanes &&
      phaseRight rank > 0 && phaseRight rank < lanes &&
      phaseIndex (phaseLeft rank) (phaseRight rank) == rank

theorem interior_codec_is_a_225_cell_bijection : phaseCodecCensus := by
  native_decide

def microDestination (cell : Nat) : Nat := cell / lanes
def microOrdinal (cell : Nat) : Nat := cell % lanes
def microCell (destination ordinal : Nat) : Nat := destination * lanes + ordinal
def microPartner (destination ordinal : Nat) : Nat := destination ^^^ ordinal

def microprogramIndexCensus : Bool :=
  (List.range cells).all fun cell =>
    microDestination cell < lanes &&
      microOrdinal cell < lanes &&
      microPartner (microDestination cell) (microOrdinal cell) < lanes &&
      microCell (microDestination cell) (microOrdinal cell) == cell

theorem destination_major_microprogram_indexes_all_256_cells :
    microprogramIndexCensus := by
  native_decide

abbrev SignTable (Cell : Type) := Cell -> Bool

theorem exact_cell_separator_implies_distinct_tables
    {Cell : Type} (candidate prior : SignTable Cell) (witness : Cell)
    (separates : candidate witness != prior witness) :
    Not (candidate = prior) := by
  intro equal
  have same := congrFun equal witness
  simp [same] at separates

def baselineClasses : Nat := 30
def mutationAttempts : Nat := 33
def equivalentCollapses : Nat := 1
def admittedClasses : Nat := 32
def finalClasses : Nat := baselineClasses + admittedClasses

def separatorCertificates : Nat :=
  (List.range admittedClasses).foldl
    (fun count admitted => count + baselineClasses + admitted) 0

theorem admission_accounting_is_33_minus_1_equals_32 :
    mutationAttempts - equivalentCollapses = admittedClasses := by
  decide

theorem class_accounting_is_30_plus_32_equals_62 :
    finalClasses = 62 := by
  decide

theorem separator_accounting_is_32_times_30_plus_choose_32_2 :
    separatorCertificates =
        admittedClasses * baselineClasses +
          admittedClasses * (admittedClasses - 1) / 2 ∧
      separatorCertificates = 1456 := by
  native_decide

structure ClaimBoundary where
  fullSpaceExhausted : Bool
  nonlinearPermutationComplete : Bool
  isotopyComplete : Bool
  unrestrictedIsomorphismComplete : Bool
  algebraicNovelty : Bool
  algorithmicNovelty : Bool
  materialNovelty : Bool
  performanceNovelty : Bool
  scientificNovelty : Bool
  globalNovelty : Bool
  historicalNovelty : Bool
  priorityClaim : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def claimBoundary : ClaimBoundary :=
  { fullSpaceExhausted := false
  , nonlinearPermutationComplete := false
  , isotopyComplete := false
  , unrestrictedIsomorphismComplete := false
  , algebraicNovelty := false
  , algorithmicNovelty := false
  , materialNovelty := false
  , performanceNovelty := false
  , scientificNovelty := false
  , globalNovelty := false
  , historicalNovelty := false
  , priorityClaim := false
  , claimReady := false }

structure FormalParitySummary where
  sounioSourceSha256 : String
  sourceManifestSha256 : String
  sounioSemanticsSha256 : String
  parentSemanticsSha256 : String
  freezeReceiptSha256 : String
  matrixCodes : Nat
  invertibleMatrices : Nat
  linearSwapViews : Nat
  linearSwapOperatorDistinctnessProved : Bool
  gaugeBits : Nat
  gaugeWords : Nat
  gaugeCodecComplete : Bool
  interiorCells : Nat
  interiorCodecComplete : Bool
  microprogramCells : Nat
  microprogramIndexComplete : Bool
  baselineClasses : Nat
  mutationAttempts : Nat
  equivalentCollapses : Nat
  admittedClasses : Nat
  finalClasses : Nat
  separatorCertificates : Nat
  abstractCodecEquivalencesProved : Bool
  concreteSounioClassReconstructionProved : Bool
  formalParityClosed : Bool
  claimBoundary : ClaimBoundary
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  { sounioSourceSha256 := frozenSounioSourceSha256
  , sourceManifestSha256 := frozenSourceManifestSha256
  , sounioSemanticsSha256 := frozenSounioSemanticsSha256
  , parentSemanticsSha256 := frozenParentSemanticsSha256
  , freezeReceiptSha256 := frozenFreezeReceiptSha256
  , matrixCodes := matrixCodes
  , invertibleMatrices := invertibleMatrixCount
  , linearSwapViews := declaredLinearSwapViews.length
  , linearSwapOperatorDistinctnessProved := false
  , gaugeBits := gaugeBits
  , gaugeWords := gaugeWords
  , gaugeCodecComplete := gaugeRoundTripCensus && gaugeLaneCensus
  , interiorCells := interiorCells
  , interiorCodecComplete := phaseCodecCensus
  , microprogramCells := cells
  , microprogramIndexComplete := microprogramIndexCensus
  , baselineClasses := baselineClasses
  , mutationAttempts := mutationAttempts
  , equivalentCollapses := equivalentCollapses
  , admittedClasses := admittedClasses
  , finalClasses := finalClasses
  , separatorCertificates := separatorCertificates
  , abstractCodecEquivalencesProved := false
  , concreteSounioClassReconstructionProved := false
  , formalParityClosed := false
  , claimBoundary := claimBoundary }

def frozenFormalParitySummary : FormalParitySummary :=
  { sounioSourceSha256 :=
      "7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae"
  , sourceManifestSha256 :=
      "022fda14573d31009c3740f0cb374b8ac06b1047fa0a90ae9ac5f44074c3e44d"
  , sounioSemanticsSha256 :=
      "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"
  , parentSemanticsSha256 :=
      "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"
  , freezeReceiptSha256 :=
      "11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84"
  , matrixCodes := 65536
  , invertibleMatrices := 20160
  , linearSwapViews := 40320
  , linearSwapOperatorDistinctnessProved := false
  , gaugeBits := 11
  , gaugeWords := 2048
  , gaugeCodecComplete := true
  , interiorCells := 225
  , interiorCodecComplete := true
  , microprogramCells := 256
  , microprogramIndexComplete := true
  , baselineClasses := 30
  , mutationAttempts := 33
  , equivalentCollapses := 1
  , admittedClasses := 32
  , finalClasses := 62
  , separatorCertificates := 1456
  , abstractCodecEquivalencesProved := false
  , concreteSounioClassReconstructionProved := false
  , formalParityClosed := false
  , claimBoundary := claimBoundary }

theorem formal_parity_summary_matches_frozen_sounio_snapshot :
    formalParitySummary = frozenFormalParitySummary := by
  native_decide

theorem formal_parity_is_bound_to_frozen_v13_hashes :
    formalParitySummary.sounioSourceSha256 =
        "7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae" ∧
      formalParitySummary.sourceManifestSha256 =
        "022fda14573d31009c3740f0cb374b8ac06b1047fa0a90ae9ac5f44074c3e44d" ∧
      formalParitySummary.sounioSemanticsSha256 =
        "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c" ∧
      formalParitySummary.parentSemanticsSha256 =
        "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4" ∧
      formalParitySummary.freezeReceiptSha256 =
        "11893a34450729ff06ac40ade86c90decb7a6947daea3cc108cae17f73572f84" := by
  rw [formal_parity_summary_matches_frozen_sounio_snapshot]
  decide

theorem formal_parity_remains_open_on_concrete_class_reconstruction :
    !formalParitySummary.linearSwapOperatorDistinctnessProved &&
      !formalParitySummary.abstractCodecEquivalencesProved &&
      !formalParitySummary.concreteSounioClassReconstructionProved &&
      !formalParitySummary.formalParityClosed := by
  rw [formal_parity_summary_matches_frozen_sounio_snapshot]
  decide

theorem executable_scope_does_not_promote_broader_classification :
    !claimBoundary.fullSpaceExhausted &&
      !claimBoundary.nonlinearPermutationComplete &&
      !claimBoundary.isotopyComplete &&
      !claimBoundary.unrestrictedIsomorphismComplete &&
      !claimBoundary.algebraicNovelty &&
      !claimBoundary.algorithmicNovelty &&
      !claimBoundary.materialNovelty &&
      !claimBoundary.performanceNovelty &&
      !claimBoundary.scientificNovelty &&
      !claimBoundary.globalNovelty &&
      !claimBoundary.historicalNovelty &&
      !claimBoundary.priorityClaim &&
      !claimBoundary.claimReady := by
  decide

end SounioPireusOperatorOrbitCanonicalization
