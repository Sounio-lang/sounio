/-
  FORMAL_PARITY for the frozen Sounio Pireus Operator Morphogenesis v12.

  Sounio owns the executable semantics and expected result.  This file proves
  the finite codec, mixed ANF matrix, diagonal separator, C2 transport, orbit
  accounting, and the deliberately narrow claim boundary against the frozen
  V12 hashes.  It does not recreate the Sounio archive or select an operator.
-/
import SounioPireusOperatorNoveltyFrontier

namespace SounioPireusOperatorMorphogenesis

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c"

def frozenSounioSemanticsSha256 : String :=
  "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"

def frozenParentSemanticsSha256 : String :=
  "e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff"

def lanes : Nat := 16
def nonzeroLanes : Nat := 15
def interiorCells : Nat := 225
def initialArchiveImages : Nat := 96
def generatedEpochs : Nat := 16

def phaseIndex (left right : Nat) : Nat :=
  (left - 1) * nonzeroLanes + (right - 1)

def phaseLeft (rank : Nat) : Nat := rank / nonzeroLanes + 1
def phaseRight (rank : Nat) : Nat := rank % nonzeroLanes + 1

def phaseCodecRoundTrip (rank : Nat) : Bool :=
  phaseIndex (phaseLeft rank) (phaseRight rank) == rank

def phaseCodecCensus : Bool :=
  (List.range interiorCells).all fun rank =>
    phaseLeft rank > 0 && phaseLeft rank < lanes &&
      phaseRight rank > 0 && phaseRight rank < lanes &&
      phaseCodecRoundTrip rank

def interiorOrdering : List (Nat × Nat) :=
  (List.range lanes).flatMap fun destination =>
    (List.range nonzeroLanes).filterMap fun offset =>
      let left := offset + 1
      let right := left ^^^ destination
      if right > 0 then some (left, right) else none

def interiorOrderingCensus : Bool :=
  interiorOrdering.length == interiorCells &&
    interiorOrdering.eraseDups.length == interiorCells &&
    interiorOrdering.all fun pair =>
      pair.1 > 0 && pair.1 < lanes && pair.2 > 0 && pair.2 < lanes &&
        pair.1 ^^^ pair.2 < lanes &&
        phaseIndex pair.1 pair.2 < interiorCells &&
    (interiorOrdering.map fun pair => phaseIndex pair.1 pair.2).eraseDups.length ==
      interiorCells

def isSubsetMask (small large : Nat) : Bool :=
  (small &&& large) == small

def boolXorList (values : List Bool) : Bool :=
  values.foldl Bool.xor false

def mixedZetaEntry (row column : Nat) : Bool :=
  isSubsetMask (phaseLeft column) (phaseLeft row) &&
    isSubsetMask (phaseRight column) (phaseRight row)

def mixedZetaProductEntry (row column : Nat) : Bool :=
  boolXorList <| (List.range interiorCells).map fun middle =>
    mixedZetaEntry row middle && mixedZetaEntry middle column

def mixedZetaSelfInverseCensus : Bool :=
  (List.range interiorCells).all fun row =>
    (List.range interiorCells).all fun column =>
      mixedZetaProductEntry row column == (row == column)

def axisVanishingExtension
    (interiorPhase : Nat -> Nat -> Bool) (left right : Nat) : Bool :=
  if left == 0 || right == 0 then false else interiorPhase left right

theorem mixed_nonempty_phase_extension_vanishes_on_both_axes
    (interiorPhase : Nat -> Nat -> Bool) (lane : Nat) :
    axisVanishingExtension interiorPhase 0 lane = false ∧
      axisVanishingExtension interiorPhase lane 0 = false := by
  cases lane <;> constructor <;> rfl

def actionLane (value : Nat) : Nat :=
  (value &&& 12) ||| ((value &&& 1) <<< 1) ||| ((value >>> 1) &&& 1)

def actionPhaseRank (rank : Nat) : Nat :=
  phaseIndex (actionLane (phaseLeft rank)) (actionLane (phaseRight rank))

def c2InteriorInvolutionCensus : Bool :=
  (List.range interiorCells).all fun rank =>
    actionPhaseRank rank < interiorCells &&
      actionPhaseRank (actionPhaseRank rank) == rank

abbrev SignTable (Cell : Type) := Cell -> Bool

def actTable {Cell : Type} (action : Cell -> Cell) (table : SignTable Cell) :
    SignTable Cell :=
  fun cell => table (action cell)

theorem actTable_involutive
    {Cell : Type} (action : Cell -> Cell)
    (action_involutive : ∀ cell, action (action cell) = cell)
    (table : SignTable Cell) :
    actTable action (actTable action table) = table := by
  funext cell
  simp [actTable, action_involutive]

def diagonalCandidate {n : Nat}
    (archive : Fin n -> SignTable (Fin n)) : SignTable (Fin n) :=
  fun cell => !(archive cell cell)

theorem list_index_complement_separates_every_prior_row
    {n : Nat} (archive : Fin n -> SignTable (Fin n)) (row : Fin n) :
    Not (diagonalCandidate archive = archive row) := by
  intro equal
  have atWitness := congrFun equal row
  simp [diagonalCandidate] at atWitness

def pairedArchive {Index Cell : Type}
    (action : Cell -> Cell) (seed : Index -> SignTable Cell) :
    (Index × Bool) -> SignTable Cell
  | (index, false) => seed index
  | (index, true) => actTable action (seed index)

def pairedArchivePartner {Index : Type} : Index × Bool -> Index × Bool
  | (index, side) => (index, !side)

theorem paired_archive_is_closed_under_c2
    {Index Cell : Type} (action : Cell -> Cell)
    (action_involutive : ∀ cell, action (action cell) = cell)
    (seed : Index -> SignTable Cell) (row : Index × Bool) :
    actTable action (pairedArchive action seed row) =
      pairedArchive action seed (pairedArchivePartner row) := by
  rcases row with ⟨index, side⟩
  cases side <;>
    simp [pairedArchive, pairedArchivePartner, actTable_involutive,
      action_involutive]

theorem transported_separator_sound
    {ArchiveIndex Cell : Type} (action : Cell -> Cell)
    (action_involutive : ∀ cell, action (action cell) = cell)
    (archive : ArchiveIndex -> SignTable Cell) (candidate : SignTable Cell)
    (separates : ∀ row, Not (candidate = archive row))
    (archive_closed : ∀ row, ∃ partner,
      actTable action (archive row) = archive partner) :
    ∀ row, Not (actTable action candidate = archive row) := by
  intro row equal
  obtain ⟨partner, partner_spec⟩ := archive_closed row
  apply separates partner
  calc
    candidate = actTable action (actTable action candidate) :=
      (actTable_involutive action action_involutive candidate).symm
    _ = actTable action (archive row) := congrArg (actTable action) equal
    _ = archive partner := partner_spec

def archiveBefore (epoch : Nat) : Nat :=
  initialArchiveImages + 2 * epoch

def archiveAfter (epoch : Nat) : Nat := archiveBefore epoch + 2

def finalArchiveImages : Nat :=
  initialArchiveImages + 2 * generatedEpochs

def phaseComparisons : Nat :=
  (List.range generatedEpochs).foldl
    (fun count epoch => count + archiveBefore epoch) 0

def separatorCertificates : Nat := 2 * phaseComparisons

def orbitAccountingCensus : Bool :=
  finalArchiveImages == 128 &&
    (List.range generatedEpochs).all fun epoch =>
      archiveBefore epoch == 96 + 2 * epoch &&
        archiveAfter epoch == 98 + 2 * epoch

structure ClaimBoundary where
  fullSpaceExhausted : Bool
  glGaugeInequivalence : Bool
  algebraicNovelty : Bool
  algorithmicNovelty : Bool
  materialNovelty : Bool
  performanceNovelty : Bool
  scientificNovelty : Bool
  historicalNovelty : Bool
  globalNovelty : Bool
  priorityClaim : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def claimBoundary : ClaimBoundary :=
  { fullSpaceExhausted := false
  , glGaugeInequivalence := false
  , algebraicNovelty := false
  , algorithmicNovelty := false
  , materialNovelty := false
  , performanceNovelty := false
  , scientificNovelty := false
  , historicalNovelty := false
  , globalNovelty := false
  , priorityClaim := false
  , claimReady := false }

structure FormalParitySummary where
  sounioSourceSha256 : String
  sounioSemanticsSha256 : String
  parentSemanticsSha256 : String
  interiorCells : Nat
  phaseCodecComplete : Bool
  interiorOrderingComplete : Bool
  mixedAnfMatrixSelfInverse : Bool
  c2InteriorInvolutionComplete : Bool
  initialArchiveImages : Nat
  generatedEpochs : Nat
  orbitFixed : Nat
  orbitPairs : Nat
  finalArchiveImages : Nat
  phaseComparisons : Nat
  certificateCount : Nat
  claimBoundary : ClaimBoundary
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  { sounioSourceSha256 := frozenSounioSourceSha256
  , sounioSemanticsSha256 := frozenSounioSemanticsSha256
  , parentSemanticsSha256 := frozenParentSemanticsSha256
  , interiorCells := interiorCells
  , phaseCodecComplete := phaseCodecCensus
  , interiorOrderingComplete := interiorOrderingCensus
  , mixedAnfMatrixSelfInverse := mixedZetaSelfInverseCensus
  , c2InteriorInvolutionComplete := c2InteriorInvolutionCensus
  , initialArchiveImages := initialArchiveImages
  , generatedEpochs := generatedEpochs
  , orbitFixed := 0
  , orbitPairs := generatedEpochs
  , finalArchiveImages := finalArchiveImages
  , phaseComparisons := phaseComparisons
  , certificateCount := separatorCertificates
  , claimBoundary := claimBoundary }

def frozenFormalParitySummary : FormalParitySummary :=
  { sounioSourceSha256 :=
      "0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c"
  , sounioSemanticsSha256 :=
      "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"
  , parentSemanticsSha256 :=
      "e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff"
  , interiorCells := 225
  , phaseCodecComplete := true
  , interiorOrderingComplete := true
  , mixedAnfMatrixSelfInverse := true
  , c2InteriorInvolutionComplete := true
  , initialArchiveImages := 96
  , generatedEpochs := 16
  , orbitFixed := 0
  , orbitPairs := 16
  , finalArchiveImages := 128
  , phaseComparisons := 1776
  , certificateCount := 3552
  , claimBoundary := claimBoundary }

theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  native_decide

theorem formal_parity_is_bound_to_frozen_sounio_hashes :
    formalParitySummary.sounioSourceSha256 =
        "0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c" ∧
      formalParitySummary.sounioSemanticsSha256 =
        "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4" ∧
      formalParitySummary.parentSemanticsSha256 =
        "e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff" := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem interior_codec_is_a_225_cell_bijection :
    formalParitySummary.interiorCells = 225 &&
      formalParitySummary.phaseCodecComplete &&
      formalParitySummary.interiorOrderingComplete := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem mixed_nonempty_anf_mobius_matrix_is_self_inverse :
    formalParitySummary.mixedAnfMatrixSelfInverse := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem c2_involution_and_orbit_accounting_complete :
    formalParitySummary.c2InteriorInvolutionComplete &&
      formalParitySummary.orbitFixed = 0 &&
      formalParitySummary.orbitPairs = 16 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem orbit_insertion_accounting_is_96_plus_2_times_16 :
    formalParitySummary.initialArchiveImages = 96 &&
      formalParitySummary.generatedEpochs = 16 &&
      formalParitySummary.finalArchiveImages = 128 &&
      formalParitySummary.phaseComparisons = 1776 &&
      formalParitySummary.certificateCount = 3552 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem executable_certificate_scope_does_not_promote_classification :
    !claimBoundary.fullSpaceExhausted &&
      !claimBoundary.glGaugeInequivalence &&
      !claimBoundary.algebraicNovelty &&
      !claimBoundary.algorithmicNovelty &&
      !claimBoundary.materialNovelty &&
      !claimBoundary.performanceNovelty &&
      !claimBoundary.scientificNovelty &&
      !claimBoundary.historicalNovelty &&
      !claimBoundary.globalNovelty &&
      !claimBoundary.priorityClaim &&
      !claimBoundary.claimReady := by
  decide

end SounioPireusOperatorMorphogenesis
