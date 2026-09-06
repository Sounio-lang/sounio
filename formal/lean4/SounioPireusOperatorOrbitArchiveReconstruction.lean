/-
  Concrete FORMAL_PARITY reconstruction of the 128-image archive consumed by
  frozen Sounio Pireus Operator Orbit Canonicalization v13.

  The archive is recomputed from the 48 cubic children and the declared frozen
  V12 epoch recurrence. Imported formal-parity definitions and hash literals
  identify that dependency; they are not cryptographic proof of external
  files. Sounio remains semantic authority. This module does not reconstruct
  the 30 GL(4,F2) x swap x gauge classes and does not close formal parity or
  any broader novelty claim.
-/
import SounioPireusOperatorOrbitCanonicalization
import SounioPireusQuotientNoveltyForge

namespace SounioPireusOperatorOrbitArchiveReconstruction

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

abbrev Table := Array Nat

def lanes : Nat := 16
def cells : Nat := 256
def interiorCells : Nat := 225
def cubicChildren : Nat := 48
def initialImages : Nat := 96
def generatedEpochs : Nat := 16
def finalImages : Nat := 128

def frozenQuotientSemanticsSha256 : String :=
  "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"

def frozenMorphogenesisSemanticsSha256 : String :=
  "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"

def frozenOrbitSemanticsSha256 : String :=
  "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"

def actionLane (value : Nat) : Nat :=
  (value &&& 12) ||| ((value &&& 1) <<< 1) ||| ((value >>> 1) &&& 1)

def actionCell (cell : Nat) : Nat :=
  actionLane (cell / lanes) * lanes + actionLane (cell % lanes)

def actionTable (table : Table) : Table :=
  ((List.range cells).map fun cell => table.getD (actionCell cell) 0).toArray

def actionInvolutionCensus : Bool :=
  (List.range cells).all fun cell =>
    actionCell cell < cells && actionCell (actionCell cell) == cell

def firstDifference? (left right : Table) : Option Nat :=
  (List.range cells).find? fun cell => left.getD cell 0 != right.getD cell 0

def tableLess (left right : Table) : Bool :=
  match firstDifference? left right with
  | none => false
  | some cell => left.getD cell 0 < right.getD cell 0

def appendUnique (archive : List Table) (table : Table) : List Table :=
  if archive.contains table then archive else archive ++ [table]

def insertOrbit (archive : List Table) (table : Table) : List Table :=
  let acted := actionTable table
  if tableLess acted table then
    appendUnique (appendUnique archive acted) table
  else
    appendUnique (appendUnique archive table) acted

def cubicChildTables : List Table :=
  SounioPireusQuotientNoveltyForge.childTables.toList

def initialArchive : List Table :=
  cubicChildTables.foldl insertOrbit []

def interiorOrdering : List (Nat × Nat) :=
  (List.range lanes).flatMap fun destination =>
    (List.range (lanes - 1)).filterMap fun offset =>
      let left := offset + 1
      let right := left ^^^ destination
      if right > 0 then some (left, right) else none

def phaseIndex (left right : Nat) : Nat :=
  (left - 1) * (lanes - 1) + (right - 1)

def interiorOrderingCensus : Bool :=
  interiorOrdering.length == interiorCells &&
    interiorOrdering.eraseDups.length == interiorCells &&
    interiorOrdering.all fun pair =>
      pair.1 > 0 && pair.1 < lanes && pair.2 > 0 && pair.2 < lanes &&
        phaseIndex pair.1 pair.2 < interiorCells &&
    (interiorOrdering.map fun pair => phaseIndex pair.1 pair.2).eraseDups.length ==
      interiorCells

def interiorCell (rank : Nat) : Nat :=
  let pair := interiorOrdering.getD rank (0, 0)
  pair.1 * lanes + pair.2

def cdBit (left right : Nat) : Nat :=
  if SounioCDCocycle.cdSigma left right 4 < 0 then 1 else 0

def phaseForArchive (archive : List Table) : Array Nat :=
  (List.range archive.length).foldl
    (fun phase archiveIndex =>
      let witness := interiorCell archiveIndex
      let left := witness / lanes
      let right := witness % lanes
      let archivedSign := (archive.getD archiveIndex #[]).getD witness 0
      let archivedPhase := archivedSign ^^^ cdBit left right
      phase.set! (phaseIndex left right) (archivedPhase ^^^ 1))
    (Array.replicate interiorCells 0)

def candidateForArchive (archive : List Table) : Table :=
  let phase := phaseForArchive archive
  ((List.range cells).map fun cell =>
    let left := cell / lanes
    let right := cell % lanes
    let phaseBit :=
      if left == 0 || right == 0 then 0
      else phase.getD (phaseIndex left right) 0
    cdBit left right ^^^ phaseBit).toArray

def buildEpochs (epochCount : Nat) : List Table :=
  (List.range epochCount).foldl
    (fun archive _epoch => insertOrbit archive (candidateForArchive archive))
    initialArchive

def finalArchive : List Table := buildEpochs generatedEpochs

def allRowsHaveExactWidth (archive : List Table) : Bool :=
  archive.all fun table => table.size == cells

def actionClosed (archive : List Table) : Bool :=
  archive.all fun table => archive.contains (actionTable table)

def epochGrowthCensus : Bool :=
  (List.range (generatedEpochs + 1)).all fun epoch =>
    (buildEpochs epoch).length == initialImages + 2 * epoch

def generatedOrbitCensus : Bool :=
  (List.range generatedEpochs).all fun epoch =>
    let archive := buildEpochs epoch
    let candidate := candidateForArchive archive
    let acted := actionTable candidate
    !archive.contains candidate &&
      !archive.contains acted &&
      candidate != acted &&
      (insertOrbit archive candidate).length == archive.length + 2

structure ArchiveReconstructionSummary where
  quotientSemanticsSha256 : String
  morphogenesisSemanticsSha256 : String
  orbitSemanticsSha256 : String
  cubicChildren : Nat
  initialImages : Nat
  generatedEpochs : Nat
  finalImages : Nat
  interiorCells : Nat
  interiorOrderingComplete : Bool
  actionInvolutionComplete : Bool
  initialRowsExactWidth : Bool
  finalRowsExactWidth : Bool
  initialArchiveUnique : Bool
  finalArchiveUnique : Bool
  initialArchiveActionClosed : Bool
  finalArchiveActionClosed : Bool
  epochGrowthComplete : Bool
  generatedOrbitsFreshAndPaired : Bool
  concrete128ImageCensusComplete : Bool
  concrete30ClassReconstructionProved : Bool
  canonicalRepresentativeIffOrbitProved : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def archiveReconstructionSummary : ArchiveReconstructionSummary :=
  let cubicCount := cubicChildTables.length
  let initialCount := initialArchive.length
  let finalCount := finalArchive.length
  let orderingComplete := interiorOrderingCensus
  let involutionComplete := actionInvolutionCensus
  let initialWidth := allRowsHaveExactWidth initialArchive
  let finalWidth := allRowsHaveExactWidth finalArchive
  let initialUnique := initialArchive.eraseDups.length == initialArchive.length
  let finalUnique := finalArchive.eraseDups.length == finalArchive.length
  let initialClosed := actionClosed initialArchive
  let finalClosed := actionClosed finalArchive
  let growthComplete := epochGrowthCensus
  let pairedComplete := generatedOrbitCensus
  let concreteComplete :=
    cubicCount == cubicChildren && initialCount == initialImages &&
      finalCount == finalImages && orderingComplete && involutionComplete &&
      initialWidth && finalWidth && initialUnique && finalUnique &&
      initialClosed && finalClosed && growthComplete && pairedComplete
  { quotientSemanticsSha256 := frozenQuotientSemanticsSha256
  , morphogenesisSemanticsSha256 := frozenMorphogenesisSemanticsSha256
  , orbitSemanticsSha256 := frozenOrbitSemanticsSha256
  , cubicChildren := cubicCount
  , initialImages := initialCount
  , generatedEpochs := generatedEpochs
  , finalImages := finalCount
  , interiorCells := interiorOrdering.length
  , interiorOrderingComplete := orderingComplete
  , actionInvolutionComplete := involutionComplete
  , initialRowsExactWidth := initialWidth
  , finalRowsExactWidth := finalWidth
  , initialArchiveUnique := initialUnique
  , finalArchiveUnique := finalUnique
  , initialArchiveActionClosed := initialClosed
  , finalArchiveActionClosed := finalClosed
  , epochGrowthComplete := growthComplete
  , generatedOrbitsFreshAndPaired := pairedComplete
  , concrete128ImageCensusComplete := concreteComplete
  , concrete30ClassReconstructionProved := false
  , canonicalRepresentativeIffOrbitProved := false
  , formalParityClosed := false
  , claimReady := false }

def frozenArchiveReconstructionSummary : ArchiveReconstructionSummary :=
  { quotientSemanticsSha256 :=
      "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"
  , morphogenesisSemanticsSha256 :=
      "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"
  , orbitSemanticsSha256 :=
      "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"
  , cubicChildren := 48
  , initialImages := 96
  , generatedEpochs := 16
  , finalImages := 128
  , interiorCells := 225
  , interiorOrderingComplete := true
  , actionInvolutionComplete := true
  , initialRowsExactWidth := true
  , finalRowsExactWidth := true
  , initialArchiveUnique := true
  , finalArchiveUnique := true
  , initialArchiveActionClosed := true
  , finalArchiveActionClosed := true
  , epochGrowthComplete := true
  , generatedOrbitsFreshAndPaired := true
  , concrete128ImageCensusComplete := true
  , concrete30ClassReconstructionProved := false
  , canonicalRepresentativeIffOrbitProved := false
  , formalParityClosed := false
  , claimReady := false }

theorem concrete_archive_reconstruction_matches_declared_frozen_summary :
    archiveReconstructionSummary = frozenArchiveReconstructionSummary := by
  native_decide

theorem reconstructed_archive_has_exactly_128_concrete_tables :
    finalArchive.length = 128 := by
  native_decide

theorem forty_eight_cubic_children_reconstruct_ninety_six_action_images :
    archiveReconstructionSummary.cubicChildren = 48 ∧
      archiveReconstructionSummary.initialImages = 96 ∧
      archiveReconstructionSummary.actionInvolutionComplete ∧
      archiveReconstructionSummary.initialArchiveUnique ∧
      archiveReconstructionSummary.initialArchiveActionClosed := by
  rw [concrete_archive_reconstruction_matches_declared_frozen_summary]
  decide

theorem sixteen_fresh_paired_epochs_reconstruct_128_images :
    archiveReconstructionSummary.generatedEpochs = 16 ∧
      archiveReconstructionSummary.finalImages = 128 ∧
      archiveReconstructionSummary.epochGrowthComplete ∧
      archiveReconstructionSummary.generatedOrbitsFreshAndPaired ∧
      archiveReconstructionSummary.finalArchiveUnique ∧
      archiveReconstructionSummary.finalArchiveActionClosed := by
  rw [concrete_archive_reconstruction_matches_declared_frozen_summary]
  decide

theorem archive_reconstruction_carries_declared_frozen_hash_literals :
    archiveReconstructionSummary.quotientSemanticsSha256 =
        "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21" ∧
      archiveReconstructionSummary.morphogenesisSemanticsSha256 =
        "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4" ∧
      archiveReconstructionSummary.orbitSemanticsSha256 =
        "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c" := by
  rw [concrete_archive_reconstruction_matches_declared_frozen_summary]
  decide

theorem archive_reconstruction_does_not_close_class_parity :
    archiveReconstructionSummary.concrete128ImageCensusComplete &&
      !archiveReconstructionSummary.concrete30ClassReconstructionProved &&
      !archiveReconstructionSummary.canonicalRepresentativeIffOrbitProved &&
      !archiveReconstructionSummary.formalParityClosed &&
      !archiveReconstructionSummary.claimReady := by
  rw [concrete_archive_reconstruction_matches_declared_frozen_summary]
  decide

end SounioPireusOperatorOrbitArchiveReconstruction
