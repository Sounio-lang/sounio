/-
  Concrete FORMAL_PARITY census of the 30 declared-action classes in the
  frozen 128-image Pireus v13 archive.

  This module recomputes the exact GL(4,F2) x input-swap x basis-fixed gauge
  lexicographic canonicalizer over the archive reconstructed in Lean. Sounio
  remains semantic authority. Equality of canonical representatives iff the
  abstract declared orbit relation is left open until its generic proof is
  discharged; no broader quotient or novelty claim is made here.
-/
import SounioPireusOperatorOrbitArchiveReconstruction

namespace SounioPireusOperatorOrbitClassReconstruction

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

abbrev Table := Array Nat
abbrev LaneMap := Array Nat

def lanes : Nat := 16
def cells : Nat := 256
def expectedActions : Nat := 40320
def expectedImages : Nat := 128
def expectedClasses : Nat := 30

def frozenOrbitSemanticsSha256 : String :=
  "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"

def matrixApply (code vector : Nat) : Nat :=
  SounioPireusOperatorOrbitCanonicalization.matrixApply code vector

def invertibleMatrixCodes : List Nat :=
  SounioPireusOperatorOrbitCanonicalization.invertibleMatrixCodes

def matrixLaneMap (code : Nat) : LaneMap :=
  ((List.range lanes).map (matrixApply code)).toArray

def invertibleMatrixMaps : List LaneMap :=
  invertibleMatrixCodes.map matrixLaneMap

def isBasis (value : Nat) : Bool :=
  value == 1 || value == 2 || value == 4 || value == 8

def highestBasis (value : Nat) : Nat :=
  if value >= 8 then 8 else if value >= 4 then 4 else if value >= 2 then 2 else 1

def gaugeVectors : List Nat := [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

def gaugeIndex (vector : Nat) : Nat :=
  (gaugeVectors.takeWhile fun candidate => candidate != vector).length

def gaugeWord (values : Array Nat) : Nat :=
  gaugeVectors.foldl
    (fun word vector =>
      if values.getD vector 0 == 1 then word + 2 ^ gaugeIndex vector else word)
    0

def actionValue
    (table : Table) (mapping : LaneMap) (swap left right : Nat) : Nat :=
  let mappedLeft := mapping.getD left 0
  let mappedRight := mapping.getD right 0
  if swap == 0 then table.getD (mappedLeft * lanes + mappedRight) 0
  else table.getD (mappedRight * lanes + mappedLeft) 0

def actionSectionValues (table : Table) (mapping : LaneMap) (swap : Nat) : Array Nat :=
  (List.range lanes).foldl
    (fun values vector =>
      if vector == 0 || isBasis vector then values
      else
        let edge := highestBasis vector
        let parent := vector ^^^ edge
        values.set! vector
          (actionValue table mapping swap parent edge ^^^ values.getD parent 0))
    (Array.replicate lanes 0)

def normalizedActionValue
    (table : Table) (mapping : LaneMap) (swap : Nat)
    (gauge : Array Nat) (cell : Nat) : Nat :=
  let left := cell / lanes
  let right := cell % lanes
  actionValue table mapping swap left right ^^^
    gauge.getD left 0 ^^^ gauge.getD right 0 ^^^ gauge.getD (left ^^^ right) 0

def normalizedActionTable
    (table : Table) (mapping : LaneMap) (swap : Nat) (gauge : Array Nat) : Table :=
  ((List.range cells).map fun cell =>
    normalizedActionValue table mapping swap gauge cell).toArray

def normalizedActionLess
    (table : Table) (mapping : LaneMap) (swap : Nat)
    (gauge : Array Nat) (best : Table) : Bool :=
  if best.size != cells then true
  else
    match (List.range cells).find? fun cell =>
        normalizedActionValue table mapping swap gauge cell != best.getD cell 0 with
    | none => false
    | some cell => normalizedActionValue table mapping swap gauge cell < best.getD cell 0

def considerAction (source best : Table) (mapping : LaneMap) (swap : Nat) : Table :=
  let gauge := actionSectionValues source mapping swap
  if normalizedActionLess source mapping swap gauge best then
    normalizedActionTable source mapping swap gauge
  else best

def considerMatrix (source best : Table) (mapping : LaneMap) : Table :=
  let withoutSwap := considerAction source best mapping 0
  considerAction source withoutSwap mapping 1

def canonicalizeWithMaps (mappings : List LaneMap) (source : Table) : Table :=
  mappings.foldl (considerMatrix source) #[]

def canonicalize (source : Table) : Table :=
  canonicalizeWithMaps invertibleMatrixMaps source

def reconstructedArchive : List Table :=
  SounioPireusOperatorOrbitArchiveReconstruction.finalArchive

def canonicalArchive : List Table :=
  let mappings := invertibleMatrixMaps
  reconstructedArchive.map (canonicalizeWithMaps mappings)

def appendUnique (tables : List Table) (table : Table) : List Table :=
  if tables.contains table then tables else tables ++ [table]

def canonicalClasses : List Table := canonicalArchive.foldl appendUnique []

def allRowsHaveExactWidth (tables : List Table) : Bool :=
  tables.all fun table => table.size == cells

def classMembershipComplete : Bool :=
  canonicalArchive.all fun table => canonicalClasses.contains table

structure ClassReconstructionSummary where
  orbitSemanticsSha256 : String
  matrixCodesScanned : Nat
  invertibleMatrices : Nat
  actionViewsPerImage : Nat
  archiveImages : Nat
  canonicalizations : Nat
  canonicalRowsExactWidth : Bool
  classRowsExactWidth : Bool
  canonicalClassCount : Nat
  canonicalClassesDistinct : Bool
  classMembershipComplete : Bool
  concrete30ClassCensusComplete : Bool
  canonicalRepresentativeIffDeclaredOrbitProved : Bool
  concrete32AdmissionReconstructionProved : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def classReconstructionSummary : ClassReconstructionSummary :=
  let images := reconstructedArchive.length
  let canonicalized := canonicalArchive
  let classes := canonicalized.foldl appendUnique []
  let canonicalWidth := allRowsHaveExactWidth canonicalized
  let classWidth := allRowsHaveExactWidth classes
  let distinct := classes.eraseDups.length == classes.length
  let membership := canonicalized.all fun table => classes.contains table
  let complete :=
    invertibleMatrixCodes.length == 20160 && images == expectedImages &&
      canonicalized.length == expectedImages && canonicalWidth && classWidth &&
      classes.length == expectedClasses && distinct && membership
  { orbitSemanticsSha256 := frozenOrbitSemanticsSha256
  , matrixCodesScanned := 65536
  , invertibleMatrices := invertibleMatrixCodes.length
  , actionViewsPerImage := invertibleMatrixCodes.length * 2
  , archiveImages := images
  , canonicalizations := canonicalized.length
  , canonicalRowsExactWidth := canonicalWidth
  , classRowsExactWidth := classWidth
  , canonicalClassCount := classes.length
  , canonicalClassesDistinct := distinct
  , classMembershipComplete := membership
  , concrete30ClassCensusComplete := complete
  , canonicalRepresentativeIffDeclaredOrbitProved := false
  , concrete32AdmissionReconstructionProved := false
  , formalParityClosed := false
  , claimReady := false }

def frozenClassReconstructionSummary : ClassReconstructionSummary :=
  { orbitSemanticsSha256 :=
      "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"
  , matrixCodesScanned := 65536
  , invertibleMatrices := 20160
  , actionViewsPerImage := 40320
  , archiveImages := 128
  , canonicalizations := 128
  , canonicalRowsExactWidth := true
  , classRowsExactWidth := true
  , canonicalClassCount := 30
  , canonicalClassesDistinct := true
  , classMembershipComplete := true
  , concrete30ClassCensusComplete := true
  , canonicalRepresentativeIffDeclaredOrbitProved := false
  , concrete32AdmissionReconstructionProved := false
  , formalParityClosed := false
  , claimReady := false }

theorem concrete_class_reconstruction_matches_declared_frozen_summary :
    classReconstructionSummary = frozenClassReconstructionSummary ∧
      canonicalArchive.length = 128 ∧ canonicalClasses.length = 30 := by
  native_decide

theorem reconstructed_128_image_archive_has_exactly_30_canonical_tables :
    canonicalArchive.length = 128 ∧ canonicalClasses.length = 30 := by
  exact concrete_class_reconstruction_matches_declared_frozen_summary.2

theorem every_reconstructed_image_maps_to_one_of_30_distinct_classes :
    classReconstructionSummary.canonicalRowsExactWidth ∧
      classReconstructionSummary.classRowsExactWidth ∧
      classReconstructionSummary.canonicalClassesDistinct ∧
      classReconstructionSummary.classMembershipComplete := by
  rw [concrete_class_reconstruction_matches_declared_frozen_summary.1]
  decide

theorem class_census_does_not_yet_prove_canonical_iff_orbit :
    classReconstructionSummary.concrete30ClassCensusComplete &&
      !classReconstructionSummary.canonicalRepresentativeIffDeclaredOrbitProved &&
      !classReconstructionSummary.concrete32AdmissionReconstructionProved &&
      !classReconstructionSummary.formalParityClosed &&
      !classReconstructionSummary.claimReady := by
  rw [concrete_class_reconstruction_matches_declared_frozen_summary.1]
  decide

end SounioPireusOperatorOrbitClassReconstruction
