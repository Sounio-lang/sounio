/-
  FORMAL_PARITY for the frozen Sounio Pireus Operator Novelty Frontier v11.

  Sounio owns the result. This file independently reconstructs the frozen V10
  parent and six-representative atlas through its already-open formal parity,
  then computes the V11 codec, supports, ambient C2_diag action, and claim
  boundary. It cannot create semantics or select a candidate.
-/
import SounioPireusOperatorDiscoveryEngine

namespace SounioPireusOperatorNoveltyFrontier

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b"

def frozenSounioSemanticsSha256 : String :=
  "f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc"

def frozenParentSemanticsSha256 : String :=
  "2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5"

def dimension : Nat := 16
def tensorCells : Nat := 4096
def grammarCandidates : Nat := 7200
def atlasRepresentatives : Nat := 6

def countWhere (upper : Nat) (predicate : Nat -> Bool) : Nat :=
  (List.range upper).foldl
    (fun count value => if predicate value then count + 1 else count) 0

def tensorIndex (output input0 input1 : Nat) : Nat :=
  (input0 * dimension + input1) * dimension + output

def cellOutput (cell : Nat) : Nat := cell % dimension
def cellInput0 (cell : Nat) : Nat := (cell / dimension) / dimension
def cellInput1 (cell : Nat) : Nat := (cell / dimension) % dimension

def candidateCoordinate (id : Nat) : Nat := id / 2
def candidateOutput (id : Nat) : Nat := candidateCoordinate id % 16
def candidatePair (id : Nat) : Nat := candidateCoordinate id / 16
def candidateInput1 (id : Nat) : Nat := candidatePair id % 15 + 1
def candidateInput0 (id : Nat) : Nat := candidatePair id / 15 + 1

def candidateCell (id : Nat) : Nat :=
  tensorIndex (candidateOutput id) (candidateInput0 id) (candidateInput1 id)

def candidateDelta (id : Nat) : Int :=
  if id % 2 == 0 then 1 else -1

def encodeCandidate (cell : Nat) (delta : Int) : Option Nat :=
  if cell < tensorCells && (delta == 1 || delta == -1) then
    let output := cellOutput cell
    let input0 := cellInput0 cell
    let input1 := cellInput1 cell
    if 1 <= input0 && input0 < dimension &&
        1 <= input1 && input1 < dimension then
      let pair := (input0 - 1) * 15 + (input1 - 1)
      let coordinate := pair * 16 + output
      some (coordinate * 2 + if delta == 1 then 0 else 1)
    else none
  else none

def codecRoundTrip (id : Nat) : Bool :=
  encodeCandidate (candidateCell id) (candidateDelta id) == some id

def codecRoundTripCensus : Bool :=
  (List.range grammarCandidates).all codecRoundTrip

def unitBoundary (id : Nat) : Bool :=
  1 <= candidateInput0 id && candidateInput0 id < dimension &&
    1 <= candidateInput1 id && candidateInput1 id < dimension

def unitBoundaryCensus : Bool :=
  (List.range grammarCandidates).all unitBoundary

def parentCoefficient (output input0 input1 : Nat) : Int :=
  SounioPireusOperatorDiscoveryEngine.parentCoefficient output input0 input1

def representativeCoefficient
    (representative output input0 input1 : Nat) : Int :=
  let classId := representative / 2
  let action := representative % 2
  SounioPireusOperatorDiscoveryEngine.representativeCoefficient
    classId action output input0 input1

def atlasDifferenceSupport (representative : Nat) : Nat :=
  countWhere tensorCells fun cell =>
    let output := cellOutput cell
    let input0 := cellInput0 cell
    let input1 := cellInput1 cell
    parentCoefficient output input0 input1 !=
      representativeCoefficient representative output input0 input1

def atlasDifferenceSupports : List Nat :=
  (List.range atlasRepresentatives).map atlasDifferenceSupport

def candidateCoefficient (candidate output input0 input1 : Nat) : Int :=
  let parent := parentCoefficient output input0 input1
  if tensorIndex output input0 input1 == candidateCell candidate then
    parent + candidateDelta candidate
  else parent

def candidateDifferenceSupport (candidate : Nat) : Nat :=
  countWhere tensorCells fun cell =>
    let output := cellOutput cell
    let input0 := cellInput0 cell
    let input1 := cellInput1 cell
    candidateCoefficient candidate output input0 input1 !=
      parentCoefficient output input0 input1

def candidateDifferenceSupportCensus : Bool :=
  (List.range grammarCandidates).all fun candidate =>
    candidateDifferenceSupport candidate == 1

def atlasSeparationCensus : Bool :=
  candidateDifferenceSupportCensus &&
    (List.range atlasRepresentatives).all fun representative =>
      atlasDifferenceSupport representative != 1

def swap01 (value : Nat) : Nat :=
  if value == 0 then 1 else if value == 1 then 0 else value

def permuteCell (cell : Nat) : Nat :=
  tensorIndex
    (swap01 (cellOutput cell))
    (swap01 (cellInput0 cell))
    (swap01 (cellInput1 cell))

def c2CellInvolutionCensus : Bool :=
  (List.range tensorCells).all fun cell =>
    permuteCell cell < tensorCells && permuteCell (permuteCell cell) == cell

def basisCharacter (_cell : Nat) : Int := 1

def unsignedCharacterCensus : Bool :=
  (List.range tensorCells).all fun cell => basisCharacter cell == 1

def transportedMutationCell (candidate : Nat) : Nat :=
  permuteCell (candidateCell candidate)

def transportedMutationInsideGrammar (candidate : Nat) : Bool :=
  let cell := transportedMutationCell candidate
  1 <= cellInput0 cell && cellInput0 cell < dimension &&
    1 <= cellInput1 cell && cellInput1 cell < dimension

def transportedMutationInsideGrammarCount : Nat :=
  countWhere grammarCandidates transportedMutationInsideGrammar

def transportedMutationOutsideGrammarCount : Nat :=
  grammarCandidates - transportedMutationInsideGrammarCount

def actedParentCoefficient (output input0 input1 : Nat) : Int :=
  parentCoefficient (swap01 output) (swap01 input0) (swap01 input1)

def actionBaseDifferenceSupport : Nat :=
  countWhere tensorCells fun cell =>
    let output := cellOutput cell
    let input0 := cellInput0 cell
    let input1 := cellInput1 cell
    actedParentCoefficient output input0 input1 !=
      parentCoefficient output input0 input1

def actedCandidateCoefficient
    (candidate output input0 input1 : Nat) : Int :=
  candidateCoefficient candidate (swap01 output) (swap01 input0) (swap01 input1)

def actionDifferenceSupport (candidate : Nat) : Nat :=
  countWhere tensorCells fun cell =>
    let output := cellOutput cell
    let input0 := cellInput0 cell
    let input1 := cellInput1 cell
    actedCandidateCoefficient candidate output input0 input1 !=
      parentCoefficient output input0 input1

def actionDifferenceCoefficient (candidate cell : Nat) : Int :=
  let output := cellOutput cell
  let input0 := cellInput0 cell
  let input1 := cellInput1 cell
  actedCandidateCoefficient candidate output input0 input1 -
    parentCoefficient output input0 input1

def firstActionDifferenceCell (candidate : Nat) : Option Nat :=
  (List.range tensorCells).find? fun cell =>
    actionDifferenceCoefficient candidate cell != 0

def candidateActionImageFromSupport (candidate support : Nat) : Option Nat :=
  if support == 1 then
    match firstActionDifferenceCell candidate with
    | some cell => encodeCandidate cell (actionDifferenceCoefficient candidate cell)
    | none => none
  else none

def candidateActionImage (candidate : Nat) : Option Nat :=
  candidateActionImageFromSupport candidate (actionDifferenceSupport candidate)

structure QuotientCensus where
  actionDifferenceSupportComplete : Bool
  outside : Nat
  fixed : Nat
  pairs : Nat
  symmetryComplete : Bool
deriving Repr, BEq, DecidableEq

def quotientCensus : QuotientCensus :=
  (List.range grammarCandidates).foldl
    (fun census candidate =>
      let support := actionDifferenceSupport candidate
      let supportComplete := census.actionDifferenceSupportComplete && support != 1
      match candidateActionImageFromSupport candidate support with
      | none =>
          { census with
            actionDifferenceSupportComplete := supportComplete
            outside := census.outside + 1 }
      | some mapped =>
          let symmetric := mapped < grammarCandidates &&
            candidateActionImage mapped == some candidate
          if mapped == candidate then
            { census with
              actionDifferenceSupportComplete := supportComplete
              fixed := census.fixed + 1
              symmetryComplete := census.symmetryComplete && symmetric }
          else if candidate < mapped then
            { census with
              actionDifferenceSupportComplete := supportComplete
              pairs := census.pairs + 1
              symmetryComplete := census.symmetryComplete && symmetric }
          else
            { census with
              actionDifferenceSupportComplete := supportComplete
              symmetryComplete := census.symmetryComplete && symmetric })
    { actionDifferenceSupportComplete := true
    , outside := 0
    , fixed := 0
    , pairs := 0
    , symmetryComplete := true }

def actionDifferenceSupportCensus : Bool :=
  quotientCensus.actionDifferenceSupportComplete

def quotientPartitionCensus : Bool :=
  let quotient := quotientCensus
  quotient.symmetryComplete &&
    quotient.outside + quotient.fixed + 2 * quotient.pairs == grammarCandidates

def supportCorollaryCertificate : Bool :=
  candidateDifferenceSupportCensus &&
    (atlasDifferenceSupports.all fun support => support != 1) &&
    actionDifferenceSupportCensus

structure ClaimBoundary where
  candidateSelected : Bool
  n3Novelty : Bool
  n4Novelty : Bool
  algorithmicNovelty : Bool
  materialNovelty : Bool
  scientificNovelty : Bool
  historicalNovelty : Bool
  priorityClaim : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def claimBoundary : ClaimBoundary :=
  { candidateSelected := false
  , n3Novelty := false
  , n4Novelty := false
  , algorithmicNovelty := false
  , materialNovelty := false
  , scientificNovelty := false
  , historicalNovelty := false
  , priorityClaim := false
  , claimReady := false }

structure FormalParitySummary where
  grammarCandidates : Nat
  generated : Nat
  typedAdmitted : Nat
  typedRejected : Nat
  codecChecks : Nat
  unitBoundaryChecks : Nat
  codecComplete : Bool
  unitBoundaryComplete : Bool
  candidateDifferenceSupportComplete : Bool
  atlasDifferenceSupports : List Nat
  atlasCollisionCandidates : Nat
  n2RelativeNovelty : Nat
  separators : Nat
  c2InvolutionComplete : Bool
  unsignedCharacterComplete : Bool
  actionBaseSupport : Nat
  transportedMutationInsideGrammar : Nat
  transportedMutationOutsideGrammar : Nat
  actionDifferenceSupportComplete : Bool
  quotientSymmetryComplete : Bool
  quotientPartitionComplete : Bool
  quotientOutside : Nat
  quotientFixed : Nat
  quotientPairs : Nat
  quotientSingletons : Nat
  quotientClasses : Nat
  supportCorollaryNotEmpiricalSearch : Bool
  claimBoundary : ClaimBoundary
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  let codecComplete := codecRoundTripCensus
  let boundaryComplete := unitBoundaryCensus
  let candidateSupportComplete := candidateDifferenceSupportCensus
  let atlasSupports := atlasDifferenceSupports
  let atlasSeparated := candidateSupportComplete &&
    atlasSupports.all fun support => support != 1
  let c2Complete := c2CellInvolutionCensus
  let characterComplete := unsignedCharacterCensus
  let baseSupport := actionBaseDifferenceSupport
  let transportedInside := transportedMutationInsideGrammarCount
  let transportedOutside := grammarCandidates - transportedInside
  let quotient := quotientCensus
  let actionSupportComplete := quotient.actionDifferenceSupportComplete
  let quotientSymmetryComplete := quotient.symmetryComplete
  let quotientPartitionComplete := quotient.symmetryComplete &&
    quotient.outside + quotient.fixed + 2 * quotient.pairs == grammarCandidates
  let quotientOutside := quotient.outside
  let quotientFixed := quotient.fixed
  let quotientPairs := quotient.pairs
  let quotientSingletons := quotientOutside + quotientFixed
  let quotientClasses := quotientSingletons + quotientPairs
  let supportCorollary := candidateSupportComplete &&
    atlasSupports.all (fun support => support != 1) && actionSupportComplete
  { grammarCandidates := grammarCandidates
  , generated := (List.range grammarCandidates).length
  , typedAdmitted := if codecComplete && boundaryComplete then
      grammarCandidates else 0
  , typedRejected := if codecComplete && boundaryComplete then
      0 else grammarCandidates
  , codecChecks := grammarCandidates * 2
  , unitBoundaryChecks := grammarCandidates * 2
  , codecComplete := codecComplete
  , unitBoundaryComplete := boundaryComplete
  , candidateDifferenceSupportComplete := candidateSupportComplete
  , atlasDifferenceSupports := atlasSupports
  , atlasCollisionCandidates := if atlasSeparated then 0 else grammarCandidates
  , n2RelativeNovelty := if atlasSeparated then grammarCandidates else 0
  , separators := if atlasSeparated then
      grammarCandidates * atlasRepresentatives else 0
  , c2InvolutionComplete := c2Complete
  , unsignedCharacterComplete := characterComplete
  , actionBaseSupport := baseSupport
  , transportedMutationInsideGrammar := transportedInside
  , transportedMutationOutsideGrammar := transportedOutside
  , actionDifferenceSupportComplete := actionSupportComplete
  , quotientSymmetryComplete := quotientSymmetryComplete
  , quotientPartitionComplete := quotientPartitionComplete
  , quotientOutside := quotientOutside
  , quotientFixed := quotientFixed
  , quotientPairs := quotientPairs
  , quotientSingletons := quotientSingletons
  , quotientClasses := quotientClasses
  , supportCorollaryNotEmpiricalSearch := supportCorollary
  , claimBoundary := claimBoundary }

def frozenFormalParitySummary : FormalParitySummary :=
  { grammarCandidates := 7200
  , generated := 7200
  , typedAdmitted := 7200
  , typedRejected := 0
  , codecChecks := 14400
  , unitBoundaryChecks := 14400
  , codecComplete := true
  , unitBoundaryComplete := true
  , candidateDifferenceSupportComplete := true
  , atlasDifferenceSupports := [0, 176, 512, 474, 96, 272]
  , atlasCollisionCandidates := 0
  , n2RelativeNovelty := 7200
  , separators := 43200
  , c2InvolutionComplete := true
  , unsignedCharacterComplete := true
  , actionBaseSupport := 176
  , transportedMutationInsideGrammar := 6272
  , transportedMutationOutsideGrammar := 928
  , actionDifferenceSupportComplete := true
  , quotientSymmetryComplete := true
  , quotientPartitionComplete := true
  , quotientOutside := 7200
  , quotientFixed := 0
  , quotientPairs := 0
  , quotientSingletons := 7200
  , quotientClasses := 7200
  , supportCorollaryNotEmpiricalSearch := true
  , claimBoundary :=
      { candidateSelected := false
      , n3Novelty := false
      , n4Novelty := false
      , algorithmicNovelty := false
      , materialNovelty := false
      , scientificNovelty := false
      , historicalNovelty := false
      , priorityClaim := false
      , claimReady := false } }

theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  native_decide

theorem grammar_codec_bijection_exact :
    formalParitySummary.grammarCandidates = 7200 &&
      formalParitySummary.generated = 7200 &&
      formalParitySummary.typedAdmitted = 7200 &&
      formalParitySummary.typedRejected = 0 &&
      formalParitySummary.codecComplete &&
      formalParitySummary.unitBoundaryComplete := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem non_singleton_atlas_support_excludes_one_sparse_collision :
    formalParitySummary.candidateDifferenceSupportComplete &&
      formalParitySummary.atlasDifferenceSupports = [0, 176, 512, 474, 96, 272] &&
      formalParitySummary.atlasCollisionCandidates = 0 &&
      formalParitySummary.n2RelativeNovelty = 7200 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem support_176_plus_one_transport_cannot_have_support_one :
    formalParitySummary.actionBaseSupport = 176 &&
      formalParitySummary.actionDifferenceSupportComplete &&
      formalParitySummary.quotientOutside = 7200 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem c2_involution_and_outside_singleton_partition :
    formalParitySummary.c2InvolutionComplete &&
      formalParitySummary.unsignedCharacterComplete &&
      formalParitySummary.transportedMutationInsideGrammar = 6272 &&
      formalParitySummary.transportedMutationOutsideGrammar = 928 &&
      formalParitySummary.quotientSymmetryComplete &&
      formalParitySummary.quotientPartitionComplete &&
      formalParitySummary.quotientOutside = 7200 &&
      formalParitySummary.quotientFixed = 0 &&
      formalParitySummary.quotientPairs = 0 &&
      formalParitySummary.quotientSingletons = 7200 &&
      formalParitySummary.quotientClasses = 7200 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem separator_count_is_7200_times_6 :
    formalParitySummary.separators = 43200 &&
      formalParitySummary.separators =
        grammarCandidates * atlasRepresentatives := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem n2_does_not_promote_stronger_or_public_claims :
    formalParitySummary.supportCorollaryNotEmpiricalSearch &&
      !claimBoundary.candidateSelected &&
      !claimBoundary.n3Novelty &&
      !claimBoundary.n4Novelty &&
      !claimBoundary.algorithmicNovelty &&
      !claimBoundary.materialNovelty &&
      !claimBoundary.scientificNovelty &&
      !claimBoundary.historicalNovelty &&
      !claimBoundary.priorityClaim &&
      !claimBoundary.claimReady := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

end SounioPireusOperatorNoveltyFrontier
