/-
  Shared FORMAL_PARITY definitions for the Sounio-owned Pireus Operator
  Novelty Feedback v7.

  Sounio remains SEMANTIC_AUTHORITY. This file starts only after the exact v7
  Sounio source and semantics reached PARITY_OPEN. It reconstructs the frozen
  CD16 challenge, checks the 12 inherited parent actions, evaluates all 14x12
  representative/action residuals, and checks the canonical bridge-or-seed
  result. It creates no semantics, expected result, operator, material
  lowering, broad novelty statement, or claim.
-/
import SounioPireusQuotientNoveltyForge

namespace SounioPireusOperatorNoveltyFeedback

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf"

def frozenSounioSemanticsSha256 : String :=
  "a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726"

def frozenQuotientParentSemanticsSha256 : String :=
  "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"

def frozenChallengeParentSemanticsSha256 : String :=
  "9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970"

def dimension : Nat := 16
def cellCount : Nat := 256
def wordCount : Nat := 8
def wordBits : Nat := 32

abbrev ParentAction := SounioPireusQuotientNoveltyForge.Action

def challengeBit (left right : Nat) : Nat :=
  if SounioCDCocycle.cdSigma left right 4 < 0 then 1 else 0

def frozenParentActions : List ParentAction :=
  [ { matrix := 33345, swap := 0 }
  , { matrix := 33345, swap := 1 }
  , { matrix := 33377, swap := 0 }
  , { matrix := 33377, swap := 1 }
  , { matrix := 33825, swap := 0 }
  , { matrix := 33825, swap := 1 }
  , { matrix := 33889, swap := 0 }
  , { matrix := 33889, swap := 1 }
  , { matrix := 34337, swap := 0 }
  , { matrix := 34337, swap := 1 }
  , { matrix := 34369, swap := 0 }
  , { matrix := 34369, swap := 1 } ]

def frozenRepresentatives : List Nat :=
  [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]

def actionCode (action : ParentAction) : Nat := action.matrix * 2 + action.swap

def countWhere (upper : Nat) (predicate : Nat -> Bool) : Nat :=
  (List.range upper).foldl
    (fun count value => if predicate value then count + 1 else count) 0

def packBitWord (cellBit : Nat -> Nat -> Nat) (word : Nat) : Nat :=
  (List.range wordBits).foldl
    (fun packed bit =>
      let cell := word * wordBits + bit
      if cellBit (cell / dimension) (cell % dimension) == 1 then packed + 2 ^ bit
      else packed)
    0

def packBitWords (cellBit : Nat -> Nat -> Nat) : List Nat :=
  (List.range wordCount).map (packBitWord cellBit)

def wordNonzero (word : Nat) : Nat :=
  countWhere wordBits fun bit => SounioPireusQuotientNoveltyForge.bit word bit == 1

def packedNonzero (words : List Nat) : Nat :=
  (words.map wordNonzero).sum

def representativeDescriptor (classId : Nat) : SounioPireusQuotientNoveltyForge.Descriptor :=
  let child := frozenRepresentatives.getD classId 0
  SounioPireusQuotientNoveltyForge.descriptors.getD child
    { kind := 0, r := 0, s := 0, t := 0 }

def representativeBit (classId left right : Nat) : Nat :=
  let descriptor := representativeDescriptor classId
  SounioPireusQuotientNoveltyForge.parentBit left right ^^^
    SounioPireusQuotientNoveltyForge.mutationBit descriptor left right

def transformBit
    (source : Nat -> Nat -> Nat) (action : ParentAction) (left right : Nat) : Nat :=
  let mappedLeft := SounioPireusQuotientNoveltyForge.matrixApply action.matrix left
  let mappedRight := SounioPireusQuotientNoveltyForge.matrixApply action.matrix right
  if action.swap == 1 then source mappedRight mappedLeft else source mappedLeft mappedRight

def normalizerValuesFor (cellBit : Nat -> Nat -> Nat) : Array Nat := Id.run do
  let mut values := Array.replicate dimension 0
  for vector in List.range dimension do
    if vector != 0 && !SounioPireusQuotientNoveltyForge.isBasis vector then
      let edge := SounioPireusQuotientNoveltyForge.highestBasis vector
      let parent := vector ^^^ edge
      values := values.set! vector (cellBit parent edge ^^^ values.getD parent 0)
  return values

def normalizerGaugeFor (cellBit : Nat -> Nat -> Nat) : Nat :=
  let values := normalizerValuesFor cellBit
  SounioPireusQuotientNoveltyForge.gaugeVectors.foldl
    (fun word vector =>
      if values.getD vector 0 == 1 then
        word + 2 ^ SounioPireusQuotientNoveltyForge.gaugeIndex vector
      else word)
    0

def parentDifferenceBit (action : ParentAction) (left right : Nat) : Nat :=
  transformBit SounioPireusQuotientNoveltyForge.parentBit action left right ^^^
    SounioPireusQuotientNoveltyForge.parentBit left right

def parentGauge (action : ParentAction) : Nat :=
  normalizerGaugeFor (parentDifferenceBit action)

def pairDifferenceBit (classId : Nat) (action : ParentAction) (left right : Nat) : Nat :=
  challengeBit left right ^^^ transformBit (representativeBit classId) action left right

def residualBit
    (difference : Nat -> Nat -> Nat) (gauge left right : Nat) : Nat :=
  difference left right ^^^
    SounioPireusQuotientNoveltyForge.coboundary gauge left right

theorem xor_replay_identity (transformed coboundary challenge : Nat) :
    transformed ^^^ coboundary ^^^ ((challenge ^^^ transformed) ^^^ coboundary) =
      challenge := by
  calc
    transformed ^^^ coboundary ^^^ ((challenge ^^^ transformed) ^^^ coboundary) =
        (transformed ^^^ transformed) ^^^ (coboundary ^^^ coboundary) ^^^ challenge := by
      ac_rfl
    _ = challenge := by
      rw [Nat.xor_self, Nat.zero_xor, Nat.xor_self, Nat.zero_xor]

theorem residual_replay_identity
    (classId : Nat) (action : ParentAction) (gauge left right : Nat) :
    transformBit (representativeBit classId) action left right ^^^
        SounioPireusQuotientNoveltyForge.coboundary gauge left right ^^^
        residualBit (pairDifferenceBit classId action) gauge left right =
      challengeBit left right := by
  simpa [residualBit, pairDifferenceBit] using
    xor_replay_identity
      (transformBit (representativeBit classId) action left right)
      (SounioPireusQuotientNoveltyForge.coboundary gauge left right)
      (challengeBit left right)

structure PairWitness where
  classId : Nat
  representative : Nat
  actionIndex : Nat
  actionCode : Nat
  matrix : Nat
  swap : Nat
  parentGauge : Nat
  challengeGauge : Nat
  residualWords : List Nat
  residualNonzero : Nat
  replayChecks : Nat
  replayFailures : Nat
deriving Repr, BEq, DecidableEq

def pairWitness (classId actionIndex : Nat) : PairWitness :=
  let representative := frozenRepresentatives.getD classId 0
  let action := frozenParentActions.getD actionIndex { matrix := 0, swap := 0 }
  let difference := pairDifferenceBit classId action
  let gauge := normalizerGaugeFor difference
  let residualWords := packBitWords (residualBit difference gauge)
  { classId := classId
    representative := representative
    actionIndex := actionIndex
    actionCode := actionCode action
    matrix := action.matrix
    swap := action.swap
    parentGauge := parentGauge action
    challengeGauge := gauge
    residualWords := residualWords
    residualNonzero := packedNonzero residualWords
    -- residual_replay_identity discharges every one of the 256 replay equations.
    replayChecks := cellCount
    replayFailures := 0 }

def wordsLess : List Nat -> List Nat -> Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | left :: leftTail, right :: rightTail =>
      if left < right then true
      else if left > right then false
      else wordsLess leftTail rightTail

def pairLess (left right : PairWitness) : Bool :=
  if left.residualNonzero < right.residualNonzero then true
  else if left.residualNonzero > right.residualNonzero then false
  else if wordsLess left.residualWords right.residualWords then true
  else if left.residualWords != right.residualWords then false
  else if left.classId < right.classId then true
  else if left.classId > right.classId then false
  else if left.representative < right.representative then true
  else if left.representative > right.representative then false
  else if left.matrix < right.matrix then true
  else if left.matrix > right.matrix then false
  else if left.swap < right.swap then true
  else if left.swap > right.swap then false
  else left.challengeGauge < right.challengeGauge

def emptyPair : PairWitness :=
  { classId := 0, representative := 0, actionIndex := 0, actionCode := 0
    matrix := 0, swap := 0, parentGauge := 0, challengeGauge := 0
    residualWords := [], residualNonzero := cellCount + 1
    replayChecks := 0, replayFailures := cellCount }

def frozenBestPair : PairWitness :=
  { classId := 8
    representative := 13
    actionIndex := 8
    actionCode := 68674
    matrix := 34337
    swap := 0
    parentGauge := 1097
    challengeGauge := 1813
    residualWords :=
      [0, 0, 1010580540, 4042322160, 2863311530, 2863311530, 2526451350, 1515870810]
    residualNonzero := 96
    replayChecks := 256
    replayFailures := 0 }

def classCertificate (classId : Nat) : Bool :=
  (List.range frozenParentActions.length).all fun actionIndex =>
    let candidate := pairWitness classId actionIndex
    candidate.residualNonzero > 0 && !pairLess candidate frozenBestPair

def allClassCertificates : Bool :=
  classCertificate 0 && classCertificate 1 && classCertificate 2 &&
  classCertificate 3 && classCertificate 4 && classCertificate 5 &&
  classCertificate 6 && classCertificate 7 && classCertificate 8 &&
  classCertificate 9 && classCertificate 10 && classCertificate 11 &&
  classCertificate 12 && classCertificate 13

def frozenBestMember : Bool := pairWitness 8 8 == frozenBestPair

def parentActionsAdmitted : Bool :=
  frozenParentActions.all
    (SounioPireusQuotientNoveltyForge.parentActionAdmitted
      SounioPireusQuotientNoveltyForge.parentTable)

def representativesBoundAndUnique : Bool :=
  -- Roster hygiene only. Quotient-class membership belongs to the frozen v5
  -- parent relation and is bound externally by its source and semantics hashes.
  frozenRepresentatives.length == 14 &&
    frozenRepresentatives.eraseDups.length == 14 &&
    frozenRepresentatives.all fun representative => representative < 48

structure ChallengeProfile where
  positive : Nat
  negative : Nat
  words : List Nat
deriving Repr, BEq, DecidableEq

def challengeProfile : ChallengeProfile :=
  let words := packBitWords challengeBit
  let negative := packedNonzero words
  { positive := cellCount - negative, negative := negative, words := words }

def frozenChallengeProfile : ChallengeProfile :=
  { positive := 136
    negative := 120
    words :=
      [2523529216, 1521237190, 2859790366, 3434243800,
       2543059454, 1532116280, 2878009824, 3444336422] }

structure ParentProfile where
  actionCount : Nat
  actionCodes : List Nat
  gauges : List Nat
  admitted : Bool
  closure : Bool
  inverses : Bool
deriving Repr, BEq, DecidableEq

def parentProfile : ParentProfile :=
  { actionCount := frozenParentActions.length
    actionCodes := frozenParentActions.map actionCode
    gauges := frozenParentActions.map parentGauge
    admitted := parentActionsAdmitted
    closure := SounioPireusQuotientNoveltyForge.actionClosure frozenParentActions
    inverses := frozenParentActions.all
      (SounioPireusQuotientNoveltyForge.actionInverseExists frozenParentActions) }

def frozenParentProfile : ParentProfile :=
  { actionCount := 12
    actionCodes :=
      [66690, 66691, 66754, 66755, 67650, 67651,
       67778, 67779, 68674, 68675, 68738, 68739]
    gauges := [0, 2027, 1097, 930, 0, 2027, 1290, 737, 1097, 930, 1290, 737]
    admitted := true
    closure := true
    inverses := true }

structure AtlasProfile where
  classCount : Nat
  representatives : List Nat
  representativesBoundAndUnique : Bool
  pairCount : Nat
  pairReplayFailures : Nat
  zeroResidualHits : Nat
  exhaustiveNonmembership : Bool
  outcomeKind : Nat
  existingClassBridge : Bool
  operatorSeedGenerated : Bool
  best : PairWitness
deriving Repr, BEq, DecidableEq

def atlasProfile : AtlasProfile :=
  -- This parity module reconstructs the nonmembership branch observed in the
  -- frozen Sounio execution. A failed certificate yields neither bridge nor
  -- seed; it does not synthesize the unobserved positive-bridge branch.
  let exhaustive := allClassCertificates && frozenBestMember
  let pairCount := frozenRepresentatives.length * frozenParentActions.length
  let zeroHits := if exhaustive then 0 else pairCount
  let replayFailureCount := if exhaustive then 0 else pairCount * cellCount
  let best := if exhaustive then frozenBestPair else emptyPair
  let seedGenerated := pairCount > 0 && zeroHits == 0 && best.residualNonzero > 0
  { classCount := frozenRepresentatives.length
    representatives := frozenRepresentatives
    representativesBoundAndUnique := representativesBoundAndUnique
    pairCount := pairCount
    pairReplayFailures := replayFailureCount
    zeroResidualHits := zeroHits
    exhaustiveNonmembership := exhaustive && pairCount == 168 && zeroHits == 0
    outcomeKind := if seedGenerated then 2 else 0
    existingClassBridge := false
    operatorSeedGenerated := seedGenerated
    best := best }

def frozenAtlasProfile : AtlasProfile :=
  { classCount := 14
    representatives := [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]
    representativesBoundAndUnique := true
    pairCount := 168
    pairReplayFailures := 0
    zeroResidualHits := 0
    exhaustiveNonmembership := true
    outcomeKind := 2
    existingClassBridge := false
    operatorSeedGenerated := true
    best := frozenBestPair }

structure FormalParitySummary where
  sourceSha256 : String
  semanticsSha256 : String
  quotientParentSemanticsSha256 : String
  challengeParentSemanticsSha256 : String
  challengePositive : Nat
  challengeNegative : Nat
  challengeWords : List Nat
  actionCount : Nat
  actionCodes : List Nat
  parentGauges : List Nat
  parentActionsAdmitted : Bool
  parentActionClosure : Bool
  parentActionInverses : Bool
  classCount : Nat
  representatives : List Nat
  representativesBoundAndUnique : Bool
  pairCount : Nat
  pairReplayFailures : Nat
  zeroResidualHits : Nat
  exhaustiveNonmembership : Bool
  outcomeKind : Nat
  existingClassBridge : Bool
  operatorSeedGenerated : Bool
  bestClass : Nat
  bestRepresentative : Nat
  bestActionIndex : Nat
  bestActionCode : Nat
  bestMatrix : Nat
  bestSwap : Nat
  bestParentGauge : Nat
  bestChallengeGauge : Nat
  bestResidualWords : List Nat
  bestResidualNonzero : Nat
  bestReplayChecks : Nat
  bestReplayFailures : Nat
  broadNovelty : Bool
  historicalNovelty : Bool
  priorityClaim : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  let challenge := challengeProfile
  let parent := parentProfile
  let atlas := atlasProfile
  { sourceSha256 := frozenSounioSourceSha256
    semanticsSha256 := frozenSounioSemanticsSha256
    quotientParentSemanticsSha256 := frozenQuotientParentSemanticsSha256
    challengeParentSemanticsSha256 := frozenChallengeParentSemanticsSha256
    challengePositive := challenge.positive
    challengeNegative := challenge.negative
    challengeWords := challenge.words
    actionCount := parent.actionCount
    actionCodes := parent.actionCodes
    parentGauges := parent.gauges
    parentActionsAdmitted := parent.admitted
    parentActionClosure := parent.closure
    parentActionInverses := parent.inverses
    classCount := atlas.classCount
    representatives := atlas.representatives
    representativesBoundAndUnique := atlas.representativesBoundAndUnique
    pairCount := atlas.pairCount
    pairReplayFailures := atlas.pairReplayFailures
    zeroResidualHits := atlas.zeroResidualHits
    exhaustiveNonmembership := atlas.exhaustiveNonmembership
    outcomeKind := atlas.outcomeKind
    existingClassBridge := atlas.existingClassBridge
    operatorSeedGenerated := atlas.operatorSeedGenerated
    bestClass := atlas.best.classId
    bestRepresentative := atlas.best.representative
    bestActionIndex := atlas.best.actionIndex
    bestActionCode := atlas.best.actionCode
    bestMatrix := atlas.best.matrix
    bestSwap := atlas.best.swap
    bestParentGauge := atlas.best.parentGauge
    bestChallengeGauge := atlas.best.challengeGauge
    bestResidualWords := atlas.best.residualWords
    bestResidualNonzero := atlas.best.residualNonzero
    bestReplayChecks := atlas.best.replayChecks
    bestReplayFailures := atlas.best.replayFailures
    broadNovelty := false
    historicalNovelty := false
    priorityClaim := false
    claimReady := false }

def frozenFormalParitySummary : FormalParitySummary :=
  { sourceSha256 :=
      "b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf"
    semanticsSha256 :=
      "a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726"
    quotientParentSemanticsSha256 :=
      "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"
    challengeParentSemanticsSha256 :=
      "9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970"
    challengePositive := 136
    challengeNegative := 120
    challengeWords :=
      [2523529216, 1521237190, 2859790366, 3434243800,
       2543059454, 1532116280, 2878009824, 3444336422]
    actionCount := 12
    actionCodes :=
      [66690, 66691, 66754, 66755, 67650, 67651,
       67778, 67779, 68674, 68675, 68738, 68739]
    parentGauges := [0, 2027, 1097, 930, 0, 2027, 1290, 737, 1097, 930, 1290, 737]
    parentActionsAdmitted := true
    parentActionClosure := true
    parentActionInverses := true
    classCount := 14
    representatives := [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]
    representativesBoundAndUnique := true
    pairCount := 168
    pairReplayFailures := 0
    zeroResidualHits := 0
    exhaustiveNonmembership := true
    outcomeKind := 2
    existingClassBridge := false
    operatorSeedGenerated := true
    bestClass := 8
    bestRepresentative := 13
    bestActionIndex := 8
    bestActionCode := 68674
    bestMatrix := 34337
    bestSwap := 0
    bestParentGauge := 1097
    bestChallengeGauge := 1813
    bestResidualWords :=
      [0, 0, 1010580540, 4042322160, 2863311530, 2863311530, 2526451350, 1515870810]
    bestResidualNonzero := 96
    bestReplayChecks := 256
    bestReplayFailures := 0
    broadNovelty := false
    historicalNovelty := false
    priorityClaim := false
    claimReady := false }

end SounioPireusOperatorNoveltyFeedback
