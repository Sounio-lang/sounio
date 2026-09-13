/-
  FORMAL_PARITY for the Sounio-owned Pireus Operator-Lowering Forge v6.

  Sounio remains SEMANTIC_AUTHORITY. This file begins only after the exact v6
  Sounio semantics reached PARITY_OPEN. It independently reconstructs the
  declared finite candidate indexing, the serialization quotient, target
  envelope partition, seed taxonomy, and obligation ledger. It does not add a
  candidate, discharge a target obligation, admit or select a lowering, or
  make a novelty claim.
-/

namespace SounioPireusOperatorLoweringForge

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0"

def frozenSounioSemanticsSha256 : String :=
  "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"

def operatorClassCount : Nat := 14
def parentChildCount : Nat := 48
def targetEnvelopeCount : Nat := 4
def packWidthCount : Nat := 5
def routeCount : Nat := 2
def serializationCount : Nat := 2
def candidateCount : Nat := 1120
def programClassCount : Nat := 560
def obligationsPerCandidate : Nat := 12
def dischargedPerCandidate : Nat := 3
def unresolvedPerCandidate : Nat := 9

def countWhere (upper : Nat) (predicate : Nat -> Bool) : Nat :=
  (List.range upper).foldl
    (fun count value => if predicate value then count + 1 else count) 0

def serializationIndex (candidate : Nat) : Nat := candidate % serializationCount
def routeIndex (candidate : Nat) : Nat := (candidate / serializationCount) % routeCount
def packIndex (candidate : Nat) : Nat :=
  (candidate / (serializationCount * routeCount)) % packWidthCount
def targetIndex (candidate : Nat) : Nat :=
  (candidate / (serializationCount * routeCount * packWidthCount)) %
    targetEnvelopeCount
def operatorIndex (candidate : Nat) : Nat :=
  candidate /
    (serializationCount * routeCount * packWidthCount * targetEnvelopeCount)

def encodeCandidate
    (operator target pack route serialization : Nat) : Nat :=
  (((operator * targetEnvelopeCount + target) * packWidthCount + pack) *
      routeCount + route) * serializationCount + serialization

def candidateIndexRoundtrip : Bool :=
  (List.range candidateCount).all fun candidate =>
    encodeCandidate
      (operatorIndex candidate)
      (targetIndex candidate)
      (packIndex candidate)
      (routeIndex candidate)
      (serializationIndex candidate) == candidate

def candidateCoordinatesInRange : Bool :=
  (List.range candidateCount).all fun candidate =>
    operatorIndex candidate < operatorClassCount &&
      targetIndex candidate < targetEnvelopeCount &&
      packIndex candidate < packWidthCount &&
      routeIndex candidate < routeCount &&
      serializationIndex candidate < serializationCount

def programKey (candidate : Nat) : Nat := candidate / serializationCount

def relationByKey (key : Nat -> Nat) (left right : Nat) : Prop :=
  key left = key right

theorem relationByKey_equivalence (key : Nat -> Nat) :
    Equivalence (relationByKey key) := by
  constructor
  · intro value
    rfl
  · intro left right equality
    exact equality.symm
  · intro left middle right leftMiddle middleRight
    exact Eq.trans leftMiddle middleRight

def programEquivalent : Nat -> Nat -> Prop := relationByKey programKey

theorem program_equivalence_is_equivalence : Equivalence programEquivalent :=
  relationByKey_equivalence programKey

def exactSerializationPairs : Bool :=
  (List.range programClassCount).all fun program =>
    let canonical := program * serializationCount
    let alternate := canonical + 1
    canonical < candidateCount && alternate < candidateCount &&
      programKey canonical == program && programKey alternate == program &&
      serializationIndex canonical == 0 && serializationIndex alternate == 1

def distinctProgramKeysSeparate : Bool :=
  (List.range programClassCount).all fun left =>
    (List.range programClassCount).all fun right =>
      left == right || programKey (left * serializationCount) !=
        programKey (right * serializationCount)

def targetPopulation (target : Nat) : Nat :=
  countWhere candidateCount fun candidate => targetIndex candidate == target

def targetEnvelopePartitionExact : Bool :=
  targetPopulation 0 == 280 && targetPopulation 1 == 280 &&
    targetPopulation 2 == 280 && targetPopulation 3 == 280

def loweringSeed : Nat := 1
def primitiveSeed : Nat := 2
def fabricSeed : Nat := 3
def operatorSeed : Nat := 4

def seedKind (candidate : Nat) : Nat :=
  if routeIndex candidate == 1 then loweringSeed
  else if targetIndex candidate == 3 then fabricSeed
  else primitiveSeed

def seedPopulation (kind : Nat) : Nat :=
  countWhere candidateCount fun candidate => seedKind candidate == kind

def candidateAdmitted (candidate : Nat) : Bool :=
  candidate < candidateCount && unresolvedPerCandidate == 0

def admittedPopulation : Nat := countWhere candidateCount candidateAdmitted

def allCandidatesCarryDeclaredDebt : Bool :=
  (List.range candidateCount).all fun candidate =>
    candidate < candidateCount && unresolvedPerCandidate > 0 &&
      dischargedPerCandidate + unresolvedPerCandidate == obligationsPerCandidate

def operatorRepresentatives : List Nat :=
  [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]

def parentRepresentativesBoundAndUnique : Bool :=
  operatorRepresentatives.length == operatorClassCount &&
    operatorRepresentatives.eraseDups.length == operatorClassCount &&
    operatorRepresentatives.all fun representative => representative < parentChildCount

structure FormalParitySummary where
  sourceSha256 : String
  semanticsSha256 : String
  operatorClasses : Nat
  targetEnvelopes : Nat
  packWidths : Nat
  routes : Nat
  serializations : Nat
  candidates : Nat
  candidateRoundtrip : Bool
  coordinateBounds : Bool
  programClasses : Nat
  serializationPairsExact : Bool
  distinctProgramKeysSeparate : Bool
  machineEnvelopeClasses : Nat
  machineEnvelopePopulation0 : Nat
  machineEnvelopePopulation1 : Nat
  machineEnvelopePopulation2 : Nat
  machineEnvelopePopulation3 : Nat
  loweringSeeds : Nat
  primitiveSeeds : Nat
  fabricSeeds : Nat
  operatorSeeds : Nat
  dischargedObligations : Nat
  unresolvedObligations : Nat
  obligationLedgerExact : Bool
  admittedLowerings : Nat
  selectedCandidate : Int
  parentRepresentatives : List Nat
  parentRepresentativesBoundAndUnique : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  { sourceSha256 := frozenSounioSourceSha256
    semanticsSha256 := frozenSounioSemanticsSha256
    operatorClasses := operatorClassCount
    targetEnvelopes := targetEnvelopeCount
    packWidths := packWidthCount
    routes := routeCount
    serializations := serializationCount
    candidates := operatorClassCount * targetEnvelopeCount * packWidthCount *
      routeCount * serializationCount
    candidateRoundtrip := candidateIndexRoundtrip
    coordinateBounds := candidateCoordinatesInRange
    programClasses := candidateCount / serializationCount
    serializationPairsExact := exactSerializationPairs
    distinctProgramKeysSeparate := distinctProgramKeysSeparate
    machineEnvelopeClasses := targetEnvelopeCount
    machineEnvelopePopulation0 := targetPopulation 0
    machineEnvelopePopulation1 := targetPopulation 1
    machineEnvelopePopulation2 := targetPopulation 2
    machineEnvelopePopulation3 := targetPopulation 3
    loweringSeeds := seedPopulation loweringSeed
    primitiveSeeds := seedPopulation primitiveSeed
    fabricSeeds := seedPopulation fabricSeed
    operatorSeeds := seedPopulation operatorSeed
    dischargedObligations := candidateCount * dischargedPerCandidate
    unresolvedObligations := candidateCount * unresolvedPerCandidate
    obligationLedgerExact := allCandidatesCarryDeclaredDebt
    admittedLowerings := admittedPopulation
    selectedCandidate := -1
    parentRepresentatives := operatorRepresentatives
    parentRepresentativesBoundAndUnique := parentRepresentativesBoundAndUnique
    claimReady := false }

def frozenFormalParitySummary : FormalParitySummary :=
  { sourceSha256 :=
      "178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0"
    semanticsSha256 :=
      "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"
    operatorClasses := 14
    targetEnvelopes := 4
    packWidths := 5
    routes := 2
    serializations := 2
    candidates := 1120
    candidateRoundtrip := true
    coordinateBounds := true
    programClasses := 560
    serializationPairsExact := true
    distinctProgramKeysSeparate := true
    machineEnvelopeClasses := 4
    machineEnvelopePopulation0 := 280
    machineEnvelopePopulation1 := 280
    machineEnvelopePopulation2 := 280
    machineEnvelopePopulation3 := 280
    loweringSeeds := 560
    primitiveSeeds := 420
    fabricSeeds := 140
    operatorSeeds := 0
    dischargedObligations := 3360
    unresolvedObligations := 10080
    obligationLedgerExact := true
    admittedLowerings := 0
    selectedCandidate := -1
    parentRepresentatives :=
      [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19]
    parentRepresentativesBoundAndUnique := true
    claimReady := false }

theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  decide

theorem candidate_index_roundtrip_and_grammar_cardinality :
    formalParitySummary.candidates = 1120 &&
      formalParitySummary.candidateRoundtrip &&
      formalParitySummary.coordinateBounds := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem program_serialization_quotient_exact :
    formalParitySummary.programClasses = 560 &&
      formalParitySummary.serializationPairsExact &&
      formalParitySummary.distinctProgramKeysSeparate := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem machine_envelope_partition_exact :
    formalParitySummary.machineEnvelopeClasses = 4 &&
      formalParitySummary.machineEnvelopePopulation0 = 280 &&
      formalParitySummary.machineEnvelopePopulation1 = 280 &&
      formalParitySummary.machineEnvelopePopulation2 = 280 &&
      formalParitySummary.machineEnvelopePopulation3 = 280 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem parent_representative_lineage_bound :
    formalParitySummary.parentRepresentatives =
        [0, 1, 2, 3, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19] &&
      formalParitySummary.parentRepresentativesBoundAndUnique := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem residual_seed_partition_exact :
    formalParitySummary.loweringSeeds = 560 &&
      formalParitySummary.primitiveSeeds = 420 &&
      formalParitySummary.fabricSeeds = 140 &&
      formalParitySummary.operatorSeeds = 0 &&
      formalParitySummary.loweringSeeds + formalParitySummary.primitiveSeeds +
        formalParitySummary.fabricSeeds + formalParitySummary.operatorSeeds =
          formalParitySummary.candidates := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem obligation_ledger_and_no_admission :
    formalParitySummary.dischargedObligations = 3360 &&
      formalParitySummary.unresolvedObligations = 10080 &&
      formalParitySummary.obligationLedgerExact &&
      formalParitySummary.admittedLowerings = 0 &&
      formalParitySummary.selectedCandidate = -1 &&
      !formalParitySummary.claimReady := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

end SounioPireusOperatorLoweringForge
