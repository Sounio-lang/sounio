/-
  Concrete FORMAL_PARITY reconstruction of the frozen Pireus v13 mutation
  admission trace.

  The parent ANFs are derived from the V12 archive reconstructed in Lean, then
  the exact Sounio coefficient-toggle order is replayed through the declared
  GL(4,F2) x input-swap x basis-fixed gauge canonicalizer. Sounio remains
  semantic authority. This closes only the bounded 33-attempt / 32-admission
  census; it does not prove abstract orbit completeness or broader novelty.
-/
import SounioPireusOperatorOrbitClassReconstruction

namespace SounioPireusOperatorOrbitAdmissionReconstruction

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

abbrev Table := Array Nat
abbrev LaneMap := Array Nat

def lanes : Nat := 16
def cells : Nat := 256
def interiorCells : Nat := 225
def generatedEpochs : Nat := 16
def mutationRequests : Nat := 3600
def admissionQuota : Nat := 32
def expectedBaselineClasses : Nat := 30
def expectedMutationAttempts : Nat := 33
def expectedFinalClasses : Nat := 62

def frozenOrbitSemanticsSha256 : String :=
  "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"

def phaseIndex (left right : Nat) : Nat :=
  SounioPireusOperatorOrbitArchiveReconstruction.phaseIndex left right

def nonzeroSubmasks (mask : Nat) : List Nat :=
  (List.range lanes).filter fun value =>
    value > 0 && (value &&& mask) == value

def mobius2 (values : Array Nat) : Array Nat :=
  ((List.range interiorCells).map fun index =>
    let left := index / (lanes - 1) + 1
    let right := index % (lanes - 1) + 1
    (nonzeroSubmasks left).foldl
      (fun outer leftValue =>
        (nonzeroSubmasks right).foldl
          (fun inner rightValue =>
            inner ^^^ values.getD (phaseIndex leftValue rightValue) 0)
          outer)
      0).toArray

def phaseToAnf (phase : Array Nat) : Array Nat := mobius2 phase

def anfToPhase (anf : Array Nat) : Array Nat := mobius2 anf

def generatedParentPhases : List (Array Nat) :=
  (List.range generatedEpochs).map fun epoch =>
    SounioPireusOperatorOrbitArchiveReconstruction.phaseForArchive
      (SounioPireusOperatorOrbitArchiveReconstruction.buildEpochs epoch)

def generatedParentAnfs : List (Array Nat) :=
  generatedParentPhases.map phaseToAnf

def mutateCoefficient (parent : Array Nat) (coefficient : Nat) : Array Nat :=
  parent.set! coefficient (parent.getD coefficient 0 ^^^ 1)

def differingCells (left right : Array Nat) : Nat :=
  (List.range interiorCells).foldl
    (fun count index =>
      if left.getD index 0 != right.getD index 0 then count + 1 else count)
    0

def cdBit (left right : Nat) : Nat :=
  SounioPireusOperatorOrbitArchiveReconstruction.cdBit left right

def buildSigns (phase : Array Nat) : Table :=
  ((List.range cells).map fun cell =>
    let left := cell / lanes
    let right := cell % lanes
    let phaseBit :=
      if left == 0 || right == 0 then 0
      else phase.getD (phaseIndex left right) 0
    cdBit left right ^^^ phaseBit).toArray

def allBits (values : Array Nat) : Bool :=
  values.all fun value => value == 0 || value == 1

def unitAxesExact (table : Table) : Bool :=
  (List.range lanes).all fun lane =>
    table.getD lane 1 == 0 && table.getD (lane * lanes) 1 == 0

def mutationWellFormed
    (parent rawAnf rawPhase : Array Nat) (rawSigns : Table) : Bool :=
  parent.size == interiorCells && rawAnf.size == interiorCells &&
    rawPhase.size == interiorCells && rawSigns.size == cells &&
    allBits rawAnf && allBits rawPhase && allBits rawSigns &&
    differingCells parent rawAnf == 1 && phaseToAnf rawPhase == rawAnf &&
    unitAxesExact rawSigns

def separatorFailuresAgainst (candidate : Table) (classes : List Table) : Nat :=
  classes.foldl
    (fun failures prior =>
      if SounioPireusOperatorOrbitArchiveReconstruction.firstDifference?
          candidate prior |>.isSome
      then failures else failures + 1)
    0

structure AdmissionState where
  baseline : List Table
  classes : List Table
  attempts : Nat
  collapses : Nat
  admitted : Nat
  separatorCertificates : Nat
  separatorFailures : Nat
  mutationChecks : Nat
  mutationFailures : Nat
  transformFailures : Nat
  collapseAttempts : List Nat
  collapseClassIds : List Nat
  parentEpochs : List Nat
deriving Repr, BEq, DecidableEq

def initialAdmissionState : AdmissionState :=
  let baseline :=
    SounioPireusOperatorOrbitClassReconstruction.canonicalClasses
  { baseline := baseline
  , classes := baseline
  , attempts := 0
  , collapses := 0
  , admitted := 0
  , separatorCertificates := 0
  , separatorFailures := 0
  , mutationChecks := 0
  , mutationFailures := 0
  , transformFailures := 0
  , collapseAttempts := []
  , collapseClassIds := []
  , parentEpochs := [] }

def admissionStep
    (mappings : List LaneMap) (parentAnfs : List (Array Nat))
    (state : AdmissionState) (attempt : Nat) : AdmissionState :=
  if state.admitted >= admissionQuota then state
  else
    let parentEpoch := attempt / interiorCells
    let coefficient := attempt % interiorCells
    let parent := parentAnfs.getD parentEpoch #[]
    let rawAnf := mutateCoefficient parent coefficient
    let rawPhase := anfToPhase rawAnf
    let rawSigns := buildSigns rawPhase
    let canonical :=
      SounioPireusOperatorOrbitClassReconstruction.canonicalizeWithMaps
        mappings rawSigns
    let existingClass := state.classes.idxOf canonical
    let mutationFailures :=
      state.mutationFailures + if differingCells parent rawAnf == 1 then 0 else 1
    let transformFailures :=
      state.transformFailures +
        if mutationWellFormed parent rawAnf rawPhase rawSigns then 0 else 1
    let attempted : AdmissionState :=
      { state with
        attempts := state.attempts + 1
        mutationChecks := state.mutationChecks + interiorCells
        mutationFailures := mutationFailures
        transformFailures := transformFailures
        parentEpochs := state.parentEpochs ++ [parentEpoch] }
    if existingClass < state.classes.length then
      { attempted with
        collapses := state.collapses + 1
        collapseAttempts := state.collapseAttempts ++ [attempt]
        collapseClassIds := state.collapseClassIds ++ [existingClass] }
    else
      { attempted with
        classes := state.classes ++ [canonical]
        admitted := state.admitted + 1
        separatorCertificates :=
          state.separatorCertificates + state.classes.length
        separatorFailures :=
          state.separatorFailures +
            separatorFailuresAgainst canonical state.classes }

def finalAdmissionState : AdmissionState :=
  let mappings :=
    SounioPireusOperatorOrbitClassReconstruction.invertibleMatrixMaps
  let parentAnfs := generatedParentAnfs
  (List.range mutationRequests).foldl
    (admissionStep mappings parentAnfs) initialAdmissionState

def onlyParentEpochZero (state : AdmissionState) : Bool :=
  state.parentEpochs.length == state.attempts &&
    state.parentEpochs.all fun epoch => epoch == 0

def baselinePrefixPreserved (state : AdmissionState) : Bool :=
  state.baseline.length == expectedBaselineClasses &&
    state.classes.take state.baseline.length == state.baseline

def allFinalClassesDistinct (state : AdmissionState) : Bool :=
  state.classes.eraseDups.length == state.classes.length

structure AdmissionReconstructionSummary where
  orbitSemanticsSha256 : String
  mutationRequests : Nat
  baselineClasses : Nat
  mutationAttempts : Nat
  equivalentCollapses : Nat
  collapseAttempts : List Nat
  collapseClassIds : List Nat
  admittedClasses : Nat
  finalClasses : Nat
  mutationChecks : Nat
  mutationFailures : Nat
  transformFailures : Nat
  separatorCertificates : Nat
  separatorFailures : Nat
  parentEpochsUsed : Nat
  onlyParentEpochZero : Bool
  baselinePrefixPreserved : Bool
  finalClassesDistinct : Bool
  baselineCanonicalizations : Nat
  mutationCanonicalizations : Nat
  totalCanonicalizations : Nat
  actionViewsPerCanonicalization : Nat
  totalActionViews : Nat
  concrete32AdmissionReconstructionComplete : Bool
  canonicalRepresentativeIffDeclaredOrbitProved : Bool
  nonlinearPermutationQuotientComplete : Bool
  unrestrictedAlgebraIsomorphismComplete : Bool
  globalNoveltyProved : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def admissionReconstructionSummary : AdmissionReconstructionSummary :=
  let state := finalAdmissionState
  let baselineCanonicalizations :=
    SounioPireusOperatorOrbitClassReconstruction.expectedImages
  let mutationCanonicalizations := state.attempts
  let totalCanonicalizations := baselineCanonicalizations + mutationCanonicalizations
  let actionViewsPerCanonicalization :=
    SounioPireusOperatorOrbitClassReconstruction.expectedActions
  let complete :=
    baselinePrefixPreserved state &&
      state.attempts == expectedMutationAttempts && state.collapses == 1 &&
      state.collapseAttempts == [15] && state.collapseClassIds == [31] &&
      state.admitted == admissionQuota &&
      state.classes.length == expectedFinalClasses &&
      state.mutationChecks == 7425 && state.mutationFailures == 0 &&
      state.transformFailures == 0 &&
      state.separatorCertificates == 1456 && state.separatorFailures == 0 &&
      onlyParentEpochZero state && allFinalClassesDistinct state &&
      totalCanonicalizations == 161 &&
      totalCanonicalizations * actionViewsPerCanonicalization == 6491520
  { orbitSemanticsSha256 := frozenOrbitSemanticsSha256
  , mutationRequests := mutationRequests
  , baselineClasses := state.baseline.length
  , mutationAttempts := state.attempts
  , equivalentCollapses := state.collapses
  , collapseAttempts := state.collapseAttempts
  , collapseClassIds := state.collapseClassIds
  , admittedClasses := state.admitted
  , finalClasses := state.classes.length
  , mutationChecks := state.mutationChecks
  , mutationFailures := state.mutationFailures
  , transformFailures := state.transformFailures
  , separatorCertificates := state.separatorCertificates
  , separatorFailures := state.separatorFailures
  , parentEpochsUsed := state.parentEpochs.eraseDups.length
  , onlyParentEpochZero := onlyParentEpochZero state
  , baselinePrefixPreserved := baselinePrefixPreserved state
  , finalClassesDistinct := allFinalClassesDistinct state
  , baselineCanonicalizations := baselineCanonicalizations
  , mutationCanonicalizations := mutationCanonicalizations
  , totalCanonicalizations := totalCanonicalizations
  , actionViewsPerCanonicalization := actionViewsPerCanonicalization
  , totalActionViews := totalCanonicalizations * actionViewsPerCanonicalization
  , concrete32AdmissionReconstructionComplete := complete
  , canonicalRepresentativeIffDeclaredOrbitProved := false
  , nonlinearPermutationQuotientComplete := false
  , unrestrictedAlgebraIsomorphismComplete := false
  , globalNoveltyProved := false
  , formalParityClosed := false
  , claimReady := false }

def frozenAdmissionReconstructionSummary : AdmissionReconstructionSummary :=
  { orbitSemanticsSha256 :=
      "0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c"
  , mutationRequests := 3600
  , baselineClasses := 30
  , mutationAttempts := 33
  , equivalentCollapses := 1
  , collapseAttempts := [15]
  , collapseClassIds := [31]
  , admittedClasses := 32
  , finalClasses := 62
  , mutationChecks := 7425
  , mutationFailures := 0
  , transformFailures := 0
  , separatorCertificates := 1456
  , separatorFailures := 0
  , parentEpochsUsed := 1
  , onlyParentEpochZero := true
  , baselinePrefixPreserved := true
  , finalClassesDistinct := true
  , baselineCanonicalizations := 128
  , mutationCanonicalizations := 33
  , totalCanonicalizations := 161
  , actionViewsPerCanonicalization := 40320
  , totalActionViews := 6491520
  , concrete32AdmissionReconstructionComplete := true
  , canonicalRepresentativeIffDeclaredOrbitProved := false
  , nonlinearPermutationQuotientComplete := false
  , unrestrictedAlgebraIsomorphismComplete := false
  , globalNoveltyProved := false
  , formalParityClosed := false
  , claimReady := false }

theorem concrete_admission_reconstruction_matches_declared_frozen_summary :
    admissionReconstructionSummary = frozenAdmissionReconstructionSummary := by
  native_decide

theorem coefficient_toggle_trace_has_one_collapse_and_32_admissions :
    admissionReconstructionSummary.mutationAttempts = 33 ∧
      admissionReconstructionSummary.equivalentCollapses = 1 ∧
      admissionReconstructionSummary.collapseAttempts = [15] ∧
      admissionReconstructionSummary.collapseClassIds = [31] ∧
      admissionReconstructionSummary.admittedClasses = 32 := by
  rw [concrete_admission_reconstruction_matches_declared_frozen_summary]
  decide

theorem every_admission_has_exact_noncollision_separators :
    admissionReconstructionSummary.finalClasses = 62 ∧
      admissionReconstructionSummary.separatorCertificates = 1456 ∧
      admissionReconstructionSummary.separatorFailures = 0 ∧
      admissionReconstructionSummary.baselinePrefixPreserved ∧
      admissionReconstructionSummary.finalClassesDistinct := by
  rw [concrete_admission_reconstruction_matches_declared_frozen_summary]
  decide

theorem bounded_admission_census_does_not_prove_global_novelty :
    admissionReconstructionSummary.concrete32AdmissionReconstructionComplete &&
      !admissionReconstructionSummary.canonicalRepresentativeIffDeclaredOrbitProved &&
      !admissionReconstructionSummary.nonlinearPermutationQuotientComplete &&
      !admissionReconstructionSummary.unrestrictedAlgebraIsomorphismComplete &&
      !admissionReconstructionSummary.globalNoveltyProved &&
      !admissionReconstructionSummary.formalParityClosed &&
      !admissionReconstructionSummary.claimReady := by
  rw [concrete_admission_reconstruction_matches_declared_frozen_summary]
  decide

end SounioPireusOperatorOrbitAdmissionReconstruction
