/-
  FORMAL_PARITY for the Sounio-owned Pireus Operator Discovery Engine v10.

  The source and semantics hashes below identify the already-frozen Sounio
  authority. This file independently reconstructs the finite C2_diag action,
  the three-class bilinear atlas, candidate zero, its six separators, and the
  bounded law spectrum. It cannot create semantics, an expected result, a
  material observation, historical novelty, priority, or CLAIM_READY status.
-/
import SounioCDCocycle

namespace SounioPireusOperatorDiscoveryEngine

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117"

def frozenSounioSemanticsSha256 : String :=
  "2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5"

def frozenParentSemanticsSha256 : String :=
  "8fad13c09d2b17ea1adcce0a6b89612964e80ddb4a7a576916c2c1da30286df6"

def dimension : Nat := 16
def tensorCells : Nat := 4096
def atlasClasses : Nat := 3
def groupOrder : Nat := 2
def separatorCapacity : Nat := 6
def grammarCandidates : Nat := 7200
def searchBudget : Nat := 64

def frozenSeedWords : Array Nat :=
  #[0, 0, 1010580540, 4042322160,
    2863311530, 2863311530, 2526451350, 1515870810]

def bit (value coordinate : Nat) : Nat := (value / 2 ^ coordinate) % 2

def seedBit (left right : Nat) : Nat :=
  let cell := left * dimension + right
  bit (frozenSeedWords.getD (cell / 32) 0) (cell % 32)

def signedCoefficient (value : Nat) : Int :=
  if value == 0 then 1 else if value == 1 then -1 else 0

def tensorIndex (output input0 input1 : Nat) : Nat :=
  (input0 * dimension + input1) * dimension + output

def parentCoefficient (output input0 input1 : Nat) : Int :=
  if output == (input0 ^^^ input1) then signedCoefficient (seedBit input0 input1)
  else 0

def countWhere (upper : Nat) (predicate : Nat -> Bool) : Nat :=
  (List.range upper).foldl
    (fun count value => if predicate value then count + 1 else count) 0

def parentAssociatorFailure (input0 input1 input2 : Nat) : Bool :=
  let left := parentCoefficient ((input0 ^^^ input1) ^^^ input2)
      (input0 ^^^ input1) input2 *
    parentCoefficient (input0 ^^^ input1) input0 input1
  let right := parentCoefficient (input0 ^^^ (input1 ^^^ input2))
      input0 (input1 ^^^ input2) *
    parentCoefficient (input1 ^^^ input2) input1 input2
  left != right

def parentPairCount (input0 input1 : Nat) : Nat :=
  countWhere dimension fun input2 => parentAssociatorFailure input0 input1 input2

def rotl4 (value : Nat) : Nat := ((value <<< 1) &&& 15) ||| ((value >>> 3) &&& 1)

def affineAddressCoefficient (output input0 input1 : Nat) : Int :=
  if output == (rotl4 input0 ^^^ input1 ^^^ 2) then
    signedCoefficient (seedBit input0 input1)
  else 0

def coefficientLiftCoefficient (output input0 input1 : Nat) : Int :=
  if output == (input0 ^^^ input1) then
    signedCoefficient (seedBit input0 input1) * Int.ofNat (1 + parentPairCount input0 input1)
  else 0

def atlasCoefficient (classId output input0 input1 : Nat) : Int :=
  if classId == 0 then parentCoefficient output input0 input1
  else if classId == 1 then affineAddressCoefficient output input0 input1
  else if classId == 2 then coefficientLiftCoefficient output input0 input1
  else 0

def swap01 (value : Nat) : Nat :=
  if value == 0 then 1 else if value == 1 then 0 else value

def permute (action value : Nat) : Nat :=
  if action == 0 then value else if action == 1 then swap01 value else dimension

def groupMultiply (left right : Nat) : Nat := left ^^^ right

def actCoefficient
    (coefficient : Nat -> Nat -> Nat -> Int)
    (action output input0 input1 : Nat) : Int :=
  coefficient (permute action output) (permute action input0) (permute action input1)

def representativeCoefficient
    (classId action output input0 input1 : Nat) : Int :=
  actCoefficient (atlasCoefficient classId) action output input0 input1

def candidateCoefficient (output input0 input1 : Nat) : Int :=
  let parent := parentCoefficient output input0 input1
  if tensorIndex output input0 input1 == 272 then parent + 1 else parent

def decodeOutput (cell : Nat) : Nat := cell % dimension
def decodeInput1 (cell : Nat) : Nat := (cell / dimension) % dimension
def decodeInput0 (cell : Nat) : Nat := cell / (dimension * dimension)

def firstWitness
    (left right : Nat -> Nat -> Nat -> Int) : Option Nat :=
  (List.range tensorCells).find? fun cell =>
    let output := decodeOutput cell
    let input1 := decodeInput1 cell
    let input0 := decodeInput0 cell
    left output input0 input1 != right output input0 input1

inductive SearchOutcome where
  | quotientCollision
  | n2RelativeNovelty
  | searchIncomplete
deriving Repr, BEq, DecidableEq

structure SearchResult where
  outcome : SearchOutcome
  comparisonsRequired : Nat
  comparisonsCompleted : Nat
  matchedClass : Option Nat
  matchedAction : Option Nat
  separators : List Nat
  complete : Bool
deriving Repr, BEq, DecidableEq

def comparisonPairs : List (Nat × Nat) :=
  [ (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1) ]

def discoverLoop
    (candidate : Nat -> Nat -> Nat -> Int) :
    List (Nat × Nat) -> Nat -> Nat -> List Nat -> SearchResult
  | [], _, completed, separators =>
      if completed == separatorCapacity && separators.length == separatorCapacity then
        { outcome := SearchOutcome.n2RelativeNovelty
        , comparisonsRequired := separatorCapacity
        , comparisonsCompleted := completed
        , matchedClass := none
        , matchedAction := none
        , separators := separators
        , complete := true }
      else
        { outcome := SearchOutcome.searchIncomplete
        , comparisonsRequired := separatorCapacity
        , comparisonsCompleted := completed
        , matchedClass := none
        , matchedAction := none
        , separators := separators
        , complete := false }
  | _ :: _, 0, completed, separators =>
      { outcome := SearchOutcome.searchIncomplete
      , comparisonsRequired := separatorCapacity
      , comparisonsCompleted := completed
      , matchedClass := none
      , matchedAction := none
      , separators := separators
      , complete := false }
  | pair :: remaining, budget + 1, completed, separators =>
      match firstWitness candidate
          (fun output input0 input1 =>
            representativeCoefficient pair.1 pair.2 output input0 input1) with
      | none =>
          { outcome := SearchOutcome.quotientCollision
          , comparisonsRequired := separatorCapacity
          , comparisonsCompleted := completed + 1
          , matchedClass := some pair.1
          , matchedAction := some pair.2
          , separators := separators
          , complete := true }
      | some witness =>
          discoverLoop candidate remaining budget (completed + 1)
            (separators ++ [witness])

def discover (candidate : Nat -> Nat -> Nat -> Int) (budget : Nat) : SearchResult :=
  discoverLoop candidate comparisonPairs budget 0 []

structure Separator where
  classId : Nat
  action : Nat
  witness : Nat
  output : Nat
  input0 : Nat
  input1 : Nat
deriving Repr, BEq, DecidableEq

def frozenSeparators : List Separator :=
  [ { classId := 0, action := 0, witness := 272, output := 0, input0 := 1, input1 := 1 }
  , { classId := 0, action := 1, witness := 0, output := 0, input0 := 0, input1 := 0 }
  , { classId := 1, action := 0, witness := 0, output := 0, input0 := 0, input1 := 0 }
  , { classId := 1, action := 1, witness := 257, output := 1, input0 := 1, input1 := 0 }
  , { classId := 2, action := 0, witness := 272, output := 0, input0 := 1, input1 := 1 }
  , { classId := 2, action := 1, witness := 0, output := 0, input0 := 0, input1 := 0 } ]

def separatorValid (separator : Separator) : Bool :=
  separator.classId < atlasClasses &&
  separator.action < groupOrder &&
  separator.output < dimension &&
  separator.input0 < dimension &&
  separator.input1 < dimension &&
  separator.witness == tensorIndex separator.output separator.input0 separator.input1 &&
  firstWitness candidateCoefficient
      (fun output input0 input1 =>
        representativeCoefficient separator.classId separator.action
          output input0 input1) == some separator.witness

def allSeparatorsValid : Bool := frozenSeparators.all separatorValid

def separatorCoverage : Bool :=
  (List.range atlasClasses).all fun classId =>
    (List.range groupOrder).all fun action =>
      frozenSeparators.any fun separator =>
        separator.classId == classId && separator.action == action &&
          separatorValid separator

def candidateSearch : SearchResult :=
  discover candidateCoefficient separatorCapacity

def controlCoefficient (output input0 input1 : Nat) : Int :=
  actCoefficient parentCoefficient 1 output input0 input1

def controlSearch : SearchResult :=
  discover controlCoefficient separatorCapacity

def incompleteControlSearch : SearchResult :=
  discover candidateCoefficient 1

def collisionControlExact : Bool :=
  controlSearch.outcome == SearchOutcome.quotientCollision &&
    controlSearch.comparisonsRequired == 6 &&
    controlSearch.comparisonsCompleted == 2 &&
    controlSearch.matchedClass == some 0 &&
    controlSearch.matchedAction == some 1 &&
    controlSearch.separators == [0] && controlSearch.complete

def incompleteControlExact : Bool :=
  incompleteControlSearch.outcome == SearchOutcome.searchIncomplete &&
    incompleteControlSearch.comparisonsRequired == 6 &&
    incompleteControlSearch.comparisonsCompleted == 1 &&
    incompleteControlSearch.matchedClass == none &&
    incompleteControlSearch.matchedAction == none &&
    incompleteControlSearch.separators == [272] &&
    !incompleteControlSearch.complete

def permutationCertificate : Bool :=
  (List.range groupOrder).all fun action =>
    (List.range dimension).all fun value =>
      permute action value < dimension &&
        permute action (permute action value) == value

def multiplicationCertificate : Bool :=
  (List.range groupOrder).all fun left =>
    (List.range groupOrder).all fun right =>
      groupMultiply left right < groupOrder &&
        groupMultiply 0 left == left && groupMultiply left 0 == left &&
        groupMultiply left left == 0

def associativityCertificate : Bool :=
  (List.range groupOrder).all fun first =>
    (List.range groupOrder).all fun second =>
      (List.range groupOrder).all fun third =>
        groupMultiply (groupMultiply first second) third ==
          groupMultiply first (groupMultiply second third)

def actionCompositionCertificate : Bool :=
  (List.range atlasClasses).all fun classId =>
    (List.range groupOrder).all fun left =>
      (List.range groupOrder).all fun right =>
        (List.range tensorCells).all fun cell =>
          let output := decodeOutput cell
          let input1 := decodeInput1 cell
          let input0 := decodeInput0 cell
          actCoefficient
              (fun o i j => representativeCoefficient classId right o i j)
              left output input0 input1 ==
            representativeCoefficient classId (groupMultiply left right)
              output input0 input1

def candidateCommutatorFailures : Nat :=
  countWhere tensorCells fun cell =>
    let output := decodeOutput cell
    let input1 := decodeInput1 cell
    let input0 := decodeInput0 cell
    candidateCoefficient output input0 input1 !=
      candidateCoefficient output input1 input0

def associatorLeft (output input0 input1 input2 : Nat) : Int :=
  (List.range dimension).foldl
    (fun total middle => total +
      candidateCoefficient middle input0 input1 *
        candidateCoefficient output middle input2) 0

def associatorRight (output input0 input1 input2 : Nat) : Int :=
  (List.range dimension).foldl
    (fun total middle => total +
      candidateCoefficient middle input1 input2 *
        candidateCoefficient output input0 middle) 0

def candidateAssociatorFailures : Nat :=
  countWhere (dimension ^ 4) fun cell =>
    let output := cell % dimension
    let input2 := (cell / dimension) % dimension
    let input1 := (cell / (dimension ^ 2)) % dimension
    let input0 := cell / (dimension ^ 3)
    associatorLeft output input0 input1 input2 !=
      associatorRight output input0 input1 input2

structure ClaimBoundary where
  n3Novelty : Bool
  n4Novelty : Bool
  algorithmicNovelty : Bool
  materialNovelty : Bool
  historicalNovelty : Bool
  priorityClaim : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def claimBoundary : ClaimBoundary :=
  { n3Novelty := false
  , n4Novelty := false
  , algorithmicNovelty := false
  , materialNovelty := false
  , historicalNovelty := false
  , priorityClaim := false
  , claimReady := false }

structure FormalParitySummary where
  seedWeight : Nat
  parentAssociatorFailures : Nat
  parentCommutatorFailures : Nat
  permutationChecks : Nat
  multiplicationChecks : Nat
  lawChecks : Nat
  actionChecks : Nat
  groupFailures : Nat
  candidateId : Nat
  mutationTensorIndex : Nat
  mutationDelta : Int
  candidateOutcome : SearchOutcome
  separatorCount : Nat
  separatorWitnesses : List Nat
  separatorsValid : Bool
  separatorCoverage : Bool
  collisionControlExact : Bool
  incompleteControlOutcome : SearchOutcome
  incompleteControlExact : Bool
  declaredGrammarBudgetExhaustive : Bool
  searchIncompletePromoted : Bool
  commutatorChecks : Nat
  commutatorFailures : Nat
  associatorChecks : Nat
  associatorFailures : Nat
  claimBoundary : ClaimBoundary
deriving Repr, BEq, DecidableEq

def seedWeight : Nat :=
  countWhere (dimension * dimension) fun cell => seedBit (cell / dimension) (cell % dimension) == 1

def parentAssociatorFailures : Nat :=
  countWhere (dimension ^ 3) fun cell =>
    let input2 := cell % dimension
    let input1 := (cell / dimension) % dimension
    let input0 := cell / (dimension ^ 2)
    parentAssociatorFailure input0 input1 input2

def parentCommutatorFailures : Nat :=
  countWhere (dimension ^ 2) fun cell =>
    let input1 := cell % dimension
    let input0 := cell / dimension
    parentCoefficient (input0 ^^^ input1) input0 input1 !=
      parentCoefficient (input0 ^^^ input1) input1 input0

def formalParitySummary : FormalParitySummary :=
  { seedWeight := seedWeight
  , parentAssociatorFailures := parentAssociatorFailures
  , parentCommutatorFailures := parentCommutatorFailures
  , permutationChecks := groupOrder * dimension
  , multiplicationChecks := groupOrder * groupOrder
  -- Four pairs each check two identities, two elements check inverses, and
  -- eight triples check associativity: 4*2 + 2 + 8 = 18.
  , lawChecks := groupOrder * groupOrder * 2 + groupOrder + groupOrder ^ 3
  , actionChecks := atlasClasses * groupOrder * groupOrder * tensorCells
  , groupFailures :=
      if permutationCertificate && multiplicationCertificate &&
          associativityCertificate && actionCompositionCertificate then 0 else 1
  , candidateId := 0
  , mutationTensorIndex := tensorIndex 0 1 1
  , mutationDelta := 1
  , candidateOutcome := candidateSearch.outcome
  , separatorCount := candidateSearch.separators.length
  , separatorWitnesses := candidateSearch.separators
  , separatorsValid := allSeparatorsValid
  , separatorCoverage := separatorCoverage
  , collisionControlExact := collisionControlExact
  , incompleteControlOutcome := incompleteControlSearch.outcome
  , incompleteControlExact := incompleteControlExact
  , declaredGrammarBudgetExhaustive := grammarCandidates <= searchBudget
  , searchIncompletePromoted := false
  , commutatorChecks := tensorCells
  , commutatorFailures := candidateCommutatorFailures
  , associatorChecks := dimension ^ 4
  , associatorFailures := candidateAssociatorFailures
  , claimBoundary := claimBoundary }

def frozenFormalParitySummary : FormalParitySummary :=
  { seedWeight := 96
  , parentAssociatorFailures := 768
  , parentCommutatorFailures := 112
  , permutationChecks := 32
  , multiplicationChecks := 4
  , lawChecks := 18
  , actionChecks := 49152
  , groupFailures := 0
  , candidateId := 0
  , mutationTensorIndex := 272
  , mutationDelta := 1
  , candidateOutcome := SearchOutcome.n2RelativeNovelty
  , separatorCount := 6
  , separatorWitnesses := [272, 0, 0, 257, 272, 0]
  , separatorsValid := true
  , separatorCoverage := true
  , collisionControlExact := true
  , incompleteControlOutcome := SearchOutcome.searchIncomplete
  , incompleteControlExact := true
  , declaredGrammarBudgetExhaustive := false
  , searchIncompletePromoted := false
  , commutatorChecks := 4096
  , commutatorFailures := 112
  , associatorChecks := 65536
  , associatorFailures := 824
  , claimBoundary :=
      { n3Novelty := false
      , n4Novelty := false
      , algorithmicNovelty := false
      , materialNovelty := false
      , historicalNovelty := false
      , priorityClaim := false
      , claimReady := false } }

theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  native_decide

theorem c2_diag_group_and_full_atlas_action_exact :
    formalParitySummary.permutationChecks = 32 &&
      formalParitySummary.multiplicationChecks = 4 &&
      formalParitySummary.lawChecks = 18 &&
      formalParitySummary.actionChecks = 49152 &&
      formalParitySummary.groupFailures = 0 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem candidate_zero_has_all_six_frozen_separators :
    formalParitySummary.candidateId = 0 &&
      formalParitySummary.mutationTensorIndex = 272 &&
      formalParitySummary.separatorCount = 6 &&
      formalParitySummary.separatorWitnesses = [272, 0, 0, 257, 272, 0] &&
      formalParitySummary.separatorsValid &&
      formalParitySummary.separatorCoverage &&
      formalParitySummary.candidateOutcome = SearchOutcome.n2RelativeNovelty := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem collision_and_incomplete_controls_are_not_n2 :
    formalParitySummary.collisionControlExact &&
      formalParitySummary.incompleteControlOutcome = SearchOutcome.searchIncomplete &&
      formalParitySummary.incompleteControlExact &&
      !formalParitySummary.declaredGrammarBudgetExhaustive &&
      !formalParitySummary.searchIncompletePromoted := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem candidate_law_spectrum_matches_frozen_sounio :
    formalParitySummary.commutatorChecks = 4096 &&
      formalParitySummary.commutatorFailures = 112 &&
      formalParitySummary.associatorChecks = 65536 &&
      formalParitySummary.associatorFailures = 824 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem n2_does_not_promote_stronger_or_public_claims :
    !formalParitySummary.claimBoundary.n3Novelty &&
      !formalParitySummary.claimBoundary.n4Novelty &&
      !formalParitySummary.claimBoundary.algorithmicNovelty &&
      !formalParitySummary.claimBoundary.materialNovelty &&
      !formalParitySummary.claimBoundary.historicalNovelty &&
      !formalParitySummary.claimBoundary.priorityClaim &&
      !formalParitySummary.claimBoundary.claimReady := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

end SounioPireusOperatorDiscoveryEngine
