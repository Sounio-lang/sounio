/-
  FORMAL_PARITY for the Sounio-owned Pireus Quotient Novelty Forge v5.

  Sounio remains SEMANTIC_AUTHORITY. This file is admitted only after the
  frozen Sounio source and semantics hashes below reached PARITY_OPEN. It
  independently recomputes the finite gauge quotient, the parent stabilizer
  in GL(4,2) x C2, and the Q0/Q1/Q2 atlas. It creates no expected result,
  selects no child, and makes no global novelty claim.
-/
import SounioCDCocycle

namespace SounioPireusQuotientNoveltyForge

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def frozenSounioSourceSha256 : String :=
  "791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667"

def frozenSounioSemanticsSha256 : String :=
  "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"

def dimension : Nat := 16
def childCount : Nat := 48
def identityMatrix : Nat := 33825
def selectedParentMatrix : Nat := 1128

abbrev Table := Array Nat

structure Descriptor where
  kind : Nat
  r : Nat
  s : Nat
  t : Nat
deriving Repr, BEq, DecidableEq

structure Action where
  matrix : Nat
  swap : Nat
deriving Repr, BEq, DecidableEq

structure Normalization where
  gaugeWord : Nat
  canonical : Table
deriving Repr, BEq, DecidableEq

def bit (value coordinate : Nat) : Nat := (value / 2 ^ coordinate) % 2

def parity4 (value : Nat) : Nat :=
  (List.range 4).foldl (fun parity coordinate => parity ^^^ bit value coordinate) 0

def matrixRow (code row : Nat) : Nat := (code / 2 ^ (row * 4)) % 16

def matrixApply (code vector : Nat) : Nat :=
  (List.range 4).foldl
    (fun result row =>
      if parity4 (matrixRow code row &&& vector) == 1 then
        result + 2 ^ row
      else result)
    0

def matrixInvertible (code : Nat) : Bool :=
  ((List.range dimension).map (matrixApply code)).eraseDups.length == dimension

def matrixCompose (left right : Nat) : Nat :=
  (List.range 4).foldl
    (fun code row =>
      let selector := matrixRow left row
      let composedRow := (List.range 4).foldl
        (fun value sourceRow =>
          if bit selector sourceRow == 1 then value ^^^ matrixRow right sourceRow
          else value)
        0
      code + composedRow * 2 ^ (row * 4))
    0

def isBasis (value : Nat) : Bool :=
  value == 1 || value == 2 || value == 4 || value == 8

def highestBasis (value : Nat) : Nat :=
  if value >= 8 then 8
  else if value >= 4 then 4
  else if value >= 2 then 2
  else 1

def gaugeVectors : List Nat := [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

def gaugeIndex (vector : Nat) : Nat :=
  (gaugeVectors.takeWhile fun candidate => candidate != vector).length

def gaugeValue (word vector : Nat) : Nat :=
  if vector == 0 || isBasis vector then 0 else bit word (gaugeIndex vector)

def coboundary (word left right : Nat) : Nat :=
  gaugeValue word left ^^^ gaugeValue word right ^^^ gaugeValue word (left ^^^ right)

def tableOf (cell : Nat -> Nat -> Nat) : Table :=
  ((List.range dimension).flatMap fun left =>
    (List.range dimension).map fun right => cell left right).toArray

def normalizerValues (table : Table) : Array Nat := Id.run do
  let mut values := Array.replicate dimension 0
  for vector in List.range dimension do
    if vector != 0 && !isBasis vector then
      let edge := highestBasis vector
      let parent := vector ^^^ edge
      values := values.set! vector
        ((table.getD (parent * dimension + edge) 0) ^^^ (values.getD parent 0))
  return values

def normalize (table : Table) : Normalization :=
  let values := normalizerValues table
  let gaugeWord := gaugeVectors.foldl
    (fun word vector =>
      if values.getD vector 0 == 1 then word + 2 ^ gaugeIndex vector else word)
    0
  let canonical := tableOf fun left right =>
    (table.getD (left * dimension + right) 0) ^^^
      (values.getD left 0) ^^^ (values.getD right 0) ^^^
      (values.getD (left ^^^ right) 0)
  { gaugeWord := gaugeWord, canonical := canonical }

def gaugeTable (word : Nat) : Table := tableOf (coboundary word)

def pullbackGaugeWord (word matrix : Nat) : Nat :=
  gaugeVectors.foldl
    (fun pulled vector =>
      let value0 := gaugeValue word (matrixApply matrix vector)
      let value := (List.range 4).foldl
        (fun current coordinate =>
          if bit vector coordinate == 1 then
            current ^^^ gaugeValue word (matrixApply matrix (2 ^ coordinate))
          else current)
        value0
      if value == 1 then pulled + 2 ^ gaugeIndex vector else pulled)
    0

def bilinearBit (matrix left right : Nat) : Nat :=
  (List.range 4).foldl
    (fun value row =>
      if bit left row == 1 then
        (List.range 4).foldl
          (fun inner column =>
            if bit right column == 1 && bit matrix (4 * row + column) == 1 then
              inner ^^^ 1
            else inner)
          value
      else value)
    0

def parentBit (left right : Nat) : Nat :=
  let base := SounioCDCocycle.cdSigma left right 4
  (if base < 0 then 1 else 0) ^^^ bilinearBit selectedParentMatrix left right

def parentTable : Table := tableOf parentBit

def l2r1Descriptors : List Descriptor :=
  (List.range 4).flatMap fun r =>
    ((List.range 4).filter fun s => r < s).flatMap fun s =>
      (List.range 4).map fun t => { kind := 1, r := r, s := s, t := t }

def l1r2Descriptors : List Descriptor :=
  (List.range 4).flatMap fun r =>
    (List.range 4).flatMap fun s =>
      ((List.range 4).filter fun t => s < t).map fun t =>
        { kind := 2, r := r, s := s, t := t }

def descriptors : List Descriptor := l2r1Descriptors ++ l1r2Descriptors

def mutationBit (descriptor : Descriptor) (left right : Nat) : Nat :=
  if descriptor.kind == 1 then
    bit left descriptor.r * bit left descriptor.s * bit right descriptor.t
  else if descriptor.kind == 2 then
    bit left descriptor.r * bit right descriptor.s * bit right descriptor.t
  else 0

def childTable (descriptor : Descriptor) : Table :=
  tableOf fun left right => parentBit left right ^^^ mutationBit descriptor left right

def childTables : Array Table := (descriptors.map childTable).toArray

def transformTable (table : Table) (action : Action) : Table :=
  tableOf fun left right =>
    let mappedLeft := matrixApply action.matrix left
    let mappedRight := matrixApply action.matrix right
    if action.swap == 1 then
      table.getD (mappedRight * dimension + mappedLeft) 0
    else
      table.getD (mappedLeft * dimension + mappedRight) 0

def tableXor (left right : Table) : Table :=
  ((List.range (dimension * dimension)).map fun cell =>
    left.getD cell 0 ^^^ right.getD cell 0).toArray

def tableZero (table : Table) : Bool := table.toList.all fun value => value == 0

def parentActionAdmitted (parent : Table) (action : Action) : Bool :=
  matrixInvertible action.matrix &&
    tableZero (normalize (tableXor (transformTable parent action) parent)).canonical

def invertibleMatrices : List Nat :=
  (List.range 65536).filter matrixInvertible

def candidateActions (matrices : List Nat) : List Action :=
  matrices.flatMap fun matrix => [{ matrix := matrix, swap := 0 }, { matrix := matrix, swap := 1 }]

def admittedActionsFor (parent : Table) (matrices : List Nat) : List Action :=
  (candidateActions matrices).filter (parentActionAdmitted parent)

def composeAction (left right : Action) : Action :=
  { matrix := matrixCompose left.matrix right.matrix, swap := left.swap ^^^ right.swap }

def actionIdentity (action : Action) : Bool :=
  action.matrix == identityMatrix && action.swap == 0

def actionInverseExists (actions : List Action) (action : Action) : Bool :=
  actions.any fun inverse =>
    actionIdentity (composeAction action inverse) &&
      actionIdentity (composeAction inverse action)

def actionClosure (actions : List Action) : Bool :=
  actions.all fun left => actions.all fun right => actions.contains (composeAction left right)

def actionApplicationComposition (actions : List Action) : Bool :=
  actions.all fun left => actions.all fun right =>
    (List.range dimension).all fun vector =>
      matrixApply (matrixCompose left.matrix right.matrix) vector ==
        matrixApply left.matrix (matrixApply right.matrix vector)

def gaugeEquivariance (actions : List Action) : Bool :=
  actions.all fun action => (List.range gaugeVectors.length).all fun basis =>
    let word := 2 ^ basis
    let pulled := pullbackGaugeWord word action.matrix
    (List.range dimension).all fun left => (List.range dimension).all fun right =>
      let mappedLeft := matrixApply action.matrix left
      let mappedRight := matrixApply action.matrix right
      let sourceLeft := if action.swap == 1 then mappedRight else mappedLeft
      let sourceRight := if action.swap == 1 then mappedLeft else mappedRight
      coboundary word sourceLeft sourceRight == coboundary pulled left right

def relationArray (relation : Nat -> Nat -> Bool) : Array Bool :=
  ((List.range childCount).flatMap fun left =>
    (List.range childCount).map fun right => relation left right).toArray

def relationGet (relation : Array Bool) (left right : Nat) : Bool :=
  relation.getD (left * childCount + right) false

def canonicalChildren (tables : Array Table) : Array Normalization :=
  ((List.range childCount).map fun child => normalize (tables.getD child #[])).toArray

def transformedChildren (tables : Array Table) (actions : List Action) : Array Normalization :=
  ((List.range childCount).flatMap fun child =>
    actions.map fun action => normalize (transformTable (tables.getD child #[]) action)).toArray

def q0RelationFor (tables : Array Table) : Array Bool :=
  relationArray fun left right => tables.getD left #[] == tables.getD right #[]

def q1RelationFor (normalized : Array Normalization) : Array Bool :=
  relationArray fun left right =>
    (normalized.getD left { gaugeWord := 0, canonical := #[] }).canonical ==
      (normalized.getD right { gaugeWord := 0, canonical := #[] }).canonical

def q2RelationFor
    (normalized : Array Normalization)
    (transformed : Array Normalization)
    (actionCount : Nat) : Array Bool :=
  relationArray fun child target =>
    (List.range actionCount).any fun action =>
      (transformed.getD (child * actionCount + action)
        { gaugeWord := 0, canonical := #[] }).canonical ==
      (normalized.getD target { gaugeWord := 0, canonical := #[] }).canonical

def relationEquivalence (relation : Array Bool) : Bool :=
  let reflexive := (List.range childCount).all fun child => relationGet relation child child
  let symmetric := (List.range childCount).all fun left =>
    (List.range childCount).all fun right =>
      relationGet relation left right == relationGet relation right left
  let transitive := (List.range childCount).all fun left =>
    (List.range childCount).all fun middle =>
      (List.range childCount).all fun right =>
        !(relationGet relation left middle && relationGet relation middle right) ||
          relationGet relation left right
  reflexive && symmetric && transitive

def refines (finer coarser : Array Bool) : Bool :=
  (List.range childCount).all fun left => (List.range childCount).all fun right =>
    !relationGet finer left right || relationGet coarser left right

def representative (relation : Array Bool) (child : Nat) : Nat :=
  ((List.range childCount).find? fun candidate => relationGet relation child candidate).getD child

def representatives (relation : Array Bool) : List Nat :=
  (List.range childCount).filter fun child => representative relation child == child

def classSizes (relation : Array Bool) : List Nat :=
  (representatives relation).map fun representative =>
    ((List.range childCount).filter fun child => relationGet relation child representative).length

def minimumClassSize (relation : Array Bool) : Nat :=
  (classSizes relation).foldl Nat.min childCount

def maximumClassSize (relation : Array Bool) : Nat :=
  (classSizes relation).foldl Nat.max 0

def classSizeSum (relation : Array Bool) : Nat := (classSizes relation).sum

def relationPartitionSound (relation : Array Bool) : Bool :=
  let reps := representatives relation
  (List.range childCount).all (fun child =>
      let rep := representative relation child
      rep < childCount && relationGet relation child rep && reps.contains rep) &&
    reps.all (fun left => reps.all fun right => left == right || !relationGet relation left right) &&
    classSizeSum relation == childCount

def witnessAction?
    (normalized : Array Normalization)
    (transformed : Array Normalization)
    (actions : List Action)
    (child target : Nat) : Option (Nat × Action) :=
  ((List.range actions.length).find? fun action =>
    (transformed.getD (child * actions.length + action)
      { gaugeWord := 0, canonical := #[] }).canonical ==
    (normalized.getD target { gaugeWord := 0, canonical := #[] }).canonical).map
      fun index => (index, actions.getD index { matrix := 0, swap := 0 })

def tableWitnessValid
    (source target : Table) (action : Action) (gauge : Nat) : Bool :=
  let transformed := transformTable source action
  (List.range (dimension * dimension)).all fun cell =>
    let left := cell / dimension
    let right := cell % dimension
    (transformed.getD cell 0 ^^^ target.getD cell 0) == coboundary gauge left right

def profileWitnessesSound
    (tables : Array Table)
    (normalized : Array Normalization)
    (transformed : Array Normalization)
    (actions : List Action)
    (profile : Nat)
    (relation : Array Bool) : Bool :=
  (List.range childCount).all fun child =>
    let target := representative relation child
    if profile == 0 then
      tableWitnessValid (tables.getD child #[]) (tables.getD target #[])
        { matrix := identityMatrix, swap := 0 } 0
    else if profile == 1 then
      let sourceGauge := (normalized.getD child { gaugeWord := 0, canonical := #[] }).gaugeWord
      let targetGauge := (normalized.getD target { gaugeWord := 0, canonical := #[] }).gaugeWord
      tableWitnessValid (tables.getD child #[]) (tables.getD target #[])
        { matrix := identityMatrix, swap := 0 } (sourceGauge ^^^ targetGauge)
    else
      match witnessAction? normalized transformed actions child target with
      | none => false
      | some (index, action) =>
          let sourceGauge := (transformed.getD (child * actions.length + index)
            { gaugeWord := 0, canonical := #[] }).gaugeWord
          let targetGauge := (normalized.getD target
            { gaugeWord := 0, canonical := #[] }).gaugeWord
          tableWitnessValid (tables.getD child #[]) (tables.getD target #[])
            action (sourceGauge ^^^ targetGauge)

def gaugeNormalizerExact : Bool :=
  gaugeVectors.length == 11 &&
    (List.range 2048).all fun word =>
      let normalized := normalize (gaugeTable word)
      normalized.gaugeWord == word && tableZero normalized.canonical

structure FormalParitySummary where
  sourceSha256 : String
  semanticsSha256 : String
  descriptors : Nat
  gaugeBits : Nat
  gaugeWords : Nat
  gaugeNormalizerExact : Bool
  matrixEncodings : Nat
  invertibleMatrices : Nat
  actionsConsidered : Nat
  admittedActions : Nat
  admittedNoSwap : Nat
  admittedSwap : Nat
  identityPresent : Bool
  inverseLaw : Bool
  closureLaw : Bool
  actionComposition : Bool
  gaugeEquivariance : Bool
  q0Classes : Nat
  q1Classes : Nat
  q2Classes : Nat
  q2MinimumClassSize : Nat
  q2MaximumClassSize : Nat
  q2ClassSizeSum : Nat
  equivalenceLaws : Bool
  refinementLaws : Bool
  partitionsSound : Bool
  witnessesSound : Bool
  selectedChild : Int
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def formalParitySummary : FormalParitySummary :=
  let parent := parentTable
  let matrices := invertibleMatrices
  let actions := admittedActionsFor parent matrices
  let tables := childTables
  let normalized := canonicalChildren tables
  let transformed := transformedChildren tables actions
  let q0 := q0RelationFor tables
  let q1 := q1RelationFor normalized
  let q2 := q2RelationFor normalized transformed actions.length
  { sourceSha256 := frozenSounioSourceSha256
    semanticsSha256 := frozenSounioSemanticsSha256
    descriptors := descriptors.length
    gaugeBits := gaugeVectors.length
    gaugeWords := 2048
    gaugeNormalizerExact := gaugeNormalizerExact
    matrixEncodings := 65536
    invertibleMatrices := matrices.length
    actionsConsidered := matrices.length * 2
    admittedActions := actions.length
    admittedNoSwap := (actions.filter fun action => action.swap == 0).length
    admittedSwap := (actions.filter fun action => action.swap == 1).length
    identityPresent := actions.contains { matrix := identityMatrix, swap := 0 }
    inverseLaw := actions.all (actionInverseExists actions)
    closureLaw := actionClosure actions
    actionComposition := actionApplicationComposition actions
    gaugeEquivariance := gaugeEquivariance actions
    q0Classes := (representatives q0).length
    q1Classes := (representatives q1).length
    q2Classes := (representatives q2).length
    q2MinimumClassSize := minimumClassSize q2
    q2MaximumClassSize := maximumClassSize q2
    q2ClassSizeSum := classSizeSum q2
    equivalenceLaws := relationEquivalence q0 && relationEquivalence q1 && relationEquivalence q2
    refinementLaws := refines q0 q1 && refines q1 q2
    partitionsSound := relationPartitionSound q0 && relationPartitionSound q1 &&
      relationPartitionSound q2
    witnessesSound := profileWitnessesSound tables normalized transformed actions 0 q0 &&
      profileWitnessesSound tables normalized transformed actions 1 q1 &&
      profileWitnessesSound tables normalized transformed actions 2 q2
    selectedChild := -1
    claimReady := false }

def frozenFormalParitySummary : FormalParitySummary :=
  { sourceSha256 :=
      "791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667"
    semanticsSha256 :=
      "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"
    descriptors := 48
    gaugeBits := 11
    gaugeWords := 2048
    gaugeNormalizerExact := true
    matrixEncodings := 65536
    invertibleMatrices := 20160
    actionsConsidered := 40320
    admittedActions := 12
    admittedNoSwap := 6
    admittedSwap := 6
    identityPresent := true
    inverseLaw := true
    closureLaw := true
    actionComposition := true
    gaugeEquivariance := true
    q0Classes := 48
    q1Classes := 48
    q2Classes := 14
    q2MinimumClassSize := 2
    q2MaximumClassSize := 4
    q2ClassSizeSum := 48
    equivalenceLaws := true
    refinementLaws := true
    partitionsSound := true
    witnessesSound := true
    selectedChild := -1
    claimReady := false }

/-- One kernel-checked certificate computes all four declared parity obligations. -/
theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  native_decide

theorem gauge_kernel_dimension_and_normalizer_uniqueness :
    formalParitySummary.gaugeBits = 11 &&
      formalParitySummary.gaugeWords = 2048 &&
      formalParitySummary.gaugeNormalizerExact := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem parent_stabilizer_in_GL4_x_C2_group_law_inverse_and_action_equivariance :
    formalParitySummary.invertibleMatrices = 20160 &&
      formalParitySummary.admittedActions = 12 &&
      formalParitySummary.inverseLaw && formalParitySummary.closureLaw &&
      formalParitySummary.actionComposition && formalParitySummary.gaugeEquivariance := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem Q0_Q1_Q2_equivalence_and_refinement :
    formalParitySummary.q0Classes = 48 &&
      formalParitySummary.q1Classes = 48 &&
      formalParitySummary.q2Classes = 14 &&
      formalParitySummary.equivalenceLaws && formalParitySummary.refinementLaws := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem canonical_partition_and_witness_soundness :
    formalParitySummary.partitionsSound && formalParitySummary.witnessesSound &&
      formalParitySummary.q2MinimumClassSize = 2 &&
      formalParitySummary.q2MaximumClassSize = 4 &&
      formalParitySummary.q2ClassSizeSum = 48 &&
      formalParitySummary.selectedChild = -1 && !formalParitySummary.claimReady := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

end SounioPireusQuotientNoveltyForge
