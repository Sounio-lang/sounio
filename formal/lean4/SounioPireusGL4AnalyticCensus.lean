/-
  Analytic FORMAL_PARITY census for ordered bases of the four-dimensional
  XOR vector space.

  This proof does not scan the 65536 matrix codes and does not consume the
  imported native census. It constructs the basis list by removing
  spans of sizes 1, 2, 4, and 8, then proves the fiber product
  15 * 14 * 12 * 8 = 20160 by kernel-checked list algebra.

  The bijection between this analytic ordered-basis list and the frozen
  matrix-code subtype remains a separate obligation. Therefore the frozen
  scan count, the 40320 action count, Target-03, and claim readiness remain
  open in this module.
-/
import SounioPireusGL4ActionEnumeration

namespace SounioPireusGL4AnalyticCensus

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

open SounioPireusLinearSwapGaugeDescent

def eraseMany {Alpha : Type} [BEq Alpha]
    (pool forbidden : List Alpha) : List Alpha :=
  forbidden.foldl (fun remaining value => remaining.erase value) pool

theorem eraseMany_nil {Alpha : Type} [BEq Alpha] (pool : List Alpha) :
    eraseMany pool [] = pool := by
  rfl

theorem eraseMany_cons {Alpha : Type} [BEq Alpha]
    (pool : List Alpha) (head : Alpha) (tail : List Alpha) :
    eraseMany pool (head :: tail) =
      eraseMany (pool.erase head) tail := by
  rfl

theorem eraseMany_subset {Alpha : Type} [BEq Alpha]
    (pool forbidden : List Alpha) :
    eraseMany pool forbidden ⊆ pool := by
  induction forbidden generalizing pool with
  | nil => simp [eraseMany]
  | cons head tail ih =>
      rw [eraseMany_cons]
      exact (ih (pool.erase head)).trans List.erase_subset

theorem nodup_not_mem_erase_self {Alpha : Type} [BEq Alpha] [LawfulBEq Alpha]
    {value : Alpha} {values : List Alpha} (nodup : values.Nodup) :
    value ∉ values.erase value := by
  induction values with
  | nil => simp
  | cons head tail ih =>
      have facts := List.nodup_cons.mp nodup
      by_cases equal : head = value
      · subst head
        simpa using facts.1
      · have notBeq : ¬(head == value) = true := by
          simpa using equal
        rw [List.erase_cons_tail notBeq]
        simp only [List.mem_cons, not_or]
        exact ⟨fun reversed => equal reversed.symm, ih facts.2⟩

theorem mem_eraseMany_not_mem_forbidden
    {Alpha : Type} [BEq Alpha] [LawfulBEq Alpha]
    {pool forbidden : List Alpha} {value : Alpha}
    (poolNodup : pool.Nodup)
    (membership : value ∈ eraseMany pool forbidden) :
    value ∉ forbidden := by
  induction forbidden generalizing pool with
  | nil => simp
  | cons head tail ih =>
      rw [eraseMany_cons] at membership
      have inErased : value ∈ pool.erase head :=
        eraseMany_subset (pool.erase head) tail membership
      have notHead : value ≠ head := by
        intro equal
        subst value
        exact (nodup_not_mem_erase_self poolNodup) inErased
      have notTail := ih (poolNodup.erase head) membership
      simpa [notHead] using notTail

theorem eraseMany_length_eq_sub
    {Alpha : Type} [BEq Alpha] [LawfulBEq Alpha]
    {pool forbidden : List Alpha}
    (poolNodup : pool.Nodup)
    (forbiddenNodup : forbidden.Nodup)
    (subset : forbidden ⊆ pool) :
    (eraseMany pool forbidden).length =
      pool.length - forbidden.length := by
  induction forbidden generalizing pool with
  | nil => simp [eraseMany]
  | cons head tail ih =>
      have facts := List.nodup_cons.mp forbiddenNodup
      have headMem : head ∈ pool := subset List.mem_cons_self
      have tailSubset : tail ⊆ pool.erase head := by
        intro value membership
        have valueMem : value ∈ pool :=
          subset (List.mem_cons_of_mem head membership)
        have valueNe : value ≠ head := by
          intro equal
          subst value
          exact facts.1 membership
        exact (List.mem_erase_of_ne valueNe).2 valueMem
      rw [eraseMany_cons]
      rw [ih (poolNodup.erase head) facts.2 tailSubset]
      rw [List.length_erase_of_mem headMem]
      have lengthBound := forbiddenNodup.length_le_of_subset subset
      simp only [List.length_cons] at lengthBound ⊢
      omega

@[simp] theorem lane_xor_zero (lane : Lane) : lane ^^^ 0 = lane := by
  apply Fin.ext
  simp [lane_xor_val]

@[simp] theorem lane_zero_xor (lane : Lane) : 0 ^^^ lane = lane := by
  apply Fin.ext
  simp [lane_xor_val]

@[simp] theorem lane_xor_self (lane : Lane) : lane ^^^ lane = 0 := by
  apply Fin.ext
  simp [lane_xor_val]

theorem lane_xor_assoc (first second third : Lane) :
    (first ^^^ second) ^^^ third = first ^^^ (second ^^^ third) := by
  apply Fin.ext
  simp only [lane_xor_val]
  exact Nat.xor_assoc first.val second.val third.val

theorem lane_xor_right_injective (right : Lane) :
    Function.Injective (fun left : Lane => left ^^^ right) := by
  intro first second equal
  have transported := congrArg (fun lane : Lane => lane ^^^ right) equal
  simpa [lane_xor_assoc] using transported

theorem lane_xor_swap_middle (first second third : Lane) :
    (first ^^^ second) ^^^ third = (first ^^^ third) ^^^ second := by
  calc
    (first ^^^ second) ^^^ third =
        first ^^^ (second ^^^ third) := lane_xor_assoc first second third
    _ = first ^^^ (third ^^^ second) := by
      rw [lane_xor_comm second third]
    _ = (first ^^^ third) ^^^ second :=
      (lane_xor_assoc first third second).symm

theorem lane_xor_cancel_coset (first second outside : Lane) :
    (first ^^^ outside) ^^^ (second ^^^ outside) = first ^^^ second := by
  calc
    (first ^^^ outside) ^^^ (second ^^^ outside) =
        ((first ^^^ outside) ^^^ second) ^^^ outside :=
      (lane_xor_assoc (first ^^^ outside) second outside).symm
    _ = ((first ^^^ second) ^^^ outside) ^^^ outside := by
      rw [lane_xor_swap_middle first outside second]
    _ = first ^^^ second := by
      rw [lane_xor_assoc, lane_xor_self, lane_xor_zero]

theorem nodup_map_of_injective
    {Alpha Beta : Type} {function : Alpha -> Beta} {values : List Alpha}
    (injective : Function.Injective function)
    (nodup : values.Nodup) :
    (values.map function).Nodup := by
  induction values with
  | nil => simp
  | cons head tail ih =>
      have facts := List.nodup_cons.mp nodup
      rw [List.map_cons, List.nodup_cons]
      constructor
      · intro membership
        have witness := List.mem_map.mp membership
        obtain ⟨value, valueMem, equal⟩ := witness
        exact facts.1 (injective equal ▸ valueMem)
      · exact ih facts.2

def spanZero : List Lane := [0]

def spanOne (first : Lane) : List Lane := [0, first]

def spanTwo (first second : Lane) : List Lane :=
  spanOne first ++ (spanOne first).map (fun lane => lane ^^^ second)

def spanThree (first second third : Lane) : List Lane :=
  spanTwo first second ++
    (spanTwo first second).map (fun lane => lane ^^^ third)

theorem spanZero_nodup : spanZero.Nodup := by
  simp [spanZero]

theorem spanOne_closed (first left right : Lane)
    (leftMem : left ∈ spanOne first)
    (rightMem : right ∈ spanOne first) :
    left ^^^ right ∈ spanOne first := by
  simp only [spanOne, List.mem_cons, List.not_mem_nil, or_false] at leftMem rightMem ⊢
  rcases leftMem with rfl | rfl <;>
    rcases rightMem with rfl | rfl <;>
    simp

theorem nodup_append_xor_coset
    (span : List Lane) (outside : Lane)
    (spanNodup : span.Nodup)
    (spanClosed : ∀ left ∈ span, ∀ right ∈ span,
      left ^^^ right ∈ span)
    (outsideNotMem : outside ∉ span) :
    (span ++ span.map (fun lane => lane ^^^ outside)).Nodup := by
  rw [List.nodup_append]
  refine ⟨spanNodup,
    nodup_map_of_injective (lane_xor_right_injective outside) spanNodup, ?_⟩
  intro left leftMem mapped mappedMem
  obtain ⟨right, rightMem, rightMap⟩ := List.mem_map.mp mappedMem
  subst mapped
  intro equal
  have transported := congrArg (fun lane : Lane => right ^^^ lane) equal
  have reachesOutside : right ^^^ left = outside := by
    calc
      right ^^^ left = right ^^^ (right ^^^ outside) := transported
      _ = (right ^^^ right) ^^^ outside :=
        (lane_xor_assoc right right outside).symm
      _ = outside := by simp
  have closed := spanClosed right rightMem left leftMem
  exact outsideNotMem (reachesOutside ▸ closed)

theorem spanOne_nodup_of_not_zero {first : Lane} (notZero : first ≠ 0) :
    (spanOne first).Nodup := by
  simp [spanOne, Ne.symm notZero]

theorem spanTwo_nodup_of_outside
    {first second : Lane}
    (spanOneNodup : (spanOne first).Nodup)
    (secondOutside : second ∉ spanOne first) :
    (spanTwo first second).Nodup := by
  exact nodup_append_xor_coset (spanOne first) second spanOneNodup
    (fun left leftMem right rightMem =>
      spanOne_closed first left right leftMem rightMem)
    secondOutside

theorem spanTwo_closed (first second left right : Lane)
    (leftMem : left ∈ spanTwo first second)
    (rightMem : right ∈ spanTwo first second) :
    left ^^^ right ∈ spanTwo first second := by
  rw [spanTwo] at leftMem rightMem ⊢
  rcases List.mem_append.mp leftMem with leftBaseMem | leftCosetMem
  · rcases List.mem_append.mp rightMem with rightBaseMem | rightCosetMem
    · exact List.mem_append.mpr (.inl
        (spanOne_closed first left right leftBaseMem rightBaseMem))
    · obtain ⟨rightBase, rightBaseMem, rfl⟩ :=
        List.mem_map.mp rightCosetMem
      apply List.mem_append.mpr (.inr ?_)
      apply List.mem_map.mpr
      exact ⟨left ^^^ rightBase,
        spanOne_closed first left rightBase leftBaseMem rightBaseMem,
        lane_xor_assoc left rightBase second⟩
  · rcases List.mem_append.mp rightMem with rightBaseMem | rightCosetMem
    · obtain ⟨leftBase, leftBaseMem, rfl⟩ :=
        List.mem_map.mp leftCosetMem
      apply List.mem_append.mpr (.inr ?_)
      apply List.mem_map.mpr
      exact ⟨leftBase ^^^ right,
        spanOne_closed first leftBase right leftBaseMem rightBaseMem,
        (lane_xor_swap_middle leftBase second right).symm⟩
    · obtain ⟨leftBase, leftBaseMem, rfl⟩ :=
        List.mem_map.mp leftCosetMem
      obtain ⟨rightBase, rightBaseMem, rfl⟩ :=
        List.mem_map.mp rightCosetMem
      apply List.mem_append.mpr (.inl ?_)
      rw [lane_xor_cancel_coset]
      exact spanOne_closed first leftBase rightBase leftBaseMem rightBaseMem

theorem spanThree_nodup_of_outside
    {first second third : Lane}
    (spanTwoNodup : (spanTwo first second).Nodup)
    (thirdOutside : third ∉ spanTwo first second) :
    (spanThree first second third).Nodup := by
  exact nodup_append_xor_coset (spanTwo first second) third spanTwoNodup
    (fun left leftMem right rightMem =>
      spanTwo_closed first second left right leftMem rightMem)
    thirdOutside

def laneUniverse : List Lane := List.finRange 16

def choicesOutside (span : List Lane) : List Lane :=
  eraseMany laneUniverse span

theorem choicesOutside_length (span : List Lane) (spanNodup : span.Nodup) :
    (choicesOutside span).length = 16 - span.length := by
  have subset : span ⊆ List.finRange 16 := by
    intro lane _
    exact List.mem_finRange lane
  have count := eraseMany_length_eq_sub
    (poolNodup := List.nodup_finRange 16)
    (forbiddenNodup := spanNodup)
    (subset := subset)
  simpa [choicesOutside, laneUniverse] using count

theorem mem_choicesOutside_not_mem_span
    {span : List Lane} {lane : Lane}
    (membership : lane ∈ choicesOutside span) :
    lane ∉ span := by
  exact mem_eraseMany_not_mem_forbidden (List.nodup_finRange 16) membership

def firstChoices : List Lane := choicesOutside spanZero

def secondChoices (first : Lane) : List Lane :=
  choicesOutside (spanOne first)

def thirdChoices (first second : Lane) : List Lane :=
  choicesOutside (spanTwo first second)

def fourthChoices (first second third : Lane) : List Lane :=
  choicesOutside (spanThree first second third)

theorem first_choices_length : firstChoices.length = 15 := by
  rw [firstChoices, choicesOutside_length spanZero spanZero_nodup]
  decide

theorem spanOne_nodup_of_first_choice
    {first : Lane} (firstMem : first ∈ firstChoices) :
    (spanOne first).Nodup := by
  have outside : first ∉ spanZero :=
    mem_choicesOutside_not_mem_span (span := spanZero) firstMem
  have notZero : first ≠ 0 := by
    simpa [spanZero] using outside
  exact spanOne_nodup_of_not_zero notZero

theorem second_choices_length
    {first : Lane} (firstMem : first ∈ firstChoices) :
    (secondChoices first).length = 14 := by
  rw [secondChoices, choicesOutside_length (spanOne first)
    (spanOne_nodup_of_first_choice firstMem)]
  simp [spanOne]

theorem spanTwo_nodup_of_second_choice
    {first second : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first) :
    (spanTwo first second).Nodup := by
  have outside : second ∉ spanOne first :=
    mem_choicesOutside_not_mem_span (span := spanOne first) secondMem
  exact spanTwo_nodup_of_outside
    (spanOne_nodup_of_first_choice firstMem) outside

theorem third_choices_length
    {first second : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first) :
    (thirdChoices first second).length = 12 := by
  rw [thirdChoices, choicesOutside_length (spanTwo first second)
    (spanTwo_nodup_of_second_choice firstMem secondMem)]
  simp [spanTwo, spanOne]

theorem spanThree_nodup_of_third_choice
    {first second third : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first)
    (thirdMem : third ∈ thirdChoices first second) :
    (spanThree first second third).Nodup := by
  have outside : third ∉ spanTwo first second :=
    mem_choicesOutside_not_mem_span (span := spanTwo first second) thirdMem
  exact spanThree_nodup_of_outside
    (spanTwo_nodup_of_second_choice firstMem secondMem) outside

theorem fourth_choices_length
    {first second third : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first)
    (thirdMem : third ∈ thirdChoices first second) :
    (fourthChoices first second third).length = 8 := by
  rw [fourthChoices, choicesOutside_length (spanThree first second third)
    (spanThree_nodup_of_third_choice firstMem secondMem thirdMem)]
  simp [spanThree, spanTwo, spanOne]

theorem length_flatMap_of_constant
    {Alpha Beta : Type} {values : List Alpha} {fiber : Alpha -> List Beta}
    {count : Nat}
    (constant : ∀ value ∈ values, (fiber value).length = count) :
    (values.flatMap fiber).length = values.length * count := by
  induction values with
  | nil => simp
  | cons head tail ih =>
      have headCount : (fiber head).length = count := constant head (by simp)
      have tailCount : ∀ value ∈ tail, (fiber value).length = count := by
        intro value membership
        exact constant value (List.mem_cons_of_mem head membership)
      simp [headCount, ih tailCount, Nat.add_mul, Nat.add_comm]

abbrev OrderedBasis4 := Lane × Lane × Lane × Lane

def fourthCompletions (first second third : Lane) : List OrderedBasis4 :=
  (fourthChoices first second third).map fun fourth =>
    (first, second, third, fourth)

def thirdCompletions (first second : Lane) : List OrderedBasis4 :=
  (thirdChoices first second).flatMap fun third =>
    fourthCompletions first second third

def secondCompletions (first : Lane) : List OrderedBasis4 :=
  (secondChoices first).flatMap fun second =>
    thirdCompletions first second

def analyticOrderedBases : List OrderedBasis4 :=
  firstChoices.flatMap secondCompletions

theorem fourth_completions_length
    {first second third : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first)
    (thirdMem : third ∈ thirdChoices first second) :
    (fourthCompletions first second third).length = 8 := by
  simp [fourthCompletions,
    fourth_choices_length firstMem secondMem thirdMem]

theorem third_completions_length
    {first second : Lane}
    (firstMem : first ∈ firstChoices)
    (secondMem : second ∈ secondChoices first) :
    (thirdCompletions first second).length = 96 := by
  rw [thirdCompletions, length_flatMap_of_constant (count := 8)]
  · rw [third_choices_length firstMem secondMem]
  · intro third thirdMem
    exact fourth_completions_length firstMem secondMem thirdMem

theorem second_completions_length
    {first : Lane} (firstMem : first ∈ firstChoices) :
    (secondCompletions first).length = 1344 := by
  rw [secondCompletions, length_flatMap_of_constant (count := 96)]
  · rw [second_choices_length firstMem]
  · intro second secondMem
    exact third_completions_length firstMem secondMem

theorem analytic_ordered_basis_census_is_20160 :
    analyticOrderedBases.length = 20160 := by
  rw [analyticOrderedBases, length_flatMap_of_constant (count := 1344)]
  · rw [first_choices_length]
  · intro first firstMem
    exact second_completions_length firstMem

structure GL4AnalyticCensusBoundary where
  parentTypedActionFamilyInstantiated : Bool
  analyticOrderedBasisListInstantiated : Bool
  firstFiberCount15Proved : Bool
  secondFiberCount14Proved : Bool
  thirdFiberCount12Proved : Bool
  fourthFiberCount8Proved : Bool
  analyticOrderedBasisCount20160Proved : Bool
  nativeMatrixScanConsumed : Bool
  orderedFramesIdentifiedWithGl4Group : Bool
  spanListsEqualLinearCombinationImagesProved : Bool
  analyticBasisToFrozenScanBijectionProved : Bool
  frozenScanCount20160ProvedAnalytically : Bool
  outer40320ActionCountProved : Bool
  actionListDistinctnessProved : Bool
  outer40320ViewMinimumProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gl4AnalyticCensusBoundary : GL4AnalyticCensusBoundary :=
  { parentTypedActionFamilyInstantiated := true
  , analyticOrderedBasisListInstantiated := true
  , firstFiberCount15Proved := true
  , secondFiberCount14Proved := true
  , thirdFiberCount12Proved := true
  , fourthFiberCount8Proved := true
  , analyticOrderedBasisCount20160Proved := true
  , nativeMatrixScanConsumed := false
  , orderedFramesIdentifiedWithGl4Group := false
  , spanListsEqualLinearCombinationImagesProved := false
  , analyticBasisToFrozenScanBijectionProved := false
  , frozenScanCount20160ProvedAnalytically := false
  , outer40320ActionCountProved := false
  , actionListDistinctnessProved := false
  , outer40320ViewMinimumProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem analytic_census_does_not_close_frozen_scan_or_target03 :
    gl4AnalyticCensusBoundary.parentTypedActionFamilyInstantiated &&
      gl4AnalyticCensusBoundary.analyticOrderedBasisListInstantiated &&
      gl4AnalyticCensusBoundary.firstFiberCount15Proved &&
      gl4AnalyticCensusBoundary.secondFiberCount14Proved &&
      gl4AnalyticCensusBoundary.thirdFiberCount12Proved &&
      gl4AnalyticCensusBoundary.fourthFiberCount8Proved &&
      gl4AnalyticCensusBoundary.analyticOrderedBasisCount20160Proved &&
      !gl4AnalyticCensusBoundary.nativeMatrixScanConsumed &&
      !gl4AnalyticCensusBoundary.orderedFramesIdentifiedWithGl4Group &&
      !gl4AnalyticCensusBoundary.spanListsEqualLinearCombinationImagesProved &&
      !gl4AnalyticCensusBoundary.analyticBasisToFrozenScanBijectionProved &&
      !gl4AnalyticCensusBoundary.frozenScanCount20160ProvedAnalytically &&
      !gl4AnalyticCensusBoundary.outer40320ActionCountProved &&
      !gl4AnalyticCensusBoundary.actionListDistinctnessProved &&
      !gl4AnalyticCensusBoundary.outer40320ViewMinimumProved &&
      !gl4AnalyticCensusBoundary.formalTarget03Closed &&
      !gl4AnalyticCensusBoundary.formalParityClosed &&
      !gl4AnalyticCensusBoundary.claimReady := by
  decide

end SounioPireusGL4AnalyticCensus
