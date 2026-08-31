/-
  FORMAL_PARITY correspondence for the minimum-update loop used by the frozen
  Sounio Pireus V13 canonicalizer.

  The executable initializes the best row with the first candidate and then
  replaces it exactly when the next row is strictly smaller.  This module
  proves that update is the lawful `min` on the exact normalized BitVec state,
  so the complete streaming fold is definitionally the `List.min?` consumed by
  the concrete quotient action theorem.

  This closes the minimum-fold algorithm and proves set-level equality between
  the modeled frozen matrix-code candidates and the analytic 40320-view
  quotient orbit.  It does not prove that these Lean definitions are bound to
  the frozen executable Sounio source hash, so executed Sounio parity remains
  open.
-/
import SounioPireusConcreteQuotientAction

namespace SounioPireusStreamingMinimumCorrespondence

set_option maxRecDepth 100000

open SounioPireusFiniteActionCanonicalization
open SounioPireusOperatorOrbitCanonicalization
open SounioPireusGaugeCoboundaryAction
open SounioPireusLinearSwapGaugeDescent
open SounioPireusMatrixCodeXorEquiv
open SounioPireusGL4ActionEnumeration
open SounioPireusGL4AnalyticCensus
open SounioPireusGL4AnalyticBasisEncoder
open SounioPireusGL4AnalyticScanBijection
open SounioPireusGL4AnalyticActionCensus
open SounioPireusAnalyticActionClosure
open SounioPireusConcreteQuotientAction

def sounioStreamingStep
    (best candidate : NormalizedBits) : NormalizedBits :=
  if candidate.val < best.val then candidate else best

theorem sounio_streaming_step_eq_min
    (best candidate : NormalizedBits) :
    sounioStreamingStep best candidate = min best candidate := by
  by_cases candidateLtBest : candidate.val < best.val
  · rw [sounioStreamingStep, if_pos candidateLtBest]
    change candidate = if best.val ≤ candidate.val then best else candidate
    rw [if_neg (BitVec.not_le.mpr candidateLtBest)]
  · rw [sounioStreamingStep, if_neg candidateLtBest]
    change best = if best.val ≤ candidate.val then best else candidate
    rw [if_pos (BitVec.not_lt.mp candidateLtBest)]

def sounioStreamingMinimum :
    List NormalizedBits -> Option NormalizedBits
  | [] => none
  | head :: tail => some (tail.foldl sounioStreamingStep head)

theorem sounio_streaming_minimum_eq_list_min?
    (candidates : List NormalizedBits) :
    sounioStreamingMinimum candidates = candidates.min? := by
  cases candidates with
  | nil => rfl
  | cons head tail =>
      have stepEquality :
          sounioStreamingStep =
            (fun best candidate : NormalizedBits => min best candidate) := by
        funext best candidate
        exact sounio_streaming_step_eq_min best candidate
      rw [sounioStreamingMinimum, stepEquality]
      rfl

def quotientStreamingCanonicalOption
    (state : NormalizedBits) : Option NormalizedBits :=
  sounioStreamingMinimum (concreteQuotientActionSystem.orbit state)

theorem quotient_streaming_minimum_eq_abstract_canonical
    (state : NormalizedBits) :
    quotientStreamingCanonicalOption state =
      concreteQuotientActionSystem.canonicalOption state := by
  exact sounio_streaming_minimum_eq_list_min?
    (concreteQuotientActionSystem.orbit state)

abbrev FrozenScanActionView := GL4ScanEntry × Bool

def frozenScanActionViews : List FrozenScanActionView :=
  frozenScanEntries.flatMap fun entry =>
    [(entry, false), (entry, true)]

def scanViewToAnalytic
    (view : FrozenScanActionView) : AnalyticActionView :=
  (analyticScanEquiv.toAnalytic view.1, view.2)

theorem every_frozen_scan_entry_mem
    (entry : GL4ScanEntry) :
    entry ∈ frozenScanEntries := by
  exact List.mem_attach invertibleMatrixCodes entry

theorem every_analytic_basis_entry_mem
    (entry : AnalyticBasisEntry) :
    entry ∈ analyticBasisEntries := by
  exact List.mem_attach analyticOrderedBases entry

theorem every_frozen_scan_action_view_mem
    (view : FrozenScanActionView) :
    view ∈ frozenScanActionViews := by
  rcases view with ⟨entry, swap⟩
  unfold frozenScanActionViews
  rw [List.mem_flatMap]
  refine ⟨entry, every_frozen_scan_entry_mem entry, ?_⟩
  cases swap <;> simp

theorem every_analytic_action_view_mem
    (view : AnalyticActionView) :
    view ∈ analyticActionViews := by
  rcases view with ⟨entry, swap⟩
  unfold analyticActionViews
  rw [List.mem_flatMap]
  refine ⟨entry, every_analytic_basis_entry_mem entry, ?_⟩
  cases swap <;> simp

theorem mapped_frozen_scan_action_views_membership
    (view : AnalyticActionView) :
    view ∈ frozenScanActionViews.map scanViewToAnalytic ↔
      view ∈ analyticActionViews := by
  constructor
  · intro _
    exact every_analytic_action_view_mem view
  · intro _
    let preimage : FrozenScanActionView :=
      (analyticScanEquiv.toScan view.1, view.2)
    apply List.mem_map.mpr
    refine ⟨preimage, every_frozen_scan_action_view_mem preimage, ?_⟩
    apply Prod.ext
    · exact analyticScanEquiv.toAnalyticToScan view.1
    · rfl

theorem scan_witness_eq_mapped_analytic_witness
    (entry : GL4ScanEntry) :
    matrixWitnessOfScanEntry entry =
      matrixWitnessOfAnalyticBasis (analyticScanEquiv.toAnalytic entry) := by
  calc
    matrixWitnessOfScanEntry entry =
        matrixWitnessOfScanEntry
          (analyticScanEquiv.toScan (analyticScanEquiv.toAnalytic entry)) :=
      congrArg matrixWitnessOfScanEntry
        (analyticScanEquiv.toScanToAnalytic entry).symm
    _ = matrixWitnessOfAnalyticBasis
        (analyticScanEquiv.toAnalytic entry) := by
      rfl

def actionOfFrozenScanView
    (view : FrozenScanActionView) : LinearSwapAction :=
  matrixCodeLinearSwapAction (matrixWitnessOfScanEntry view.1) view.2

theorem action_of_frozen_scan_view_eq_mapped_analytic_action
    (view : FrozenScanActionView) :
    actionOfFrozenScanView view = actionOfView (scanViewToAnalytic view) := by
  rw [actionOfFrozenScanView, actionOfView, scanViewToAnalytic,
    scan_witness_eq_mapped_analytic_witness]

def frozenScanCandidate
    (table : SignTable) (view : FrozenScanActionView) : NormalizedBits :=
  normalizedBitsOfTable (rawAct (actionOfFrozenScanView view) table)

theorem frozen_scan_candidate_eq_mapped_quotient_action
    (table : SignTable) (view : FrozenScanActionView) :
    frozenScanCandidate table view =
      quotientAct (scanViewToAnalytic view) (normalizedBitsOfTable table) := by
  rw [quotient_act_on_normalized_table]
  unfold frozenScanCandidate
  rw [action_of_frozen_scan_view_eq_mapped_analytic_action]

def frozenScanCandidateList (table : SignTable) : List NormalizedBits :=
  frozenScanActionViews.map (frozenScanCandidate table)

theorem frozen_scan_candidate_membership_eq_analytic_orbit
    (table : SignTable) (candidate : NormalizedBits) :
    candidate ∈ frozenScanCandidateList table ↔
      candidate ∈ concreteQuotientActionSystem.orbit
        (normalizedBitsOfTable table) := by
  change candidate ∈ frozenScanActionViews.map (frozenScanCandidate table) ↔
    candidate ∈ analyticActionViews.map
      (fun view => quotientAct view (normalizedBitsOfTable table))
  constructor
  · intro candidateMem
    rcases List.mem_map.mp candidateMem with
      ⟨scanView, scanViewMem, candidateEqual⟩
    apply List.mem_map.mpr
    refine ⟨scanViewToAnalytic scanView,
      every_analytic_action_view_mem (scanViewToAnalytic scanView), ?_⟩
    exact (frozen_scan_candidate_eq_mapped_quotient_action
      table scanView).symm.trans candidateEqual
  · intro candidateMem
    rcases List.mem_map.mp candidateMem with
      ⟨analyticView, analyticViewMem, candidateEqual⟩
    let scanView : FrozenScanActionView :=
      (analyticScanEquiv.toScan analyticView.1, analyticView.2)
    have mappedViewEqual : scanViewToAnalytic scanView = analyticView := by
      apply Prod.ext
      · exact analyticScanEquiv.toAnalyticToScan analyticView.1
      · rfl
    apply List.mem_map.mpr
    refine ⟨scanView, every_frozen_scan_action_view_mem scanView, ?_⟩
    rw [frozen_scan_candidate_eq_mapped_quotient_action, mappedViewEqual]
    exact candidateEqual

def frozenScanModelStreamingCanonicalOption
    (table : SignTable) : Option NormalizedBits :=
  sounioStreamingMinimum (frozenScanCandidateList table)

theorem frozen_scan_model_streaming_minimum_eq_declared_canonical
    (table : SignTable) :
    frozenScanModelStreamingCanonicalOption table =
      declaredCanonicalOption table := by
  rw [frozenScanModelStreamingCanonicalOption,
    sounio_streaming_minimum_eq_list_min?]
  change (frozenScanCandidateList table).min? =
    (concreteQuotientActionSystem.orbit
      (normalizedBitsOfTable table)).min?
  apply minOption_eq_of_membership_iff
  exact frozen_scan_candidate_membership_eq_analytic_orbit table

structure StreamingMinimumCorrespondenceBoundary where
  strictCandidateReplacementEqualsLawfulMinProved : Bool
  firstCandidateInitializationAndFoldProved : Bool
  streamingFoldEqualsListMinimumProved : Bool
  quotientStreamingEqualsAbstractCanonicalProved : Bool
  frozenScanCandidateListEqualsAnalyticOrbitProved : Bool
  frozenScanModelStreamingMinimumEqualityProved : Bool
  frozenSounioSourceHashBindingProved : Bool
  executedSounioStreamingMinimumEqualityProved : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def streamingMinimumCorrespondenceBoundary :
    StreamingMinimumCorrespondenceBoundary :=
  { strictCandidateReplacementEqualsLawfulMinProved := true
  , firstCandidateInitializationAndFoldProved := true
  , streamingFoldEqualsListMinimumProved := true
  , quotientStreamingEqualsAbstractCanonicalProved := true
  , frozenScanCandidateListEqualsAnalyticOrbitProved := true
  , frozenScanModelStreamingMinimumEqualityProved := true
  , frozenSounioSourceHashBindingProved := false
  , executedSounioStreamingMinimumEqualityProved := false
  , formalParityClosed := false
  , claimReady := false }

theorem streaming_fold_closed_without_executed_parity_promotion :
    (streamingMinimumCorrespondenceBoundary.strictCandidateReplacementEqualsLawfulMinProved &&
      streamingMinimumCorrespondenceBoundary.firstCandidateInitializationAndFoldProved &&
      streamingMinimumCorrespondenceBoundary.streamingFoldEqualsListMinimumProved &&
      streamingMinimumCorrespondenceBoundary.quotientStreamingEqualsAbstractCanonicalProved &&
      streamingMinimumCorrespondenceBoundary.frozenScanCandidateListEqualsAnalyticOrbitProved &&
      streamingMinimumCorrespondenceBoundary.frozenScanModelStreamingMinimumEqualityProved &&
      !streamingMinimumCorrespondenceBoundary.frozenSounioSourceHashBindingProved &&
      !streamingMinimumCorrespondenceBoundary.executedSounioStreamingMinimumEqualityProved &&
      !streamingMinimumCorrespondenceBoundary.formalParityClosed &&
      !streamingMinimumCorrespondenceBoundary.claimReady) = true := by
  decide

end SounioPireusStreamingMinimumCorrespondence
