/-
  Proof-carrying check for the V13 streaming-minimum correspondence.

  This file cites the exact theorem chain consumed by the child gate.  The
  final example deliberately keeps source-hash binding and executed Sounio
  parity false; those are pre-execution receipt obligations outside Lean.
-/
import SounioPireusStreamingMinimumCorrespondence

open SounioPireusGaugeCoboundaryAction
open SounioPireusGL4ActionEnumeration
open SounioPireusGL4AnalyticBasisEncoder
open SounioPireusGL4AnalyticScanBijection
open SounioPireusGL4AnalyticActionCensus
open SounioPireusAnalyticActionClosure
open SounioPireusConcreteQuotientAction
open SounioPireusStreamingMinimumCorrespondence

example (best candidate : NormalizedBits) :
    sounioStreamingStep best candidate = min best candidate :=
  sounio_streaming_step_eq_min best candidate

example (candidates : List NormalizedBits) :
    sounioStreamingMinimum candidates = candidates.min? :=
  sounio_streaming_minimum_eq_list_min? candidates

example (view : FrozenScanActionView) :
    view ∈ frozenScanActionViews :=
  every_frozen_scan_action_view_mem view

example (view : AnalyticActionView) :
    view ∈ frozenScanActionViews.map scanViewToAnalytic ↔
      view ∈ analyticActionViews :=
  mapped_frozen_scan_action_views_membership view

example (entry : GL4ScanEntry) :
    matrixWitnessOfScanEntry entry =
      matrixWitnessOfAnalyticBasis (analyticScanEquiv.toAnalytic entry) :=
  scan_witness_eq_mapped_analytic_witness entry

example (table : SignTable) (view : FrozenScanActionView) :
    frozenScanCandidate table view =
      quotientAct (scanViewToAnalytic view) (normalizedBitsOfTable table) :=
  frozen_scan_candidate_eq_mapped_quotient_action table view

example (table : SignTable) (candidate : NormalizedBits) :
    candidate ∈ frozenScanCandidateList table ↔
      candidate ∈ concreteQuotientActionSystem.orbit
        (normalizedBitsOfTable table) :=
  frozen_scan_candidate_membership_eq_analytic_orbit table candidate

example (table : SignTable) :
    frozenScanModelStreamingCanonicalOption table =
      declaredCanonicalOption table :=
  frozen_scan_model_streaming_minimum_eq_declared_canonical table

example :
    (streamingMinimumCorrespondenceBoundary.strictCandidateReplacementEqualsLawfulMinProved &&
      streamingMinimumCorrespondenceBoundary.firstCandidateInitializationAndFoldProved &&
      streamingMinimumCorrespondenceBoundary.streamingFoldEqualsListMinimumProved &&
      streamingMinimumCorrespondenceBoundary.quotientStreamingEqualsAbstractCanonicalProved &&
      streamingMinimumCorrespondenceBoundary.frozenScanCandidateListEqualsAnalyticOrbitProved &&
      streamingMinimumCorrespondenceBoundary.frozenScanModelStreamingMinimumEqualityProved &&
      !streamingMinimumCorrespondenceBoundary.frozenSounioSourceHashBindingProved &&
      !streamingMinimumCorrespondenceBoundary.executedSounioStreamingMinimumEqualityProved &&
      !streamingMinimumCorrespondenceBoundary.formalParityClosed &&
      !streamingMinimumCorrespondenceBoundary.claimReady) = true :=
  streaming_fold_closed_without_executed_parity_promotion
