import SounioPireusExecutedStreamingProbeCertificate

namespace SounioPireusExecutedStreamingProbeCheck

open SounioPireusGaugeSectionCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusConcreteQuotientAction
open SounioPireusStreamingMinimumCorrespondence
open SounioPireusExecutedStreamingProbe

example : admittedProbeWinnerEntry.val = admittedProbeMatrixCode :=
  admitted_probe_winner_matrix_code

example :
    (directSectionWord
      (rawAct (actionOfFrozenScanView admittedProbeWinnerView)
        admittedProbeRawTable)).val = admittedProbeGaugeWord :=
  admitted_probe_winner_gauge_word

example :
    frozenScanCandidate admittedProbeRawTable admittedProbeWinnerView =
      admittedProbeCanonical :=
  admitted_probe_winner_candidate

example : allCodeBlocks = List.finRange codeBlockCount :=
  all_code_blocks_are_exactly_fin_range

example : allCodeBlockViews = frozenScanActionViews :=
  all_code_block_views_eq_frozen_scan_action_views

example : allCodeBlockViews.length = 40320 :=
  all_code_block_views_count_is_40320

example (view : FrozenScanActionView) :
    admittedProbeCanonical ≤
      frozenScanCandidate admittedProbeRawTable view :=
  admitted_probe_canonical_le_every_frozen_candidate view

example :
    (frozenScanCandidateList admittedProbeRawTable).min? =
      some admittedProbeCanonical :=
  admitted_probe_frozen_candidate_list_minimum

example :
    frozenScanModelStreamingCanonicalOption admittedProbeRawTable =
      some admittedProbeCanonical :=
  admitted_probe_streaming_minimum_eq_packaged_canonical

example :
    declaredCanonicalOption admittedProbeRawTable =
      some admittedProbeCanonical :=
  admitted_probe_declared_canonical_eq_packaged_canonical

example :
    (executedStreamingProbeBoundary.packagedTranscriptConstantsCheckedAgainstModel &&
      executedStreamingProbeBoundary.all65536MatrixCodesPartitioned &&
      executedStreamingProbeBoundary.all40320InvertibleSwapViewsCovered &&
      executedStreamingProbeBoundary.emittedWinnerReconstructed &&
      executedStreamingProbeBoundary.emittedWinnerGaugeReconstructed &&
      executedStreamingProbeBoundary.emittedCanonicalIsModeledMinimum &&
      executedStreamingProbeBoundary.nativeDecideTrustAssumed &&
      !executedStreamingProbeBoundary.frozenTranscriptBytesBoundInsideLeanKernel &&
      !executedStreamingProbeBoundary.frozenSounioSourceHashBindingProved &&
      !executedStreamingProbeBoundary.generalExecutedSounioStreamingEqualityProved &&
      executedStreamingProbeBoundary.singleFrozenProbeReadyForExternalHashAndTrustGate &&
      !executedStreamingProbeBoundary.formalParityClosed &&
      !executedStreamingProbeBoundary.claimReady) = true :=
  single_probe_closed_without_general_execution_promotion

end SounioPireusExecutedStreamingProbeCheck
