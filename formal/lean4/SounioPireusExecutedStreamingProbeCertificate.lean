/-
  Aggregate FORMAL_PARITY certificate for the admitted=0 record emitted by the
  frozen Sounio V13 canonicalizer.

  Sixty-four independently compiled native reductions establish that the
  packaged canonical table is no larger than every normalized action in each
  1024-code block.  This file combines those certificates, proves that the
  blocks cover the exact frozen scan carrier, and derives the concrete list
  minimum.  The result is one frozen execution probe, not general equivalence
  between the Sounio implementation and the Lean model.
-/
import SounioPireusExecutedStreamingProbeBlocks0
import SounioPireusExecutedStreamingProbeBlocks1
import SounioPireusExecutedStreamingProbeBlocks2
import SounioPireusExecutedStreamingProbeBlocks3
import SounioPireusExecutedStreamingProbeBlocks4
import SounioPireusExecutedStreamingProbeBlocks5
import SounioPireusExecutedStreamingProbeBlocks6
import SounioPireusExecutedStreamingProbeBlocks7

namespace SounioPireusExecutedStreamingProbe

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusFiniteActionCanonicalization
open SounioPireusGL4AnalyticScanBijection
open SounioPireusConcreteQuotientAction
open SounioPireusStreamingMinimumCorrespondence

def allCodeBlocks : List (Fin codeBlockCount) :=
  [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩, ⟨3, by decide⟩,
   ⟨4, by decide⟩, ⟨5, by decide⟩, ⟨6, by decide⟩, ⟨7, by decide⟩,
   ⟨8, by decide⟩, ⟨9, by decide⟩, ⟨10, by decide⟩, ⟨11, by decide⟩,
   ⟨12, by decide⟩, ⟨13, by decide⟩, ⟨14, by decide⟩, ⟨15, by decide⟩,
   ⟨16, by decide⟩, ⟨17, by decide⟩, ⟨18, by decide⟩, ⟨19, by decide⟩,
   ⟨20, by decide⟩, ⟨21, by decide⟩, ⟨22, by decide⟩, ⟨23, by decide⟩,
   ⟨24, by decide⟩, ⟨25, by decide⟩, ⟨26, by decide⟩, ⟨27, by decide⟩,
   ⟨28, by decide⟩, ⟨29, by decide⟩, ⟨30, by decide⟩, ⟨31, by decide⟩,
   ⟨32, by decide⟩, ⟨33, by decide⟩, ⟨34, by decide⟩, ⟨35, by decide⟩,
   ⟨36, by decide⟩, ⟨37, by decide⟩, ⟨38, by decide⟩, ⟨39, by decide⟩,
   ⟨40, by decide⟩, ⟨41, by decide⟩, ⟨42, by decide⟩, ⟨43, by decide⟩,
   ⟨44, by decide⟩, ⟨45, by decide⟩, ⟨46, by decide⟩, ⟨47, by decide⟩,
   ⟨48, by decide⟩, ⟨49, by decide⟩, ⟨50, by decide⟩, ⟨51, by decide⟩,
   ⟨52, by decide⟩, ⟨53, by decide⟩, ⟨54, by decide⟩, ⟨55, by decide⟩,
   ⟨56, by decide⟩, ⟨57, by decide⟩, ⟨58, by decide⟩, ⟨59, by decide⟩,
   ⟨60, by decide⟩, ⟨61, by decide⟩, ⟨62, by decide⟩, ⟨63, by decide⟩]

def allCodeBlockViews : List FrozenScanActionView :=
  allCodeBlocks.flatMap codeBlockViews

theorem every_code_block_dominates_frozen_canonical :
    allCodeBlocks.all codeBlockDominatesFrozenCanonical = true := by
  simp only [allCodeBlocks, List.all_cons, List.all_nil,
    code_block_00_dominates, code_block_01_dominates,
    code_block_02_dominates, code_block_03_dominates,
    code_block_04_dominates, code_block_05_dominates,
    code_block_06_dominates, code_block_07_dominates,
    code_block_08_dominates, code_block_09_dominates,
    code_block_10_dominates, code_block_11_dominates,
    code_block_12_dominates, code_block_13_dominates,
    code_block_14_dominates, code_block_15_dominates,
    code_block_16_dominates, code_block_17_dominates,
    code_block_18_dominates, code_block_19_dominates,
    code_block_20_dominates, code_block_21_dominates,
    code_block_22_dominates, code_block_23_dominates,
    code_block_24_dominates, code_block_25_dominates,
    code_block_26_dominates, code_block_27_dominates,
    code_block_28_dominates, code_block_29_dominates,
    code_block_30_dominates, code_block_31_dominates,
    code_block_32_dominates, code_block_33_dominates,
    code_block_34_dominates, code_block_35_dominates,
    code_block_36_dominates, code_block_37_dominates,
    code_block_38_dominates, code_block_39_dominates,
    code_block_40_dominates, code_block_41_dominates,
    code_block_42_dominates, code_block_43_dominates,
    code_block_44_dominates, code_block_45_dominates,
    code_block_46_dominates, code_block_47_dominates,
    code_block_48_dominates, code_block_49_dominates,
    code_block_50_dominates, code_block_51_dominates,
    code_block_52_dominates, code_block_53_dominates,
    code_block_54_dominates, code_block_55_dominates,
    code_block_56_dominates, code_block_57_dominates,
    code_block_58_dominates, code_block_59_dominates,
    code_block_60_dominates, code_block_61_dominates,
    code_block_62_dominates, code_block_63_dominates]
  decide

theorem all_code_blocks_are_exactly_fin_range :
    allCodeBlocks = List.finRange codeBlockCount := by
  native_decide

theorem all_code_block_views_eq_frozen_scan_action_views :
    allCodeBlockViews = frozenScanActionViews := by
  native_decide

theorem two_views_per_entry_length (entries : List GL4ScanEntry) :
    (entries.flatMap fun entry => [(entry, false), (entry, true)]).length =
      entries.length * 2 := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
      simp only [List.flatMap_cons, List.length_append, List.length_cons,
        List.length_nil, ih]
      omega

theorem all_code_block_views_count_is_40320 :
    allCodeBlockViews.length = 40320 := by
  rw [all_code_block_views_eq_frozen_scan_action_views]
  rw [frozenScanActionViews, two_views_per_entry_length,
    frozen_scan_census_is_20160_analytically]

theorem all_code_block_views_dominate_frozen_canonical :
    allCodeBlockViews.all candidateNotBelowFrozenCanonical = true := by
  apply List.all_eq_true.mpr
  intro view viewMem
  obtain ⟨block, blockMem, viewInBlock⟩ := List.mem_flatMap.mp viewMem
  have blockDominates :
      codeBlockDominatesFrozenCanonical block = true :=
    (List.all_eq_true.mp every_code_block_dominates_frozen_canonical)
      block blockMem
  exact (List.all_eq_true.mp blockDominates) view viewInBlock

theorem frozen_scan_views_dominate_admitted_probe_canonical :
    frozenScanActionViews.all candidateNotBelowFrozenCanonical = true := by
  rw [← all_code_block_views_eq_frozen_scan_action_views]
  exact all_code_block_views_dominate_frozen_canonical

theorem admitted_probe_canonical_le_every_frozen_candidate
    (view : FrozenScanActionView) :
    admittedProbeCanonical ≤
      frozenScanCandidate admittedProbeRawTable view := by
  have decision :=
    (List.all_eq_true.mp
      frozen_scan_views_dominate_admitted_probe_canonical)
      view (every_frozen_scan_action_view_mem view)
  change admittedProbeCanonical.val ≤
    (frozenScanCandidate admittedProbeRawTable view).val
  apply of_decide_eq_true
  simpa [candidateNotBelowFrozenCanonical] using decision

theorem admitted_probe_canonical_mem_frozen_candidate_list :
    admittedProbeCanonical ∈
      frozenScanCandidateList admittedProbeRawTable := by
  apply List.mem_map.mpr
  exact ⟨admittedProbeWinnerView,
    every_frozen_scan_action_view_mem admittedProbeWinnerView,
    admitted_probe_winner_candidate⟩

theorem admitted_probe_frozen_candidate_list_minimum :
    (frozenScanCandidateList admittedProbeRawTable).min? =
      some admittedProbeCanonical := by
  apply List.min?_eq_some_iff.mpr
  refine ⟨admitted_probe_canonical_mem_frozen_candidate_list, ?_⟩
  intro candidate candidateMem
  rcases List.mem_map.mp candidateMem with ⟨view, _, rfl⟩
  exact admitted_probe_canonical_le_every_frozen_candidate view

theorem admitted_probe_streaming_minimum_eq_packaged_canonical :
    frozenScanModelStreamingCanonicalOption admittedProbeRawTable =
      some admittedProbeCanonical := by
  rw [frozenScanModelStreamingCanonicalOption,
    sounio_streaming_minimum_eq_list_min?]
  exact admitted_probe_frozen_candidate_list_minimum

theorem admitted_probe_declared_canonical_eq_packaged_canonical :
    declaredCanonicalOption admittedProbeRawTable =
      some admittedProbeCanonical := by
  rw [← frozen_scan_model_streaming_minimum_eq_declared_canonical]
  exact admitted_probe_streaming_minimum_eq_packaged_canonical

structure ExecutedStreamingProbeBoundary where
  packagedTranscriptConstantsCheckedAgainstModel : Bool
  all65536MatrixCodesPartitioned : Bool
  all40320InvertibleSwapViewsCovered : Bool
  emittedWinnerReconstructed : Bool
  emittedWinnerGaugeReconstructed : Bool
  emittedCanonicalIsModeledMinimum : Bool
  nativeDecideTrustAssumed : Bool
  frozenTranscriptBytesBoundInsideLeanKernel : Bool
  frozenSounioSourceHashBindingProved : Bool
  generalExecutedSounioStreamingEqualityProved : Bool
  singleFrozenProbeReadyForExternalHashAndTrustGate : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def executedStreamingProbeBoundary : ExecutedStreamingProbeBoundary :=
  { packagedTranscriptConstantsCheckedAgainstModel := true
  , all65536MatrixCodesPartitioned := true
  , all40320InvertibleSwapViewsCovered := true
  , emittedWinnerReconstructed := true
  , emittedWinnerGaugeReconstructed := true
  , emittedCanonicalIsModeledMinimum := true
  , nativeDecideTrustAssumed := true
  , frozenTranscriptBytesBoundInsideLeanKernel := false
  , frozenSounioSourceHashBindingProved := false
  , generalExecutedSounioStreamingEqualityProved := false
  , singleFrozenProbeReadyForExternalHashAndTrustGate := true
  , formalParityClosed := false
  , claimReady := false }

theorem single_probe_closed_without_general_execution_promotion :
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
      !executedStreamingProbeBoundary.claimReady) = true := by
  native_decide

end SounioPireusExecutedStreamingProbe
