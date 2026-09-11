/-
  Analytic FORMAL_PARITY census for the concrete GL(4,F2) x input-swap
  action family used by frozen Pireus V13.

  This module consumes the compiled duplicate-free 20160 scan/basis census,
  pairs every analytic basis with both swap values, and maps those 40320 views
  to distinct concrete LinearSwapActions.  Gauge integration, executed-
  normalizer equality, Target-03, and CLAIM_READY remain open.
-/
import SounioPireusGL4AnalyticScanBijection

namespace SounioPireusGL4AnalyticActionCensus

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusOperatorOrbitCanonicalization
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusMatrixCodeXorEquiv
open SounioPireusGL4AnalyticCensus
open SounioPireusGL4AnalyticBasisEncoder
open SounioPireusGL4AnalyticScanBijection

def analyticActionViews : List (AnalyticBasisEntry × Bool) :=
  analyticBasisEntries.flatMap fun entry =>
    [(entry, false), (entry, true)]

def actionOfView (view : AnalyticBasisEntry × Bool) : LinearSwapAction :=
  matrixCodeLinearSwapAction (matrixWitnessOfAnalyticBasis view.1) view.2

def analyticConcreteActionList : List LinearSwapAction :=
  analyticActionViews.map actionOfView

theorem analytic_concrete_action_list_length_is_40320 :
    analyticConcreteActionList.length = 40320 := by
  rw [analyticConcreteActionList, List.length_map,
    analyticActionViews,
    length_flatMap_of_constant (count := 2)]
  · rw [analyticBasisEntries, List.length_attach,
      analytic_ordered_basis_census_is_20160]
  · intro entry entryMem
    simp

theorem action_view_fiber_nodup (entry : AnalyticBasisEntry) :
    ([(entry, false), (entry, true)] :
      List (AnalyticBasisEntry × Bool)).Nodup := by
  simp

theorem analytic_action_views_nodup : analyticActionViews.Nodup := by
  unfold analyticActionViews
  apply nodup_flatMap_of_tagged_fibers
  · exact analytic_basis_entries_nodup
  · intro entry _
    exact action_view_fiber_nodup entry
  · intro left _ right _ different value leftMem rightMem
    simp only [List.mem_cons, List.not_mem_nil, or_false] at leftMem rightMem
    rcases leftMem with leftEqual | leftEqual <;>
      rcases rightMem with rightEqual | rightEqual <;>
      exact different ((congrArg Prod.fst leftEqual).symm.trans
        (congrArg Prod.fst rightEqual))

theorem action_of_view_injective : Function.Injective actionOfView := by
  intro left right actionEqual
  have swapEqual : left.2 = right.2 :=
    congrArg (fun action : LinearSwapAction => action.swap) actionEqual
  have lane1Equal := congrArg
    (fun action : LinearSwapAction => action.linear.toFun lane1) actionEqual
  have lane2Equal := congrArg
    (fun action : LinearSwapAction => action.linear.toFun lane2) actionEqual
  have lane4Equal := congrArg
    (fun action : LinearSwapAction => action.linear.toFun lane4) actionEqual
  have lane8Equal := congrArg
    (fun action : LinearSwapAction => action.linear.toFun lane8) actionEqual
  simp only [actionOfView, matrixCodeLinearSwapAction, matrixCodeXorEquiv,
    matrixWitnessOfAnalyticBasis, boundedMatrixCodeOfBasis] at lane1Equal lane2Equal lane4Equal lane8Equal
  rw [matrix_lane_map_code_of_basis_lane1,
    matrix_lane_map_code_of_basis_lane1] at lane1Equal
  rw [matrix_lane_map_code_of_basis_lane2,
    matrix_lane_map_code_of_basis_lane2] at lane2Equal
  rw [matrix_lane_map_code_of_basis_lane4,
    matrix_lane_map_code_of_basis_lane4] at lane4Equal
  rw [matrix_lane_map_code_of_basis_lane8,
    matrix_lane_map_code_of_basis_lane8] at lane8Equal
  apply Prod.ext
  · apply Subtype.ext
    apply Prod.ext
    · exact lane1Equal
    · apply Prod.ext
      · exact lane2Equal
      · apply Prod.ext
        · exact lane4Equal
        · exact lane8Equal
  · exact swapEqual

theorem analytic_concrete_action_list_nodup :
    analyticConcreteActionList.Nodup := by
  exact nodup_map_of_injective action_of_view_injective
    analytic_action_views_nodup

theorem any_enumeration_containing_declared_actions_has_at_least_40320
    (candidate : List LinearSwapAction)
    (complete : analyticConcreteActionList ⊆ candidate) :
    40320 ≤ candidate.length := by
  have bound := analytic_concrete_action_list_nodup.length_le_of_subset complete
  simpa [analytic_concrete_action_list_length_is_40320] using bound

structure GL4AnalyticScanBijectionBoundary where
  parentAnalyticCensusProved : Bool
  scanEmbeddingProved : Bool
  analyticBasisEncoderProved : Bool
  analyticScanBijectionProved : Bool
  analyticOrderedBasesNodupProved : Bool
  frozenScanCount20160Proved : Bool
  outerActionListCount40320Proved : Bool
  concreteActionListDistinctnessProved : Bool
  outer40320MinimumProved : Bool
  nativeMatrixScanEvaluatedForCount : Bool
  concreteCanonicalEqualityIffFullDeclaredOrbitProved : Bool
  formalTarget03Closed : Bool
  formalParityClosed : Bool
  claimReady : Bool
deriving Repr, BEq, DecidableEq

def gl4AnalyticScanBijectionBoundary : GL4AnalyticScanBijectionBoundary :=
  { parentAnalyticCensusProved := true
  , scanEmbeddingProved := true
  , analyticBasisEncoderProved := true
  , analyticScanBijectionProved := true
  , analyticOrderedBasesNodupProved := true
  , frozenScanCount20160Proved := true
  , outerActionListCount40320Proved := true
  , concreteActionListDistinctnessProved := true
  , outer40320MinimumProved := true
  , nativeMatrixScanEvaluatedForCount := false
  , concreteCanonicalEqualityIffFullDeclaredOrbitProved := false
  , formalTarget03Closed := false
  , formalParityClosed := false
  , claimReady := false }

theorem scan_basis_extraction_is_partial_not_target03 :
    gl4AnalyticScanBijectionBoundary.parentAnalyticCensusProved &&
      gl4AnalyticScanBijectionBoundary.scanEmbeddingProved &&
      gl4AnalyticScanBijectionBoundary.analyticBasisEncoderProved &&
      gl4AnalyticScanBijectionBoundary.analyticScanBijectionProved &&
      gl4AnalyticScanBijectionBoundary.analyticOrderedBasesNodupProved &&
      gl4AnalyticScanBijectionBoundary.frozenScanCount20160Proved &&
      gl4AnalyticScanBijectionBoundary.outerActionListCount40320Proved &&
      gl4AnalyticScanBijectionBoundary.concreteActionListDistinctnessProved &&
      gl4AnalyticScanBijectionBoundary.outer40320MinimumProved &&
      !gl4AnalyticScanBijectionBoundary.nativeMatrixScanEvaluatedForCount &&
      !gl4AnalyticScanBijectionBoundary.concreteCanonicalEqualityIffFullDeclaredOrbitProved &&
      !gl4AnalyticScanBijectionBoundary.formalTarget03Closed &&
      !gl4AnalyticScanBijectionBoundary.formalParityClosed &&
      !gl4AnalyticScanBijectionBoundary.claimReady := by
  decide

end SounioPireusGL4AnalyticActionCensus
