/-
  Concrete FORMAL_PARITY closure of the 40320 frozen Pireus GL(4,F2) x C2
  views.

  Every XOR-linear equivalence is re-encoded from the images of the standard
  basis lanes 1, 2, 4, and 8.  Injectivity transports the four successive
  outside-span obligations, so the re-encoded basis belongs to the analytic
  20160-entry census.  Equality on those four lanes determines the whole
  XOR-linear map.  This supplies identity, composition, and inverse directly
  in the analytic view type without evaluating the native 65536-code scan.
-/
import SounioPireusSignTableBitVecLex

namespace SounioPireusAnalyticActionClosure

set_option maxHeartbeats 0
set_option maxRecDepth 100000

open SounioPireusGaugeCoboundaryAction
open SounioPireusLinearSwapGaugeDescent
open SounioPireusBasisFixedGaugeRebase
open SounioPireusMatrixCodeXorEquiv
open SounioPireusGL4AnalyticCensus
open SounioPireusGL4AnalyticScanEmbedding
open SounioPireusGL4AnalyticBasisEncoder
open SounioPireusGL4AnalyticScanBijection
open SounioPireusGL4AnalyticActionCensus

theorem xor_lane_equiv_injective (linear : XorLaneEquiv) :
    Function.Injective linear.toFun := by
  intro left right imagesEqual
  have preimagesEqual := congrArg linear.invFun imagesEqual
  simpa only [linear.leftInverse] using preimagesEqual

theorem linear_map_span_zero (linear : XorLaneEquiv) :
    spanZero.map linear.toFun = spanZero := by
  simp [spanZero, linear.mapZero]

theorem linear_map_span_one (linear : XorLaneEquiv) (first : Lane) :
    (spanOne first).map linear.toFun =
      spanOne (linear.toFun first) := by
  simp [spanOne, linear.mapZero]

theorem linear_map_span_two
    (linear : XorLaneEquiv) (first second : Lane) :
    (spanTwo first second).map linear.toFun =
      spanTwo (linear.toFun first) (linear.toFun second) := by
  rw [spanTwo, List.map_append, linear_map_span_one, spanTwo]
  congr 1
  rw [List.map_map, ← linear_map_span_one linear first, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact linear.mapXor lane second

theorem linear_map_span_three
    (linear : XorLaneEquiv) (first second third : Lane) :
    (spanThree first second third).map linear.toFun =
      spanThree (linear.toFun first) (linear.toFun second)
        (linear.toFun third) := by
  rw [spanThree, List.map_append, linear_map_span_two, spanThree]
  congr 1
  rw [List.map_map,
    ← linear_map_span_two linear first second, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact linear.mapXor lane third

theorem linear_map_span_four
    (linear : XorLaneEquiv) (first second third fourth : Lane) :
    (spanFour first second third fourth).map linear.toFun =
      spanFour (linear.toFun first) (linear.toFun second)
        (linear.toFun third) (linear.toFun fourth) := by
  rw [spanFour, List.map_append, linear_map_span_three, spanFour]
  congr 1
  rw [List.map_map,
    ← linear_map_span_three linear first second third, List.map_map]
  apply List.map_congr_left
  intro lane _
  exact linear.mapXor lane fourth

def basisOfLinear (linear : XorLaneEquiv) : OrderedBasis4 :=
  (linear.toFun lane1, linear.toFun lane2,
    linear.toFun lane4, linear.toFun lane8)

theorem basis_of_linear_first_mem (linear : XorLaneEquiv) :
    linear.toFun lane1 ∈ firstChoices := by
  have outsideMapped := image_not_mem_map_of_injective
    (xor_lane_equiv_injective linear) standard_first_outside
  rw [linear_map_span_zero] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_linear_second_mem (linear : XorLaneEquiv) :
    linear.toFun lane2 ∈ secondChoices (linear.toFun lane1) := by
  have outsideMapped := image_not_mem_map_of_injective
    (xor_lane_equiv_injective linear) standard_second_outside
  rw [linear_map_span_one] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_linear_third_mem (linear : XorLaneEquiv) :
    linear.toFun lane4 ∈
      thirdChoices (linear.toFun lane1) (linear.toFun lane2) := by
  have outsideMapped := image_not_mem_map_of_injective
    (xor_lane_equiv_injective linear) standard_third_outside
  rw [linear_map_span_two] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_linear_fourth_mem (linear : XorLaneEquiv) :
    linear.toFun lane8 ∈ fourthChoices (linear.toFun lane1)
      (linear.toFun lane2) (linear.toFun lane4) := by
  have outsideMapped := image_not_mem_map_of_injective
    (xor_lane_equiv_injective linear) standard_fourth_outside
  rw [linear_map_span_three] at outsideMapped
  exact mem_choicesOutside_of_not_mem_span outsideMapped

theorem basis_of_linear_mem_analytic_ordered_bases
    (linear : XorLaneEquiv) :
    basisOfLinear linear ∈ analyticOrderedBases := by
  rw [analyticOrderedBases, List.mem_flatMap]
  refine ⟨linear.toFun lane1, basis_of_linear_first_mem linear, ?_⟩
  rw [secondCompletions, List.mem_flatMap]
  refine ⟨linear.toFun lane2, basis_of_linear_second_mem linear, ?_⟩
  rw [thirdCompletions, List.mem_flatMap]
  refine ⟨linear.toFun lane4, basis_of_linear_third_mem linear, ?_⟩
  rw [fourthCompletions, List.mem_map]
  exact ⟨linear.toFun lane8, basis_of_linear_fourth_mem linear, rfl⟩

def analyticBasisEntryOfLinear (linear : XorLaneEquiv) :
    AnalyticBasisEntry :=
  ⟨basisOfLinear linear,
    basis_of_linear_mem_analytic_ordered_bases linear⟩

abbrev AnalyticActionView := AnalyticBasisEntry × Bool

def viewOfAction (action : LinearSwapAction) : AnalyticActionView :=
  (analyticBasisEntryOfLinear action.linear, action.swap)

theorem analytic_basis_entry_of_linear_mem (linear : XorLaneEquiv) :
    analyticBasisEntryOfLinear linear ∈ analyticBasisEntries := by
  unfold analyticBasisEntries
  exact List.mem_attach analyticOrderedBases _

theorem view_of_action_mem (action : LinearSwapAction) :
    viewOfAction action ∈ analyticActionViews := by
  unfold analyticActionViews
  rw [List.mem_flatMap]
  refine ⟨analyticBasisEntryOfLinear action.linear,
    analytic_basis_entry_of_linear_mem action.linear, ?_⟩
  cases action.swap <;> simp [viewOfAction]

theorem xor_linear_to_fun_eq_of_basis
    (left right : XorLaneEquiv)
    (lane1Equal : left.toFun lane1 = right.toFun lane1)
    (lane2Equal : left.toFun lane2 = right.toFun lane2)
    (lane4Equal : left.toFun lane4 = right.toFun lane4)
    (lane8Equal : left.toFun lane8 = right.toFun lane8) :
    left.toFun = right.toFun := by
  have mappedLists : laneUniverse.map left.toFun =
      laneUniverse.map right.toFun := by
    rw [← standard_span_four_is_lane_universe,
      linear_map_span_four, linear_map_span_four,
      lane1Equal, lane2Equal, lane4Equal, lane8Equal]
  funext lane
  have atLane := congrArg
    (fun values : List Lane => values[lane.val]?) mappedLists
  simpa [laneUniverse] using atLane

theorem recoded_action_lane1 (action : LinearSwapAction) :
    (actionOfView (viewOfAction action)).linear.toFun lane1 =
      action.linear.toFun lane1 := by
  change matrixLaneMap (matrixCodeOfBasis (basisOfLinear action.linear)) lane1 =
    action.linear.toFun lane1
  rw [matrix_lane_map_code_of_basis_lane1]
  rfl

theorem recoded_action_lane2 (action : LinearSwapAction) :
    (actionOfView (viewOfAction action)).linear.toFun lane2 =
      action.linear.toFun lane2 := by
  change matrixLaneMap (matrixCodeOfBasis (basisOfLinear action.linear)) lane2 =
    action.linear.toFun lane2
  rw [matrix_lane_map_code_of_basis_lane2]
  rfl

theorem recoded_action_lane4 (action : LinearSwapAction) :
    (actionOfView (viewOfAction action)).linear.toFun lane4 =
      action.linear.toFun lane4 := by
  change matrixLaneMap (matrixCodeOfBasis (basisOfLinear action.linear)) lane4 =
    action.linear.toFun lane4
  rw [matrix_lane_map_code_of_basis_lane4]
  rfl

theorem recoded_action_lane8 (action : LinearSwapAction) :
    (actionOfView (viewOfAction action)).linear.toFun lane8 =
      action.linear.toFun lane8 := by
  change matrixLaneMap (matrixCodeOfBasis (basisOfLinear action.linear)) lane8 =
    action.linear.toFun lane8
  rw [matrix_lane_map_code_of_basis_lane8]
  rfl

theorem recoded_action_to_fun (action : LinearSwapAction) :
    (actionOfView (viewOfAction action)).linear.toFun =
      action.linear.toFun := by
  exact xor_linear_to_fun_eq_of_basis _ _
    (recoded_action_lane1 action) (recoded_action_lane2 action)
    (recoded_action_lane4 action) (recoded_action_lane8 action)

theorem raw_act_view_of_action
    (action : LinearSwapAction) (table : SignTable) :
    rawAct (actionOfView (viewOfAction action)) table =
      rawAct action table := by
  funext cell
  have leftEqual := congrFun (recoded_action_to_fun action) cell.1
  have rightEqual := congrFun (recoded_action_to_fun action) cell.2
  have swapEqual :
      (actionOfView (viewOfAction action)).swap = action.swap := rfl
  simp only [rawAct]
  rw [swapEqual, leftEqual, rightEqual]

def identityView : AnalyticActionView := viewOfAction identityAction

def composeView
    (outer inner : AnalyticActionView) : AnalyticActionView :=
  viewOfAction (composeAction (actionOfView outer) (actionOfView inner))

def inverseView (view : AnalyticActionView) : AnalyticActionView :=
  viewOfAction (inverseAction (actionOfView view))

theorem identity_view_mem : identityView ∈ analyticActionViews :=
  view_of_action_mem identityAction

theorem compose_view_mem
    {outer inner : AnalyticActionView}
    (_outerMem : outer ∈ analyticActionViews)
    (_innerMem : inner ∈ analyticActionViews) :
    composeView outer inner ∈ analyticActionViews :=
  view_of_action_mem _

theorem inverse_view_mem
    {view : AnalyticActionView} (_viewMem : view ∈ analyticActionViews) :
    inverseView view ∈ analyticActionViews :=
  view_of_action_mem _

theorem raw_act_identity_view (table : SignTable) :
    rawAct (actionOfView identityView) table = table := by
  rw [identityView, raw_act_view_of_action, raw_action_identity]

theorem raw_act_compose_view
    (outer inner : AnalyticActionView) (table : SignTable) :
    rawAct (actionOfView (composeView outer inner)) table =
      rawAct (actionOfView outer) (rawAct (actionOfView inner) table) := by
  rw [composeView, raw_act_view_of_action, raw_action_compose]

theorem raw_act_inverse_view
    (view : AnalyticActionView) (table : SignTable) :
    rawAct (actionOfView (inverseView view))
        (rawAct (actionOfView view) table) = table := by
  rw [inverseView, raw_act_view_of_action, raw_action_inverse]

end SounioPireusAnalyticActionClosure
