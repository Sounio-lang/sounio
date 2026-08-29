import SounioPireusOperatorNoveltyFeedbackParentAction11

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem parent_profile_exact : parentProfile = frozenParentProfile := by
  unfold parentProfile parentActionsAdmitted frozenParentActions
  simp only [List.map, List.all]
  rw [parent_action_00_exact.1, parent_action_01_exact.1, parent_action_02_exact.1,
    parent_action_03_exact.1, parent_action_04_exact.1, parent_action_05_exact.1,
    parent_action_06_exact.1, parent_action_07_exact.1, parent_action_08_exact.1,
    parent_action_09_exact.1, parent_action_10_exact.1, parent_action_11_exact.1,
    parent_action_00_exact.2, parent_action_01_exact.2, parent_action_02_exact.2,
    parent_action_03_exact.2, parent_action_04_exact.2, parent_action_05_exact.2,
    parent_action_06_exact.2, parent_action_07_exact.2, parent_action_08_exact.2,
    parent_action_09_exact.2, parent_action_10_exact.2, parent_action_11_exact.2]
  decide

end SounioPireusOperatorNoveltyFeedback
