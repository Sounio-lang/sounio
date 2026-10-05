import SounioPireusOperatorNoveltyFeedbackParentAction01

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem parent_action_02_exact :
    parentGauge { matrix := 33377, swap := 0 } = 1097 /\
      SounioPireusQuotientNoveltyForge.parentActionAdmitted
        SounioPireusQuotientNoveltyForge.parentTable { matrix := 33377, swap := 0 } = true := by
  decide

end SounioPireusOperatorNoveltyFeedback
