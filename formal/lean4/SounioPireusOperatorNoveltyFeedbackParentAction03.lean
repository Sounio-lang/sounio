import SounioPireusOperatorNoveltyFeedbackParentAction02

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem parent_action_03_exact :
    parentGauge { matrix := 33377, swap := 1 } = 930 /\
      SounioPireusQuotientNoveltyForge.parentActionAdmitted
        SounioPireusQuotientNoveltyForge.parentTable { matrix := 33377, swap := 1 } = true := by
  decide

end SounioPireusOperatorNoveltyFeedback
