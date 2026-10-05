import SounioPireusOperatorNoveltyFeedbackParentAction00

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem parent_action_01_exact :
    parentGauge { matrix := 33345, swap := 1 } = 2027 /\
      SounioPireusQuotientNoveltyForge.parentActionAdmitted
        SounioPireusQuotientNoveltyForge.parentTable { matrix := 33345, swap := 1 } = true := by
  decide

end SounioPireusOperatorNoveltyFeedback
