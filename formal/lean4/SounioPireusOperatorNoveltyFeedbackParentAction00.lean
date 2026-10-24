import SounioPireusOperatorNoveltyFeedbackChallenge

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem parent_action_00_exact :
    parentGauge { matrix := 33345, swap := 0 } = 0 /\
      SounioPireusQuotientNoveltyForge.parentActionAdmitted
        SounioPireusQuotientNoveltyForge.parentTable { matrix := 33345, swap := 0 } = true := by
  decide

end SounioPireusOperatorNoveltyFeedback
