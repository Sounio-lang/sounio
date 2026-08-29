import SounioPireusOperatorNoveltyFeedbackParent

namespace SounioPireusOperatorNoveltyFeedback
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem all_class_certificates_exact : allClassCertificates = true := by
  simp [allClassCertificates, class_certificate_0, class_certificate_1,
    class_certificate_2, class_certificate_3, class_certificate_4,
    class_certificate_5, class_certificate_6, class_certificate_7,
    class_certificate_8, class_certificate_9, class_certificate_10,
    class_certificate_11, class_certificate_12, class_certificate_13]

theorem frozen_best_member_exact : frozenBestMember = true := by decide

theorem atlas_profile_exact : atlasProfile = frozenAtlasProfile := by
  unfold atlasProfile
  rw [all_class_certificates_exact, frozen_best_member_exact]
  decide

end SounioPireusOperatorNoveltyFeedback
