/-
  Composed FORMAL_PARITY certificate for the Sounio-owned Pireus Operator
  Novelty Feedback v7. The 14 class certificates are chained build shards so
  each kernel process releases its finite-census working set before the next.
-/
import SounioPireusOperatorNoveltyFeedbackAtlas

namespace SounioPireusOperatorNoveltyFeedback

set_option maxHeartbeats 0
set_option maxRecDepth 1000000

theorem formal_parity_summary_matches_frozen_sounio :
    formalParitySummary = frozenFormalParitySummary := by
  unfold formalParitySummary
  rw [challenge_profile_exact, parent_profile_exact, atlas_profile_exact]
  decide

theorem cd16_challenge_census_and_words_exact :
    formalParitySummary.challengePositive = 136 &&
      formalParitySummary.challengeNegative = 120 &&
      formalParitySummary.challengeWords =
        [2523529216, 1521237190, 2859790366, 3434243800,
         2543059454, 1532116280, 2878009824, 3444336422] := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem frozen_parent_actions_admitted_closed_and_invertible :
    formalParitySummary.actionCount = 12 &&
      formalParitySummary.parentActionsAdmitted &&
      formalParitySummary.parentActionClosure &&
      formalParitySummary.parentActionInverses &&
      formalParitySummary.parentGauges =
        [0, 2027, 1097, 930, 0, 2027, 1290, 737, 1097, 930, 1290, 737] := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem atlas_enumeration_and_nonmembership_exact :
    formalParitySummary.classCount = 14 &&
      formalParitySummary.representativesBoundAndUnique &&
      formalParitySummary.pairCount = 168 &&
      formalParitySummary.pairReplayFailures = 0 &&
      formalParitySummary.zeroResidualHits = 0 &&
      formalParitySummary.exhaustiveNonmembership := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem canonical_operator_seed_witness_exact :
    formalParitySummary.bestClass = 8 &&
      formalParitySummary.bestRepresentative = 13 &&
      formalParitySummary.bestActionIndex = 8 &&
      formalParitySummary.bestActionCode = 68674 &&
      formalParitySummary.bestMatrix = 34337 &&
      formalParitySummary.bestSwap = 0 &&
      formalParitySummary.bestParentGauge = 1097 &&
      formalParitySummary.bestChallengeGauge = 1813 &&
      formalParitySummary.bestResidualWords =
        [0, 0, 1010580540, 4042322160, 2863311530, 2863311530, 2526451350, 1515870810] &&
      formalParitySummary.bestResidualNonzero = 96 &&
      formalParitySummary.bestReplayChecks = 256 &&
      formalParitySummary.bestReplayFailures = 0 := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

theorem operator_seed_and_claim_bound_exact :
    formalParitySummary.outcomeKind = 2 &&
      !formalParitySummary.existingClassBridge &&
      formalParitySummary.operatorSeedGenerated &&
      !formalParitySummary.broadNovelty &&
      !formalParitySummary.historicalNovelty &&
      !formalParitySummary.priorityClaim &&
      !formalParitySummary.claimReady := by
  rw [formal_parity_summary_matches_frozen_sounio]
  decide

end SounioPireusOperatorNoveltyFeedback
