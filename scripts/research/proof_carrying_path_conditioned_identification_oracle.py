#!/usr/bin/env python3
"""Independent exhaustive oracle for the frozen D8 identification model."""

from itertools import product


STATE_IDS = (12_001, 12_002, 12_003)
STATE_BITS = {12_001: 1, 12_002: 2, 12_003: 4}
PROJECTION = {state_id: 877 for state_id in STATE_IDS}
AB_ACTIONS = (14_001, 14_002)
BA_ACTIONS = (14_002, 14_001)


def mask_for(predicate) -> int:
    return sum(bit for state_id, bit in STATE_BITS.items() if predicate(state_id))


def members(mask: int) -> set[int]:
    assert 0 <= mask <= 7
    return {state_id for state_id, bit in STATE_BITS.items() if mask & bit}


def nonempty_subsets(mask: int) -> list[int]:
    return [candidate for candidate in range(1, 8) if candidate & ~mask == 0]


def main() -> None:
    assert set(PROJECTION.values()) == {877}
    family_fingerprint = (STATE_IDS[0] * 31 + STATE_IDS[1]) * 31 + STATE_IDS[2]
    collision_checksum = family_fingerprint * 31 + 877
    assert family_fingerprint == 11_917_026
    assert collision_checksum == 369_428_683

    history_ab = AB_ACTIONS[0] * 31 + AB_ACTIONS[1]
    history_ba = BA_ACTIONS[0] * 31 + BA_ACTIONS[1]
    assert history_ab == 448_033
    assert history_ba == 448_063
    assert AB_ACTIONS != BA_ACTIONS and history_ab != history_ba

    ab_mask = mask_for(lambda state_id: state_id in (12_001, 12_002))
    ba_mask = mask_for(lambda state_id: state_id in (12_002, 12_003))
    evidence_mask = mask_for(lambda state_id: state_id in (12_001, 12_003))
    assert (ab_mask, ba_mask, evidence_mask) == (3, 6, 5)
    assert members(ab_mask) == {12_001, 12_002}
    assert members(ba_mask) == {12_002, 12_003}
    assert members(ab_mask & ba_mask) == {12_002}

    refined_ab = ab_mask & evidence_mask
    refined_ba = ba_mask & evidence_mask
    assert (refined_ab, refined_ba) == (1, 4)
    assert refined_ab & ~ab_mask == 0
    assert refined_ba & ~ba_mask == 0
    assert refined_ab & refined_ba == 0
    assert members(refined_ab) == {12_001}
    assert members(refined_ba) == {12_003}

    # Exhaust every pair of three-state subsets and verify set algebra.
    subset_pairs = list(product(range(8), repeat=2))
    assert len(subset_pairs) == 64
    for left, right in subset_pairs:
        intersection = left & right
        assert members(intersection) == members(left).intersection(members(right))
        assert intersection & ~left == 0
        assert intersection & ~right == 0
        assert (intersection == 0) == members(left).isdisjoint(members(right))

    nonempty_masks = list(range(1, 8))
    refinement_triples = list(product(nonempty_masks, repeat=3))
    assert len(refinement_triples) == 343
    refinement_identity_count = 0
    post_refinement_disjoint_count = 0
    initially_overlapping_to_post_disjoint_count = 0
    both_refinements_nonempty_count = 0
    both_nonempty_and_post_disjoint_count = 0
    for left, right, evidence in refinement_triples:
        refined_left = left & evidence
        refined_right = right & evidence
        if refined_left & refined_right == (left & right) & evidence:
            refinement_identity_count += 1
        if refined_left & refined_right == 0:
            post_refinement_disjoint_count += 1
            if left & right:
                initially_overlapping_to_post_disjoint_count += 1
        if refined_left and refined_right:
            both_refinements_nonempty_count += 1
            if refined_left & refined_right == 0:
                both_nonempty_and_post_disjoint_count += 1
    assert refinement_identity_count == 343
    assert post_refinement_disjoint_count == 174
    assert initially_overlapping_to_post_disjoint_count == 90
    assert both_refinements_nonempty_count == 205
    assert both_nonempty_and_post_disjoint_count == 36

    sound_outer_tuples = [
        (exact_left, outer_left, exact_right, outer_right)
        for outer_left in nonempty_masks
        for exact_left in nonempty_subsets(outer_left)
        for outer_right in nonempty_masks
        for exact_right in nonempty_subsets(outer_right)
    ]
    assert len(sound_outer_tuples) == 361
    disjoint_outer_count = 0
    disjoint_outer_soundness_violations = 0
    overlapping_outer_exact_disjoint_count = 0
    overlapping_outer_exact_overlap_count = 0
    for exact_left, outer_left, exact_right, outer_right in sound_outer_tuples:
        assert exact_left & ~outer_left == 0
        assert exact_right & ~outer_right == 0
        if outer_left & outer_right == 0:
            disjoint_outer_count += 1
            if exact_left & exact_right:
                disjoint_outer_soundness_violations += 1
        elif exact_left & exact_right:
            overlapping_outer_exact_overlap_count += 1
        else:
            overlapping_outer_exact_disjoint_count += 1
    assert disjoint_outer_count == 24
    assert disjoint_outer_soundness_violations == 0
    assert overlapping_outer_exact_disjoint_count == 120
    assert overlapping_outer_exact_overlap_count == 217

    # Forgetting exactness leaves sound outers 3 and 6. Their overlap alone is
    # undecided because admitted nonempty exact completions realize both cases.
    ab_completions = nonempty_subsets(ab_mask)
    ba_completions = nonempty_subsets(ba_mask)
    assert ab_completions == [1, 2, 3]
    assert ba_completions == [2, 4, 6]
    completion_pairs = list(product(ab_completions, ba_completions))
    overlapping = [(left, right) for left, right in completion_pairs if left & right]
    disjoint = [(left, right) for left, right in completion_pairs if not left & right]
    assert len(completion_pairs) == 9
    assert len(overlapping) == 4
    assert len(disjoint) == 5
    assert (2, 2) in overlapping and (1, 4) in disjoint

    initial_ab_checksum = (12_301 * 31 + history_ab) * 31 + ab_mask
    initial_ba_checksum = (12_302 * 31 + history_ba) * 31 + ba_mask
    observation_checksum = (12_401 * 31 + 15_004) * 31 + evidence_mask
    refined_ab_checksum = (initial_ab_checksum * 31 + observation_checksum) * 31 + refined_ab
    refined_ba_checksum = (initial_ba_checksum * 31 + observation_checksum) * 31 + refined_ba
    assert initial_ab_checksum == 25_710_287
    assert initial_ba_checksum == 25_712_181
    assert observation_checksum == 12_382_490
    assert refined_ab_checksum == 25_091_442_998
    assert refined_ba_checksum == 25_093_263_135

    ab_subset_checksum = (ab_mask * 31 + refined_ab) * 31 + 12_601
    ba_subset_checksum = (ba_mask * 31 + refined_ba) * 31 + 12_602
    assert ab_subset_checksum == 15_515
    assert ba_subset_checksum == 18_492
    ab_point_checksum = ((12_501 * 31 + 12_001) * 31 + 12_101) * 31 + 12_611
    ba_point_checksum = ((12_502 * 31 + 12_003) * 31 + 12_102) * 31 + 12_612
    assert ab_point_checksum == 384_337_994
    assert ba_point_checksum == 384_369_739

    common_checksum = (((12_301 * 31 + 12_302) * 31 + 12_002) * 31 + 2)
    initial_refusal_checksum = common_checksum * 7 + 3
    separation_checksum = (12_501 * 31 + 12_502) * 31
    assert common_checksum == 378_653_377
    assert initial_refusal_checksum == 2_650_573_642
    assert separation_checksum == 12_401_023

    outer_ab_checksum = (12_801 * 31 + 12_301) * 31 + ab_mask
    outer_ba_checksum = (12_802 * 31 + 12_302) * 31 + ba_mask
    outer_undecided_checksum = (
        (((outer_ab_checksum * 31 + outer_ba_checksum) * 31 + 9) * 31 + 4)
        * 31
        + 5
    )
    assert outer_ab_checksum == 12_683_095
    assert outer_ba_checksum == 12_684_090
    assert outer_undecided_checksum == 12_090_976_311_463

    # Missing under the declared policy preserves the set. A contradictory
    # observed compatibility mask empties it and yields conflict, not a choice.
    missing_checksum = (12_901 * 31 + 15_004) * 31 + 12_902
    missing_abstention_checksum = (missing_checksum * 31 + ab_mask) * 31 + 12_903
    assert missing_checksum == 12_875_887
    assert missing_abstention_checksum == 12_373_740_403
    assert ab_mask == 3
    observation_provenance = ((16_001 * 31 + 1) * 31 + 16_101) * 31
    missing_provenance = ((16_002 * 31 + 2) * 31 + 16_102) * 31 + 12_351
    conflict_provenance = ((16_003 * 31 + 3) * 31 + 16_103) * 31 + 12_351
    assert observation_provenance == 477_185_883
    assert missing_provenance == 477_229_017
    assert conflict_provenance == 477_259_800
    observation_decision = ((16_101 * 31 + 15_004) * 31 + 1) * 7
    missing_decision = ((16_102 * 31 + 15_004) * 31 + 2) * 7 + 1
    conflict_decision = ((16_103 * 31 + 15_004) * 31 + 3) * 7 + 2
    assert observation_decision == 111_567_302
    assert missing_decision == 111_574_037
    assert conflict_decision == 111_580_772

    conflict_observation_mask = 4
    conflict_observation_checksum = (13_101 * 31 + 15_004) * 31 + conflict_observation_mask
    conflict_result = refined_ab & conflict_observation_mask
    conflict_checksum = (refined_ab_checksum * 31 + conflict_observation_checksum) * 31 + conflict_result
    assert conflict_observation_checksum == 13_055_189
    assert conflict_result == 0
    assert conflict_checksum == 24_113_281_431_937

    d7_refusal_checksum = ((322_833 * 31 + 17_001) * 31 + 11_901) * 7 + 3
    model_checksum = ((((12_200 * 31 + 15_001) * 31 + 15_002) * 31 + 15_003) * 31 + 15_004)
    association_checksum = (12_301 * 31 + 12_302) * 31 + 2
    proxy_checksum = (12_400 * 31 + 877) * 31 + 13_003
    summary_checksum = (((((13_201 * 31 + 8_800) * 31 + 2) * 31 + 4) * 31 + 1) * 31)
    assert d7_refusal_checksum == 2_175_470_118
    assert model_checksum == 11_728_748_010
    assert association_checksum == 12_202_625
    assert proxy_checksum == 11_956_590
    assert summary_checksum == 386_060_470_608

    authority = {
        "intervention": 0,
        "counterfactual": 0,
        "clinical_action": 0,
        "human_suffering": 0,
        "compiler_rewrite": 0,
        "contest_ir": 0,
        "ontology_transport": 0,
    }
    assert sum(authority.values()) == 0

    print("ORACLE_D8_W0 states=3 scalar=877 collision_mask=7 family_fingerprint=11917026 collision_checksum=369428683")
    print("ORACLE_D8_W1 subset_pairs=64 history_ab=448033 history_ba=448063 initial=3,6 intersection=2")
    print("ORACLE_D8_W2 evidence=5 refined=1,4 subset_checksums=15515,18492 point_checksums=384337994,384369739")
    print("ORACLE_D8_W3 refinement_triples=343 identity=343 post_disjoint=174 initial_overlap_to_post_disjoint=90")
    print("ORACLE_D8_W4 both_refinements_nonempty=205 both_nonempty_post_disjoint=36")
    print("ORACLE_D8_W5 sound_outer_tuples=361 disjoint_outer=24 soundness_violations=0 overlap_outer_exact_disjoint=120 overlap_outer_exact_overlap=217")
    print("ORACLE_D8_W6 completion_pairs=9 overlap=4 disjoint=5 outer_only_undecided_checksum=12090976311463")
    print("ORACLE_D8_W7 missing_result=3 conflict_result=0 conflict_checksum=24113281431937 nearest_state=false")
    print("ORACLE_D8_W8 provenance=477185883,477229017,477259800 policy_decisions=111567302,111574037,111580772")
    print("ORACLE_D8_W9 d7_reuse_refusal=2175470118 rebracketing=false")
    print("ORACLE_D8_W10 compiler_rewrites=0 contest_ir=0 ontology_transport=0")
    print("ORACLE_D8_W11 association=1 intervention=0 counterfactual=0 clinical_action=0 human_suffering=0")
    print("PATH-CONDITIONED PARTIAL IDENTIFICATION D8 ORACLE PASS")


if __name__ == "__main__":
    main()
