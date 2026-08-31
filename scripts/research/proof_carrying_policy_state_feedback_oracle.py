#!/usr/bin/env python3
"""Independent exhaustive oracle for the finite D5 policy-state fixture."""

from itertools import permutations, product


HYPOTHESES = (
    {"id": 510, "mode": 0, "anchor": 2, "target": 2},
    {"id": 511, "mode": 1, "anchor": 2, "target": 8},
)
ANCHOR_COST = 3
PROBE_COST = 4
BUDGET_CAP = 7
ANCHOR_PROVENANCE = 8101
POLICY_DECISION_ID = 8102
PROBE_PROVENANCE = 8103


def mask_for_anchor(value: int) -> int:
    return sum((1 << h["mode"]) for h in HYPOTHESES if h["anchor"] == value)


def mask_for_target(value: int) -> int:
    return sum((1 << h["mode"]) for h in HYPOTHESES if h["target"] == value)


def adaptive_policy(summary_code: int) -> tuple[bool, bool]:
    assert summary_code == 2
    return False, False


def apply_optional_evidence(prior_mask: int, evidence_mask: int | None) -> int:
    """An inadmissible artifact abstains; mask zero remains real refutation."""
    if evidence_mask is None:
        return prior_mask
    return prior_mask & evidence_mask


def main() -> None:
    assert len(HYPOTHESES) == 2
    assert {h["mode"] for h in HYPOTHESES} == {0, 1}
    assert {h["target"] for h in HYPOTHESES} == {2, 8}

    anchor_mask = mask_for_anchor(2)
    assert anchor_mask == 3
    summary_code = 2
    first_policy = adaptive_policy(summary_code)
    second_policy = adaptive_policy(summary_code)
    assert first_policy == second_policy == (False, False)

    observed_traces = {
        h["mode"]: (h["anchor"], summary_code, first_policy, second_policy)
        for h in HYPOTHESES
    }
    assert observed_traces[0] == observed_traces[1]
    assert HYPOTHESES[0]["target"] != HYPOTHESES[1]["target"]

    eligible_opportunities = int(first_policy[0]) + int(second_policy[0])
    observed_opportunities = int(first_policy[1]) + int(second_policy[1])
    assert eligible_opportunities == observed_opportunities == 0
    statistical_positivity = False
    policy_value_identified = False
    assert not statistical_positivity and not policy_value_identified

    low_mask = anchor_mask & mask_for_target(2)
    high_mask = anchor_mask & mask_for_target(8)
    assert low_mask == 1 and high_mask == 2

    assert ANCHOR_COST + PROBE_COST == BUDGET_CAP
    remaining_before_probe = BUDGET_CAP - ANCHOR_COST
    remaining_after_probe = BUDGET_CAP - ANCHOR_COST - PROBE_COST
    assert remaining_before_probe == 4 and remaining_after_probe == 0
    assert PROBE_COST > remaining_after_probe

    evidence_fingerprint = ANCHOR_PROVENANCE * 31 + PROBE_PROVENANCE
    feedback_fingerprint = ANCHOR_PROVENANCE * 31 + POLICY_DECISION_ID
    assert evidence_fingerprint == 259234
    assert feedback_fingerprint == 259233

    # Every Boolean custody tuple and value code is considered. Only a
    # considered, exogenously authorized, scheduled, present value of 2 or 8
    # can be admitted as a coverage probe in this frozen fixture.
    action_tuples = list(product((False, True), repeat=4))
    values = (-1, 2, 8)
    valid_probe_tuples = []
    for considered, authorized, scheduled, present in action_tuples:
        for value in values:
            valid = (
                considered
                and authorized
                and scheduled
                and present
                and value in (2, 8)
            )
            if valid:
                valid_probe_tuples.append(
                    (considered, authorized, scheduled, present, value)
                )
    assert len(action_tuples) * len(values) == 48
    assert len(valid_probe_tuples) == 2

    relabelings = list(permutations((610, 611)))
    assert len(relabelings) == 2
    for ids in relabelings:
        relabeled_family = tuple(
            {**h, "id": ids[h["mode"]]}
            for h in HYPOTHESES
        )
        by_id = {h["id"]: h for h in relabeled_family}
        assert set(by_id) == set(ids)
        assert {
            h["id"] for h in relabeled_family if h["anchor"] == 2
        } == set(ids)
        assert {
            hypothesis_id: adaptive_policy(summary_code)
            for hypothesis_id in by_id
        } == {hypothesis_id: (False, False) for hypothesis_id in ids}

        for target, expected_mode in ((2, 0), (8, 1)):
            selected_ids = {
                h["id"]
                for h in relabeled_family
                if h["anchor"] == 2 and h["target"] == target
            }
            assert len(selected_ids) == 1
            selected_id = next(iter(selected_ids))
            assert by_id[selected_id]["mode"] == expected_mode
            assert by_id[selected_id]["target"] == target

    policy_erased_evidence_mask = None
    disconnected_probe_evidence_mask = None
    policy_erased_mask = apply_optional_evidence(
        anchor_mask, policy_erased_evidence_mask
    )
    disconnected_probe_mask = apply_optional_evidence(
        anchor_mask, disconnected_probe_evidence_mask
    )
    assert policy_erased_mask == disconnected_probe_mask == anchor_mask
    assert apply_optional_evidence(anchor_mask, 0) == 0
    maximum_two_link_fingerprint = 1_000_000 * 31 + 1_000_000
    assert maximum_two_link_fingerprint == 32_000_000

    print("ORACLE_D5_W0 anchor=2 mask=3 summary=2 targets=2|8")
    print("ORACLE_D5_W1 feedback=withhold,withhold traces=equal absorbing=bounded")
    print("ORACLE_D5_W2 coverage=0/0 positivity=false policy_value=false")
    print("ORACLE_D5_W3 exogenous=2->1,8->2 burden=3+4=7 fingerprint=259234")
    print("ORACLE_D5_W4 budget_before=4 budget_after=0 second_probe=refused")
    print(
        "ORACLE_D5_W5 relabelings=2 policy_erased="
        f"{policy_erased_mask} disconnected={disconnected_probe_mask}"
    )
    print(
        "ORACLE_D5_W6 exhaustive_actions=48 valid_probes=2 "
        f"fingerprint_max2={maximum_two_link_fingerprint} i64_safe=true"
    )
    print("PROOF-CARRYING POLICY-STATE FEEDBACK D5 ORACLE PASS")


if __name__ == "__main__":
    main()
