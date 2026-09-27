#!/usr/bin/env python3
"""Independent exact oracle for the frozen D10 deployment-validity fixture.

This script enumerates bounded synthetic contests only. It establishes no
external validation, production deployment authority, patient state, clinical
action authority, general anytime-valid theorem, affine consumption, or live
revocation guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product


OUTCOMES = (0, 0, 1, 1)
SCORES_A_PERMILLE = (100, 200, 800, 900)
SCORES_B_PERMILLE = (400, 450, 550, 600)
DECISION_THRESHOLD = Fraction(1, 2)

PATH_MASSES = {
    (True, True): 9,
    (True, False): 3,
    (False, True): 3,
    (False, False): 1,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def probabilities(permille_scores: tuple[int, ...]) -> tuple[Fraction, ...]:
    return tuple(Fraction(score, 1000) for score in permille_scores)


def brier_score(
    outcomes: tuple[int, ...], scores: tuple[Fraction, ...]
) -> Fraction:
    require(len(outcomes) == len(scores), "Brier inputs must have equal length")
    require(len(outcomes) > 0, "Brier inputs must be nonempty")
    squared_errors = tuple(
        (score - outcome) ** 2 for outcome, score in zip(outcomes, scores)
    )
    return sum(squared_errors, Fraction(0)) / len(squared_errors)


def ranking_wins(
    outcomes: tuple[int, ...], scores: tuple[Fraction, ...]
) -> tuple[int, int]:
    negative_scores = tuple(
        score for outcome, score in zip(outcomes, scores) if outcome == 0
    )
    positive_scores = tuple(
        score for outcome, score in zip(outcomes, scores) if outcome == 1
    )
    comparisons = tuple(
        positive > negative
        for positive in positive_scores
        for negative in negative_scores
    )
    return sum(comparisons), len(comparisons)


def threshold_decisions(scores: tuple[Fraction, ...]) -> tuple[int, ...]:
    return tuple(int(score >= DECISION_THRESHOLD) for score in scores)


def expanded_paths() -> tuple[tuple[bool, bool], ...]:
    paths: list[tuple[bool, bool]] = []
    for path in product((True, False), repeat=2):
        paths.extend((path,) * PATH_MASSES[path])
    return tuple(paths)


def stop_at_first_miss(path: tuple[bool, bool]) -> bool:
    return path[0] and path[1]


def bounded_time_uniform_path(path: tuple[bool, bool]) -> tuple[bool, bool]:
    """Frozen two-look control: both looks inherit the first-look coverage."""
    return path[0], path[0]


@dataclass(frozen=True)
class SiteFixture:
    site_id: str
    abstention_code: str
    workflow_declared: bool
    operator_qualified: bool
    override_available: bool
    stop_path_available: bool
    destination_present: bool
    required_handoffs: int
    destination_capacity: int
    acknowledgements: int
    response_time_limit_minutes: int
    response_time_observed_minutes: int
    unresolved_case_policy: bool


def local_deployment_ready(site: SiteFixture) -> bool:
    return (
        site.workflow_declared
        and site.operator_qualified
        and site.override_available
        and site.stop_path_available
    )


def safe_deferral(site: SiteFixture) -> bool:
    return (
        site.destination_present
        and site.destination_capacity >= site.required_handoffs
        and site.acknowledgements >= site.required_handoffs
        and site.response_time_observed_minutes
        <= site.response_time_limit_minutes
        and site.unresolved_case_policy
    )


@dataclass(frozen=True)
class ChangeFixture:
    name: str
    performance_signature: tuple[int, int]
    modification_described: bool
    protocol_followed: bool
    acceptance_criteria_met: bool
    impact_assessed: bool
    declared_impact_assessment_id: int
    observed_impact_receipt_id: int


def change_authorized(change: ChangeFixture) -> bool:
    return (
        change.modification_described
        and change.protocol_followed
        and change.acceptance_criteria_met
        and change.impact_assessed
        and change.declared_impact_assessment_id
        == change.observed_impact_receipt_id
    )


@dataclass(frozen=True)
class LocalSpendingTrace:
    capacity: int
    ledger_epoch: int = 1
    spent: int = 0
    nonces: frozenset[int] = frozenset()


def attempt_spend(
    ledger: LocalSpendingTrace, amount: int, nonce: int
) -> tuple[LocalSpendingTrace, str]:
    require(amount > 0, "spend amount must be positive")
    if nonce in ledger.nonces:
        return ledger, "reuse_refused"
    if ledger.spent + amount > ledger.capacity:
        return ledger, "overspend_refused"
    return (
        replace(
            ledger,
            spent=ledger.spent + amount,
            nonces=ledger.nonces | {nonce},
        ),
        "accepted",
    )


@dataclass(frozen=True)
class LeaseFacet:
    lease_id: int
    spend_nonce: int
    ledger_epoch: int
    epoch: int
    status: str


def advance_epoch(
    facet: LeaseFacet, next_epoch: int, revoked_lease_ids: frozenset[int]
) -> LeaseFacet:
    require(next_epoch == facet.epoch + 1, "epochs must advance by exactly one")
    revoked = facet.status == "revoked" or facet.lease_id in revoked_lease_ids
    return LeaseFacet(
        lease_id=facet.lease_id,
        spend_nonce=facet.spend_nonce,
        ledger_epoch=facet.ledger_epoch,
        epoch=next_epoch,
        status="revoked" if revoked else "live",
    )


def main() -> None:
    scores_a = probabilities(SCORES_A_PERMILLE)
    scores_b = probabilities(SCORES_B_PERMILLE)
    brier_a = brier_score(OUTCOMES, scores_a)
    brier_b = brier_score(OUTCOMES, scores_b)
    rank_a = ranking_wins(OUTCOMES, scores_a)
    rank_b = ranking_wins(OUTCOMES, scores_b)
    decisions_a = threshold_decisions(scores_a)
    decisions_b = threshold_decisions(scores_b)
    require(brier_a == Fraction(1, 40), "unexpected Brier score for A")
    require(brier_b == Fraction(29, 160), "unexpected Brier score for B")
    require(rank_a == rank_b == (4, 4), "ranking must be perfectly tied")
    require(
        decisions_a == decisions_b == OUTCOMES,
        "threshold decisions must be identical and perfect",
    )

    paths = expanded_paths()
    require(len(paths) == 16, "unexpected two-look path mass")
    look_one_covered = sum(path[0] for path in paths)
    look_two_covered = sum(path[1] for path in paths)
    stopped_covered = sum(stop_at_first_miss(path) for path in paths)
    require(
        (look_one_covered, look_two_covered, stopped_covered) == (12, 12, 9),
        "fixed-horizon/stopping contest changed",
    )
    require(
        Fraction(look_one_covered, len(paths))
        == Fraction(look_two_covered, len(paths))
        == Fraction(3, 4),
        "fixed-horizon margins must each be three quarters",
    )
    require(
        Fraction(stopped_covered, len(paths)) == Fraction(9, 16),
        "stopped fixed-horizon coverage changed",
    )

    uniform_paths = tuple(bounded_time_uniform_path(path) for path in paths)
    simultaneous_covered = sum(all(path) for path in uniform_paths)
    uniform_stopped_covered = sum(
        stop_at_first_miss(path) for path in uniform_paths
    )
    require(
        (simultaneous_covered, uniform_stopped_covered) == (12, 12),
        "bounded time-uniform contest changed",
    )
    require(
        Fraction(simultaneous_covered, len(paths))
        == Fraction(uniform_stopped_covered, len(paths))
        == Fraction(3, 4),
        "bounded time-uniform rates changed",
    )

    fair_tree = tuple(product(("H", "T"), repeat=2))
    fair_probability = Fraction(1, len(fair_tree))
    require(fair_probability == Fraction(1, 4), "fair-tree mass changed")
    e0 = Fraction(1)
    e1_values = tuple(
        Fraction(2) if path[0] == "H" else Fraction(0) for path in fair_tree
    )
    e2_values = tuple(
        Fraction(4) if path == ("H", "H") else Fraction(0)
        for path in fair_tree
    )
    stopped_values = tuple(
        e1_value if path[0] == "H" else e2_value
        for path, e1_value, e2_value in zip(fair_tree, e1_values, e2_values)
    )
    expected_e1 = sum(e1_values, Fraction(0)) * fair_probability
    expected_e2 = sum(e2_values, Fraction(0)) * fair_probability
    expected_stopped = sum(stopped_values, Fraction(0)) * fair_probability
    require(
        e0 == expected_e1 == expected_e2 == expected_stopped == Fraction(1),
        "fair-tree e-process expectations changed",
    )
    require(
        (e2_values[0] + e2_values[1]) / 2 == e1_values[0]
        and (e2_values[2] + e2_values[3]) / 2 == e1_values[2],
        "fair-tree conditional expectation check failed",
    )

    site_a = SiteFixture(
        site_id="A",
        abstention_code="model_abstention",
        workflow_declared=True,
        operator_qualified=True,
        override_available=True,
        stop_path_available=True,
        destination_present=True,
        required_handoffs=2,
        destination_capacity=2,
        acknowledgements=2,
        response_time_limit_minutes=30,
        response_time_observed_minutes=20,
        unresolved_case_policy=True,
    )
    site_b = SiteFixture(
        site_id="B",
        abstention_code="model_abstention",
        workflow_declared=False,
        operator_qualified=False,
        override_available=False,
        stop_path_available=False,
        destination_present=True,
        required_handoffs=2,
        destination_capacity=1,
        acknowledgements=0,
        response_time_limit_minutes=30,
        response_time_observed_minutes=45,
        unresolved_case_policy=False,
    )
    require(site_a.abstention_code == site_b.abstention_code, "abstentions differ")
    require(local_deployment_ready(site_a), "site A must be locally ready")
    require(not local_deployment_ready(site_b), "site B must be quarantined")
    require(safe_deferral(site_a), "site A deferral must be safe")
    require(not safe_deferral(site_b), "site B deferral must be unsafe")

    performance_signature = (4, 4)
    authorized_change = ChangeFixture(
        name="authorized",
        performance_signature=performance_signature,
        modification_described=True,
        protocol_followed=True,
        acceptance_criteria_met=True,
        impact_assessed=True,
        declared_impact_assessment_id=30671,
        observed_impact_receipt_id=30671,
    )
    out_of_protocol_change = replace(
        authorized_change,
        name="out_of_protocol",
        protocol_followed=False,
    )
    authorization_truth_table = tuple(
        change_authorized(
            ChangeFixture(
                name="enumerated",
                performance_signature=performance_signature,
                modification_described=description,
                protocol_followed=protocol,
                acceptance_criteria_met=acceptance,
                impact_assessed=impact,
                declared_impact_assessment_id=30671,
                observed_impact_receipt_id=30671,
            )
        )
        for description, protocol, acceptance, impact in product(
            (False, True), repeat=4
        )
    )
    require(
        authorized_change.performance_signature
        == out_of_protocol_change.performance_signature,
        "change-control metric collision failed",
    )
    require(change_authorized(authorized_change), "declared change must pass")
    require(
        not change_authorized(out_of_protocol_change),
        "out-of-protocol change must be quarantined",
    )
    require(
        sum(authorization_truth_table) == 1,
        "change-control truth table must authorize only one combination",
    )

    drift_categories = (
        "input_distribution_shift",
        "performance_drift",
        "calibration_drift",
    )
    no_detected_shift = "no_detected_shift"
    no_shift = "no_shift"
    brier_delta = brier_b - brier_a
    require(len(set(drift_categories)) == 3, "drift categories collapsed")
    require(no_detected_shift != no_shift, "absence observations collapsed")
    require(brier_delta == Fraction(5, 32), "unexpected Brier delta")
    require(rank_a == rank_b, "rank control changed across calibration profiles")

    ledger0 = LocalSpendingTrace(capacity=100)
    ledger1, spend40 = attempt_spend(ledger0, amount=40, nonce=30811)
    ledger2, spend60 = attempt_spend(ledger1, amount=60, nonce=30812)
    ledger_reuse, repeat_status = attempt_spend(
        ledger2, amount=60, nonce=30812
    )
    ledger_overspend, overspend_status = attempt_spend(
        ledger2, amount=10, nonce=30813
    )
    require((spend40, spend60) == ("accepted", "accepted"), "valid spends failed")
    require(ledger2.spent == 100, "ledger must spend exactly 40 + 60")
    require(ledger2.capacity - ledger2.spent == 0, "ledger remainder changed")
    require(ledger2.ledger_epoch == 1, "local trace epoch changed")
    require(
        ledger2.nonces == frozenset({30811, 30812}),
        "local trace nonces changed",
    )
    require(repeat_status == "reuse_refused", "nonce reuse was not refused")
    require(overspend_status == "overspend_refused", "overspend was not refused")
    require(ledger_reuse == ledger2 == ledger_overspend, "refusals mutated ledger")

    epoch_one = LeaseFacet(
        lease_id=9001,
        spend_nonce=30812,
        ledger_epoch=1,
        epoch=1,
        status="live",
    )
    epoch_two = advance_epoch(epoch_one, 2, frozenset({9001}))
    epoch_three = advance_epoch(epoch_two, 3, frozenset())
    require(epoch_one.status == "live", "epoch-one lease must be live")
    require(
        epoch_one.spend_nonce == 30812
        and epoch_one.ledger_epoch == epoch_one.epoch,
        "lease did not preserve the local spend trace",
    )
    require(epoch_two.status == "revoked", "epoch-two lease must be revoked")
    require(epoch_three.status == "revoked", "revocation must remain sticky")

    print(
        "ORACLE_D10_W0 brier_a=1/40 brier_b=29/160 "
        "ranking_a=4/4 ranking_b=4/4 decisions=0,0,1,1 threshold=1/2"
    )
    print(
        "ORACLE_D10_W1 fixed_look1=12/16 fixed_look2=12/16 "
        "stopped=9/16 path_masses=9,3,3,1"
    )
    print(
        "ORACLE_D10_W2 bounded_time_uniform_simultaneous=12/16 "
        "stopped=12/16 general_theorem=false"
    )
    print("ORACLE_D10_W3 e0=1 expected_e1=1 expected_e2=1 expected_stopped=1")
    print(
        "ORACLE_D10_W4 same_abstention=true site_a=ready,safe "
        "site_b=quarantined,unsafe capacity_a=2 capacity_b=1 ack_a=2 ack_b=0"
    )
    print(
        "ORACLE_D10_W5 metrics_equal=true authorized=true "
        "out_of_protocol=quarantined authorization_truth_table=1/16"
    )
    print(
        "ORACLE_D10_W6 drift_categories=input,performance,calibration "
        "distinct=true brier_delta=5/32 no_detected_is_no_shift=false"
    )
    print(
        "ORACLE_D10_W7 ledger=40+60=100 capacity=100 remaining=0 "
        "repeat=reuse_refused overspend10=overspend_refused"
    )
    print(
        "ORACLE_D10_W8 epoch1=live epoch2=revoked epoch3=revoked "
        "old_facet_statically_invalidated=false"
    )
    print("PROOF-CARRYING DEPLOYMENT VALIDITY AND REVOCABLE AUTHORITY D10 ORACLE PASS")


if __name__ == "__main__":
    main()
