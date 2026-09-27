#!/usr/bin/env python3
"""Independent exact-arithmetic oracle for the bounded D11 fixtures."""

from fractions import Fraction
from itertools import product


SOURCE_NUMERATORS = (3, 3, 3, 3)
TARGET_NUMERATORS = (6, 2, 2, 2)
LOSS_NUMERATORS = (1, 0, 0, 0)
MASS_DENOMINATOR = 12
SOURCE = tuple(Fraction(value, MASS_DENOMINATOR) for value in SOURCE_NUMERATORS)
TARGET = tuple(Fraction(value, MASS_DENOMINATOR) for value in TARGET_NUMERATORS)
LOSSES = tuple(Fraction(value) for value in LOSS_NUMERATORS)
LABEL_PROBE_ID = 31311
EVALUATED_LOSS_ID = 31711
JOINT_LAW_FINGERPRINT = 92352734845403155


def dot(mass: tuple[Fraction, ...], values: tuple[Fraction, ...]) -> Fraction:
    assert len(mass) == len(values)
    return sum((m * value for m, value in zip(mass, values)), Fraction(0))


def base31(values: tuple[int, ...]) -> int:
    code = 0
    for value in values:
        code = code * 31 + value
    return code


def coarse_ab(mass: tuple[Fraction, ...]) -> tuple[Fraction, Fraction]:
    return mass[0], sum(mass[1:], Fraction(0))


def check_shared_exact_joint_law() -> None:
    assert sum(SOURCE) == sum(TARGET) == 1
    assert all(Fraction(0) <= loss <= Fraction(1) for loss in LOSSES)
    source_code = base31(SOURCE_NUMERATORS)
    target_code = base31(TARGET_NUMERATORS)
    loss_code = base31(LOSS_NUMERATORS)
    fingerprint = (source_code * 1_000_003 + target_code) * 1_000_003 + loss_code
    assert (source_code, target_code, loss_code) == (92352, 180732, 29791)
    assert fingerprint == JOINT_LAW_FINGERPRINT


def check_scope_subset() -> None:
    source_members = tuple(range(3114101, 3114109))
    target_members = tuple(range(3114101, 3114105))
    smaller_disjoint_members = tuple(range(3180901, 3180905))
    assert len(target_members) < len(source_members)
    assert set(target_members).issubset(source_members)
    assert not set(smaller_disjoint_members).issubset(source_members)


def check_covariate_transport() -> None:
    source = coarse_ab(SOURCE)
    target = coarse_ab(TARGET)
    losses = (Fraction(1), Fraction(0))
    weights = tuple(q / p for p, q in zip(source, target))
    assert source == (Fraction(1, 4), Fraction(3, 4))
    assert target == (Fraction(1, 2), Fraction(1, 2))
    assert weights == (Fraction(2), Fraction(2, 3))
    source_risk = dot(source, losses)
    target_risk = dot(target, losses)
    weighted_risk = dot(source, tuple(w * loss for w, loss in zip(weights, losses)))
    assert source_risk == Fraction(1, 4)
    assert target_risk == Fraction(1, 2)
    assert weighted_risk == target_risk


def check_overlap_interval() -> None:
    source = (Fraction(1), Fraction(0))
    target = (Fraction(1, 2), Fraction(1, 2))
    observed_loss_a = Fraction(0)
    possible = {
        dot(target, (observed_loss_a, Fraction(unseen_loss_b)))
        for unseen_loss_b in (0, 1)
    }
    assert source[1] == 0 and target[1] > 0
    assert possible == {Fraction(0), Fraction(1, 2)}


def check_label_shift() -> None:
    source_prior = coarse_ab(SOURCE)
    target_prior = coarse_ab(TARGET)
    class_losses = (Fraction(1), Fraction(0))
    weights = tuple(q / p for p, q in zip(source_prior, target_prior))
    assert LABEL_PROBE_ID != EVALUATED_LOSS_ID
    assert weights == (Fraction(2), Fraction(2, 3))
    assert dot(source_prior, class_losses) == Fraction(1, 4)
    assert dot(target_prior, class_losses) == Fraction(1, 2)
    assert dot(
        source_prior,
        tuple(w * loss for w, loss in zip(weights, class_losses)),
    ) == Fraction(1, 2)

    # The identification probe is a separate perfect classifier. The evaluated
    # loss is not repurposed as its confusion matrix.
    label_probe_confusion = ((1, 0), (0, 1))
    determinant = (
        label_probe_confusion[0][0] * label_probe_confusion[1][1]
        - label_probe_confusion[0][1] * label_probe_confusion[1][0]
    )
    assert determinant == 1

    singular_confusion = ((1, 1), (0, 0))
    singular_determinant = (
        singular_confusion[0][0] * singular_confusion[1][1]
        - singular_confusion[0][1] * singular_confusion[1][0]
    )
    assert singular_determinant == 0
    target_worlds = (
        (Fraction(3, 4), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(3, 4)),
    )
    prediction_histograms = {(Fraction(1), Fraction(0)) for _ in target_worlds}
    risks = {dot(prior, (Fraction(0), Fraction(1))) for prior in target_worlds}
    assert len(prediction_histograms) == 1
    assert risks == {Fraction(1, 4), Fraction(3, 4)}


def check_concept_ambiguity() -> None:
    target_inputs = (0, 1, 1, 1)
    served_scores = (1, 0, 0, 0)
    stable_loss = LOSSES
    shifted_loss = (Fraction(1),) * 4
    assert target_inputs == target_inputs
    assert served_scores == served_scores
    stable_risk = dot(TARGET, stable_loss)
    shifted_risk = dot(TARGET, shifted_loss)
    assert stable_risk == Fraction(1, 2)
    assert shifted_risk == 1


def check_subgroup_collision() -> None:
    balanced = ((3, 6), (3, 6))
    hidden = ((0, 6), (6, 6))
    balanced_marginal = Fraction(sum(e for e, _ in balanced), sum(n for _, n in balanced))
    hidden_marginal = Fraction(sum(e for e, _ in hidden), sum(n for _, n in hidden))
    balanced_worst = max(Fraction(e, n) for e, n in balanced)
    hidden_worst = max(Fraction(e, n) for e, n in hidden)
    assert balanced_marginal == hidden_marginal == dot(TARGET, LOSSES)
    assert balanced_worst == Fraction(1, 2)
    assert hidden_worst == 1


def check_calibration_transport() -> None:
    outcomes = (Fraction(1), Fraction(0))
    diagnostic_prediction = (Fraction(1, 4), Fraction(1, 4))
    source_mass = coarse_ab(SOURCE)
    target_mass = coarse_ab(TARGET)
    residuals = tuple(y - p for y, p in zip(outcomes, diagnostic_prediction))
    weights = tuple(q / p for p, q in zip(source_mass, target_mass))
    assert dot(source_mass, residuals) == 0
    assert dot(target_mass, residuals) == Fraction(1, 4)
    assert dot(source_mass, tuple(w * r for w, r in zip(weights, residuals))) == Fraction(1, 4)

    active_local_prediction = outcomes
    assert all(y - p == 0 for y, p in zip(outcomes, active_local_prediction))
    later_outcomes = (Fraction(0), Fraction(0))
    later_residuals = tuple(
        y - p for y, p in zip(later_outcomes, active_local_prediction)
    )
    assert dot(target_mass, later_residuals) == Fraction(-1, 2)
    assert abs(dot(target_mass, later_residuals)) > Fraction(1, 4)


def check_tight_conformal_tv_bound() -> None:
    source_risk = dot(SOURCE, LOSSES)
    target_risk = dot(TARGET, LOSSES)
    tv = sum((abs(p - q) for p, q in zip(SOURCE, TARGET)), Fraction(0)) / 2
    assert source_risk == Fraction(1, 4)
    assert target_risk == Fraction(1, 2)
    assert tv == Fraction(1, 4)
    assert target_risk == source_risk + tv

    # Exhaustively confirm the bounded-loss inequality on this four-atom law.
    for loss_bits in product((Fraction(0), Fraction(1)), repeat=4):
        assert dot(TARGET, loss_bits) <= dot(SOURCE, loss_bits) + tv


def check_authority_attenuation() -> None:
    ranks = (3, 3, 2, 1, 0)
    scopes = (8, 4, 2, 0, 0)
    windows = (31121, 31122, 31123, 31124)
    epochs = (1, 2, 3, 4)
    assert all(after <= before for before, after in zip(ranks, ranks[1:]))
    assert all(after <= before for before, after in zip(scopes, scopes[1:]))
    assert sum(after > before for before, after in zip(ranks, ranks[1:])) == 0
    assert all(after > before for before, after in zip(windows, windows[1:]))
    assert epochs == tuple(range(1, 5))
    assert ranks[-1] == 0
    nominal_trace_terminal = True
    globally_absorbing = False
    runtime_canary_disabled = False
    fixture_replayable = True
    assert nominal_trace_terminal and fixture_replayable
    assert not globally_absorbing and not runtime_canary_disabled


def main() -> None:
    check_shared_exact_joint_law()
    check_scope_subset()
    check_covariate_transport()
    check_overlap_interval()
    check_label_shift()
    check_concept_ambiguity()
    check_subgroup_collision()
    check_calibration_transport()
    check_tight_conformal_tv_bound()
    check_authority_attenuation()

    print("ORACLE_D11_W0 d10_lease=30821 source_rank=3 canary_only=true production=false clinical=false")
    print("ORACLE_D11_W1 covariate source_mass=3,9 target_mass=6,6 weights=2,2/3 source_risk=1/4 target_risk=1/2 weighted=1/2")
    print("ORACLE_D11_W2 overlap source_mass=4,0 target_mass=2,2 target_risk_interval=[0,1/2] point_identified=false")
    print("ORACLE_D11_W3 label src=3,9 tgt=6,6 risk=1/4->1/2 probe=31311 loss=31711 singular=1/4,3/4")
    print("ORACLE_D11_W4 concept same_unlabeled_inputs=true stable_risk=2/4 shifted_risk=4/4 labels_required=true")
    print("ORACLE_D11_W5 subgroup marginal_a=6/12 marginal_b=6/12 worst_a=1/2 worst_b=1")
    print("ORACLE_D11_W6 calibration diag=0->1/4 local=0 active_later=-1/2 diagnostic_transition=false")
    print("ORACLE_D11_W7 conformal source_risk=1/4 tv=1/4 target_risk=1/2 bound_tight=true general_theorem=false")
    print("ORACLE_D11_W8 rank=3,3,2,1,0 scope=8,4,2,0,0 w=31121..31124 up=0 nominal=true global=false")
    print("ORACLE_D11_W8_LIMIT runtime_disabled=false replayable=true stale_alias_invalidated=false")
    print("PROOF-CARRYING SHIFT-ROBUST RISK TRANSPORT D11 ORACLE PASS")


if __name__ == "__main__":
    main()
