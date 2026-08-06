#!/usr/bin/env python3
"""Fail-closed verifier for the predictor-centered event-time receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-centered.v1"
EXPECTED_CENTER = Fraction(
    -229481176335118857750998868969216857385473740087351435274360730211439744304527,
    29642774844752946028434172162224104410437116074403984394101141506025761187823616,
)
EXPECTED_PATH = [
    "DOWN_RHO0L",
    "DOWN_ETAH",
    "DOWN_RHO0L",
    "DOWN_ETAL",
    "DOWN_RHO0L",
    "DOWN_ETAL",
    "DOWN_RHO0L",
    "DOWN_RHO1H",
    "DOWN_ETAH",
    "DOWN_RHO0L",
    "DOWN_RHO1H",
    "DOWN_ETAH",
]
EXPECTED_POWERS = list(range(18, 6, -1))


def fail(message: str) -> None:
    raise SystemExit(f"event-centered verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{label} is not an interval")
    lower, upper = (Fraction(str(item)) for item in value)
    if lower > upper:
        fail(f"{label} has reversed endpoints")
    return lower, upper


def positive_weights(value: object, label: str) -> None:
    if not isinstance(value, list) or len(value) != 6:
        fail(f"{label} does not contain six weights")
    for index, item in enumerate(value):
        _lower, upper = interval(item, f"{label}[{index}]")
        if upper <= 0:
            fail(f"{label}[{index}] is not positive")


def validate_chart(chart: object) -> bool:
    if not isinstance(chart, dict):
        fail("centered event chart is absent")
    if Fraction(str(chart.get("fixed_shift_q"))) != EXPECTED_CENTER:
        fail("centered chart used the wrong fixed shift")
    scales = chart.get("scales")
    if not isinstance(scales, list):
        fail("centered chart scales are absent")
    powers = [item.get("power") for item in scales]
    accepted = chart.get("accepted") is True
    if accepted:
        if chart.get("status") != "ACCEPTED":
            fail("accepted chart has the wrong terminal status")
        accepted_power = chart.get("accepted_power")
        if powers != EXPECTED_POWERS[: len(powers)] or not powers:
            fail("accepted chart has a malformed scale prefix")
        if powers[-1] != accepted_power or scales[-1].get("status") != "ACCEPTED":
            fail("accepted chart summary does not match its final scale")
        if any(item.get("status") == "ACCEPTED" for item in scales[:-1]):
            fail("accepted chart continued past an earlier accepted scale")
    else:
        if not scales:
            if chart.get("status") != "CENTERED_FIXED_FLOW_REFUSED":
                fail("empty chart is not a classified fixed-flow refusal")
            if not isinstance(chart.get("failure_class"), str) or not isinstance(
                chart.get("detail"), str
            ):
                fail("fixed-flow refusal lacks its failure class or detail")
            return False
        if powers != EXPECTED_POWERS:
            fail("refused chart did not exhaust the frozen scale sequence")
        if chart.get("accepted_power") is not None:
            fail("refused chart retained an accepted power")
        if chart.get("status") != "CENTERED_EVENT_CHART_REFUSED":
            fail("refused chart has the wrong terminal status")

    if "centered_variable_weights" in chart:
        positive_weights(
            chart["centered_variable_weights"], "centered variable weights"
        )
        if chart.get("centered_variables_preserved") is not True:
            fail("centered-state preservation summary is false")

    if accepted:
        final = scales[-1]
        radius = Fraction(str(final.get("radius_q")))
        event_lower, event_upper = interval(final.get("event_time"), "event time")
        if not -radius < event_lower <= event_upper < radius:
            fail("accepted event time is not strictly inside its residual slab")
        _derivative_lower, derivative_upper = interval(
            final.get("derivative"), "accepted derivative"
        )
        if derivative_upper >= 0:
            fail("accepted derivative is not strictly negative")
        domain_lower, domain_upper = interval(
            final.get("newton_domain"), "Newton domain"
        )
        image_lower, image_upper = interval(
            final.get("newton_image"), "Newton image"
        )
        if not domain_lower < image_lower <= image_upper < domain_upper:
            fail("Newton image is not strictly inside its a-priori domain")
        if final.get("newton_strict_inclusion") is not True:
            fail("Newton strict-inclusion summary is false")
        predictor_lower, predictor_upper = interval(
            final.get("predictor"), "centered predictor"
        )
        if not (
            -radius
            < predictor_lower + domain_lower
            <= predictor_upper + domain_upper
            < radius
        ):
            fail("predictor plus Newton domain escapes the residual Picard slab")
        if final.get("newton_domain_in_picard_slab") is not True:
            fail("Newton-domain Picard-slab summary is false")
        _normal_lower, normal_upper = interval(
            final.get("projected_normal"), "projected normal"
        )
        if normal_upper >= 0:
            fail("accepted projected normal is not strictly negative")
        if final.get("exact_section") is not True:
            fail("accepted projection is not exactly on w=0")
        if final.get("projected_variables_preserved") is not True:
            fail("accepted projection lost a symbolic variable")
        positive_weights(
            final.get("projected_variable_weights"),
            "projected variable weights",
        )
        if not isinstance(final.get("pure_source_monomials_retained"), int) or (
            final["pure_source_monomials_retained"] <= 0
        ):
            fail("accepted projection retained no pure source monomial")
        combined_lower, combined_upper = interval(
            final.get("combined_event_time"), "combined event time"
        )
        if (combined_lower, combined_upper) != (
            EXPECTED_CENTER + event_lower,
            EXPECTED_CENTER + event_upper,
        ):
            fail("combined event time is inconsistent with the centered chart")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--prior-worker", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    require(payload, "worker_source_sha256", sha256(args.worker))
    require(payload, "prior_worker_source_sha256", sha256(args.prior_worker))
    require(payload, "carrier_source_sha256", sha256(args.carrier))
    require(payload, "chain_source_sha256", sha256(args.chain))
    require(payload, "adaptive_source_sha256", sha256(args.adaptive))
    require(payload, "event_source_sha256", sha256(args.event))
    require(payload, "base_source_sha256", sha256(args.base))
    require(payload, "prior_receipt_sha256", sha256(args.prior_receipt))
    require(payload, "tile_id", "XLEL")
    require(payload, "critical_path", EXPECTED_PATH)
    require(payload, "critical_depth", len(EXPECTED_PATH))
    require(payload, "source_degree", 2)
    require(payload, "time_taylor_order", 12)
    require(payload, "arb_precision_bits", 256)
    require(
        payload,
        "reconditioner",
        "cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker."
        "point_coefficient_recondition",
    )
    require(payload, "all_six_symbolic_variables_required", True)
    require(payload, "diagnostic_complete", True)
    require(payload, "point_fallback_used", False)
    require(payload, "box_flattening_used", False)
    require(payload, "full_transport_attempted", False)
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    if Fraction(str(payload.get("predictor_center_q"))) != EXPECTED_CENTER:
        fail("receipt predictor center does not match the frozen exact center")
    predictor_lower, predictor_upper = interval(
        payload.get("predictor_range"), "original predictor"
    )
    boundary = Fraction(-1, 128)
    if not predictor_lower < boundary < predictor_upper:
        fail("original predictor does not reproduce the lower-boundary escape")
    positive_weights(payload.get("critical_variable_weights"), "critical weights")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks]
    if len(names) != len(set(names)):
        fail("implementation check names are not unique")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    chart_accepted = validate_chart(payload.get("centered_event_chart"))
    accepted = checks_passed and chart_accepted
    require(payload, "predictor_centered_event_accepted", accepted)
    expected_classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not checks_passed
        else (
            "PREDICTOR_CENTERED_EVENT_ACCEPTED"
            if accepted
            else "PREDICTOR_CENTERED_EVENT_REFUSED"
        )
    )
    require(payload, "classification", expected_classification)

    print(f"SCHEMA={SCHEMA}")
    print(f"CLASSIFICATION={expected_classification}")
    print(f"IMPLEMENTATION_CHECKS_PASSED={str(checks_passed).lower()}")
    print(f"PREDICTOR_CENTERED_EVENT_ACCEPTED={str(accepted).lower()}")
    print("ALL_SIX_SYMBOLIC_VARIABLES_REQUIRED=true")
    print("FULL_TRANSPORT_ATTEMPTED=false")
    print("COVERING_RELATION_CERTIFIED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
