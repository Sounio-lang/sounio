#!/usr/bin/env python3
"""Validate a predictor-centered event-time chart on the critical XLEL leaf."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_event_local_diagnostic_worker as prior


carrier = prior.carrier
base = prior.base
adaptive = prior.adaptive
chain = prior.chain
event = prior.event

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-centered.v1"
TILE_ID = prior.TILE_ID
CRITICAL_PATH = prior.CRITICAL_PATH
SLAB_POWERS = prior.SLAB_POWERS
ANCHOR_RADIUS = Fraction(1, 128)
EXPECTED_CENTER = Fraction(
    -229481176335118857750998868969216857385473740087351435274360730211439744304527,
    29642774844752946028434172162224104410437116074403984394101141506025761187823616,
)
EXPECTED_PRIOR_RECEIPT_SHA256 = (
    "0a47c711f8442bfe4bb3ce844cb247c74aaf9922f5dcf224411f58acd2bb3146"
)
STATE_ROWS = (0, 1, 2, 3)
# The event projection sets w (row 2) exactly to zero by construction.
SECTION_ROWS = (0, 1, 3)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def variable_weights(
    state: list[base.TM2R], rows: tuple[int, ...] = STATE_ROWS
) -> list[arb]:
    weights: list[arb] = []
    for variable in range(base.VARIABLES):
        weight = arb(0)
        for row in rows:
            for monomial, coefficient in state[row].coefficients.items():
                exponent = monomial[variable]
                if exponent:
                    weight += base.upper_abs(coefficient) * exponent
        weights.append(weight)
    return weights


def model_variable_weights(model: base.TM2R) -> list[arb]:
    weights: list[arb] = []
    for variable in range(base.VARIABLES):
        weight = arb(0)
        for monomial, coefficient in model.coefficients.items():
            exponent = monomial[variable]
            if exponent:
                weight += base.upper_abs(coefficient) * exponent
        weights.append(weight)
    return weights


def retained_source_monomials(state: list[base.TM2R]) -> int:
    return sum(
        1
        for row in SECTION_ROWS
        for monomial, coefficient in state[row].coefficients.items()
        if event.pure_source_nonconstant(monomial)
        and (coefficient.lower() != 0 or coefficient.upper() != 0)
    )


def critical_state(
    checks: list[dict[str, object]],
) -> tuple[list[base.TM2R], object, int, int, int]:
    tiles, source_split_checks = carrier.source_tiles()
    state, _domain = tiles[TILE_ID]
    first = event.integrate_positive_return(state)
    first_projection = event.interval_newton_project(first)
    approach = chain.integrate_downward_return(first_projection.carrier)
    state = approach.endpoint
    critical_split_checks = 0
    for depth, token in enumerate(CRITICAL_PATH):
        body = token.removeprefix("DOWN_")
        expected_name, side_token = body[:-1], body[-1]
        variable, _weight = adaptive.dominant_variable(state)
        bool_check(
            checks,
            f"split_{depth + 1}_dominant_variable_matches_frozen_path",
            adaptive.VARIABLE_NAMES[variable] == expected_name,
        )
        left, right, reconstruction_checks = adaptive.split_state(state, variable)
        critical_split_checks += reconstruction_checks
        state = left if side_token == "L" else right
    return (
        state,
        approach,
        first.end_step,
        source_split_checks,
        critical_split_checks,
    )


def frozen_predictor(
    state: list[base.TM2R], checks: list[dict[str, object]]
) -> tuple[base.TM2R, arb, Fraction, list[arb], arb, dict[str, object]]:
    ranges = [component.range() for component in state]
    radius = base.rational_ball(ANCHOR_RADIUS)
    backward, backward_iterations, backward_contraction = event.signed_picard_box(
        ranges, -radius
    )
    forward, forward_iterations, forward_contraction = event.signed_picard_box(
        ranges, radius
    )
    tube = [
        left.union(right)
        for left, right in zip(backward, forward, strict=True)
    ]
    derivative = tube[0] * tube[1] - tube[2] - base.ZS
    predictor = -state[2] / derivative.mid()
    predictor_range = predictor.range()
    center = Fraction(base.exact_fraction(predictor_range.mid()))
    bool_check(checks, "anchor_derivative_strictly_negative", derivative.upper() < 0)
    bool_check(checks, "predictor_center_matches_frozen_receipt", center == EXPECTED_CENTER)
    bool_check(
        checks,
        "predictor_center_is_strictly_inside_anchor_slab",
        -ANCHOR_RADIUS < center < ANCHOR_RADIUS,
    )
    bool_check(
        checks,
        "predictor_straddles_anchor_lower_boundary",
        predictor_range.lower() < -radius < predictor_range.upper(),
    )
    details = {
        "anchor_radius_q": str(ANCHOR_RADIUS),
        "predictor": interval_json(predictor_range),
        "predictor_width": interval_json(base.width(predictor_range)),
        "predictor_center_q": str(center),
        "derivative": interval_json(derivative),
        "backward_picard_iterations": backward_iterations,
        "forward_picard_iterations": forward_iterations,
        "backward_picard_contraction": interval_json(backward_contraction),
        "forward_picard_contraction": interval_json(forward_contraction),
    }
    return predictor, predictor_range, center, tube, derivative, details


def centered_event_chart(
    state: list[base.TM2R], center: Fraction
) -> dict[str, object]:
    result: dict[str, object] = {
        "accepted": False,
        "fixed_shift_q": str(center),
        "scales": [],
    }
    try:
        centered_state, fixed_iterations, fixed_contraction = event.fixed_time_flow(
            state, base.rational_ball(center)
        )
    except base.Refusal as refusal:
        result.update(
            status="CENTERED_FIXED_FLOW_REFUSED",
            failure_class=refusal.failure_class,
            detail=refusal.detail,
        )
        return result

    centered_ranges = [component.range() for component in centered_state]
    centered_weights = variable_weights(centered_state)
    result.update(
        fixed_picard_iterations=fixed_iterations,
        fixed_picard_contraction=interval_json(fixed_contraction),
        centered_state=prior.state_metrics(centered_state),
        centered_variable_weights=[interval_json(value) for value in centered_weights],
        centered_variables_preserved=all(value.upper() > 0 for value in centered_weights),
    )

    scales: list[dict[str, object]] = []
    for power in SLAB_POWERS:
        radius_q = Fraction(1, 2**power)
        radius = base.rational_ball(radius_q)
        record: dict[str, object] = {"power": power, "radius_q": str(radius_q)}
        try:
            backward, backward_iterations, backward_contraction = (
                event.signed_picard_box(centered_ranges, -radius)
            )
            forward, forward_iterations, forward_contraction = (
                event.signed_picard_box(centered_ranges, radius)
            )
        except base.Refusal as refusal:
            record.update(status=refusal.failure_class, detail=refusal.detail)
            scales.append(record)
            continue
        tube = [
            left.union(right)
            for left, right in zip(backward, forward, strict=True)
        ]
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        record.update(
            backward_picard_iterations=backward_iterations,
            forward_picard_iterations=forward_iterations,
            backward_picard_contraction=interval_json(backward_contraction),
            forward_picard_contraction=interval_json(forward_contraction),
            derivative=interval_json(derivative),
        )
        if derivative.upper() >= 0:
            record["status"] = "DERIVATIVE_SIGN_UNRESOLVED"
            scales.append(record)
            continue

        predictor = -centered_state[2] / derivative.mid()
        predictor_range = predictor.range()
        record.update(
            predictor=interval_json(predictor_range),
            predictor_width=interval_json(base.width(predictor_range)),
        )
        if predictor_range.lower() <= -radius or predictor_range.upper() >= radius:
            record["status"] = "CENTERED_PREDICTOR_ESCAPED"
            scales.append(record)
            continue

        predicted_state = chain.variable_time_flow(centered_state, predictor, tube)
        predictor_lower_q = Fraction(base.lower_fraction(predictor_range))
        predictor_upper_q = Fraction(base.upper_fraction(predictor_range))
        safe_lower_q = -radius_q - predictor_lower_q
        safe_upper_q = radius_q - predictor_upper_q
        if safe_lower_q >= 0 or safe_upper_q <= 0:
            record["status"] = "CENTERED_NEWTON_DOMAIN_EMPTY"
            scales.append(record)
            continue
        # This fixed residual domain keeps predictor(P) + domain strictly
        # inside the symmetric Picard slab. The factor 1/2 supplies an exact
        # rational margin on both sides for the interval-Newton inclusion.
        newton_domain_lower_q = safe_lower_q / 2
        newton_domain_upper_q = safe_upper_q / 2
        newton_domain = base.rational_ball(newton_domain_lower_q).union(
            base.rational_ball(newton_domain_upper_q)
        )
        newton_domain_in_picard_slab = (
            predictor_range.lower() + newton_domain.lower() > -radius
            and predictor_range.upper() + newton_domain.upper() < radius
        )
        if not newton_domain_in_picard_slab:
            record.update(
                newton_domain=interval_json(newton_domain),
                newton_domain_in_picard_slab=False,
                status="CENTERED_NEWTON_DOMAIN_ESCAPED_PICARD_SLAB",
            )
            scales.append(record)
            continue
        newton_image = -predicted_state[2].range() / derivative
        newton_strict_inclusion = (
            newton_image.lower() > newton_domain.lower()
            and newton_image.upper() < newton_domain.upper()
        )
        record.update(
            newton_domain=interval_json(newton_domain),
            newton_domain_in_picard_slab=True,
            newton_image=interval_json(newton_image),
            newton_strict_inclusion=newton_strict_inclusion,
        )
        if not newton_strict_inclusion:
            record["status"] = "CENTERED_NEWTON_NOT_SELF_MAPPING"
            scales.append(record)
            continue

        event_time_model = predictor.with_remainder(newton_image)
        event_time_range = event_time_model.range()
        record.update(
            correction=interval_json(newton_image),
            event_time=interval_json(event_time_range),
            combined_event_time=interval_json(
                base.rational_ball(center) + event_time_range
            ),
            event_time_variable_weights=[
                interval_json(value)
                for value in model_variable_weights(event_time_model)
            ],
        )
        if not (
            event_time_range.lower() > -radius
            and event_time_range.upper() < radius
        ):
            record["status"] = "CENTERED_NEWTON_ESCAPED"
            scales.append(record)
            continue

        event_state = chain.variable_time_flow(centered_state, event_time_model, tube)
        raw_projection = [
            component if row != 2 else base.TM2R.constant(0)
            for row, component in enumerate(event_state)
        ]
        projected = base.recondition(raw_projection)
        projected_ranges = [component.range() for component in projected]
        normal = projected_ranges[0] * projected_ranges[1] - base.ZS
        projected_weights = variable_weights(projected, rows=SECTION_ROWS)
        exact_section = (
            projected_ranges[2].lower() == 0
            and projected_ranges[2].upper() == 0
        )
        variables_preserved = all(value.upper() > 0 for value in projected_weights)
        retained = retained_source_monomials(projected)
        record.update(
            projected_state=prior.state_metrics(projected),
            projected_normal=interval_json(normal),
            projected_variable_weights=[
                interval_json(value) for value in projected_weights
            ],
            projected_variables_preserved=variables_preserved,
            pure_source_monomials_retained=retained,
            exact_section=exact_section,
        )
        if not exact_section:
            record["status"] = "CENTERED_SECTION_PROJECTION_DRIFT"
            scales.append(record)
            continue
        if normal.upper() >= 0:
            record["status"] = "CENTERED_TRANSVERSALITY_UNRESOLVED"
            scales.append(record)
            continue
        if not variables_preserved or retained == 0:
            record["status"] = "CENTERED_SYMBOLIC_DEPENDENCE_LOST"
            scales.append(record)
            continue

        record["status"] = "ACCEPTED"
        scales.append(record)
        result.update(
            accepted=True,
            accepted_power=power,
            status="ACCEPTED",
            scales=scales,
        )
        return result

    result.update(
        accepted_power=None,
        status="CENTERED_EVENT_CHART_REFUSED",
        scales=scales,
    )
    return result


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("event-centered diagnostic requires Python >= 3.10")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    event.MAX_PHASE_STEPS = carrier.MAX_FIRST_RETURN_STEPS

    source_path = Path(__file__)
    receipt_dir = source_path.parent / "receipts" / (
        "cs6_v7b_target23_arb_tm2r_event_local_v1"
    )
    prior_receipt = receipt_dir / "event_local_diagnostic.json"
    if not prior_receipt.is_file():
        raise SystemExit("the frozen event-local receipt is missing")

    checks: list[dict[str, object]] = []
    bool_check(
        checks,
        "prior_receipt_hash_matches_frozen_diagnostic",
        sha256(prior_receipt) == EXPECTED_PRIOR_RECEIPT_SHA256,
    )
    state, approach, first_end_step, source_checks, critical_checks = (
        critical_state(checks)
    )
    raw_weights = variable_weights(state)
    bool_check(
        checks,
        "critical_state_preserves_all_six_variables",
        all(value.upper() > 0 for value in raw_weights),
    )
    predictor, predictor_range, center, _tube, _derivative, anchor = (
        frozen_predictor(state, checks)
    )
    bool_check(
        checks,
        "predictor_model_preserves_all_six_variables",
        all(value.upper() > 0 for value in model_variable_weights(predictor)),
    )
    centered = centered_event_chart(state, center)
    if "centered_variables_preserved" in centered:
        bool_check(
            checks,
            "centered_state_preserves_all_six_variables",
            bool(centered["centered_variables_preserved"]),
        )
    if centered.get("accepted") is True:
        accepted = centered["scales"][-1]
        bool_check(
            checks,
            "accepted_projection_preserves_all_six_variables",
            bool(accepted["projected_variables_preserved"]),
        )
        bool_check(
            checks,
            "accepted_event_time_is_strictly_inside_residual_slab",
            accepted["status"] == "ACCEPTED",
        )

    implementation_ok = all(bool(item["passed"]) for item in checks)
    accepted = implementation_ok and centered.get("accepted") is True
    classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not implementation_ok
        else (
            "PREDICTOR_CENTERED_EVENT_ACCEPTED"
            if accepted
            else "PREDICTOR_CENTERED_EVENT_REFUSED"
        )
    )
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "prior_worker_source_sha256": sha256(Path(prior.__file__)),
        "carrier_source_sha256": sha256(Path(carrier.__file__)),
        "chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "prior_receipt_sha256": sha256(prior_receipt),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_degree": base.SOURCE_DEGREE,
        "time_taylor_order": base.TIME_TAYLOR_ORDER,
        "reconditioner": (
            f"{base.recondition.__module__}.{base.recondition.__qualname__}"
        ),
        "tile_id": TILE_ID,
        "critical_path": list(CRITICAL_PATH),
        "critical_depth": len(CRITICAL_PATH),
        "first_return_end_step": first_end_step,
        "downward_reference_time_q": str(approach.reference_time),
        "source_split_reconstruction_checks": source_checks,
        "critical_split_reconstruction_checks": critical_checks,
        "critical_state": prior.state_metrics(state),
        "critical_variable_weights": [interval_json(value) for value in raw_weights],
        "anchor": anchor,
        "predictor_range": interval_json(predictor_range),
        "predictor_center_q": str(center),
        "centered_event_chart": centered,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "predictor_centered_event_accepted": accepted,
        "diagnostic_complete": True,
        "all_six_symbolic_variables_required": True,
        "point_fallback_used": False,
        "box_flattening_used": False,
        "full_transport_attempted": False,
        "covering_relation_certified": False,
        "recurrent_graph_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
