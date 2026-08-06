#!/usr/bin/env python3
"""Replay one critical event branch and separate three failure hypotheses."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_composability_carrier_worker as carrier


base = carrier.base
adaptive = carrier.adaptive
transport = carrier.transport
chain = transport.chain
event = transport.event

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-local-diagnostic.v1"
TILE_ID = "XLEL"
CRITICAL_PATH = (
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
)
SLAB_POWERS = tuple(range(18, 6, -1))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def same_interval(left: arb, right: arb) -> bool:
    return left.contains(right) and right.contains(left)


def local_slab_scan(state: list[base.TM2R]) -> dict[str, object]:
    """Run the production symmetric-slab criterion with full observability."""
    initial_ranges = [component.range() for component in state]
    scales: list[dict[str, object]] = []
    accepted_power: int | None = None
    for power in SLAB_POWERS:
        radius = Fraction(1, 2**power)
        record: dict[str, object] = {
            "power": power,
            "radius_q": str(radius),
        }
        try:
            backward_box, backward_iterations, backward_contraction = (
                event.signed_picard_box(
                    initial_ranges, base.rational_ball(-radius)
                )
            )
            forward_box, forward_iterations, forward_contraction = (
                event.signed_picard_box(
                    initial_ranges, base.rational_ball(radius)
                )
            )
        except base.Refusal as refusal:
            record.update(
                status=refusal.failure_class,
                detail=refusal.detail,
            )
            scales.append(record)
            continue
        tube = [
            backward.union(forward)
            for backward, forward in zip(
                backward_box, forward_box, strict=True
            )
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
        predictor = -state[2] / derivative.mid()
        predictor_range = predictor.range()
        radius_ball = base.rational_ball(radius)
        record.update(
            predictor=interval_json(predictor_range),
            predictor_width=interval_json(base.width(predictor_range)),
        )
        if (
            predictor_range.lower() <= -radius_ball
            or predictor_range.upper() >= radius_ball
        ):
            record["status"] = "PREDICTOR_ESCAPED"
            scales.append(record)
            continue
        predicted_state = chain.variable_time_flow(state, predictor, tube)
        correction = -predicted_state[2].range() / derivative
        event_time_model = predictor.with_remainder(correction)
        event_time_range = event_time_model.range()
        record["correction"] = interval_json(correction)
        record["event_time"] = interval_json(event_time_range)
        if (
            event_time_range.lower() > -radius_ball
            and event_time_range.upper() < radius_ball
        ):
            record["status"] = "ACCEPTED"
            scales.append(record)
            accepted_power = power
            break
        record["status"] = "NEWTON_ESCAPED"
        scales.append(record)
    return {
        "accepted": accepted_power is not None,
        "accepted_power": accepted_power,
        "scales": scales,
    }


def anchored_event_step_newton(
    state: list[base.TM2R],
    event_tube: list[arb],
) -> dict[str, object]:
    """Project using the already validated crossing step as the Newton anchor."""
    ranges = [component.range() for component in state]
    derivative = event_tube[0] * event_tube[1] - event_tube[2] - base.ZS
    result: dict[str, object] = {
        "accepted": False,
        "endpoint_w": interval_json(ranges[2]),
        "derivative": interval_json(derivative),
        "endpoint_in_crossing_tube": all(
            tube_component.contains(component)
            for tube_component, component in zip(event_tube, ranges, strict=True)
        ),
    }
    if ranges[2].upper() >= 0:
        result["status"] = "ENDPOINT_NOT_STRICTLY_NEGATIVE"
        return result
    if derivative.upper() >= 0:
        result["status"] = "DERIVATIVE_SIGN_UNRESOLVED"
        return result
    delta = -ranges[2] / derivative
    result["delta"] = interval_json(delta)
    step = base.rational_ball(Fraction(1, 2**8))
    if delta.lower() <= -step or delta.upper() >= 0:
        result["status"] = "ANCHORED_NEWTON_TIME_ESCAPED"
        return result
    try:
        slab_box, iterations, contraction = event.signed_picard_box(
            ranges, delta.lower()
        )
    except base.Refusal as refusal:
        result.update(status=refusal.failure_class, detail=refusal.detail)
        return result
    slab_contained = all(
        event_component.contains(slab_component)
        for event_component, slab_component in zip(
            event_tube, slab_box, strict=True
        )
    )
    result.update(
        slab_picard_iterations=iterations,
        slab_picard_contraction=interval_json(contraction),
        slab_in_crossing_tube=slab_contained,
    )
    if not slab_contained:
        result["status"] = "ANCHORED_SLAB_ESCAPES_CROSSING_TUBE"
        return result

    fixed_shift = delta.mid()
    residual_shift = delta - fixed_shift
    try:
        fixed_state, fixed_iterations, fixed_contraction = event.fixed_time_flow(
            state, fixed_shift
        )
    except base.Refusal as refusal:
        result.update(status=refusal.failure_class, detail=refusal.detail)
        return result
    vector_field = base.field_interval(event_tube)
    raw_projection = [
        base.TM2R.constant(0)
        if row == 2
        else component.with_remainder(vector_field[row] * residual_shift)
        for row, component in enumerate(fixed_state)
    ]
    projected = base.recondition(raw_projection)
    projected_ranges = [component.range() for component in projected]
    normal = projected_ranges[0] * projected_ranges[1] - base.ZS
    exact_section = (
        projected_ranges[2].lower() == 0
        and projected_ranges[2].upper() == 0
    )
    result.update(
        fixed_picard_iterations=fixed_iterations,
        fixed_picard_contraction=interval_json(fixed_contraction),
        residual_shift=interval_json(residual_shift),
        projected_normal=interval_json(normal),
        exact_section=exact_section,
    )
    if not exact_section:
        result["status"] = "ANCHORED_SECTION_PROJECTION_DRIFT"
        return result
    if normal.upper() >= 0:
        result["status"] = "ANCHORED_TRANSVERSALITY_UNRESOLVED"
        return result
    result.update(status="ACCEPTED", accepted=True)
    return result


def state_metrics(state: list[base.TM2R]) -> dict[str, object]:
    ranges = [component.range() for component in state]
    return {
        "ranges": [interval_json(value) for value in ranges],
        "widths": [interval_json(base.width(value)) for value in ranges],
        "max_width": interval_json(
            base.max_upper([base.width(value) for value in ranges])
        ),
    }


def classify(
    implementation_ok: bool,
    raw_accepted: bool,
    reconditioned_accepted: bool,
    anchored_accepted: bool,
) -> str:
    if not implementation_ok:
        return "IMPLEMENTATION_INCONSISTENCY"
    if raw_accepted:
        return "CURRENT_CRITERION_ACCEPTS"
    chart = reconditioned_accepted
    criterion = anchored_accepted
    if chart and criterion:
        return "MIXED_CHART_AND_EVENT_CRITERION"
    if chart:
        return "CHART_DRIFT"
    if criterion:
        return "EVENT_CRITERION"
    return "UNRESOLVED_ENCLOSURE"


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("event-local diagnostic requires Python >= 3.10")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    event.MAX_PHASE_STEPS = carrier.MAX_FIRST_RETURN_STEPS

    checks: list[dict[str, object]] = []
    tiles, source_split_checks = carrier.source_tiles()
    tile_state, _domain = tiles[TILE_ID]
    first = event.integrate_positive_return(tile_state)
    first_projection = event.interval_newton_project(first)
    approach = chain.integrate_downward_return(first_projection.carrier)

    endpoint_ranges = [component.range() for component in approach.endpoint]
    bool_check(
        checks,
        "approach_endpoint_in_crossing_tube",
        all(
            tube.contains(component)
            for tube, component in zip(
                approach.event_tube, endpoint_ranges, strict=True
            )
        ),
    )
    derivative_direct = (
        approach.event_tube[0] * approach.event_tube[1]
        - approach.event_tube[2]
        - base.ZS
    )
    derivative_field = base.field_interval(approach.event_tube)[2]
    bool_check(
        checks,
        "event_derivative_formula_identity",
        same_interval(derivative_direct, derivative_field),
    )
    bool_check(
        checks,
        "stored_crossing_derivative_identity",
        same_interval(derivative_direct, approach.derivative),
    )
    bool_check(
        checks,
        "crossing_tube_strictly_downward",
        derivative_direct.upper() < 0,
    )

    state = approach.endpoint
    prefixes: list[dict[str, object]] = []
    split_reconstruction_checks = 0
    for depth in range(len(CRITICAL_PATH) + 1):
        raw_scan = local_slab_scan(state)
        reconditioned = adaptive.point_coefficient_recondition(state)
        reconditioned_scan = local_slab_scan(reconditioned)
        prefix: dict[str, object] = {
            "depth": depth,
            "path": list(CRITICAL_PATH[:depth]),
            "state": state_metrics(state),
            "reconditioned_state": state_metrics(reconditioned),
            "raw_symmetric_slab": raw_scan,
            "reconditioned_symmetric_slab": reconditioned_scan,
        }
        prefixes.append(prefix)
        if depth == len(CRITICAL_PATH):
            break

        token = CRITICAL_PATH[depth]
        body = token.removeprefix("DOWN_")
        expected_name, side_token = body[:-1], body[-1]
        variable, _weight = adaptive.dominant_variable(state)
        actual_name = adaptive.VARIABLE_NAMES[variable]
        bool_check(
            checks,
            f"split_{depth + 1}_dominant_variable_matches_log",
            actual_name == expected_name,
        )
        bool_check(
            checks,
            f"split_{depth + 1}_current_criterion_refuses_parent",
            not raw_scan["accepted"],
        )
        left, right, reconstruction_checks = adaptive.split_state(state, variable)
        split_reconstruction_checks += reconstruction_checks
        child = left if side_token == "L" else right
        parent_ranges = [component.range() for component in state]
        child_ranges = [component.range() for component in child]
        bool_check(
            checks,
            f"split_{depth + 1}_child_range_in_parent",
            all(
                parent.contains(candidate)
                for parent, candidate in zip(
                    parent_ranges, child_ranges, strict=True
                )
            ),
        )
        state = child

    final_raw = prefixes[-1]["raw_symmetric_slab"]
    final_reconditioned = prefixes[-1]["reconditioned_symmetric_slab"]
    anchored = anchored_event_step_newton(state, approach.event_tube)

    tm_derivative = state[0] * state[1] - state[2] - base.ZS
    coefficient_derivative = base.tm_flow_coefficients(state, 1)[2][1]
    monomials = (
        tm_derivative.coefficients.keys()
        | coefficient_derivative.coefficients.keys()
    )
    bool_check(
        checks,
        "tm_event_derivative_coefficient_identity",
        all(
            same_interval(
                tm_derivative.coefficients.get(monomial, arb(0)),
                coefficient_derivative.coefficients.get(monomial, arb(0)),
            )
            for monomial in monomials
        )
        and same_interval(tm_derivative.remainder, coefficient_derivative.remainder),
    )
    bool_check(
        checks,
        "anchored_endpoint_in_crossing_tube",
        bool(anchored["endpoint_in_crossing_tube"]),
    )

    implementation_ok = all(bool(item["passed"]) for item in checks)
    classification = classify(
        implementation_ok,
        bool(final_raw["accepted"]),
        bool(final_reconditioned["accepted"]),
        bool(anchored["accepted"]),
    )
    source_path = Path(__file__)
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "carrier_source_sha256": sha256(Path(carrier.__file__)),
        "chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "reconditioner": (
            f"{base.recondition.__module__}.{base.recondition.__qualname__}"
        ),
        "tile_id": TILE_ID,
        "critical_path": list(CRITICAL_PATH),
        "critical_depth": len(CRITICAL_PATH),
        "first_return_end_step": first.end_step,
        "downward_reference_time_q": str(approach.reference_time),
        "source_split_reconstruction_checks": source_split_checks,
        "critical_split_reconstruction_checks": split_reconstruction_checks,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "prefix_diagnostics": prefixes,
        "anchored_crossing_step_newton": anchored,
        "final_raw_accepted": bool(final_raw["accepted"]),
        "final_reconditioned_accepted": bool(final_reconditioned["accepted"]),
        "final_anchored_accepted": bool(anchored["accepted"]),
        "classification": classification,
        "diagnostic_complete": True,
        "point_fallback_used": False,
        "box_flattening_used": False,
        "full_transport_attempted": False,
        "covering_relation_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
