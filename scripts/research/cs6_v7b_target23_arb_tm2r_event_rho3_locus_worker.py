#!/usr/bin/env python3
"""Locate where residual rho3 weight vanishes on the predictor-centered chart."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_event_centered_worker as centered


base = centered.base
adaptive = centered.adaptive
chain = centered.chain
event = centered.event
prior = centered.prior
carrier = centered.carrier

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-rho3-locus.v1"
TILE_ID = centered.TILE_ID
CRITICAL_PATH = centered.CRITICAL_PATH
SLAB_POWERS = centered.SLAB_POWERS
EXPECTED_CENTER = centered.EXPECTED_CENTER
EXPECTED_EVENT_LOCAL_SHA256 = centered.EXPECTED_PRIOR_RECEIPT_SHA256
EXPECTED_EVENT_CENTERED_SHA256 = (
    "37464bf6b240e9dc621aed3b4fcdf02b8276ce44a21623001aea0d62f6ce29f7"
)
RHO3 = 5
VARIABLE_NAMES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
SECTION_ROWS = centered.SECTION_ROWS
STATE_ROWS = centered.STATE_ROWS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return centered.interval_json(value)


def weights_json(weights: list[arb]) -> list[list[str]]:
    return [interval_json(value) for value in weights]


def positive(weights: list[arb]) -> list[bool]:
    return [value.upper() > 0 for value in weights]


def residual_radii_after_recondition(state: list[base.TM2R]) -> list[arb]:
    """Rebuild residual pure-direction radii produced by the active reconditioner."""
    radii = [arb(0) for _ in range(base.RESIDUAL_VARIABLES)]
    for component in state:
        for monomial, coefficient in component.coefficients.items():
            if sum(monomial) != 1:
                continue
            for residual in range(base.RESIDUAL_VARIABLES):
                if monomial[base.SOURCE_VARIABLES + residual] == 1:
                    radii[residual] += base.upper_abs(coefficient)
    return radii


def classify_scale(
    event_pos: list[bool],
    raw_pos: list[bool],
    recond_pos: list[bool],
) -> str:
    if not event_pos[RHO3]:
        return "FLOW_ERASES_RHO3"
    if not raw_pos[RHO3]:
        return "PROJECTION_ERASES_RHO3"
    if not recond_pos[RHO3]:
        return "RECONDITION_COLLAPSES_RESIDUAL_RANK"
    return "RHO3_PRESERVED_ALL_STAGES"


def instrumented_chart(state: list[base.TM2R], center: Fraction) -> dict[str, object]:
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

    centered_weights = centered.variable_weights(centered_state)
    result.update(
        fixed_picard_iterations=fixed_iterations,
        fixed_picard_contraction=interval_json(fixed_contraction),
        centered_variable_weights=weights_json(centered_weights),
        centered_variables_preserved=all(value.upper() > 0 for value in centered_weights),
        centered_rho3_positive=centered_weights[RHO3].upper() > 0,
    )

    scales: list[dict[str, object]] = []
    locus_counts: dict[str, int] = {}
    for power in SLAB_POWERS:
        radius_q = Fraction(1, 2**power)
        radius = base.rational_ball(radius_q)
        record: dict[str, object] = {"power": power, "radius_q": str(radius_q)}
        try:
            backward, backward_iterations, backward_contraction = (
                event.signed_picard_box(
                    [component.range() for component in centered_state], -radius
                )
            )
            forward, forward_iterations, forward_contraction = (
                event.signed_picard_box(
                    [component.range() for component in centered_state], radius
                )
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
                status="CENTERED_NEWTON_DOMAIN_ESCAPED_PICARD_SLAB",
                newton_domain=interval_json(newton_domain),
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
        reconditioned = base.recondition(raw_projection)

        event_weights_full = centered.variable_weights(event_state, rows=STATE_ROWS)
        event_weights_section = centered.variable_weights(event_state, rows=SECTION_ROWS)
        raw_weights = centered.variable_weights(raw_projection, rows=SECTION_ROWS)
        recond_weights = centered.variable_weights(reconditioned, rows=SECTION_ROWS)
        residual_radii = residual_radii_after_recondition(reconditioned)
        event_pos = positive(event_weights_section)
        raw_pos = positive(raw_weights)
        recond_pos = positive(recond_weights)
        locus = classify_scale(event_pos, raw_pos, recond_pos)
        locus_counts[locus] = locus_counts.get(locus, 0) + 1

        projected_ranges = [component.range() for component in reconditioned]
        normal = projected_ranges[0] * projected_ranges[1] - base.ZS
        exact_section = (
            projected_ranges[2].lower() == 0 and projected_ranges[2].upper() == 0
        )
        retained_raw = centered.retained_source_monomials(raw_projection)
        retained_recond = centered.retained_source_monomials(reconditioned)

        record.update(
            status=locus,
            event_time=interval_json(event_time_range),
            event_time_variable_weights=weights_json(
                centered.model_variable_weights(event_time_model)
            ),
            event_state_variable_weights_full=weights_json(event_weights_full),
            event_state_variable_weights_section=weights_json(event_weights_section),
            event_state_variables_preserved=all(event_pos),
            raw_projection_variable_weights=weights_json(raw_weights),
            raw_projection_variables_preserved=all(raw_pos),
            raw_projection_rho3_positive=raw_pos[RHO3],
            reconditioned_variable_weights=weights_json(recond_weights),
            reconditioned_variables_preserved=all(recond_pos),
            reconditioned_rho3_positive=recond_pos[RHO3],
            residual_pure_direction_radii=weights_json(residual_radii),
            residual_pure_direction_rank=sum(1 for value in residual_radii if value.upper() > 0),
            projected_normal=interval_json(normal),
            exact_section=exact_section,
            pure_source_monomials_raw=retained_raw,
            pure_source_monomials_reconditioned=retained_recond,
            variable_names=list(VARIABLE_NAMES),
        )
        scales.append(record)

    modes = sorted(locus_counts)
    if not modes:
        overall = "NO_SCALE_REACHED_PROJECTION"
    elif modes == ["RECONDITION_COLLAPSES_RESIDUAL_RANK"]:
        overall = "RECONDITION_COLLAPSES_RESIDUAL_RANK"
    elif modes == ["FLOW_ERASES_RHO3"]:
        overall = "FLOW_ERASES_RHO3"
    elif modes == ["PROJECTION_ERASES_RHO3"]:
        overall = "PROJECTION_ERASES_RHO3"
    elif modes == ["RHO3_PRESERVED_ALL_STAGES"]:
        overall = "RHO3_PRESERVED_ALL_STAGES"
    else:
        overall = "MIXED_RHO3_LOCI"

    result.update(
        accepted=False,
        accepted_power=None,
        status=overall,
        scales=scales,
        locus_counts=locus_counts,
        projection_scales=sum(1 for scale in scales if "raw_projection_rho3_positive" in scale),
    )
    return result


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("rho3 locus diagnostic requires Python >= 3.10")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    base.recondition = adaptive.point_coefficient_recondition
    event.MAX_PHASE_STEPS = carrier.MAX_FIRST_RETURN_STEPS

    source_path = Path(__file__)
    research = source_path.parent
    event_local_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_local_v1"
        / "event_local_diagnostic.json"
    )
    event_centered_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_centered_v1"
        / "event_centered.json"
    )
    if not event_local_receipt.is_file():
        raise SystemExit("frozen event-local receipt is missing")
    if not event_centered_receipt.is_file():
        raise SystemExit("frozen event-centered receipt is missing")

    checks: list[dict[str, object]] = []
    centered.bool_check(
        checks,
        "prior_event_local_hash_matches",
        sha256(event_local_receipt) == EXPECTED_EVENT_LOCAL_SHA256,
    )
    centered.bool_check(
        checks,
        "prior_event_centered_hash_matches",
        sha256(event_centered_receipt) == EXPECTED_EVENT_CENTERED_SHA256,
    )
    state, approach, first_end_step, source_checks, critical_checks = (
        centered.critical_state(checks)
    )
    raw_weights = centered.variable_weights(state)
    centered.bool_check(
        checks,
        "critical_state_preserves_all_six_variables",
        all(value.upper() > 0 for value in raw_weights),
    )
    predictor, predictor_range, center, _tube, _derivative, anchor = (
        centered.frozen_predictor(state, checks)
    )
    # frozen_predictor already records predictor_center_matches_frozen_receipt.
    chart = instrumented_chart(state, center)
    if chart.get("centered_rho3_positive") is True:
        centered.bool_check(checks, "centered_state_keeps_rho3", True)
    else:
        centered.bool_check(checks, "centered_state_keeps_rho3", False)

    implementation_ok = all(bool(item["passed"]) for item in checks)
    classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not implementation_ok
        else str(chart.get("status"))
    )
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "centered_worker_source_sha256": sha256(Path(centered.__file__)),
        "prior_worker_source_sha256": sha256(Path(prior.__file__)),
        "carrier_source_sha256": sha256(Path(carrier.__file__)),
        "chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "prior_event_local_receipt_sha256": sha256(event_local_receipt),
        "prior_event_centered_receipt_sha256": sha256(event_centered_receipt),
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
        "critical_variable_weights": weights_json(raw_weights),
        "anchor": anchor,
        "predictor_range": interval_json(predictor_range),
        "predictor_center_q": str(center),
        "lost_variable_index": RHO3,
        "lost_variable_name": VARIABLE_NAMES[RHO3],
        "variable_names": list(VARIABLE_NAMES),
        "rho3_locus_chart": chart,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "diagnostic_complete": True,
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
