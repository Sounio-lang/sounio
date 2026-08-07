#!/usr/bin/env python3
"""Predictor-centered residual chart with pre-QR symbolic acceptance."""

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

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-prerecond.v1"
TILE_ID = centered.TILE_ID
CRITICAL_PATH = centered.CRITICAL_PATH
SLAB_POWERS = centered.SLAB_POWERS
EXPECTED_CENTER = centered.EXPECTED_CENTER
EXPECTED_EVENT_LOCAL_SHA256 = centered.EXPECTED_PRIOR_RECEIPT_SHA256
EXPECTED_EVENT_CENTERED_SHA256 = (
    "37464bf6b240e9dc621aed3b4fcdf02b8276ce44a21623001aea0d62f6ce29f7"
)
EXPECTED_RHO3_LOCUS_SHA256 = (
    "6b3ee1c7244e6618d4256a0fc5afad42c9fcd328e3dc70d79c6a35028ffe3016"
)
SECTION_ROWS = centered.SECTION_ROWS
RHO3 = 5
VARIABLE_NAMES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return centered.interval_json(value)


def weights_json(weights: list[arb]) -> list[list[str]]:
    return [interval_json(value) for value in weights]


def residual_pure_direction_rank(state: list[base.TM2R]) -> int:
    radii = [arb(0) for _ in range(base.RESIDUAL_VARIABLES)]
    for component in state:
        for monomial, coefficient in component.coefficients.items():
            if sum(monomial) != 1:
                continue
            for residual in range(base.RESIDUAL_VARIABLES):
                if monomial[base.SOURCE_VARIABLES + residual] == 1:
                    radii[residual] += base.upper_abs(coefficient)
    return sum(1 for value in radii if value.upper() > 0)


def prerecond_event_chart(
    state: list[base.TM2R], center: Fraction
) -> dict[str, object]:
    result: dict[str, object] = {
        "accepted": False,
        "fixed_shift_q": str(center),
        "symbolic_policy": "raw_projection_pre_qr",
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
    )

    scales: list[dict[str, object]] = []
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
        # Exact rational shift: combined endpoints = frozen center + serialized
        # residual event-time endpoints. Do not re-round through Arb addition.
        event_lower_q = Fraction(base.lower_fraction(event_time_range))
        event_upper_q = Fraction(base.upper_fraction(event_time_range))
        record.update(
            correction=interval_json(newton_image),
            event_time=[str(event_lower_q), str(event_upper_q)],
            combined_event_time=[
                str(center + event_lower_q),
                str(center + event_upper_q),
            ],
            event_time_variable_weights=weights_json(
                centered.model_variable_weights(event_time_model)
            ),
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
        reconditioned = base.recondition(raw_projection)

        raw_ranges = [component.range() for component in raw_projection]
        normal = raw_ranges[0] * raw_ranges[1] - base.ZS
        exact_section = raw_ranges[2].lower() == 0 and raw_ranges[2].upper() == 0
        raw_weights = centered.variable_weights(raw_projection, rows=SECTION_ROWS)
        recond_weights = centered.variable_weights(reconditioned, rows=SECTION_ROWS)
        raw_preserved = all(value.upper() > 0 for value in raw_weights)
        recond_preserved = all(value.upper() > 0 for value in recond_weights)
        retained_raw = centered.retained_source_monomials(raw_projection)
        retained_recond = centered.retained_source_monomials(reconditioned)
        residual_rank = residual_pure_direction_rank(reconditioned)

        record.update(
            raw_projection_state=prior.state_metrics(raw_projection),
            reconditioned_state=prior.state_metrics(reconditioned),
            projected_normal=interval_json(normal),
            raw_projection_variable_weights=weights_json(raw_weights),
            raw_projection_variables_preserved=raw_preserved,
            raw_projection_rho3_positive=raw_weights[RHO3].upper() > 0,
            reconditioned_variable_weights=weights_json(recond_weights),
            reconditioned_variables_preserved=recond_preserved,
            reconditioned_rho3_positive=recond_weights[RHO3].upper() > 0,
            residual_pure_direction_rank=residual_rank,
            pure_source_monomials_raw=retained_raw,
            pure_source_monomials_reconditioned=retained_recond,
            exact_section=exact_section,
            variable_names=list(VARIABLE_NAMES),
            symbolic_gate_uses_raw_projection=True,
        )
        if not exact_section:
            record["status"] = "CENTERED_SECTION_PROJECTION_DRIFT"
            scales.append(record)
            continue
        if normal.upper() >= 0:
            record["status"] = "CENTERED_TRANSVERSALITY_UNRESOLVED"
            scales.append(record)
            continue
        if not raw_preserved or retained_raw == 0:
            record["status"] = "CENTERED_RAW_SYMBOLIC_DEPENDENCE_LOST"
            scales.append(record)
            continue

        # Forensic note only: post-QR collapse is expected and does not refuse.
        record["status"] = "ACCEPTED"
        record["post_qr_residual_rank_forensic"] = residual_rank
        record["post_qr_symbolic_preserved_forensic"] = recond_preserved
        scales.append(record)
        result.update(
            accepted=True,
            accepted_power=power,
            status="ACCEPTED",
            scales=scales,
            accepted_raw_projection_variables_preserved=True,
            accepted_residual_pure_direction_rank_forensic=residual_rank,
        )
        return result

    result.update(
        accepted_power=None,
        status="CENTERED_PRERECOND_EVENT_CHART_REFUSED",
        scales=scales,
    )
    return result


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("prerecond event diagnostic requires Python >= 3.10")
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
    rho3_locus_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_rho3_locus_v1"
        / "rho3_locus.json"
    )
    for path, label in (
        (event_local_receipt, "event-local"),
        (event_centered_receipt, "event-centered"),
        (rho3_locus_receipt, "rho3-locus"),
    ):
        if not path.is_file():
            raise SystemExit(f"frozen {label} receipt is missing: {path}")

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
    centered.bool_check(
        checks,
        "prior_rho3_locus_hash_matches",
        sha256(rho3_locus_receipt) == EXPECTED_RHO3_LOCUS_SHA256,
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
    centered.bool_check(
        checks,
        "predictor_model_preserves_all_six_variables",
        all(
            value.upper() > 0
            for value in centered.model_variable_weights(predictor)
        ),
    )
    chart = prerecond_event_chart(state, center)
    if "centered_variables_preserved" in chart:
        centered.bool_check(
            checks,
            "centered_state_preserves_all_six_variables",
            bool(chart["centered_variables_preserved"]),
        )
    if chart.get("accepted") is True:
        accepted_scale = chart["scales"][-1]
        centered.bool_check(
            checks,
            "accepted_raw_projection_preserves_all_six_variables",
            bool(accepted_scale["raw_projection_variables_preserved"]),
        )
        centered.bool_check(
            checks,
            "accepted_raw_projection_keeps_rho3",
            bool(accepted_scale["raw_projection_rho3_positive"]),
        )
        centered.bool_check(
            checks,
            "accepted_event_time_is_strictly_inside_residual_slab",
            accepted_scale["status"] == "ACCEPTED",
        )
        centered.bool_check(
            checks,
            "symbolic_gate_uses_raw_projection_not_post_qr",
            accepted_scale.get("symbolic_gate_uses_raw_projection") is True,
        )

    implementation_ok = all(bool(item["passed"]) for item in checks)
    accepted = implementation_ok and chart.get("accepted") is True
    classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not implementation_ok
        else (
            "PREDICTOR_CENTERED_PRERECOND_EVENT_ACCEPTED"
            if accepted
            else "PREDICTOR_CENTERED_PRERECOND_EVENT_REFUSED"
        )
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
        "prior_rho3_locus_receipt_sha256": sha256(rho3_locus_receipt),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_degree": base.SOURCE_DEGREE,
        "time_taylor_order": base.TIME_TAYLOR_ORDER,
        "reconditioner": (
            f"{base.recondition.__module__}.{base.recondition.__qualname__}"
        ),
        "symbolic_gate_policy": "raw_projection_pre_qr",
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
        "prerecond_event_chart": chart,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "predictor_centered_prerecond_event_accepted": accepted,
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
