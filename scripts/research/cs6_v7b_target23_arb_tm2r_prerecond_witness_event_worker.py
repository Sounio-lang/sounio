#!/usr/bin/env python3
"""Diagnose the exact pre-QR witness that refused the next upward event."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_prerecond_transport_worker as prior


base = prior.base
adaptive = prior.adaptive
chain = prior.chain
event = prior.event
centered = prior.centered
composability = prior.composability
transport = prior.transport

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-witness-event.v1"
WITNESS_PATH = (
    "RHO1L",
    "RHO0L",
    "RHO1L",
    "ETAL",
    "RHO0L",
    "RHO2L",
    "RHO1L",
    "ETAL",
)
EXPECTED_WITNESS_BOUNDS = {
    "xi": ["-1", "0"],
    "eta": ["-13/32", "-51/128"],
    "rho0": ["-1", "-63/64"],
    "rho1": ["1/2", "9/16"],
    "rho2": ["-1", "0"],
    "rho3": ["-1", "1"],
}
PRODUCTION_TIME_DEPTH = chain.MAX_TIME_REFINEMENT_DEPTH
DIAGNOSTIC_TIME_DEPTH = 18
INITIAL_STEP = Fraction(1, 2**8)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def state_json(state: list[base.TM2R]) -> dict[str, object]:
    ranges = [component.range() for component in state]
    return {
        "ranges": [interval_json(value) for value in ranges],
        "widths": [interval_json(base.width(value)) for value in ranges],
        "remainders": [interval_json(component.remainder) for component in state],
        "components": [transport.tm2r_json(component) for component in state],
    }


def tube_json(tube: list[arb]) -> dict[str, object]:
    derivative = tube[0] * tube[1] - tube[2] - base.ZS
    return {
        "ranges": [interval_json(value) for value in tube],
        "w": interval_json(tube[2]),
        "derivative": interval_json(derivative),
        "contains_section": tube[2].lower() <= 0 <= tube[2].upper(),
        "strictly_upward": derivative.lower() > 0,
    }


def endpoint_record(
    state: list[base.TM2R], reference_time: Fraction
) -> dict[str, object]:
    ranges = [component.range() for component in state]
    return {
        "reference_time_q": str(reference_time),
        "w": interval_json(ranges[2]),
        "sign": base.strict_sign(ranges[2]),
        "state": state_json(state),
    }


def reconstruct_witness(
    checks: list[dict[str, object]],
) -> tuple[list[base.TM2R], composability.SymbolicDomain, dict[str, object]]:
    state, _approach, _first_end_step, _source_checks, _critical_checks = (
        centered.critical_state(checks)
    )
    _predictor, _predictor_range, center, _tube, _derivative, _anchor = (
        centered.frozen_predictor(state, checks)
    )
    raw_projection, chart, captured_count = prior.capture_accepted_raw_projection(
        state, center
    )
    # The capture helper restores the production point reconditioner. Switch to
    # the frozen six-variable lineage policy before any new witness split or
    # event projection.
    base.recondition = prior.lineage_preserving_recondition
    bool_check(checks, "raw_projection_capture_is_unique", captured_count == 1)
    bool_check(checks, "prerecond_event_replay_is_accepted", chart.get("accepted") is True)
    root_domain = prior.critical_domain()
    witness_state, stabilization_checks = chain.outward_stabilize_carrier(
        raw_projection
    )
    domain = root_domain
    split_checks = 0
    observed_path: list[str] = []
    for depth, token in enumerate(WITNESS_PATH, start=1):
        expected_name, side = token[:-1], token[-1]
        variable, _weight = adaptive.dominant_variable(witness_state)
        actual_name = adaptive.VARIABLE_NAMES[variable]
        bool_check(
            checks,
            f"witness_split_{depth}_dominant_variable_matches",
            actual_name == expected_name,
        )
        left, right, reconstructions = adaptive.split_state(
            witness_state, variable
        )
        left_domain, right_domain = composability.split_domain_pair(domain, variable)
        split_checks += reconstructions
        if side == "L":
            witness_state, domain = left, left_domain
        else:
            witness_state, domain = right, right_domain
        observed_path.append(actual_name + side)
    bool_check(checks, "witness_path_replayed_exactly", observed_path == list(WITNESS_PATH))
    bool_check(
        checks,
        "witness_domain_matches_frozen_bounds",
        domain.as_json().get("bounds") == EXPECTED_WITNESS_BOUNDS,
    )
    return witness_state, domain, {
        "raw_projection": state_json(raw_projection),
        "raw_projection_weights": [
            interval_json(value)
            for value in centered.variable_weights(raw_projection, rows=prior.SECTION_ROWS)
        ],
        "prerecond_accepted_power": chart.get("accepted_power"),
        "outward_stabilization_checks": stabilization_checks,
        "witness_split_reconstruction_checks": split_checks,
    }


def try_projection(
    state: list[base.TM2R], reference_time: Fraction, label: str
) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate": label,
        "reference_time_q": str(reference_time),
        "endpoint_w": interval_json(state[2].range()),
        "accepted": False,
    }
    try:
        projections, split_nodes, split_reconstructions = chain.project_upward_cover(
            state, reference_time
        )
    except base.Refusal as refusal:
        record.update(
            status=refusal.failure_class,
            detail=refusal.detail,
        )
        return record
    carriers: list[dict[str, object]] = []
    all_variable_weights_present = True
    for projection in projections:
        weights = centered.variable_weights(
            projection.carrier, rows=prior.SECTION_ROWS
        )
        weights_present = all(value.upper() > 0 for value in weights)
        all_variable_weights_present = all_variable_weights_present and weights_present
        carriers.append(
            {
                "event_time": interval_json(projection.event_time),
                "event_derivative": interval_json(projection.derivative),
                "event_normal": interval_json(projection.normal),
                "variable_weights": [interval_json(value) for value in weights],
                "all_six_variable_weights_present": weights_present,
            }
        )
    accepted = bool(carriers) and all_variable_weights_present
    record.update(
        status="ACCEPTED" if accepted else "EVENT_SYMBOLIC_DEPENDENCE_UNRESOLVED",
        accepted=accepted,
        projected_leaves=len(carriers),
        split_nodes=split_nodes,
        split_reconstructions=split_reconstructions,
        all_six_variable_weights_present=all_variable_weights_present,
        carriers=carriers,
    )
    return record


def diagnose_upward_event(initial: list[base.TM2R]) -> dict[str, object]:
    state = initial
    elapsed = Fraction(0)
    accepted_substeps = 0
    time_bisections = 0
    seen_strict_negative = False
    last_strict_negative: dict[str, object] | None = None
    first_positive: dict[str, object] | None = None
    production_boundary: dict[str, object] | None = None
    first_ambiguous: dict[str, object] | None = None
    terminal_ambiguous: dict[str, object] | None = None
    projection_attempts: list[dict[str, object]] = []
    pending: list[tuple[Fraction, int]] = [(INITIAL_STEP, 0)]

    while elapsed < chain.MAX_UPWARD_TIME:
        step_fraction, depth = pending.pop()
        before_ranges = [component.range() for component in state]
        before_sign = base.strict_sign(before_ranges[2])
        if before_sign < 0:
            last_strict_negative = endpoint_record(state, elapsed)
        try:
            next_state, tube = adaptive.advance_with_endpoint_intersection(
                state, base.rational_ball(step_fraction)
            )
        except base.Refusal as refusal:
            if (
                refusal.failure_class in {"PICARD_NO_CLOSURE", "PICARD_NONCONTRACTION"}
                and depth < DIAGNOSTIC_TIME_DEPTH
            ):
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                time_bisections += 1
                continue
            return {
                "status": "WITNESS_ENCLOSURE_UNRESOLVED",
                "detail": refusal.detail,
                "accepted": False,
                "production_boundary_reproduced": production_boundary is not None,
                "production_boundary": production_boundary,
                "last_strict_negative": last_strict_negative,
                "first_positive": first_positive,
                "first_ambiguous": first_ambiguous,
                "terminal_ambiguous": terminal_ambiguous,
                "terminal_refusal": {
                    "failure_class": refusal.failure_class,
                    "detail": refusal.detail,
                    "time_depth": depth,
                    "step_q": str(step_fraction),
                    "reference_time_q": str(elapsed),
                },
                "projection_attempts": projection_attempts,
                "accepted_substeps": accepted_substeps,
                "time_bisections": time_bisections,
            }

        after_ranges = [component.range() for component in next_state]
        after_sign = base.strict_sign(after_ranges[2])
        derivative = tube[0] * tube[1] - tube[2] - base.ZS
        contains_section = tube[2].lower() <= 0 <= tube[2].upper()

        if after_sign < 0:
            seen_strict_negative = True
        elif after_sign > 0 and seen_strict_negative and first_positive is None:
            first_positive = endpoint_record(next_state, elapsed + step_fraction)

        if not contains_section:
            pass
        elif elapsed == 0:
            if before_sign == 0 and after_sign < 0 and derivative.upper() < 0:
                pass
            elif depth < DIAGNOSTIC_TIME_DEPTH:
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                time_bisections += 1
                continue
            else:
                terminal_ambiguous = {
                    "phase": "initial_departure",
                    "time_depth": depth,
                    "step_q": str(step_fraction),
                    "before": endpoint_record(state, elapsed),
                    "after": endpoint_record(next_state, elapsed + step_fraction),
                    "tube": tube_json(tube),
                }
                break
        elif seen_strict_negative:
            ambiguity = {
                "phase": "pre_target",
                "time_depth": depth,
                "step_q": str(step_fraction),
                "before": endpoint_record(state, elapsed),
                "after": endpoint_record(next_state, elapsed + step_fraction),
                "tube": tube_json(tube),
            }
            if first_ambiguous is None:
                first_ambiguous = ambiguity
            candidates = [
                ("before", state, elapsed),
                ("after", next_state, elapsed + step_fraction),
            ]
            candidates.sort(
                key=lambda item: float(base.width(item[1][2].range()).upper())
            )
            accepted_projection: dict[str, object] | None = None
            for label, candidate_state, reference_time in candidates:
                attempt = try_projection(candidate_state, reference_time, label)
                attempt.update(time_depth=depth, step_q=str(step_fraction))
                projection_attempts.append(attempt)
                if attempt["accepted"] is True:
                    accepted_projection = attempt
                    break
            if accepted_projection is not None:
                if production_boundary is None:
                    return {
                        "status": "EARLY_ACCEPTANCE_BEFORE_FROZEN_REFUSAL",
                        "accepted": False,
                        "early_projection": accepted_projection,
                    "production_boundary_reproduced": False,
                    "production_boundary": None,
                        "last_strict_negative": last_strict_negative,
                        "first_positive": first_positive,
                        "first_ambiguous": first_ambiguous,
                        "terminal_ambiguous": None,
                        "terminal_refusal": None,
                        "projection_attempts": projection_attempts,
                        "accepted_substeps": accepted_substeps + 1,
                        "time_bisections": time_bisections,
                    }
                return {
                    "status": "LOCAL_INTERVAL_NEWTON_ACCEPTED",
                    "accepted": True,
                    "accepted_projection": accepted_projection,
                    "accepted_after_production_boundary": True,
                    "production_boundary_reproduced": True,
                    "production_boundary": production_boundary,
                    "last_strict_negative": last_strict_negative,
                    "first_positive": first_positive,
                    "first_ambiguous": first_ambiguous,
                    "terminal_ambiguous": None,
                    "projection_attempts": projection_attempts,
                    "accepted_substeps": accepted_substeps + 1,
                    "time_bisections": time_bisections,
                }
            if depth == PRODUCTION_TIME_DEPTH and production_boundary is None:
                production_boundary = ambiguity
                production_boundary["failure_class"] = (
                    "SECOND_PRIOR_ORIENTATION_UNRESOLVED"
                )
            if depth < DIAGNOSTIC_TIME_DEPTH:
                half = step_fraction / 2
                pending.extend(((half, depth + 1), (half, depth + 1)))
                time_bisections += 1
                continue
            terminal_ambiguous = ambiguity
            break

        state = next_state
        elapsed += step_fraction
        accepted_substeps += 1
        if not pending:
            pending.append((INITIAL_STEP, 0))

    terminal_derivative_strict = False
    if terminal_ambiguous is not None:
        derivative_bounds = terminal_ambiguous["tube"]["derivative"]
        terminal_derivative_strict = Fraction(derivative_bounds[0]) > 0
    status = (
        "WITNESS_ENCLOSURE_UNRESOLVED"
        if terminal_derivative_strict
        else "WITNESS_TRANSVERSALITY_UNRESOLVED"
    )
    return {
        "status": status,
        "accepted": False,
        "production_boundary_reproduced": production_boundary is not None,
        "production_boundary": production_boundary,
        "last_strict_negative": last_strict_negative,
        "first_positive": first_positive,
        "first_ambiguous": first_ambiguous,
        "terminal_ambiguous": terminal_ambiguous,
        "terminal_refusal": None,
        "projection_attempts": projection_attempts,
        "accepted_substeps": accepted_substeps,
        "time_bisections": time_bisections,
    }


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("witness event diagnostic requires Python >= 3.10")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    # Replay the frozen production path through the raw event capture first.
    # reconstruct_witness switches to lineage preservation immediately after
    # that capture and before the new witness-local work begins.
    base.recondition = adaptive.point_coefficient_recondition
    event.MAX_PHASE_STEPS = composability.MAX_FIRST_RETURN_STEPS

    source_path = Path(__file__)
    prerecond_receipt = (
        source_path.parent
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_prerecond_v1"
        / "event_prerecond.json"
    )
    transport_receipt = (
        source_path.parent
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_prerecond_transport_v1"
        / "prerecond_transport.json"
    )
    checks: list[dict[str, object]] = []
    bool_check(
        checks,
        "production_reconditioner_active_before_replay",
        base.recondition is adaptive.point_coefficient_recondition,
    )
    if not prerecond_receipt.is_file():
        raise SystemExit(f"frozen preconditioned event receipt is missing: {prerecond_receipt}")
    if not transport_receipt.is_file():
        raise SystemExit(f"frozen transport refusal receipt is missing: {transport_receipt}")
    bool_check(checks, "prerecond_receipt_exists", True)
    bool_check(checks, "transport_receipt_exists", True)
    transport_payload = json.loads(transport_receipt.read_text(encoding="ascii"))
    unresolved = transport_payload.get("transport", {}).get("unresolved", [])
    bool_check(checks, "prior_transport_has_one_frozen_witness", len(unresolved) == 1)
    if unresolved:
        bool_check(
            checks,
            "prior_witness_path_matches",
            unresolved[0].get("path") == list(WITNESS_PATH),
        )
        bool_check(
            checks,
            "prior_witness_failure_matches",
            unresolved[0].get("failure_class")
            == "SECOND_PRIOR_ORIENTATION_UNRESOLVED",
        )
        bool_check(
            checks,
            "prior_witness_domain_matches",
            unresolved[0].get("domain", {}).get("bounds")
            == EXPECTED_WITNESS_BOUNDS,
        )

    witness_state, witness_domain, reconstruction = reconstruct_witness(checks)
    bool_check(
        checks,
        "lineage_reconditioner_active_for_witness_event",
        base.recondition is prior.lineage_preserving_recondition,
    )
    diagnostic = diagnose_upward_event(witness_state)
    bool_check(
        checks,
        "production_refusal_boundary_reproduced",
        diagnostic.get("production_boundary_reproduced") is True,
    )
    bool_check(
        checks,
        "strict_negative_departure_observed",
        diagnostic.get("last_strict_negative") is not None,
    )
    bool_check(
        checks,
        "ambiguous_pre_target_tube_observed",
        diagnostic.get("first_ambiguous") is not None,
    )
    implementation_ok = all(item["passed"] is True for item in checks)
    if not implementation_ok:
        classification = "IMPLEMENTATION_INCONSISTENCY"
    elif diagnostic.get("accepted") is True:
        classification = "EVENT_REFINEMENT_BUDGET_LIMIT"
    else:
        classification = str(diagnostic.get("status"))

    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "prior_worker_source_sha256": sha256(Path(prior.__file__)),
        "prerecond_worker_source_sha256": sha256(Path(prior.prerecond.__file__)),
        "centered_worker_source_sha256": sha256(Path(centered.__file__)),
        "composability_source_sha256": sha256(Path(composability.__file__)),
        "transport_source_sha256": sha256(Path(transport.__file__)),
        "chain_source_sha256": sha256(Path(chain.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "event_source_sha256": sha256(Path(event.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "prerecond_receipt_sha256": sha256(prerecond_receipt),
        "transport_receipt_sha256": sha256(transport_receipt),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_degree": base.SOURCE_DEGREE,
        "time_taylor_order": base.TIME_TAYLOR_ORDER,
        "tile_id": prior.TILE_ID,
        "critical_path": list(prior.CRITICAL_PATH),
        "witness_path": list(WITNESS_PATH),
        "witness_domain": witness_domain.as_json(),
        "production_time_refinement_depth": PRODUCTION_TIME_DEPTH,
        "diagnostic_time_refinement_depth": DIAGNOSTIC_TIME_DEPTH,
        "symbolic_policy": "original_six_variables_no_qr_renumbering",
        "reconstruction": reconstruction,
        "diagnostic": diagnostic,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "diagnostic_complete": True,
        "full_transport_attempted": False,
        "point_fallback_used": False,
        "box_flattening_used": False,
        "covering_relation_certified": False,
        "recurrent_graph_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
