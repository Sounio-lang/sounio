#!/usr/bin/env python3
"""Fail-closed verifier for the event-normal QR carrier receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-normal-carrier.v1"
EXPECTED_WITNESS_SHA256 = "76115e2b3e7dee3a2a3b85fe91c15250f25e3f8643efe4ee56a42a9a68a2f8b7"
EXPECTED_BUDGET_SHA256 = "f5b0f3ac5936c7814b20194bd13dc24e10408d4e9f9139a8c4c07c38084fbe21"
MODES = ("EVENT_NORMAL_DOUBLETON", "EVENT_NORMAL_TRIPLETON")
PRIMARY = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
CARRIER = ("sigma0", "sigma1", "sigma2", "sigma3")


def fail(message: str) -> None:
    raise SystemExit(f"event-normal carrier verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def rational(value: object, label: str) -> Fraction:
    if isinstance(value, bool):
        fail(f"{label} is boolean, not rational")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        fail(f"{label} is not rational: {error}")


def interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{label} is not a two-endpoint interval")
    lower = rational(value[0], f"{label} lower")
    upper = rational(value[1], f"{label} upper")
    if lower > upper:
        fail(f"{label} has reversed endpoints")
    return lower, upper


def dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def matrix_product(
    left: list[list[Fraction]], right: list[list[Fraction]]
) -> list[list[Fraction]]:
    return [
        [
            sum((left[row][k] * right[k][column] for k in range(4)), Fraction(0))
            for column in range(4)
        ]
        for row in range(4)
    ]


def parse_matrix(value: object, label: str) -> list[list[Fraction]]:
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{label} is not a four-row matrix")
    result: list[list[Fraction]] = []
    for row, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 4:
            fail(f"{label} row {row} is malformed")
        result.append([rational(item, f"{label}[{row}]") for item in raw])
    return result


def validate_basis_history(value: object, label: str) -> tuple[int, Fraction]:
    if not isinstance(value, list) or not value:
        fail(f"{label} basis history is absent")
    previous = 0
    maximum_row_sum = Fraction(0)
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            fail(f"{label} basis record {index} is malformed")
        reconditioning = record.get("reconditioning")
        if not isinstance(reconditioning, int) or reconditioning <= previous:
            fail(f"{label} basis history is not chronological")
        previous = reconditioning
        covector_raw = record.get("event_covector")
        if not isinstance(covector_raw, list) or len(covector_raw) != 4:
            fail(f"{label} event covector is malformed")
        covector = [rational(item, f"{label} covector") for item in covector_raw]
        if covector[2:] != [Fraction(-1), Fraction(0)]:
            fail(f"{label} event covector has wrong w/ell coefficients")
        basis = parse_matrix(record.get("basis"), f"{label} basis")
        inverse = parse_matrix(record.get("inverse"), f"{label} inverse")
        identity = [
            [Fraction(int(row == column)) for column in range(4)]
            for row in range(4)
        ]
        if matrix_product(basis, inverse) != identity:
            fail(f"{label} basis times inverse is not identity")
        columns = [[basis[row][column] for row in range(4)] for column in range(4)]
        normal_pairing = dot(covector, columns[0])
        if not normal_pairing:
            fail(f"{label} normal column lies in the event kernel")
        if rational(record.get("normal_pairing_q"), f"{label} normal pairing") != normal_pairing:
            fail(f"{label} normal pairing is incorrect")
        kernel_pairings = record.get("kernel_pairings_q")
        if not isinstance(kernel_pairings, list) or len(kernel_pairings) != 3:
            fail(f"{label} kernel pairings are malformed")
        for column, raw_pairing in enumerate(kernel_pairings, start=1):
            pairing = dot(covector, columns[column])
            if pairing or rational(raw_pairing, f"{label} kernel pairing") != 0:
                fail(f"{label} complement column {column} leaves the exact kernel")
        radii = record.get("coordinate_radii")
        if not isinstance(radii, list) or len(radii) != 4:
            fail(f"{label} coordinate radii are malformed")
        for coordinate, radius in enumerate(radii):
            lower, upper = interval(radius, f"{label} radius {coordinate}")
            if lower < 0 or upper <= 0:
                fail(f"{label} radius {coordinate} is not strictly present")
        maximum_row_sum = max(
            maximum_row_sum,
            max(sum(abs(item) for item in row) for row in inverse),
        )
    return len(value), maximum_row_sum


def validate_stats(value: object, label: str, complete_history: bool) -> None:
    if not isinstance(value, dict):
        fail(f"{label} stats are absent")
    reconditionings = value.get("reconditionings")
    reconstructions = value.get("generator_reconstructions")
    checks = value.get("reconstruction_checks")
    kernel_checks = value.get("kernel_orthogonality_checks")
    normal_form_checks = value.get("normal_form_checks")
    if not isinstance(reconditionings, int) or reconditionings <= 0:
        fail(f"{label} has no reconditionings")
    if not isinstance(reconstructions, int) or reconstructions <= 0 or checks != reconstructions:
        fail(f"{label} generator reconstruction count is inconsistent")
    if kernel_checks != 3 * reconditionings:
        fail(f"{label} kernel check count is inconsistent")
    if normal_form_checks != reconditionings:
        fail(f"{label} normal-form check count is inconsistent")
    history_count, history_max = validate_basis_history(value.get("basis_history"), label)
    reported_max = rational(
        value.get("maximum_basis_inverse_row_sum_q"),
        f"{label} maximum inverse row sum",
    )
    if reported_max < history_max:
        fail(f"{label} maximum inverse row sum misses retained history")
    if complete_history and history_count != reconditionings:
        fail(f"{label} preflight history is incomplete")
    maximum_generators = value.get("maximum_generator_count")
    if not isinstance(maximum_generators, int) or maximum_generators <= 0:
        fail(f"{label} maximum generator count is invalid")


def validate_budget(value: object, label: str) -> Fraction:
    if not isinstance(value, dict):
        fail(f"{label} derivative budget is absent")
    enclosure = interval(value.get("range"), f"{label} range")
    exact_width = enclosure[1] - enclosure[0]
    width = rational(value.get("width_q"), f"{label} width")
    tolerance = Fraction(1, 2**230)
    if width < exact_width or width - exact_width > tolerance:
        fail(f"{label} width disagrees with range")
    midpoint = rational(value.get("midpoint_q"), f"{label} midpoint")
    radius = rational(value.get("radius_q"), f"{label} radius")
    exact_midpoint = sum(enclosure) / 2
    exact_radius = exact_width / 2
    if abs(midpoint - exact_midpoint) > tolerance:
        fail(f"{label} midpoint disagrees with range")
    if radius < exact_radius or radius - exact_radius > tolerance:
        fail(f"{label} midpoint or radius disagrees with range")
    remainder = interval(value.get("remainder"), f"{label} remainder")
    exact_remainder_width = remainder[1] - remainder[0]
    reported_remainder_width = rational(
        value.get("remainder_width_q"), f"{label} remainder width"
    )
    if (
        reported_remainder_width < exact_remainder_width
        or reported_remainder_width - exact_remainder_width > tolerance
    ):
        fail(f"{label} remainder width is incorrect")
    return width


def validate_components(value: object, label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{label} does not contain four components")
    for row, component in enumerate(value):
        if not isinstance(component, dict):
            fail(f"{label} component {row} is malformed")
        coefficients = component.get("coefficients")
        if not isinstance(coefficients, list) or not coefficients:
            fail(f"{label} component {row} has no coefficients")
        seen: set[tuple[int, ...]] = set()
        for item in coefficients:
            if not isinstance(item, dict):
                fail(f"{label} component {row} coefficient is malformed")
            raw = item.get("monomial")
            if not isinstance(raw, list) or len(raw) != 10:
                fail(f"{label} component {row} is not a ten-variable carrier")
            monomial = tuple(int(exponent) for exponent in raw)
            if any(exponent < 0 for exponent in monomial) or sum(monomial) > 2:
                fail(f"{label} component {row} exceeds degree two")
            if monomial in seen:
                fail(f"{label} component {row} duplicates a monomial")
            carrier = monomial[6:]
            if any(carrier) and not (
                sum(monomial[:6]) == 0 and sum(carrier) == 1
            ):
                fail(f"{label} component {row} leaves carrier normal form")
            seen.add(monomial)
            interval(item.get("interval"), f"{label} coefficient interval")
        remainder = interval(component.get("remainder"), f"{label} component remainder")
        if remainder != (Fraction(0), Fraction(0)):
            fail(f"{label} component {row} has a nonzero remainder after reconditioning")


def validate_mode(
    value: object,
    baseline: dict[str, object],
) -> tuple[str, bool]:
    if not isinstance(value, dict) or value.get("mode") not in MODES:
        fail("preflight mode is malformed")
    mode = str(value["mode"])
    analyses = value.get("analyses")
    if not isinstance(analyses, dict) or set(analyses) != {"production_before", "terminal_before"}:
        fail(f"{mode} endpoint analysis set is not closed")
    for name, analysis in analyses.items():
        if not isinstance(analysis, dict):
            fail(f"{mode} {name} analysis is malformed")
        validate_components(analysis.get("conditioned_components"), f"{mode} {name}")
        width = validate_budget(
            analysis.get("conditioned_derivative_budget"), f"{mode} {name}"
        )
        expected_baseline = rational(
            baseline["analyses"][name]["derivative_budget"]["width_q"],
            f"baseline {name}",
        )
        if rational(analysis.get("baseline_derivative_width_q"), f"{mode} baseline") != expected_baseline:
            fail(f"{mode} {name} baseline width is incorrect")
        improvement = expected_baseline / width if width else Fraction(0)
        if rational(analysis.get("derivative_width_improvement_factor_q"), f"{mode} endpoint improvement") != improvement:
            fail(f"{mode} {name} endpoint improvement is incorrect")
        if analysis.get("target_improvement_met") is not (improvement >= 18):
            fail(f"{mode} {name} target flag is incorrect")
        validate_stats(analysis.get("stats"), f"{mode} {name}", True)

    initial = value.get("initial_witness_analysis")
    if not isinstance(initial, dict):
        fail(f"{mode} initial witness analysis is absent")
    validate_components(initial.get("conditioned_initial_components"), f"{mode} initial")
    validate_components(initial.get("carrier_one_step_components"), f"{mode} one step")
    lineage_width = validate_budget(
        initial.get("lineage_one_step_derivative_budget"), f"{mode} lineage one step"
    )
    carrier_width = validate_budget(
        initial.get("carrier_one_step_derivative_budget"), f"{mode} carrier one step"
    )
    validate_budget(
        initial.get("conditioned_initial_derivative_budget"), f"{mode} conditioned initial"
    )
    interval(initial.get("carrier_one_step_tube_derivative"), f"{mode} one-step tube")
    improvement = lineage_width / carrier_width if carrier_width else Fraction(0)
    if rational(initial.get("one_step_derivative_width_improvement_factor_q"), f"{mode} one-step improvement") != improvement:
        fail(f"{mode} one-step improvement factor is incorrect")
    margin = lineage_width - carrier_width
    tolerance = Fraction(1, 2**230)
    if rational(initial.get("one_step_derivative_width_margin_q"), f"{mode} one-step margin") != margin:
        fail(f"{mode} one-step width margin is incorrect")
    if rational(initial.get("receipt_rounding_tolerance_q"), f"{mode} receipt tolerance") != tolerance:
        fail(f"{mode} receipt rounding tolerance is incorrect")
    if initial.get("width_margin_exceeds_receipt_rounding_tolerance") is not (margin > tolerance):
        fail(f"{mode} width-margin tolerance flag is incorrect")
    improves = improvement > 1
    if initial.get("one_step_improves_lineage") is not improves:
        fail(f"{mode} one-step improvement flag is incorrect")
    validate_stats(initial.get("stats"), f"{mode} initial", True)
    stats = initial["stats"]
    certificate = (
        stats["generator_reconstructions"] > 0
        and stats["generator_reconstructions"] == stats["reconstruction_checks"]
        and stats["kernel_orthogonality_checks"] == 3 * stats["reconditionings"]
        and stats["normal_form_checks"] == stats["reconditionings"]
    )
    if initial.get("generator_reconstruction_certificate") is not certificate:
        fail(f"{mode} reconstruction certificate is incorrect")
    expected_class = (
        "EVENT_NORMAL_PREFLIGHT_RECONSTRUCTION_FAILED"
        if not certificate
        else (
            "EVENT_NORMAL_PREFLIGHT_ONE_STEP_IMPROVED"
            if improves
            else "EVENT_NORMAL_PREFLIGHT_ONE_STEP_WORSENED"
        )
    )
    if value.get("classification") != expected_class:
        fail(f"{mode} classification is incorrect")
    if value.get("post_hoc_endpoint_recovery_is_control_only") is not True:
        fail(f"{mode} post-hoc control boundary is absent")
    return mode, expected_class == "EVENT_NORMAL_PREFLIGHT_ONE_STEP_IMPROVED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--witness-worker", type=Path, required=True)
    parser.add_argument("--budget-worker", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--witness-receipt", type=Path, required=True)
    parser.add_argument("--budget-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))
    baseline = json.loads(args.budget_receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    for key, path in (
        ("worker_source_sha256", args.worker),
        ("witness_worker_source_sha256", args.witness_worker),
        ("budget_worker_source_sha256", args.budget_worker),
        ("base_source_sha256", args.base),
        ("adaptive_source_sha256", args.adaptive),
        ("witness_receipt_sha256", args.witness_receipt),
        ("budget_receipt_sha256", args.budget_receipt),
    ):
        require(payload, key, sha256(path))
    require(payload, "witness_receipt_sha256", EXPECTED_WITNESS_SHA256)
    require(payload, "budget_receipt_sha256", EXPECTED_BUDGET_SHA256)
    require(payload, "arb_precision_bits", 256)
    require(payload, "source_degree", 2)
    require(payload, "time_taylor_order", 12)
    require(payload, "primary_variables", list(PRIMARY))
    require(payload, "carrier_variables", list(CARRIER))
    require(payload, "target_improvement_factor_q", "18")
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks if isinstance(item, dict)]
    if len(names) != len(checks) or len(names) != len(set(names)):
        fail("implementation checks are malformed or duplicated")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    result = payload.get("result")
    if not isinstance(result, dict):
        fail("result is absent")
    execution_mode = result.get("execution_mode")
    if execution_mode == "PREFLIGHT":
        require(payload, "interval_newton_attempted", False)
        require(result, "full_transport_attempted", False)
        modes = result.get("modes")
        if not isinstance(modes, list) or len(modes) != 2:
            fail("preflight does not contain both modes")
        outcomes = [validate_mode(item, baseline) for item in modes]
        if [name for name, _viable in outcomes] != list(MODES):
            fail("preflight mode order is incorrect")
        viable = [name for name, accepted in outcomes if accepted]
        require(result, "transport_candidates", viable)
        expected = (
            "EVENT_NORMAL_PREFLIGHT_CANDIDATE_FOUND"
            if viable
            else "EVENT_NORMAL_PREFLIGHT_NO_CANDIDATE"
        )
        require(result, "classification", expected)
    elif execution_mode == "TRANSPORT":
        require(result, "full_transport_attempted", True)
        if result.get("mode") not in MODES:
            fail("transport mode is invalid")
        validate_stats(result.get("carrier_stats"), "transport", False)
        diagnostic = result.get("diagnostic")
        if not isinstance(diagnostic, dict) or "status" not in diagnostic:
            fail("transport diagnostic is absent")
        attempts = bool(diagnostic.get("projection_attempts"))
        require(payload, "interval_newton_attempted", attempts)
        expected = (
            "IMPLEMENTATION_INCONSISTENCY"
            if result.get("implementation_checks_passed") is not True
            else (
                "EVENT_NORMAL_TRANSPORT_ACCEPTED"
                if diagnostic.get("accepted") is True
                else diagnostic.get("status")
            )
        )
        require(result, "classification", expected)
    else:
        fail("execution mode is invalid")

    print("CS6_EVENT_NORMAL_CARRIER_VERIFIED=true")


if __name__ == "__main__":
    main()
