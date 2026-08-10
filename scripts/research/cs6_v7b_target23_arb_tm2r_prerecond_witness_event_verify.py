#!/usr/bin/env python3
"""Fail-closed verifier for the pre-QR witness-local event diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-witness-event.v1"
PRERECOND_SHA256 = "4b615c5632ba9537d639d4fe831c924aff1586a0d4a9db1f2f4efd9c1f1daa3a"
TRANSPORT_SHA256 = "3a3371dcdabe66f5f0f1c79d5988b73e42cc0232a02eb9c85fd19ade2f5238f5"
WITNESS_PATH = [
    "RHO1L", "RHO0L", "RHO1L", "ETAL",
    "RHO0L", "RHO2L", "RHO1L", "ETAL",
]
WITNESS_BOUNDS = {
    "xi": (Fraction(-1), Fraction(0)),
    "eta": (Fraction(-13, 32), Fraction(-51, 128)),
    "rho0": (Fraction(-1), Fraction(-63, 64)),
    "rho1": (Fraction(1, 2), Fraction(9, 16)),
    "rho2": (Fraction(-1), Fraction(0)),
    "rho3": (Fraction(-1), Fraction(1)),
}
VARIABLES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
CLASSIFICATIONS = {
    "EVENT_REFINEMENT_BUDGET_LIMIT",
    "WITNESS_ENCLOSURE_UNRESOLVED",
    "WITNESS_TRANSVERSALITY_UNRESOLVED",
    "IMPLEMENTATION_INCONSISTENCY",
}


def fail(message: str) -> None:
    raise SystemExit(f"witness event verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{label} is not a two-endpoint interval")
    try:
        lower, upper = (Fraction(str(item)) for item in value)
    except (ValueError, ZeroDivisionError) as error:
        fail(f"{label} has invalid rational endpoints: {error}")
    if lower > upper:
        fail(f"{label} has reversed endpoints")
    return lower, upper


def positive_weights(value: object, label: str) -> None:
    if not isinstance(value, list) or len(value) != len(VARIABLES):
        fail(f"{label} does not contain six weights")
    for index, item in enumerate(value):
        lower, upper = interval(item, f"{label}[{index}]")
        if lower < 0 or upper <= 0:
            fail(f"{label}[{index}] is not a positive-presence enclosure")


def validate_domain(value: object) -> None:
    if not isinstance(value, dict):
        fail("witness domain is absent")
    bounds = value.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(VARIABLES):
        fail("witness domain does not bind exactly the original six variables")
    for variable, expected in WITNESS_BOUNDS.items():
        if interval(bounds[variable], f"witness domain {variable}") != expected:
            fail(f"witness domain {variable} differs from the frozen witness")
    lineage = value.get("split_lineage")
    if not isinstance(lineage, list) or lineage[-len(WITNESS_PATH):] != WITNESS_PATH:
        fail("witness domain lineage does not end in the frozen eight splits")
    trace = value.get("split_trace")
    if not isinstance(trace, list) or len(trace) != len(lineage):
        fail("witness domain trace and lineage lengths differ")


def validate_endpoint(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict):
        fail(f"{label} is absent")
    reference = Fraction(str(value.get("reference_time_q")))
    if reference < 0:
        fail(f"{label} has negative reference time")
    w = interval(value.get("w"), f"{label} w")
    sign = value.get("sign")
    exact_sign = -1 if w[1] < 0 else (1 if w[0] > 0 else 0)
    if sign != exact_sign:
        fail(f"{label} sign disagrees with its w interval")
    return w


def validate_ambiguity(value: object, label: str, depth: int | None = None) -> bool:
    if not isinstance(value, dict):
        fail(f"{label} is absent")
    if value.get("phase") != "pre_target":
        fail(f"{label} is not a pre-target witness")
    if depth is not None and value.get("time_depth") != depth:
        fail(f"{label} has wrong time depth")
    step = Fraction(str(value.get("step_q")))
    if step <= 0:
        fail(f"{label} has nonpositive step")
    before_w = validate_endpoint(value.get("before"), f"{label} before")
    after_w = validate_endpoint(value.get("after"), f"{label} after")
    tube = value.get("tube")
    if not isinstance(tube, dict):
        fail(f"{label} tube is absent")
    tube_w = interval(tube.get("w"), f"{label} tube w")
    if not (tube_w[0] <= 0 <= tube_w[1]):
        fail(f"{label} tube does not contain the section")
    if tube.get("contains_section") is not True:
        fail(f"{label} section flag is false")
    derivative = interval(tube.get("derivative"), f"{label} derivative")
    if tube.get("strictly_upward") != (derivative[0] > 0):
        fail(f"{label} derivative flag disagrees with its interval")
    if before_w[0] > tube_w[1] or after_w[1] < tube_w[0]:
        fail(f"{label} endpoint data is incompatible with its tube")
    return derivative[0] > 0


def validate_accepted_projection(value: object) -> None:
    if not isinstance(value, dict) or value.get("accepted") is not True:
        fail("accepted projection is absent")
    if value.get("status") != "ACCEPTED":
        fail("accepted projection has a non-accepted status")
    if value.get("all_six_variable_weights_present") is not True:
        fail("accepted projection loses representation of an original symbolic variable")
    carriers = value.get("carriers")
    if not isinstance(carriers, list) or not carriers:
        fail("accepted projection has no carrier leaves")
    if value.get("projected_leaves") != len(carriers):
        fail("accepted projection leaf count is inconsistent")
    for index, carrier in enumerate(carriers):
        if not isinstance(carrier, dict):
            fail(f"accepted carrier {index} is malformed")
        derivative = interval(carrier.get("event_derivative"), f"carrier {index} derivative")
        normal = interval(carrier.get("event_normal"), f"carrier {index} normal")
        if derivative[0] <= 0 or normal[0] <= 0:
            fail(f"accepted carrier {index} is not strictly upward-transversal")
        if carrier.get("all_six_variable_weights_present") is not True:
            fail(f"accepted carrier {index} loses symbolic-variable representation")
        positive_weights(carrier.get("variable_weights"), f"carrier {index} weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--prior-worker", type=Path, required=True)
    parser.add_argument("--prerecond-worker", type=Path, required=True)
    parser.add_argument("--centered-worker", type=Path, required=True)
    parser.add_argument("--composability", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prerecond-receipt", type=Path, required=True)
    parser.add_argument("--transport-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    for key, path in (
        ("worker_source_sha256", args.worker),
        ("prior_worker_source_sha256", args.prior_worker),
        ("prerecond_worker_source_sha256", args.prerecond_worker),
        ("centered_worker_source_sha256", args.centered_worker),
        ("composability_source_sha256", args.composability),
        ("transport_source_sha256", args.transport),
        ("chain_source_sha256", args.chain),
        ("adaptive_source_sha256", args.adaptive),
        ("event_source_sha256", args.event),
        ("base_source_sha256", args.base),
    ):
        require(payload, key, sha256(path))
    require(payload, "prerecond_receipt_sha256", sha256(args.prerecond_receipt))
    require(payload, "prerecond_receipt_sha256", PRERECOND_SHA256)
    require(payload, "transport_receipt_sha256", sha256(args.transport_receipt))
    require(payload, "transport_receipt_sha256", TRANSPORT_SHA256)
    require(payload, "tile_id", "XLEL")
    require(payload, "witness_path", WITNESS_PATH)
    require(payload, "production_time_refinement_depth", 10)
    diagnostic_depth = payload.get("diagnostic_time_refinement_depth")
    if not isinstance(diagnostic_depth, int) or diagnostic_depth <= 10:
        fail("diagnostic time refinement does not exceed production depth")
    require(payload, "source_degree", 2)
    require(payload, "time_taylor_order", 12)
    require(payload, "arb_precision_bits", 256)
    require(payload, "symbolic_policy", "original_six_variables_no_qr_renumbering")
    require(payload, "diagnostic_complete", True)
    require(payload, "full_transport_attempted", False)
    require(payload, "point_fallback_used", False)
    require(payload, "box_flattening_used", False)
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)
    validate_domain(payload.get("witness_domain"))

    reconstruction = payload.get("reconstruction")
    if not isinstance(reconstruction, dict):
        fail("witness reconstruction is absent")
    require(reconstruction, "prerecond_accepted_power", 12)
    positive_weights(reconstruction.get("raw_projection_weights"), "raw projection weights")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks if isinstance(item, dict)]
    if len(names) != len(checks) or len(names) != len(set(names)):
        fail("implementation checks are malformed or duplicated")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)
    if "lineage_reconditioner_active_for_witness_event" not in names:
        fail("lineage reconditioner control is absent")
    if "production_reconditioner_active_before_replay" not in names:
        fail("production reconditioner control is absent")

    diagnostic = payload.get("diagnostic")
    if not isinstance(diagnostic, dict):
        fail("diagnostic payload is absent")
    require(diagnostic, "production_boundary_reproduced", True)
    last_negative = validate_endpoint(
        diagnostic.get("last_strict_negative"), "last strict-negative endpoint"
    )
    if last_negative[1] >= 0:
        fail("last strict-negative endpoint is not strictly negative")
    validate_ambiguity(diagnostic.get("first_ambiguous"), "first ambiguous tube")
    production = diagnostic.get("production_boundary")
    validate_ambiguity(production, "production boundary", 10)
    if production.get("failure_class") != "SECOND_PRIOR_ORIENTATION_UNRESOLVED":
        fail("production boundary has the wrong refusal class")

    classification = payload.get("classification")
    if classification not in CLASSIFICATIONS:
        fail("classification is outside the closed diagnostic set")
    if not checks_passed:
        expected = "IMPLEMENTATION_INCONSISTENCY"
    elif diagnostic.get("accepted") is True:
        expected = "EVENT_REFINEMENT_BUDGET_LIMIT"
        validate_accepted_projection(diagnostic.get("accepted_projection"))
        if diagnostic.get("accepted_after_production_boundary") is not True:
            fail("local Newton accepted before reproducing the frozen refusal")
    else:
        terminal = diagnostic.get("terminal_ambiguous")
        terminal_refusal = diagnostic.get("terminal_refusal")
        if terminal is None:
            if not isinstance(terminal_refusal, dict):
                fail("non-accepted diagnostic has no terminal witness")
            if terminal_refusal.get("failure_class") not in {
                "PICARD_NO_CLOSURE", "PICARD_NONCONTRACTION"
            }:
                fail("terminal refusal is not an enclosure-closure failure")
            if terminal_refusal.get("time_depth") != diagnostic_depth:
                fail("terminal refusal did not exhaust the diagnostic depth")
            expected = "WITNESS_ENCLOSURE_UNRESOLVED"
        else:
            terminal_strict = validate_ambiguity(
                terminal, "terminal ambiguous tube", diagnostic_depth
            )
            expected = (
                "WITNESS_ENCLOSURE_UNRESOLVED"
                if terminal_strict
                else "WITNESS_TRANSVERSALITY_UNRESOLVED"
            )
        if diagnostic.get("status") != expected:
            fail("diagnostic status disagrees with terminal derivative sign")
    if classification != expected:
        fail(f"classification mismatch: expected {expected}, got {classification}")

    print(f"CLASSIFICATION={classification}")
    print("PRODUCTION_BOUNDARY_REPRODUCED=true")
    print("ORIGINAL_SYMBOLIC_VARIABLES=6")
    print("WITNESS_EVENT_VERIFY=PASS")


if __name__ == "__main__":
    main()
