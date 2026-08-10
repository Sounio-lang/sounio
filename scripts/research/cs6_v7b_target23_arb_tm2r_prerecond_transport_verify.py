#!/usr/bin/env python3
"""Fail-closed verifier for the lineage-preserving pre-QR transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-transport.v1"
EXPECTED_PRERECOND_SHA256 = (
    "4b615c5632ba9537d639d4fe831c924aff1586a0d4a9db1f2f4efd9c1f1daa3a"
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
VARIABLES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")


def fail(message: str) -> None:
    raise SystemExit(f"prerecond transport verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{label} is not an interval")
    try:
        lower, upper = (Fraction(str(item)) for item in value)
    except (ValueError, ZeroDivisionError) as error:
        fail(f"{label} has a non-rational endpoint: {error}")
    if lower > upper:
        fail(f"{label} has reversed endpoints")
    return lower, upper


def weight_presence(value: object, label: str) -> bool:
    if not isinstance(value, list) or len(value) != len(VARIABLES):
        fail(f"{label} does not contain six weights")
    present = True
    for index, item in enumerate(value):
        lower, upper = interval(item, f"{label}[{index}]")
        if lower < 0:
            fail(f"{label}[{index}] is negative")
        present = present and upper > 0
    return present


def positive_weights(value: object, label: str) -> None:
    if not weight_presence(value, label):
        fail(f"{label} does not preserve all six variables")


def exact_zero_component(value: object, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} is not a TM2R component")
    coefficients = value.get("coefficients")
    if not isinstance(coefficients, list):
        fail(f"{label} coefficients are absent")
    for index, coefficient in enumerate(coefficients):
        if not isinstance(coefficient, dict):
            fail(f"{label} coefficient {index} is malformed")
        if interval(coefficient.get("interval"), f"{label} coefficient {index}") != (0, 0):
            fail(f"{label} coefficient {index} is not exact zero")
    if interval(value.get("remainder"), f"{label} remainder") != (0, 0):
        fail(f"{label} remainder is not exact zero")


def validate_components(value: object, label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        fail(f"{label} does not contain four phase-space components")
    exact_zero_component(value[2], f"{label} w")


def validate_domain(value: object, label: str) -> list[str]:
    if not isinstance(value, dict):
        fail(f"{label} is absent")
    bounds = value.get("bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(VARIABLES):
        fail(f"{label} does not bind exactly the six symbolic variables")
    for variable in VARIABLES:
        lower, upper = interval(bounds[variable], f"{label} {variable}")
        if lower >= upper:
            fail(f"{label} {variable} is not a nondegenerate interval")
    lineage = value.get("split_lineage")
    trace = value.get("split_trace")
    if not isinstance(lineage, list) or not all(isinstance(item, str) for item in lineage):
        fail(f"{label} lineage is malformed")
    if not isinstance(trace, list) or len(trace) != len(lineage):
        fail(f"{label} trace does not match its lineage")
    reconstructed = {variable: (Fraction(-1), Fraction(1)) for variable in VARIABLES}
    for index, step in enumerate(trace):
        if not isinstance(step, dict):
            fail(f"{label} trace step {index} is malformed")
        variable_name = step.get("variable")
        if not isinstance(variable_name, str) or variable_name.lower() not in reconstructed:
            fail(f"{label} trace step {index} names an unknown variable")
        variable = variable_name.lower()
        side = step.get("side")
        if side not in {"LEFT", "RIGHT"}:
            fail(f"{label} trace step {index} has an invalid side")
        parent = interval(step.get("parent"), f"{label} trace parent {index}")
        if parent != reconstructed[variable]:
            fail(f"{label} trace step {index} has the wrong parent")
        cut = Fraction(str(step.get("cut")))
        if cut != (parent[0] + parent[1]) / 2:
            fail(f"{label} trace step {index} is not an exact bisection")
        expected_child = (
            (parent[0], cut) if side == "LEFT" else (cut, parent[1])
        )
        child = interval(step.get("child"), f"{label} trace child {index}")
        if child != expected_child:
            fail(f"{label} trace step {index} has the wrong child")
        suffix = "L" if side == "LEFT" else "H"
        if lineage[index] != variable_name + suffix:
            fail(f"{label} trace step {index} disagrees with its lineage")
        expected_center = "-1/2" if side == "LEFT" else "1/2"
        if step.get("tm2r_substitution_center") != expected_center:
            fail(f"{label} trace step {index} has the wrong TM2R center")
        if step.get("tm2r_substitution_radius") != "1/2":
            fail(f"{label} trace step {index} has the wrong TM2R radius")
        reconstructed[variable] = child
    for variable in VARIABLES:
        if interval(bounds[variable], f"{label} final {variable}") != reconstructed[variable]:
            fail(f"{label} final {variable} bounds disagree with the split trace")
    return lineage


def lineage_cover(root: list[str], leaves: list[list[str]]) -> bool:
    terminal = {tuple(path) for path in leaves}
    prefix = tuple(root)
    if not terminal or any(path[: len(prefix)] != prefix for path in terminal):
        return False

    def covers(current: tuple[str, ...]) -> bool:
        if current in terminal:
            return not any(
                path != current and path[: len(current)] == current for path in terminal
            )
        tokens = {
            path[len(current)]
            for path in terminal
            if len(path) > len(current) and path[: len(current)] == current
        }
        variables = {token[:-1] for token in tokens if token[-1:] in {"L", "H"}}
        if len(variables) != 1:
            return False
        variable = next(iter(variables))
        return tokens == {variable + "L", variable + "H"} and covers(
            current + (variable + "L",)
        ) and covers(current + (variable + "H",))

    return covers(prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--prerecond-worker", type=Path, required=True)
    parser.add_argument("--centered-worker", type=Path, required=True)
    parser.add_argument("--composability", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prerecond-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    for key, path in (
        ("worker_source_sha256", args.worker),
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
    require(
        payload,
        "prior_prerecond_receipt_sha256",
        sha256(args.prerecond_receipt),
    )
    require(payload, "prior_prerecond_receipt_sha256", EXPECTED_PRERECOND_SHA256)
    require(payload, "tile_id", "XLEL")
    require(payload, "critical_path", EXPECTED_PATH)
    require(payload, "critical_depth", len(EXPECTED_PATH))
    require(payload, "source_degree", 2)
    require(payload, "time_taylor_order", 12)
    require(payload, "arb_precision_bits", 256)
    require(payload, "prerecond_accepted_power", 12)
    require(payload, "symbolic_transport_policy", "original_six_variables_no_qr_renumbering")
    require(payload, "diagnostic_complete", True)
    require(payload, "point_fallback_used", False)
    require(payload, "box_flattening_used", False)
    require(payload, "full_transport_attempted", True)
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    root_lineage = validate_domain(payload.get("critical_domain"), "critical domain")
    validate_components(payload.get("raw_projection_components"), "raw projection")
    positive_weights(payload.get("raw_projection_variable_weights"), "raw projection weights")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks if isinstance(item, dict)]
    if len(names) != len(checks) or len(names) != len(set(names)):
        fail("implementation checks are malformed or non-unique")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    transport = payload.get("transport")
    if not isinstance(transport, dict):
        fail("transport result is absent")
    require(
        transport,
        "reconditioner",
        "__main__.lineage_preserving_recondition",
    )
    require(transport, "split_depth_limit", 8)
    require(transport, "split_node_limit", 255)
    require(transport, "stop_after_first_unresolved", True)
    for count_key in (
        "outward_stabilization_checks",
        "split_nodes",
        "split_reconstructions",
    ):
        if not isinstance(transport.get(count_key), int) or transport[count_key] < 0:
            fail(f"transport {count_key} is not a nonnegative integer")
    if transport["split_nodes"] > transport["split_node_limit"]:
        fail("transport exceeded its split-node limit")
    unresolved = transport.get("unresolved")
    carriers = transport.get("carriers")
    if not isinstance(unresolved, list) or not isinstance(carriers, list):
        fail("transport branches are malformed")

    complete_summary = transport.get("complete") is True
    all_preserved = True
    leaf_lineages: list[list[str]] = []
    carrier_event_times: list[tuple[Fraction, Fraction]] = []
    carrier_derivatives: list[tuple[Fraction, Fraction]] = []
    carrier_normals: list[tuple[Fraction, Fraction]] = []
    for index, carrier in enumerate(carriers):
        if not isinstance(carrier, dict):
            fail(f"carrier {index} is malformed")
        validate_components(carrier.get("components"), f"carrier {index}")
        carrier_preserved = weight_presence(
            carrier.get("variable_weights"), f"carrier {index} weights"
        )
        if carrier.get("all_six_variables_preserved") is not carrier_preserved:
            fail(f"carrier {index} symbolic-preservation summary is inconsistent")
        all_preserved = all_preserved and carrier_preserved
        carrier_event_times.append(
            interval(carrier.get("event_time"), f"carrier {index} event time")
        )
        carrier_derivatives.append(
            interval(carrier.get("event_derivative"), f"carrier {index} derivative")
        )
        carrier_normals.append(
            interval(carrier.get("event_normal"), f"carrier {index} normal")
        )
        if carrier_derivatives[-1][0] <= 0 or carrier_normals[-1][0] <= 0:
            fail(f"carrier {index} is not strictly upward-transversal")
        leaf_lineages.append(validate_domain(carrier.get("domain"), f"carrier {index} domain"))

    if complete_summary:
        if unresolved or not carriers:
            fail("complete transport has unresolved or no terminal carriers")
        require(transport, "status", "COMPLETE")
        require(transport, "terminal_domain_cover_certified", True)
        require(transport, "all_six_variables_preserved", True)
        if not all_preserved:
            fail("complete transport has a carrier without all six variables")
        if not lineage_cover(root_lineage, leaf_lineages):
            fail("terminal carrier lineages do not exactly cover the critical domain")
        if not isinstance(transport.get("terminal_domain_cover_checks"), int) or (
            transport["terminal_domain_cover_checks"] <= 0
        ):
            fail("complete transport lacks domain-cover checks")
    else:
        if transport.get("status") not in {
            "TRANSPORT_REFUSED",
            "FINAL_SYMBOLIC_DEPENDENCE_LOST",
        }:
            fail("incomplete transport has an unclassified status")
        if not unresolved and not carriers:
            fail("incomplete transport records neither a refusal nor a terminal carrier")
        if transport.get("status") == "TRANSPORT_REFUSED":
            if len(unresolved) != 1:
                fail("fail-fast refused transport must retain exactly one witness")
            if transport.get("terminal_domain_cover_certified") is True:
                fail("refused transport claims a terminal domain cover")
            for key in ("event_time", "event_derivative", "event_normal"):
                if key in transport:
                    fail(f"refused transport exposes a partial {key} hull")
        else:
            require(transport, "terminal_domain_cover_certified", True)
            if unresolved or not carriers or not lineage_cover(root_lineage, leaf_lineages):
                fail("symbolic-loss result lacks a complete terminal domain cover")
    if not unresolved and carriers:
        event_hull = interval(transport.get("event_time"), "transport event time")
        derivative_hull = interval(
            transport.get("event_derivative"), "transport derivative"
        )
        normal_hull = interval(
            transport.get("event_normal"), "transport normal"
        )
        expected_hulls = (
            (
                min(item[0] for item in carrier_event_times),
                max(item[1] for item in carrier_event_times),
            ),
            (
                min(item[0] for item in carrier_derivatives),
                max(item[1] for item in carrier_derivatives),
            ),
            (
                min(item[0] for item in carrier_normals),
                max(item[1] for item in carrier_normals),
            ),
        )
        if (event_hull, derivative_hull, normal_hull) != expected_hulls:
            fail("transport hulls are not the exact hulls of all terminal carriers")
        if derivative_hull[0] <= 0 or normal_hull[0] <= 0:
            fail("transport hull is not strictly upward-transversal")
    require(transport, "all_six_variables_preserved", all_preserved)

    recomputed_complete = (
        checks_passed
        and complete_summary
        and not unresolved
        and bool(carriers)
        and transport.get("terminal_domain_cover_certified") is True
        and transport.get("all_six_variables_preserved") is True
        and all_preserved
    )
    require(payload, "next_return_complete", recomputed_complete)
    expected_classification = (
        "IMPLEMENTATION_INCONSISTENCY"
        if not checks_passed
        else (
            "PRERECOND_NEXT_RETURN_COMPLETE"
            if recomputed_complete
            else (
                "PRERECOND_FINAL_SYMBOLIC_DEPENDENCE_LOST"
                if transport.get("status") == "FINAL_SYMBOLIC_DEPENDENCE_LOST"
                else "PRERECOND_NEXT_RETURN_REFUSED"
            )
        )
    )
    require(payload, "classification", expected_classification)

    print(f"SCHEMA={SCHEMA}")
    print(f"CLASSIFICATION={expected_classification}")
    print(f"IMPLEMENTATION_CHECKS_PASSED={str(checks_passed).lower()}")
    print(f"NEXT_RETURN_COMPLETE={str(recomputed_complete).lower()}")
    print(f"TERMINAL_CARRIERS={len(carriers)}")
    print("SYMBOLIC_TRANSPORT_POLICY=original_six_variables_no_qr_renumbering")
    print("COVERING_RELATION_CERTIFIED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
