#!/usr/bin/env python3
"""Fail-closed exact verifier for the witness derivative width budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-witness-derivative-budget.v1"
EXPECTED_RECEIPT_SHA256 = "76115e2b3e7dee3a2a3b85fe91c15250f25e3f8643efe4ee56a42a9a68a2f8b7"
VARIABLES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
ANALYSES = ("production_before", "terminal_before")
GROUPS = ("constant", "linear", "pure_quadratic", "mixed_quadratic", "interval_remainder")
REMAINDER_PARTS = {
    "truncated_degree_gt_2",
    "u_polynomial_times_v_remainder",
    "v_polynomial_times_u_remainder",
    "u_remainder_times_v_remainder",
    "minus_w_remainder",
}
CLASSIFICATIONS = {
    "IMPLEMENTATION_INCONSISTENCY",
    "ONE_SPLIT_TRANSVERSALITY_CERTIFIED",
    "DERIVATIVE_INTERVAL_REMAINDER_DOMINANT",
    "DERIVATIVE_SYMBOLIC_DIRECTION_DOMINANT",
    "DERIVATIVE_BUDGET_MIXED",
}


def fail(message: str) -> None:
    raise SystemExit(f"derivative budget verify error: {message}")


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


def interval_add(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def interval_mul(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]
) -> tuple[Fraction, Fraction]:
    products = (
        left[0] * right[0], left[0] * right[1],
        left[1] * right[0], left[1] * right[1],
    )
    return min(products), max(products)


def monomial_range(monomial: tuple[int, ...]) -> tuple[Fraction, Fraction]:
    if not any(monomial):
        return Fraction(1), Fraction(1)
    if any(exponent % 2 for exponent in monomial):
        return Fraction(-1), Fraction(1)
    return Fraction(0), Fraction(1)


def term_kind(monomial: tuple[int, ...]) -> str:
    degree = sum(monomial)
    if degree == 0:
        return "constant"
    if degree == 1:
        return "linear"
    if any(exponent == 2 for exponent in monomial):
        return "pure_quadratic"
    return "mixed_quadratic"


def validate_budget(value: object, label: str, expect_parts: bool) -> tuple[Fraction, Fraction]:
    if not isinstance(value, dict):
        fail(f"{label} is absent")
    total_range = interval(value.get("range"), f"{label} range")
    total_width = total_range[1] - total_range[0]
    if rational(value.get("width_q"), f"{label} width") != total_width:
        fail(f"{label} width disagrees with range")
    if rational(value.get("midpoint_q"), f"{label} midpoint") != sum(total_range) / 2:
        fail(f"{label} midpoint disagrees with range")
    if rational(value.get("radius_q"), f"{label} radius") != total_width / 2:
        fail(f"{label} radius disagrees with range")

    remainder = interval(value.get("remainder"), f"{label} remainder")
    remainder_width = remainder[1] - remainder[0]
    if rational(value.get("remainder_width_q"), f"{label} remainder width") != remainder_width:
        fail(f"{label} remainder width disagrees with interval")

    terms = value.get("terms")
    if not isinstance(terms, list) or not terms:
        fail(f"{label} terms are absent")
    seen: set[tuple[int, ...]] = set()
    polynomial = (Fraction(0), Fraction(0))
    groups = {name: Fraction(0) for name in GROUPS}
    groups["interval_remainder"] = remainder_width
    variables = {name: Fraction(0) for name in VARIABLES}
    constant_uncertainty = Fraction(0)
    for index, term in enumerate(terms):
        if not isinstance(term, dict):
            fail(f"{label} term {index} is malformed")
        raw_monomial = term.get("monomial")
        if not isinstance(raw_monomial, list) or len(raw_monomial) != len(VARIABLES):
            fail(f"{label} term {index} has malformed monomial")
        try:
            monomial = tuple(int(exponent) for exponent in raw_monomial)
        except (TypeError, ValueError):
            fail(f"{label} term {index} has noninteger exponent")
        if any(exponent < 0 for exponent in monomial) or sum(monomial) > 2:
            fail(f"{label} term {index} lies outside degree two")
        if monomial in seen:
            fail(f"{label} has duplicate monomial")
        seen.add(monomial)
        coefficient = interval(term.get("coefficient"), f"{label} coefficient {index}")
        contribution = interval_mul(coefficient, monomial_range(monomial))
        if interval(term.get("range_contribution"), f"{label} contribution {index}") != contribution:
            fail(f"{label} term {index} contribution is incorrect")
        width = contribution[1] - contribution[0]
        if rational(term.get("width_q"), f"{label} term width {index}") != width:
            fail(f"{label} term {index} width is incorrect")
        kind = term_kind(monomial)
        if term.get("kind") != kind:
            fail(f"{label} term {index} kind is incorrect")
        support = [VARIABLES[i] for i, exponent in enumerate(monomial) if exponent]
        if term.get("support") != support:
            fail(f"{label} term {index} support is incorrect")
        groups[kind] += width
        if support:
            for variable in support:
                variables[variable] += width / len(support)
        else:
            constant_uncertainty += width
        polynomial = interval_add(polynomial, contribution)

    if interval_add(polynomial, remainder) != total_range:
        fail(f"{label} total range is not polynomial plus remainder")
    raw_groups = value.get("group_widths_q")
    if not isinstance(raw_groups, dict) or set(raw_groups) != set(GROUPS):
        fail(f"{label} width groups are malformed")
    for name, expected in groups.items():
        if rational(raw_groups[name], f"{label} group {name}") != expected:
            fail(f"{label} group {name} is incorrect")
    width_sum = sum(groups.values(), Fraction(0))
    if rational(value.get("width_sum_q"), f"{label} width sum") != width_sum:
        fail(f"{label} width sum is incorrect")
    if value.get("width_sum_identity") is not (width_sum == total_width):
        fail(f"{label} width identity flag is incorrect")

    raw_variables = value.get("variable_attributed_widths_q")
    if not isinstance(raw_variables, dict) or set(raw_variables) != set(VARIABLES):
        fail(f"{label} variable attribution is malformed")
    for name, expected in variables.items():
        if rational(raw_variables[name], f"{label} variable {name}") != expected:
            fail(f"{label} variable attribution for {name} is incorrect")
    attributed = sum(variables.values(), Fraction(0)) + constant_uncertainty + remainder_width
    if rational(value.get("constant_coefficient_uncertainty_width_q"), f"{label} constant uncertainty") != constant_uncertainty:
        fail(f"{label} constant uncertainty is incorrect")
    if rational(value.get("attributed_width_sum_q"), f"{label} attributed sum") != attributed:
        fail(f"{label} attributed sum is incorrect")
    if value.get("attribution_sum_identity") is not (attributed == total_width):
        fail(f"{label} attribution identity flag is incorrect")
    expected_rank = sorted(variables.items(), key=lambda item: item[1], reverse=True)
    ranked = value.get("ranked_variables")
    if not isinstance(ranked, list) or len(ranked) != len(VARIABLES):
        fail(f"{label} ranked variables are malformed")
    for item, (name, width) in zip(ranked, expected_rank, strict=True):
        if not isinstance(item, dict) or item.get("variable") != name:
            fail(f"{label} ranked variable order is incorrect")
        if rational(item.get("attributed_width_q"), f"{label} ranked width") != width:
            fail(f"{label} ranked variable width is incorrect")

    parts = value.get("remainder_parts")
    if expect_parts:
        if not isinstance(parts, list) or len(parts) != len(REMAINDER_PARTS):
            fail(f"{label} remainder decomposition is malformed")
        part_sum = (Fraction(0), Fraction(0))
        names: set[str] = set()
        widths: list[Fraction] = []
        for index, part in enumerate(parts):
            if not isinstance(part, dict) or not isinstance(part.get("name"), str):
                fail(f"{label} remainder part {index} is malformed")
            names.add(part["name"])
            part_interval = interval(part.get("interval"), f"{label} remainder part {index}")
            width = part_interval[1] - part_interval[0]
            widths.append(width)
            if rational(part.get("width_q"), f"{label} remainder part width {index}") != width:
                fail(f"{label} remainder part {index} width is incorrect")
            expected_fraction = width / remainder_width if remainder_width else Fraction(0)
            if rational(part.get("fraction_of_total_remainder_q"), f"{label} remainder fraction {index}") != expected_fraction:
                fail(f"{label} remainder part {index} fraction is incorrect")
            part_sum = interval_add(part_sum, part_interval)
        if names != REMAINDER_PARTS or widths != sorted(widths, reverse=True):
            fail(f"{label} remainder parts have wrong names or order")
        if part_sum != remainder:
            fail(f"{label} remainder parts do not sum to remainder")
        if rational(value.get("remainder_parts_width_sum_q"), f"{label} remainder width sum") != sum(widths, Fraction(0)):
            fail(f"{label} remainder parts width sum is incorrect")
        if value.get("remainder_parts_width_sum_identity") is not (sum(widths, Fraction(0)) == remainder_width):
            fail(f"{label} remainder parts width identity is incorrect")
    elif parts is not None:
        fail(f"{label} unexpectedly contains a remainder decomposition")
    return total_range


def split_bounds(
    bounds: dict[str, object], variable: str, side: str
) -> dict[str, list[str]]:
    expected = {name: list(value) for name, value in bounds.items()}
    lower, upper = interval(expected[variable], f"source domain {variable}")
    cut = (lower + upper) / 2
    child = (lower, cut) if side == "LEFT" else (cut, upper)
    expected[variable] = [str(child[0]), str(child[1])]
    return expected


def validate_analysis(value: object, label: str, bounds: dict[str, object]) -> tuple[bool, Fraction]:
    if not isinstance(value, dict):
        fail(f"{label} analysis is absent")
    parent = value.get("derivative_budget")
    validate_budget(parent, f"{label} parent", True)
    assert isinstance(parent, dict)
    parent_radius = rational(parent.get("radius_q"), f"{label} parent radius")
    scans = value.get("one_level_split_scan")
    if not isinstance(scans, list) or len(scans) != len(VARIABLES):
        fail(f"{label} split scan is malformed")
    variables: set[str] = set()
    factors: list[Fraction] = []
    any_positive = False
    for scan_index, scan in enumerate(scans):
        if not isinstance(scan, dict) or scan.get("variable") not in VARIABLES:
            fail(f"{label} split {scan_index} has invalid variable")
        variable = str(scan["variable"])
        variables.add(variable)
        children = scan.get("children")
        if not isinstance(children, list) or [child.get("side") for child in children if isinstance(child, dict)] != ["LEFT", "RIGHT"]:
            fail(f"{label} split {variable} children are malformed")
        radii: list[Fraction] = []
        lowers: list[Fraction] = []
        for child in children:
            assert isinstance(child, dict)
            side = str(child["side"])
            if child.get("domain") != split_bounds(bounds, variable, side):
                fail(f"{label} split {variable} {side} has wrong domain")
            child_range = validate_budget(
                child.get("derivative_budget"),
                f"{label} split {variable} {side}",
                False,
            )
            budget_value = child["derivative_budget"]
            assert isinstance(budget_value, dict)
            radii.append(rational(budget_value.get("radius_q"), f"{label} child radius"))
            lowers.append(child_range[0])
        worst_radius = max(radii)
        factor = parent_radius / worst_radius if worst_radius else Fraction(0)
        factors.append(factor)
        if rational(scan.get("worst_child_lower_q"), f"{label} worst lower") != min(lowers):
            fail(f"{label} split {variable} worst lower is incorrect")
        if rational(scan.get("worst_child_radius_q"), f"{label} worst radius") != worst_radius:
            fail(f"{label} split {variable} worst radius is incorrect")
        if rational(scan.get("radius_contraction_factor_q"), f"{label} factor") != factor:
            fail(f"{label} split {variable} contraction is incorrect")
        both_positive = min(lowers) > 0
        if scan.get("both_children_strictly_positive") is not both_positive:
            fail(f"{label} split {variable} positivity flag is incorrect")
        any_positive |= both_positive
    if variables != set(VARIABLES) or factors != sorted(factors, reverse=True):
        fail(f"{label} split variables or ranking are incorrect")
    best = scans[0]
    if value.get("best_split_variable") != best.get("variable"):
        fail(f"{label} best split variable is incorrect")
    if rational(value.get("best_split_contraction_factor_q"), f"{label} best factor") != factors[0]:
        fail(f"{label} best split factor is incorrect")
    if value.get("one_split_certifies_transversality") is not any_positive:
        fail(f"{label} one-split conclusion is incorrect")
    return any_positive, rational(parent.get("remainder_width_q"), f"{label} remainder") / rational(parent.get("width_q"), f"{label} width")


def validate_observed(value: object, source: object, label: str) -> None:
    if not isinstance(value, dict) or not isinstance(source, dict):
        fail(f"{label} observed tube is absent")
    tube = source.get("tube")
    if not isinstance(tube, dict):
        fail(f"{label} source tube is absent")
    derivative = interval(value.get("derivative"), f"{label} derivative")
    if derivative != interval(tube.get("derivative"), f"{label} source derivative"):
        fail(f"{label} derivative differs from witness receipt")
    require(value, "time_depth", source.get("time_depth"))
    require(value, "step_q", source.get("step_q"))
    midpoint = sum(derivative) / 2
    radius = (derivative[1] - derivative[0]) / 2
    if rational(value.get("midpoint_q"), f"{label} midpoint") != midpoint:
        fail(f"{label} midpoint is incorrect")
    if rational(value.get("radius_q"), f"{label} radius") != radius:
        fail(f"{label} radius is incorrect")
    expected = radius / midpoint if midpoint > 0 else Fraction(0)
    if rational(value.get("fixed_midpoint_required_contraction_q"), f"{label} required contraction") != expected:
        fail(f"{label} required contraction is incorrect")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--witness-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))
    witness = json.loads(args.witness_receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    require(payload, "worker_source_sha256", sha256(args.worker))
    require(payload, "witness_receipt_sha256", sha256(args.witness_receipt))
    require(payload, "witness_receipt_sha256", EXPECTED_RECEIPT_SHA256)
    require(payload, "arithmetic", "fractions.Fraction exact rational interval arithmetic")
    require(payload, "variables", list(VARIABLES))
    require(payload, "witness_domain", witness["witness_domain"]["bounds"])
    require(payload, "diagnostic_complete", True)
    require(payload, "full_transport_attempted", False)
    require(payload, "interval_newton_attempted", False)
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    diagnostic = witness["diagnostic"]
    validate_observed(payload.get("observed_production_tube"), diagnostic["production_boundary"], "production")
    validate_observed(payload.get("observed_terminal_tube"), diagnostic["terminal_ambiguous"], "terminal")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks if isinstance(item, dict)]
    if len(names) != len(checks) or len(names) != len(set(names)):
        fail("implementation checks are malformed or duplicated")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    analyses = payload.get("analyses")
    if not isinstance(analyses, dict) or set(analyses) != set(ANALYSES):
        fail("analysis set is not closed")
    results = {
        name: validate_analysis(analyses[name], name, payload["witness_domain"])
        for name in ANALYSES
    }
    terminal_positive, terminal_remainder_fraction = results["terminal_before"]
    terminal_budget = analyses["terminal_before"]["derivative_budget"]
    leading_width = rational(terminal_budget["ranked_variables"][0]["attributed_width_q"], "terminal leading variable")
    terminal_width = rational(terminal_budget["width_q"], "terminal width")
    classification = payload.get("classification")
    if classification not in CLASSIFICATIONS:
        fail("classification lies outside the closed result set")
    if not checks_passed:
        expected = "IMPLEMENTATION_INCONSISTENCY"
    elif terminal_positive:
        expected = "ONE_SPLIT_TRANSVERSALITY_CERTIFIED"
    elif terminal_remainder_fraction > Fraction(1, 2):
        expected = "DERIVATIVE_INTERVAL_REMAINDER_DOMINANT"
    elif leading_width * 2 > terminal_width:
        expected = "DERIVATIVE_SYMBOLIC_DIRECTION_DOMINANT"
    else:
        expected = "DERIVATIVE_BUDGET_MIXED"
    if classification != expected:
        fail(f"classification is {classification!r}, expected {expected!r}")

    print("CS6_WITNESS_DERIVATIVE_BUDGET_VERIFIED=true")


if __name__ == "__main__":
    main()
