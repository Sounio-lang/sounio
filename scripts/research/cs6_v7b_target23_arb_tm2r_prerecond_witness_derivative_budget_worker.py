#!/usr/bin/env python3
"""Exact-rational width budget for the refused pre-QR witness derivative."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-prerecond-witness-derivative-budget.v1"
EXPECTED_RECEIPT_SHA256 = "76115e2b3e7dee3a2a3b85fe91c15250f25e3f8643efe4ee56a42a9a68a2f8b7"
VARIABLES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
ZERO = (0,) * len(VARIABLES)
ZS = Fraction("22.3274637391")


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("reversed interval")

    @classmethod
    def point(cls, value: Fraction | int) -> "Interval":
        exact = Fraction(value)
        return cls(exact, exact)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: "Interval") -> "Interval":
        return self + (-other)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    def scale(self, value: Fraction | int) -> "Interval":
        return self * Interval.point(value)

    def width(self) -> Fraction:
        return self.upper - self.lower

    def midpoint(self) -> Fraction:
        return (self.lower + self.upper) / 2

    def radius(self) -> Fraction:
        return self.width() / 2

    def as_json(self) -> list[str]:
        return [str(self.lower), str(self.upper)]


@dataclass
class Component:
    coefficients: dict[tuple[int, ...], Interval]
    remainder: Interval


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_interval(value: object) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("malformed interval")
    return Interval(Fraction(str(value[0])), Fraction(str(value[1])))


def parse_component(value: object) -> Component:
    if not isinstance(value, dict):
        raise ValueError("malformed TM2R component")
    coefficients: dict[tuple[int, ...], Interval] = {}
    raw_coefficients = value.get("coefficients")
    if not isinstance(raw_coefficients, list):
        raise ValueError("missing TM2R coefficients")
    for item in raw_coefficients:
        if not isinstance(item, dict):
            raise ValueError("malformed TM2R coefficient")
        monomial = tuple(int(exponent) for exponent in item["monomial"])
        if len(monomial) != len(VARIABLES) or sum(monomial) > 2:
            raise ValueError("coefficient lies outside TM2R degree-2 support")
        if monomial in coefficients:
            raise ValueError("duplicate TM2R monomial")
        coefficients[monomial] = parse_interval(item["interval"])
    return Component(coefficients, parse_interval(value.get("remainder")))


def monomial_range(monomial: tuple[int, ...]) -> Interval:
    if not any(monomial):
        return Interval.point(1)
    if any(exponent % 2 for exponent in monomial):
        return Interval(Fraction(-1), Fraction(1))
    return Interval(Fraction(0), Fraction(1))


def add_component(left: Component, right: Component) -> Component:
    keys = left.coefficients.keys() | right.coefficients.keys()
    zero = Interval.point(0)
    return Component(
        {
            key: left.coefficients.get(key, zero) + right.coefficients.get(key, zero)
            for key in keys
        },
        left.remainder + right.remainder,
    )


def negate_component(value: Component) -> Component:
    return Component(
        {monomial: -coefficient for monomial, coefficient in value.coefficients.items()},
        -value.remainder,
    )


def polynomial_range(value: Component) -> Interval:
    result = Interval.point(0)
    for monomial, coefficient in value.coefficients.items():
        result = result + coefficient * monomial_range(monomial)
    return result


def component_range(value: Component) -> Interval:
    return polynomial_range(value) + value.remainder


def multiply_component_with_parts(
    left: Component, right: Component
) -> tuple[Component, dict[str, Interval]]:
    retained: dict[tuple[int, ...], Interval] = {}
    tail = Interval.point(0)
    for left_monomial, left_coefficient in left.coefficients.items():
        for right_monomial, right_coefficient in right.coefficients.items():
            monomial = tuple(
                left_monomial[index] + right_monomial[index]
                for index in range(len(VARIABLES))
            )
            coefficient = left_coefficient * right_coefficient
            if sum(monomial) <= 2:
                retained[monomial] = retained.get(
                    monomial, Interval.point(0)
                ) + coefficient
            else:
                tail = tail + coefficient * monomial_range(monomial)
    parts = {
        "truncated_degree_gt_2": tail,
        "u_polynomial_times_v_remainder": polynomial_range(left) * right.remainder,
        "v_polynomial_times_u_remainder": polynomial_range(right) * left.remainder,
        "u_remainder_times_v_remainder": left.remainder * right.remainder,
    }
    cross = sum(parts.values(), Interval.point(0))
    return Component(retained, cross), parts


def multiply_component(left: Component, right: Component) -> Component:
    result, _parts = multiply_component_with_parts(left, right)
    return result


def derivative_model_with_parts(
    state: list[Component],
) -> tuple[Component, dict[str, Interval]]:
    product, parts = multiply_component_with_parts(state[0], state[1])
    result = add_component(
        product,
        negate_component(state[2]),
    )
    constant = Component({ZERO: Interval.point(-ZS)}, Interval.point(0))
    derivative = add_component(result, constant)
    remainder_parts = dict(parts)
    remainder_parts["minus_w_remainder"] = -state[2].remainder
    return derivative, remainder_parts


def derivative_model(state: list[Component]) -> Component:
    result, _parts = derivative_model_with_parts(state)
    return result


def term_kind(monomial: tuple[int, ...]) -> str:
    degree = sum(monomial)
    if degree == 0:
        return "constant"
    if degree == 1:
        return "linear"
    if any(exponent == 2 for exponent in monomial):
        return "pure_quadratic"
    return "mixed_quadratic"


def budget(
    value: Component,
    remainder_parts: dict[str, Interval] | None = None,
) -> dict[str, object]:
    terms: list[dict[str, object]] = []
    group_widths: dict[str, Fraction] = {
        "constant": Fraction(0),
        "linear": Fraction(0),
        "pure_quadratic": Fraction(0),
        "mixed_quadratic": Fraction(0),
        "interval_remainder": value.remainder.width(),
    }
    variable_widths = {variable: Fraction(0) for variable in VARIABLES}
    constant_uncertainty = Fraction(0)
    for monomial, coefficient in sorted(value.coefficients.items()):
        contribution = coefficient * monomial_range(monomial)
        kind = term_kind(monomial)
        width = contribution.width()
        group_widths[kind] += width
        support = [
            VARIABLES[index]
            for index, exponent in enumerate(monomial)
            if exponent
        ]
        if support:
            share = width / len(support)
            for variable in support:
                variable_widths[variable] += share
        else:
            constant_uncertainty += width
        terms.append(
            {
                "monomial": list(monomial),
                "kind": kind,
                "support": support,
                "coefficient": coefficient.as_json(),
                "range_contribution": contribution.as_json(),
                "width_q": str(width),
            }
        )
    total = component_range(value)
    width_sum = sum(group_widths.values(), Fraction(0))
    attributed_sum = (
        sum(variable_widths.values(), Fraction(0))
        + constant_uncertainty
        + value.remainder.width()
    )
    ranked_variables = sorted(
        (
            {"variable": variable, "attributed_width_q": str(width)}
            for variable, width in variable_widths.items()
        ),
        key=lambda item: Fraction(item["attributed_width_q"]),
        reverse=True,
    )
    result = {
        "range": total.as_json(),
        "width_q": str(total.width()),
        "midpoint_q": str(total.midpoint()),
        "radius_q": str(total.radius()),
        "terms": terms,
        "group_widths_q": {name: str(width) for name, width in group_widths.items()},
        "variable_attribution_policy": "split each mixed-term interval width equally across its symbolic support",
        "variable_attributed_widths_q": {
            variable: str(width) for variable, width in variable_widths.items()
        },
        "ranked_variables": ranked_variables,
        "constant_coefficient_uncertainty_width_q": str(constant_uncertainty),
        "remainder": value.remainder.as_json(),
        "remainder_width_q": str(value.remainder.width()),
        "width_sum_q": str(width_sum),
        "attributed_width_sum_q": str(attributed_sum),
        "width_sum_identity": width_sum == total.width(),
        "attribution_sum_identity": attributed_sum == total.width(),
    }
    if remainder_parts is not None:
        part_width_sum = sum(
            (part.width() for part in remainder_parts.values()), Fraction(0)
        )
        result["remainder_parts"] = [
            {
                "name": name,
                "interval": part.as_json(),
                "width_q": str(part.width()),
                "fraction_of_total_remainder_q": str(
                    part.width() / value.remainder.width()
                    if value.remainder.width()
                    else Fraction(0)
                ),
            }
            for name, part in sorted(
                remainder_parts.items(),
                key=lambda item: item[1].width(),
                reverse=True,
            )
        ]
        result["remainder_parts_width_sum_q"] = str(part_width_sum)
        result["remainder_parts_width_sum_identity"] = (
            part_width_sum == value.remainder.width()
        )
    return result


def binomial(exp: int, power: int) -> int:
    return math.comb(exp, power)


def split_component(value: Component, variable: int, side: int) -> Component:
    if side not in (-1, 1):
        raise ValueError("split side must be -1 or 1")
    center = Fraction(side, 2)
    radius = Fraction(1, 2)
    result: dict[tuple[int, ...], Interval] = {}
    for monomial, coefficient in value.coefficients.items():
        exponent = monomial[variable]
        for new_power in range(exponent + 1):
            factor = (
                Fraction(binomial(exponent, new_power))
                * center ** (exponent - new_power)
                * radius ** new_power
            )
            child = list(monomial)
            child[variable] = new_power
            child_monomial = tuple(child)
            result[child_monomial] = result.get(
                child_monomial, Interval.point(0)
            ) + coefficient.scale(factor)
    return Component(result, value.remainder)


def split_domain(
    bounds: dict[str, list[str]], variable: str, side: int
) -> dict[str, list[str]]:
    result = {name: list(value) for name, value in bounds.items()}
    lower, upper = (Fraction(token) for token in result[variable])
    cut = (lower + upper) / 2
    child = (lower, cut) if side < 0 else (cut, upper)
    result[variable] = [str(child[0]), str(child[1])]
    return result


def split_scan(
    state: list[Component], bounds: dict[str, list[str]], parent: dict[str, object]
) -> list[dict[str, object]]:
    parent_radius = Fraction(parent["radius_q"])
    scans: list[dict[str, object]] = []
    for variable_index, variable in enumerate(VARIABLES):
        children: list[dict[str, object]] = []
        child_radii: list[Fraction] = []
        child_lowers: list[Fraction] = []
        for side, side_name in ((-1, "LEFT"), (1, "RIGHT")):
            child_state = [
                split_component(component, variable_index, side)
                for component in state
            ]
            child_budget = budget(derivative_model(child_state))
            child_range = parse_interval(child_budget["range"])
            child_radii.append(Fraction(child_budget["radius_q"]))
            child_lowers.append(child_range.lower)
            children.append(
                {
                    "side": side_name,
                    "domain": split_domain(bounds, variable, side),
                    "derivative_budget": child_budget,
                }
            )
        worst_radius = max(child_radii)
        scans.append(
            {
                "variable": variable,
                "children": children,
                "worst_child_lower_q": str(min(child_lowers)),
                "worst_child_radius_q": str(worst_radius),
                "radius_contraction_factor_q": str(
                    parent_radius / worst_radius if worst_radius else Fraction(0)
                ),
                "both_children_strictly_positive": min(child_lowers) > 0,
            }
        )
    scans.sort(
        key=lambda item: Fraction(item["radius_contraction_factor_q"]),
        reverse=True,
    )
    return scans


def observed_tube_budget(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("missing observed tube")
    derivative = parse_interval(value["tube"]["derivative"])
    return {
        "time_depth": value["time_depth"],
        "step_q": value["step_q"],
        "derivative": derivative.as_json(),
        "midpoint_q": str(derivative.midpoint()),
        "radius_q": str(derivative.radius()),
        "fixed_midpoint_required_contraction_q": str(
            derivative.radius() / derivative.midpoint()
            if derivative.midpoint() > 0
            else Fraction(0)
        ),
    }


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("derivative budget requires Python >= 3.10")
    source_path = Path(__file__)
    receipt_path = (
        source_path.parent
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_prerecond_witness_event_v1"
        / "witness_event.json"
    )
    if not receipt_path.is_file():
        raise SystemExit(f"frozen witness receipt is missing: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="ascii"))
    checks: list[dict[str, object]] = []
    bool_check(checks, "witness_receipt_hash_matches", sha256(receipt_path) == EXPECTED_RECEIPT_SHA256)
    bool_check(checks, "witness_classification_matches", payload.get("classification") == "WITNESS_TRANSVERSALITY_UNRESOLVED")
    bool_check(checks, "witness_implementation_checks_passed", payload.get("implementation_checks_passed") is True)
    diagnostic = payload["diagnostic"]
    domain = payload["witness_domain"]["bounds"]
    states = {
        "production_before": diagnostic["production_boundary"]["before"]["state"]["components"],
        "terminal_before": diagnostic["terminal_ambiguous"]["before"]["state"]["components"],
    }
    analyses: dict[str, object] = {}
    for name, raw_state in states.items():
        state = [parse_component(component) for component in raw_state]
        derivative, remainder_parts = derivative_model_with_parts(state)
        derivative_budget = budget(derivative, remainder_parts)
        bool_check(checks, f"{name}_width_sum_identity", bool(derivative_budget["width_sum_identity"]))
        bool_check(checks, f"{name}_attribution_sum_identity", bool(derivative_budget["attribution_sum_identity"]))
        bool_check(checks, f"{name}_remainder_parts_width_sum_identity", bool(derivative_budget["remainder_parts_width_sum_identity"]))
        scans = split_scan(state, domain, derivative_budget)
        analyses[name] = {
            "derivative_budget": derivative_budget,
            "one_level_split_scan": scans,
            "best_split_variable": scans[0]["variable"],
            "best_split_contraction_factor_q": scans[0]["radius_contraction_factor_q"],
            "one_split_certifies_transversality": any(
                scan["both_children_strictly_positive"] for scan in scans
            ),
        }

    terminal_budget = analyses["terminal_before"]["derivative_budget"]
    terminal_total_width = Fraction(terminal_budget["width_q"])
    terminal_remainder_width = Fraction(terminal_budget["remainder_width_q"])
    ranked = terminal_budget["ranked_variables"]
    leading_variable_width = Fraction(ranked[0]["attributed_width_q"])
    one_split = analyses["terminal_before"]["one_split_certifies_transversality"]
    implementation_ok = all(item["passed"] is True for item in checks)
    if not implementation_ok:
        classification = "IMPLEMENTATION_INCONSISTENCY"
    elif one_split:
        classification = "ONE_SPLIT_TRANSVERSALITY_CERTIFIED"
    elif terminal_remainder_width * 2 > terminal_total_width:
        classification = "DERIVATIVE_INTERVAL_REMAINDER_DOMINANT"
    elif leading_variable_width * 2 > terminal_total_width:
        classification = "DERIVATIVE_SYMBOLIC_DIRECTION_DOMINANT"
    else:
        classification = "DERIVATIVE_BUDGET_MIXED"

    result = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "witness_receipt_sha256": sha256(receipt_path),
        "python_version": platform.python_version(),
        "arithmetic": "fractions.Fraction exact rational interval arithmetic",
        "variables": list(VARIABLES),
        "witness_domain": domain,
        "observed_production_tube": observed_tube_budget(diagnostic["production_boundary"]),
        "observed_terminal_tube": observed_tube_budget(diagnostic["terminal_ambiguous"]),
        "analyses": analyses,
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "classification": classification,
        "diagnostic_complete": True,
        "full_transport_attempted": False,
        "interval_newton_attempted": False,
        "covering_relation_certified": False,
        "recurrent_graph_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
