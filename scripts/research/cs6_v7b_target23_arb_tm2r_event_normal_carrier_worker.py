#!/usr/bin/env python3
"""Event-normal QR doubleton/tripleton for the refused pre-QR witness."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

import cs6_v7b_target23_arb_tm2r_prerecond_witness_event_worker as witness
import cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_worker as budget_worker


base = witness.base
adaptive = witness.adaptive
event = witness.event
composability = witness.composability
transport = witness.transport

SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-normal-carrier.v1"
EXPECTED_WITNESS_SHA256 = "76115e2b3e7dee3a2a3b85fe91c15250f25e3f8643efe4ee56a42a9a68a2f8b7"
EXPECTED_BUDGET_SHA256 = "f5b0f3ac5936c7814b20194bd13dc24e10408d4e9f9139a8c4c07c38084fbe21"
MODES = ("EVENT_NORMAL_DOUBLETON", "EVENT_NORMAL_TRIPLETON")
EXECUTION_MODES = ("PREFLIGHT", "TRANSPORT")
TARGET_IMPROVEMENT = Fraction(18)
SECTION_ROWS = (0, 1, 3)
PRIMARY_VARIABLES = 6
CARRIER_VARIABLES = 4
TOTAL_VARIABLES = PRIMARY_VARIABLES + CARRIER_VARIABLES
PRIMARY_NAMES = ("xi", "eta", "rho0", "rho1", "rho2", "rho3")
CARRIER_NAMES = ("sigma0", "sigma1", "sigma2", "sigma3")


@dataclass
class CarrierStats:
    reconditionings: int = 0
    generator_reconstructions: int = 0
    coefficient_uncertainty_generators: int = 0
    normal_direction_seeds: int = 0
    normal_direction_transports: int = 0
    kernel_direction_seeds: int = 0
    kernel_direction_transports: int = 0
    maximum_basis_inverse_row_sum: Fraction = Fraction(0)
    maximum_generator_count: int = 0
    kernel_orthogonality_checks: int = 0
    reconstruction_checks: int = 0
    normal_form_checks: int = 0
    basis_history: list[dict[str, object]] = field(default_factory=list)


ACTIVE_MODE = "EVENT_NORMAL_TRIPLETON"
STATS = CarrierStats()
CARRIER_INITIALIZED = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_json(value: arb) -> list[str]:
    return [base.lower_fraction(value), base.upper_fraction(value)]


def bool_check(checks: list[dict[str, object]], name: str, passed: bool) -> None:
    checks.append({"name": name, "passed": bool(passed)})


def arb_fraction(value: arb) -> Fraction:
    return Fraction(str(value.fmpq()))


def midpoint_fraction(value: arb) -> Fraction:
    return arb_fraction(value.mid())


def arb_interval(value: object) -> arb:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("serialized interval must contain two endpoints")
    lower = base.rational_ball(Fraction(str(value[0])))
    upper = base.rational_ball(Fraction(str(value[1])))
    return lower.union(upper)


def enable_extended_carrier() -> None:
    base.RESIDUAL_VARIABLES = TOTAL_VARIABLES - base.SOURCE_VARIABLES
    base.VARIABLES = TOTAL_VARIABLES
    base.ZERO_MONOMIAL = (0,) * TOTAL_VARIABLES


def extend_component(component: base.TM2R) -> base.TM2R:
    coefficients: dict[tuple[int, ...], arb] = {}
    for monomial, coefficient in component.coefficients.items():
        if len(monomial) == TOTAL_VARIABLES:
            key = monomial
        elif len(monomial) == PRIMARY_VARIABLES:
            key = monomial + (0,) * CARRIER_VARIABLES
        else:
            raise ValueError("TM2R component has an unexpected variable count")
        coefficients[key] = coefficient
    return base.TM2R(coefficients, arb(component.remainder))


def extend_state(state: list[base.TM2R]) -> list[base.TM2R]:
    enable_extended_carrier()
    return [extend_component(component) for component in state]


def parse_tm2r(value: object) -> base.TM2R:
    if not isinstance(value, dict):
        raise ValueError("serialized TM2R component must be an object")
    raw_coefficients = value.get("coefficients")
    if not isinstance(raw_coefficients, list):
        raise ValueError("serialized TM2R coefficients are absent")
    coefficients: dict[tuple[int, ...], arb] = {}
    for item in raw_coefficients:
        if not isinstance(item, dict):
            raise ValueError("serialized TM2R coefficient is malformed")
        monomial = tuple(int(exponent) for exponent in item["monomial"])
        if len(monomial) not in {PRIMARY_VARIABLES, TOTAL_VARIABLES} or sum(monomial) > 2:
            raise ValueError("serialized TM2R monomial lies outside degree two")
        if monomial in coefficients:
            raise ValueError("serialized TM2R monomial is duplicated")
        coefficients[monomial] = arb_interval(item["interval"])
    return base.TM2R(coefficients, arb_interval(value.get("remainder")))


def dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def normalize(vector: list[Fraction]) -> list[Fraction]:
    scale = max((abs(value) for value in vector), default=Fraction(0))
    if not scale:
        raise base.Refusal("EVENT_NORMAL_ZERO_DIRECTION", "cannot normalize a zero direction")
    return [value / scale for value in vector]


def orthogonalize(
    vector: list[Fraction], directions: list[list[Fraction]]
) -> list[Fraction]:
    result = list(vector)
    for direction in directions:
        denominator = dot(direction, direction)
        if not denominator:
            continue
        factor = dot(result, direction) / denominator
        result = [
            result[index] - factor * direction[index]
            for index in range(4)
        ]
    return result


def fraction_vector(vector: list[arb]) -> list[Fraction]:
    return [midpoint_fraction(value) for value in vector]


def event_covector(state: list[base.TM2R]) -> list[Fraction]:
    ranges = [component.range() for component in state]
    # D = x*y - w - z_s, so grad(D) = (y, x, -1, 0).
    return [
        midpoint_fraction(ranges[1]),
        midpoint_fraction(ranges[0]),
        Fraction(-1),
        Fraction(0),
    ]


def kernel_projection(
    vector: list[Fraction], normal_direction: list[Fraction], covector: list[Fraction]
) -> list[Fraction]:
    denominator = dot(covector, normal_direction)
    if not denominator:
        raise base.Refusal(
            "EVENT_NORMAL_DIRECTION_IN_KERNEL",
            "the privileged normal direction has zero event-covector pairing",
        )
    factor = dot(covector, vector) / denominator
    projected = [
        vector[index] - factor * normal_direction[index]
        for index in range(4)
    ]
    if dot(covector, projected):
        raise base.Refusal(
            "EVENT_KERNEL_PROJECTION_FAILED",
            "exact projection did not land in the event-covector kernel",
        )
    return projected


def matrix_rank(columns: list[list[Fraction]]) -> int:
    if not columns:
        return 0
    matrix = [
        [columns[column][row] for column in range(len(columns))]
        for row in range(4)
    ]
    rank = 0
    for column in range(len(columns)):
        pivot = next(
            (row for row in range(rank, 4) if matrix[row][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        if not scale:
            raise base.Refusal(
                "EVENT_NORMAL_RANK_PIVOT_FAILED",
                "exact rank elimination selected a zero pivot",
            )
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(4):
            if row == rank:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][index] - factor * matrix[rank][index]
                for index in range(len(columns))
            ]
        rank += 1
        if rank == 4:
            break
    return rank


def fixed_kernel_candidates(covector: list[Fraction]) -> list[list[Fraction]]:
    pivot = 2  # The w coefficient is exactly -1.
    if covector[pivot] != -1:
        raise base.Refusal(
            "EVENT_COVECTOR_W_COEFFICIENT_DRIFT",
            "event covector no longer has exact w coefficient -1",
        )
    candidates: list[list[Fraction]] = []
    for coordinate in (0, 1, 3):
        vector = [Fraction(0) for _ in range(4)]
        vector[coordinate] = Fraction(1)
        vector[pivot] = covector[coordinate]
        if dot(covector, vector):
            raise base.Refusal(
                "FIXED_EVENT_KERNEL_DIRECTION_FAILED",
                "fixed complement direction is not in the exact kernel",
            )
        candidates.append(vector)
    return candidates


def select_normal_direction(
    generators: list[list[arb]], transported: list[arb] | None, covector: list[Fraction]
) -> tuple[list[Fraction], bool]:
    if transported is not None:
        candidate = fraction_vector(transported)
        if any(candidate) and dot(covector, candidate):
            return normalize(candidate), True
    # Seed from the Euclidean dual of the exact rational covector. Subsequent
    # calls transport rho0, so only the first carrier construction is frozen to
    # the local normal rather than to a legacy residual coordinate.
    return normalize(list(covector)), False


def select_kernel_direction(
    generators: list[list[arb]],
    transported: list[arb] | None,
    normal_direction: list[Fraction],
    covector: list[Fraction],
) -> tuple[list[Fraction] | None, bool]:
    candidates: list[tuple[list[Fraction], bool]] = []
    if transported is not None:
        candidates.append((fraction_vector(transported), True))
    candidates.extend((fraction_vector(generator), False) for generator in generators)
    best: list[Fraction] | None = None
    best_transported = False
    best_norm = Fraction(0)
    for vector, is_transported in candidates:
        projected = kernel_projection(vector, normal_direction, covector)
        norm = sum((value * value for value in projected), Fraction(0))
        if norm > best_norm:
            best = normalize(projected)
            best_norm = norm
            best_transported = is_transported
    return best, best_transported


def event_normal_basis(
    generators: list[list[arb]],
    privileged0: list[arb] | None,
    privileged1: list[arb] | None,
    covector: list[Fraction],
) -> tuple[list[list[Fraction]], list[list[Fraction]], bool, bool]:
    normal_direction, normal_transported = select_normal_direction(
        generators, privileged0, covector
    )
    columns = [normal_direction]
    kernel_transported = False
    if ACTIVE_MODE == "EVENT_NORMAL_TRIPLETON":
        kernel_direction, kernel_transported = select_kernel_direction(
            generators, privileged1, normal_direction, covector
        )
        if kernel_direction is not None:
            columns.append(kernel_direction)

    # Complete from an analytic exact kernel basis. Greedy generator-derived
    # columns can be almost parallel and make the interval inverse unusable.
    # Tripleton already gets one deliberately selected dynamic kernel column.
    for candidate in fixed_kernel_candidates(covector):
        candidate = orthogonalize(candidate, columns[1:])
        if not any(candidate):
            continue
        if dot(covector, candidate):
            raise base.Refusal(
                "EVENT_KERNEL_ORTHOGONALITY_FAILED",
                "candidate complement direction left the exact kernel",
            )
        if matrix_rank([*columns, candidate]) > len(columns):
            columns.append(normalize(candidate))
        if len(columns) == 4:
            break
    if len(columns) != 4:
        raise base.Refusal(
            "EVENT_NORMAL_BASIS_INCOMPLETE",
            "could not complete the event-normal carrier basis",
        )
    for column in columns[1:]:
        if dot(covector, column):
            raise base.Refusal(
                "EVENT_NORMAL_KERNEL_NOT_EXACT",
                "a complement column has nonzero exact covector pairing",
            )
        STATS.kernel_orthogonality_checks += 1
    basis = [
        [columns[column][row] for column in range(4)]
        for row in range(4)
    ]
    inverse = base.fraction_inverse(basis)
    row_sum = max(sum(abs(value) for value in row) for row in inverse)
    STATS.maximum_basis_inverse_row_sum = max(
        STATS.maximum_basis_inverse_row_sum, row_sum
    )
    return basis, inverse, normal_transported, kernel_transported


def pure_carrier_key(index: int) -> tuple[int, ...]:
    monomial = [0] * base.VARIABLES
    monomial[PRIMARY_VARIABLES + index] = 1
    return tuple(monomial)


def carrier_normal_form(state: list[base.TM2R]) -> bool:
    """Recognize primary TM2 plus four independent linear carrier terms."""
    for component in state:
        if component.remainder != 0:
            return False
        for monomial in component.coefficients:
            if len(monomial) != TOTAL_VARIABLES or sum(monomial) > 2:
                return False
            carrier = monomial[PRIMARY_VARIABLES:]
            if any(carrier) and not (
                sum(monomial[:PRIMARY_VARIABLES]) == 0 and sum(carrier) == 1
            ):
                return False
    return True


def event_normal_recondition(state: list[base.TM2R]) -> list[base.TM2R]:
    """Move all non-source uncertainty into an event-normal rational carrier."""
    global CARRIER_INITIALIZED
    STATS.reconditionings += 1
    base.STATS.reconditionings += 1
    source_coefficients: list[dict[tuple[int, ...], arb]] = [
        {} for _ in range(4)
    ]
    residual_monomials: set[tuple[int, ...]] = set()
    generators: list[list[arb]] = []

    for row, component in enumerate(state):
        for monomial, coefficient in component.coefficients.items():
            midpoint = coefficient.mid()
            radius = coefficient.rad()
            if any(monomial[PRIMARY_VARIABLES:]):
                residual_monomials.add(monomial)
            else:
                source_coefficients[row][monomial] = midpoint
            if radius.upper() > 0:
                generator = [arb(0) for _ in range(4)]
                generator[row] = radius
                generators.append(generator)
                STATS.coefficient_uncertainty_generators += 1

    privileged0: list[arb] | None = None
    privileged1: list[arb] | None = None
    sigma0_key = pure_carrier_key(0)
    sigma1_key = pure_carrier_key(1)
    for monomial in sorted(residual_monomials):
        generator = [
            component.coefficients.get(monomial, arb(0)).mid()
            for component in state
        ]
        if not base.vector_nonzero(generator):
            continue
        if monomial == sigma0_key:
            privileged0 = generator
        elif monomial == sigma1_key:
            privileged1 = generator
        if all(exponent % 2 == 0 for exponent in monomial):
            half = [value / 2 for value in generator]
            for row in range(4):
                source_coefficients[row][base.ZERO_MONOMIAL] = (
                    source_coefficients[row].get(base.ZERO_MONOMIAL, arb(0))
                    + half[row]
                )
            generators.append(half)
        else:
            generators.append(generator)

    for row, component in enumerate(state):
        midpoint = component.remainder.mid()
        radius = component.remainder.rad()
        source_coefficients[row][base.ZERO_MONOMIAL] = (
            source_coefficients[row].get(base.ZERO_MONOMIAL, arb(0)) + midpoint
        )
        if radius.upper() > 0:
            generator = [arb(0) for _ in range(4)]
            generator[row] = radius
            generators.append(generator)

    if not generators:
        raise base.Refusal(
            "EVENT_NORMAL_EMPTY_GENERATOR_FAMILY",
            "event-normal reconditioning received no residual generators",
        )
    STATS.maximum_generator_count = max(STATS.maximum_generator_count, len(generators))
    covector = event_covector(state)
    basis, inverse, normal_transported, kernel_transported = event_normal_basis(
        generators,
        privileged0 if CARRIER_INITIALIZED else None,
        privileged1 if CARRIER_INITIALIZED else None,
        covector,
    )
    if normal_transported:
        STATS.normal_direction_transports += 1
    else:
        STATS.normal_direction_seeds += 1
    if ACTIVE_MODE == "EVENT_NORMAL_TRIPLETON":
        if kernel_transported:
            STATS.kernel_direction_transports += 1
        else:
            STATS.kernel_direction_seeds += 1
    CARRIER_INITIALIZED = True

    radii = [arb(0) for _ in range(4)]
    for generator in generators:
        coordinates = [
            sum(
                (
                    base.rational_ball(inverse[coordinate][row]) * generator[row]
                    for row in range(4)
                ),
                arb(0),
            )
            for coordinate in range(4)
        ]
        for coordinate, projected in enumerate(coordinates):
            radii[coordinate] += base.upper_abs(projected)
        reconstructed = [
            sum(
                (
                    base.rational_ball(basis[row][coordinate])
                    * coordinates[coordinate]
                    for coordinate in range(4)
                ),
                arb(0),
            )
            for row in range(4)
        ]
        if not all(
            enclosure.contains(component)
            for enclosure, component in zip(reconstructed, generator, strict=True)
        ):
            raise base.Refusal(
                "EVENT_NORMAL_GENERATOR_RECONSTRUCTION_FAILED",
                "exact rational basis reconstruction did not enclose a generator",
            )
        STATS.generator_reconstructions += 1
        STATS.reconstruction_checks += 1

    result: list[base.TM2R] = []
    for row in range(4):
        coefficients = dict(source_coefficients[row])
        for coordinate in range(4):
            coefficients[pure_carrier_key(coordinate)] = (
                base.rational_ball(basis[row][coordinate]) * radii[coordinate]
            )
        result.append(base.TM2R(coefficients, arb(0)))

    if not carrier_normal_form(result):
        raise base.Refusal(
            "EVENT_NORMAL_FORM_FAILED",
            "reconditioning did not restore primary-TM2 plus linear-carrier form",
        )
    STATS.normal_form_checks += 1

    if len(STATS.basis_history) < 8 or STATS.reconditionings % 256 == 0:
        STATS.basis_history.append(
            {
                "reconditioning": STATS.reconditionings,
                "event_covector": [str(value) for value in covector],
                "basis": [[str(value) for value in row] for row in basis],
                "inverse": [[str(value) for value in row] for row in inverse],
                "coordinate_radii": [interval_json(value) for value in radii],
                "normal_pairing_q": str(
                    sum(covector[row] * basis[row][0] for row in range(4))
                ),
                "kernel_pairings_q": [
                    str(sum(covector[row] * basis[row][column] for row in range(4)))
                    for column in range(1, 4)
                ],
            }
        )
    return result


def stats_json() -> dict[str, object]:
    return {
        "reconditionings": STATS.reconditionings,
        "generator_reconstructions": STATS.generator_reconstructions,
        "coefficient_uncertainty_generators": STATS.coefficient_uncertainty_generators,
        "normal_direction_seeds": STATS.normal_direction_seeds,
        "normal_direction_transports": STATS.normal_direction_transports,
        "kernel_direction_seeds": STATS.kernel_direction_seeds,
        "kernel_direction_transports": STATS.kernel_direction_transports,
        "maximum_basis_inverse_row_sum_q": str(STATS.maximum_basis_inverse_row_sum),
        "maximum_generator_count": STATS.maximum_generator_count,
        "kernel_orthogonality_checks": STATS.kernel_orthogonality_checks,
        "reconstruction_checks": STATS.reconstruction_checks,
        "normal_form_checks": STATS.normal_form_checks,
        "basis_history": STATS.basis_history,
    }


def reset_stats() -> None:
    global STATS, CARRIER_INITIALIZED
    STATS = CarrierStats()
    CARRIER_INITIALIZED = False


def serialized_state(state: list[base.TM2R]) -> list[dict[str, object]]:
    return [transport.tm2r_json(component) for component in state]


def exact_budget(state: list[base.TM2R]) -> dict[str, object]:
    if all(
        len(monomial) == PRIMARY_VARIABLES
        for component in state
        for monomial in component.coefficients
    ):
        parsed = [
            budget_worker.parse_component(component)
            for component in serialized_state(state)
        ]
        derivative, parts = budget_worker.derivative_model_with_parts(parsed)
        return budget_worker.budget(derivative, parts)
    derivative = state[0] * state[1] - state[2] - base.ZS
    derivative_range = derivative.range()
    derivative_width = base.width(derivative_range)
    midpoint = derivative_range.mid()
    return {
        "variables": [*PRIMARY_NAMES, *CARRIER_NAMES],
        "range": interval_json(derivative_range),
        "width_q": base.upper_fraction(derivative_width),
        "midpoint_q": str(midpoint.fmpq()),
        "radius_q": str((derivative_width / 2).upper().fmpq()),
        "remainder": interval_json(derivative.remainder),
        "remainder_width_q": base.upper_fraction(base.width(derivative.remainder)),
        "polynomial_range": interval_json(derivative.polynomial_range()),
        "coefficient_count": len(derivative.coefficients),
    }


def frozen_witness_state(
    witness_payload: dict[str, object], checks: list[dict[str, object]]
) -> list[base.TM2R]:
    raw = witness_payload["reconstruction"]["raw_projection"]["components"]
    state = [parse_tm2r(component) for component in raw]
    state, stabilization_checks = witness.chain.outward_stabilize_carrier(state)
    observed_path: list[str] = []
    for depth, token in enumerate(witness.WITNESS_PATH, start=1):
        expected_name, side = token[:-1], token[-1]
        variable, _weight = adaptive.dominant_variable(state)
        actual_name = adaptive.VARIABLE_NAMES[variable]
        bool_check(
            checks,
            f"serialized_witness_split_{depth}_matches",
            actual_name == expected_name,
        )
        left, right, _reconstructions = adaptive.split_state(state, variable)
        state = left if side == "L" else right
        observed_path.append(actual_name + side)
    bool_check(
        checks,
        "serialized_witness_path_replayed_exactly",
        observed_path == list(witness.WITNESS_PATH),
    )
    bool_check(
        checks,
        "serialized_raw_projection_stabilized",
        stabilization_checks > 0,
    )
    return state


def lineage_one_step(state: list[base.TM2R]) -> tuple[list[base.TM2R], list[arb]]:
    original = base.recondition
    base.recondition = witness.prior.lineage_preserving_recondition
    try:
        return adaptive.advance_with_endpoint_intersection(
            state, base.rational_ball(Fraction(1, 2**8))
        )
    finally:
        base.recondition = original


def preflight_mode(
    mode: str,
    states: dict[str, list[base.TM2R]],
    baseline: dict[str, object],
    initial_witness: list[base.TM2R],
    lineage_step_budget: dict[str, object],
) -> dict[str, object]:
    global ACTIVE_MODE
    ACTIVE_MODE = mode
    analyses: dict[str, object] = {}
    for name, state in states.items():
        reset_stats()
        state = extend_state(state)
        before_ranges = [component.range() for component in state]
        conditioned = event_normal_recondition(state)
        after_ranges = [component.range() for component in conditioned]
        contains_before = [
            after.contains(before)
            for before, after in zip(before_ranges, after_ranges, strict=True)
        ]
        conditioned_budget = exact_budget(conditioned)
        baseline_budget = baseline["analyses"][name]["derivative_budget"]
        baseline_width = Fraction(str(baseline_budget["width_q"]))
        conditioned_width = Fraction(str(conditioned_budget["width_q"]))
        improvement = baseline_width / conditioned_width if conditioned_width else Fraction(0)
        lower = Fraction(str(conditioned_budget["range"][0]))
        analyses[name] = {
            "state_range_hull_containments_forensic": contains_before,
            "all_state_range_hulls_contained_forensic": all(contains_before),
            "conditioned_components": serialized_state(conditioned),
            "baseline_derivative_width_q": str(baseline_width),
            "conditioned_derivative_budget": conditioned_budget,
            "derivative_width_improvement_factor_q": str(improvement),
            "target_improvement_met": improvement >= TARGET_IMPROVEMENT,
            "endpoint_derivative_strictly_positive": lower > 0,
            "stats": stats_json(),
        }
    reset_stats()
    initial_witness = extend_state(initial_witness)
    initial_ranges = [component.range() for component in initial_witness]
    conditioned_initial = event_normal_recondition(initial_witness)
    initial_containments = [
        after.range().contains(before)
        for before, after in zip(initial_ranges, conditioned_initial, strict=True)
    ]
    original = base.recondition
    base.recondition = event_normal_recondition
    try:
        carrier_step, carrier_tube = adaptive.advance_with_endpoint_intersection(
            conditioned_initial, base.rational_ball(Fraction(1, 2**8))
        )
    finally:
        base.recondition = original
    carrier_step_budget = exact_budget(carrier_step)
    lineage_width = Fraction(str(lineage_step_budget["width_q"]))
    carrier_width = Fraction(str(carrier_step_budget["width_q"]))
    step_improvement = lineage_width / carrier_width if carrier_width else Fraction(0)
    width_margin = lineage_width - carrier_width
    receipt_rounding_tolerance = Fraction(1, 2**230)
    initial_analysis = {
        "initial_state_range_hull_containments_forensic": initial_containments,
        "all_initial_state_range_hulls_contained_forensic": all(initial_containments),
        "conditioned_initial_components": serialized_state(conditioned_initial),
        "carrier_one_step_components": serialized_state(carrier_step),
        "conditioned_initial_derivative_budget": exact_budget(conditioned_initial),
        "lineage_one_step_derivative_budget": lineage_step_budget,
        "carrier_one_step_derivative_budget": carrier_step_budget,
        "carrier_one_step_tube_derivative": interval_json(
            carrier_tube[0] * carrier_tube[1] - carrier_tube[2] - base.ZS
        ),
        "one_step_derivative_width_improvement_factor_q": str(step_improvement),
        "one_step_derivative_width_margin_q": str(width_margin),
        "receipt_rounding_tolerance_q": str(receipt_rounding_tolerance),
        "width_margin_exceeds_receipt_rounding_tolerance": (
            width_margin > receipt_rounding_tolerance
        ),
        "one_step_improves_lineage": step_improvement > 1,
        "stats": stats_json(),
    }
    stats = initial_analysis["stats"]
    reconstruction_certified = (
        stats["generator_reconstructions"] > 0
        and stats["generator_reconstructions"] == stats["reconstruction_checks"]
        and stats["kernel_orthogonality_checks"]
        == 3 * stats["reconditionings"]
        and stats["normal_form_checks"] == stats["reconditionings"]
    )
    initial_analysis["generator_reconstruction_certificate"] = reconstruction_certified
    if not reconstruction_certified:
        classification = "EVENT_NORMAL_PREFLIGHT_RECONSTRUCTION_FAILED"
    elif initial_analysis["one_step_improves_lineage"]:
        classification = "EVENT_NORMAL_PREFLIGHT_ONE_STEP_IMPROVED"
    else:
        classification = "EVENT_NORMAL_PREFLIGHT_ONE_STEP_WORSENED"
    return {
        "mode": mode,
        "classification": classification,
        "target_improvement_factor_q": str(TARGET_IMPROVEMENT),
        "analyses": analyses,
        "initial_witness_analysis": initial_analysis,
        "post_hoc_endpoint_recovery_is_control_only": True,
    }


def run_preflight(
    witness_payload: dict[str, object],
    baseline: dict[str, object],
    checks: list[dict[str, object]],
) -> dict[str, object]:
    diagnostic = witness_payload["diagnostic"]
    states = {
        "production_before": [
            parse_tm2r(component)
            for component in diagnostic["production_boundary"]["before"]["state"]["components"]
        ],
        "terminal_before": [
            parse_tm2r(component)
            for component in diagnostic["terminal_ambiguous"]["before"]["state"]["components"]
        ],
    }
    initial_witness = frozen_witness_state(witness_payload, checks)
    lineage_step, _lineage_tube = lineage_one_step(initial_witness)
    lineage_step_budget = exact_budget(lineage_step)
    modes = [
        preflight_mode(
            mode, states, baseline, initial_witness, lineage_step_budget
        )
        for mode in MODES
    ]
    viable = [
        item["mode"]
        for item in modes
        if item["classification"] == "EVENT_NORMAL_PREFLIGHT_ONE_STEP_IMPROVED"
    ]
    return {
        "execution_mode": "PREFLIGHT",
        "modes": modes,
        "transport_candidates": viable,
        "full_transport_attempted": False,
        "classification": (
            "EVENT_NORMAL_PREFLIGHT_CANDIDATE_FOUND"
            if viable
            else "EVENT_NORMAL_PREFLIGHT_NO_CANDIDATE"
        ),
    }


def run_transport(
    mode: str,
    checks: list[dict[str, object]],
    witness_payload: dict[str, object],
) -> dict[str, object]:
    global ACTIVE_MODE
    ACTIVE_MODE = mode
    reset_stats()
    # Rebuild the exact frozen residual subdomain from the hash-bound raw event
    # projection, then install the new carrier before any new flow step.
    witness_state = frozen_witness_state(witness_payload, checks)
    witness_domain = witness_payload["witness_domain"]
    reconstruction = {
        "source": "hash_bound_serialized_raw_projection",
        "witness_path": list(witness.WITNESS_PATH),
    }
    base.recondition = event_normal_recondition
    witness_state = extend_state(witness_state)
    witness_state = event_normal_recondition(witness_state)
    initial_weights = witness.centered.variable_weights(
        witness_state, rows=SECTION_ROWS
    )
    bool_check(
        checks,
        "initial_xi_eta_preserved",
        initial_weights[0].upper() > 0 and initial_weights[1].upper() > 0,
    )
    diagnostic = witness.diagnose_upward_event(witness_state)
    xi_eta_preserved: bool | None = None
    accepted = diagnostic.get("accepted_projection")
    if isinstance(accepted, dict):
        xi_eta_preserved = True
        for carrier in accepted.get("carriers", []):
            weights = carrier.get("variable_weights", [])
            xi_eta_preserved = (
                xi_eta_preserved
                and len(weights) >= 2
                and Fraction(str(weights[0][1])) > 0
                and Fraction(str(weights[1][1])) > 0
            )
    bool_check(checks, "event_normal_reconditioner_active", base.recondition is event_normal_recondition)
    if xi_eta_preserved is not None:
        bool_check(checks, "terminal_xi_eta_coordinates_retained", xi_eta_preserved)
    implementation_ok = all(item["passed"] is True for item in checks)
    return {
        "execution_mode": "TRANSPORT",
        "mode": mode,
        "witness_domain": witness_domain,
        "reconstruction": reconstruction,
        "diagnostic": diagnostic,
        "carrier_stats": stats_json(),
        "terminal_xi_eta_preserved": xi_eta_preserved,
        "full_transport_attempted": True,
        "implementation_checks_passed": implementation_ok,
        "classification": (
            "IMPLEMENTATION_INCONSISTENCY"
            if not implementation_ok
            else (
                "EVENT_NORMAL_TRANSPORT_ACCEPTED"
                if diagnostic.get("accepted") is True
                else str(diagnostic.get("status"))
            )
        ),
    }


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("event-normal carrier requires Python >= 3.10")
    execution_mode = os.environ.get("CS6_EXECUTION_MODE", "PREFLIGHT")
    if execution_mode not in EXECUTION_MODES:
        raise SystemExit(f"CS6_EXECUTION_MODE must be one of {EXECUTION_MODES}")
    carrier_mode = os.environ.get("CS6_CARRIER_MODE", "EVENT_NORMAL_TRIPLETON")
    if carrier_mode not in MODES:
        raise SystemExit(f"CS6_CARRIER_MODE must be one of {MODES}")
    base.SOURCE_DEGREE = 2
    base.TIME_TAYLOR_ORDER = 12
    event.MAX_PHASE_STEPS = composability.MAX_FIRST_RETURN_STEPS

    source_path = Path(__file__)
    research = source_path.parent
    witness_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_prerecond_witness_event_v1"
        / "witness_event.json"
    )
    budget_receipt = (
        research
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_prerecond_witness_derivative_budget_v1"
        / "derivative_budget.json"
    )
    for path, label in ((witness_receipt, "witness"), (budget_receipt, "budget")):
        if not path.is_file():
            raise SystemExit(f"frozen {label} receipt is missing: {path}")
    witness_payload = json.loads(witness_receipt.read_text(encoding="ascii"))
    baseline = json.loads(budget_receipt.read_text(encoding="ascii"))
    checks: list[dict[str, object]] = []
    bool_check(checks, "witness_receipt_hash_matches", sha256(witness_receipt) == EXPECTED_WITNESS_SHA256)
    bool_check(checks, "budget_receipt_hash_matches", sha256(budget_receipt) == EXPECTED_BUDGET_SHA256)
    bool_check(checks, "witness_classification_matches", witness_payload.get("classification") == "WITNESS_TRANSVERSALITY_UNRESOLVED")
    bool_check(checks, "budget_classification_matches", baseline.get("classification") == "DERIVATIVE_INTERVAL_REMAINDER_DOMINANT")

    result = (
        run_preflight(witness_payload, baseline, checks)
        if execution_mode == "PREFLIGHT"
        else run_transport(carrier_mode, checks, witness_payload)
    )
    implementation_ok = all(item["passed"] is True for item in checks)
    payload = {
        "schema": SCHEMA,
        "worker_source_sha256": sha256(source_path),
        "witness_worker_source_sha256": sha256(Path(witness.__file__)),
        "budget_worker_source_sha256": sha256(Path(budget_worker.__file__)),
        "base_source_sha256": sha256(Path(base.__file__)),
        "adaptive_source_sha256": sha256(Path(adaptive.__file__)),
        "witness_receipt_sha256": sha256(witness_receipt),
        "budget_receipt_sha256": sha256(budget_receipt),
        "python_version": platform.python_version(),
        "python_flint_version": flint.__version__,
        "arb_precision_bits": base.PRECISION_BITS,
        "source_degree": base.SOURCE_DEGREE,
        "time_taylor_order": base.TIME_TAYLOR_ORDER,
        "event_covector_definition": ["mid(y)", "mid(x)", "-1", "0"],
        "carrier_policy": "preserve_six_primary_tm2_variables_append_event_normal_qr_carrier",
        "primary_variables": list(PRIMARY_NAMES),
        "carrier_variables": list(CARRIER_NAMES),
        "target_improvement_factor_q": str(TARGET_IMPROVEMENT),
        "implementation_checks": checks,
        "implementation_checks_passed": implementation_ok,
        "result": result,
        "interval_newton_attempted": (
            execution_mode == "TRANSPORT"
            and bool(result.get("diagnostic", {}).get("projection_attempts"))
        ),
        "covering_relation_certified": False,
        "recurrent_graph_certified": False,
        "chaos_certified": False,
        "open_problem_solved": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
