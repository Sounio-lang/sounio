#!/usr/bin/env python3
"""Rigorous Arb TM2 with QR-derived residual transport for CS6 leaf 331."""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb, ctx


PRECISION_BITS = 256
TIME_TAYLOR_ORDER = 12
SOURCE_DEGREE = 2
SOURCE_VARIABLES = 2
RESIDUAL_VARIABLES = 4
VARIABLES = SOURCE_VARIABLES + RESIDUAL_VARIABLES
MAX_STEPS = 2000
LEAF_ID = "U08-0000000223_S09-0000000325"
U_DEPTH, U_INDEX, S_DEPTH, S_INDEX = 8, 223, 9, 325
ZERO_MONOMIAL = (0,) * VARIABLES

ctx.prec = PRECISION_BITS
ctx.threads = 1


def exact_decimal(token: str) -> arb:
    value = Fraction(token)
    return arb(value.numerator) / arb(value.denominator)


def rational_ball(value: Fraction) -> arb:
    return arb(value.numerator) / arb(value.denominator)


ZS = exact_decimal("22.3274637391")
ORIGIN_X = exact_decimal("15.186446520640786")
ORIGIN_Y = exact_decimal("10.908543194765466")
UNSTABLE_X = exact_decimal("-0.67430316214199759")
UNSTABLE_Y = exact_decimal("-0.73845463335624273")
STABLE_X = exact_decimal("-0.94170446778164518")
STABLE_Y = exact_decimal("0.33644122125579123")
STEP = arb(1) / arb(2**8)
UNIT = arb(0, 1)
UNIT_SQUARE = arb(arb(1) / 2, arb(1) / 2)


class Refusal(RuntimeError):
    def __init__(self, failure_class: str, detail: str) -> None:
        super().__init__(detail)
        self.failure_class = failure_class
        self.detail = detail


def exact_fraction(value: arb) -> str:
    return str(value.fmpq())


def lower_fraction(value: arb) -> str:
    return exact_fraction(value.lower())


def upper_fraction(value: arb) -> str:
    return exact_fraction(value.upper())


def upper_abs(value: arb) -> arb:
    return abs(value).upper()


def width(value: arb) -> arb:
    return value.upper() - value.lower()


def max_upper(values: list[arb]) -> arb:
    if not values:
        raise ValueError("empty maximum")
    result = values[0].upper()
    for value in values[1:]:
        if value.upper() > result:
            result = value.upper()
    return result


def monomial_range(monomial: tuple[int, ...]) -> arb:
    if not any(monomial):
        return arb(1)
    if any(exponent % 2 for exponent in monomial):
        return UNIT
    return UNIT_SQUARE


@dataclass
class TM2R:
    coefficients: dict[tuple[int, ...], arb]
    remainder: arb = dataclass_field(default_factory=arb)

    @classmethod
    def constant(cls, value: arb | int) -> "TM2R":
        return cls({ZERO_MONOMIAL: arb(value)}, arb(0))

    def polynomial_range(self) -> arb:
        result = arb(0)
        for monomial, coefficient in self.coefficients.items():
            result += coefficient * monomial_range(monomial)
        return result

    def range(self) -> arb:
        return self.polynomial_range() + self.remainder

    def __add__(self, other: "TM2R" | arb | int) -> "TM2R":
        rhs = other if isinstance(other, TM2R) else TM2R.constant(other)
        keys = self.coefficients.keys() | rhs.coefficients.keys()
        coefficients = {
            key: self.coefficients.get(key, arb(0)) + rhs.coefficients.get(key, arb(0))
            for key in keys
        }
        return TM2R(coefficients, self.remainder + rhs.remainder)

    __radd__ = __add__

    def __neg__(self) -> "TM2R":
        return TM2R({key: -value for key, value in self.coefficients.items()}, -self.remainder)

    def __sub__(self, other: "TM2R" | arb | int) -> "TM2R":
        return self + (-other if isinstance(other, TM2R) else -arb(other))

    def __rsub__(self, other: arb | int) -> "TM2R":
        return TM2R.constant(other) - self

    def __mul__(self, other: "TM2R" | arb | int) -> "TM2R":
        if not isinstance(other, TM2R):
            scalar = arb(other)
            return TM2R(
                {key: value * scalar for key, value in self.coefficients.items()},
                self.remainder * scalar,
            )
        retained: dict[tuple[int, ...], arb] = {}
        tail = arb(0)
        for left_monomial, left in self.coefficients.items():
            for right_monomial, right in other.coefficients.items():
                monomial = tuple(
                    left_monomial[index] + right_monomial[index]
                    for index in range(VARIABLES)
                )
                coefficient = left * right
                if sum(monomial) <= SOURCE_DEGREE:
                    retained[monomial] = retained.get(monomial, arb(0)) + coefficient
                else:
                    tail += coefficient * monomial_range(monomial)
        cross = (
            self.polynomial_range() * other.remainder
            + other.polynomial_range() * self.remainder
            + self.remainder * other.remainder
        )
        return TM2R(retained, tail + cross)

    __rmul__ = __mul__

    def __truediv__(self, divisor: arb | int) -> "TM2R":
        return self * (arb(1) / arb(divisor))

    def with_remainder(self, addition: arb) -> "TM2R":
        return TM2R(dict(self.coefficients), self.remainder + addition)


def field_interval(state: list[arb]) -> list[arb]:
    x, y, w, _ell = state
    return [
        2 * y * y - x * y,
        x * y - y * (w + ZS) / 2,
        x * y - w - ZS,
        x - y - (w + ZS) / 2 - 1,
    ]


def interval_flow_coefficients(initial: list[arb], degree: int) -> list[list[arb]]:
    coefficients = [[initial[i]] + [arb(0) for _ in range(degree)] for i in range(4)]
    for n in range(degree):
        xy = sum((coefficients[0][j] * coefficients[1][n - j] for j in range(n + 1)), arb(0))
        yy = sum((coefficients[1][j] * coefficients[1][n - j] for j in range(n + 1)), arb(0))
        yw = sum((coefficients[1][j] * coefficients[2][n - j] for j in range(n + 1)), arb(0))
        divisor = arb(n + 1)
        coefficients[0][n + 1] = (2 * yy - xy) / divisor
        coefficients[1][n + 1] = (xy - (yw + ZS * coefficients[1][n]) / 2) / divisor
        coefficients[2][n + 1] = (xy - coefficients[2][n] - (ZS if n == 0 else 0)) / divisor
        coefficients[3][n + 1] = (
            coefficients[0][n] - coefficients[1][n] - coefficients[2][n] / 2
            - ((ZS / 2 + 1) if n == 0 else 0)
        ) / divisor
    return coefficients


def tm_flow_coefficients(initial: list[TM2R], degree: int) -> list[list[TM2R]]:
    coefficients = [[initial[i]] + [TM2R.constant(0) for _ in range(degree)] for i in range(4)]
    for n in range(degree):
        xy = sum((coefficients[0][j] * coefficients[1][n - j] for j in range(n + 1)), TM2R.constant(0))
        yy = sum((coefficients[1][j] * coefficients[1][n - j] for j in range(n + 1)), TM2R.constant(0))
        yw = sum((coefficients[1][j] * coefficients[2][n - j] for j in range(n + 1)), TM2R.constant(0))
        divisor = n + 1
        coefficients[0][n + 1] = (2 * yy - xy) / divisor
        coefficients[1][n + 1] = (xy - (yw + ZS * coefficients[1][n]) / 2) / divisor
        coefficients[2][n + 1] = (xy - coefficients[2][n] - (ZS if n == 0 else 0)) / divisor
        coefficients[3][n + 1] = (
            coefficients[0][n] - coefficients[1][n] - coefficients[2][n] / 2
            - ((ZS / 2 + 1) if n == 0 else 0)
        ) / divisor
    return coefficients


def inflate(value: arb) -> arb:
    return arb(value.mid(), value.rad() * arb(17) / arb(16) + arb(1) / (arb(2) ** 220))


def lipschitz_bound(box: list[arb]) -> arb:
    x, y, w, _ell = box
    return max_upper([
        abs(y) + abs(4 * y - x),
        abs(y) + abs(x - (w + ZS) / 2) + abs(y) / 2,
        abs(y) + abs(x) + 1,
        arb(5) / 2,
    ])


@dataclass
class Statistics:
    attempted_steps: int = 0
    completed_steps: int = 0
    picard_calls: int = 0
    picard_containments: int = 0
    endpoint_picard_containments: int = 0
    reconditionings: int = 0
    generator_reconstructions: int = 0
    max_picard_iterations: int = 0
    max_picard_contraction: arb = dataclass_field(default_factory=arb)
    max_raw_width: arb = dataclass_field(default_factory=arb)
    max_reconditioned_width: arb = dataclass_field(default_factory=arb)
    max_basis_inverse_row_sum: Fraction = Fraction(0)


STATS = Statistics()


def picard_box(initial: list[arb], step: arb) -> list[arb]:
    STATS.picard_calls += 1
    time_interval = arb(step / 2, step / 2)
    box = [
        inflate(component.union(component + time_interval * derivative))
        for component, derivative in zip(initial, field_interval(initial), strict=True)
    ]
    for iteration in range(1, 51):
        image = [
            component + time_interval * derivative
            for component, derivative in zip(initial, field_interval(box), strict=True)
        ]
        if all(container.contains(candidate) for container, candidate in zip(box, image, strict=True)):
            contraction = lipschitz_bound(box) * step
            STATS.max_picard_contraction = max_upper([STATS.max_picard_contraction, contraction])
            if contraction.upper() >= 1:
                raise Refusal("PICARD_NONCONTRACTION", "step-times-Lipschitz bound reached one")
            STATS.picard_containments += 1
            STATS.max_picard_iterations = max(STATS.max_picard_iterations, iteration)
            return box
        box = [inflate(container.union(candidate)) for container, candidate in zip(box, image, strict=True)]
    raise Refusal("PICARD_NO_CLOSURE", "Picard enclosure did not close in 50 iterations")


def fraction_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    dimension = len(matrix)
    augmented = [
        list(row) + [Fraction(int(i == j)) for j in range(dimension)]
        for i, row in enumerate(matrix)
    ]
    for column in range(dimension):
        pivot = next((row for row in range(column, dimension) if augmented[row][column]), None)
        if pivot is None:
            raise Refusal("QR_BASIS_SINGULAR", "rationalized QR basis was singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(2 * dimension)
            ]
    return [row[dimension:] for row in augmented]


def qr_derived_basis(generators: list[list[arb]]) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    candidates: list[list[float]] = []
    for generator in generators:
        vector = [float(value.mid()) for value in generator]
        if all(math.isfinite(value) for value in vector):
            candidates.append(vector)
    candidates.sort(key=lambda vector: sum(value * value for value in vector), reverse=True)
    candidates.extend([[float(i == j) for i in range(4)] for j in range(4)])
    columns: list[list[float]] = []
    for candidate in candidates:
        vector = list(candidate)
        for column in columns:
            projection = sum(vector[i] * column[i] for i in range(4))
            vector = [vector[i] - projection * column[i] for i in range(4)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1e-12:
            columns.append([value / norm for value in vector])
        if len(columns) == 4:
            break
    if len(columns) != 4:
        raise Refusal("QR_BASIS_INCOMPLETE", "could not construct four residual directions")
    basis = [
        [Fraction(format(columns[column][row], ".17g")) for column in range(4)]
        for row in range(4)
    ]
    inverse = fraction_inverse(basis)
    STATS.max_basis_inverse_row_sum = max(
        STATS.max_basis_inverse_row_sum,
        max(sum(abs(value) for value in row) for row in inverse),
    )
    return basis, inverse


def vector_nonzero(vector: list[arb]) -> bool:
    return any(value.lower() != 0 or value.upper() != 0 for value in vector)


def recondition(state: list[TM2R]) -> list[TM2R]:
    STATS.reconditionings += 1
    source_coefficients: list[dict[tuple[int, ...], arb]] = [dict() for _ in range(4)]
    residual_monomials = set()
    for component in state:
        residual_monomials.update(
            monomial for monomial in component.coefficients
            if any(monomial[SOURCE_VARIABLES:])
        )
    for row, component in enumerate(state):
        for monomial, coefficient in component.coefficients.items():
            if not any(monomial[SOURCE_VARIABLES:]):
                source_coefficients[row][monomial] = coefficient

    generators: list[list[arb]] = []
    for monomial in sorted(residual_monomials):
        generator = [component.coefficients.get(monomial, arb(0)) for component in state]
        if not vector_nonzero(generator):
            continue
        if all(exponent % 2 == 0 for exponent in monomial):
            half = [value / 2 for value in generator]
            for row in range(4):
                source_coefficients[row][ZERO_MONOMIAL] = (
                    source_coefficients[row].get(ZERO_MONOMIAL, arb(0)) + half[row]
                )
            generators.append(half)
        else:
            generators.append(generator)

    for row, component in enumerate(state):
        midpoint = component.remainder.mid()
        radius = component.remainder.rad()
        source_coefficients[row][ZERO_MONOMIAL] = (
            source_coefficients[row].get(ZERO_MONOMIAL, arb(0)) + midpoint
        )
        if radius.upper() > 0:
            generator = [arb(0) for _ in range(4)]
            generator[row] = radius
            generators.append(generator)

    basis, inverse = qr_derived_basis(generators)
    radii = [arb(0) for _ in range(4)]
    for generator in generators:
        coordinates: list[arb] = []
        for coordinate in range(4):
            projected = sum(
                (rational_ball(inverse[coordinate][row]) * generator[row] for row in range(4)),
                arb(0),
            )
            coordinates.append(projected)
            radii[coordinate] += upper_abs(projected)
        reconstructed = [
            sum(
                (rational_ball(basis[row][coordinate]) * coordinates[coordinate] for coordinate in range(4)),
                arb(0),
            )
            for row in range(4)
        ]
        if not all(
            enclosure.contains(component)
            for enclosure, component in zip(reconstructed, generator, strict=True)
        ):
            raise Refusal("GENERATOR_RECONSTRUCTION_FAILED", "Q-times-Q-inverse failed to enclose a residual generator")
        STATS.generator_reconstructions += 1

    result: list[TM2R] = []
    for row in range(4):
        coefficients = dict(source_coefficients[row])
        for coordinate in range(4):
            monomial = [0] * VARIABLES
            monomial[SOURCE_VARIABLES + coordinate] = 1
            coefficients[tuple(monomial)] = rational_ball(basis[row][coordinate]) * radii[coordinate]
        result.append(TM2R(coefficients, arb(0)))

    conditioned_ranges = [component.range() for component in result]
    STATS.max_reconditioned_width = max_upper([
        STATS.max_reconditioned_width, *[width(value) for value in conditioned_ranges]
    ])
    return result


def advance(initial: list[TM2R], step: arb) -> tuple[list[TM2R], list[arb]]:
    STATS.attempted_steps += 1
    initial_range = [component.range() for component in initial]
    box = picard_box(initial_range, step)
    coefficients = tm_flow_coefficients(initial, TIME_TAYLOR_ORDER)
    polynomial = [
        sum(
            (coefficients[row][power] * step**power for power in range(TIME_TAYLOR_ORDER + 1)),
            TM2R.constant(0),
        )
        for row in range(4)
    ]
    remainder_coefficients = interval_flow_coefficients(box, TIME_TAYLOR_ORDER + 1)
    raw: list[TM2R] = []
    for row, component in enumerate(polynomial):
        time_remainder = upper_abs(remainder_coefficients[row][TIME_TAYLOR_ORDER + 1]) * step ** (TIME_TAYLOR_ORDER + 1)
        candidate = component.with_remainder(arb(0, time_remainder))
        if not box[row].contains(candidate.range()):
            raise Refusal("ENDPOINT_ESCAPES_PICARD", "raw TM endpoint escaped its Picard tube")
        raw.append(candidate)
    STATS.endpoint_picard_containments += 1
    STATS.max_raw_width = max_upper([STATS.max_raw_width, *[width(component.range()) for component in raw]])
    result = recondition(raw)
    STATS.completed_steps += 1
    return result, box


def strict_sign(value: arb) -> int:
    if value.upper() < 0:
        return -1
    if value.lower() > 0:
        return 1
    return 0


def initial_leaf() -> tuple[list[TM2R], arb, arb]:
    u_low = Fraction(-4, 1000) + Fraction(U_INDEX * 8, 1000 * 2**U_DEPTH)
    u_high = Fraction(-4, 1000) + Fraction((U_INDEX + 1) * 8, 1000 * 2**U_DEPTH)
    s_low = Fraction(-3, 10) + Fraction(S_INDEX * 6, 10 * 2**S_DEPTH)
    s_high = Fraction(-3, 10) + Fraction((S_INDEX + 1) * 6, 10 * 2**S_DEPTH)
    u_center, u_radius = (u_low + u_high) / 2, (u_high - u_low) / 2
    s_center, s_radius = (s_low + s_high) / 2, (s_high - s_low) / 2
    xi = [0] * VARIABLES
    xi[0] = 1
    eta = [0] * VARIABLES
    eta[1] = 1
    x = TM2R({
        ZERO_MONOMIAL: ORIGIN_X + UNSTABLE_X * rational_ball(u_center) + STABLE_X * rational_ball(s_center),
        tuple(xi): UNSTABLE_X * rational_ball(u_radius),
        tuple(eta): STABLE_X * rational_ball(s_radius),
    })
    y = TM2R({
        ZERO_MONOMIAL: ORIGIN_Y + UNSTABLE_Y * rational_ball(u_center) + STABLE_Y * rational_ball(s_center),
        tuple(xi): UNSTABLE_Y * rational_ball(u_radius),
        tuple(eta): STABLE_Y * rational_ball(s_radius),
    })
    u = arb(rational_ball((u_low + u_high) / 2), rational_ball((u_high - u_low) / 2))
    s = arb(rational_ball((s_low + s_high) / 2), rational_ball((s_high - s_low) / 2))
    return [x, y, TM2R.constant(0), TM2R.constant(0)], u, s


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={upper_fraction(value)}")


def main() -> None:
    initial, u_interval, s_interval = initial_leaf()
    state = initial
    initial_range = [component.range() for component in state]
    events: list[list[arb]] = []
    event_before_w = arb(0)
    event_after_w = arb(0)
    event_end_step = -1
    initial_departure_tubes = 0
    prior_downward_tubes = 0
    zero_free_prior_tubes = 0
    seen_strict_negative = False
    event_w_derivative = arb(0)
    failure_class = "NONE"
    failure_detail = "NONE"
    failure_step = -1
    try:
        for step_index in range(MAX_STEPS):
            before_range = [component.range() for component in state]
            before_sign = strict_sign(before_range[2])
            next_state, step_tube = advance(state, STEP)
            next_range = [component.range() for component in next_state]
            after_sign = strict_sign(next_range[2])
            if after_sign < 0:
                seen_strict_negative = True
            tube_contains_section = step_tube[2].lower() <= 0 <= step_tube[2].upper()
            tube_w_derivative = step_tube[0] * step_tube[1] - step_tube[2] - ZS
            if not tube_contains_section:
                zero_free_prior_tubes += 1
            elif step_index == 0:
                if before_sign != 0 or after_sign <= 0 or tube_w_derivative.lower() <= 0:
                    raise Refusal("INITIAL_DEPARTURE_UNRESOLVED", "initial section departure was not strictly positive")
                initial_departure_tubes += 1
            elif seen_strict_negative and before_sign < 0 and after_sign > 0:
                normal = step_tube[0] * step_tube[1] - ZS
                if normal.lower() <= 0 or tube_w_derivative.lower() <= 0:
                    raise Refusal("EVENT_TRANSVERSALITY_UNRESOLVED", "event tube did not have strictly positive w derivative")
                events.append(step_tube)
                event_before_w = before_range[2]
                event_after_w = next_range[2]
                event_end_step = STATS.completed_steps
                event_w_derivative = tube_w_derivative
                state = next_state
                break
            elif tube_w_derivative.upper() < 0:
                prior_downward_tubes += 1
            else:
                raise Refusal("PRIOR_ORIENTATION_UNRESOLVED", "a prior section-touching tube lacked strict downward orientation")
            state = next_state
        if len(events) != 1:
            raise Refusal("EVENT_COUNT_UNRESOLVED", f"validated event count was {len(events)} after {STATS.completed_steps} steps")
    except Refusal as refusal:
        failure_class = refusal.failure_class
        failure_detail = refusal.detail.replace(" ", "_")
        failure_step = STATS.completed_steps

    first_return = failure_class == "NONE" and len(events) == 1
    event_normal = arb(0)
    if first_return:
        event_normal = events[0][0] * events[0][1] - ZS

    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print("SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-first-return-worker.v1")
    print(f"WORKER_SOURCE_SHA256={source_sha}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_FLINT_VERSION={flint.__version__}")
    print(f"LEAF_ID={LEAF_ID}")
    print("ARB_PRECISION_BITS=256")
    print("ARB_THREADS=1")
    print(f"SOURCE_DEGREE={SOURCE_DEGREE}")
    print(f"SOURCE_VARIABLES={SOURCE_VARIABLES}")
    print(f"RESIDUAL_VARIABLES={RESIDUAL_VARIABLES}")
    print("RECONDITIONING=QR_DERIVED_RATIONAL_BASIS_ZONOTOPE_HULL_EVERY_STEP")
    print(f"TIME_TAYLOR_ORDER={TIME_TAYLOR_ORDER}")
    print("TIME_STEP_POWER=-8")
    emit_interval("LEAF_U", u_interval)
    emit_interval("LEAF_S", s_interval)
    emit_interval("INITIAL_X", initial_range[0])
    emit_interval("INITIAL_Y", initial_range[1])
    print(f"ATTEMPTED_STEPS={STATS.attempted_steps}")
    print(f"COMPLETED_STEPS={STATS.completed_steps}")
    print(f"PICARD_CALLS={STATS.picard_calls}")
    print(f"PICARD_CONTAINMENTS={STATS.picard_containments}")
    print(f"ENDPOINT_PICARD_CONTAINMENTS={STATS.endpoint_picard_containments}")
    print(f"RECONDITIONINGS={STATS.reconditionings}")
    print(f"GENERATOR_RECONSTRUCTIONS={STATS.generator_reconstructions}")
    print(f"MAX_PICARD_ITERATIONS={STATS.max_picard_iterations}")
    print(f"EVENTS_VALIDATED={len(events)}")
    print(f"INITIAL_DEPARTURE_TUBES={initial_departure_tubes}")
    print(f"PRIOR_DOWNWARD_TUBES={prior_downward_tubes}")
    print(f"ZERO_FREE_PRIOR_TUBES={zero_free_prior_tubes}")
    print(f"FAILURE_CLASS={failure_class}")
    print(f"FAILURE_DETAIL={failure_detail}")
    print(f"FAILURE_STEP={failure_step}")
    print(f"FIRST_RETURN_END_STEP={event_end_step}")
    print(f"FIRST_RETURN_TIME_LOWER_Q={Fraction(max(event_end_step - 1, 0), 2**8)}")
    print(f"FIRST_RETURN_TIME_UPPER_Q={Fraction(max(event_end_step, 0), 2**8)}")
    print(f"MAX_RAW_WIDTH_UPPER_Q={upper_fraction(STATS.max_raw_width)}")
    print(f"MAX_RECONDITIONED_WIDTH_UPPER_Q={upper_fraction(STATS.max_reconditioned_width)}")
    print(f"MAX_PICARD_CONTRACTION_UPPER_Q={upper_fraction(STATS.max_picard_contraction)}")
    print(f"MAX_BASIS_INVERSE_ROW_SUM_Q={STATS.max_basis_inverse_row_sum}")
    emit_interval("FIRST_RETURN_W_BEFORE", event_before_w)
    emit_interval("FIRST_RETURN_W_AFTER", event_after_w)
    emit_interval("FIRST_RETURN_W_DERIVATIVE", event_w_derivative)
    emit_interval("FIRST_RETURN_NORMAL", event_normal)
    print("BOUNDED_METHOD_RESULT=true")
    print(f"FULL_LEAF_FIRST_RETURN_CERTIFICATE={str(first_return).lower()}")
    print("FULL_LEAF_SECOND_RETURN_CERTIFICATE=false")
    print("CAPD_USED_BY_WORKER=false")
    print("POINT_FALLBACK_USED=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("CHAOS_PROVED=false")
    print("CHAOTIC_ATTRACTOR_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
