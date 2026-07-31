#!/usr/bin/env python3
"""Exact verifier for the CS6 C1-dependency A/B receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            fail("inverted interval")

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower

    def contains_zero(self) -> bool:
        return self.lower <= 0 <= self.upper

    def contains(self, other: "Interval") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def overlaps(self, other: "Interval") -> bool:
        return self.lower <= other.upper and other.lower <= self.upper

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.contains_zero():
            fail("division by interval containing zero")
        reciprocal = Interval(
            min(Fraction(1) / other.lower, Fraction(1) / other.upper),
            max(Fraction(1) / other.lower, Fraction(1) / other.upper),
        )
        return self * reciprocal

    def scale(self, factor: int | Fraction) -> "Interval":
        return self * point(factor)


def point(value: int | Fraction) -> Interval:
    rational = value if isinstance(value, Fraction) else Fraction(value)
    return Interval(rational, rational)


ZERO = point(0)
ONE = point(1)
TWO = point(2)
HALF = point(Fraction(1, 2))
ZS = point(Fraction("22.3274637391"))
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
HEX_RE = re.compile(
    r"^-?0x(?:[0-9a-f]+(?:\.[0-9a-f]*)?|\.[0-9a-f]+)p[+-][0-9]+$"
)
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")


def inward_binary64(value: Fraction, toward_positive: bool) -> Fraction:
    binary = float(value)
    if Fraction.from_float(binary) != value:
        fail("serialized endpoint is not binary64")
    direction = math.inf if toward_positive else -math.inf
    return Fraction.from_float(math.nextafter(binary, direction))


def parse_interval(token: str) -> Interval:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        fail(f"malformed interval: {token}")
    lower_token, upper_token = match.groups()
    if HEX_RE.fullmatch(lower_token) is None or HEX_RE.fullmatch(upper_token) is None:
        fail(f"noncanonical hexadecimal endpoint: {token}")
    outer_lower = Fraction.from_float(float.fromhex(lower_token))
    outer_upper = Fraction.from_float(float.fromhex(upper_token))
    if outer_lower > outer_upper:
        fail("inverted serialized interval")
    lower = inward_binary64(outer_lower, True)
    upper = inward_binary64(outer_upper, False)
    return Interval(lower, upper)


def decimal_interval(text: str) -> Interval:
    exact = Fraction(text)
    nearest_float = float(text)
    nearest = Fraction.from_float(nearest_float)
    lower = nearest_float
    upper = nearest_float
    if nearest > exact:
        lower = math.nextafter(nearest_float, -math.inf)
    elif nearest < exact:
        upper = math.nextafter(nearest_float, math.inf)
    return Interval(Fraction.from_float(lower), Fraction.from_float(upper))


def centered_square(value: Interval) -> Interval:
    if value.contains_zero():
        radius = max(abs(value.lower), abs(value.upper))
        return Interval(Fraction(0), radius * radius)
    return value * value


def vector_keys(prefix: str, size: int = 3) -> list[str]:
    return [f"{prefix}{row}" for row in range(size)]


def matrix_keys(prefix: str) -> list[str]:
    return [f"{prefix}{row}{column}" for row in range(3) for column in range(3)]


def hessian_keys(prefix: str) -> list[str]:
    return [
        f"{prefix}{image}{first}{second}"
        for image in range(3)
        for first in range(3)
        for second in range(first, 3)
    ]


FIXED_HEADERS: tuple[tuple[str, str | None], ...] = (
    ("SCHEMA", "sounio.cs6.c1-dependency-affine-projective.v1"),
    ("WORKER_SOURCE_SHA256", None),
    ("INPUT_SHA256", None),
    ("RUN_CHALLENGE", None),
    ("CAPD_SOURCE_TREE_DECLARED", "capd-5.3.0"),
    ("INTERVAL_BACKEND_DECLARED", "FILIB"),
    ("INTERVAL_SERIALIZATION", "ONE_ULP_OUTWARD_BINARY64_HEX"),
    ("SOURCE", "N0"),
    ("U_INDEX", "20000"),
    ("S_INDEX", "15000"),
    ("U_TILES", "40000"),
    ("S_TILES", "30000"),
    ("ORDER", "8"),
    ("RETURN_COUNT", "2"),
    ("SECTION", "COORDINATE_W_EQUALS_ZERO"),
    ("CROSSING_DIRECTION", "MINUS_PLUS"),
    (
        "C2_POINCARE_CONVERSION",
        "CAPD_COMPUTE_DP_WITH_RETURN_TIME_CORRECTION",
    ),
    (
        "IMPACT_TIME_CROSSCHECK",
        "CAPD_OUTPUT_VS_COORDINATE_SECTION_RECONSTRUCTION",
    ),
    (
        "C2_HESSIAN_ROLE",
        "NORMALIZED_TAYLOR_COEFFICIENTS_OF_RETURN_MAP",
    ),
    ("DIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR", "2"),
    ("OFFDIAGONAL_TAYLOR_TO_DERIVATIVE_FACTOR", "1"),
    (
        "AFFINE_CARRIER_FORM",
        "M_PLUS_A0_DELTA0_PLUS_A1_DELTA1_PLUS_R",
    ),
    (
        "AFFINE_REMAINDER_RULE",
        "CENTER_DP_RADIUS_PLUS_HESSIAN_RADIUS_TIMES_DELTA",
    ),
    ("PROJECTIVE_CONTROL", "FINAL_COLUMN_SLOPE_CHARTS"),
    ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
    ("LIOUVILLE_ROLE", "INDEPENDENT_SIGN_CROSS_CHECK_ONLY"),
    ("EXECUTION_SCOPE", "BOUNDED_LOCAL_CAPD_CPU_PROBE"),
    ("EXECUTION_PROVENANCE_ATTESTED", "false"),
    ("INDEPENDENT_REPLAY_REQUIRED", "true"),
    ("PROMOTION_ELIGIBLE", "false"),
    ("FULL_SOURCE_CARRIER_PROVED", "false"),
    ("HYPERBOLICITY_PROVED", "false"),
    ("CHAOTIC_ATTRACTOR_PROVED", "false"),
)

C2_KEYS = (
    ["TIME"]
    + vector_keys("X")
    + matrix_keys("FLOW")
    + hessian_keys("FLOW_H")
    + vector_keys("DT")
    + matrix_keys("D2T")
    + vector_keys("D2PHIDT2")
    + vector_keys("DT_RECON")
    + matrix_keys("D2T_RECON")
    + matrix_keys("DP")
    + hessian_keys("D2P")
    + ["NU"]
)

SUMMARY_KEYS = [
    "ALL_FINITE",
    "CENTER_FULL_DP_OVERLAP",
    "C1_C2_DP_OVERLAP",
    "EVENT_TRANSVERSALITY_CERTIFIED",
    "IMPACT_TIME_CROSSCHECK",
    "C1_ORIENTATION_UNRESOLVED",
    "C2_HULL_ORIENTATION_UNRESOLVED",
    "AFFINE_ORIENTATION_CERTIFIED",
    "LIOUVILLE_ORIENTATION_CERTIFIED",
    "AFFINE_LIOUVILLE_OVERLAP",
    "AFFINE_LIOUVILLE_SAME_SIGN",
    "PROJECTIVE_X_ORIENTATION_CERTIFIED",
    "PROJECTIVE_Y_ORIENTATION_CERTIFIED",
    "ANY_PROJECTIVE_ORIENTATION_CERTIFIED",
    "AFFINE_STRICTLY_NARROWER_THAN_C1",
    "AFFINE_STRICTLY_NARROWER_THAN_C2_HULL",
    "STRUCTURAL_PASS",
    "CERTIFICATE_PASS",
    "PROBE_PASS",
]

RECORD_SPECS: tuple[tuple[str, Sequence[str], set[str]], ...] = (
    (
        "SOURCE_TILE",
        ["U", "S"] + vector_keys("DELTA") + matrix_keys("Q0"),
        set(),
    ),
    (
        "C1_P1_TRANSVERSALITY",
        ["TIME"] + vector_keys("X") + ["NU"],
        set(),
    ),
    (
        "C1_P2_CONTROL",
        ["TIME"] + vector_keys("X") + matrix_keys("DP") + ["NU", "DET"],
        set(),
    ),
    ("C2_FULL_P2", C2_KEYS + ["HULL_DET"], set()),
    ("C2_CENTER_P2", C2_KEYS, set()),
    (
        "AFFINE_CARRIER",
        matrix_keys("M")
        + matrix_keys("A0")
        + matrix_keys("A1")
        + matrix_keys("R")
        + matrix_keys("HULL")
        + ["DET_POLYNOMIAL", "DET_REMAINDER", "DET"],
        set(),
    ),
    (
        "PROJECTIVE_X",
        ["ELIGIBLE", "FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"],
        {"ELIGIBLE"},
    ),
    (
        "PROJECTIVE_Y",
        ["ELIGIBLE", "FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"],
        {"ELIGIBLE"},
    ),
    (
        "LIOUVILLE",
        ["TIME"] + vector_keys("X", 4) + ["NU0", "NU2", "ELL", "EXP_ELL", "DET"],
        set(),
    ),
    ("SUMMARY", SUMMARY_KEYS, set(SUMMARY_KEYS)),
)


@dataclass
class Ledger:
    headers: dict[str, str]
    records: dict[str, dict[str, Interval | bool]]
    receipt_sha256: str
    physical_sha256: str


def parse_bool(token: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    fail(f"malformed boolean: {token}")


def parse_ledger(path: Path) -> Ledger:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise VerificationError("receipt must be ASCII") from error
    if not text.endswith("\n") or "\r" in text:
        fail("receipt has noncanonical line endings")
    lines = text.splitlines()
    expected_lines = len(FIXED_HEADERS) + len(RECORD_SPECS)
    if len(lines) != expected_lines:
        fail(f"receipt line count mismatch: {len(lines)} != {expected_lines}")

    headers: dict[str, str] = {}
    for line, (expected_key, fixed_value) in zip(lines, FIXED_HEADERS):
        if line.count("=") != 1:
            fail("malformed header")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            fail(f"header order mismatch: expected {expected_key}")
        if fixed_value is not None and value != fixed_value:
            fail(f"fixed header mismatch: {key}")
        headers[key] = value

    records: dict[str, dict[str, Interval | bool]] = {}
    offset = len(FIXED_HEADERS)
    for line, (marker, expected_keys, bool_keys) in zip(lines[offset:], RECORD_SPECS):
        tokens = line.split(" ")
        if not tokens or tokens[0] != marker or len(tokens) != len(expected_keys) + 1:
            fail(f"record grammar mismatch: {marker}")
        values: dict[str, Interval | bool] = {}
        for token, expected_key in zip(tokens[1:], expected_keys):
            if token.count("=") != 1:
                fail(f"bad token in {marker}")
            key, value = token.split("=", 1)
            if key != expected_key or key in values:
                fail(f"key order mismatch in {marker}: expected {expected_key}")
            values[key] = parse_bool(value) if key in bool_keys else parse_interval(value)
        records[marker] = values

    physical_lines = lines[len(FIXED_HEADERS):-1]
    physical = hashlib.sha256(("\n".join(physical_lines) + "\n").encode("ascii")).hexdigest()
    return Ledger(headers, records, hashlib.sha256(raw).hexdigest(), physical)


def interval(record: Mapping[str, Interval | bool], key: str) -> Interval:
    value = record[key]
    if not isinstance(value, Interval):
        fail(f"expected interval: {key}")
    return value


def boolean(record: Mapping[str, Interval | bool], key: str) -> bool:
    value = record[key]
    if not isinstance(value, bool):
        fail(f"expected boolean: {key}")
    return value


def vector(record: Mapping[str, Interval | bool], prefix: str, size: int = 3) -> list[Interval]:
    return [interval(record, f"{prefix}{row}") for row in range(size)]


def matrix(record: Mapping[str, Interval | bool], prefix: str) -> list[list[Interval]]:
    return [
        [interval(record, f"{prefix}{row}{column}") for column in range(3)]
        for row in range(3)
    ]


def hessian_entry(
    record: Mapping[str, Interval | bool], prefix: str, image: int, first: int, second: int
) -> Interval:
    low, high = sorted((first, second))
    return interval(record, f"{prefix}{image}{low}{high}")


def matrix_multiply(left: list[list[Interval]], right: list[list[Interval]]) -> list[list[Interval]]:
    return [
        [sum_intervals(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def matrix_vector(left: list[list[Interval]], right: list[Interval]) -> list[Interval]:
    return [sum_intervals(left[row][k] * right[k] for k in range(3)) for row in range(3)]


def sum_intervals(values: Sequence[Interval] | object) -> Interval:
    result = ZERO
    for value in values:  # type: ignore[union-attr]
        result = result + value
    return result


def determinant_xy(value: list[list[Interval]]) -> Interval:
    return value[0][0] * value[1][1] - value[0][1] * value[1][0]


def require_contains(outer: Interval, inner: Interval, label: str) -> None:
    if not outer.contains(inner):
        fail(f"containment failed: {label}")


def require_overlap(left: Interval, right: Interval, label: str) -> None:
    if not left.overlaps(right):
        fail(f"overlap failed: {label}")


def require_tight_contains(
    reported: Interval, calculated: Interval, label: str, max_ulps: int = 4096
) -> None:
    if not reported.contains(calculated):
        fail(f"tight containment misses reconstruction: {label}")
    magnitude = max(abs(calculated.lower), abs(calculated.upper))
    ulp = Fraction.from_float(math.ulp(float(magnitude)))
    allowed = Interval(
        calculated.lower - max_ulps * ulp,
        calculated.upper + max_ulps * ulp,
    )
    if not allowed.contains(reported):
        fail(f"reported enclosure exceeds rounding budget: {label}")


def require_matrix_reconstruction(
    reported: Sequence[Sequence[Interval]],
    calculated: Sequence[Sequence[Interval]],
    label: str,
) -> None:
    for row in range(3):
        for column in range(3):
            require_tight_contains(
                reported[row][column],
                calculated[row][column],
                f"{label}[{row},{column}]",
            )


def joint_interval(values: Sequence[Interval]) -> bool:
    return bool(values) and max(value.lower for value in values) <= min(
        value.upper for value in values
    )


def joint_vectors(values: Sequence[Sequence[Interval]]) -> bool:
    return bool(values) and all(
        joint_interval([value[row] for value in values])
        for row in range(len(values[0]))
    )


def matrix_overlap(left: list[list[Interval]], right: list[list[Interval]]) -> bool:
    return all(left[i][j].overlaps(right[i][j]) for i in range(3) for j in range(3))


def field_and_derivative(state: list[Interval]) -> tuple[list[Interval], list[list[Interval]]]:
    x, y, w = state
    field = [
        TWO * y * y - x * y,
        x * y - y * (w + ZS) * HALF,
        x * y - w - ZS,
    ]
    derivative = [[ZERO for _ in range(3)] for _ in range(3)]
    derivative[0][0] = -y
    derivative[0][1] = y.scale(4) - x
    derivative[1][0] = y
    derivative[1][1] = x - (w + ZS) * HALF
    derivative[1][2] = -y * HALF
    derivative[2][0] = y
    derivative[2][1] = x
    derivative[2][2] = point(-1)
    return field, derivative


def frozen_geometry() -> dict[str, Interval]:
    return {
        "zs": decimal_interval("22.3274637391"),
        "origin_x": decimal_interval("15.186446520640786"),
        "origin_y": decimal_interval("10.908543194765466"),
        "unstable_x": decimal_interval("-0.67430316214199759"),
        "unstable_y": decimal_interval("-0.73845463335624273"),
        "stable_x": decimal_interval("-0.94170446778164518"),
        "stable_y": decimal_interval("0.33644122125579123"),
        "radius_u": decimal_interval("0.004"),
        "radius_s": decimal_interval("0.3"),
    }


def frozen_tile(radius: Interval, index: int, count: int) -> tuple[Interval, Fraction]:
    left = ZERO - radius
    step = radius.scale(Fraction(2, count))
    logical = Interval(
        (left + step.scale(index)).lower,
        (left + step.scale(index + 1)).upper,
    )
    scale = max(abs(radius.lower), abs(radius.upper))
    slack = 32 * Fraction.from_float(math.ulp(float(scale)))
    return logical, slack


def require_tile(
    reported: Interval, logical: Interval, slack: Fraction, label: str
) -> None:
    if not reported.contains(logical):
        fail(f"indexed tile is not contained: {label}")
    if not Interval(logical.lower - slack, logical.upper + slack).contains(reported):
        fail(f"tile exceeds rounding budget: {label}")


def point_midpoint(value: Interval) -> Interval:
    midpoint = 0.5 * float(value.lower) + 0.5 * float(value.upper)
    return point(Fraction.from_float(midpoint))


def normal_velocity(
    state: Sequence[Interval], geometry: Mapping[str, Interval]
) -> Interval:
    return state[0] * state[1] - state[2] - geometry["zs"]


def exp_enclosure_negative(value: Interval, terms: int = 192) -> Interval:
    """Enclose exp(value) using exact-rational Taylor bounds."""
    if value.upper >= 0:
        fail("Liouville exponential requires a strictly negative exponent")

    def positive_exp_bounds(argument: Fraction) -> tuple[Fraction, Fraction]:
        if argument < 0:
            fail("positive exponential helper received a negative argument")
        partial = Fraction(1)
        term = Fraction(1)
        for order in range(1, terms + 1):
            term = term * argument / order
            partial += term
        next_term = term * argument / (terms + 1)
        ratio = argument / (terms + 2)
        if ratio >= 1:
            fail("Taylor tail ratio is not contractive")
        return partial, partial + next_term / (1 - ratio)

    largest_positive = -value.lower
    smallest_positive = -value.upper
    _, largest_upper = positive_exp_bounds(largest_positive)
    smallest_lower, _ = positive_exp_bounds(smallest_positive)
    return Interval(Fraction(1) / largest_upper, Fraction(1) / smallest_lower)


def verify_normal_velocity(record: Mapping[str, Interval | bool], label: str) -> Interval:
    field, _ = field_and_derivative(vector(record, "X"))
    nu = interval(record, "NU")
    require_contains(nu, field[2], f"{label} normal velocity")
    if nu.lower <= 0:
        fail(f"transversality not certified: {label}")
    return nu


def verify_c2(record: Mapping[str, Interval | bool], label: str) -> None:
    state = vector(record, "X")
    flow = matrix(record, "FLOW")
    dp = matrix(record, "DP")
    dt = vector(record, "DT")
    d2t = matrix(record, "D2T")
    dt_recon = vector(record, "DT_RECON")
    d2t_recon = matrix(record, "D2T_RECON")
    d2phi = vector(record, "D2PHIDT2")
    field, derivative = field_and_derivative(state)
    nu = interval(record, "NU")
    require_contains(nu, field[2], f"{label} nu")
    if nu.lower <= 0:
        fail(f"{label} section denominator crosses zero")

    direct_d2phi = [value * HALF for value in matrix_vector(derivative, field)]
    for row in range(3):
        require_overlap(d2phi[row], direct_d2phi[row], f"{label} d2phi[{row}]")

    for column in range(3):
        expected_dt = -flow[2][column] / nu
        require_contains(dt[column], expected_dt, f"{label} dt[{column}]")
        require_contains(dt_recon[column], expected_dt, f"{label} reconstructed dt[{column}]")
        require_overlap(dt[column], dt_recon[column], f"{label} dt crosscheck[{column}]")
        for row in range(2):
            expected_dp = flow[row][column] + field[row] * dt[column]
            require_contains(dp[row][column], expected_dp, f"{label} dp[{row},{column}]")
    for column in range(3):
        if dp[2][column] != ZERO or dp[column][2] != ZERO:
            fail(f"{label} dummy tangent slot not exact zero")

    dfield_flow = matrix_multiply(derivative, flow)
    for first in range(3):
        for second in range(first, 3):
            uncorrected: list[Interval] = []
            for image in range(3):
                base = hessian_entry(record, "FLOW_H", image, first, second)
                if first == second:
                    base = base + dt[first] * (
                        dfield_flow[image][first] + d2phi[image] * dt[first]
                    )
                else:
                    base = (
                        base
                        + dfield_flow[image][first] * dt[second]
                        + dfield_flow[image][second] * dt[first]
                        + d2phi[image] * dt[first] * dt[second] * TWO
                    )
                uncorrected.append(base)
            expected_d2t = -uncorrected[2] / nu
            require_contains(d2t[first][second], expected_d2t, f"{label} d2t[{first},{second}]")
            require_contains(
                d2t_recon[first][second], expected_d2t, f"{label} reconstructed d2t[{first},{second}]"
            )
            require_overlap(
                d2t[first][second], d2t_recon[first][second], f"{label} d2t crosscheck[{first},{second}]"
            )
            require_overlap(d2t[first][second], d2t[second][first], f"{label} d2t symmetry")
            for image in range(3):
                expected_d2p = uncorrected[image] + field[image] * d2t[first][second]
                recorded_d2p = hessian_entry(record, "D2P", image, first, second)
                require_contains(recorded_d2p, expected_d2p, f"{label} d2p[{image},{first},{second}]")

    for image in range(3):
        for index in range(3):
            if hessian_entry(record, "D2P", image, index, 2) != ZERO:
                fail(f"{label} dummy Hessian slot not exact zero")


def hessian_derivative(
    record: Mapping[str, Interval | bool], image: int, column: int, variable: int
) -> Interval:
    value = hessian_entry(record, "D2P", image, column, variable)
    return value.scale(2) if column == variable else value


def affine_hull(
    center: list[list[Interval]], coefficients: list[list[list[Interval]]], delta: list[Interval]
) -> list[list[Interval]]:
    result = [[center[i][j] for j in range(3)] for i in range(3)]
    for row in range(2):
        for column in range(2):
            for variable in range(2):
                result[row][column] = (
                    result[row][column] + coefficients[variable][row][column] * delta[variable]
                )
    return result


def verify_affine(ledger: Ledger) -> dict[str, Interval | bool]:
    source = ledger.records["SOURCE_TILE"]
    full = ledger.records["C2_FULL_P2"]
    center_record = ledger.records["C2_CENTER_P2"]
    carrier = ledger.records["AFFINE_CARRIER"]
    delta = vector(source, "DELTA")
    center_dp = matrix(center_record, "DP")
    m = matrix(carrier, "M")
    coefficients = [matrix(carrier, "A0"), matrix(carrier, "A1")]
    residual = matrix(carrier, "R")
    recorded_hull = matrix(carrier, "HULL")

    for row in range(2):
        for column in range(2):
            if m[row][column].width != 0:
                fail("affine center is not point-valued")
            require_contains(center_dp[row][column], m[row][column], f"M[{row},{column}]")
            expected_residual = center_dp[row][column] - m[row][column]
            for variable in range(2):
                derivative = hessian_derivative(full, row, column, variable)
                coefficient = coefficients[variable][row][column]
                if coefficient.width != 0:
                    fail("affine coefficient is not point-valued")
                require_contains(derivative, coefficient, f"A{variable}[{row},{column}]")
                expected_residual = expected_residual + (derivative - coefficient) * delta[variable]
            require_contains(residual[row][column], expected_residual, f"R[{row},{column}]")

    for index in range(3):
        if (
            m[2][index] != ZERO
            or m[index][2] != ZERO
            or residual[2][index] != ZERO
            or residual[index][2] != ZERO
        ):
            fail("affine carrier escaped the tangent 2x2 block")
        for variable in range(2):
            if (
                coefficients[variable][2][index] != ZERO
                or coefficients[variable][index][2] != ZERO
            ):
                fail("affine coefficient escaped the tangent 2x2 block")

    base_hull = affine_hull(m, coefficients, delta)
    for row in range(3):
        for column in range(3):
            require_contains(
                recorded_hull[row][column],
                base_hull[row][column] + residual[row][column],
                f"affine hull[{row},{column}]",
            )

    constant = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    linear: list[Interval] = []
    square: list[Interval] = []
    for variable in range(2):
        a = coefficients[variable]
        linear.append(
            a[0][0] * m[1][1]
            + m[0][0] * a[1][1]
            - a[0][1] * m[1][0]
            - m[0][1] * a[1][0]
        )
        square.append(a[0][0] * a[1][1] - a[0][1] * a[1][0])
    cross = (
        coefficients[0][0][0] * coefficients[1][1][1]
        + coefficients[1][0][0] * coefficients[0][1][1]
        - coefficients[0][0][1] * coefficients[1][1][0]
        - coefficients[1][0][1] * coefficients[0][1][0]
    )
    polynomial = (
        constant
        + linear[0] * delta[0]
        + linear[1] * delta[1]
        + square[0] * centered_square(delta[0])
        + square[1] * centered_square(delta[1])
        + cross * delta[0] * delta[1]
    )
    recorded_polynomial = interval(carrier, "DET_POLYNOMIAL")
    require_contains(recorded_polynomial, polynomial, "determinant polynomial")

    e = residual
    correction = (
        base_hull[0][0] * e[1][1]
        + e[0][0] * base_hull[1][1]
        + e[0][0] * e[1][1]
        - base_hull[0][1] * e[1][0]
        - e[0][1] * base_hull[1][0]
        - e[0][1] * e[1][0]
    )
    recorded_correction = interval(carrier, "DET_REMAINDER")
    require_contains(recorded_correction, correction, "determinant remainder")
    determinant = interval(carrier, "DET")
    require_contains(determinant, recorded_polynomial + recorded_correction, "affine determinant sum")
    return {
        "det": determinant,
        "hull_det": determinant_xy(recorded_hull),
    }


def verify_projective(
    record: Mapping[str, Interval | bool], jacobian: list[list[Interval]], chart: str
) -> tuple[bool, bool]:
    if chart == "X":
        eligible = not jacobian[0][0].contains_zero() and not jacobian[0][1].contains_zero()
    else:
        eligible = not jacobian[1][0].contains_zero() and not jacobian[1][1].contains_zero()
    if boolean(record, "ELIGIBLE") != eligible:
        fail(f"projective {chart} eligibility mismatch")
    if not eligible:
        for key in ("FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"):
            if interval(record, key) != ZERO:
                fail(f"ineligible projective {chart} chart has nonzero payload")
        return eligible, False

    if chart == "X":
        first = jacobian[1][0] / jacobian[0][0]
        second = jacobian[1][1] / jacobian[0][1]
        separation = second - first
        scale = jacobian[0][0] * jacobian[0][1]
    else:
        first = jacobian[0][0] / jacobian[1][0]
        second = jacobian[0][1] / jacobian[1][1]
        separation = first - second
        scale = jacobian[1][0] * jacobian[1][1]
    determinant = scale * separation
    for key, expected in (
        ("FIRST_SLOPE", first),
        ("SECOND_SLOPE", second),
        ("SEPARATION", separation),
        ("SCALE", scale),
        ("DET", determinant),
    ):
        require_contains(interval(record, key), expected, f"projective {chart} {key}")
    return eligible, not interval(record, "DET").contains_zero()


def verify_ledger(
    ledger: Ledger, expected_source: str, expected_input: str, expected_challenge: str
) -> None:
    for key, expected in (
        ("WORKER_SOURCE_SHA256", expected_source),
        ("INPUT_SHA256", expected_input),
        ("RUN_CHALLENGE", expected_challenge),
    ):
        if SHA_RE.fullmatch(ledger.headers[key]) is None or ledger.headers[key] != expected:
            fail(f"dynamic header mismatch: {key}")

    geometry = frozen_geometry()
    source = ledger.records["SOURCE_TILE"]
    source_u = interval(source, "U")
    source_s = interval(source, "S")
    logical_u, slack_u = frozen_tile(geometry["radius_u"], 20000, 40000)
    logical_s, slack_s = frozen_tile(geometry["radius_s"], 15000, 30000)
    require_tile(source_u, logical_u, slack_u, "SOURCE U")
    require_tile(source_s, logical_s, slack_s, "SOURCE S")

    q0 = matrix(source, "Q0")
    expected_q0 = [[ZERO for _ in range(3)] for _ in range(3)]
    expected_q0[0][0] = geometry["unstable_x"] * geometry["radius_u"]
    expected_q0[1][0] = geometry["unstable_y"] * geometry["radius_u"]
    expected_q0[0][1] = geometry["stable_x"] * geometry["radius_s"]
    expected_q0[1][1] = geometry["stable_y"] * geometry["radius_s"]
    require_matrix_reconstruction(q0, expected_q0, "SOURCE Q0")

    delta = vector(source, "DELTA")
    expected_delta = [
        (source_u - point_midpoint(source_u)) / geometry["radius_u"],
        (source_s - point_midpoint(source_s)) / geometry["radius_s"],
        ZERO,
    ]
    for index in range(3):
        require_tight_contains(
            delta[index], expected_delta[index], f"SOURCE DELTA{index}"
        )
    if delta[0].lower != -delta[0].upper or delta[1].lower != -delta[1].upper:
        fail("normalized tile is not centered")
    if delta[0].width <= 0 or delta[1].width <= 0:
        fail("normalized tile radius was erased")
    if determinant_xy(q0).upper >= 0:
        fail("Q0 orientation is not strictly negative")

    p1 = ledger.records["C1_P1_TRANSVERSALITY"]
    p2 = ledger.records["C1_P2_CONTROL"]
    p1_nu = verify_normal_velocity(p1, "C1 P1")
    p2_nu = verify_normal_velocity(p2, "C1 P2")
    if interval(p2, "TIME").lower <= interval(p1, "TIME").upper:
        fail("second event is not strictly later")
    c1_dp = matrix(p2, "DP")
    c1_det = determinant_xy(c1_dp)
    require_contains(interval(p2, "DET"), c1_det, "C1 determinant")
    if not interval(p2, "DET").contains_zero():
        fail("C1 boxed determinant unexpectedly sign-definite")

    full = ledger.records["C2_FULL_P2"]
    center = ledger.records["C2_CENTER_P2"]
    verify_c2(full, "C2 full")
    verify_c2(center, "C2 center")
    for index in range(3):
        require_overlap(interval(p2, f"X{index}"), interval(full, f"X{index}"), "C1/C2 state")
    require_overlap(interval(p2, "TIME"), interval(full, "TIME"), "C1/C2 time")
    if not matrix_overlap(c1_dp, matrix(full, "DP")):
        fail("C1/C2 DP mismatch")
    if not matrix_overlap(matrix(center, "DP"), matrix(full, "DP")):
        fail("center/full DP mismatch")
    full_hull_det = determinant_xy(matrix(full, "DP"))
    require_contains(interval(full, "HULL_DET"), full_hull_det, "C2 hull determinant")
    if not interval(full, "HULL_DET").contains_zero():
        fail("C2 boxed determinant unexpectedly sign-definite")

    affine = verify_affine(ledger)
    affine_det = affine["det"]
    assert isinstance(affine_det, Interval)
    if affine_det.upper >= 0:
        fail("affine carrier did not certify negative orientation")
    if affine_det.width >= interval(p2, "DET").width:
        fail("affine determinant did not improve C1 width")
    if affine_det.width >= interval(full, "HULL_DET").width:
        fail("affine determinant did not improve C2 hull width")

    affine_hull_matrix = matrix(ledger.records["AFFINE_CARRIER"], "HULL")
    _, projective_x = verify_projective(
        ledger.records["PROJECTIVE_X"], affine_hull_matrix, "X"
    )
    _, projective_y = verify_projective(
        ledger.records["PROJECTIVE_Y"], affine_hull_matrix, "Y"
    )

    liouville = ledger.records["LIOUVILLE"]
    liouville_time = interval(liouville, "TIME")
    liouville_state = vector(liouville, "X", 4)
    ell = interval(liouville, "ELL")
    exp_ell = interval(liouville, "EXP_ELL")
    liouville_nu0 = interval(liouville, "NU0")
    liouville_nu2 = interval(liouville, "NU2")
    if ell != liouville_state[3]:
        fail("Liouville ELL differs from the integrated fourth state")
    independent_exp = exp_enclosure_negative(ell)
    require_tight_contains(exp_ell, independent_exp, "Liouville EXP_ELL", 8192)

    initial_x = (
        geometry["origin_x"]
        + geometry["unstable_x"] * source_u
        + geometry["stable_x"] * source_s
    )
    initial_y = (
        geometry["origin_y"]
        + geometry["unstable_y"] * source_u
        + geometry["stable_y"] * source_s
    )
    calculated_nu0 = initial_x * initial_y - geometry["zs"]
    calculated_nu2 = normal_velocity(liouville_state[:3], geometry)
    require_tight_contains(liouville_nu0, calculated_nu0, "Liouville NU0")
    require_tight_contains(liouville_nu2, calculated_nu2, "Liouville NU2")
    if (
        liouville_time.lower <= 0
        or exp_ell.lower <= 0
        or liouville_nu0.lower <= 0
        or liouville_nu2.lower <= 0
    ):
        fail("Liouville time, exponential, or normal velocity is not positive")

    frame_det = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    oriented_q0_area = frame_det * geometry["radius_u"] * geometry["radius_s"]
    liouville_formula = exp_ell * liouville_nu0 / liouville_nu2 * oriented_q0_area
    liouville_det = interval(liouville, "DET")
    require_tight_contains(
        liouville_det, liouville_formula, "Liouville determinant identity"
    )
    if liouville_det.upper >= 0:
        fail("Liouville determinant is not strictly negative")
    if not joint_interval(
        [interval(p2, "TIME"), interval(full, "TIME"), liouville_time]
    ):
        fail("Liouville time does not share the C1/C2 second return")
    if not joint_vectors(
        [vector(p2, "X"), vector(full, "X"), liouville_state[:3]]
    ):
        fail("Liouville state does not share the C1/C2 second return")
    if not joint_interval([p2_nu, interval(full, "NU"), liouville_nu2]):
        fail("Liouville normal velocity does not share the C1/C2 second return")
    require_overlap(affine_det, liouville_det, "affine/Liouville determinant")

    computed = {
        "ALL_FINITE": True,
        "CENTER_FULL_DP_OVERLAP": matrix_overlap(matrix(center, "DP"), matrix(full, "DP")),
        "C1_C2_DP_OVERLAP": matrix_overlap(c1_dp, matrix(full, "DP")),
        "EVENT_TRANSVERSALITY_CERTIFIED": (
            p1_nu.lower > 0
            and p2_nu.lower > 0
            and interval(full, "NU").lower > 0
            and interval(center, "NU").lower > 0
        ),
        "IMPACT_TIME_CROSSCHECK": True,
        "C1_ORIENTATION_UNRESOLVED": interval(p2, "DET").contains_zero(),
        "C2_HULL_ORIENTATION_UNRESOLVED": interval(full, "HULL_DET").contains_zero(),
        "AFFINE_ORIENTATION_CERTIFIED": affine_det.upper < 0 or affine_det.lower > 0,
        "LIOUVILLE_ORIENTATION_CERTIFIED": liouville_det.upper < 0 or liouville_det.lower > 0,
        "AFFINE_LIOUVILLE_OVERLAP": affine_det.overlaps(liouville_det),
        "AFFINE_LIOUVILLE_SAME_SIGN": affine_det.upper < 0 and liouville_det.upper < 0,
        "PROJECTIVE_X_ORIENTATION_CERTIFIED": projective_x,
        "PROJECTIVE_Y_ORIENTATION_CERTIFIED": projective_y,
        "ANY_PROJECTIVE_ORIENTATION_CERTIFIED": projective_x or projective_y,
        "AFFINE_STRICTLY_NARROWER_THAN_C1": affine_det.width < interval(p2, "DET").width,
        "AFFINE_STRICTLY_NARROWER_THAN_C2_HULL": affine_det.width < interval(full, "HULL_DET").width,
    }
    computed["STRUCTURAL_PASS"] = all(
        computed[key]
        for key in (
            "ALL_FINITE",
            "CENTER_FULL_DP_OVERLAP",
            "C1_C2_DP_OVERLAP",
            "EVENT_TRANSVERSALITY_CERTIFIED",
            "IMPACT_TIME_CROSSCHECK",
            "C1_ORIENTATION_UNRESOLVED",
            "C2_HULL_ORIENTATION_UNRESOLVED",
            "LIOUVILLE_ORIENTATION_CERTIFIED",
            "AFFINE_STRICTLY_NARROWER_THAN_C1",
            "AFFINE_STRICTLY_NARROWER_THAN_C2_HULL",
        )
    )
    computed["CERTIFICATE_PASS"] = (
        computed["STRUCTURAL_PASS"]
        and computed["AFFINE_ORIENTATION_CERTIFIED"]
        and computed["AFFINE_LIOUVILLE_OVERLAP"]
        and computed["AFFINE_LIOUVILLE_SAME_SIGN"]
    )
    computed["PROBE_PASS"] = computed["STRUCTURAL_PASS"] and computed["CERTIFICATE_PASS"]
    summary = ledger.records["SUMMARY"]
    for key in SUMMARY_KEYS:
        if boolean(summary, key) != computed[key]:
            fail(f"summary mismatch: {key}")


Mutation = tuple[str, Callable[[Ledger], None]]


def set_interval(ledger: Ledger, marker: str, key: str, value: Interval) -> None:
    ledger.records[marker][key] = value


def mutation_suite() -> list[Mutation]:
    def swap_axes(ledger: Ledger) -> None:
        record = ledger.records["AFFINE_CARRIER"]
        record["A000"], record["A100"] = record["A100"], record["A000"]

    def swap_delta_axes(ledger: Ledger) -> None:
        record = ledger.records["SOURCE_TILE"]
        record["DELTA0"], record["DELTA1"] = record["DELTA1"], record["DELTA0"]

    def scale_delta(ledger: Ledger, key: str, factor: int) -> None:
        record = ledger.records["SOURCE_TILE"]
        set_interval(ledger, "SOURCE_TILE", key, interval(record, key).scale(factor))

    def make_delta_asymmetric(ledger: Ledger) -> None:
        value = interval(ledger.records["SOURCE_TILE"], "DELTA0")
        set_interval(ledger, "SOURCE_TILE", "DELTA0", Interval(Fraction(0), value.upper))

    def zero_ell_and_state(ledger: Ledger) -> None:
        set_interval(ledger, "LIOUVILLE", "ELL", ZERO)
        set_interval(ledger, "LIOUVILLE", "X3", ZERO)

    def zero_liouville_state(ledger: Ledger) -> None:
        for key in ("X0", "X1", "X2"):
            set_interval(ledger, "LIOUVILLE", key, ZERO)

    def header_mutation(key: str) -> Callable[[Ledger], None]:
        return lambda ledger: ledger.headers.__setitem__(key, "0" * 64)

    return [
        ("diagonal_factor_two_removed", lambda x: set_interval(x, "AFFINE_CARRIER", "A000", interval(x.records["C2_FULL_P2"], "D2P000"))),
        ("flow_hessian_used_as_return_hessian", lambda x: set_interval(x, "AFFINE_CARRIER", "A000", interval(x.records["C2_FULL_P2"], "FLOW_H000"))),
        ("source_axes_swapped", swap_axes),
        ("mixed_coefficient_removed", lambda x: set_interval(x, "AFFINE_CARRIER", "A001", ZERO)),
        ("remainder_erased", lambda x: set_interval(x, "AFFINE_CARRIER", "R00", ZERO)),
        ("determinant_polynomial_erased", lambda x: set_interval(x, "AFFINE_CARRIER", "DET_POLYNOMIAL", ZERO)),
        ("affine_determinant_erased", lambda x: set_interval(x, "AFFINE_CARRIER", "DET", ZERO)),
        ("impact_time_derivative_erased", lambda x: set_interval(x, "C2_FULL_P2", "DT0", ZERO)),
        ("impact_time_crosscheck_erased", lambda x: set_interval(x, "C2_FULL_P2", "DT_RECON0", ZERO)),
        ("second_impact_time_erased", lambda x: set_interval(x, "C2_FULL_P2", "D2T01", ZERO)),
        ("poincare_hessian_replaced_by_flow", lambda x: set_interval(x, "C2_FULL_P2", "D2P000", interval(x.records["C2_FULL_P2"], "FLOW_H000"))),
        ("transversality_crosses_zero", lambda x: set_interval(x, "C1_P1_TRANSVERSALITY", "NU", Interval(Fraction(-1), Fraction(1)))),
        ("c1_determinant_erased", lambda x: set_interval(x, "C1_P2_CONTROL", "DET", ZERO)),
        ("c2_hull_determinant_erased", lambda x: set_interval(x, "C2_FULL_P2", "HULL_DET", ZERO)),
        ("projective_pole_accepted", lambda x: x.records["PROJECTIVE_X"].__setitem__("ELIGIBLE", True)),
        ("liouville_sign_flipped", lambda x: set_interval(x, "LIOUVILLE", "DET", ONE)),
        ("summary_claim_flipped", lambda x: x.records["SUMMARY"].__setitem__("AFFINE_ORIENTATION_CERTIFIED", False)),
        ("challenge_substituted", header_mutation("RUN_CHALLENGE")),
        ("source_hash_substituted", header_mutation("WORKER_SOURCE_SHA256")),
        ("source_u_outside_frozen_tile", lambda x: set_interval(x, "SOURCE_TILE", "U", ONE)),
        ("source_s_outside_frozen_tile", lambda x: set_interval(x, "SOURCE_TILE", "S", ONE)),
        ("q0_00_substituted", lambda x: set_interval(x, "SOURCE_TILE", "Q000", ONE)),
        ("q0_10_substituted", lambda x: set_interval(x, "SOURCE_TILE", "Q010", ONE)),
        ("q0_01_substituted", lambda x: set_interval(x, "SOURCE_TILE", "Q001", ONE)),
        ("q0_11_substituted", lambda x: set_interval(x, "SOURCE_TILE", "Q011", ONE)),
        ("dummy_q0_column_activated", lambda x: set_interval(x, "SOURCE_TILE", "Q002", ONE)),
        ("delta0_wrong_scale", lambda x: scale_delta(x, "DELTA0", 2)),
        ("delta1_wrong_scale", lambda x: scale_delta(x, "DELTA1", 2)),
        ("delta_axes_swapped", swap_delta_axes),
        ("delta_asymmetric", make_delta_asymmetric),
        ("source_delta_erased", lambda x: set_interval(x, "SOURCE_TILE", "DELTA0", ZERO)),
        ("liouville_ell_and_state_erased", zero_ell_and_state),
        ("liouville_ell_state_mismatch", lambda x: set_interval(x, "LIOUVILLE", "ELL", ZERO)),
        ("liouville_exp_wrong_positive", lambda x: set_interval(x, "LIOUVILLE", "EXP_ELL", ONE)),
        ("liouville_nu0_wrong_positive", lambda x: set_interval(x, "LIOUVILLE", "NU0", ONE)),
        ("liouville_nu2_wrong_positive", lambda x: set_interval(x, "LIOUVILLE", "NU2", ONE)),
        ("liouville_state_erased", zero_liouville_state),
        ("liouville_time_erased", lambda x: set_interval(x, "LIOUVILLE", "TIME", ZERO)),
        ("liouville_formula_wrong_negative", lambda x: set_interval(x, "LIOUVILLE", "DET", point(-1))),
    ]


def run_mutations(
    ledger: Ledger, expected_source: str, expected_input: str, expected_challenge: str
) -> tuple[int, int]:
    rejected = 0
    suite = mutation_suite()
    for name, mutation in suite:
        candidate = copy.deepcopy(ledger)
        mutation(candidate)
        try:
            verify_ledger(candidate, expected_source, expected_input, expected_challenge)
        except VerificationError:
            rejected += 1
        else:
            fail(f"mutation escaped verifier: {name}")
    return len(suite), rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--input-sha", required=True)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--self-test-mutations", action="store_true")
    args = parser.parse_args(argv)
    for label, value in (
        ("source", args.source_sha),
        ("input", args.input_sha),
        ("challenge", args.challenge),
    ):
        if SHA_RE.fullmatch(value) is None:
            fail(f"{label} digest must be lowercase SHA-256")

    ledger = parse_ledger(args.receipt)
    verify_ledger(ledger, args.source_sha, args.input_sha, args.challenge)
    total = rejected = 0
    if args.self_test_mutations:
        total, rejected = run_mutations(
            ledger, args.source_sha, args.input_sha, args.challenge
        )
    print("VERIFICATION_SCHEMA=sounio.cs6.c1-dependency-verification.v1")
    print(f"RECEIPT_SHA256={ledger.receipt_sha256}")
    print(f"PHYSICAL_SHA256={ledger.physical_sha256}")
    print(f"MUTATION_TESTS={total}")
    print(f"MUTATIONS_REJECTED={rejected}")
    print("CERTIFICATE_PASS=true")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
