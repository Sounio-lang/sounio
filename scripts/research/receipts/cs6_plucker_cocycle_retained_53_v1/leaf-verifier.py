#!/usr/bin/env python3
"""Exact verifier for one dyadic CS6 C1 full-source-cover leaf."""

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
CANONICAL_INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
HEX_RE = re.compile(
    r"^-?0x[01]\.[0-9a-f]{0,12}[1-9a-f]p(?:\+[1-9][0-9]*|\+0|-[1-9][0-9]*)$"
)
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
LEAF_INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"


@dataclass(frozen=True)
class LeafInput:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    sha256: str


def read_leaf_input(path: Path) -> LeafInput:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise VerificationError("leaf input must be ASCII") from error
    except OSError as error:
        raise VerificationError(f"cannot read leaf input: {path}") from error
    if b"\x00" in raw or b"\r" in raw or not raw.endswith(b"\n"):
        fail("leaf input must use canonical LF-terminated ASCII")

    expected_keys = ("SCHEMA", "SOURCE", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
    lines = text[:-1].split("\n")
    if len(lines) != len(expected_keys):
        fail("leaf input has the wrong number of lines")
    fields: dict[str, str] = {}
    for line, expected_key in zip(lines, expected_keys, strict=True):
        if line.count("=") != 1:
            fail("malformed leaf input line")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            fail(f"noncanonical leaf input field: {expected_key}")
        fields[key] = value
    if fields["SCHEMA"] != LEAF_INPUT_SCHEMA or fields["SOURCE"] != "N0":
        fail("leaf input schema or source mismatch")

    values: dict[str, int] = {}
    for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX"):
        token = fields[key]
        if CANONICAL_INT_RE.fullmatch(token) is None:
            fail(f"noncanonical leaf input integer: {key}")
        values[key] = int(token)
    if values["U_DEPTH"] > 30 or values["S_DEPTH"] > 30:
        fail("leaf input depth exceeds worker contract")
    if not (0 <= values["U_INDEX"] < 1 << values["U_DEPTH"]):
        fail("leaf input U index out of range")
    if not (0 <= values["S_INDEX"] < 1 << values["S_DEPTH"]):
        fail("leaf input S index out of range")

    canonical = (
        f"SCHEMA={LEAF_INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={values['U_DEPTH']}\n"
        f"U_INDEX={values['U_INDEX']}\n"
        f"S_DEPTH={values['S_DEPTH']}\n"
        f"S_INDEX={values['S_INDEX']}\n"
    ).encode("ascii")
    if raw != canonical:
        fail("leaf input bytes are not canonical")
    return LeafInput(
        u_depth=values["U_DEPTH"],
        u_index=values["U_INDEX"],
        s_depth=values["S_DEPTH"],
        s_index=values["S_INDEX"],
        sha256=hashlib.sha256(raw).hexdigest(),
    )


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
    try:
        outer_lower = Fraction.from_float(float.fromhex(lower_token))
        outer_upper = Fraction.from_float(float.fromhex(upper_token))
    except (ValueError, OverflowError) as error:
        raise VerificationError(f"invalid hexadecimal endpoint: {token}") from error
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
    ("SCHEMA", "sounio.cs6.plucker-cocycle-leaf.v1"),
    ("WORKER_SOURCE_SHA256", None),
    ("INPUT_SHA256", None),
    ("RUN_CHALLENGE", None),
    ("CAPD_SOURCE_TREE_DECLARED", "capd-5.3.0"),
    ("INTERVAL_BACKEND_DECLARED", "FILIB"),
    ("INTERVAL_SERIALIZATION", "ONE_ULP_OUTWARD_BINARY64_HEX"),
    ("SOURCE", "N0"),
    ("U_DEPTH", None),
    ("U_INDEX", None),
    ("S_DEPTH", None),
    ("S_INDEX", None),
    ("U_TILES", None),
    ("S_TILES", None),
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
    (
        "PROJECTIVE_CONTROL",
        "FOUR_FIXED_COVECTOR_FINAL_COLUMN_SLOPE_CHARTS",
    ),
    ("PROJECTIVE_COVECTORS", "X,Y,X_PLUS_Y,X_MINUS_Y"),
    (
        "PROJECTIVE_TRANSFORMS",
        "X_1_0_0_1_DET_1,Y_0_1_1_0_DET_NEG1,PLUS_1_1_NEG1_1_DET_2,MINUS_1_NEG1_1_1_DET_2",
    ),
    (
        "PROJECTIVE_DETERMINANT_IDENTITY",
        "PIVOT0_TIMES_PIVOT1_TIMES_SLOPE_SEPARATION_OVER_TRANSFORM_DET",
    ),
    ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
    ("HOMOGENEOUS_CARRIER", "EVENT1_NORMALIZE_LOCAL_P2_EVENT2_NORMALIZE"),
    ("EVENTWISE_NORMALIZATION", "true"),
    ("DISCRETE_POINCARE_COCYCLE", "true"),
    ("CHART_NORMALIZATION_SCOPE", "EVENT_BOUNDARIES"),
    ("GRASSMANN_OBJECT", "GR_1_2_EQUALS_REAL_PROJECTIVE_LINE"),
    ("GRASSMANN_SCOPE", "GR_1_2_EQUALS_P1"),
    (
        "PLUCKER_ROLE",
        "HOMOGENEOUS_RAYS_WITH_FACTORED_SIGNED_EXTERIOR_RECONSTRUCTION",
    ),
    ("PLUCKER_RELATIONS_NONTRIVIAL", "false"),
    ("EVENT_NORMALIZATION_COUNT", "2"),
    ("DYNAMIC_CHART_SET", "X,Y,PLUS,MINUS"),
    (
        "DYNAMIC_CHART_SELECTION",
        "MAX_CERTIFIED_PIVOT_MARGIN_SQUARED_OVER_COVECTOR_NORM_SQUARED",
    ),
    ("EVENT1_C0_CARRIER", "MIDPOINT_MEAN_VALUE_CORRELATED_SOURCE_DELTA"),
    ("TANGENT_DEPENDENCY_MODEL", "INTERVAL_RAYS_WITH_EVENT_NORMALIZATION"),
    ("TANGENT_SOURCE_DEPENDENCY_AT_SWITCH", "BOXED_NOT_AFFINE"),
    ("CONTINUOUS_RICCATI_INTEGRATED", "false"),
    ("CONTINUOUS_PROJECTIVE_FLOW_INTEGRATED", "false"),
    ("GENERAL_GRASSMANN_PLUCKER_INTEGRATOR", "false"),
    ("AUTONOMOUS_VECTOR_FIELD", "true"),
    ("EVENT_TIME_SENSITIVITY_PROPAGATED", "false"),
    ("NONAUTONOMOUS_GENERALIZATION_PROVED", "false"),
    ("LIOUVILLE_ROLE", "INDEPENDENT_SIGN_CROSS_CHECK_ONLY"),
    ("EXECUTION_SCOPE", "DYADIC_LEAF_EVENT_NORMALIZED_CAPD_CPU_PROBE"),
    ("FPGA_EXECUTION", "false"),
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
    "P1_CENTER_FULL_DP_OVERLAP",
    "CENTER_FULL_DP_OVERLAP",
    "C1_C2_DP_OVERLAP",
    "EVENT1_MEAN_VALUE_OVERLAP",
    "EVENT1_RAY_RECONSTRUCTION",
    "EVENT2_RAY_RECONSTRUCTION",
    "CUMULATIVE_MATRIX_OVERLAP",
    "EVENT_ORDER_CERTIFIED",
    "POSTSECTION_PLUS_SIDE",
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
    "PROJECTIVE_PLUS_ORIENTATION_CERTIFIED",
    "PROJECTIVE_MINUS_ORIENTATION_CERTIFIED",
    "ANY_PROJECTIVE_ORIENTATION_CERTIFIED",
    "AFFINE_STRICTLY_NARROWER_THAN_C1",
    "AFFINE_STRICTLY_NARROWER_THAN_C2_HULL",
    "EVENT1_CHARTS_CERTIFIED",
    "EVENT2_CHARTS_CERTIFIED",
    "HOMOGENEOUS_ORIENTATION_CERTIFIED",
    "HOMOGENEOUS_LIOUVILLE_OVERLAP",
    "HOMOGENEOUS_LIOUVILLE_SAME_SIGN",
    "HOMOGENEOUS_JOINT_OVERLAP",
    "HOMOGENEOUS_STRICTLY_NARROWER_THAN_AFFINE",
    "HOMOGENEOUS_STRICTLY_NARROWER_THAN_BEST_FIXED",
    "HOMOGENEOUS_COMPUTATION_VALID",
    "HOMOGENEOUS_CERTIFICATE_PASS",
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
    ("C2_FULL_P1", C2_KEYS, set()),
    ("C2_CENTER_P1", C2_KEYS, set()),
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
        "PROJECTIVE_PLUS",
        ["ELIGIBLE", "FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"],
        {"ELIGIBLE"},
    ),
    (
        "PROJECTIVE_MINUS",
        ["ELIGIBLE", "FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"],
        {"ELIGIBLE"},
    ),
    (
        "EVENT1_MEAN_VALUE",
        ["TIME"]
        + vector_keys("CENTER")
        + matrix_keys("BASIS")
        + vector_keys("DELTA")
        + vector_keys("RESIDUAL")
        + vector_keys("HULL"),
        set(),
    ),
    *tuple(
        (
            marker,
            [
                "ELIGIBLE",
                "CHART",
                "PIVOT_X",
                "PIVOT_Y",
                "PIVOT_PLUS",
                "PIVOT_MINUS",
                "PIVOT",
                "SLOPE",
                "SCALE",
            ]
            + vector_keys("N"),
            {"ELIGIBLE"},
        )
        for marker in ("HOMOGENEOUS_EVENT1_RAY0", "HOMOGENEOUS_EVENT1_RAY1")
    ),
    (
        "HOMOGENEOUS_LOCAL_P2",
        ["TIME", "DURATION"]
        + vector_keys("X")
        + matrix_keys("FLOW")
        + matrix_keys("DP")
        + ["POST_TIME"]
        + vector_keys("POST_X")
        + ["NU"]
        + matrix_keys("RECON_DP")
        + ["RECON_DET"],
        set(),
    ),
    *tuple(
        (
            marker,
            [
                "ELIGIBLE",
                "CHART",
                "PIVOT_X",
                "PIVOT_Y",
                "PIVOT_PLUS",
                "PIVOT_MINUS",
                "PIVOT",
                "SLOPE",
                "SCALE",
            ]
            + vector_keys("N"),
            {"ELIGIBLE"},
        )
        for marker in ("HOMOGENEOUS_EVENT2_RAY0", "HOMOGENEOUS_EVENT2_RAY1")
    ),
    (
        "PLUCKER_COCYCLE",
        ["TOTAL_SCALE0", "TOTAL_SCALE1", "NORMALIZED_EXTERIOR", "DET"],
        set(),
    ),
    (
        "LIOUVILLE",
        ["TIME"] + vector_keys("X", 4) + ["NU0", "NU2", "ELL", "EXP_ELL", "DET"],
        set(),
    ),
    (
        "LEAF_RESULT",
        [
            "TERMINAL_CERTIFIED",
            "SUBDIVISION_REQUIRED",
            "AFFINE_CERTIFICATE_PASS",
            "PROJECTIVE_X_CERTIFICATE_PASS",
            "PROJECTIVE_Y_CERTIFICATE_PASS",
            "PROJECTIVE_PLUS_CERTIFICATE_PASS",
            "PROJECTIVE_MINUS_CERTIFICATE_PASS",
            "HOMOGENEOUS_CERTIFICATE_PASS",
        ],
        {
            "TERMINAL_CERTIFIED",
            "SUBDIVISION_REQUIRED",
            "AFFINE_CERTIFICATE_PASS",
            "PROJECTIVE_X_CERTIFICATE_PASS",
            "PROJECTIVE_Y_CERTIFICATE_PASS",
            "PROJECTIVE_PLUS_CERTIFICATE_PASS",
            "PROJECTIVE_MINUS_CERTIFICATE_PASS",
            "HOMOGENEOUS_CERTIFICATE_PASS",
        },
    ),
    ("SUMMARY", SUMMARY_KEYS, set(SUMMARY_KEYS)),
)


@dataclass
class Ledger:
    headers: dict[str, str]
    records: dict[str, dict[str, Interval | bool | str]]
    receipt_sha256: str
    physical_sha256: str


def parse_bool(token: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    fail(f"malformed boolean: {token}")


def canonical_header_int(headers: Mapping[str, str], key: str) -> int:
    token = headers[key]
    if CANONICAL_INT_RE.fullmatch(token) is None:
        fail(f"noncanonical integer header: {key}")
    return int(token)


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

    records: dict[str, dict[str, Interval | bool | str]] = {}
    offset = len(FIXED_HEADERS)
    for line, (marker, expected_keys, bool_keys) in zip(lines[offset:], RECORD_SPECS):
        tokens = line.split(" ")
        if not tokens or tokens[0] != marker or len(tokens) != len(expected_keys) + 1:
            fail(f"record grammar mismatch: {marker}")
        values: dict[str, Interval | bool | str] = {}
        for token, expected_key in zip(tokens[1:], expected_keys):
            if token.count("=") != 1:
                fail(f"bad token in {marker}")
            key, value = token.split("=", 1)
            if key != expected_key or key in values:
                fail(f"key order mismatch in {marker}: expected {expected_key}")
            if key in bool_keys:
                values[key] = parse_bool(value)
            elif key == "CHART":
                if value not in {"NONE", "X", "Y", "PLUS", "MINUS"}:
                    fail(f"unknown chart in {marker}")
                values[key] = value
            else:
                values[key] = parse_interval(value)
        records[marker] = values

    physical_record_count = len(RECORD_SPECS) - 2
    physical_lines = lines[
        len(FIXED_HEADERS) : len(FIXED_HEADERS) + physical_record_count
    ]
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


def string_value(record: Mapping[str, Interval | bool | str], key: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        fail(f"expected string: {key}")
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
    record: Mapping[str, Interval | bool],
    jacobian: list[list[Interval]],
    chart: str,
    transform: tuple[int, int, int, int, int],
) -> tuple[bool, bool, Interval]:
    l0, l1, m0, m1, transform_det = transform
    pivot0 = jacobian[0][0].scale(l0) + jacobian[1][0].scale(l1)
    pivot1 = jacobian[0][1].scale(l0) + jacobian[1][1].scale(l1)
    complement0 = jacobian[0][0].scale(m0) + jacobian[1][0].scale(m1)
    complement1 = jacobian[0][1].scale(m0) + jacobian[1][1].scale(m1)
    eligible = not pivot0.contains_zero() and not pivot1.contains_zero()
    if boolean(record, "ELIGIBLE") != eligible:
        fail(f"projective {chart} eligibility mismatch")
    if not eligible:
        for key in ("FIRST_SLOPE", "SECOND_SLOPE", "SEPARATION", "SCALE", "DET"):
            if interval(record, key) != ZERO:
                fail(f"ineligible projective {chart} chart has nonzero payload")
        return eligible, False, ZERO

    first = complement0 / pivot0
    second = complement1 / pivot1
    separation = second - first
    scale = pivot0 * pivot1 / point(transform_det)
    determinant = scale * separation
    for key, expected in (
        ("FIRST_SLOPE", first),
        ("SECOND_SLOPE", second),
        ("SEPARATION", separation),
        ("SCALE", scale),
        ("DET", determinant),
    ):
        require_contains(interval(record, key), expected, f"projective {chart} {key}")
    recorded_determinant = interval(record, "DET")
    return eligible, not recorded_determinant.contains_zero(), recorded_determinant


def interval_midpoint(value: Interval) -> Interval:
    return point((value.lower + value.upper) / 2)


def require_midpoint_value(reported: Interval, source: Interval, label: str) -> None:
    target = (source.lower + source.upper) / 2
    ulp = Fraction.from_float(math.ulp(float(target)))
    if not Interval(target - 4 * ulp, target + 4 * ulp).contains(reported):
        fail(f"midpoint reconstruction mismatch: {label}")


def verify_normalized_ray(
    record: Mapping[str, Interval | bool | str],
    source: list[list[Interval]],
    column: int,
    label: str,
) -> tuple[bool, str, Interval, list[Interval], Fraction | None]:
    x = source[0][column]
    y = source[1][column]
    candidates = [x, y, x + y, x - y]
    names = ["X", "Y", "PLUS", "MINUS"]
    norm_squared = [Fraction(1), Fraction(1), Fraction(2), Fraction(2)]
    for key, expected in zip(
        ("PIVOT_X", "PIVOT_Y", "PIVOT_PLUS", "PIVOT_MINUS"),
        candidates,
        strict=True,
    ):
        require_tight_contains(interval(record, key), expected, f"{label} {key}")

    scores: list[Fraction | None] = []
    for pivot, norm in zip(candidates, norm_squared, strict=True):
        if pivot.contains_zero():
            scores.append(None)
        else:
            margin = min(abs(pivot.lower), abs(pivot.upper))
            scores.append(margin * margin / norm)
    selected: int | None = None
    selected_score: Fraction | None = None
    for index, score in enumerate(scores):
        if score is not None and (selected_score is None or score > selected_score):
            selected = index
            selected_score = score

    eligible = selected is not None
    if boolean(record, "ELIGIBLE") != eligible:
        fail(f"{label} eligibility mismatch")
    expected_chart = names[selected] if selected is not None else "NONE"
    if string_value(record, "CHART") != expected_chart:
        fail(f"{label} chart selection mismatch")

    if not eligible:
        if any(interval(record, key) != ZERO for key in ("PIVOT", "SLOPE")):
            fail(f"{label} ineligible ray has a chart payload")
        if interval(record, "SCALE") != ONE:
            fail(f"{label} ineligible ray does not preserve unit scale")
        normalized = vector(record, "N")
        for row in range(3):
            require_tight_contains(normalized[row], source[row][column], f"{label} fallback N{row}")
        return False, "NONE", ONE, normalized, None

    assert selected is not None
    chosen = candidates[selected]
    require_tight_contains(interval(record, "PIVOT"), chosen, f"{label} chosen pivot")
    require_tight_contains(interval(record, "SCALE"), chosen, f"{label} scale")
    if expected_chart == "X":
        slope = y / x
        expected_normalized = [ONE, slope, ZERO]
    elif expected_chart == "Y":
        slope = x / y
        expected_normalized = [slope, ONE, ZERO]
    elif expected_chart == "PLUS":
        slope = (-x + y) / (x + y)
        expected_normalized = [(ONE - slope) / TWO, (ONE + slope) / TWO, ZERO]
    else:
        slope = (x + y) / (x - y)
        expected_normalized = [(ONE + slope) / TWO, (-ONE + slope) / TWO, ZERO]
    require_tight_contains(interval(record, "SLOPE"), slope, f"{label} slope")
    normalized = vector(record, "N")
    for row in range(3):
        require_tight_contains(
            normalized[row], expected_normalized[row], f"{label} normalized[{row}]"
        )
        reconstruction = interval(record, "SCALE") * normalized[row]
        require_contains(reconstruction, source[row][column], f"{label} ray reconstruction[{row}]")
    return True, expected_chart, interval(record, "SCALE"), normalized, selected_score


def verify_ledger(
    ledger: Ledger,
    expected_source: str,
    expected_input: LeafInput,
    expected_challenge: str,
) -> dict[str, bool]:
    for key, expected in (
        ("WORKER_SOURCE_SHA256", expected_source),
        ("INPUT_SHA256", expected_input.sha256),
        ("RUN_CHALLENGE", expected_challenge),
    ):
        if SHA_RE.fullmatch(ledger.headers[key]) is None or ledger.headers[key] != expected:
            fail(f"dynamic header mismatch: {key}")

    u_depth = canonical_header_int(ledger.headers, "U_DEPTH")
    u_index = canonical_header_int(ledger.headers, "U_INDEX")
    s_depth = canonical_header_int(ledger.headers, "S_DEPTH")
    s_index = canonical_header_int(ledger.headers, "S_INDEX")
    u_tiles = canonical_header_int(ledger.headers, "U_TILES")
    s_tiles = canonical_header_int(ledger.headers, "S_TILES")
    if u_depth > 30 or s_depth > 30:
        fail("dyadic leaf depth exceeds worker contract")
    if u_tiles != 1 << u_depth or s_tiles != 1 << s_depth:
        fail("dyadic tile count does not match depth")
    if not (0 <= u_index < u_tiles and 0 <= s_index < s_tiles):
        fail("dyadic leaf index out of range")
    if (u_depth, u_index, s_depth, s_index) != (
        expected_input.u_depth,
        expected_input.u_index,
        expected_input.s_depth,
        expected_input.s_index,
    ):
        fail("receipt leaf coordinates differ from canonical input")

    geometry = frozen_geometry()
    source = ledger.records["SOURCE_TILE"]
    source_u = interval(source, "U")
    source_s = interval(source, "S")
    logical_u, slack_u = frozen_tile(geometry["radius_u"], u_index, u_tiles)
    logical_s, slack_s = frozen_tile(geometry["radius_s"], s_index, s_tiles)
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
    ideal_delta_u = Interval(Fraction(-1, u_tiles), Fraction(1, u_tiles))
    ideal_delta_s = Interval(Fraction(-1, s_tiles), Fraction(1, s_tiles))
    for index, reported, ideal, tile_slack, radius in (
        (0, delta[0], ideal_delta_u, slack_u, geometry["radius_u"]),
        (1, delta[1], ideal_delta_s, slack_s, geometry["radius_s"]),
    ):
        if not reported.contains(ideal):
            fail(f"normalized tile misses its logical radius: DELTA{index}")
        normalized_slack = 4 * tile_slack / radius.lower
        allowed = Interval(
            ideal.lower - normalized_slack,
            ideal.upper + normalized_slack,
        )
        if not allowed.contains(reported):
            fail(f"normalized tile exceeds rounding budget: DELTA{index}")
    if delta[2] != ZERO:
        fail("normalized dummy coordinate is not exact zero")
    if (
        not delta[0].contains_zero()
        or not delta[1].contains_zero()
        or delta[0].width <= 0
        or delta[1].width <= 0
    ):
        fail("normalized tile is not centered with positive radius")
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

    full1 = ledger.records["C2_FULL_P1"]
    center1 = ledger.records["C2_CENTER_P1"]
    verify_c2(full1, "C2 full P1")
    verify_c2(center1, "C2 center P1")
    for index in range(3):
        require_overlap(
            interval(p1, f"X{index}"), interval(full1, f"X{index}"), "C1/C2 P1 state"
        )
    require_overlap(interval(p1, "TIME"), interval(full1, "TIME"), "C1/C2 P1 time")
    if not matrix_overlap(matrix(center1, "DP"), matrix(full1, "DP")):
        fail("P1 center/full DP mismatch")

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

    affine = verify_affine(ledger)
    affine_det = affine["det"]
    assert isinstance(affine_det, Interval)
    affine_hull_matrix = matrix(ledger.records["AFFINE_CARRIER"], "HULL")
    projective_x_eligible, projective_x, projective_x_det = verify_projective(
        ledger.records["PROJECTIVE_X"], affine_hull_matrix, "X", (1, 0, 0, 1, 1)
    )
    projective_y_eligible, projective_y, projective_y_det = verify_projective(
        ledger.records["PROJECTIVE_Y"], affine_hull_matrix, "Y", (0, 1, 1, 0, -1)
    )
    projective_plus_eligible, projective_plus, projective_plus_det = verify_projective(
        ledger.records["PROJECTIVE_PLUS"],
        affine_hull_matrix,
        "PLUS",
        (1, 1, -1, 1, 2),
    )
    projective_minus_eligible, projective_minus, projective_minus_det = verify_projective(
        ledger.records["PROJECTIVE_MINUS"],
        affine_hull_matrix,
        "MINUS",
        (1, -1, 1, 1, 2),
    )

    event1 = ledger.records["EVENT1_MEAN_VALUE"]
    if interval(event1, "TIME") != interval(full1, "TIME"):
        fail("event-1 carrier time differs from C2 P1 time")
    event1_center = vector(event1, "CENTER")
    center1_state = vector(center1, "X")
    for row in range(2):
        require_midpoint_value(event1_center[row], center1_state[row], f"event1 center[{row}]")
    if event1_center[2] != ZERO:
        fail("event-1 center is not on the coordinate section")
    event1_basis = matrix(event1, "BASIS")
    full1_dp = matrix(full1, "DP")
    for row in range(3):
        for column in range(3):
            require_midpoint_value(
                event1_basis[row][column], full1_dp[row][column],
                f"event1 basis[{row},{column}]",
            )
    event1_delta = vector(event1, "DELTA")
    if event1_delta != delta:
        fail("event-1 mean-value delta differs from source delta")
    derivative_radius = [
        [full1_dp[row][column] - event1_basis[row][column] for column in range(3)]
        for row in range(3)
    ]
    expected_residual = [
        center1_state[row] - event1_center[row]
        + matrix_vector(derivative_radius, delta)[row]
        for row in range(3)
    ]
    expected_residual[2] = ZERO
    event1_residual = vector(event1, "RESIDUAL")
    for row in range(3):
        require_tight_contains(
            event1_residual[row], expected_residual[row], f"event1 residual[{row}]", 8192
        )
    expected_hull = [
        event1_center[row]
        + matrix_vector(event1_basis, delta)[row]
        + event1_residual[row]
        for row in range(3)
    ]
    event1_hull = vector(event1, "HULL")
    for row in range(3):
        require_contains(event1_hull[row], expected_hull[row], f"event1 hull[{row}]")
    event1_mean_value_overlap = joint_vectors(
        [event1_hull, vector(full1, "X"), center1_state]
    )
    if not event1_mean_value_overlap:
        fail("event-1 mean-value carrier has no joint state intersection")

    event1_ray0 = verify_normalized_ray(
        ledger.records["HOMOGENEOUS_EVENT1_RAY0"], full1_dp, 0, "event1 ray0"
    )
    event1_ray1 = verify_normalized_ray(
        ledger.records["HOMOGENEOUS_EVENT1_RAY1"], full1_dp, 1, "event1 ray1"
    )
    event1_rays = [event1_ray0, event1_ray1]

    local = ledger.records["HOMOGENEOUS_LOCAL_P2"]
    local_time = interval(local, "TIME")
    local_duration = interval(local, "DURATION")
    require_tight_contains(
        local_duration, local_time - interval(event1, "TIME"), "local P2 duration"
    )
    if local_duration.lower <= 0:
        fail("local second return duration is not positive")
    local_state = vector(local, "X")
    local_flow = matrix(local, "FLOW")
    local_dp = matrix(local, "DP")
    local_nu = interval(local, "NU")
    require_tight_contains(
        local_nu, normal_velocity(local_state, geometry), "local P2 normal velocity"
    )
    if local_nu.lower <= 0:
        fail("local P2 section denominator crosses zero")
    local_field, _ = field_and_derivative(local_state)
    for column in range(3):
        local_dt = -local_flow[2][column] / local_nu
        for row in range(2):
            expected = local_flow[row][column] + local_field[row] * local_dt
            require_contains(local_dp[row][column], expected, f"local P2 DP[{row},{column}]")
        if local_dp[2][column] != ZERO or local_dp[column][2] != ZERO:
            fail("local P2 dummy tangent slot is not exact zero")
    post_time = interval(local, "POST_TIME")
    post_state = vector(local, "POST_X")
    event_order_certified = local_time.lower > interval(event1, "TIME").upper and post_time.lower > local_time.upper
    postsection_plus_side = post_state[2].lower > 0
    if not event_order_certified or not postsection_plus_side:
        fail("local P2 event order or Plus-side witness failed")

    reconstructed_dp = matrix(local, "RECON_DP")
    expected_reconstructed = [[ZERO for _ in range(3)] for _ in range(3)]
    for column, ray in enumerate(event1_rays):
        scale = ray[2]
        for row in range(3):
            expected_reconstructed[row][column] = local_dp[row][column] * scale
    require_matrix_reconstruction(
        reconstructed_dp, expected_reconstructed, "homogeneous cumulative DP"
    )
    reconstructed_det = determinant_xy(reconstructed_dp)
    require_contains(
        interval(local, "RECON_DET"), reconstructed_det, "homogeneous reconstructed determinant"
    )
    cumulative_matrix_overlap = matrix_overlap(reconstructed_dp, c1_dp) and matrix_overlap(
        reconstructed_dp, matrix(full, "DP")
    )
    if not cumulative_matrix_overlap:
        fail("homogeneous cumulative DP misses an independent P2 enclosure")

    event2_ray0 = verify_normalized_ray(
        ledger.records["HOMOGENEOUS_EVENT2_RAY0"], local_dp, 0, "event2 ray0"
    )
    event2_ray1 = verify_normalized_ray(
        ledger.records["HOMOGENEOUS_EVENT2_RAY1"], local_dp, 1, "event2 ray1"
    )
    event2_rays = [event2_ray0, event2_ray1]
    event2_normalized = [[ZERO for _ in range(3)] for _ in range(3)]
    for column, ray in enumerate(event2_rays):
        for row in range(3):
            event2_normalized[row][column] = ray[3][row]

    plucker = ledger.records["PLUCKER_COCYCLE"]
    expected_total_scale0 = event1_ray0[2] * event2_ray0[2]
    expected_total_scale1 = event1_ray1[2] * event2_ray1[2]
    require_tight_contains(
        interval(plucker, "TOTAL_SCALE0"), expected_total_scale0, "total scale 0"
    )
    require_tight_contains(
        interval(plucker, "TOTAL_SCALE1"), expected_total_scale1, "total scale 1"
    )
    expected_exterior = determinant_xy(event2_normalized)
    require_tight_contains(
        interval(plucker, "NORMALIZED_EXTERIOR"), expected_exterior,
        "normalized exterior",
    )
    expected_homogeneous_det = (
        interval(plucker, "TOTAL_SCALE0")
        * interval(plucker, "TOTAL_SCALE1")
        * interval(plucker, "NORMALIZED_EXTERIOR")
    )
    homogeneous_det = interval(plucker, "DET")
    require_tight_contains(
        homogeneous_det, expected_homogeneous_det, "homogeneous exterior determinant", 8192
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

    homogeneous_joint_overlap = joint_interval(
        [homogeneous_det, interval(local, "RECON_DET"), affine_det, liouville_det]
    )
    if not homogeneous_joint_overlap:
        fail("homogeneous, matrix, affine, and Liouville determinants lack a joint intersection")
    fixed_widths = [
        determinant.width
        for eligible, determinant in (
            (projective_x_eligible, projective_x_det),
            (projective_y_eligible, projective_y_det),
            (projective_plus_eligible, projective_plus_det),
            (projective_minus_eligible, projective_minus_det),
        )
        if eligible
    ]

    computed = {
        "ALL_FINITE": True,
        "P1_CENTER_FULL_DP_OVERLAP": matrix_overlap(
            matrix(center1, "DP"), matrix(full1, "DP")
        ),
        "CENTER_FULL_DP_OVERLAP": matrix_overlap(matrix(center, "DP"), matrix(full, "DP")),
        "C1_C2_DP_OVERLAP": matrix_overlap(c1_dp, matrix(full, "DP")),
        "EVENT1_MEAN_VALUE_OVERLAP": event1_mean_value_overlap,
        "EVENT1_RAY_RECONSTRUCTION": True,
        "EVENT2_RAY_RECONSTRUCTION": True,
        "CUMULATIVE_MATRIX_OVERLAP": cumulative_matrix_overlap,
        "EVENT_ORDER_CERTIFIED": event_order_certified,
        "POSTSECTION_PLUS_SIDE": postsection_plus_side,
        "EVENT_TRANSVERSALITY_CERTIFIED": (
            p1_nu.lower > 0
            and p2_nu.lower > 0
            and interval(full1, "NU").lower > 0
            and interval(center1, "NU").lower > 0
            and interval(full, "NU").lower > 0
            and interval(center, "NU").lower > 0
            and local_nu.lower > 0
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
        "PROJECTIVE_PLUS_ORIENTATION_CERTIFIED": projective_plus,
        "PROJECTIVE_MINUS_ORIENTATION_CERTIFIED": projective_minus,
        "ANY_PROJECTIVE_ORIENTATION_CERTIFIED": (
            projective_x or projective_y or projective_plus or projective_minus
        ),
        "AFFINE_STRICTLY_NARROWER_THAN_C1": affine_det.width < interval(p2, "DET").width,
        "AFFINE_STRICTLY_NARROWER_THAN_C2_HULL": affine_det.width < interval(full, "HULL_DET").width,
        "EVENT1_CHARTS_CERTIFIED": event1_ray0[0] and event1_ray1[0],
        "EVENT2_CHARTS_CERTIFIED": event2_ray0[0] and event2_ray1[0],
        "HOMOGENEOUS_ORIENTATION_CERTIFIED": not homogeneous_det.contains_zero(),
        "HOMOGENEOUS_LIOUVILLE_OVERLAP": homogeneous_det.overlaps(liouville_det),
        "HOMOGENEOUS_LIOUVILLE_SAME_SIGN": (
            (homogeneous_det.upper < 0 and liouville_det.upper < 0)
            or (homogeneous_det.lower > 0 and liouville_det.lower > 0)
        ),
        "HOMOGENEOUS_JOINT_OVERLAP": homogeneous_joint_overlap,
        "HOMOGENEOUS_STRICTLY_NARROWER_THAN_AFFINE": (
            homogeneous_det.width < affine_det.width
        ),
        "HOMOGENEOUS_STRICTLY_NARROWER_THAN_BEST_FIXED": (
            bool(fixed_widths) and homogeneous_det.width < min(fixed_widths)
        ),
    }
    computed["STRUCTURAL_PASS"] = all(
        computed[key]
        for key in (
            "ALL_FINITE",
            "P1_CENTER_FULL_DP_OVERLAP",
            "CENTER_FULL_DP_OVERLAP",
            "C1_C2_DP_OVERLAP",
            "EVENT_TRANSVERSALITY_CERTIFIED",
            "IMPACT_TIME_CROSSCHECK",
            "LIOUVILLE_ORIENTATION_CERTIFIED",
        )
    )
    computed["HOMOGENEOUS_COMPUTATION_VALID"] = all(
        computed[key]
        for key in (
            "STRUCTURAL_PASS",
            "EVENT1_MEAN_VALUE_OVERLAP",
            "EVENT1_RAY_RECONSTRUCTION",
            "EVENT2_RAY_RECONSTRUCTION",
            "CUMULATIVE_MATRIX_OVERLAP",
            "EVENT_ORDER_CERTIFIED",
            "POSTSECTION_PLUS_SIDE",
            "EVENT1_CHARTS_CERTIFIED",
            "EVENT2_CHARTS_CERTIFIED",
            "HOMOGENEOUS_JOINT_OVERLAP",
        )
    )
    affine_certificate_pass = (
        computed["STRUCTURAL_PASS"]
        and computed["AFFINE_ORIENTATION_CERTIFIED"]
        and computed["AFFINE_LIOUVILLE_OVERLAP"]
        and computed["AFFINE_LIOUVILLE_SAME_SIGN"]
    )
    projective_determinants = {
        "PROJECTIVE_X_CERTIFICATE_PASS": (projective_x, projective_x_det),
        "PROJECTIVE_Y_CERTIFICATE_PASS": (projective_y, projective_y_det),
        "PROJECTIVE_PLUS_CERTIFICATE_PASS": (projective_plus, projective_plus_det),
        "PROJECTIVE_MINUS_CERTIFICATE_PASS": (projective_minus, projective_minus_det),
    }
    certificate_methods: dict[str, bool] = {
        "AFFINE_CERTIFICATE_PASS": affine_certificate_pass,
    }
    for key, (orientation_certified, determinant) in projective_determinants.items():
        certificate_methods[key] = (
            computed["STRUCTURAL_PASS"]
            and orientation_certified
            and determinant.overlaps(affine_det)
            and determinant.overlaps(liouville_det)
            and determinant.upper < 0
            and liouville_det.upper < 0
        )
    certificate_methods["HOMOGENEOUS_CERTIFICATE_PASS"] = (
        computed["HOMOGENEOUS_COMPUTATION_VALID"]
        and computed["HOMOGENEOUS_ORIENTATION_CERTIFIED"]
        and homogeneous_det.upper < 0
        and computed["HOMOGENEOUS_LIOUVILLE_OVERLAP"]
        and computed["HOMOGENEOUS_LIOUVILLE_SAME_SIGN"]
    )
    computed["HOMOGENEOUS_CERTIFICATE_PASS"] = certificate_methods[
        "HOMOGENEOUS_CERTIFICATE_PASS"
    ]
    computed["CERTIFICATE_PASS"] = any(certificate_methods.values())
    computed["PROBE_PASS"] = computed["HOMOGENEOUS_COMPUTATION_VALID"]
    summary = ledger.records["SUMMARY"]
    for key in SUMMARY_KEYS:
        if boolean(summary, key) != computed[key]:
            fail(f"summary mismatch: {key}")
    leaf_result = ledger.records["LEAF_RESULT"]
    expected_leaf_result = {
        "TERMINAL_CERTIFIED": computed["CERTIFICATE_PASS"],
        "SUBDIVISION_REQUIRED": not computed["CERTIFICATE_PASS"],
        **certificate_methods,
    }
    for key, expected in expected_leaf_result.items():
        if boolean(leaf_result, key) != expected:
            fail(f"leaf result mismatch: {key}")
    return {**computed, **expected_leaf_result}


Mutation = tuple[str, Callable[[Ledger], None]]


def set_interval(ledger: Ledger, marker: str, key: str, value: Interval) -> None:
    current = interval(ledger.records[marker], key)
    ledger.records[marker][key] = value if current != value else (
        ONE if value != ONE else ZERO
    )


def mutation_suite() -> list[Mutation]:
    def swap_axes(ledger: Ledger) -> None:
        record = ledger.records["AFFINE_CARRIER"]
        if record["A000"] == record["A100"]:
            set_interval(ledger, "AFFINE_CARRIER", "A000", ONE)
        else:
            record["A000"], record["A100"] = record["A100"], record["A000"]

    def swap_delta_axes(ledger: Ledger) -> None:
        record = ledger.records["SOURCE_TILE"]
        if record["DELTA0"] == record["DELTA1"]:
            record["DELTA0"] = ONE
        else:
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
        def mutate(ledger: Ledger) -> None:
            replacement = "0" * 64 if ledger.headers[key] != "0" * 64 else "1" * 64
            ledger.headers[key] = replacement

        return mutate

    def integer_header_mutation(key: str) -> Callable[[Ledger], None]:
        return lambda ledger: ledger.headers.__setitem__(
            key, str(int(ledger.headers[key]) + 1)
        )

    def flip_record_bool(marker: str, key: str) -> Callable[[Ledger], None]:
        return lambda ledger: ledger.records[marker].__setitem__(
            key, not boolean(ledger.records[marker], key)
        )

    def change_chart(marker: str) -> Callable[[Ledger], None]:
        def mutate(ledger: Ledger) -> None:
            current = string_value(ledger.records[marker], "CHART")
            ledger.records[marker]["CHART"] = "Y" if current != "Y" else "X"

        return mutate

    def unsigned_scale(marker: str) -> Callable[[Ledger], None]:
        def mutate(ledger: Ledger) -> None:
            current = interval(ledger.records[marker], "SCALE")
            replacement = Interval(
                min(abs(current.lower), abs(current.upper)),
                max(abs(current.lower), abs(current.upper)),
            )
            set_interval(ledger, marker, "SCALE", replacement)

        return mutate

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
        ("projective_x_eligibility_flipped", flip_record_bool("PROJECTIVE_X", "ELIGIBLE")),
        ("projective_y_eligibility_flipped", flip_record_bool("PROJECTIVE_Y", "ELIGIBLE")),
        ("projective_plus_eligibility_flipped", flip_record_bool("PROJECTIVE_PLUS", "ELIGIBLE")),
        ("projective_minus_eligibility_flipped", flip_record_bool("PROJECTIVE_MINUS", "ELIGIBLE")),
        ("event1_mean_value_center_erased", lambda x: set_interval(x, "EVENT1_MEAN_VALUE", "CENTER0", ZERO)),
        ("event1_mean_value_basis_erased", lambda x: set_interval(x, "EVENT1_MEAN_VALUE", "BASIS00", ZERO)),
        ("event1_mean_value_residual_erased", lambda x: set_interval(x, "EVENT1_MEAN_VALUE", "RESIDUAL0", ZERO)),
        ("event1_ray0_chart_changed", change_chart("HOMOGENEOUS_EVENT1_RAY0")),
        ("event1_ray1_pivot_erased", lambda x: set_interval(x, "HOMOGENEOUS_EVENT1_RAY1", "PIVOT", ZERO)),
        ("event1_ray0_scale_unsigned", unsigned_scale("HOMOGENEOUS_EVENT1_RAY0")),
        ("event1_ray1_normalized_erased", lambda x: set_interval(x, "HOMOGENEOUS_EVENT1_RAY1", "N1", ZERO)),
        ("local_p2_time_erased", lambda x: set_interval(x, "HOMOGENEOUS_LOCAL_P2", "TIME", ZERO)),
        ("local_p2_flow_erased", lambda x: set_interval(x, "HOMOGENEOUS_LOCAL_P2", "FLOW00", ZERO)),
        ("local_p2_dp_erased", lambda x: set_interval(x, "HOMOGENEOUS_LOCAL_P2", "DP00", ZERO)),
        ("local_p2_reconstruction_erased", lambda x: set_interval(x, "HOMOGENEOUS_LOCAL_P2", "RECON_DP00", ZERO)),
        ("local_p2_postside_erased", lambda x: set_interval(x, "HOMOGENEOUS_LOCAL_P2", "POST_X2", ZERO)),
        ("event2_ray0_chart_changed", change_chart("HOMOGENEOUS_EVENT2_RAY0")),
        ("event2_ray1_scale_reset", lambda x: set_interval(x, "HOMOGENEOUS_EVENT2_RAY1", "SCALE", ONE)),
        ("event2_ray0_slope_erased", lambda x: set_interval(x, "HOMOGENEOUS_EVENT2_RAY0", "SLOPE", ZERO)),
        ("plucker_scale0_reset", lambda x: set_interval(x, "PLUCKER_COCYCLE", "TOTAL_SCALE0", ONE)),
        ("plucker_exterior_erased", lambda x: set_interval(x, "PLUCKER_COCYCLE", "NORMALIZED_EXTERIOR", ZERO)),
        ("plucker_determinant_replaced_by_liouville", lambda x: set_interval(x, "PLUCKER_COCYCLE", "DET", interval(x.records["LIOUVILLE"], "DET"))),
        ("liouville_sign_flipped", lambda x: set_interval(x, "LIOUVILLE", "DET", ONE)),
        ("summary_claim_flipped", flip_record_bool("SUMMARY", "AFFINE_ORIENTATION_CERTIFIED")),
        ("leaf_terminal_claim_flipped", flip_record_bool("LEAF_RESULT", "TERMINAL_CERTIFIED")),
        ("leaf_subdivision_claim_flipped", flip_record_bool("LEAF_RESULT", "SUBDIVISION_REQUIRED")),
        ("leaf_affine_method_flipped", flip_record_bool("LEAF_RESULT", "AFFINE_CERTIFICATE_PASS")),
        ("leaf_projective_x_method_flipped", flip_record_bool("LEAF_RESULT", "PROJECTIVE_X_CERTIFICATE_PASS")),
        ("leaf_projective_y_method_flipped", flip_record_bool("LEAF_RESULT", "PROJECTIVE_Y_CERTIFICATE_PASS")),
        ("leaf_projective_plus_method_flipped", flip_record_bool("LEAF_RESULT", "PROJECTIVE_PLUS_CERTIFICATE_PASS")),
        ("leaf_projective_minus_method_flipped", flip_record_bool("LEAF_RESULT", "PROJECTIVE_MINUS_CERTIFICATE_PASS")),
        ("leaf_homogeneous_method_flipped", flip_record_bool("LEAF_RESULT", "HOMOGENEOUS_CERTIFICATE_PASS")),
        ("summary_homogeneous_valid_flipped", flip_record_bool("SUMMARY", "HOMOGENEOUS_COMPUTATION_VALID")),
        ("challenge_substituted", header_mutation("RUN_CHALLENGE")),
        ("source_hash_substituted", header_mutation("WORKER_SOURCE_SHA256")),
        ("input_hash_substituted", header_mutation("INPUT_SHA256")),
        ("u_depth_substituted", integer_header_mutation("U_DEPTH")),
        ("u_index_substituted", integer_header_mutation("U_INDEX")),
        ("s_depth_substituted", integer_header_mutation("S_DEPTH")),
        ("s_index_substituted", integer_header_mutation("S_INDEX")),
        ("u_tiles_substituted", integer_header_mutation("U_TILES")),
        ("s_tiles_substituted", integer_header_mutation("S_TILES")),
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
    ledger: Ledger,
    expected_source: str,
    expected_input: LeafInput,
    expected_challenge: str,
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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument("--require-probe", action="store_true")
    args = parser.parse_args(argv)
    for label, value in (
        ("source", args.source_sha),
        ("challenge", args.challenge),
    ):
        if SHA_RE.fullmatch(value) is None:
            fail(f"{label} digest must be lowercase SHA-256")

    leaf_input = read_leaf_input(args.input)
    ledger = parse_ledger(args.receipt)
    computed = verify_ledger(ledger, args.source_sha, leaf_input, args.challenge)
    total = rejected = 0
    if args.self_test_mutations:
        total, rejected = run_mutations(
            ledger, args.source_sha, leaf_input, args.challenge
        )
    method = "NONE"
    for candidate, key in (
        ("HOMOGENEOUS", "HOMOGENEOUS_CERTIFICATE_PASS"),
        ("AFFINE", "AFFINE_CERTIFICATE_PASS"),
        ("PROJECTIVE_X", "PROJECTIVE_X_CERTIFICATE_PASS"),
        ("PROJECTIVE_Y", "PROJECTIVE_Y_CERTIFICATE_PASS"),
        ("PROJECTIVE_PLUS", "PROJECTIVE_PLUS_CERTIFICATE_PASS"),
        ("PROJECTIVE_MINUS", "PROJECTIVE_MINUS_CERTIFICATE_PASS"),
    ):
        if computed[key]:
            method = candidate
            break
    print("VERIFICATION_SCHEMA=sounio.cs6.plucker-cocycle-leaf-verification.v1")
    print(f"RECEIPT_SHA256={ledger.receipt_sha256}")
    print(f"PHYSICAL_SHA256={ledger.physical_sha256}")
    print(f"MUTATION_TESTS={total}")
    print(f"MUTATIONS_REJECTED={rejected}")
    print(f"LEAF_METHOD={method}")
    print(f"PROBE_PASS={str(computed['PROBE_PASS']).lower()}")
    for key in (
        "AFFINE_CERTIFICATE_PASS",
        "PROJECTIVE_X_CERTIFICATE_PASS",
        "PROJECTIVE_Y_CERTIFICATE_PASS",
        "PROJECTIVE_PLUS_CERTIFICATE_PASS",
        "PROJECTIVE_MINUS_CERTIFICATE_PASS",
        "HOMOGENEOUS_CERTIFICATE_PASS",
    ):
        print(f"{key}={str(computed[key]).lower()}")
    print(f"SUBDIVISION_REQUIRED={str(computed['SUBDIVISION_REQUIRED']).lower()}")
    print(f"CERTIFICATE_PASS={str(computed['CERTIFICATE_PASS']).lower()}")
    if args.require_probe and not computed["PROBE_PASS"]:
        return 3
    return 2 if args.require_terminal and not computed["CERTIFICATE_PASS"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
