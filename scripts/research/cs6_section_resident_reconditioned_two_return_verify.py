#!/usr/bin/env python3
"""Exact verifier for the CS6 reconditioned two-return receipt."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


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

    def contains(self, other: "Interval") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def contains_zero(self) -> bool:
        return self.lower <= 0 <= self.upper

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lower - other.upper, self.upper - other.lower)

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

    def scale(self, factor: Fraction) -> "Interval":
        return self * Interval(factor, factor)


ZERO = Interval(Fraction(0), Fraction(0))
ONE = Interval(Fraction(1), Fraction(1))
TWO = Interval(Fraction(2), Fraction(2))
HEX_RE = re.compile(
    r"^-?0x(?:[0-9a-f]+(?:\.[0-9a-f]*)?|\.[0-9a-f]+)p[+-][0-9]+$"
)
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


FIXED_HEADERS = (
    ("SCHEMA", "sounio.cs6.section-resident-reconditioned-two-return.v1"),
    ("WORKER_SOURCE_SHA256", None),
    ("INPUT_SHA256", None),
    ("RUN_CHALLENGE", None),
    ("CAPD_SOURCE_TREE_DECLARED", "capd-5.3.0"),
    ("INTERVAL_BACKEND_DECLARED", "FILIB"),
    ("INTERVAL_SERIALIZATION", "ONE_ULP_OUTWARD_BINARY64_HEX"),
    ("DIRECT_FLOW_TANGENT_ROLE", "D_FLOW_TIMES_Q0"),
    ("LOCAL_P1_FLOW_TANGENT_ROLE", "D_FLOW_TIMES_Q0"),
    (
        "FLAT_LOCAL_P2_FLOW_TANGENT_ROLE",
        "D_FLOW_LOCAL_TIMES_SECTION_IDENTITY",
    ),
    ("GAUGE_FLOW_TANGENT_ROLE", "D_FLOW_LOCAL_TIMES_GAUGE_BASIS"),
    ("WIDTH_COMPARISON_FRAME", "FIXED_SOURCE_Q0_COORDINATES"),
    (
        "SOURCE_TANGENT_SEED_ROLE",
        "GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL",
    ),
    (
        "Q0_AREA_ROLE",
        "ORIENTED_XY_MINOR_OF_FIXED_GLOBAL_BASIS_NOT_TILE_AREA",
    ),
    (
        "TANGENT_ZERO_TIGHTENING",
        "COORDINATE_OUTPUT_ROW_AND_Q0_DUMMY_INPUT_COLUMN",
    ),
    ("SOURCE", "N0"),
    ("U_INDEX", "20000"),
    ("S_INDEX", "15000"),
    ("U_TILES", "40000"),
    ("S_TILES", "30000"),
    ("ORDER", "8"),
    ("CUMULATIVE_RETURN_COUNT", "2"),
    ("LOCAL_RETURN_COUNT", "1"),
    ("SECTION", "COORDINATE_W_EQUALS_ZERO"),
    ("CROSSING_DIRECTION", "MINUS_PLUS"),
    ("FAST_PATH_REQUIRED", "true"),
    ("EVENT1_CARRIER_ROLE", "TERMINAL_J1_IN_EVIDENCE"),
    (
        "CONTINUATION1_CARRIER_ROLE",
        "LOCAL_TANGENT_SEED_WITH_J1_METADATA",
    ),
    ("EVENT2_CARRIER_ROLE", "TERMINAL_J2_LOCAL_EVIDENCE"),
    (
        "CONTINUATION2_CARRIER_ROLE",
        "LOCAL_TANGENT_SEED_WITH_COMPOSED_METADATA",
    ),
    ("COMPOSITION_ORDER", "J2_LOCAL_TIMES_J1_IN"),
    (
        "GAUGE_COMPOSITION_ORDER",
        "J2_BASIS_TIMES_BASIS_INVERSE_TIMES_J1_IN",
    ),
    ("EVENT1_C0_REPRESENTATION", "MEAN_VALUE_DOUBLETON"),
    (
        "EVENT1_C0_FORM",
        "CENTER_PLUS_MID_J1_TIMES_NORMALIZED_DELTA_PLUS_RESIDUAL",
    ),
    (
        "EVENT1_C0_RESIDUAL_ROLE",
        "POINT_INTEGRATION_ERROR_PLUS_J1_RADIUS",
    ),
    ("EVENT1_AFFINE_BASIS", "MIDPOINT_J1"),
    ("TANGENT_GAUGES", "IDENTITY,MIDPOINT_M,ORIENTED_QR"),
    ("PRIMARY_TANGENT_RECONDITIONING", "ORIENTED_QR_OF_MIDPOINT_J1"),
    ("C0_FACTOR_REORGANIZATION", "DISABLED_TO_PRESERVE_SOURCE_R0"),
    ("FLATTENED_TWO_RETURN_CONTROL_RETAINED", "true"),
    (
        "FLATTENED_BASELINE_RECEIPT_SHA256",
        "14315dd35ada83d13bddaa1c653e0dea86a9da91379559e7f64d69b314077dba",
    ),
    (
        "FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256",
        "536dea89d9f841e0afedaaeb9ef116f5237fb7dd96f7774340850833b5f4b0b1",
    ),
    (
        "SCIENTIFIC_RESULT_CLASS",
        "CORRELATION_PRESERVED_ORIENTATION_UNRESOLVED",
    ),
    ("AUTONOMOUS_VECTOR_FIELD", "true"),
    ("EVENT_TIME_SENSITIVITY_PROPAGATED", "false"),
    ("NONAUTONOMOUS_GENERALIZATION_PROVED", "false"),
    ("INCOMING_DP_REINJECTED", "false"),
    ("POSTSECTION_STATE_REUSED", "false"),
    ("LIOUVILLE_REJECT_ONLY", "true"),
    ("EXECUTION_SCOPE", "BOUNDED_LOCAL_CAPD_CPU_PROBE"),
    ("EXECUTION_PROVENANCE_ATTESTED", "false"),
    ("INDEPENDENT_REPLAY_REQUIRED", "true"),
    ("PROMOTION_ELIGIBLE", "false"),
    ("FULL_SOURCE_CARRIER_PROVED", "false"),
    ("HYPERBOLICITY_PROVED", "false"),
    ("CHAOTIC_ATTRACTOR_PROVED", "false"),
)


def vector_keys(prefix: str, size: int = 3) -> list[str]:
    return [f"{prefix}{row}" for row in range(size)]


def matrix_keys(prefix: str) -> list[str]:
    return [f"{prefix}{row}{column}" for row in range(3) for column in range(3)]


C0_KEYS = (
    vector_keys("C0_X")
    + matrix_keys("C0_C")
    + vector_keys("C0_R0")
    + matrix_keys("C0_B")
    + vector_keys("C0_R")
)
C1_KEYS = (
    matrix_keys("C1_D")
    + matrix_keys("C1_C")
    + matrix_keys("C1_R0")
    + matrix_keys("C1_B")
    + matrix_keys("C1_R")
)
DIRECT_KEYS = (
    ["TIME"] + vector_keys("X") + matrix_keys("FLOW_TANGENT")
    + matrix_keys("DP") + matrix_keys("SECTION_DP") + ["NU", "DET"]
)
LOCAL_KEYS = (
    ["TIME"] + vector_keys("X") + matrix_keys("FLOW_TANGENT")
    + matrix_keys("DP") + vector_keys("SECTION_X")
    + matrix_keys("SECTION_DP") + ["NU", "DET"]
)
MIDPOINT_KEYS = ["TIME"] + vector_keys("X") + vector_keys("SECTION_X")
MEAN_VALUE_KEYS = (
    vector_keys("CENTER") + vector_keys("NORMALIZED_DELTA")
    + matrix_keys("M") + matrix_keys("RESIDUAL_BASIS")
    + vector_keys("CENTER_ERROR") + vector_keys("LINEARIZATION_ERROR")
    + vector_keys("RESIDUAL")
)
GAUGE_KEYS = (
    matrix_keys("BASIS") + matrix_keys("INVERSE_BASIS")
    + matrix_keys("TRANSITION") + matrix_keys("BASIS_TIMES_INVERSE")
    + matrix_keys("INVERSE_TIMES_BASIS")
    + matrix_keys("BASIS_TIMES_TRANSITION")
)
GAUGE_CONTINUATION_KEYS = (
    ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
    + matrix_keys("C1_HULL") + matrix_keys("INCOMING_J1")
)
GAUGE_LOCAL_KEYS = (
    ["TIME", "DURATION"] + vector_keys("X")
    + matrix_keys("FLOW_TANGENT") + matrix_keys("DP")
    + vector_keys("SECTION_X") + matrix_keys("SECTION_DP")
    + ["NU", "DET_IN_BASIS"]
)
GAUGE_COMPOSED_KEYS = (
    matrix_keys("J2_BASIS") + matrix_keys("TRANSITION")
    + matrix_keys("DP_FIXED_Q0") + ["DET_FIXED_Q0"]
)
POSTSECTION_KEYS = ["TIME"] + vector_keys("X") + ["SECTION_SIGN"]
SUMMARY_KEYS = [
    "ALL_FINITE",
    "P1_STATE_JOINT_OVERLAP",
    "P1_TIME_JOINT_OVERLAP",
    "P1_DP_JOINT_OVERLAP",
    "P2_STATE_JOINT_OVERLAP",
    "P2_TIME_JOINT_OVERLAP",
    "P2_DP_JOINT_OVERLAP",
    "P2_VELOCITY_JOINT_OVERLAP",
    "P2_DETERMINANT_JOINT_OVERLAP",
    "MEAN_VALUE_P1_STATE_JOINT_OVERLAP",
    "MEAN_VALUE_P1_DP_JOINT_OVERLAP",
    "IDENTITY_P2_JOINT_OVERLAP",
    "MIDPOINT_M_P2_JOINT_OVERLAP",
    "ORIENTED_QR_P2_JOINT_OVERLAP",
    "MEAN_VALUE_C0_SHARED",
    "MIDPOINT_M_INVERSE_CERTIFIED",
    "ORIENTED_QR_INVERSE_CERTIFIED",
    "TRANSITIONS_RECONSTRUCT_J1",
    "FLAT_DETERMINANT_CROSSES_ZERO",
    "IDENTITY_DETERMINANT_CROSSES_ZERO",
    "MIDPOINT_M_DETERMINANT_CROSSES_ZERO",
    "ORIENTED_QR_DETERMINANT_CROSSES_ZERO",
    "ANY_GAUGE_SIGN_DEFINITE",
    "LIOUVILLE_DETERMINANT_NEGATIVE",
    "IDENTITY_DETERMINANT_WIDTH_IMPROVED",
    "MIDPOINT_M_DETERMINANT_WIDTH_IMPROVED",
    "ORIENTED_QR_DETERMINANT_WIDTH_IMPROVED",
    "CORRELATED_STATE_COMPONENTWISE_NARROWER",
    "CARRIER1_C0_IDENTICAL",
    "CARRIER2_C0_IDENTICAL",
    "EVENT_SECTIONS_EXACT",
    "CONTINUATION_SEEDS_EXACT",
    "COMPOSITION_ORDER_DISCRIMINATED",
    "SECOND_EVENT_STRICTLY_LATER",
    "POSTSECTIONS_STRICTLY_LATER",
    "POSTSECTIONS_PLUS",
    "CERTIFICATE_PASS",
    "PROBE_PASS",
]
GAUGES = ("IDENTITY", "MIDPOINT_M", "ORIENTED_QR")

RECORD_GRAMMAR = (
    ("SOURCE_TILE", ["SOURCE_U", "SOURCE_S"] + matrix_keys("Q0")),
    ("MIDPOINT_P1", MIDPOINT_KEYS),
    ("MEAN_VALUE_C0", MEAN_VALUE_KEYS),
    ("DIRECT_P1", DIRECT_KEYS),
    ("LOCAL_P1", LOCAL_KEYS),
    (
        "EVENT1_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + ["NU"],
    ),
    (
        "CONTINUATION1_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + matrix_keys("INCOMING_J1"),
    ),
    (
        "RECONDITIONED_EVENT1_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL"),
    ),
    ("GAUGE_IDENTITY", GAUGE_KEYS),
    ("GAUGE_MIDPOINT_M", GAUGE_KEYS),
    ("GAUGE_ORIENTED_QR", GAUGE_KEYS),
    ("GAUGE_IDENTITY_CONTINUATION1", GAUGE_CONTINUATION_KEYS),
    ("GAUGE_MIDPOINT_M_CONTINUATION1", GAUGE_CONTINUATION_KEYS),
    ("GAUGE_ORIENTED_QR_CONTINUATION1", GAUGE_CONTINUATION_KEYS),
    ("DIRECT_P2", DIRECT_KEYS),
    (
        "LOCAL_P2",
        ["TIME", "DURATION"] + vector_keys("X")
        + matrix_keys("FLOW_TANGENT") + matrix_keys("DP")
        + vector_keys("SECTION_X") + matrix_keys("SECTION_DP")
        + ["NU", "DET"],
    ),
    (
        "COMPOSED_P2",
        matrix_keys("J1") + matrix_keys("J2_LOCAL") + matrix_keys("DP")
        + matrix_keys("REVERSED_DP") + ["DET"],
    ),
    ("GAUGE_IDENTITY_LOCAL_P2", GAUGE_LOCAL_KEYS),
    ("GAUGE_MIDPOINT_M_LOCAL_P2", GAUGE_LOCAL_KEYS),
    ("GAUGE_ORIENTED_QR_LOCAL_P2", GAUGE_LOCAL_KEYS),
    ("GAUGE_IDENTITY_COMPOSED_P2", GAUGE_COMPOSED_KEYS),
    ("GAUGE_MIDPOINT_M_COMPOSED_P2", GAUGE_COMPOSED_KEYS),
    ("GAUGE_ORIENTED_QR_COMPOSED_P2", GAUGE_COMPOSED_KEYS),
    (
        "EVENT2_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + ["NU"],
    ),
    (
        "CONTINUATION2_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + matrix_keys("INCOMING_J2_LOCAL")
        + matrix_keys("INCOMING_COMPOSED_P2"),
    ),
    ("POSTSECTION1", POSTSECTION_KEYS),
    ("POSTSECTION2", POSTSECTION_KEYS),
    ("GAUGE_IDENTITY_POSTSECTION2", POSTSECTION_KEYS),
    ("GAUGE_MIDPOINT_M_POSTSECTION2", POSTSECTION_KEYS),
    ("GAUGE_ORIENTED_QR_POSTSECTION2", POSTSECTION_KEYS),
    (
        "LIOUVILLE_P2",
        ["TIME"] + vector_keys("X", 4)
        + ["NU0", "NU2", "ELL", "EXP_ELL", "DET"],
    ),
    ("SUMMARY", SUMMARY_KEYS),
)

FLATTENED_RECORD_MARKERS = (
    "SOURCE_TILE", "DIRECT_P1", "LOCAL_P1", "EVENT1_CARRIER",
    "CONTINUATION1_CARRIER", "DIRECT_P2", "LOCAL_P2", "COMPOSED_P2",
    "EVENT2_CARRIER", "CONTINUATION2_CARRIER", "POSTSECTION1",
    "POSTSECTION2", "LIOUVILLE_P2",
)
FLATTENED_RECORD_GRAMMAR = tuple(
    entry for marker in FLATTENED_RECORD_MARKERS
    for entry in RECORD_GRAMMAR if entry[0] == marker
)


def canonical_hex(value: float) -> str:
    mantissa, exponent = value.hex().lower().split("p", 1)
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}p{exponent}"


def parse_interval(text: str, context: str) -> Interval:
    match = INTERVAL_RE.fullmatch(text)
    if match is None:
        fail(f"{context}: malformed interval")
    lower_text, upper_text = match.groups()
    if HEX_RE.fullmatch(lower_text) is None or HEX_RE.fullmatch(upper_text) is None:
        fail(f"{context}: endpoint is not canonical lowercase binary64")
    lower_outer = float.fromhex(lower_text)
    upper_outer = float.fromhex(upper_text)
    if not math.isfinite(lower_outer) or not math.isfinite(upper_outer):
        fail(f"{context}: non-finite endpoint")
    if lower_outer > upper_outer:
        fail(f"{context}: inverted serialized interval")
    lower_inner = math.nextafter(lower_outer, math.inf)
    upper_inner = math.nextafter(upper_outer, -math.inf)
    if lower_inner > upper_inner:
        fail(f"{context}: not a one-ULP outward encoding")
    expected_lower = canonical_hex(math.nextafter(lower_inner, -math.inf))
    expected_upper = canonical_hex(math.nextafter(upper_inner, math.inf))
    if lower_text != expected_lower or upper_text != expected_upper:
        fail(f"{context}: noncanonical one-ULP serialization")
    return Interval(Fraction.from_float(lower_inner), Fraction.from_float(upper_inner))


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


def parse_record(line: str, marker: str, keys: Sequence[str]) -> dict[str, str]:
    tokens = line.split(" ")
    if not tokens or tokens[0] != marker or any(not token for token in tokens):
        fail(f"{marker}: malformed record marker or whitespace")
    if len(tokens) != len(keys) + 1:
        fail(f"{marker}: wrong token count")
    result: dict[str, str] = {}
    for token, expected_key in zip(tokens[1:], keys):
        if "=" not in token:
            fail(f"{marker}: token lacks equals sign")
        key, value = token.split("=", 1)
        if key != expected_key or not value:
            fail(f"{marker}: expected key {expected_key}, got {key}")
        result[key] = value
    return result


def parse_ledger(data: bytes) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        fail("ledger must end in LF and contain no CR or NUL")
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise VerificationError("ledger is not ASCII") from error
    lines = text.splitlines()
    expected_lines = len(FIXED_HEADERS) + len(RECORD_GRAMMAR)
    if len(lines) != expected_lines or any(not line for line in lines):
        fail(f"ledger must contain exactly {expected_lines} nonempty lines")
    headers: dict[str, str] = {}
    for index, (key, expected) in enumerate(FIXED_HEADERS):
        line = lines[index]
        if line.count("=") != 1:
            fail(f"header {key}: malformed line")
        actual_key, value = line.split("=", 1)
        if actual_key != key or not value:
            fail(f"header order mismatch: expected {key}")
        if expected is not None and value != expected:
            fail(f"header {key}: expected {expected!r}, got {value!r}")
        headers[key] = value
    records: dict[str, dict[str, str]] = {}
    offset = len(FIXED_HEADERS)
    for index, (marker, keys) in enumerate(RECORD_GRAMMAR):
        records[marker] = parse_record(lines[offset + index], marker, keys)
    return headers, records


def read_vector(record: Mapping[str, str], prefix: str, size: int = 3) -> list[Interval]:
    return [parse_interval(record[f"{prefix}{row}"], f"{prefix}{row}")
            for row in range(size)]


def read_matrix(record: Mapping[str, str], prefix: str) -> list[list[Interval]]:
    return [[parse_interval(record[f"{prefix}{row}{column}"],
                            f"{prefix}{row}{column}")
             for column in range(3)] for row in range(3)]


def matmul(left: Sequence[Sequence[Interval]],
           right: Sequence[Sequence[Interval]]) -> list[list[Interval]]:
    return [[sum_intervals(left[row][inner] * right[inner][column]
                           for inner in range(3))
             for column in range(3)] for row in range(3)]


def matvec(matrix: Sequence[Sequence[Interval]],
           vector: Sequence[Interval]) -> list[Interval]:
    return [sum_intervals(matrix[row][column] * vector[column]
                          for column in range(3)) for row in range(3)]


def sum_intervals(values: Iterable[Interval]) -> Interval:
    result = ZERO
    for value in values:
        result = result + value
    return result


def add_vectors(*vectors: Sequence[Interval]) -> list[Interval]:
    return [sum_intervals(vector[row] for vector in vectors) for row in range(3)]


def add_matrices(*matrices: Sequence[Sequence[Interval]]) -> list[list[Interval]]:
    return [[sum_intervals(matrix[row][column] for matrix in matrices)
             for column in range(3)] for row in range(3)]


def interval_vector_equal(left: Sequence[Interval], right: Sequence[Interval]) -> bool:
    return list(left) == list(right)


def interval_matrix_equal(left: Sequence[Sequence[Interval]],
                          right: Sequence[Sequence[Interval]]) -> bool:
    return all(left[row][column] == right[row][column]
               for row in range(3) for column in range(3))


def interval_matrix_contains(outer: Sequence[Sequence[Interval]],
                             inner: Sequence[Sequence[Interval]]) -> bool:
    return all(outer[row][column].contains(inner[row][column])
               for row in range(3) for column in range(3))


def joint_interval(values: Sequence[Interval]) -> bool:
    return bool(values) and max(value.lower for value in values) <= min(
        value.upper for value in values
    )


def joint_vectors(values: Sequence[Sequence[Interval]]) -> bool:
    return bool(values) and all(joint_interval([value[row] for value in values])
                                for row in range(len(values[0])))


def joint_matrices(values: Sequence[Sequence[Sequence[Interval]]]) -> bool:
    return bool(values) and all(
        joint_interval([value[row][column] for value in values])
        for row in range(3) for column in range(3)
    )


def require_positive(value: Interval, context: str) -> None:
    if value.lower <= 0:
        fail(f"{context}: interval is not strictly positive")


def require_tight_contains(reported: Interval, calculated: Interval,
                           context: str, max_ulps: int = 4096) -> None:
    if not reported.contains(calculated):
        fail(f"{context}: reported interval misses exact reconstruction")
    magnitude = max(abs(calculated.lower), abs(calculated.upper))
    ulp = Fraction.from_float(math.ulp(float(magnitude)))
    allowed = Interval(calculated.lower - max_ulps * ulp,
                       calculated.upper + max_ulps * ulp)
    if not allowed.contains(reported):
        fail(f"{context}: reported enclosure exceeds rounding budget")


def require_vector_reconstruction(reported: Sequence[Interval],
                                  calculated: Sequence[Interval],
                                  context: str) -> None:
    for row in range(3):
        require_tight_contains(reported[row], calculated[row], f"{context}[{row}]")


def require_matrix_reconstruction(reported: Sequence[Sequence[Interval]],
                                  calculated: Sequence[Sequence[Interval]],
                                  context: str) -> None:
    for row in range(3):
        for column in range(3):
            require_tight_contains(reported[row][column], calculated[row][column],
                                   f"{context}[{row},{column}]")


def require_product_contains(reported: Interval, calculated: Interval,
                             context: str) -> None:
    """Allow one calculated-width of cancellation slack, still fail closed."""
    if not reported.contains(calculated):
        fail(f"{context}: reported interval misses exact product")
    magnitude = max(abs(calculated.lower), abs(calculated.upper))
    ulp = Fraction.from_float(math.ulp(float(magnitude)))
    slack = calculated.width + 4096 * ulp
    if not Interval(calculated.lower - slack,
                    calculated.upper + slack).contains(reported):
        fail(f"{context}: reported product exceeds cancellation budget")


def require_matrix_product_reconstruction(
        reported: Sequence[Sequence[Interval]],
        calculated: Sequence[Sequence[Interval]], context: str,
) -> None:
    for row in range(3):
        for column in range(3):
            require_product_contains(
                reported[row][column], calculated[row][column],
                f"{context}[{row},{column}]",
            )


def carrier_components(record: Mapping[str, str]) -> tuple[
    list[Interval], list[list[Interval]], list[Interval], list[list[Interval]],
    list[Interval], list[list[Interval]], list[list[Interval]],
    list[list[Interval]], list[list[Interval]], list[list[Interval]],
]:
    return (
        read_vector(record, "C0_X"),
        read_matrix(record, "C0_C"),
        read_vector(record, "C0_R0"),
        read_matrix(record, "C0_B"),
        read_vector(record, "C0_R"),
        read_matrix(record, "C1_D"),
        read_matrix(record, "C1_C"),
        read_matrix(record, "C1_R0"),
        read_matrix(record, "C1_B"),
        read_matrix(record, "C1_R"),
    )


def reconstruct_carrier(record: Mapping[str, str], context: str) -> tuple[
    list[Interval], list[list[Interval]], tuple[object, ...]
]:
    components = carrier_components(record)
    x, c0_c, c0_r0, c0_b, c0_r, c1_d, c1_c, c1_r0, c1_b, c1_r = components
    calculated_c0 = add_vectors(x, matvec(c0_c, c0_r0), matvec(c0_b, c0_r))
    calculated_c1 = add_matrices(c1_d, matmul(c1_c, c1_r0), matmul(c1_b, c1_r))
    reported_c0 = read_vector(record, "C0_HULL")
    reported_c1 = read_matrix(record, "C1_HULL")
    require_vector_reconstruction(reported_c0, calculated_c0, f"{context} C0")
    require_matrix_reconstruction(reported_c1, calculated_c1, f"{context} C1")
    return reported_c0, reported_c1, components


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
    logical = Interval((left + step.scale(Fraction(index))).lower,
                       (left + step.scale(Fraction(index + 1))).upper)
    scale = max(abs(radius.lower), abs(radius.upper))
    slack = 32 * Fraction.from_float(math.ulp(float(scale)))
    return logical, slack


def require_tile(reported: Interval, logical: Interval, slack: Fraction,
                 context: str) -> None:
    if not reported.contains(logical):
        fail(f"{context}: indexed tile is not contained")
    if not Interval(logical.lower - slack, logical.upper + slack).contains(reported):
        fail(f"{context}: tile exceeds rounding budget")


def vector_field(image: Sequence[Interval], geometry: Mapping[str, Interval]) -> list[Interval]:
    x, y, w = image
    zs = geometry["zs"]
    return [
        TWO * y * y - x * y,
        x * y - y * (w + zs) / TWO,
        x * y - w - zs,
    ]


def normal_velocity(image: Sequence[Interval], geometry: Mapping[str, Interval]) -> Interval:
    return image[0] * image[1] - image[2] - geometry["zs"]


def poincare_projection(image: Sequence[Interval],
                        phi: Sequence[Sequence[Interval]],
                        geometry: Mapping[str, Interval]) -> list[list[Interval]]:
    field = vector_field(image, geometry)
    require_positive(field[2], "Poincare section normal velocity")
    return [[phi[row][column] - field[row] * phi[2][column] / field[2]
             for column in range(3)] for row in range(3)]


def determinant_xy(matrix: Sequence[Sequence[Interval]]) -> Interval:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def exp_enclosure_negative(value: Interval, terms: int = 192) -> Interval:
    """Enclose exp(value) with exact rational Taylor bounds for value < 0."""
    if value.upper >= 0:
        fail("EXP_ELL reconstruction requires a strictly negative exponent")

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


def physical_digest(
        records: Mapping[str, Mapping[str, str]],
        grammar: Sequence[tuple[str, Sequence[str]]] = RECORD_GRAMMAR[:-1],
) -> str:
    chunks: list[str] = []
    for marker, keys in grammar:
        record = records[marker]
        for key in keys:
            value = record[key]
            if value.startswith("["):
                interval = parse_interval(value, f"digest {marker} {key}")
                chunks.append(
                    f"{marker}.{key}={interval.lower.numerator}/{interval.lower.denominator},"
                    f"{interval.upper.numerator}/{interval.upper.denominator}"
                )
    return hashlib.sha256("\n".join(chunks).encode("ascii")).hexdigest()


def point_midpoint(value: Interval) -> Interval:
    midpoint = float(value.lower) + 0.5 * (float(value.upper) - float(value.lower))
    exact = Fraction.from_float(midpoint)
    return Interval(exact, exact)


def require_exact_point_midpoint(reported: Interval, source: Interval,
                                 context: str) -> None:
    expected = point_midpoint(source)
    if reported != expected:
        fail(f"{context}: not the exact binary64 midpoint")


def require_tangent_structure(matrix: Sequence[Sequence[Interval]],
                              context: str) -> None:
    for index in range(3):
        if matrix[2][index] != ZERO or matrix[index][2] != ZERO:
            fail(f"{context}: normal row/column is not exactly zero")


def bool_text(value: bool) -> str:
    return str(value).lower()


def verify_two_return(data: bytes, expected_source: str, expected_input: str,
                      expected_challenge: str,
                      expected_receipt: str,
                      expected_baseline_receipt: str,
                      expected_flattened_physical: str) -> dict[str, str]:
    for value, name in (
        (expected_source, "source hash"),
        (expected_input, "input hash"),
        (expected_challenge, "run challenge"),
        (expected_receipt, "receipt hash"),
        (expected_baseline_receipt, "baseline receipt hash"),
        (expected_flattened_physical, "flattened physical hash"),
    ):
        if SHA_RE.fullmatch(value) is None:
            fail(f"expected {name} is not lowercase SHA-256")
    if hashlib.sha256(data).hexdigest() != expected_receipt:
        fail("receipt SHA-256 mismatch")
    headers, records = parse_ledger(data)
    if headers["WORKER_SOURCE_SHA256"] != expected_source:
        fail("worker source binding mismatch")
    if headers["INPUT_SHA256"] != expected_input:
        fail("input binding mismatch")
    if headers["RUN_CHALLENGE"] != expected_challenge:
        fail("run challenge mismatch")
    if headers["FLATTENED_BASELINE_RECEIPT_SHA256"] != expected_baseline_receipt:
        fail("flattened baseline receipt binding mismatch")
    if (headers["FLATTENED_BASELINE_PHYSICAL_CHAIN_SHA256"]
            != expected_flattened_physical):
        fail("flattened physical-chain binding mismatch")

    geometry = frozen_geometry()
    source = records["SOURCE_TILE"]
    source_u = parse_interval(source["SOURCE_U"], "SOURCE_U")
    source_s = parse_interval(source["SOURCE_S"], "SOURCE_S")
    logical_u, slack_u = frozen_tile(geometry["radius_u"], 20000, 40000)
    logical_s, slack_s = frozen_tile(geometry["radius_s"], 15000, 30000)
    require_tile(source_u, logical_u, slack_u, "SOURCE_U")
    require_tile(source_s, logical_s, slack_s, "SOURCE_S")
    source_q0 = read_matrix(source, "Q0")
    expected_q0 = [[ZERO for _ in range(3)] for _ in range(3)]
    expected_q0[0][0] = geometry["unstable_x"] * geometry["radius_u"]
    expected_q0[1][0] = geometry["unstable_y"] * geometry["radius_u"]
    expected_q0[0][1] = geometry["stable_x"] * geometry["radius_s"]
    expected_q0[1][1] = geometry["stable_y"] * geometry["radius_s"]
    require_matrix_reconstruction(source_q0, expected_q0, "SOURCE Q0")
    if any(source_q0[row][2] != ZERO for row in range(3)):
        fail("SOURCE Q0 dummy normal column is not exactly zero")

    def checked_return(marker: str, *, local: bool,
                       det_key: str = "DET") -> dict[str, object]:
        record = records[marker]
        time = parse_interval(record["TIME"], f"{marker} TIME")
        require_positive(time, f"{marker} TIME")
        image = read_vector(record, "X")
        flow = read_matrix(record, "FLOW_TANGENT")
        dp = read_matrix(record, "DP")
        section_dp = read_matrix(record, "SECTION_DP")
        projection = poincare_projection(image, flow, geometry)
        if not interval_matrix_contains(dp, projection):
            fail(f"{marker}: DP misses independently projected flow tangent")
        expected_section_dp = [row[:] for row in dp]
        for index in range(3):
            if (not dp[2][index].contains_zero()
                    or not dp[index][2].contains_zero()):
                fail(f"{marker}: structural tangent zero precondition failed")
            expected_section_dp[2][index] = ZERO
            expected_section_dp[index][2] = ZERO
        if not interval_matrix_equal(section_dp, expected_section_dp):
            fail(f"{marker}: SECTION_DP is not the exact structural projection")
        section_x = [image[0], image[1], ZERO]
        if local:
            reported_section_x = read_vector(record, "SECTION_X")
            if not interval_vector_equal(reported_section_x, section_x):
                fail(f"{marker}: SECTION_X is not the coordinate-section image")
        nu = parse_interval(record["NU"], f"{marker} NU")
        require_tight_contains(nu, normal_velocity(image, geometry), f"{marker} NU")
        require_positive(nu, f"{marker} NU")
        det = parse_interval(record[det_key], f"{marker} {det_key}")
        calculated_det = determinant_xy(section_dp)
        if det_key == "DET_IN_BASIS":
            require_product_contains(
                det, calculated_det, f"{marker} {det_key}"
            )
        else:
            require_tight_contains(
                det, calculated_det, f"{marker} {det_key}", 8192
            )
        return {
            "time": time, "image": image, "flow": flow, "dp": dp,
            "section_x": section_x, "section_dp": section_dp,
            "projection": projection, "nu": nu, "det": det,
        }

    direct1 = checked_return("DIRECT_P1", local=False)
    local1 = checked_return("LOCAL_P1", local=True)
    direct2 = checked_return("DIRECT_P2", local=False)
    local2 = checked_return("LOCAL_P2", local=True)
    for key in ("time", "image", "flow", "dp", "nu", "det"):
        if direct1[key] != local1[key]:
            fail(f"P1 public and protected fast paths differ at {key}")

    seed = [[ONE if row == column and row < 2 else ZERO
             for column in range(3)] for row in range(3)]

    midpoint_record = records["MIDPOINT_P1"]
    midpoint_time = parse_interval(midpoint_record["TIME"], "MIDPOINT_P1 TIME")
    require_positive(midpoint_time, "MIDPOINT_P1 TIME")
    midpoint_image = read_vector(midpoint_record, "X")
    midpoint_section = read_vector(midpoint_record, "SECTION_X")
    if not interval_vector_equal(
            midpoint_section, [midpoint_image[0], midpoint_image[1], ZERO]):
        fail("MIDPOINT_P1 SECTION_X is not the coordinate-section image")
    if not joint_interval([midpoint_time, direct1["time"], local1["time"]]):
        fail("MIDPOINT_P1 time does not overlap the P1 enclosure")

    mean_record = records["MEAN_VALUE_C0"]
    mean_center = read_vector(mean_record, "CENTER")
    normalized_delta = read_vector(mean_record, "NORMALIZED_DELTA")
    mean_basis = read_matrix(mean_record, "M")
    residual_basis = read_matrix(mean_record, "RESIDUAL_BASIS")
    center_error = read_vector(mean_record, "CENTER_ERROR")
    linearization_error = read_vector(mean_record, "LINEARIZATION_ERROR")
    mean_residual = read_vector(mean_record, "RESIDUAL")
    for row in range(3):
        require_exact_point_midpoint(
            mean_center[row], midpoint_section[row], f"MEAN_VALUE_C0 CENTER{row}"
        )
    for row in range(3):
        for column in range(3):
            require_exact_point_midpoint(
                mean_basis[row][column], local1["section_dp"][row][column],
                f"MEAN_VALUE_C0 M{row}{column}",
            )
    if not interval_matrix_equal(residual_basis, [
            [ONE if row == column else ZERO for column in range(3)]
            for row in range(3)]):
        fail("MEAN_VALUE_C0 residual basis is not exact identity")
    expected_delta = [
        (source_u - point_midpoint(source_u)) / geometry["radius_u"],
        (source_s - point_midpoint(source_s)) / geometry["radius_s"],
        ZERO,
    ]
    require_vector_reconstruction(
        normalized_delta, expected_delta, "MEAN_VALUE_C0 normalized delta"
    )
    require_vector_reconstruction(
        center_error,
        [midpoint_section[row] - mean_center[row] for row in range(3)],
        "MEAN_VALUE_C0 center error",
    )
    require_vector_reconstruction(
        linearization_error,
        matvec(add_matrices(local1["section_dp"], [
            [mean_basis[row][column].scale(Fraction(-1))
             for column in range(3)] for row in range(3)
        ]), normalized_delta),
        "MEAN_VALUE_C0 linearization error",
    )
    require_vector_reconstruction(
        mean_residual, add_vectors(center_error, linearization_error),
        "MEAN_VALUE_C0 residual",
    )
    require_tangent_structure(mean_basis, "MEAN_VALUE_C0 M")

    reconditioned_event = records["RECONDITIONED_EVENT1_CARRIER"]
    mean_c0_hull, mean_c1_hull, mean_components = reconstruct_carrier(
        reconditioned_event, "RECONDITIONED_EVENT1_CARRIER"
    )
    expected_mean_components = (
        mean_center, mean_basis, normalized_delta, residual_basis, mean_residual
    )
    if mean_components[:5] != expected_mean_components:
        fail("RECONDITIONED_EVENT1_CARRIER raw C0 is not MEAN_VALUE_C0")
    if not interval_matrix_equal(mean_c1_hull, local1["section_dp"]):
        fail("RECONDITIONED_EVENT1_CARRIER C1 is not incoming J1")
    if parse_interval(reconditioned_event["TIME"], "reconditioned event time") \
            != local1["time"]:
        fail("RECONDITIONED_EVENT1_CARRIER clock is not the P1 clock")
    if mean_c0_hull[2] != ZERO:
        fail("RECONDITIONED_EVENT1_CARRIER is not resident on w=0")

    gauge_data: dict[str, dict[str, object]] = {}
    for gauge in GAUGES:
        record = records[f"GAUGE_{gauge}"]
        basis = read_matrix(record, "BASIS")
        inverse = read_matrix(record, "INVERSE_BASIS")
        transition = read_matrix(record, "TRANSITION")
        basis_inverse = read_matrix(record, "BASIS_TIMES_INVERSE")
        inverse_basis = read_matrix(record, "INVERSE_TIMES_BASIS")
        transition_image = read_matrix(record, "BASIS_TIMES_TRANSITION")
        require_tangent_structure(basis, f"GAUGE_{gauge} BASIS")
        require_tangent_structure(inverse, f"GAUGE_{gauge} INVERSE_BASIS")
        require_tangent_structure(transition, f"GAUGE_{gauge} TRANSITION")
        basis_det = determinant_xy(basis)
        if basis_det.contains_zero():
            fail(f"GAUGE_{gauge}: singular tangent basis")
        if gauge != "IDENTITY":
            expected_inverse = [[ZERO for _ in range(3)] for _ in range(3)]
            expected_inverse[0][0] = basis[1][1] / basis_det
            expected_inverse[0][1] = basis[0][1].scale(Fraction(-1)) / basis_det
            expected_inverse[1][0] = basis[1][0].scale(Fraction(-1)) / basis_det
            expected_inverse[1][1] = basis[0][0] / basis_det
            require_matrix_reconstruction(
                inverse, expected_inverse, f"GAUGE_{gauge} canonical inverse"
            )
        require_matrix_product_reconstruction(
            basis_inverse, matmul(basis, inverse),
            f"GAUGE_{gauge} basis*inverse",
        )
        require_matrix_product_reconstruction(
            inverse_basis, matmul(inverse, basis),
            f"GAUGE_{gauge} inverse*basis",
        )
        require_matrix_product_reconstruction(
            transition, matmul(inverse, local1["section_dp"]),
            f"GAUGE_{gauge} inverse*J1",
        )
        require_matrix_product_reconstruction(
            transition_image, matmul(basis, transition),
            f"GAUGE_{gauge} basis*transition",
        )
        if (not interval_matrix_contains(basis_inverse, seed)
                or not interval_matrix_contains(inverse_basis, seed)):
            fail(f"GAUGE_{gauge}: inverse products miss tangent identity")
        if not interval_matrix_contains(
                transition_image, local1["section_dp"]):
            fail(f"GAUGE_{gauge}: transition does not reconstruct J1")
        if gauge == "IDENTITY":
            if (not interval_matrix_equal(basis, seed)
                    or not interval_matrix_equal(inverse, seed)
                    or not interval_matrix_equal(transition,
                                                 local1["section_dp"])):
                fail("GAUGE_IDENTITY is not the exact tangent identity gauge")
        elif gauge == "MIDPOINT_M":
            if not interval_matrix_equal(basis, mean_basis):
                fail("GAUGE_MIDPOINT_M basis is not MEAN_VALUE_C0 M")
        else:
            if any(basis[row][column].width != 0
                   for row in range(2) for column in range(2)):
                fail("GAUGE_ORIENTED_QR basis is not point-valued")
            if (basis[0][1] != basis[1][0].scale(Fraction(-1))
                    or basis[1][1] != basis[0][0]):
                fail("GAUGE_ORIENTED_QR lacks the oriented planar form")
            qr_norm = basis[0][0] * basis[0][0] \
                + basis[1][0] * basis[1][0]
            unit_slack = Fraction(1, 1 << 50)
            if max(abs(qr_norm.lower - 1), abs(qr_norm.upper - 1)) > unit_slack:
                fail("GAUGE_ORIENTED_QR first column is not unit length")
            parallel = basis[0][0] * mean_basis[1][0] \
                - basis[1][0] * mean_basis[0][0]
            alignment_scale = max(
                abs((basis[0][0] * mean_basis[1][0]).lower),
                abs((basis[1][0] * mean_basis[0][0]).lower),
            )
            alignment_slack = 8 * Fraction.from_float(
                math.ulp(float(alignment_scale))
            )
            aligned = basis[0][0] * mean_basis[0][0] \
                + basis[1][0] * mean_basis[1][0]
            if (max(abs(parallel.lower), abs(parallel.upper))
                    > alignment_slack or aligned.lower <= 0):
                fail("GAUGE_ORIENTED_QR is not aligned with midpoint J1")

        continuation_record = records[f"GAUGE_{gauge}_CONTINUATION1"]
        continuation_c0, continuation_c1, continuation_components = (
            reconstruct_carrier(
                continuation_record, f"GAUGE_{gauge}_CONTINUATION1"
            )
        )
        if continuation_components[:5] != expected_mean_components:
            fail(f"GAUGE_{gauge}_CONTINUATION1 lost mean-value C0")
        if (not interval_vector_equal(continuation_c0, mean_c0_hull)
                or not interval_matrix_equal(continuation_c1, basis)):
            fail(f"GAUGE_{gauge}_CONTINUATION1 hull differs from gauge seed")
        if not interval_matrix_equal(
                read_matrix(continuation_record, "INCOMING_J1"),
                local1["section_dp"]):
            fail(f"GAUGE_{gauge}_CONTINUATION1 lost incoming J1")
        if parse_interval(
                continuation_record["TIME"], f"GAUGE_{gauge} TIME"
        ) != local1["time"]:
            fail(f"GAUGE_{gauge}_CONTINUATION1 clock differs from P1")
        gauge_data[gauge] = {
            "basis": basis,
            "inverse": inverse,
            "transition": transition,
            "continuation_c0": continuation_c0,
        }

    def checked_carrier(event_marker: str, continuation_marker: str,
                        local_result: Mapping[str, object],
                        incoming_key: str) -> tuple[
                            list[Interval], list[list[Interval]],
                            list[Interval], list[list[Interval]]]:
        event = records[event_marker]
        continuation = records[continuation_marker]
        event_c0, event_c1, event_components = reconstruct_carrier(
            event, event_marker
        )
        continuation_c0, continuation_c1, continuation_components = (
            reconstruct_carrier(continuation, continuation_marker)
        )
        if event_components[:5] != continuation_components[:5]:
            fail(f"{event_marker}/{continuation_marker}: raw C0 differs")
        expected_x = local_result["section_x"]
        expected_dp = local_result["section_dp"]
        if (not interval_vector_equal(event_c0, expected_x)
                or not interval_vector_equal(continuation_c0, expected_x)):
            fail(f"{event_marker}: C0 is not the local section image")
        if not interval_matrix_equal(event_c1, expected_dp):
            fail(f"{event_marker}: C1 is not the incoming local derivative")
        if not interval_matrix_equal(continuation_c1, seed):
            fail(f"{continuation_marker}: C1 is not the exact local seed")
        if not interval_matrix_equal(
                read_matrix(continuation, incoming_key), expected_dp):
            fail(f"{continuation_marker}: incoming derivative metadata differs")
        event_time = parse_interval(event["TIME"], f"{event_marker} TIME")
        continuation_time = parse_interval(
            continuation["TIME"], f"{continuation_marker} TIME"
        )
        if event_time != local_result["time"] or continuation_time != event_time:
            fail(f"{event_marker}: carrier clock is not the event clock")
        if event_c0[2] != ZERO or continuation_c0[2] != ZERO:
            fail(f"{event_marker}: carrier is not exactly resident on w=0")
        return event_c0, event_c1, continuation_c0, continuation_c1

    event1_c0, event1_c1, continuation1_c0, _ = checked_carrier(
        "EVENT1_CARRIER", "CONTINUATION1_CARRIER", local1, "INCOMING_J1"
    )
    event2_c0, event2_c1, continuation2_c0, _ = checked_carrier(
        "EVENT2_CARRIER", "CONTINUATION2_CARRIER", local2,
        "INCOMING_J2_LOCAL"
    )

    composed = records["COMPOSED_P2"]
    j1 = read_matrix(composed, "J1")
    j2_local = read_matrix(composed, "J2_LOCAL")
    composed_dp = read_matrix(composed, "DP")
    reversed_dp = read_matrix(composed, "REVERSED_DP")
    if not interval_matrix_equal(j1, local1["section_dp"]):
        fail("COMPOSED_P2 J1 differs from LOCAL_P1 SECTION_DP")
    if not interval_matrix_equal(j2_local, local2["section_dp"]):
        fail("COMPOSED_P2 J2_LOCAL differs from LOCAL_P2 SECTION_DP")
    require_matrix_reconstruction(
        composed_dp, matmul(j2_local, j1), "COMPOSED_P2 J2_LOCAL*J1"
    )
    require_matrix_reconstruction(
        reversed_dp, matmul(j1, j2_local), "COMPOSED_P2 reversed J1*J2_LOCAL"
    )
    if all(joint_interval([composed_dp[row][column],
                           reversed_dp[row][column]])
           for row in range(3) for column in range(3)):
        fail("composition order is not distinguished on this tile")
    composed_det = parse_interval(composed["DET"], "COMPOSED_P2 DET")
    require_tight_contains(
        composed_det, determinant_xy(composed_dp), "COMPOSED_P2 DET"
    )
    continuation2 = records["CONTINUATION2_CARRIER"]
    if not interval_matrix_equal(
            read_matrix(continuation2, "INCOMING_COMPOSED_P2"), composed_dp):
        fail("CONTINUATION2_CARRIER lost cumulative derivative metadata")

    for gauge in GAUGES:
        local_marker = f"GAUGE_{gauge}_LOCAL_P2"
        gauge_local = checked_return(
            local_marker, local=True, det_key="DET_IN_BASIS"
        )
        gauge_duration = parse_interval(
            records[local_marker]["DURATION"], f"{local_marker} DURATION"
        )
        require_tight_contains(
            gauge_duration, gauge_local["time"] - local1["time"],
            f"{local_marker} DURATION",
        )
        require_positive(gauge_duration, f"{local_marker} DURATION")
        if gauge_local["time"].lower <= local1["time"].upper:
            fail(f"{local_marker}: second event is not strictly later")

        composed_marker = f"GAUGE_{gauge}_COMPOSED_P2"
        gauge_composed_record = records[composed_marker]
        j2_basis = read_matrix(gauge_composed_record, "J2_BASIS")
        transition = read_matrix(gauge_composed_record, "TRANSITION")
        fixed_dp = read_matrix(gauge_composed_record, "DP_FIXED_Q0")
        fixed_det = parse_interval(
            gauge_composed_record["DET_FIXED_Q0"],
            f"{composed_marker} DET_FIXED_Q0",
        )
        if not interval_matrix_equal(j2_basis, gauge_local["section_dp"]):
            fail(f"{composed_marker}: J2_BASIS differs from local gauge DP")
        if not interval_matrix_equal(transition, gauge_data[gauge]["transition"]):
            fail(f"{composed_marker}: transition differs from gauge certificate")
        require_matrix_product_reconstruction(
            fixed_dp, matmul(j2_basis, transition),
            f"{composed_marker} J2_BASIS*TRANSITION",
        )
        require_product_contains(
            fixed_det, determinant_xy(fixed_dp),
            f"{composed_marker} DET_FIXED_Q0",
        )
        gauge_data[gauge].update({
            "local": gauge_local,
            "duration": gauge_duration,
            "fixed_dp": fixed_dp,
            "fixed_det": fixed_det,
        })

    gauge_images = [gauge_data[gauge]["local"]["image"] for gauge in GAUGES]
    gauge_times = [gauge_data[gauge]["local"]["time"] for gauge in GAUGES]
    if not all(interval_vector_equal(gauge_images[0], image)
               for image in gauge_images[1:]):
        fail("gauge-local C0 results differ despite shared mean-value carrier")
    if not all(gauge_times[0] == time for time in gauge_times[1:]):
        fail("gauge-local event clocks differ despite shared C0 carrier")

    duration2 = parse_interval(records["LOCAL_P2"]["DURATION"],
                               "LOCAL_P2 DURATION")
    require_tight_contains(
        duration2, local2["time"] - local1["time"], "LOCAL_P2 DURATION"
    )
    require_positive(duration2, "LOCAL_P2 DURATION")
    if (local2["time"].lower <= local1["time"].upper
            or direct2["time"].lower <= direct1["time"].upper):
        fail("second event is not strictly later than the first")

    def checked_post(marker: str, event_time: Interval) -> tuple[Interval, list[Interval]]:
        record = records[marker]
        time = parse_interval(record["TIME"], f"{marker} TIME")
        image = read_vector(record, "X")
        sign = parse_interval(record["SECTION_SIGN"], f"{marker} SECTION_SIGN")
        if time.lower <= event_time.upper:
            fail(f"{marker}: post-section clock is not strictly later")
        if sign.lower <= 0 or image[2] != sign:
            fail(f"{marker}: lacks strict MinusPlus behavioral witness")
        return time, image

    post1_time, _ = checked_post("POSTSECTION1", local1["time"])
    checked_post("POSTSECTION2", local2["time"])
    for gauge in GAUGES:
        post_time, post_image = checked_post(
            f"GAUGE_{gauge}_POSTSECTION2",
            gauge_data[gauge]["local"]["time"],
        )
        gauge_data[gauge].update({
            "post_time": post_time,
            "post_image": post_image,
        })
    if local2["time"].lower <= post1_time.upper:
        fail("second event does not occur after the first post-section state")

    liouville = records["LIOUVILLE_P2"]
    liouville_time = parse_interval(liouville["TIME"], "LIOUVILLE_P2 TIME")
    liouville_x = read_vector(liouville, "X", 4)
    liouville_nu0 = parse_interval(liouville["NU0"], "LIOUVILLE_P2 NU0")
    liouville_nu2 = parse_interval(liouville["NU2"], "LIOUVILLE_P2 NU2")
    ell = parse_interval(liouville["ELL"], "LIOUVILLE_P2 ELL")
    exp_ell = parse_interval(liouville["EXP_ELL"], "LIOUVILLE_P2 EXP_ELL")
    liouville_det = parse_interval(liouville["DET"], "LIOUVILLE_P2 DET")
    if ell != liouville_x[3]:
        fail("LIOUVILLE_P2 ELL differs from the integrated fourth state")
    independent_exp = exp_enclosure_negative(ell)
    require_tight_contains(exp_ell, independent_exp, "LIOUVILLE_P2 EXP_ELL", 8192)
    calculated_nu0_x = (
        geometry["origin_x"] + geometry["unstable_x"] * source_u
        + geometry["stable_x"] * source_s
    )
    calculated_nu0_y = (
        geometry["origin_y"] + geometry["unstable_y"] * source_u
        + geometry["stable_y"] * source_s
    )
    calculated_nu0 = calculated_nu0_x * calculated_nu0_y - geometry["zs"]
    calculated_nu2 = normal_velocity(liouville_x[:3], geometry)
    require_tight_contains(liouville_nu0, calculated_nu0, "LIOUVILLE_P2 NU0")
    require_tight_contains(liouville_nu2, calculated_nu2, "LIOUVILLE_P2 NU2")
    require_positive(liouville_nu0, "LIOUVILLE_P2 NU0")
    require_positive(liouville_nu2, "LIOUVILLE_P2 NU2")
    require_positive(exp_ell, "LIOUVILLE_P2 EXP_ELL")
    frame_det = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    oriented_q0_area = (
        frame_det * geometry["radius_u"] * geometry["radius_s"]
    )
    calculated_liouville_det = (
        exp_ell * liouville_nu0 / liouville_nu2 * oriented_q0_area
    )
    require_tight_contains(
        liouville_det, calculated_liouville_det, "LIOUVILLE_P2 DET"
    )
    liouville_determinant_negative = liouville_det.upper < 0
    if not liouville_determinant_negative:
        fail("LIOUVILLE_P2 determinant is not strictly negative")

    p1_state_joint = joint_vectors(
        [direct1["image"], local1["image"], event1_c0, continuation1_c0]
    )
    p1_time_joint = joint_interval([direct1["time"], local1["time"]])
    p1_dp_joint = joint_matrices(
        [direct1["section_dp"], local1["section_dp"], event1_c1]
    )
    p2_state_joint = joint_vectors(
        [direct2["image"], local2["image"], event2_c0,
         continuation2_c0, liouville_x[:3]]
    )
    p2_time_joint = joint_interval(
        [direct2["time"], local2["time"], liouville_time]
    )
    p2_dp_joint = joint_matrices([direct2["section_dp"], composed_dp])
    p2_velocity_joint = joint_interval(
        [direct2["nu"], local2["nu"],
         parse_interval(records["EVENT2_CARRIER"]["NU"],
                        "EVENT2_CARRIER NU"), liouville_nu2]
    )
    p2_det_joint = joint_interval(
        [direct2["det"], composed_det, liouville_det]
    )
    if not all((p1_state_joint, p1_time_joint, p1_dp_joint,
                p2_state_joint, p2_time_joint, p2_dp_joint,
                p2_velocity_joint, p2_det_joint)):
        fail("a required P1/P2 joint intersection is empty")

    for marker, expected_nu in (("EVENT1_CARRIER", event1_c0),
                                ("EVENT2_CARRIER", event2_c0)):
        reported_nu = parse_interval(records[marker]["NU"], f"{marker} NU")
        require_tight_contains(
            reported_nu, normal_velocity(expected_nu, geometry), f"{marker} NU"
        )
        require_positive(reported_nu, f"{marker} NU")

    mean_p1_state_joint = joint_vectors([
        direct1["image"], local1["image"], mean_c0_hull,
        *[gauge_data[gauge]["continuation_c0"] for gauge in GAUGES],
    ])
    mean_p1_dp_joint = joint_matrices([
        direct1["section_dp"], local1["section_dp"], mean_c1_hull,
    ])
    if not mean_p1_state_joint or not mean_p1_dp_joint:
        fail("mean-value P1 certificate has an empty joint intersection")

    gauge_joint: dict[str, bool] = {}
    gauge_crosses_zero: dict[str, bool] = {}
    gauge_width_improved: dict[str, bool] = {}
    for gauge in GAUGES:
        gauge_local = gauge_data[gauge]["local"]
        fixed_dp = gauge_data[gauge]["fixed_dp"]
        fixed_det = gauge_data[gauge]["fixed_det"]
        gauge_joint[gauge] = (
            joint_vectors([direct2["image"], gauge_local["image"],
                           liouville_x[:3]])
            and joint_interval([direct2["time"], gauge_local["time"],
                                liouville_time])
            and joint_matrices([direct2["section_dp"], fixed_dp])
            and joint_interval([direct2["nu"], gauge_local["nu"],
                                liouville_nu2])
            and joint_interval([direct2["det"], fixed_det, liouville_det])
        )
        gauge_crosses_zero[gauge] = fixed_det.contains_zero()
        gauge_width_improved[gauge] = fixed_det.width < composed_det.width
        if not gauge_joint[gauge]:
            fail(f"GAUGE_{gauge}: P2 joint intersection is empty")

    flat_crosses_zero = composed_det.contains_zero()
    any_gauge_sign_definite = any(
        not gauge_crosses_zero[gauge] for gauge in GAUGES
    )
    correlated_state_narrower = all(
        gauge_data[gauge]["local"]["image"][component].width
        < local2["image"][component].width
        for gauge in GAUGES for component in (0, 1)
    )
    if not flat_crosses_zero:
        fail("flattened determinant unexpectedly excludes zero")
    if any_gauge_sign_definite:
        fail("a gauge determinant unexpectedly resolves orientation")
    if not correlated_state_narrower:
        fail("correlated C0 state is not componentwise narrower")

    derived_summary = {
        "ALL_FINITE": True,
        "P1_STATE_JOINT_OVERLAP": p1_state_joint,
        "P1_TIME_JOINT_OVERLAP": p1_time_joint,
        "P1_DP_JOINT_OVERLAP": p1_dp_joint,
        "P2_STATE_JOINT_OVERLAP": p2_state_joint,
        "P2_TIME_JOINT_OVERLAP": p2_time_joint,
        "P2_DP_JOINT_OVERLAP": p2_dp_joint,
        "P2_VELOCITY_JOINT_OVERLAP": p2_velocity_joint,
        "P2_DETERMINANT_JOINT_OVERLAP": p2_det_joint,
        "MEAN_VALUE_P1_STATE_JOINT_OVERLAP": mean_p1_state_joint,
        "MEAN_VALUE_P1_DP_JOINT_OVERLAP": mean_p1_dp_joint,
        "IDENTITY_P2_JOINT_OVERLAP": gauge_joint["IDENTITY"],
        "MIDPOINT_M_P2_JOINT_OVERLAP": gauge_joint["MIDPOINT_M"],
        "ORIENTED_QR_P2_JOINT_OVERLAP": gauge_joint["ORIENTED_QR"],
        "MEAN_VALUE_C0_SHARED": all(
            interval_vector_equal(mean_c0_hull,
                                  gauge_data[gauge]["continuation_c0"])
            for gauge in GAUGES
        ),
        "MIDPOINT_M_INVERSE_CERTIFIED": True,
        "ORIENTED_QR_INVERSE_CERTIFIED": True,
        "TRANSITIONS_RECONSTRUCT_J1": True,
        "FLAT_DETERMINANT_CROSSES_ZERO": flat_crosses_zero,
        "IDENTITY_DETERMINANT_CROSSES_ZERO": gauge_crosses_zero["IDENTITY"],
        "MIDPOINT_M_DETERMINANT_CROSSES_ZERO": gauge_crosses_zero["MIDPOINT_M"],
        "ORIENTED_QR_DETERMINANT_CROSSES_ZERO": gauge_crosses_zero["ORIENTED_QR"],
        "ANY_GAUGE_SIGN_DEFINITE": any_gauge_sign_definite,
        "LIOUVILLE_DETERMINANT_NEGATIVE": liouville_determinant_negative,
        "IDENTITY_DETERMINANT_WIDTH_IMPROVED": gauge_width_improved["IDENTITY"],
        "MIDPOINT_M_DETERMINANT_WIDTH_IMPROVED": gauge_width_improved["MIDPOINT_M"],
        "ORIENTED_QR_DETERMINANT_WIDTH_IMPROVED": gauge_width_improved["ORIENTED_QR"],
        "CORRELATED_STATE_COMPONENTWISE_NARROWER": correlated_state_narrower,
        "CARRIER1_C0_IDENTICAL": True,
        "CARRIER2_C0_IDENTICAL": True,
        "EVENT_SECTIONS_EXACT": True,
        "CONTINUATION_SEEDS_EXACT": True,
        "COMPOSITION_ORDER_DISCRIMINATED": True,
        "SECOND_EVENT_STRICTLY_LATER": True,
        "POSTSECTIONS_STRICTLY_LATER": True,
        "POSTSECTIONS_PLUS": True,
    }
    certificate_keys = (
        set(derived_summary)
        - {
            "ANY_GAUGE_SIGN_DEFINITE",
            "IDENTITY_DETERMINANT_WIDTH_IMPROVED",
            "MIDPOINT_M_DETERMINANT_WIDTH_IMPROVED",
            "ORIENTED_QR_DETERMINANT_WIDTH_IMPROVED",
        }
    )
    certificate_pass = (
        all(derived_summary[key] for key in certificate_keys)
        and not any_gauge_sign_definite
    )
    derived_summary["CERTIFICATE_PASS"] = certificate_pass
    derived_summary["PROBE_PASS"] = certificate_pass

    summary = records["SUMMARY"]
    for key in SUMMARY_KEYS:
        expected = bool_text(derived_summary[key])
        if summary[key] not in ("true", "false"):
            fail(f"SUMMARY {key}: value is not canonical boolean")
        if summary[key] != expected:
            fail(f"SUMMARY {key}: reported {summary[key]}, derived {expected}")

    complete_digest = physical_digest(records)
    flattened_digest = physical_digest(records, FLATTENED_RECORD_GRAMMAR)
    if flattened_digest != expected_flattened_physical:
        fail("flattened physical-chain SHA-256 mismatch")
    comparison_payload = (
        f"BASELINE_RECEIPT_SHA256={expected_baseline_receipt}\n"
        f"FLATTENED_PHYSICAL_CHAIN_SHA256={flattened_digest}\n"
        f"PHYSICAL_CHAIN_SHA256={complete_digest}\n"
    ).encode("ascii")
    comparison_digest = hashlib.sha256(comparison_payload).hexdigest()
    return {
        "PHYSICAL_CHAIN_SHA256": complete_digest,
        "FLATTENED_PHYSICAL_CHAIN_SHA256": flattened_digest,
        "COMPARISON_CHAIN_SHA256": comparison_digest,
        "MEAN_VALUE_C0_RECOMPUTED": "true",
        "GAUGE_TRANSITIONS_RECOMPUTED": "true",
        "FIXED_FRAME_COMPOSITIONS_RECOMPUTED": "true",
        "CORRELATED_STATE_COMPONENTWISE_NARROWER": bool_text(
            correlated_state_narrower
        ),
        "IDENTITY_DETERMINANT_CROSSES_ZERO": bool_text(
            gauge_crosses_zero["IDENTITY"]
        ),
        "MIDPOINT_M_DETERMINANT_CROSSES_ZERO": bool_text(
            gauge_crosses_zero["MIDPOINT_M"]
        ),
        "ORIENTED_QR_DETERMINANT_CROSSES_ZERO": bool_text(
            gauge_crosses_zero["ORIENTED_QR"]
        ),
        "ANY_GAUGE_SIGN_DEFINITE": bool_text(any_gauge_sign_definite),
        "LIOUVILLE_DETERMINANT_NEGATIVE": bool_text(
            liouville_determinant_negative
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-run-challenge", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-baseline-receipt-sha256", required=True)
    parser.add_argument("--expected-flattened-physical-sha256", required=True)
    args = parser.parse_args()
    try:
        data = args.ledger.read_bytes()
        result = verify_two_return(
            data,
            args.expected_source_sha256,
            args.expected_input_sha256,
            args.expected_run_challenge,
            args.expected_receipt_sha256,
            args.expected_baseline_receipt_sha256,
            args.expected_flattened_physical_sha256,
        )
    except (OSError, VerificationError) as error:
        print(f"VERIFY_ERROR={error}", file=sys.stderr)
        return 2
    print("VERIFY_SCHEMA=sounio.cs6.section-resident-reconditioned-two-return-verification.v1")
    print("VERIFY_PASS=true")
    print("SOURCE=N0")
    print("TILE=20000,15000/40000,30000")
    print("RETURN_COUNT=2")
    print("RAW_C0_RECONSTRUCTED=true")
    print("RAW_C1_RECONSTRUCTED=true")
    print("POINCARE_DP_RECOMPUTED=true")
    print("POINCARE_DP_CONTAINS_RECOMPUTATION=true")
    print("COMPOSITION_EXACT_RECOMPUTED=true")
    print("REVERSED_ORDER_EXACT_RECOMPUTED=true")
    print("EXP_ELL_RECOMPUTED=true")
    for key, value in result.items():
        print(f"{key}={value}")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
