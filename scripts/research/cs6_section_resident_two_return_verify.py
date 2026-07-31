#!/usr/bin/env python3
"""Exact, fail-closed verifier for the CS6 section-resident carrier receipt."""

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
    ("SCHEMA", "sounio.cs6.section-resident-two-return.v1"),
    ("WORKER_SOURCE_SHA256", None),
    ("INPUT_SHA256", None),
    ("RUN_CHALLENGE", None),
    ("CAPD_SOURCE_TREE_DECLARED", "capd-5.3.0"),
    ("INTERVAL_BACKEND_DECLARED", "FILIB"),
    ("INTERVAL_SERIALIZATION", "ONE_ULP_OUTWARD_BINARY64_HEX"),
    ("DIRECT_FLOW_TANGENT_ROLE", "D_FLOW_TIMES_Q0"),
    ("LOCAL_FLOW_TANGENT_ROLE", "D_FLOW_LOCAL_TIMES_SECTION_IDENTITY"),
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
RECORD_GRAMMAR = (
    ("SOURCE_TILE", ["SOURCE_U", "SOURCE_S"] + matrix_keys("Q0")),
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
    ("POSTSECTION1", ["TIME"] + vector_keys("X") + ["SECTION_SIGN"]),
    ("POSTSECTION2", ["TIME"] + vector_keys("X") + ["SECTION_SIGN"]),
    (
        "LIOUVILLE_P2",
        ["TIME"] + vector_keys("X", 4)
        + ["NU0", "NU2", "ELL", "EXP_ELL", "DET"],
    ),
    (
        "SUMMARY",
        [
            "ALL_FINITE",
            "P1_STATE_JOINT_OVERLAP",
            "P1_TIME_JOINT_OVERLAP",
            "P1_DP_JOINT_OVERLAP",
            "P2_STATE_JOINT_OVERLAP",
            "P2_TIME_JOINT_OVERLAP",
            "P2_DP_JOINT_OVERLAP",
            "P2_VELOCITY_JOINT_OVERLAP",
            "P2_DETERMINANT_JOINT_OVERLAP",
            "CARRIER1_C0_IDENTICAL",
            "CARRIER2_C0_IDENTICAL",
            "EVENT_SECTIONS_EXACT",
            "CONTINUATION_SEEDS_EXACT",
            "COMPOSITION_ORDER_DISCRIMINATED",
            "SECOND_EVENT_STRICTLY_LATER",
            "POSTSECTIONS_STRICTLY_LATER",
            "POSTSECTIONS_PLUS",
            "PROBE_PASS",
        ],
    ),
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


def physical_digest(records: Mapping[str, Mapping[str, str]]) -> str:
    chunks: list[str] = []
    for marker, keys in RECORD_GRAMMAR[:-1]:
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


def verify_two_return(data: bytes, expected_source: str, expected_input: str,
                      expected_challenge: str,
                      expected_receipt: str) -> dict[str, str]:
    for value, name in (
        (expected_source, "source hash"),
        (expected_input, "input hash"),
        (expected_challenge, "run challenge"),
        (expected_receipt, "receipt hash"),
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

    def checked_return(marker: str, *, local: bool) -> dict[str, object]:
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
        det = parse_interval(record["DET"], f"{marker} DET")
        require_tight_contains(
            det, determinant_xy(section_dp), f"{marker} DET", 8192
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

    summary = records["SUMMARY"]
    if any(summary[key] != "true" for key in dict(RECORD_GRAMMAR)["SUMMARY"]):
        fail("worker summary does not report all bounded checks true")
    return {
        "PHYSICAL_CHAIN_SHA256": physical_digest(records),
        "P1_STATE_JOINT_OVERLAP": str(p1_state_joint).lower(),
        "P1_TIME_JOINT_OVERLAP": str(p1_time_joint).lower(),
        "P1_DP_JOINT_OVERLAP": str(p1_dp_joint).lower(),
        "P2_STATE_JOINT_OVERLAP": str(p2_state_joint).lower(),
        "P2_TIME_JOINT_OVERLAP": str(p2_time_joint).lower(),
        "P2_DP_JOINT_OVERLAP": str(p2_dp_joint).lower(),
        "P2_VELOCITY_JOINT_OVERLAP": str(p2_velocity_joint).lower(),
        "P2_DETERMINANT_JOINT_OVERLAP": str(p2_det_joint).lower(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-run-challenge", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()
    try:
        data = args.ledger.read_bytes()
        result = verify_two_return(
            data,
            args.expected_source_sha256,
            args.expected_input_sha256,
            args.expected_run_challenge,
            args.expected_receipt_sha256,
        )
    except (OSError, VerificationError) as error:
        print(f"VERIFY_ERROR={error}", file=sys.stderr)
        return 2
    print("VERIFY_SCHEMA=sounio.cs6.section-resident-two-return-verification.v1")
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
