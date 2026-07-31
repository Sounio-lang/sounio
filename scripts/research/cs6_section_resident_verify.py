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
    ("SCHEMA", "sounio.cs6.section-resident-carrier.v1"),
    ("WORKER_SOURCE_SHA256", None),
    ("INPUT_SHA256", None),
    ("RUN_CHALLENGE", None),
    ("CAPD_SOURCE_TREE_DECLARED", "capd-5.3.0"),
    ("INTERVAL_BACKEND_DECLARED", "FILIB"),
    ("INTERVAL_SERIALIZATION", "ONE_ULP_OUTWARD_BINARY64_HEX"),
    ("FLOW_TANGENT_ROLE", "D_FLOW_TIMES_Q0"),
    (
        "SOURCE_TANGENT_SEED_ROLE",
        "GLOBAL_FRAME_RADII_WITH_ZERO_DUMMY_NORMAL",
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
    ("RETURN_COUNT", "1"),
    ("SECTION", "COORDINATE_W_EQUALS_ZERO"),
    ("CROSSING_DIRECTION", "MINUS_PLUS"),
    ("FAST_PATH_REQUIRED", "true"),
    ("EVENT_CARRIER_ROLE", "TERMINAL_INCOMING_DP_EVIDENCE"),
    ("CONTINUATION_CARRIER_ROLE", "LOCAL_TANGENT_SEED_ONLY"),
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
RECORD_GRAMMAR = (
    ("SOURCE_TILE", ["SOURCE_U", "SOURCE_S"] + matrix_keys("Q0")),
    (
        "DIRECT",
        ["TIME"]
        + vector_keys("X")
        + matrix_keys("FLOW_TANGENT")
        + matrix_keys("DP")
        + ["NU", "DET"],
    ),
    (
        "CANDIDATE",
        ["TIME"]
        + vector_keys("X")
        + matrix_keys("FLOW_TANGENT")
        + matrix_keys("DP")
        + vector_keys("SECTION_X")
        + matrix_keys("SECTION_DP")
        + ["NU", "DET"],
    ),
    (
        "EVENT_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + ["NU"],
    ),
    (
        "CONTINUATION_CARRIER",
        ["TIME"] + C0_KEYS + C1_KEYS + vector_keys("C0_HULL")
        + matrix_keys("C1_HULL") + matrix_keys("INCOMING_DP"),
    ),
    ("POSTSECTION", ["TIME"] + vector_keys("X") + ["SECTION_SIGN"]),
    (
        "LIOUVILLE",
        ["TIME"] + vector_keys("X", 4)
        + ["NU0", "NU1", "ELL", "EXP_ELL", "DET"],
    ),
    (
        "SUMMARY",
        [
            "ALL_FINITE",
            "STATE_JOINT_OVERLAP",
            "TIME_JOINT_OVERLAP",
            "DP_JOINT_OVERLAP",
            "VELOCITY_JOINT_OVERLAP",
            "DETERMINANT_JOINT_OVERLAP",
            "CARRIER_C0_IDENTICAL",
            "EVENT_SECTION_EXACT",
            "CONTINUATION_SEED_EXACT",
            "POSTSECTION_STRICTLY_LATER",
            "POSTSECTION_PLUS",
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


def verify(data: bytes, expected_source: str, expected_input: str,
           expected_challenge: str, expected_receipt: str) -> dict[str, str]:
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

    direct = records["DIRECT"]
    candidate = records["CANDIDATE"]
    direct_time = parse_interval(direct["TIME"], "DIRECT TIME")
    candidate_time = parse_interval(candidate["TIME"], "CANDIDATE TIME")
    direct_x = read_vector(direct, "X")
    candidate_x = read_vector(candidate, "X")
    direct_flow_tangent = read_matrix(direct, "FLOW_TANGENT")
    candidate_flow_tangent = read_matrix(candidate, "FLOW_TANGENT")
    direct_dp = read_matrix(direct, "DP")
    candidate_dp = read_matrix(candidate, "DP")
    if not (
        direct_time == candidate_time
        and interval_vector_equal(direct_x, candidate_x)
        and interval_matrix_equal(direct_flow_tangent, candidate_flow_tangent)
        and interval_matrix_equal(direct_dp, candidate_dp)
        and direct["NU"] == candidate["NU"]
        and direct["DET"] == candidate["DET"]
    ):
        fail("direct and adapter candidate are not the same fast-path enclosure")
    require_positive(direct_time, "DIRECT TIME")
    projection = poincare_projection(candidate_x, candidate_flow_tangent, geometry)
    if not interval_matrix_contains(candidate_dp, projection):
        fail("candidate DP does not contain independently projected flow tangent")
    if not interval_matrix_contains(direct_dp, projection):
        fail("public DP does not contain independently projected flow tangent")

    section_x = read_vector(candidate, "SECTION_X")
    section_dp = read_matrix(candidate, "SECTION_DP")
    expected_section_x = [candidate_x[0], candidate_x[1], ZERO]
    expected_section_dp = [[candidate_dp[row][column] for column in range(3)]
                           for row in range(3)]
    for index in range(3):
        if not candidate_dp[2][index].contains_zero() or not candidate_dp[index][2].contains_zero():
            fail("section tangent projection precondition failed")
        expected_section_dp[2][index] = ZERO
        expected_section_dp[index][2] = ZERO
    if not interval_vector_equal(section_x, expected_section_x):
        fail("candidate section state is not the coordinate-section intersection")
    if not interval_matrix_equal(section_dp, expected_section_dp):
        fail("candidate section DP is not the exact tangent projection")

    event = records["EVENT_CARRIER"]
    continuation = records["CONTINUATION_CARRIER"]
    event_c0, event_c1, event_components = reconstruct_carrier(event, "EVENT_CARRIER")
    continuation_c0, continuation_c1, continuation_components = reconstruct_carrier(
        continuation, "CONTINUATION_CARRIER"
    )
    if event_components[:5] != continuation_components[:5]:
        fail("event and continuation carriers do not share identical raw C0")
    if not interval_vector_equal(event_c0, section_x):
        fail("event carrier C0 does not equal projected event state")
    if not interval_vector_equal(continuation_c0, section_x):
        fail("continuation carrier C0 does not equal projected event state")
    if not interval_matrix_equal(event_c1, section_dp):
        fail("event carrier C1 does not equal projected incoming DP")
    seed = [[ONE if row == column and row < 2 else ZERO
             for column in range(3)] for row in range(3)]
    if not interval_matrix_equal(continuation_c1, seed):
        fail("continuation carrier is not the exact tangent seed")
    incoming_dp = read_matrix(continuation, "INCOMING_DP")
    if not interval_matrix_equal(incoming_dp, section_dp):
        fail("continuation metadata lost the incoming projected DP")
    event_time = parse_interval(event["TIME"], "EVENT_CARRIER TIME")
    continuation_time = parse_interval(continuation["TIME"], "CONTINUATION_CARRIER TIME")
    if event_time != candidate_time or continuation_time != candidate_time:
        fail("carrier current time is not the event time")
    if event_c0[2] != ZERO or continuation_c0[2] != ZERO:
        fail("carrier is not exactly resident on w=0")

    post = records["POSTSECTION"]
    post_time = parse_interval(post["TIME"], "POSTSECTION TIME")
    post_x = read_vector(post, "X")
    post_sign = parse_interval(post["SECTION_SIGN"], "POSTSECTION SECTION_SIGN")
    if post_time.lower <= candidate_time.upper:
        fail("post-section set is not strictly later than the event")
    if post_sign.lower <= 0 or post_x[2] != post_sign:
        fail("post-section set lacks strict MinusPlus behavioral witness")

    liouville = records["LIOUVILLE"]
    liouville_time = parse_interval(liouville["TIME"], "LIOUVILLE TIME")
    liouville_x = read_vector(liouville, "X", 4)
    liouville_nu0 = parse_interval(liouville["NU0"], "LIOUVILLE NU0")
    liouville_nu1 = parse_interval(liouville["NU1"], "LIOUVILLE NU1")
    ell = parse_interval(liouville["ELL"], "LIOUVILLE ELL")
    exp_ell = parse_interval(liouville["EXP_ELL"], "LIOUVILLE EXP_ELL")
    liouville_det = parse_interval(liouville["DET"], "LIOUVILLE DET")
    if ell != liouville_x[3]:
        fail("Liouville ELL differs from the integrated fourth state")
    require_positive(exp_ell, "LIOUVILLE EXP_ELL")
    calculated_nu0_x = (
        geometry["origin_x"] + geometry["unstable_x"] * source_u
        + geometry["stable_x"] * source_s
    )
    calculated_nu0_y = (
        geometry["origin_y"] + geometry["unstable_y"] * source_u
        + geometry["stable_y"] * source_s
    )
    calculated_nu0 = calculated_nu0_x * calculated_nu0_y - geometry["zs"]
    calculated_nu1 = normal_velocity(liouville_x[:3], geometry)
    require_tight_contains(liouville_nu0, calculated_nu0, "LIOUVILLE NU0")
    require_tight_contains(liouville_nu1, calculated_nu1, "LIOUVILLE NU1")
    require_positive(liouville_nu0, "LIOUVILLE NU0")
    require_positive(liouville_nu1, "LIOUVILLE NU1")
    frame_det = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    source_area = frame_det * geometry["radius_u"] * geometry["radius_s"]
    calculated_liouville_det = exp_ell * liouville_nu0 / liouville_nu1 * source_area
    require_tight_contains(liouville_det, calculated_liouville_det, "LIOUVILLE DET")

    direct_nu = parse_interval(direct["NU"], "DIRECT NU")
    candidate_nu = parse_interval(candidate["NU"], "CANDIDATE NU")
    event_nu = parse_interval(event["NU"], "EVENT_CARRIER NU")
    require_tight_contains(direct_nu, normal_velocity(direct_x, geometry), "DIRECT NU")
    require_tight_contains(candidate_nu, normal_velocity(candidate_x, geometry),
                           "CANDIDATE NU")
    require_tight_contains(event_nu, normal_velocity(event_c0, geometry),
                           "EVENT_CARRIER NU")
    for value, context in ((direct_nu, "DIRECT NU"),
                           (candidate_nu, "CANDIDATE NU"),
                           (event_nu, "EVENT_CARRIER NU")):
        require_positive(value, context)
    direct_det = parse_interval(direct["DET"], "DIRECT DET")
    candidate_det = parse_interval(candidate["DET"], "CANDIDATE DET")
    require_tight_contains(direct_det, determinant_xy(direct_dp), "DIRECT DET")
    require_tight_contains(candidate_det, determinant_xy(section_dp), "CANDIDATE DET")

    state_joint = joint_vectors(
        [direct_x, candidate_x, event_c0, continuation_c0, liouville_x[:3]]
    )
    time_joint = joint_interval(
        [direct_time, candidate_time, event_time, continuation_time, liouville_time]
    )
    dp_joint = joint_matrices([direct_dp, candidate_dp, event_c1, section_dp, projection])
    velocity_joint = joint_interval(
        [direct_nu, candidate_nu, event_nu, liouville_nu1]
    )
    determinant_joint = joint_interval(
        [direct_det, candidate_det, liouville_det]
    )
    if not all((state_joint, time_joint, dp_joint, velocity_joint, determinant_joint)):
        fail("direct/candidate/carrier/Liouville joint intersection is empty")

    summary = records["SUMMARY"]
    if any(summary[key] != "true" for key in dict(RECORD_GRAMMAR)["SUMMARY"]):
        fail("worker summary does not report all bounded checks true")
    return {
        "PHYSICAL_CHAIN_SHA256": physical_digest(records),
        "STATE_JOINT_OVERLAP": str(state_joint).lower(),
        "TIME_JOINT_OVERLAP": str(time_joint).lower(),
        "DP_JOINT_OVERLAP": str(dp_joint).lower(),
        "VELOCITY_JOINT_OVERLAP": str(velocity_joint).lower(),
        "DETERMINANT_JOINT_OVERLAP": str(determinant_joint).lower(),
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
        result = verify(
            data,
            args.expected_source_sha256,
            args.expected_input_sha256,
            args.expected_run_challenge,
            args.expected_receipt_sha256,
        )
    except (OSError, VerificationError) as error:
        print(f"VERIFY_ERROR={error}", file=sys.stderr)
        return 2
    print("VERIFY_SCHEMA=sounio.cs6.section-resident-carrier-verification.v1")
    print("VERIFY_PASS=true")
    print("SOURCE=N0")
    print("TILE=20000,15000/40000,30000")
    print("RETURN_COUNT=1")
    print("RAW_C0_RECONSTRUCTED=true")
    print("RAW_C1_RECONSTRUCTED=true")
    print("POINCARE_DP_RECOMPUTED=true")
    print("POINCARE_DP_CONTAINS_RECOMPUTATION=true")
    print("EXP_ELL_RECOMPUTED=false")
    for key, value in result.items():
        print(f"{key}={value}")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
