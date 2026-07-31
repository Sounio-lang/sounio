#!/usr/bin/env python3
"""Fail-closed verifier for the bounded CS6 synchronized C1 rebox ledger."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Interval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise VerificationError("interval lower endpoint exceeds upper endpoint")

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower

    def contains(self, other: "Interval") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def overlaps(self, other: "Interval") -> bool:
        return self.lower <= other.upper and other.lower <= self.upper

    def contains_zero(self) -> bool:
        return self.lower <= 0 <= self.upper

    def scale(self, factor: Fraction) -> "Interval":
        products = (self.lower * factor, self.upper * factor)
        return Interval(min(products), max(products))

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lower + other.lower, self.upper + other.upper)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lower - other.upper, self.upper - other.lower)

    def __truediv__(self, other: "Interval") -> "Interval":
        if other.contains_zero():
            fail("interval division by a denominator containing zero")
        reciprocal = Interval(
            min(Fraction(1, 1) / other.lower, Fraction(1, 1) / other.upper),
            max(Fraction(1, 1) / other.lower, Fraction(1, 1) / other.upper),
        )
        return self * reciprocal


Record = Dict[str, str]

FROZEN_HEADER = {
    "SCHEMA": "sounio.cs6.c1-representation-rebox-probe.v1",
    "RESET_SEMANTICS": "REPRESENTATION_PRESERVING_CUMULATIVE_JRAW_REBOX",
    "CALL_PATTERN": "6xP1_SAME_MUTABLE_SET",
    "LOCAL_FACTOR_CHAIN": "false",
    "PREFIX_DP_PRODUCT_FORBIDDEN": "true",
    "FINAL_DP": "PREFIX_6_ONLY",
    "REBOX_COUNT": "5",
    "C0_CARRIER_RESET": "false",
    "LIOUVILLE_CARRIER_RESET": "false",
    "EVENT_DP_REINJECTION": "false",
    "RIGHT_REPARAMETERIZATION_ONLY": "true",
    "LAST_MATRIX_POLICY": "PRESERVE_EXACTLY_NO_REPARAMETERIZATION",
    "TARGET_NORMALIZATION_ONLY": "true",
    "LIOUVILLE_REJECT_ONLY": "true",
    "C1_CLIPPED_BY_LIOUVILLE": "false",
    "C1_SET_SUBTYPE": "ResettableC1Rect2Set",
    "CAPD_SOURCE_TREE_DECLARED": "capd-5.3.0",
    "INTERVAL_BACKEND_DECLARED": "FILIB",
    "INTERVAL_SERIALIZATION": "ONE_ULP_OUTWARD_BINARY64_HEX",
    "ORDER": "8",
    "EXECUTION_SCOPE": "BOUNDED_LOCAL_CAPD_CPU_PROBE",
    "EXECUTION_PROVENANCE_ATTESTED": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "PROMOTION_ELIGIBLE": "false",
    "RESET_AUDIT_MODEL": "HASHED_WORKER_REPLAY_TCB_SELF_REPORTED_FLAGS",
    "VECTOR_FIELD_CAPD":
        "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;",
    "LIOUVILLE_FIELD_CAPD":
        "par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,"
        "x*y-w-zs,x-y-(w+zs)/2-1;",
}

FALSE_CLAIMS = (
    "C1_REBOX_SCALING_BLOCKER_RESOLVED",
    "FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED",
    "GLOBAL_FULL_SOURCE_HULL_TESTED",
    "PAIRWISE_CHORD_CONE_CONDITION_PROVED",
    "UNIFORM_HYPERBOLICITY_PROVED",
    "CHAOTIC_ATTRACTOR_PROVED",
)

TRUE_RESET_FLAGS = (
    "C0_UNCHANGED",
    "SCRATCH_POLICY_UNCHANGED",
    "LAST_MATRIX_UNCHANGED",
    "CURRENT_EXACT",
    "CURRENT_CONTAINS_CANDIDATE",
    "DOUBLETON_CONTAINS_CANDIDATE",
    "PHYSICAL_CARRIER_CONTAINS_PRE",
    "INVERSE_BASIS_IDENTITY",
    "CANONICAL_FORM",
    "THIRD_COLUMN_ZERO",
    "SCALE_CHAIN_VALID",
)

STRATEGIES = (
    "direct",
    "sequential",
    "canonical-rebox",
    "dyadic-right-rebox",
)

MARKERS = {"RESULT", "PREFIX", "RESET", "LIOUVILLE_STATUS", "LIOUVILLE", "SUMMARY"}
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
HEX_RE = re.compile(r"^-?0x(?:[0-9a-f]+(?:\.[0-9a-f]*)?|\.[0-9a-f]+)p[+-]?[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MIN_NORMAL = Fraction.from_float(sys.float_info.min)
MAX_FINITE = Fraction.from_float(sys.float_info.max)
INT32_MAX = (1 << 31) - 1


def fail(message: str) -> None:
    raise VerificationError(message)


def parse_fraction(text: str) -> Fraction:
    if HEX_RE.fullmatch(text) is None:
        fail(f"number is not a canonical lowercase hexadecimal binary64: {text!r}")
    try:
        value = float.fromhex(text)
        if not math.isfinite(value):
            fail(f"non-finite binary64 value: {text}")
        return Fraction.from_float(value)
    except (ValueError, ZeroDivisionError) as error:
        raise VerificationError(f"invalid exact number {text!r}") from error


def parse_interval(text: str, context: str) -> Interval:
    match = INTERVAL_RE.fullmatch(text)
    if match is None:
        fail(f"{context}: malformed interval {text!r}")
    lower_text, upper_text = match.groups()
    lower_float = float.fromhex(lower_text) if HEX_RE.fullmatch(lower_text) else None
    upper_float = float.fromhex(upper_text) if HEX_RE.fullmatch(upper_text) else None
    if lower_float is None or upper_float is None:
        fail(f"{context}: endpoints must be hexadecimal binary64")
    if not math.isfinite(lower_float) or not math.isfinite(upper_float):
        fail(f"{context}: non-finite endpoint")
    if lower_float > upper_float:
        fail(f"{context}: inverted serialized interval")
    lower = math.nextafter(lower_float, math.inf)
    upper = math.nextafter(upper_float, -math.inf)
    if lower > upper:
        fail(f"{context}: interval is not a one-ULP outward encoding")
    return Interval(Fraction.from_float(lower), Fraction.from_float(upper))


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


def parse_tokens(tokens: Sequence[str], context: str) -> Record:
    record: Record = {}
    for token in tokens:
        if "=" not in token:
            fail(f"{context}: token has no '=': {token!r}")
        key, value = token.split("=", 1)
        if not key or not value:
            fail(f"{context}: malformed token {token!r}")
        if key in record:
            fail(f"{context}: duplicate key {key}")
        record[key] = value
    return record


def parse_ledger(text: str) -> Tuple[Record, Dict[str, List[Record]], List[Tuple[str, Record]]]:
    header: Record = {}
    groups: Dict[str, List[Record]] = {marker: [] for marker in MARKERS}
    events: List[Tuple[str, Record]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] in MARKERS:
            marker = tokens.pop(0)
            record = parse_tokens(tokens, f"line {line_number} {marker}")
            groups[marker].append(record)
            events.append((marker, record))
            continue
        record = parse_tokens(tokens, f"line {line_number} header")
        for key, value in record.items():
            if key in header:
                fail(f"line {line_number}: duplicate header key {key}")
            header[key] = value
    return header, groups, events


def require_keys(record: Mapping[str, str], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(record))
    if missing:
        fail(f"{context}: missing keys {','.join(missing)}")


def require_exact_keys(record: Mapping[str, str], keys: Iterable[str], context: str) -> None:
    expected = set(keys)
    actual = set(record)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        fail(f"{context}: grammar mismatch missing={missing} unknown={unknown}")


def require_value(record: Mapping[str, str], key: str, value: str, context: str) -> None:
    actual = record.get(key)
    if actual != value:
        fail(f"{context}: {key} must be {value!r}, got {actual!r}")


def exact_int(record: Mapping[str, str], key: str, context: str) -> int:
    try:
        value = int(record[key])
    except (KeyError, ValueError) as error:
        raise VerificationError(f"{context}: invalid integer {key}") from error
    return value


def intervals(record: Mapping[str, str], prefix: str, count: int, context: str) -> List[Interval]:
    keys = [f"{prefix}{index}" for index in range(count)]
    require_keys(record, keys, context)
    return [parse_interval(record[key], f"{context} {key}") for key in keys]


def matrix(record: Mapping[str, str], prefix: str, context: str) -> List[List[Interval]]:
    keys = [f"{prefix}{row}{column}" for row in range(3) for column in range(3)]
    require_keys(record, keys, context)
    return [
        [parse_interval(record[f"{prefix}{row}{column}"],
                        f"{context} {prefix}{row}{column}")
         for column in range(3)]
        for row in range(3)
    ]


def joint_interval(values: Sequence[Interval]) -> bool:
    return bool(values) and max(value.lower for value in values) <= min(
        value.upper for value in values
    )


def joint_vectors(values: Sequence[Sequence[Interval]]) -> bool:
    return bool(values) and all(
        joint_interval([value[row] for value in values]) for row in range(3)
    )


def joint_matrices(values: Sequence[Sequence[Sequence[Interval]]]) -> bool:
    return bool(values) and all(
        joint_interval([value[row][column] for value in values])
        for row in range(3)
        for column in range(3)
    )


def matrix_width(value: Sequence[Sequence[Interval]]) -> Fraction:
    return max(value[row][column].width for row in range(3) for column in range(3))


def normalized_width(value: Sequence[Sequence[Interval]]) -> Fraction:
    return max(entry.width for row in value for entry in row)


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def is_dyadic_power(value: Fraction) -> bool:
    return (
        MIN_NORMAL <= value <= MAX_FINITE
        and is_power_of_two(value.numerator)
        and is_power_of_two(value.denominator)
    )


def dyadic_exponent(value: Fraction) -> int:
    if not is_dyadic_power(value):
        fail("value is not a positive normal binary power")
    return value.numerator.bit_length() - value.denominator.bit_length()


def require_tight_enclosure(reported: Interval, calculated: Interval,
                            context: str, max_ulps: int = 512) -> None:
    if not reported.contains(calculated):
        fail(f"{context}: reported interval does not contain recomputed value")
    magnitude = max(abs(calculated.lower), abs(calculated.upper))
    slack = max_ulps * Fraction.from_float(math.ulp(float(magnitude)))
    allowed = Interval(calculated.lower - slack, calculated.upper + slack)
    if not allowed.contains(reported):
        fail(f"{context}: reported interval is not a tight enclosure")


def require_width_metric(record: Mapping[str, str], key: str, calculated: Fraction,
                         context: str) -> None:
    reported = finite_decimal(record, key, context, nonnegative=True)
    expected = float(calculated)
    if not math.isfinite(expected):
        fail(f"{context}: recomputed {key} is non-finite")
    tolerance = 16 * math.ulp(expected)
    if abs(reported - expected) > tolerance:
        fail(f"{context}: {key} does not match the recomputed width")


def assert_zero_third(value: Sequence[Sequence[Interval]], context: str) -> None:
    for row in range(3):
        if value[row][2] != Interval(Fraction(0), Fraction(0)):
            fail(f"{context}: row {row} third tangent column is not exactly zero")


def require_positive(value: Interval, context: str) -> None:
    if value.lower <= 0:
        fail(f"{context}: interval is not strictly positive")


def require_negative(value: Interval, context: str) -> None:
    if value.upper >= 0:
        fail(f"{context}: interval is not strictly negative")


def require_ordered_times(records: Sequence[Mapping[str, str]], context: str) -> None:
    previous: Interval | None = None
    for record in records:
        current = parse_interval(record["TIME"], f"{context} return {record['RETURN']} TIME")
        require_positive(current, f"{context} return {record['RETURN']} TIME")
        if previous is not None and previous.upper >= current.lower:
            fail(f"{context}: cumulative return times are not strictly ordered")
        previous = current


def interval_max_width(value: Sequence[Sequence[Interval]]) -> Fraction:
    return max(entry.width for row in value for entry in row)


def finite_decimal(record: Mapping[str, str], key: str, context: str,
                   nonnegative: bool = False) -> float:
    try:
        value = float(record[key])
    except (KeyError, ValueError) as error:
        raise VerificationError(f"{context}: invalid decimal {key}") from error
    if not math.isfinite(value) or (nonnegative and value < 0):
        fail(f"{context}: non-finite or negative {key}")
    return value


def assert_result_prefix_equal(result: Mapping[str, str], prefix: Mapping[str, str],
                               context: str) -> None:
    keys = ["TIME", *(f"X{i}" for i in range(3)),
            *(f"DP{r}{c}" for r in range(3) for c in range(3))]
    for key in keys:
        if result.get(key) != prefix.get(key):
            fail(f"{context}: final result {key} is not serialized PREFIX_6")


def verify_reset(record: Mapping[str, str], strategy: str, expected_return: int,
                 previous_external: Sequence[Fraction] | None) -> List[Fraction]:
    context = f"RESET {strategy} return {expected_return}"
    require_exact_keys(
        record,
        (
            "STRATEGY", "RETURN", "CANDIDATE_SOURCE", *TRUE_RESET_FLAGS,
            *(f"{prefix}{column}" for prefix in ("OLD_E", "S", "NEW_E")
              for column in range(3)),
            *(f"{prefix}{row}{column}" for prefix in ("PRE", "POST", "BOX")
              for row in range(3) for column in range(3)),
        ),
        context,
    )
    require_value(record, "STRATEGY", strategy, context)
    if exact_int(record, "RETURN", context) != expected_return:
        fail(f"{context}: non-consecutive return index")
    require_value(record, "CANDIDATE_SOURCE", "POSTSECTION_CURRENT_MATRIX", context)
    for flag in TRUE_RESET_FLAGS:
        require_value(record, flag, "true", context)

    old_e = [parse_fraction(record[f"OLD_E{i}"]) for i in range(3)]
    scale = [parse_fraction(record[f"S{i}"]) for i in range(3)]
    new_e = [parse_fraction(record[f"NEW_E{i}"]) for i in range(3)]
    if previous_external is None:
        if old_e != [Fraction(1), Fraction(1), Fraction(1)]:
            fail(f"{context}: first external chart is not identity")
    elif list(previous_external) != old_e:
        fail(f"{context}: OLD_E does not continue prior NEW_E")
    if not all(is_dyadic_power(value) for value in (*old_e, *scale, *new_e)):
        fail(f"{context}: chart contains zero, subnormal, overflow, or non-power-of-two value")
    if any(not -500 <= dyadic_exponent(value) <= 500 for value in scale):
        fail(f"{context}: reset scale is outside the worker's frozen exponent range")
    if scale[2] != 1 or old_e[2] != 1 or new_e[2] != 1:
        fail(f"{context}: third chart coordinate must remain identity")
    if strategy == "canonical-rebox" and scale != [Fraction(1)] * 3:
        fail(f"{context}: canonical rebox used a non-identity chart")
    for column in range(3):
        if new_e[column] != scale[column] * old_e[column]:
            fail(f"{context}: NEW_E != S * OLD_E in column {column}")

    pre = matrix(record, "PRE", context)
    post = matrix(record, "POST", context)
    box = matrix(record, "BOX", context)
    for value, name in ((pre, "PRE"), (post, "POST"), (box, "BOX")):
        assert_zero_third(value, f"{context} {name}")
    for row in range(3):
        for column in range(3):
            candidate = pre[row][column].scale(Fraction(1, 1) / scale[column])
            if post[row][column] != candidate:
                fail(f"{context}: POST is not exactly PRE/S at {row},{column}")
            if not box[row][column].contains(post[row][column]):
                fail(f"{context}: BOX does not contain POST at {row},{column}")
            physical_before = pre[row][column].scale(old_e[column])
            physical_after = post[row][column].scale(new_e[column])
            if physical_after != physical_before:
                fail(f"{context}: physical carrier changed at {row},{column}")
    return new_e


def ordered(records: Sequence[Mapping[str, str]], context: str,
            count: int = 6) -> List[Mapping[str, str]]:
    values = list(records)
    returns = [exact_int(item, "RETURN", context) for item in values]
    if returns != list(range(1, count + 1)):
        fail(f"{context}: returns must be exactly 1..{count}, got {returns}")
    return values


def frozen_geometry() -> Dict[str, Interval]:
    return {
        "zs": decimal_interval("22.3274637391"),
        "n0_center_u": Interval(Fraction(0), Fraction(0)),
        "n1_center_u": decimal_interval("0.019771776972779206"),
        "center_s": Interval(Fraction(0), Fraction(0)),
        "origin_x": decimal_interval("15.186446520640786"),
        "origin_y": decimal_interval("10.908543194765466"),
        "unstable_x": decimal_interval("-0.67430316214199759"),
        "unstable_y": decimal_interval("-0.73845463335624273"),
        "stable_x": decimal_interval("-0.94170446778164518"),
        "stable_y": decimal_interval("0.33644122125579123"),
        "n0_ru": decimal_interval("0.004"),
        "n1_ru": decimal_interval("0.0015"),
        "rs": decimal_interval("0.3"),
    }


def frozen_tile(source: str, coordinate: str, index: int, count: int,
                geometry: Mapping[str, Interval]) -> Tuple[Interval, Fraction]:
    if coordinate == "U":
        center = geometry["n0_center_u" if source == "N0" else "n1_center_u"]
        radius = geometry["n0_ru" if source == "N0" else "n1_ru"]
    elif coordinate == "S":
        center = geometry["center_s"]
        radius = geometry["rs"]
    else:
        fail(f"unknown tile coordinate {coordinate}")
    left = center - radius
    step = radius.scale(Fraction(2, count))
    lower = left + step.scale(Fraction(index))
    upper = left + step.scale(Fraction(index + 1))
    logical = Interval(lower.lower, upper.upper)

    # FILIB rounds each operation outward. Bound that implementation-level
    # inflation by the finite operation count at the source-set scale.
    scale = max(*(abs(value) for value in (center.lower, center.upper,
                                           radius.lower, radius.upper)))
    slack = 32 * Fraction.from_float(math.ulp(float(scale)))
    return logical, slack


def require_tile_binding(reported: Interval, logical: Interval, slack: Fraction,
                         context: str) -> None:
    if not reported.contains(logical):
        fail(f"{context}: payload does not contain the indexed logical tile")
    allowed = Interval(logical.lower - slack, logical.upper + slack)
    if not allowed.contains(reported):
        fail(f"{context}: payload is wider than the frozen FILIB rounding budget")


def normal_velocity(image: Sequence[Interval], geometry: Mapping[str, Interval]) -> Interval:
    return image[0] * image[1] - image[2] - geometry["zs"]


def normalized_derivative(
    dp: Sequence[Sequence[Interval]], target: str, geometry: Mapping[str, Interval]
) -> List[List[Interval]]:
    frame_determinant = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    local00 = (
        geometry["stable_y"] * dp[0][0] - geometry["stable_x"] * dp[1][0]
    ) / frame_determinant
    local01 = (
        geometry["stable_y"] * dp[0][1] - geometry["stable_x"] * dp[1][1]
    ) / frame_determinant
    local10 = (
        Interval(Fraction(0), Fraction(0)) - geometry["unstable_y"]
    ) * dp[0][0] + geometry["unstable_x"] * dp[1][0]
    local11 = (
        Interval(Fraction(0), Fraction(0)) - geometry["unstable_y"]
    ) * dp[0][1] + geometry["unstable_x"] * dp[1][1]
    local10 = local10 / frame_determinant
    local11 = local11 / frame_determinant
    target_ru = geometry["n0_ru"] if target == "N0" else geometry["n1_ru"]
    return [
        [local00 / target_ru, local01 / target_ru],
        [local10 / geometry["rs"], local11 / geometry["rs"]],
    ]


def initial_liouville_data(
    source: str,
    source_u: Interval,
    source_s: Interval,
    geometry: Mapping[str, Interval],
) -> Tuple[Interval, Interval]:
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
    initial_velocity = initial_x * initial_y - geometry["zs"]
    frame_determinant = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    source_ru = geometry["n0_ru"] if source == "N0" else geometry["n1_ru"]
    source_frame = frame_determinant * source_ru * geometry["rs"]
    return initial_velocity, source_frame


def verify(path: Path, expect_rebox_worse: bool,
           expect_c0_nontransversal_failure: bool = False) -> Dict[str, str]:
    data = path.read_bytes()
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise VerificationError("ledger must be ASCII") from error
    header, groups, events = parse_ledger(text)

    for key, value in FROZEN_HEADER.items():
        require_value(header, key, value, "header")
    dynamic_header = ("SOURCE", "TARGET", "U_INDEX", "S_INDEX", "U_TILES",
                      "S_TILES", "SOURCE_U", "SOURCE_S", "WORKER_SOURCE_SHA256")
    require_exact_keys(
        header,
        (*FROZEN_HEADER.keys(), *dynamic_header, *FALSE_CLAIMS),
        "header",
    )
    if header["SOURCE"] not in {"N0", "N1"} or header["TARGET"] not in {"N0", "N1"}:
        fail("header: unknown h-set")
    if SHA256_RE.fullmatch(header["WORKER_SOURCE_SHA256"]) is None:
        fail("header: worker source hash is absent or malformed")
    if header["SOURCE"] == "N1" and header["TARGET"] != "N0":
        fail("header: edge is outside frozen adjacency")
    for key in ("U_INDEX", "S_INDEX", "U_TILES", "S_TILES", "ORDER"):
        value = exact_int(header, key, "header")
        if (value < 0 or value > INT32_MAX or
                (key in {"U_TILES", "S_TILES", "ORDER"} and value < 1)):
            fail(f"header: invalid {key}")
    if int(header["U_INDEX"]) >= int(header["U_TILES"]) or int(header["S_INDEX"]) >= int(header["S_TILES"]):
        fail("header: tile index outside partition")
    source_u = parse_interval(header["SOURCE_U"], "header SOURCE_U")
    source_s = parse_interval(header["SOURCE_S"], "header SOURCE_S")
    geometry = frozen_geometry()
    logical_u, u_slack = frozen_tile(
        header["SOURCE"], "U", int(header["U_INDEX"]), int(header["U_TILES"]), geometry
    )
    logical_s, s_slack = frozen_tile(
        header["SOURCE"], "S", int(header["S_INDEX"]), int(header["S_TILES"]), geometry
    )
    require_tile_binding(source_u, logical_u, u_slack, "header SOURCE_U")
    require_tile_binding(source_s, logical_s, s_slack, "header SOURCE_S")
    for claim in FALSE_CLAIMS:
        require_value(header, claim, "false", "claim boundary")

    expected_events: List[Tuple[str, str, str]] = [("RESULT", "direct", "")]
    expected_events.append(("RESULT", "sequential", ""))
    expected_events.extend(("PREFIX", "sequential", str(index)) for index in range(1, 7))
    expected_events.append(("RESULT", "canonical-rebox", ""))
    expected_events.extend(("PREFIX", "canonical-rebox", str(index)) for index in range(1, 7))
    expected_events.extend(("RESET", "canonical-rebox", str(index)) for index in range(1, 6))
    expected_events.append(("RESULT", "dyadic-right-rebox", ""))
    expected_events.extend(("PREFIX", "dyadic-right-rebox", str(index)) for index in range(1, 7))
    expected_events.extend(("RESET", "dyadic-right-rebox", str(index)) for index in range(1, 6))
    liouville_count = 5 if expect_c0_nontransversal_failure else 6
    expected_events.append(("LIOUVILLE_STATUS", "", ""))
    expected_events.extend(
        ("LIOUVILLE", "", str(index)) for index in range(1, liouville_count + 1)
    )
    expected_events.append(("SUMMARY", "", ""))
    actual_events = [
        (marker, record.get("STRATEGY", ""), record.get("RETURN", ""))
        for marker, record in events
    ]
    if actual_events != expected_events:
        fail("ledger record order or cardinality differs from the frozen grammar")

    if len(groups["RESULT"]) != 4:
        fail("ledger must contain exactly four RESULT records")
    results: Dict[str, Mapping[str, str]] = {}
    calculated_normalized: Dict[str, List[List[Interval]]] = {}
    for record in groups["RESULT"]:
        strategy = record.get("STRATEGY", "")
        if strategy not in STRATEGIES or strategy in results:
            fail(f"RESULT: invalid or duplicate strategy {strategy!r}")
        context = f"RESULT {strategy}"
        require_exact_keys(
            record,
            (
                "STRATEGY", "SUCCESS", "RESET_COUNT", "PREFIX_COUNT",
                "ELAPSED_SECONDS", "TIME", *(f"X{i}" for i in range(3)),
                *(f"DP{row}{column}" for row in range(3) for column in range(3)),
                *(f"A{row}{column}" for row in range(2) for column in range(2)),
                "DP_MAX_WIDTH", "A_MAX_WIDTH",
            ),
            context,
        )
        require_value(record, "SUCCESS", "true", context)
        expected_resets = 5 if strategy in {"canonical-rebox", "dyadic-right-rebox"} else 0
        expected_prefixes = 0 if strategy == "direct" else 6
        if exact_int(record, "RESET_COUNT", f"RESULT {strategy}") != expected_resets:
            fail(f"RESULT {strategy}: wrong reset count")
        if exact_int(record, "PREFIX_COUNT", f"RESULT {strategy}") != expected_prefixes:
            fail(f"RESULT {strategy}: wrong prefix count")
        dp = matrix(record, "DP", context)
        assert_zero_third(dp, context)
        image = intervals(record, "X", 3, context)
        if not image[2].contains_zero():
            fail(f"{context}: section coordinate excludes zero")
        require_positive(normal_velocity(image, geometry), f"{context} recomputed NU")
        require_positive(parse_interval(record["TIME"], f"{context} TIME"), f"{context} TIME")
        finite_decimal(record, "ELAPSED_SECONDS", context, nonnegative=True)
        calculated_a = normalized_derivative(dp, header["TARGET"], geometry)
        reported_a = [[parse_interval(record[f"A{row}{column}"],
                                      f"{context} A{row}{column}")
                       for column in range(2)] for row in range(2)]
        for row in range(2):
            for column in range(2):
                require_tight_enclosure(
                    reported_a[row][column], calculated_a[row][column],
                    f"{context} A{row}{column}", max_ulps=64,
                )
        require_width_metric(record, "DP_MAX_WIDTH", matrix_width(dp), context)
        require_width_metric(record, "A_MAX_WIDTH", normalized_width(reported_a), context)
        calculated_normalized[strategy] = calculated_a
        results[strategy] = record

    prefixes: Dict[str, List[Mapping[str, str]]] = {}
    for strategy in STRATEGIES[1:]:
        selected = [record for record in groups["PREFIX"] if record.get("STRATEGY") == strategy]
        prefixes[strategy] = ordered(selected, f"PREFIX {strategy}")
        assert_result_prefix_equal(results[strategy], prefixes[strategy][-1], f"RESULT {strategy}")
        for record in prefixes[strategy]:
            context = f"PREFIX {strategy} return {record['RETURN']}"
            require_exact_keys(
                record,
                (
                    "STRATEGY", "RETURN", "TIME", *(f"X{i}" for i in range(3)),
                    "NU", *(f"DP{row}{column}" for row in range(3)
                            for column in range(3)), "DET",
                ),
                context,
            )
            dp = matrix(record, "DP", context)
            assert_zero_third(dp, context)
            determinant = parse_interval(record["DET"], f"{context} DET")
            calculated = dp[0][0] * dp[1][1] - dp[0][1] * dp[1][0]
            require_tight_enclosure(determinant, calculated, f"{context} DET")
            image = intervals(record, "X", 3, context)
            if not image[2].contains_zero():
                fail(f"{context}: section coordinate excludes zero")
            reported_nu = parse_interval(record["NU"], f"{context} NU")
            calculated_nu = normal_velocity(image, geometry)
            require_tight_enclosure(reported_nu, calculated_nu, f"{context} NU")
            require_positive(reported_nu, f"{context} NU")
        require_ordered_times(prefixes[strategy], f"PREFIX {strategy}")
    unknown_prefix = [record for record in groups["PREFIX"]
                      if record.get("STRATEGY") not in STRATEGIES[1:]]
    if unknown_prefix:
        fail("PREFIX: direct or unknown strategy record is forbidden")

    for strategy in ("canonical-rebox", "dyadic-right-rebox"):
        selected = [record for record in groups["RESET"] if record.get("STRATEGY") == strategy]
        if [exact_int(item, "RETURN", f"RESET {strategy}") for item in selected] != list(range(1, 6)):
            fail(f"RESET {strategy}: returns must be exactly 1..5")
        external: Sequence[Fraction] | None = None
        for expected_return, record in enumerate(selected, 1):
            external = verify_reset(record, strategy, expected_return, external)
    if any(record.get("STRATEGY") not in {"canonical-rebox", "dyadic-right-rebox"}
           for record in groups["RESET"]):
        fail("RESET: direct or sequential reset is forbidden")
    if len(groups["RESET"]) != 10:
        fail("ledger must contain exactly ten RESET records")

    if len(groups["LIOUVILLE_STATUS"]) != 1:
        fail("ledger must contain exactly one LIOUVILLE_STATUS")
    liouville_status = groups["LIOUVILLE_STATUS"][0]
    if expect_c0_nontransversal_failure:
        require_exact_keys(
            liouville_status, ("SUCCESS", "PREFIX_COUNT", "ERROR"),
            "LIOUVILLE_STATUS",
        )
        require_value(liouville_status, "SUCCESS", "false", "LIOUVILLE_STATUS")
        require_value(liouville_status, "PREFIX_COUNT", "5", "LIOUVILLE_STATUS")
        error = liouville_status["ERROR"]
        expected_error = "PoincareMap_error:_possible_nontransversal_return_to_the_section_"
        if (not error.startswith(expected_error) or
                any(character.isspace() for character in error)):
            fail("LIOUVILLE_STATUS: expected a safe nontransversality error token")
    else:
        require_exact_keys(
            liouville_status, ("SUCCESS", "PREFIX_COUNT"), "LIOUVILLE_STATUS"
        )
        require_value(liouville_status, "SUCCESS", "true", "LIOUVILLE_STATUS")
        require_value(liouville_status, "PREFIX_COUNT", "6", "LIOUVILLE_STATUS")
    liouville = ordered(groups["LIOUVILLE"], "LIOUVILLE", liouville_count)
    initial_velocity, source_frame = initial_liouville_data(
        header["SOURCE"], source_u, source_s, geometry
    )
    require_positive(initial_velocity, "Liouville initial normal velocity")
    require_negative(source_frame, "Liouville source frame determinant")
    for record in liouville:
        context = f"LIOUVILLE return {record['RETURN']}"
        require_exact_keys(
            record,
            ("RETURN", "TIME", *(f"X{i}" for i in range(3)), "NU", "ELL",
             "EXP_ELL", "DET_SOURCE_FRAME"),
            context,
        )
        image = intervals(record, "X", 3, context)
        if not image[2].contains_zero():
            fail(f"{context}: section coordinate excludes zero")
        nu = parse_interval(record["NU"], f"{context} NU")
        calculated_nu = normal_velocity(image, geometry)
        require_tight_enclosure(nu, calculated_nu, f"{context} NU")
        require_positive(nu, f"{context} NU")
        parse_interval(record["ELL"], f"{context} ELL")
        exponential = parse_interval(record["EXP_ELL"], f"{context} EXP_ELL")
        require_positive(exponential, f"{context} EXP_ELL")
        determinant = parse_interval(record["DET_SOURCE_FRAME"], f"{context} DET")
        require_negative(determinant, f"{context} DET")
        calculated_det = exponential * initial_velocity / calculated_nu * source_frame
        require_tight_enclosure(determinant, calculated_det, f"{context} DET")
    require_ordered_times(liouville, "LIOUVILLE")

    for index in range(6):
        c1 = [prefixes[strategy][index] for strategy in STRATEGIES[1:]]
        context = f"joint prefix return {index + 1}"
        if not joint_matrices([matrix(record, "DP", context) for record in c1]):
            fail(f"{context}: C1 DP lanes have empty joint intersection")
        times = [parse_interval(record["TIME"], f"{context} TIME") for record in c1]
        images = [intervals(record, "X", 3, context) for record in c1]
        normal_velocities = [
            normal_velocity(intervals(record, "X", 3, context), geometry)
            for record in c1
        ]
        determinants = []
        for record in c1:
            dp = matrix(record, "DP", context)
            determinants.append(dp[0][0] * dp[1][1] - dp[0][1] * dp[1][0])
        if index == 5:
            direct = results["direct"]
            times.append(parse_interval(direct["TIME"], f"{context} direct TIME"))
            direct_image = intervals(direct, "X", 3, f"{context} direct")
            images.append(direct_image)
            normal_velocities.append(normal_velocity(direct_image, geometry))
            direct_dp = matrix(direct, "DP", f"{context} direct")
            determinants.append(
                direct_dp[0][0] * direct_dp[1][1]
                - direct_dp[0][1] * direct_dp[1][0]
            )
        if index < liouville_count:
            independent = liouville[index]
            times.append(
                parse_interval(independent["TIME"], f"{context} LIOUVILLE TIME")
            )
            independent_image = intervals(
                independent, "X", 3, f"{context} LIOUVILLE"
            )
            images.append(independent_image)
            independent_nu = normal_velocity(independent_image, geometry)
            normal_velocities.append(independent_nu)
            determinants.append(
                parse_interval(
                    independent["EXP_ELL"], f"{context} LIOUVILLE EXP_ELL"
                ) * initial_velocity / independent_nu * source_frame
            )
        if not joint_interval(times):
            fail(f"{context}: time lanes have empty joint intersection")
        if not joint_vectors(images):
            fail(f"{context}: state lanes have empty joint intersection")
        if not joint_interval(normal_velocities):
            fail(f"{context}: normal-velocity lanes have empty joint intersection")
        if not joint_interval(determinants):
            fail(f"{context}: determinant lanes have empty joint intersection")

    final_dp = [matrix(results[strategy], "DP", f"RESULT {strategy}")
                for strategy in STRATEGIES]
    final_x = [intervals(results[strategy], "X", 3, f"RESULT {strategy}")
               for strategy in STRATEGIES]
    final_time = [parse_interval(results[strategy]["TIME"], f"RESULT {strategy} TIME")
                  for strategy in STRATEGIES]
    if not joint_matrices(final_dp):
        fail("final DP lanes have empty joint intersection")
    if not joint_vectors(final_x):
        fail("final C0 lanes have empty joint intersection")
    if not joint_interval(final_time):
        fail("final time lanes have empty joint intersection")

    if len(groups["SUMMARY"]) != 1:
        fail("ledger must contain exactly one SUMMARY")
    summary = groups["SUMMARY"][0]
    summary_keys = ("ALL_STRATEGIES_SUCCESS", "FINAL_DP_OVERLAP", "FINAL_C0_OVERLAP",
                    "FINAL_TIME_OVERLAP", "PREFIX_C1_DP_JOINT_OVERLAP",
                    "PREFIX_LIOUVILLE_OVERLAP",
                    "PREFIX_LIOUVILLE_DETERMINANT_OVERLAP", "PROBE_PASS")
    require_exact_keys(summary, summary_keys, "SUMMARY")
    for key in summary_keys:
        require_value(
            summary, key,
            "false" if expect_c0_nontransversal_failure else "true",
            "SUMMARY",
        )

    widths = {strategy: normalized_width(calculated_normalized[strategy])
              for strategy in STRATEGIES}
    direct_width = widths["direct"]
    if direct_width <= 0:
        fail("RESULT direct: non-positive normalized width")
    canonical_ratio = widths["canonical-rebox"] / direct_width
    scaled_ratio = widths["dyadic-right-rebox"] / direct_width
    if expect_rebox_worse and not (canonical_ratio > 10 and scaled_ratio > 10):
        fail("bounded negative-control receipt no longer shows >10x rebox widening")

    physical_header_keys = (
        "SOURCE", "U_INDEX", "S_INDEX", "U_TILES", "S_TILES", "ORDER",
        "SOURCE_U", "SOURCE_S", "WORKER_SOURCE_SHA256",
    )
    physical_records = [
        "HEADER " + " ".join(f"{key}={header[key]}" for key in physical_header_keys)
    ]
    result_physical_keys = (
        "STRATEGY", "TIME", *(f"X{index}" for index in range(3)),
        *(f"DP{row}{column}" for row in range(3) for column in range(3)),
    )
    for record in groups["RESULT"]:
        physical_records.append(
            "RESULT " + " ".join(f"{key}={record[key]}" for key in result_physical_keys)
        )
    for marker in ("PREFIX", "RESET", "LIOUVILLE_STATUS", "LIOUVILLE"):
        for record in groups[marker]:
            physical_records.append(
                marker + " " + " ".join(f"{key}={record[key]}" for key in sorted(record))
            )
    physical_digest = hashlib.sha256(
        ("\n".join(sorted(physical_records)) + "\n").encode("ascii")
    ).hexdigest()

    return {
        "LEDGER_SHA256": hashlib.sha256(data).hexdigest(),
        "PHYSICAL_CHAIN_SHA256": physical_digest,
        "SOURCE": header["SOURCE"],
        "TARGET": header["TARGET"],
        "TILE": f"{header['U_INDEX']},{header['S_INDEX']}/{header['U_TILES']},{header['S_TILES']}",
        "DIRECT_A_WIDTH": f"{float(direct_width):.17g}",
        "SEQUENTIAL_A_WIDTH": f"{float(widths['sequential']):.17g}",
        "CANONICAL_A_WIDTH": f"{float(widths['canonical-rebox']):.17g}",
        "DYADIC_A_WIDTH": f"{float(widths['dyadic-right-rebox']):.17g}",
        "CANONICAL_TO_DIRECT_RATIO": f"{float(canonical_ratio):.17g}",
        "DYADIC_TO_DIRECT_RATIO": f"{float(scaled_ratio):.17g}",
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    outcomes = parser.add_mutually_exclusive_group()
    outcomes.add_argument(
        "--expect-rebox-worse", action="store_true",
        help="require both rebox controls to widen normalized A by >10x",
    )
    outcomes.add_argument(
        "--expect-c0-nontransversal-failure", action="store_true",
        help="require five valid Liouville prefixes followed by C0 nontransversality",
    )
    args = parser.parse_args(argv)
    try:
        metrics = verify(
            args.ledger, args.expect_rebox_worse,
            args.expect_c0_nontransversal_failure,
        )
    except (OSError, VerificationError) as error:
        print(f"VERIFY_ERROR={error}", file=sys.stderr)
        return 2
    print("VERIFY_PASS=true")
    for key, value in metrics.items():
        print(f"{key}={value}")
    if args.expect_c0_nontransversal_failure:
        print("RESULT_CLASS=BOUNDED_EXPECTED_C0_NONTRANSVERSAL_FAILURE")
        print("C1_STRATEGIES_COMPLETE=true")
        print("LIOUVILLE_PREFIX_COUNT=5")
    else:
        print("RESULT_CLASS=BOUNDED_NEGATIVE_EFFICIENCY_RESULT")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
