#!/usr/bin/env python3
"""Exact verifier for the retained Arb TM2R full-leaf first return."""

from __future__ import annotations

import argparse
import hashlib
import re
from fractions import Fraction
from pathlib import Path


WORKER_REL = Path("scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py")
KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
FRACTION_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")
FALSE_FLAGS = (
    "FULL_LEAF_SECOND_RETURN_CERTIFICATE", "CAPD_USED_BY_WORKER",
    "POINT_FALLBACK_USED", "GLOBAL_HPG_CERTIFICATE", "V7_B_ELIGIBILITY",
    "CHAOS_PROVED", "CHAOTIC_ATTRACTOR_PROVED", "OPEN_PROBLEM_SOLVED",
    "NOVELTY_OR_PRIORITY_CLAIMED", "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"Arb TM2R first-return verifier error: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("noncanonical output bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit("worker output is not ASCII") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail("malformed key-value row")
        key, value = line.split("=", 1)
        if not KEY_RE.fullmatch(key) or not value or key in result:
            fail(f"invalid or duplicate key: {key}")
        result[key] = value
    return result


def fraction(data: dict[str, str], key: str) -> Fraction:
    token = data.get(key, "")
    if not FRACTION_RE.fullmatch(token):
        fail(f"invalid exact fraction: {key}")
    value = Fraction(token)
    if str(value) != token:
        fail(f"noncanonical fraction: {key}")
    return value


def interval(data: dict[str, str], prefix: str) -> tuple[Fraction, Fraction]:
    lower, upper = fraction(data, f"{prefix}_LOWER_Q"), fraction(data, f"{prefix}_UPPER_Q")
    if lower > upper:
        fail(f"reversed interval: {prefix}")
    return lower, upper


def contains(enclosure: tuple[Fraction, Fraction], exact: Fraction) -> bool:
    return enclosure[0] <= exact <= enclosure[1]


def verify(path: Path) -> dict[str, str]:
    data = fields(path)
    expected = {
        "SCHEMA": "sounio.cs6.v7b-target23-arb-tm2r-first-return-worker.v1",
        "WORKER_SOURCE_SHA256": digest(Path.cwd() / WORKER_REL),
        "PYTHON_FLINT_VERSION": "0.8.0",
        "LEAF_ID": "U08-0000000223_S09-0000000325",
        "ARB_PRECISION_BITS": "256", "ARB_THREADS": "1",
        "SOURCE_DEGREE": "2", "SOURCE_VARIABLES": "2", "RESIDUAL_VARIABLES": "4",
        "RECONDITIONING": "QR_DERIVED_RATIONAL_BASIS_ZONOTOPE_HULL_EVERY_STEP",
        "TIME_TAYLOR_ORDER": "12", "TIME_STEP_POWER": "-8",
        "ATTEMPTED_STEPS": "617", "COMPLETED_STEPS": "617",
        "PICARD_CALLS": "617", "PICARD_CONTAINMENTS": "617",
        "ENDPOINT_PICARD_CONTAINMENTS": "617", "RECONDITIONINGS": "617",
        "GENERATOR_RECONSTRUCTIONS": "15810", "MAX_PICARD_ITERATIONS": "5",
        "EVENTS_VALIDATED": "1", "INITIAL_DEPARTURE_TUBES": "1",
        "PRIOR_DOWNWARD_TUBES": "1", "ZERO_FREE_PRIOR_TUBES": "614",
        "FAILURE_CLASS": "NONE", "FAILURE_DETAIL": "NONE",
        "FAILURE_STEP": "-1", "FIRST_RETURN_END_STEP": "617",
        "FULL_LEAF_FIRST_RETURN_CERTIFICATE": "true", "BOUNDED_METHOD_RESULT": "true",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            fail(f"retained field mismatch: {key}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", data.get("PYTHON_VERSION", "")):
        fail("invalid Python version")
    for key in FALSE_FLAGS:
        if data.get(key) != "false":
            fail(f"forbidden promotion or fallback: {key}")

    u_pair = (Fraction(19, 6400), Fraction(3, 1000))
    s_pair = (Fraction(207, 2560), Fraction(21, 256))
    for prefix, pair in (("LEAF_U", u_pair), ("LEAF_S", s_pair)):
        enclosure = interval(data, prefix)
        if not all(contains(enclosure, endpoint) for endpoint in pair):
            fail(f"leaf endpoint escaped: {prefix}")
    ox, oy = Fraction("15.186446520640786"), Fraction("10.908543194765466")
    ux, uy = Fraction("-0.67430316214199759"), Fraction("-0.73845463335624273")
    sx, sy = Fraction("-0.94170446778164518"), Fraction("0.33644122125579123")
    x_pair = (ox + ux * u_pair[1] + sx * s_pair[1], ox + ux * u_pair[0] + sx * s_pair[0])
    y_pair = (oy + uy * u_pair[1] + sy * s_pair[0], oy + uy * u_pair[0] + sy * s_pair[1])
    for prefix, pair in (("INITIAL_X", x_pair), ("INITIAL_Y", y_pair)):
        enclosure = interval(data, prefix)
        if not all(contains(enclosure, endpoint) for endpoint in pair):
            fail(f"physical endpoint escaped: {prefix}")

    if fraction(data, "FIRST_RETURN_TIME_LOWER_Q") != Fraction(616, 256):
        fail("first-return lower time mismatch")
    if fraction(data, "FIRST_RETURN_TIME_UPPER_Q") != Fraction(617, 256):
        fail("first-return upper time mismatch")
    before_w = interval(data, "FIRST_RETURN_W_BEFORE")
    after_w = interval(data, "FIRST_RETURN_W_AFTER")
    normal = interval(data, "FIRST_RETURN_NORMAL")
    derivative = interval(data, "FIRST_RETURN_W_DERIVATIVE")
    if before_w[1] >= 0:
        fail("pre-event w is not strictly negative")
    if after_w[0] <= 0:
        fail("post-event w is not strictly positive")
    if normal[0] <= 0:
        fail("event normal is not strictly positive")
    if derivative[0] <= 0:
        fail("target-step w derivative is not strictly positive")
    if fraction(data, "MAX_RAW_WIDTH_UPPER_Q") <= 0:
        fail("raw-width witness is not positive")
    if fraction(data, "MAX_RECONDITIONED_WIDTH_UPPER_Q") <= 0:
        fail("reconditioned-width witness is not positive")
    contraction = fraction(data, "MAX_PICARD_CONTRACTION_UPPER_Q")
    if not 0 < contraction < 1:
        fail("Picard contraction witness is invalid")
    if fraction(data, "MAX_BASIS_INVERSE_ROW_SUM_Q") <= 0:
        fail("basis inverse witness is invalid")

    return {
        "FIRST_RETURN_END_STEP": data["FIRST_RETURN_END_STEP"],
        "FIRST_RETURN_TIME_LOWER_Q": data["FIRST_RETURN_TIME_LOWER_Q"],
        "FIRST_RETURN_TIME_UPPER_Q": data["FIRST_RETURN_TIME_UPPER_Q"],
        "GENERATOR_RECONSTRUCTIONS": data["GENERATOR_RECONSTRUCTIONS"],
        "INITIAL_DEPARTURE_TUBES": data["INITIAL_DEPARTURE_TUBES"],
        "PRIOR_DOWNWARD_TUBES": data["PRIOR_DOWNWARD_TUBES"],
        "ZERO_FREE_PRIOR_TUBES": data["ZERO_FREE_PRIOR_TUBES"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-first-return-verification.v1")
    for key, value in result.items():
        print(f"{key}={value}")
    print("LEAF_GEOMETRY_EXACTLY_RECONSTRUCTED=true")
    print("STRICT_EVENT_BRACKET_VERIFIED=true")
    print("STRICT_EVENT_TRANSVERSALITY_VERIFIED=true")
    print("NO_PRIOR_POSITIVE_RETURN_VERIFIED=true")
    print("UNIQUE_TARGET_STEP_CROSSING_VERIFIED=true")
    print("FULL_LEAF_FIRST_RETURN_CERTIFICATE=true")
    print("FULL_LEAF_SECOND_RETURN_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("CHAOTIC_ATTRACTOR_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")


if __name__ == "__main__":
    main()
