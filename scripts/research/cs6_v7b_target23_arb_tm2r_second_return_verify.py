#!/usr/bin/env python3
"""Exact verifier for the retained Arb TM2R event projection and refusal."""

from __future__ import annotations

import argparse
import hashlib
import re
from fractions import Fraction
from pathlib import Path


WORKER_REL = Path("scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py")
DEPENDENCY_REL = Path("scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py")
KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
FRACTION_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")
FALSE_FLAGS = (
    "FULL_LEAF_SECOND_RETURN_CERTIFICATE", "RETURN_MAP_DETERMINANT_CERTIFICATE",
    "CAPD_USED_BY_WORKER", "POINT_FALLBACK_USED", "GLOBAL_HPG_CERTIFICATE",
    "V7_B_ELIGIBILITY", "CHAOS_PROVED", "CHAOTIC_ATTRACTOR_PROVED",
    "OPEN_PROBLEM_SOLVED", "NOVELTY_OR_PRIORITY_CLAIMED", "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"Arb TM2R event-projection verifier error: {message}")


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
    lower = fraction(data, f"{prefix}_LOWER_Q")
    upper = fraction(data, f"{prefix}_UPPER_Q")
    if lower > upper:
        fail(f"reversed interval: {prefix}")
    return lower, upper


def verify(path: Path) -> dict[str, str]:
    data = fields(path)
    expected = {
        "SCHEMA": "sounio.cs6.v7b-target23-arb-tm2r-second-return-worker.v1",
        "WORKER_SOURCE_SHA256": digest(Path.cwd() / WORKER_REL),
        "FIRST_RETURN_DEPENDENCY_SHA256": digest(Path.cwd() / DEPENDENCY_REL),
        "PYTHON_FLINT_VERSION": "0.8.0",
        "LEAF_ID": "U08-0000000223_S09-0000000325",
        "ARB_PRECISION_BITS": "256", "ARB_THREADS": "1",
        "SOURCE_VARIABLES": "2", "RESIDUAL_VARIABLES": "4",
        "FIRST_PHASE_SOURCE_DEGREE": "2", "SECOND_PHASE_SOURCE_DEGREE": "2",
        "FIRST_PHASE_TIME_TAYLOR_ORDER": "12",
        "SECOND_PHASE_TIME_TAYLOR_ORDER": "12", "TIME_STEP_POWER": "-8",
        "EVENT_PROJECTION": "INTERVAL_NEWTON_ENDPOINT_SLAB_WITH_PURE_SOURCE_COEFFICIENT_RETENTION",
        "FAILURE_CLASS": "ENDPOINT_ESCAPES_PICARD",
        "FAILURE_DETAIL": "raw_TM_endpoint_escaped_its_Picard_tube",
        "FAILURE_PHASE": "SECOND_RETURN",
        "TOTAL_ATTEMPTED_STEPS": "1325", "TOTAL_COMPLETED_STEPS": "1324",
        "TOTAL_PICARD_CONTAINMENTS": "1325",
        "TOTAL_ENDPOINT_PICARD_CONTAINMENTS": "1324",
        "TOTAL_RECONDITIONINGS": "1326",
        "TOTAL_GENERATOR_RECONSTRUCTIONS": "34213",
        "FIRST_RETURN_END_STEP": "617", "FIRST_INITIAL_DEPARTURE_TUBES": "1",
        "FIRST_PRIOR_DOWNWARD_TUBES": "1", "FIRST_ZERO_FREE_PRIOR_TUBES": "614",
        "PURE_SOURCE_MONOMIALS_RETAINED": "15", "PROJECTION_PICARD_ITERATIONS": "2",
        "PROJECTION_SLAB_PICARD_ITERATIONS": "2",
        "PROJECTION_SLAB_CONTAINED_IN_EVENT_TUBE": "true",
        "PROJECTED_W_EXACTLY_ZERO": "true", "SECOND_RETURN_ELAPSED_END_STEP": "-1",
        "SECOND_PHASE_ATTEMPTED_STEPS": "708", "SECOND_PHASE_COMPLETED_STEPS": "707",
        "SECOND_PHASE_COMPLETED_TIME_Q": "707/256", "BOUNDED_METHOD_RESULT": "true",
        "FULL_LEAF_FIRST_RETURN_CERTIFICATE": "true",
        "INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE": "true",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            fail(f"retained field mismatch: {key}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", data.get("PYTHON_VERSION", "")):
        fail("invalid Python version")
    for key in FALSE_FLAGS:
        if data.get(key) != "false":
            fail(f"forbidden promotion or fallback: {key}")

    before_w = interval(data, "FIRST_RETURN_W_BEFORE")
    after_w = interval(data, "FIRST_RETURN_W_AFTER")
    derivative = interval(data, "FIRST_RETURN_W_DERIVATIVE")
    denominator = interval(data, "NEWTON_DENOMINATOR")
    if before_w[1] >= 0 or after_w[0] <= 0:
        fail("first-return strict sign bracket is absent")
    if derivative[0] <= 0 or denominator != derivative:
        fail("event derivative is not strictly positive and retained exactly")

    delta = interval(data, "NEWTON_TIME_CORRECTION")
    fixed = interval(data, "NEWTON_FIXED_TIME_SHIFT")
    residual = interval(data, "NEWTON_RESIDUAL_TIME_SHIFT")
    if not (-Fraction(1, 256) <= delta[0] <= delta[1] <= 0):
        fail("Newton time slab escaped the validated event step")
    if fixed[0] != fixed[1]:
        fail("fixed event-flow time is not exact")
    if residual[0] != -residual[1] or residual[0] >= 0:
        fail("residual event-time interval is not a nontrivial symmetric radius")
    if (fixed[0] + residual[0], fixed[0] + residual[1]) != delta:
        fail("fixed and residual time pieces do not reconstruct the Newton slab")

    contraction = fraction(data, "PROJECTION_PICARD_CONTRACTION_UPPER_Q")
    slab_contraction = fraction(data, "PROJECTION_SLAB_PICARD_CONTRACTION_UPPER_Q")
    width = fraction(data, "PROJECTED_CARRIER_MAX_WIDTH_UPPER_Q")
    if not 0 < contraction < 1:
        fail("projection Picard contraction witness is invalid")
    if not 0 < slab_contraction < 1:
        fail("full event-slab Picard contraction witness is invalid")
    if not 0 < width < Fraction(1, 1000):
        fail("projected carrier width witness is invalid")
    if int(data["TOTAL_ATTEMPTED_STEPS"]) != 617 + int(data["SECOND_PHASE_ATTEMPTED_STEPS"]):
        fail("attempted-step accounting mismatch")
    if int(data["TOTAL_COMPLETED_STEPS"]) != 617 + int(data["SECOND_PHASE_COMPLETED_STEPS"]):
        fail("completed-step accounting mismatch")

    return {
        "FIRST_RETURN_END_STEP": data["FIRST_RETURN_END_STEP"],
        "PURE_SOURCE_MONOMIALS_RETAINED": data["PURE_SOURCE_MONOMIALS_RETAINED"],
        "SECOND_PHASE_ATTEMPTED_STEPS": data["SECOND_PHASE_ATTEMPTED_STEPS"],
        "SECOND_PHASE_COMPLETED_STEPS": data["SECOND_PHASE_COMPLETED_STEPS"],
        "SECOND_PHASE_COMPLETED_TIME_Q": data["SECOND_PHASE_COMPLETED_TIME_Q"],
        "FAILURE_CLASS": data["FAILURE_CLASS"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-second-return-verification.v1")
    for key, value in result.items():
        print(f"{key}={value}")
    print("STRICT_FIRST_EVENT_BRACKET_VERIFIED=true")
    print("INTERVAL_NEWTON_TIME_SLAB_VERIFIED=true")
    print("SIGNED_PICARD_EVENT_FLOW_VERIFIED=true")
    print("EVENT_POSITION_SLAB_CONTAINMENT_VERIFIED=true")
    print("SOURCE_VARIABLE_RETENTION_VERIFIED=true")
    print("EXACT_SECTION_CARRIER_VERIFIED=true")
    print("SECOND_PHASE_FAIL_CLOSED_VERIFIED=true")
    print("INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true")
    print("FULL_LEAF_SECOND_RETURN_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("OPEN_PROBLEM_SOLVED=false")


if __name__ == "__main__":
    main()
