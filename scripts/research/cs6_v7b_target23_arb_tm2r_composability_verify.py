#!/usr/bin/env python3
"""Fail-closed verifier for the full-support local covering B -> C."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-composability-covering.v1"
TILES = ("XLEL", "XLEH", "XHEL", "XHEH")


def fail(message: str) -> None:
    raise SystemExit(f"composability verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_fields(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for number, line in enumerate(value.splitlines(), 1):
        if not line or "=" not in line:
            fail(f"invalid aggregate line {number}")
        key, item = line.split("=", 1)
        if key in fields:
            fail(f"duplicate aggregate field {key}")
        fields[key] = item
    return fields


def require(fields: dict[str, str], key: str, expected: str) -> None:
    if fields.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {fields.get(key)!r}")


def fraction(fields: dict[str, str], key: str) -> Fraction:
    try:
        return Fraction(fields[key])
    except (KeyError, ValueError, ZeroDivisionError) as error:
        fail(f"invalid rational field {key}: {error}")


def bounds(fields: dict[str, str], prefix: str) -> tuple[Fraction, Fraction]:
    lower = fraction(fields, prefix + "_LOWER_Q")
    upper = fraction(fields, prefix + "_UPPER_Q")
    if lower > upper:
        fail(f"reversed interval {prefix}")
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    aggregate_path = args.receipts / "aggregate.txt"
    aggregate_bytes = aggregate_path.read_bytes()
    try:
        fields = parse_fields(aggregate_bytes.decode("ascii"))
    except UnicodeDecodeError as error:
        fail(f"aggregate is not ASCII: {error}")

    script_dir = Path(__file__).resolve().parent
    analyzer = script_dir / "cs6_v7b_target23_arb_tm2r_composability_analyze.py"
    receipt_paths = [args.receipts / f"support_{tile}.json" for tile in TILES]
    completed = subprocess.run(
        [sys.executable, str(analyzer), *(str(path) for path in receipt_paths)],
        cwd=script_dir.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            "deterministic reanalysis failed: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    if completed.stdout != aggregate_bytes:
        fail("aggregate differs from deterministic raw-evidence reanalysis")

    require(fields, "SCHEMA", SCHEMA)
    require(fields, "ANALYZER_SOURCE_SHA256", sha256(analyzer))
    require(fields, "LEAF_ID", "U08-0000000223_S09-0000000325")
    require(fields, "SOURCE_HSET", "B")
    require(fields, "TARGET_HSET", "C")
    require(fields, "MAP", "P2")
    require(fields, "SECTION", "w=0")
    require(fields, "SOURCE_TILE_COUNT", "4")
    require(fields, "SOURCE_TILES", ",".join(TILES))
    require(fields, "HSET_UNSTABLE_DIMENSION", "1")
    require(fields, "HSET_STABLE_DIMENSION", "1")
    require(fields, "TM_RESIDUAL_VARIABLES_ARE_ENCLOSURE_PARAMETERS", "true")
    require(fields, "COVERING_DEGREE_OBJECT", "UNSTABLE_TERMINAL_MAP_A")
    require(
        fields,
        "TARGET_ROW_DERIVATION",
        "RIGHT_MINUS_LEFT_RIGOROUS_BOUNDARY_HULL_CENTER",
    )
    require(
        fields,
        "EXIT_FACE_DERIVATION",
        "FULL_SUPPORT_BOUNDARY_RESTRICTION_WITH_XI_FIXED",
    )
    require(fields, "SOURCE_CHART_CONVENTION", "INVERSE_OF_ORTHOGONAL_ROW_COORDINATES")
    require(fields, "TARGET_CHART_CONVENTION", "INVERSE_OF_ORTHOGONAL_ROW_COORDINATES")
    require(fields, "INITIAL_NORMAL_DEFINITION", "x*y-ZS=dw/dt_ON_SECTION")
    require(fields, "POINCARE_DETERMINANT_FORMULA", "exp(ell)*initial_normal/final_normal")
    for tile in TILES:
        require(fields, f"CARRIER_RECEIPT_SHA256_{tile}", sha256(args.receipts / f"support_{tile}.json"))
        if fraction(fields, f"TERMINAL_COVER_CHECKS_{tile}") <= 0:
            fail(f"tile {tile} has no terminal-cover checks")
        profile = fields.get(f"EXECUTION_PROFILE_{tile}")
        wrapper_hash = fields.get(f"EXECUTION_WRAPPER_SOURCE_SHA256_{tile}")
        if profile == "BASE_SPLIT_BUDGET_V1":
            if wrapper_hash != "NONE":
                fail(f"base tile {tile} unexpectedly binds a retry wrapper")
        elif profile == "EXTENDED_SPLIT_BUDGET_V1":
            retry_worker = script_dir / "cs6_v7b_target23_arb_tm2r_composability_retry_worker.py"
            if wrapper_hash != sha256(retry_worker):
                fail(f"retry wrapper hash mismatch for tile {tile}")
        else:
            fail(f"unknown execution profile for tile {tile}")
    if fraction(fields, "SECOND_EVENT_CARRIER_COUNT") <= 0:
        fail("full support has no carriers")
    if fraction(fields, "LEFT_EXIT_CARRIER_COUNT") <= 0:
        fail("left exit face has no carriers")
    if fraction(fields, "RIGHT_EXIT_CARRIER_COUNT") <= 0:
        fail("right exit face has no carriers")
    if fraction(fields, "LEFT_EXIT_COVER_CHECKS") <= 0:
        fail("left exit face has no recursive cover checks")
    if fraction(fields, "RIGHT_EXIT_COVER_CHECKS") <= 0:
        fail("right exit face has no recursive cover checks")

    nx = fraction(fields, "TARGET_UNSTABLE_ROW_X_Q")
    ny = fraction(fields, "TARGET_UNSTABLE_ROW_Y_Q")
    require(fields, "TARGET_STABLE_ROW_X_Q", str(-ny))
    require(fields, "TARGET_STABLE_ROW_Y_Q", str(nx))
    if nx == 0 and ny == 0:
        fail("target unstable row is zero")
    target_linear = bounds(fields, "TARGET_LINEAR_DETERMINANT")
    if target_linear[0] <= 0:
        fail("target linear chart determinant touches zero")
    if bounds(fields, "SOURCE_LINEAR_DETERMINANT")[0] <= 0:
        fail("source linear chart determinant touches zero")
    if bounds(fields, "SOURCE_CHART_DETERMINANT")[0] <= 0:
        fail("source scaled chart determinant touches zero")
    if bounds(fields, "TARGET_CHART_DETERMINANT")[0] <= 0:
        fail("target scaled chart determinant touches zero")

    left_raw = bounds(fields, "LEFT_EXIT_UNSTABLE_RAW")
    right_raw = bounds(fields, "RIGHT_EXIT_UNSTABLE_RAW")
    stable_raw = bounds(fields, "SUPPORT_STABLE_RAW")
    gap = fraction(fields, "SIGNED_EXIT_FACE_GAP_Q")
    if gap != right_raw[0] - left_raw[1] or gap <= 0:
        fail("raw exit-face gap identity failed")
    target_u_center = fraction(fields, "TARGET_U_CENTER_Q")
    target_u_radius = fraction(fields, "TARGET_U_RADIUS_Q")
    if target_u_center != (left_raw[1] + right_raw[0]) / 2:
        fail("target unstable center identity failed")
    if target_u_radius != gap / 4 or target_u_radius <= 0:
        fail("target unstable radius identity failed")
    target_s_center = fraction(fields, "TARGET_S_CENTER_Q")
    target_s_radius = fraction(fields, "TARGET_S_RADIUS_Q")
    if target_s_center != (stable_raw[0] + stable_raw[1]) / 2:
        fail("target stable center identity failed")
    if target_s_radius != stable_raw[1] - stable_raw[0] or target_s_radius <= 0:
        fail("target stable radius identity failed")

    left = bounds(fields, "LEFT_EXIT_UNSTABLE_IMAGE")
    right = bounds(fields, "RIGHT_EXIT_UNSTABLE_IMAGE")
    stable = bounds(fields, "SUPPORT_STABLE_IMAGE")
    if left[1] >= -1 or right[0] <= 1:
        fail("an exit-face inequality is not strict")
    if stable[0] <= -1 or stable[1] >= 1:
        fail("an entry-boundary inequality is not strict")
    expected_exit_margin = min(-1 - left[1], right[0] - 1)
    expected_entry_margin = min(stable[0] + 1, 1 - stable[1])
    if expected_exit_margin <= 0 or fraction(fields, "EXIT_MARGIN_Q") != expected_exit_margin:
        fail("exit margin mismatch")
    if expected_entry_margin <= 0 or fraction(fields, "ENTRY_MARGIN_Q") != expected_entry_margin:
        fail("entry margin mismatch")

    if bounds(fields, "INITIAL_NORMAL")[0] <= 0:
        fail("initial section normal touches zero")
    if bounds(fields, "FINAL_NORMAL")[0] <= 0:
        fail("final section normal touches zero")
    if bounds(fields, "PHYSICAL_RETURN_DETERMINANT")[0] <= 0:
        fail("physical return determinant touches zero")
    if bounds(fields, "NORMALIZED_RETURN_DETERMINANT")[0] <= 0:
        fail("normalized return determinant touches zero")
    require(fields, "COVERING_DEGREE", "1")
    require(fields, "DEGREE_ARGUMENT", "ONE_DIMENSIONAL_BOUNDARY_SIGN_HOMOTOPY")

    for key in (
        "FULL_SUPPORT_CERTIFICATE",
        "DERIVED_EXIT_FACE_COVER_CERTIFICATE",
        "HSET_C_COORDINATES_CERTIFICATE",
        "ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE",
        "EXIT_FACE_INEQUALITIES_CERTIFICATE",
        "COVERING_DEGREE_CERTIFICATE",
        "RETURN_MAP_DETERMINANT_CERTIFICATE",
        "LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE",
    ):
        require(fields, key, "true")
    for key in (
        "RECURRENT_COVERING_GRAPH_CERTIFICATE",
        "CHAOS_PROVED",
        "OPEN_PROBLEM_SOLVED",
        "NOVELTY_OR_PRIORITY_CLAIMED",
        "CAPD_USED",
        "POINT_FALLBACK_USED",
        "BOX_FLATTENING_USED",
    ):
        require(fields, key, "false")

    print(f"SCHEMA={SCHEMA}")
    print(f"SECOND_EVENT_CARRIER_COUNT={fields['SECOND_EVENT_CARRIER_COUNT']}")
    print(f"SIGNED_EXIT_FACE_GAP_Q={gap}")
    print(f"ENTRY_MARGIN_Q={expected_entry_margin}")
    print(f"EXIT_MARGIN_Q={expected_exit_margin}")
    print("COVERING_DEGREE=1")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=true")
    print("LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE=true")
    print("RECURRENT_COVERING_GRAPH_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
