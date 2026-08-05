#!/usr/bin/env python3
"""Fail-closed verifier for the QR falsifier and anchored C2 covering receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-c2-anchored-local-hset-covering.v1"


def fail(message: str) -> None:
    raise SystemExit(f"covector QR/C2 verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_fields(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(value.splitlines(), 1):
        if not line or "=" not in line:
            fail(f"invalid aggregate line {number}")
        key, item = line.split("=", 1)
        if key in result:
            fail(f"duplicate aggregate field {key}")
        result[key] = item
    return result


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
        aggregate_text = aggregate_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        fail(f"aggregate is not ASCII: {error}")
    fields = parse_fields(aggregate_text)

    script_dir = Path(__file__).resolve().parent
    analyzer = script_dir / "cs6_v7b_target23_arb_tm2r_covector_qr_analyze.py"
    completed = subprocess.run(
        [sys.executable, str(analyzer)],
        cwd=script_dir.parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        fail("deterministic reanalysis failed: " + completed.stderr.decode("utf-8", "replace").strip())
    if aggregate_bytes != completed.stdout:
        fail("aggregate differs from deterministic raw-evidence reanalysis")

    require(fields, "SCHEMA", SCHEMA)
    require(fields, "LEAF_ID", "U08-0000000223_S09-0000000325")
    require(fields, "MAP", "P2")
    require(fields, "SECTION", "w=0")
    require(fields, "CAPD_VERSION", "5.3.0")
    require(fields, "CAPD_INTERVAL_BACKEND", "FILIB")
    require(fields, "CAPD_RETURN_COUNT", "2")
    require(fields, "CAPD_HESSIAN_DIAGONAL_FACTOR", "2")
    require(fields, "CAPD_HESSIAN_OFFDIAGONAL_FACTOR", "1")
    require(fields, "CAPD_DELTA0_RADIUS_Q", "1/256")
    require(fields, "CAPD_DELTA1_RADIUS_Q", "1/512")
    require(fields, "ANALYZER_SOURCE_SHA256", sha256(analyzer))

    tripleton_path = args.receipts / "face_LEFT_XLEH_ROOT.json"
    doubleton_stderr = args.receipts / "doubleton_LEFT_XLEH_ROOT.stderr.txt"
    doubleton_incomplete = args.receipts / "doubleton_LEFT_XLEH_ROOT.incomplete.json"
    require(fields, "TRIPLETON_RECEIPT_SHA256", sha256(tripleton_path))
    require(fields, "DOUBLETON_STDERR_SHA256", sha256(doubleton_stderr))
    require(fields, "DOUBLETON_INCOMPLETE_SHA256", sha256(doubleton_incomplete))
    if doubleton_incomplete.read_bytes() != b"":
        fail("doubleton incomplete marker is not empty")
    if doubleton_stderr.read_text(encoding="ascii") != "first-event-projection tile=XLEH end_step=617\n":
        fail("doubleton first-event evidence mismatch")
    tripleton = json.loads(tripleton_path.read_text(encoding="ascii"))
    if tripleton.get("carrier_kind") != "DYNAMIC_TRIPLETON":
        fail("wrong tripleton carrier kind")
    if tripleton.get("selected_source_face_chain_certificate") is not True:
        fail("missing tripleton second-section chain certificate")
    if tripleton.get("point_fallback_used") is not False:
        fail("tripleton used point fallback")
    if tripleton.get("reconditionings") != 2570:
        fail("tripleton reconditioning count mismatch")
    if tripleton.get("dynamic_direction_transport_count") != 2569:
        fail("tripleton transport count mismatch")

    c1 = bounds(fields, "C1_DIRECT_DU_DGLOBAL0")
    c2 = bounds(fields, "C2_MEAN_VALUE_DU_DXI")
    if not (c1[0] < 0 < c1[1]):
        fail("direct C1 interval must retain its zero crossing")
    if c2[0] <= 0:
        fail("C2 mean-value derivative is not strictly positive")

    left = bounds(fields, "ANCHORED_LEFT_FACE_U_RAW")
    right_direct = bounds(fields, "DIRECT_RIGHT_FACE_U_RAW")
    right_anchored = fraction(fields, "ANCHORED_RIGHT_FACE_LOWER_Q")
    gap = fraction(fields, "ANCHORED_EXIT_FACE_GAP_Q")
    if gap != right_anchored - left[1] or gap <= 0:
        fail("anchored exit-face gap identity failed")
    if right_anchored != left[0] + 2 * c2[0]:
        fail("anchored monotonicity integration identity failed")
    if right_direct[0] < right_anchored:
        fail("direct right face does not cross-check the anchored lower bound")

    target_center = fraction(fields, "TARGET_U_CENTER_Q")
    target_radius = fraction(fields, "TARGET_U_RADIUS_Q")
    if target_center != (left[1] + right_anchored) / 2:
        fail("target unstable center identity failed")
    if target_radius != gap / 4 or target_radius <= 0:
        fail("target unstable radius identity failed")
    normalized_left = bounds(fields, "NORMALIZED_LEFT_EXIT_IMAGE")
    normalized_right = fraction(fields, "NORMALIZED_RIGHT_EXIT_LOWER_Q")
    if normalized_left[1] != -2 or normalized_right != 2:
        fail("normalized anchored faces are not at -2 and +2")
    if fraction(fields, "EXIT_MARGIN_Q") != 1:
        fail("exit margin is not exactly one")

    stable = bounds(fields, "SUPPORT_STABLE_IMAGE")
    expected_entry_margin = min(stable[0] + 1, 1 - stable[1])
    if expected_entry_margin <= 0 or fraction(fields, "ENTRY_MARGIN_Q") != expected_entry_margin:
        fail("stable entry margin mismatch")
    if bounds(fields, "TARGET_LINEAR_DETERMINANT")[0] <= 0:
        fail("target linear chart is singular or orientation-reversing")
    if bounds(fields, "TARGET_CHART_DETERMINANT")[0] <= 0:
        fail("target scaled chart is singular or orientation-reversing")
    if bounds(fields, "PHYSICAL_RETURN_DETERMINANT")[1] >= 0:
        fail("physical return determinant is not strictly negative")
    if bounds(fields, "NORMALIZED_RETURN_DETERMINANT")[1] >= 0:
        fail("normalized return determinant is not strictly negative")

    baseline = fraction(fields, "BASELINE_DIRECTIONAL_TOTAL_RADIUS_Q")
    triple = fraction(fields, "TRIPLETON_DIRECTIONAL_TOTAL_RADIUS_Q")
    ratio = fraction(fields, "TRIPLETON_TO_BASELINE_RADIUS_RATIO_Q")
    improvement = fraction(fields, "TRIPLETON_IMPROVEMENT_FACTOR_Q")
    if baseline <= 0 or triple <= 0:
        fail("directional radii must be positive")
    if ratio != triple / baseline or ratio <= 1:
        fail("tripleton worsening ratio mismatch")
    if improvement != baseline / triple or improvement >= 1:
        fail("tripleton improvement factor mismatch")
    if triple <= baseline / 18:
        fail("18x target was not falsified")

    for key, expected in {
        "FIXED_COVECTOR_QR_CERTIFICATE": "false",
        "DYNAMIC_DOUBLETON_SECOND_SECTION_CERTIFICATE": "false",
        "DYNAMIC_TRIPLETON_SECOND_SECTION_CERTIFICATE": "true",
        "DYNAMIC_TRIPLETON_DIRECTIONAL_IMPROVEMENT_CERTIFICATE": "false",
        "DYNAMIC_TRIPLETON_DIRECTIONAL_RADIUS_RATIO_GT_ONE": "true",
        "C1_DIRECT_MONOTONICITY_CERTIFICATE": "false",
        "C2_MEAN_VALUE_MONOTONICITY_CERTIFICATE": "true",
        "ANCHORED_EXIT_FACE_INEQUALITIES_CERTIFICATE": "true",
        "HSET_COORDINATES_CERTIFICATE": "true",
        "ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE": "true",
        "COVERING_DEGREE": "1",
        "COVERING_DEGREE_CERTIFICATE": "true",
        "RETURN_MAP_DETERMINANT_CERTIFICATE": "true",
        "LOCAL_HSET_COVERING_RELATION_CERTIFICATE": "true",
        "CAPD_USED": "true",
        "POINT_FALLBACK_USED": "false",
    }.items():
        require(fields, key, expected)
    for key in (
        "RECURRENT_COVERING_GRAPH_CERTIFICATE",
        "FIBONACCI_COVERING_CERTIFICATE",
        "GLOBAL_HPG_CERTIFICATE",
        "CHAOS_PROVED",
        "CHAOTIC_ATTRACTOR_PROVED",
        "OPEN_PROBLEM_SOLVED",
        "NOVELTY_OR_PRIORITY_CLAIMED",
    ):
        require(fields, key, "false")

    print(f"SCHEMA={SCHEMA}")
    print(f"C2_MEAN_VALUE_DU_DXI_LOWER_Q={c2[0]}")
    print(f"ANCHORED_EXIT_FACE_GAP_Q={gap}")
    print(f"ENTRY_MARGIN_Q={expected_entry_margin}")
    print("EXIT_MARGIN_Q=1")
    print(f"TRIPLETON_TO_BASELINE_RADIUS_RATIO_Q={ratio}")
    print("COVERING_DEGREE=1")
    print("LOCAL_HSET_COVERING_RELATION_CERTIFICATE=true")
    print("CHAOS_PROVED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
