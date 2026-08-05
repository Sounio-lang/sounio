#!/usr/bin/env python3
"""Fail-closed verifier for the bounded local h-set covering candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SUPPORT_FILES = {
    "XLEL": "support_XLEL.json",
    "XLEH": "support_XLEH.json",
    "XHEL": "support_XHEL.json",
    "XHEH": "support_XHEH.json",
}
FACE_FILES = (
    "face_LEFT_XLEL_L.json",
    "face_LEFT_XLEL_H.json",
    "face_LEFT_XLEH_ROOT.json",
    "face_RIGHT_XHEL_ROOT.json",
    "face_RIGHT_XHEH_L.json",
    "face_RIGHT_XHEH_H.json",
)
SUPPORT_SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-carrier.v1"
FACE_SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-hset-exit-face.v1"
AGGREGATE_SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-local-hset-covering.v1"


def fail(message: str) -> None:
    raise SystemExit(f"h-set covering verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if not raw or "=" not in raw:
            fail(f"invalid aggregate line {line_number}")
        key, value = raw.split("=", 1)
        if key in result:
            fail(f"duplicate aggregate field {key}")
        result[key] = value
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


def verify_exact_section(carrier: dict[str, object]) -> None:
    components = carrier.get("components")
    if not isinstance(components, list) or len(components) != 4:
        fail("serialized carrier does not have four components")
    w = components[2]
    if not isinstance(w, dict) or w.get("remainder") != ["0", "0"]:
        fail("serialized carrier w remainder is not exact zero")
    for coefficient in w.get("coefficients", []):
        if coefficient.get("interval") != ["0", "0"]:
            fail("serialized carrier w coefficient is not exact zero")


def load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.name}: {error}")
    if not isinstance(value, dict):
        fail(f"{path.name} is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    receipts = args.receipts
    aggregate_path = receipts / "aggregate.txt"
    fields = parse_fields(aggregate_path)
    require(fields, "SCHEMA", AGGREGATE_SCHEMA)
    require(fields, "LEAF_ID", "U08-0000000223_S09-0000000325")
    require(fields, "MAP", "P2")
    require(fields, "SECTION", "w=0")
    require(fields, "SOURCE_TILES", "XLEL,XLEH,XHEL,XHEH")
    require(fields, "SOURCE_TILE_COUNT", "4")
    require(fields, "SECOND_EVENT_CARRIER_COUNT", "5")
    require(fields, "FACE_RECEIPT_COUNT", "6")

    script_dir = Path(__file__).resolve().parent
    require(
        fields,
        "ANALYZER_SOURCE_SHA256",
        sha256(script_dir / "cs6_v7b_target23_arb_tm2r_hset_covering_analyze.py"),
    )
    support_worker_hash = sha256(
        script_dir / "cs6_v7b_target23_arb_tm2r_hset_covering_carrier_worker.py"
    )
    face_worker_hash = sha256(
        script_dir / "cs6_v7b_target23_arb_tm2r_hset_covering_face_worker.py"
    )
    require(fields, "CARRIER_WORKER_SOURCE_SHA256", support_worker_hash)
    require(fields, "FACE_WORKER_SOURCE_SHA256", face_worker_hash)

    support_count = 0
    for tile_id, filename in SUPPORT_FILES.items():
        path = receipts / filename
        require(fields, f"CARRIER_RECEIPT_SHA256_{tile_id}", sha256(path))
        raw = load_json(path)
        if raw.get("schema") != SUPPORT_SCHEMA or raw.get("tile_id") != tile_id:
            fail(f"wrong support identity in {filename}")
        if raw.get("worker_source_sha256") != support_worker_hash:
            fail(f"wrong support worker hash in {filename}")
        if raw.get("selected_source_chain_certificate") is not True:
            fail(f"missing chain certificate in {filename}")
        if raw.get("point_fallback_used") is not False:
            fail(f"point fallback in {filename}")
        carriers = raw.get("carriers")
        if not isinstance(carriers, list) or not carriers:
            fail(f"no support carrier in {filename}")
        support_count += len(carriers)
        for carrier in carriers:
            verify_exact_section(carrier)
            if Fraction(carrier["event_normal"][0]) <= 0:
                fail(f"non-positive support normal in {filename}")
    if support_count != 5:
        fail(f"support carrier count is {support_count}, expected 5")

    face_ids = {
        fields[f"FACE_RECEIPT_ID_{index}"]: fields[f"FACE_RECEIPT_SHA256_{index}"]
        for index in range(1, 7)
    }
    eta_cover = {"LEFT": [], "RIGHT": []}
    observed_files: set[str] = set()
    for filename in FACE_FILES:
        path = receipts / filename
        raw = load_json(path)
        face = str(raw.get("source_face"))
        tile = str(raw.get("tile_id"))
        refinement = str(raw.get("eta_refinement"))
        receipt_id = f"{face}_{tile}_{refinement}"
        if face_ids.get(receipt_id) != sha256(path):
            fail(f"face hash binding failed for {filename}")
        if raw.get("schema") != FACE_SCHEMA:
            fail(f"wrong face schema in {filename}")
        if raw.get("worker_source_sha256") != face_worker_hash:
            fail(f"wrong face worker hash in {filename}")
        if raw.get("support_worker_source_sha256") != support_worker_hash:
            fail(f"face does not bind support worker in {filename}")
        if raw.get("selected_source_face_chain_certificate") is not True:
            fail(f"missing face chain certificate in {filename}")
        if raw.get("point_fallback_used") is not False:
            fail(f"point fallback in {filename}")
        domain = raw["source_domain"]
        expected_xi = Fraction(-1) if face == "LEFT" else Fraction(1)
        if tuple(map(Fraction, domain["xi"])) != (expected_xi, expected_xi):
            fail(f"wrong exact xi face in {filename}")
        eta_cover[face].append(tuple(map(Fraction, domain["eta"])))
        carriers = raw.get("carriers")
        if not isinstance(carriers, list) or not carriers:
            fail(f"no face carrier in {filename}")
        for carrier in carriers:
            verify_exact_section(carrier)
            if Fraction(carrier["event_normal"][0]) <= 0:
                fail(f"non-positive face normal in {filename}")
        observed_files.add(filename)
    if observed_files != set(FACE_FILES):
        fail("face file set mismatch")
    for face, intervals in eta_cover.items():
        ordered = sorted(intervals)
        if ordered[0][0] != -1 or ordered[-1][1] != 1:
            fail(f"{face} eta face does not cover [-1,1]")
        if any(a[1] != b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
            fail(f"{face} eta face has a gap or overlap")

    if fraction(fields, "SOURCE_U_RADIUS_Q") <= 0:
        fail("source unstable radius is non-positive")
    if fraction(fields, "SOURCE_S_RADIUS_Q") <= 0:
        fail("source stable radius is non-positive")
    if fraction(fields, "TARGET_U_RADIUS_Q") <= 0:
        fail("target unstable radius is non-positive")
    if fraction(fields, "TARGET_S_RADIUS_Q") <= 0:
        fail("target stable radius is non-positive")
    frame = bounds(fields, "FRAME_DETERMINANT")
    target_linear = bounds(fields, "TARGET_LINEAR_DETERMINANT")
    source_chart = bounds(fields, "SOURCE_CHART_DETERMINANT")
    target_chart = bounds(fields, "TARGET_CHART_DETERMINANT")
    if frame[1] >= 0 or source_chart[1] >= 0:
        fail("source chart orientation is not strictly negative")
    if target_linear[0] <= 0 or target_chart[0] <= 0:
        fail("target chart is not strictly invertible and positive")

    stable = bounds(fields, "SUPPORT_STABLE_IMAGE")
    if stable[0] <= -1 or stable[1] >= 1:
        fail("support meets target entry boundary")
    expected_entry_margin = min(stable[0] + 1, 1 - stable[1])
    if fraction(fields, "ENTRY_MARGIN_Q") != expected_entry_margin or expected_entry_margin <= 0:
        fail("entry margin mismatch")
    require(fields, "ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE", "true")

    left = bounds(fields, "LEFT_EXIT_UNSTABLE_IMAGE")
    right = bounds(fields, "RIGHT_EXIT_UNSTABLE_IMAGE")
    require(fields, "COVERING_DEGREE_CANDIDATE", "1")
    expected_exit_margin = min(-1 - left[1], right[0] - 1)
    if fraction(fields, "EXIT_MARGIN_Q") != expected_exit_margin or expected_exit_margin >= 0:
        fail("negative exit margin was not retained exactly")
    signed_gap = fraction(fields, "SIGNED_EXIT_FACE_GAP_Q")
    overlap = fraction(fields, "EXIT_FACE_OVERLAP_Q")
    if signed_gap >= 0 or overlap != -signed_gap or overlap <= 0:
        fail("strictly positive face overlap was not retained exactly")
    require(fields, "EXIT_FACE_INEQUALITIES_CERTIFICATE", "false")
    require(fields, "COVERING_DEGREE_CERTIFICATE", "false")

    initial_normal = bounds(fields, "INITIAL_NORMAL")
    physical_det = bounds(fields, "PHYSICAL_RETURN_DETERMINANT")
    normalized_det = bounds(fields, "NORMALIZED_RETURN_DETERMINANT")
    if initial_normal[0] <= 0:
        fail("initial normal is not strictly positive")
    if physical_det[1] >= 0 or normalized_det[1] >= 0:
        fail("return determinant is not strictly negative")
    require(fields, "RETURN_MAP_DETERMINANT_CERTIFICATE", "true")
    require(fields, "HSET_COORDINATES_CERTIFICATE", "true")
    require(fields, "LOCAL_HSET_COVERING_RELATION_CERTIFICATE", "false")
    require(fields, "COVERING_CANDIDATE_FALSIFIED_BY", "EXIT_FACE_OVERLAP")
    for key in (
        "RECURRENT_COVERING_GRAPH_CERTIFICATE",
        "FIBONACCI_COVERING_CERTIFICATE",
        "GLOBAL_HPG_CERTIFICATE",
        "CHAOS_PROVED",
        "CHAOTIC_ATTRACTOR_PROVED",
        "OPEN_PROBLEM_SOLVED",
        "NOVELTY_OR_PRIORITY_CLAIMED",
        "CAPD_USED",
        "POINT_FALLBACK_USED",
    ):
        require(fields, key, "false")

    print(f"SCHEMA={AGGREGATE_SCHEMA}")
    print("SOURCE_TILE_COUNT=4")
    print("SECOND_EVENT_CARRIER_COUNT=5")
    print("FACE_RECEIPT_COUNT=6")
    print(f"ENTRY_MARGIN_Q={expected_entry_margin}")
    print(f"EXIT_MARGIN_Q={expected_exit_margin}")
    print(f"EXIT_FACE_OVERLAP_Q={overlap}")
    print(f"NORMALIZED_RETURN_DETERMINANT_UPPER_Q={normalized_det[1]}")
    print("HSET_COORDINATES_CERTIFICATE=true")
    print("ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE=true")
    print("EXIT_FACE_INEQUALITIES_CERTIFICATE=false")
    print("RETURN_MAP_DETERMINANT_CERTIFICATE=true")
    print("LOCAL_HSET_COVERING_RELATION_CERTIFICATE=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
