#!/usr/bin/env python3
"""Verify target-23 depth-5 boundary-refinement receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from cs6_v7b_subdivision_ladder_run import CARRIERS, extract_summary
from cs6_v7b_target23_depth5_boundary_refine_run import (
    COORDINATE_COLUMNS, EXPECTED_ATTEMPTS, EXPECTED_CELLS, RESULT_COLUMNS,
    SOURCE_PASS_CELLS, SOURCE_REJECTED_PARENTS, build_attempts, node_id,
)


SUMMARY_KEYS = (
    "SCHEMA", "RUN_COMPLETE", "RUN_VALID", "SOURCE_DEPTH4_PASS_CELLS",
    "SOURCE_REJECTED_PARENT_CELLS", "GRANDCHILD_CELLS_EVALUATED",
    "ATTEMPTS_COMPLETED", "PROBE_PASS_ATTEMPTS", "PROBE_REJECTED_ATTEMPTS",
    "SECTION_RESIDENT_CROSSING_UNAVAILABLE", "UNKNOWN_FAILURE",
    "CERTIFICATE_PASS_ATTEMPTS", "BOTH_CARRIERS_PROBE_PASS_CELLS",
    "BOTH_CARRIERS_PROBE_REJECT_CELLS", "MIXED_CARRIER_CELLS",
    "CARRIER_STATUS_AGREEMENT_CELLS", "REFINED_PARENTS_FULL_PROBE_COVER",
    "REFINED_PARENTS_WITH_REJECTION", "REFINED_REGION_EVALUATED",
    "REFINED_REGION_PROBE_COVER_PASS", "ADAPTIVE_PARENT_PROBE_COVER_EVALUATED",
    "ADAPTIVE_PARENT_PROBE_COVER_PASS", "ADAPTIVE_COVER_LEAF_CELLS",
    "ADAPTIVE_PARENT_CERTIFICATE_COVER_PASS",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", "V7_B_ELIGIBILITY",
    "V7_B_WINNER", "PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED", "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"V7-B target-23 boundary verify error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def parse_tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(canonical(path).splitlines(), delimiter="\t"))


def parse_summary(path: Path) -> dict[str, str]:
    lines = canonical(path).splitlines()
    if len(lines) != len(SUMMARY_KEYS):
        fail("summary line count mismatch")
    fields = {}
    for line, expected in zip(lines, SUMMARY_KEYS, strict=True):
        if line.count("=") != 1:
            fail("malformed summary line")
        key, value = line.split("=", 1)
        if key != expected or not value:
            fail(f"summary key mismatch: expected {expected}")
        fields[key] = value
    return fields


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_dir", type=Path)
    args = parser.parse_args()
    root, receipt = Path.cwd(), args.receipt_dir
    summary = parse_summary(receipt / "summary.txt")
    coordinates = parse_tsv(receipt / "coordinate-manifest.tsv")
    results = parse_tsv(receipt / "results.tsv")
    expected = build_attempts(root)
    if len(coordinates) != EXPECTED_CELLS or len(results) != EXPECTED_ATTEMPTS:
        fail("coordinate or result cardinality mismatch")
    if tuple(coordinates[0]) != COORDINATE_COLUMNS or tuple(results[0]) != RESULT_COLUMNS:
        fail("column order drifted")
    if summary["SCHEMA"] != "sounio.cs6.v7b-target23-depth5-boundary-refine-summary.v1":
        fail("summary schema drifted")
    if summary["RUN_COMPLETE"] != "true" or summary["RUN_VALID"] != "true":
        fail("run is incomplete or invalid")

    coordinate_nodes = set()
    source_children: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for index, row in enumerate(coordinates, 1):
        planned = expected[(index - 1) * len(CARRIERS)]
        if row["CELL_INDEX"] != str(index):
            fail("coordinate order drifted")
        fields = {
            "SOURCE_CELL_INDEX": str(planned.source.cell_index),
            "SOURCE_U_OFFSET": str(planned.source.u_offset),
            "SOURCE_S_OFFSET": str(planned.source.s_offset),
            "SUB_U_OFFSET": str(planned.sub_u_offset),
            "SUB_S_OFFSET": str(planned.sub_s_offset),
            "NODE_ID": planned.node, "U_DEPTH": str(planned.u_depth),
            "U_INDEX": str(planned.u_index), "S_DEPTH": str(planned.s_depth),
            "S_INDEX": str(planned.s_index), "INPUT_SHA256": planned.input_sha256,
        }
        for key, value in fields.items():
            if row[key] != value:
                fail(f"coordinate {index} field mismatch: {key}")
        if row["NODE_ID"] in coordinate_nodes:
            fail("duplicate grandchild node")
        coordinate_nodes.add(row["NODE_ID"])
        source_children[planned.source.cell_index].add((planned.sub_u_offset, planned.sub_s_offset))
    if len(source_children) != SOURCE_REJECTED_PARENTS:
        fail("not every rejected parent was refined")
    if any(children != {(0, 0), (0, 1), (1, 0), (1, 1)} for children in source_children.values()):
        fail("grandchild partition incomplete")

    for index, (row, planned) in enumerate(zip(results, expected, strict=True), 1):
        fields = {
            "ATTEMPT_INDEX": str(index), "SOURCE_CELL_INDEX": str(planned.source.cell_index),
            "SOURCE_U_OFFSET": str(planned.source.u_offset),
            "SOURCE_S_OFFSET": str(planned.source.s_offset),
            "SUB_U_OFFSET": str(planned.sub_u_offset), "SUB_S_OFFSET": str(planned.sub_s_offset),
            "ORIGINAL_PARENT_DEPTH_DELTA": "5", "NODE_ID": planned.node,
            "CARRIER": planned.carrier,
            "INPUT_SHA256": planned.input_sha256, "RUN_CHALLENGE": planned.run_challenge,
            "ATTEMPT_BINDING": planned.attempt_binding,
        }
        for key, value in fields.items():
            if row[key] != value:
                fail(f"attempt {index} field mismatch: {key}")
        attempt_dir = receipt / "attempts" / planned.identity
        stdout_path, stderr_path = attempt_dir / "stdout.txt", attempt_dir / "stderr.txt"
        command_path = attempt_dir / "command.txt"
        if not all(path.is_file() for path in (stdout_path, stderr_path, command_path)):
            fail(f"attempt evidence missing: {index}")
        if row["STDOUT_SHA256"] != sha256(stdout_path) or row["STDERR_SHA256"] != sha256(stderr_path):
            fail(f"attempt output hash mismatch: {index}")
        summary_sha, worker = extract_summary(stdout_path.read_bytes())
        if row["SUMMARY_SHA256"] != summary_sha:
            fail(f"worker summary hash mismatch: {index}")
        if row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE":
            if b"one-step Newton crossing was not available" not in stderr_path.read_bytes() or worker:
                fail(f"crossing evidence mismatch: {index}")
        elif row["STATUS"] in {"DESCENDANT_PROBE_PASS", "DESCENDANT_PROBE_REJECTED"}:
            expected_pass = str(row["STATUS"] == "DESCENDANT_PROBE_PASS").lower()
            if worker.get("PROBE_PASS") != expected_pass:
                fail(f"probe status mismatch: {index}")
            for key in (
                "C1_ORIENTATION_UNRESOLVED", "C2_HULL_ORIENTATION_UNRESOLVED",
                "EVENT1_CHARTS_CERTIFIED", "EVENT2_CHARTS_CERTIFIED",
                "HOMOGENEOUS_COMPUTATION_VALID", "CERTIFICATE_PASS", "PROBE_PASS",
            ):
                if row[key] != worker.get(key):
                    fail(f"worker field mismatch: attempt {index} field {key}")
        else:
            fail(f"unaccepted status: {row['STATUS']}")
        command = canonical(command_path).split()
        if len(command) != 13 or node_id(*map(int, command[1:5])) != planned.node:
            fail(f"command coordinate mismatch: {index}")
        if command[7] != planned.carrier:
            fail(f"command carrier mismatch: {index}")

    pairs = [results[i:i + len(CARRIERS)] for i in range(0, len(results), len(CARRIERS))]
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in results:
        by_source[row["SOURCE_CELL_INDEX"]].append(row)
    counts = {
        "SOURCE_DEPTH4_PASS_CELLS": SOURCE_PASS_CELLS,
        "SOURCE_REJECTED_PARENT_CELLS": len(by_source),
        "GRANDCHILD_CELLS_EVALUATED": len(pairs), "ATTEMPTS_COMPLETED": len(results),
        "PROBE_PASS_ATTEMPTS": sum(row["PROBE_PASS"] == "true" for row in results),
        "PROBE_REJECTED_ATTEMPTS": sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in results),
        "SECTION_RESIDENT_CROSSING_UNAVAILABLE": sum(row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in results),
        "UNKNOWN_FAILURE": sum(row["STATUS"] == "UNKNOWN_FAILURE" for row in results),
        "CERTIFICATE_PASS_ATTEMPTS": sum(row["CERTIFICATE_PASS"] == "true" for row in results),
        "BOTH_CARRIERS_PROBE_PASS_CELLS": sum(all(row["PROBE_PASS"] == "true" for row in pair) for pair in pairs),
        "BOTH_CARRIERS_PROBE_REJECT_CELLS": sum(all(row["PROBE_PASS"] == "false" for row in pair) for pair in pairs),
        "MIXED_CARRIER_CELLS": sum(len({row["PROBE_PASS"] for row in pair}) != 1 for pair in pairs),
        "CARRIER_STATUS_AGREEMENT_CELLS": sum(len({row["STATUS"] for row in pair}) == 1 for pair in pairs),
        "REFINED_PARENTS_FULL_PROBE_COVER": sum(all(row["PROBE_PASS"] == "true" for row in group) for group in by_source.values()),
        "REFINED_PARENTS_WITH_REJECTION": sum(any(row["PROBE_PASS"] != "true" for row in group) for group in by_source.values()),
    }
    for key, count in counts.items():
        if summary[key] != str(count):
            fail(f"summary count mismatch: {key}")
    refined_cover = counts["PROBE_PASS_ATTEMPTS"] == EXPECTED_ATTEMPTS
    for key in ("REFINED_REGION_EVALUATED", "ADAPTIVE_PARENT_PROBE_COVER_EVALUATED"):
        if summary[key] != "true":
            fail(f"evaluation flag missing: {key}")
    for key in ("REFINED_REGION_PROBE_COVER_PASS", "ADAPTIVE_PARENT_PROBE_COVER_PASS"):
        if summary[key] != str(refined_cover).lower():
            fail(f"probe-cover verdict mismatch: {key}")
    if summary["ADAPTIVE_COVER_LEAF_CELLS"] != str(SOURCE_PASS_CELLS + EXPECTED_CELLS):
        fail("adaptive leaf count mismatch")
    for key in (
        "ADAPTIVE_PARENT_CERTIFICATE_COVER_PASS", "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
        "V7_B_ELIGIBILITY", "PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED", "FPGA_EXECUTION",
    ):
        if summary[key] != "false":
            fail(f"forbidden claim enabled: {key}")
    if summary["V7_B_WINNER"] != "NONE":
        fail("winner must remain NONE")

    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-depth5-boundary-refine-verification.v1")
    print("RUN_VALID=true")
    print(f"SOURCE_REJECTED_PARENTS_VERIFIED={SOURCE_REJECTED_PARENTS}")
    print(f"GRANDCHILD_CELLS_VERIFIED={EXPECTED_CELLS}")
    print(f"ATTEMPTS_VERIFIED={EXPECTED_ATTEMPTS}")
    print(f"PROBE_PASS_ATTEMPTS={counts['PROBE_PASS_ATTEMPTS']}")
    print(f"REFINED_REGION_PROBE_COVER_PASS={str(refined_cover).lower()}")
    print(f"ADAPTIVE_PARENT_PROBE_COVER_PASS={str(refined_cover).lower()}")
    print("ADAPTIVE_PARENT_CERTIFICATE_COVER_PASS=false")
    print("V7_B_ELIGIBILITY=false")


if __name__ == "__main__":
    main()
