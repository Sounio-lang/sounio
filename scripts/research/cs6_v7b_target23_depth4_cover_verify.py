#!/usr/bin/env python3
"""Verify the complete target-23 depth-4 sibling-cover receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from cs6_v7b_subdivision_ladder_run import CARRIERS, extract_summary
from cs6_v7b_target23_depth4_cover_run import (
    COORDINATE_COLUMNS,
    DEPTH_DELTA,
    EXPECTED_ATTEMPTS,
    EXPECTED_CELLS,
    OFFSET_DOMAIN,
    RESULT_COLUMNS,
    build_attempts,
    node_id,
)


SUMMARY_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "RUN_VALID",
    "CHILD_CELLS_EVALUATED",
    "ATTEMPTS_COMPLETED",
    "PROBE_PASS_ATTEMPTS",
    "PROBE_REJECTED_ATTEMPTS",
    "SECTION_RESIDENT_CROSSING_UNAVAILABLE",
    "UNKNOWN_FAILURE",
    "CERTIFICATE_PASS_ATTEMPTS",
    "BOTH_CARRIERS_PROBE_PASS_CELLS",
    "BOTH_CARRIERS_PROBE_REJECT_CELLS",
    "MIXED_CARRIER_CELLS",
    "CARRIER_STATUS_AGREEMENT_CELLS",
    "PARENT_COVER_EVALUATED",
    "PARENT_PROBE_COVER_PASS",
    "PARENT_CERTIFICATE_COVER_PASS",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
    "V7_B_ELIGIBILITY",
    "V7_B_WINNER",
    "PROMOTION_ELIGIBLE",
    "OPEN_PROBLEM_SOLVED",
    "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"V7-B target-23 cover verify error: {message}")


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
    fields: dict[str, str] = {}
    for line, expected in zip(lines, SUMMARY_KEYS, strict=True):
        if line.count("=") != 1:
            fail(f"malformed summary line: {line}")
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
    root = Path.cwd()
    receipt_dir = args.receipt_dir
    summary = parse_summary(receipt_dir / "summary.txt")
    coordinates = parse_tsv(receipt_dir / "coordinate-manifest.tsv")
    results = parse_tsv(receipt_dir / "results.tsv")
    expected = build_attempts(root)
    if len(coordinates) != EXPECTED_CELLS or len(results) != EXPECTED_ATTEMPTS:
        fail("coordinate or result row count mismatch")
    if tuple(coordinates[0]) != COORDINATE_COLUMNS:
        fail("coordinate column order drifted")
    if tuple(results[0]) != RESULT_COLUMNS:
        fail("result column order drifted")
    if summary["SCHEMA"] != "sounio.cs6.v7b-target23-depth4-cover-summary.v1":
        fail("summary schema mismatch")
    if summary["RUN_COMPLETE"] != "true" or summary["RUN_VALID"] != "true":
        fail("run is incomplete or invalid")

    coordinate_keys: set[tuple[int, int]] = set()
    for index, row in enumerate(coordinates, 1):
        key = (int(row["CHILD_U_OFFSET"]), int(row["CHILD_S_OFFSET"]))
        if key in coordinate_keys or key[0] not in OFFSET_DOMAIN or key[1] not in OFFSET_DOMAIN:
            fail(f"coordinate duplicate or outside partition: {key}")
        coordinate_keys.add(key)
        if row["CELL_INDEX"] != str(index):
            fail("coordinate order drifted")
        planned = expected[(index - 1) * len(CARRIERS)]
        expected_fields = {
            "NODE_ID": planned.node,
            "U_DEPTH": str(planned.u_depth),
            "U_INDEX": str(planned.u_index),
            "S_DEPTH": str(planned.s_depth),
            "S_INDEX": str(planned.s_index),
            "INPUT_SHA256": planned.input_sha256,
        }
        for field, value in expected_fields.items():
            if row[field] != value:
                fail(f"coordinate mismatch: row {index} field {field}")
    expected_keys = {(u, s) for u in OFFSET_DOMAIN for s in OFFSET_DOMAIN}
    if coordinate_keys != expected_keys:
        fail("coordinate partition is incomplete")

    for index, (row, planned) in enumerate(zip(results, expected, strict=True), 1):
        if row["ATTEMPT_INDEX"] != str(index):
            fail("attempt order drifted")
        expected_fields = {
            "PARENT_V7_ORDINAL": "23",
            "CHECKPOINT_ROLE": "MASKED_TARGET",
            "DEPTH_DELTA": str(DEPTH_DELTA),
            "CHILD_U_OFFSET": str(planned.child_u_offset),
            "CHILD_S_OFFSET": str(planned.child_s_offset),
            "NODE_ID": planned.node,
            "CARRIER": planned.carrier,
            "INPUT_SHA256": planned.input_sha256,
            "RUN_CHALLENGE": planned.run_challenge,
            "ATTEMPT_BINDING": planned.attempt_binding,
        }
        for field, value in expected_fields.items():
            if row[field] != value:
                fail(f"attempt mismatch: row {index} field {field}")
        attempt_dir = receipt_dir / "attempts" / planned.identity
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"
        command_path = attempt_dir / "command.txt"
        if not stdout_path.is_file() or not stderr_path.is_file() or not command_path.is_file():
            fail(f"attempt evidence missing: {index}")
        if row["STDOUT_SHA256"] != sha256(stdout_path):
            fail(f"stdout hash mismatch: {index}")
        if row["STDERR_SHA256"] != sha256(stderr_path):
            fail(f"stderr hash mismatch: {index}")
        summary_sha, worker = extract_summary(stdout_path.read_bytes())
        if row["SUMMARY_SHA256"] != summary_sha:
            fail(f"worker summary hash mismatch: {index}")
        status = row["STATUS"]
        if status == "SECTION_RESIDENT_CROSSING_UNAVAILABLE":
            if b"one-step Newton crossing was not available" not in stderr_path.read_bytes():
                fail(f"crossing classification lacks evidence: {index}")
            if worker:
                fail(f"crossing failure emitted a summary: {index}")
        elif status in {"DESCENDANT_PROBE_PASS", "DESCENDANT_PROBE_REJECTED"}:
            expected_pass = status == "DESCENDANT_PROBE_PASS"
            if worker.get("PROBE_PASS") != str(expected_pass).lower():
                fail(f"probe classification mismatch: {index}")
            for field in (
                "C1_ORIENTATION_UNRESOLVED",
                "C2_HULL_ORIENTATION_UNRESOLVED",
                "EVENT1_CHARTS_CERTIFIED",
                "EVENT2_CHARTS_CERTIFIED",
                "HOMOGENEOUS_COMPUTATION_VALID",
                "CERTIFICATE_PASS",
                "PROBE_PASS",
            ):
                if row[field] != worker.get(field):
                    fail(f"worker field mismatch: row {index} field {field}")
        else:
            fail(f"unaccepted result status: {status}")
        command = canonical(command_path).split()
        if len(command) != 13:
            fail(f"command arity mismatch: {index}")
        child = tuple(map(int, command[1:5]))
        if node_id(*child) != planned.node or command[7] != planned.carrier:
            fail(f"command coordinate/carrier mismatch: {index}")

    pairs = [results[index : index + len(CARRIERS)] for index in range(0, len(results), len(CARRIERS))]
    counts = {
        "PROBE_PASS_ATTEMPTS": sum(row["PROBE_PASS"] == "true" for row in results),
        "PROBE_REJECTED_ATTEMPTS": sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in results),
        "SECTION_RESIDENT_CROSSING_UNAVAILABLE": sum(row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in results),
        "CERTIFICATE_PASS_ATTEMPTS": sum(row["CERTIFICATE_PASS"] == "true" for row in results),
        "BOTH_CARRIERS_PROBE_PASS_CELLS": sum(all(row["PROBE_PASS"] == "true" for row in pair) for pair in pairs),
        "BOTH_CARRIERS_PROBE_REJECT_CELLS": sum(all(row["PROBE_PASS"] == "false" for row in pair) for pair in pairs),
        "MIXED_CARRIER_CELLS": sum(len({row["PROBE_PASS"] for row in pair}) != 1 for pair in pairs),
        "CARRIER_STATUS_AGREEMENT_CELLS": sum(len({row["STATUS"] for row in pair}) == 1 for pair in pairs),
    }
    for key, count in counts.items():
        if summary[key] != str(count):
            fail(f"summary count mismatch: {key}")
    if summary["CHILD_CELLS_EVALUATED"] != str(EXPECTED_CELLS):
        fail("evaluated cell count mismatch")
    if summary["ATTEMPTS_COMPLETED"] != str(EXPECTED_ATTEMPTS):
        fail("attempt count mismatch")
    if summary["UNKNOWN_FAILURE"] != "0":
        fail("unknown failures are forbidden")
    if summary["PARENT_COVER_EVALUATED"] != "true":
        fail("complete partition was not marked evaluated")
    expected_probe_cover = str(counts["PROBE_PASS_ATTEMPTS"] == EXPECTED_ATTEMPTS).lower()
    expected_certificate_cover = str(counts["CERTIFICATE_PASS_ATTEMPTS"] == EXPECTED_ATTEMPTS).lower()
    if summary["PARENT_PROBE_COVER_PASS"] != expected_probe_cover:
        fail("parent probe-cover verdict mismatch")
    if summary["PARENT_CERTIFICATE_COVER_PASS"] != expected_certificate_cover:
        fail("parent certificate-cover verdict mismatch")
    for key in (
        "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
        "V7_B_ELIGIBILITY",
        "PROMOTION_ELIGIBLE",
        "OPEN_PROBLEM_SOLVED",
        "FPGA_EXECUTION",
    ):
        if summary[key] != "false":
            fail(f"forbidden claim enabled: {key}")
    if summary["V7_B_WINNER"] != "NONE":
        fail("winner must remain NONE")

    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-depth4-cover-verification.v1")
    print("RUN_VALID=true")
    print(f"CHILD_CELLS_VERIFIED={EXPECTED_CELLS}")
    print(f"ATTEMPTS_VERIFIED={EXPECTED_ATTEMPTS}")
    print(f"PROBE_PASS_ATTEMPTS={counts['PROBE_PASS_ATTEMPTS']}")
    print(f"CERTIFICATE_PASS_ATTEMPTS={counts['CERTIFICATE_PASS_ATTEMPTS']}")
    print(f"PARENT_PROBE_COVER_PASS={expected_probe_cover}")
    print(f"PARENT_CERTIFICATE_COVER_PASS={expected_certificate_cover}")
    print("V7_B_ELIGIBILITY=false")


if __name__ == "__main__":
    main()
