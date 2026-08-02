#!/usr/bin/env python3
"""Verify the bounded V7-B descendant-depth scout receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from cs6_v7b_subdivision_ladder_run import (
    CARRIERS,
    DEPTH_DELTAS,
    EXPECTED_ATTEMPTS,
    PARENTS,
    RESULT_COLUMNS,
    extract_summary,
    node_id,
)


SUMMARY_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "RUN_VALID",
    "ATTEMPTS_COMPLETED",
    "DESCENDANT_PROBE_PASS",
    "DESCENDANT_PROBE_REJECTED",
    "SECTION_RESIDENT_CROSSING_UNAVAILABLE",
    "UNKNOWN_FAILURE",
    "CERTIFICATE_PASS",
    "TARGET_FIRST_CROSSING_RECOVERY_DELTA",
    "TARGET_FIRST_PROBE_PASS_DELTA",
    "ALL_PARENT_CARRIERS_HAVE_CANDIDATE",
    "DESCENDANT_CANDIDATE_DISCOVERED",
    "PARENT_COVER_EVALUATED",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
    "V7_B_ELIGIBILITY",
    "V7_B_WINNER",
    "PROMOTION_ELIGIBLE",
    "OPEN_PROBLEM_SOLVED",
    "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"V7-B subdivision ladder verify error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


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
    receipt_dir = args.receipt_dir
    summary = parse_summary(receipt_dir / "summary.txt")
    rows = list(
        csv.DictReader(canonical(receipt_dir / "results.tsv").splitlines(), delimiter="\t")
    )
    if len(rows) != EXPECTED_ATTEMPTS:
        fail("result row count must be 24")
    if tuple(rows[0]) != RESULT_COLUMNS:
        fail("result column order drifted")
    if summary["SCHEMA"] != "sounio.cs6.v7b-subdivision-ladder-summary.v1":
        fail("summary schema mismatch")
    if summary["RUN_COMPLETE"] != "true" or summary["RUN_VALID"] != "true":
        fail("run is incomplete or invalid")
    if summary["ATTEMPTS_COMPLETED"] != str(EXPECTED_ATTEMPTS):
        fail("attempt count mismatch")

    expected_index = 0
    parent_nodes: dict[str, tuple[int, int, int, int]] = {}
    for row in rows:
        expected_index += 1
        if row["ATTEMPT_INDEX"] != str(expected_index):
            fail("attempt order drifted")
        parent = row["PARENT_V7_ORDINAL"]
        depth_delta = int(row["DEPTH_DELTA"])
        carrier = row["CARRIER"]
        if parent not in PARENTS or depth_delta not in DEPTH_DELTAS or carrier not in CARRIERS:
            fail("attempt matrix value outside frozen domain")
        key = (parent, depth_delta, carrier)
        if sum(
            other["PARENT_V7_ORDINAL"] == key[0]
            and int(other["DEPTH_DELTA"]) == key[1]
            and other["CARRIER"] == key[2]
            for other in rows
        ) != 1:
            fail(f"attempt matrix duplicate or gap: {key}")

        attempt_dir = next(
            receipt_dir.glob(
                f"attempts/A{expected_index:04d}_P{parent}_D{depth_delta}_{carrier}"
            ),
            None,
        )
        if attempt_dir is None:
            fail(f"attempt directory missing: {expected_index}")
        stdout_path = attempt_dir / "stdout.txt"
        stderr_path = attempt_dir / "stderr.txt"
        if row["STDOUT_SHA256"] != sha256(stdout_path):
            fail(f"stdout hash mismatch: {expected_index}")
        if row["STDERR_SHA256"] != sha256(stderr_path):
            fail(f"stderr hash mismatch: {expected_index}")
        summary_sha, worker = extract_summary(stdout_path.read_bytes())
        if row["SUMMARY_SHA256"] != summary_sha:
            fail(f"worker summary hash mismatch: {expected_index}")
        if row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE":
            if b"one-step Newton crossing was not available" not in stderr_path.read_bytes():
                fail(f"crossing classification lacks evidence: {expected_index}")
            if worker:
                fail(f"crossing failure unexpectedly emitted summary: {expected_index}")
        elif row["STATUS"] in {"DESCENDANT_PROBE_PASS", "DESCENDANT_PROBE_REJECTED"}:
            expected_pass = row["STATUS"] == "DESCENDANT_PROBE_PASS"
            if worker.get("PROBE_PASS") != str(expected_pass).lower():
                fail(f"probe classification mismatch: {expected_index}")
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
                    fail(f"worker field mismatch: {field} attempt {expected_index}")
        else:
            fail(f"unaccepted status: {row['STATUS']}")

        command = canonical(attempt_dir / "command.txt").split()
        if len(command) != 13:
            fail(f"command arity mismatch: {expected_index}")
        child = tuple(map(int, (command[1], command[2], command[3], command[4])))
        if row["NODE_ID"] != node_id(*child):
            fail(f"node/command mismatch: {expected_index}")
        if parent not in parent_nodes:
            scale = 1 << depth_delta
            parent_nodes[parent] = (
                child[0] - depth_delta,
                child[1] // scale,
                child[2] - depth_delta,
                child[3] // scale,
            )
        base = parent_nodes[parent]
        expected_child = (
            base[0] + depth_delta,
            base[1] << depth_delta,
            base[2] + depth_delta,
            base[3] << depth_delta,
        )
        if child != expected_child:
            fail(f"not a lower-left descendant: {expected_index}")

    counts = {
        "DESCENDANT_PROBE_PASS": sum(row["STATUS"] == "DESCENDANT_PROBE_PASS" for row in rows),
        "DESCENDANT_PROBE_REJECTED": sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in rows),
        "SECTION_RESIDENT_CROSSING_UNAVAILABLE": sum(
            row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in rows
        ),
        "CERTIFICATE_PASS": sum(row["CERTIFICATE_PASS"] == "true" for row in rows),
    }
    for key, count in counts.items():
        if summary[key] != str(count):
            fail(f"summary count mismatch: {key}")
    if summary["UNKNOWN_FAILURE"] != "0":
        fail("unknown failures are not accepted")
    for key in (
        "PARENT_COVER_EVALUATED",
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
    if summary["DESCENDANT_CANDIDATE_DISCOVERED"] != str(counts["DESCENDANT_PROBE_PASS"] > 0).lower():
        fail("candidate-discovery flag mismatch")

    def target_at(depth_delta: int) -> list[dict[str, str]]:
        return [
            row
            for row in rows
            if row["PARENT_V7_ORDINAL"] == "23"
            and int(row["DEPTH_DELTA"]) == depth_delta
        ]

    crossing_delta = next(
        (
            str(depth_delta)
            for depth_delta in DEPTH_DELTAS
            if all(
                row["STATUS"]
                in {"DESCENDANT_PROBE_REJECTED", "DESCENDANT_PROBE_PASS"}
                for row in target_at(depth_delta)
            )
        ),
        "NONE",
    )
    probe_delta = next(
        (
            str(depth_delta)
            for depth_delta in DEPTH_DELTAS
            if all(row["PROBE_PASS"] == "true" for row in target_at(depth_delta))
        ),
        "NONE",
    )
    if summary["TARGET_FIRST_CROSSING_RECOVERY_DELTA"] != crossing_delta:
        fail("target crossing-recovery delta mismatch")
    if summary["TARGET_FIRST_PROBE_PASS_DELTA"] != probe_delta:
        fail("target probe-pass delta mismatch")
    all_parent_carriers = all(
        any(
            row["PARENT_V7_ORDINAL"] == parent
            and row["CARRIER"] == carrier
            and row["PROBE_PASS"] == "true"
            for row in rows
        )
        for parent in PARENTS
        for carrier in CARRIERS
    )
    if summary["ALL_PARENT_CARRIERS_HAVE_CANDIDATE"] != str(all_parent_carriers).lower():
        fail("all-parent/carrier candidate flag mismatch")

    print("VERIFY_SCHEMA=sounio.cs6.v7b-subdivision-ladder-verification.v1")
    print("RUN_VALID=true")
    print(f"ATTEMPTS_VERIFIED={len(rows)}")
    print(f"DESCENDANT_PROBE_PASS={counts['DESCENDANT_PROBE_PASS']}")
    print("PARENT_COVER_EVALUATED=false")
    print("V7_B_ELIGIBILITY=false")
    print("PROMOTION_ELIGIBLE=false")


if __name__ == "__main__":
    main()
