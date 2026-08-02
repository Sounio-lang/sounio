#!/usr/bin/env python3
"""Verify a V7-B bridge execution receipt."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_SUMMARY_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "RUN_VALID",
    "ATTEMPTS_COMPLETED",
    "FULL_BRIDGE_PROBE_PASS",
    "SECTION_RESIDENT_CROSSING_UNAVAILABLE",
    "UNKNOWN_FAILURE",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
    "FULL_HPG_PIPELINE_EVALUATED",
    "V7_B_ELIGIBILITY",
    "V7_B_WINNER",
    "PROMOTION_ELIGIBLE",
    "OPEN_PROBLEM_SOLVED",
    "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"V7-B execution verify error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def parse_kv(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line, expected in zip(canonical(path).splitlines(), EXPECTED_SUMMARY_KEYS, strict=True):
        if line.count("=") != 1:
            fail(f"malformed summary line: {line}")
        key, value = line.split("=", 1)
        if key != expected or not value or key in fields:
            fail(f"summary key mismatch: {expected}")
        fields[key] = value
    return fields


def parse_results(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(canonical(path).splitlines(), delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt_dir", type=Path)
    args = parser.parse_args()
    summary = parse_kv(args.receipt_dir / "summary.txt")
    results = parse_results(args.receipt_dir / "results.tsv")
    if len(results) != 6:
        fail("result row count must be six")
    if summary["SCHEMA"] != "sounio.cs6.v7b-full-hpg-bridge-execution-summary.v1":
        fail("summary schema mismatch")
    if summary["RUN_COMPLETE"] != "true" or summary["RUN_VALID"] != "true":
        fail("run is incomplete or invalid")
    if summary["ATTEMPTS_COMPLETED"] != "6":
        fail("attempt count mismatch")
    if summary["UNKNOWN_FAILURE"] != "0":
        fail("unknown failures are not accepted")
    for key in ("PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED", "FPGA_EXECUTION"):
        if summary[key] != "false":
            fail(f"forbidden claim enabled: {key}")
    if summary["V7_B_WINNER"] != "NONE":
        fail("winner must remain NONE in this verifier")

    passes = sum(row["STATUS"] == "FULL_BRIDGE_PROBE_PASS" for row in results)
    negatives = sum(row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in results)
    if int(summary["FULL_BRIDGE_PROBE_PASS"]) != passes:
        fail("pass count mismatch")
    if int(summary["SECTION_RESIDENT_CROSSING_UNAVAILABLE"]) != negatives:
        fail("classified negative count mismatch")
    eligibility = passes == 6
    expected_bool = str(eligibility).lower()
    for key in (
        "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
        "FULL_HPG_PIPELINE_EVALUATED",
        "V7_B_ELIGIBILITY",
    ):
        if summary[key] != expected_bool:
            fail(f"eligibility-dependent field mismatch: {key}")
    if eligibility:
        fail("unexpected V7-B eligibility: winner scoring is intentionally not implemented in this checkpoint")

    print("VERIFY_SCHEMA=sounio.cs6.v7b-full-hpg-bridge-execution-verification.v1")
    print("RUN_VALID=true")
    print(f"ATTEMPTS_VERIFIED={len(results)}")
    print(f"CLASSIFIED_NEGATIVES={negatives}")
    print("V7_B_ELIGIBILITY=false")
    print("PROMOTION_ELIGIBLE=false")


if __name__ == "__main__":
    main()
