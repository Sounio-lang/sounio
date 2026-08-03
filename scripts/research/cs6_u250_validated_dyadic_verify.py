#!/usr/bin/env python3
"""Independently verify S1.I31.F96 vectors and the retained HLS CSim transcript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from fractions import Fraction
from pathlib import Path


BITS = 128
FRAC_BITS = 96
ONE = 1 << FRAC_BITS
DOMAIN_LIMIT = 1 << 111
DIVISORS = (2, 3, 6, 41)
FALSE_CLAIMS = (
    "FPGA_EXECUTION",
    "DYADIC_ARITHMETIC_CERTIFICATE",
    "PICARD_STEP_CERTIFICATE",
    "NOVELTY_OR_PRIORITY_CLAIMED",
    "OPEN_PROBLEM_SOLVED",
)


def fail(message: str) -> None:
    raise ValueError(f"validated dyadic verify error: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kv(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        fail(f"missing regular file: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.count("=") != 1:
            fail(f"malformed key-value line in {path.name}")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or key in fields:
            fail(f"invalid key-value field in {path.name}: {key}")
        fields[key] = value
    return fields


def decode_words(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 16:
        fail(f"misaligned binary: {path.name}")
    return [int.from_bytes(raw[offset:offset + 16], "little", signed=True) for offset in range(0, len(raw), 16)]


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def operation_status(a_lo: int, a_hi: int, b_lo: int, b_hi: int, divisor: int) -> int:
    if a_lo > a_hi or b_lo > b_hi:
        return -1
    if any(not -DOMAIN_LIMIT < value < DOMAIN_LIMIT for value in (a_lo, a_hi, b_lo, b_hi)):
        return -2
    if divisor not in DIVISORS:
        return -3
    return 1


def exact_outputs(a_lo: int, a_hi: int, b_lo: int, b_hi: int, divisor: int) -> tuple[int, ...]:
    status = operation_status(a_lo, a_hi, b_lo, b_hi, divisor)
    if status != 1:
        return (0, 0, 0, 0, 0, 0, 0, 0, status)
    corners = tuple(Fraction(a, ONE) * Fraction(b, ONE) for a in (a_lo, a_hi) for b in (b_lo, b_hi))
    result = (
        a_lo + b_lo,
        a_hi + b_hi,
        a_lo - b_hi,
        a_hi - b_lo,
        floor_fraction(min(corners) * ONE),
        ceil_fraction(max(corners) * ONE),
        floor_fraction(Fraction(a_lo, divisor)),
        ceil_fraction(Fraction(a_hi, divisor)),
        status,
    )
    exact_ranges = (
        (Fraction(a_lo + b_lo, ONE), Fraction(result[0], ONE), Fraction(result[1], ONE)),
        (Fraction(a_hi + b_hi, ONE), Fraction(result[0], ONE), Fraction(result[1], ONE)),
        (Fraction(a_lo - b_hi, ONE), Fraction(result[2], ONE), Fraction(result[3], ONE)),
        (Fraction(a_hi - b_lo, ONE), Fraction(result[2], ONE), Fraction(result[3], ONE)),
        (min(corners), Fraction(result[4], ONE), Fraction(result[5], ONE)),
        (max(corners), Fraction(result[4], ONE), Fraction(result[5], ONE)),
        (Fraction(a_lo, ONE * divisor), Fraction(result[6], ONE), Fraction(result[7], ONE)),
        (Fraction(a_hi, ONE * divisor), Fraction(result[6], ONE), Fraction(result[7], ONE)),
    )
    if any(not lower <= value <= upper for value, lower, upper in exact_ranges):
        fail("internal independent containment reconstruction failed")
    return result


def verify(receipt: Path, require_csim: bool = True) -> dict[str, str]:
    root = Path.cwd()
    summary = kv(receipt / "summary.txt")
    expected_summary = {
        "SCHEMA": "sounio.cs6.u250-validated-dyadic-vectors.v1",
        "CONTRACT_SHA256": digest(root / "scripts/research/cs6_u250_validated_dyadic_contract_v1.txt"),
        "GENERATOR_SHA256": digest(root / "scripts/research/cs6_u250_validated_dyadic_generate.py"),
        "CASES": "96",
        "VALID_CASES": "80",
        "REFUSAL_CASES": "16",
        "INPUT_WORDS": "480",
        "OUTPUT_WORDS": "864",
        "CASES_SHA256": digest(receipt / "cases.tsv"),
        "INPUTS_SHA256": digest(receipt / "inputs.bin"),
        "EXPECTED_SHA256": digest(receipt / "expected.bin"),
        "EXACT_RATIONAL_CONTAINMENT_PASS": "true",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            fail(f"summary mismatch: {key}")
    for key in FALSE_CLAIMS:
        if summary.get(key) != "false":
            fail(f"forbidden claim enabled: {key}")

    rows = list(csv.DictReader((receipt / "cases.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(rows) != 96:
        fail("case cardinality mismatch")
    input_words = decode_words(receipt / "inputs.bin")
    output_words = decode_words(receipt / "expected.bin")
    if len(input_words) != 480 or len(output_words) != 864:
        fail("binary cardinality mismatch")
    statuses: list[int] = []
    for index, row in enumerate(rows, 1):
        if row["CASE_INDEX"] != str(index) or not re.fullmatch(r"[a-z0-9_]+", row["CASE_ID"]):
            fail(f"case identity drift at {index}")
        values = tuple(map(int, (row["A_LO_RAW"], row["A_HI_RAW"], row["B_LO_RAW"], row["B_HI_RAW"], row["DIVISOR"])))
        if tuple(input_words[5 * (index - 1):5 * index]) != values:
            fail(f"input binding mismatch at {index}")
        exact = exact_outputs(*values)
        if tuple(output_words[9 * (index - 1):9 * index]) != exact:
            fail(f"exact output mismatch at {index}")
        if row["STATUS"] != str(exact[8]) or row["MUL_LO_RAW"] != str(exact[4]) or row["MUL_HI_RAW"] != str(exact[5]):
            fail(f"case transcript mismatch at {index}")
        statuses.append(exact[8])
    if statuses.count(1) != 80 or statuses.count(-1) != 6 or statuses.count(-2) != 5 or statuses.count(-3) != 5:
        fail("status population mismatch")

    if require_csim:
        csim = kv(receipt / "csim-summary.txt")
        required_csim = {
            "SCHEMA": "sounio.cs6.u250-validated-dyadic-csim.v1",
            "KERNEL_SHA256": digest(root / "hardware/fpga/u250_validated_dyadic/kernel.cpp"),
            "TESTBENCH_SHA256": digest(root / "hardware/fpga/u250_validated_dyadic/testbench.cpp"),
            "TCL_SHA256": digest(root / "hardware/fpga/u250_validated_dyadic/run_hls_csim.tcl"),
            "INPUTS_SHA256": digest(receipt / "inputs.bin"),
            "EXPECTED_SHA256": digest(receipt / "expected.bin"),
            "CSIM_LOG_SHA256": digest(receipt / "csim.log"),
            "CSIM_CASES": "96",
            "CSIM_WORDS": "864",
            "CSIM_MISMATCHES": "0",
            "VALIDATED_DYADIC_CSIM_PASS": "true",
            "PHYSICAL_FPGA_EXECUTION": "false",
        }
        for key, value in required_csim.items():
            if csim.get(key) != value:
                fail(f"CSim mismatch: {key}")

    manifest = receipt / "artifact-files.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="ascii").splitlines():
            sha, relative = line.split("  ", 1)
            target = receipt / relative
            if target == manifest or not target.is_file() or target.is_symlink() or digest(target) != sha:
                fail(f"artifact manifest mismatch: {relative}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--vectors-only", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.receipt, require_csim=not args.vectors_only)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("VERIFY_SCHEMA=sounio.cs6.u250-validated-dyadic-verification.v1")
    print("VERIFIED_CASES=96")
    print("VERIFIED_VALID_CASES=80")
    print("VERIFIED_REFUSAL_CASES=16")
    print(f"VERIFIED_HLS_CSIM={str(not args.vectors_only).lower()}")
    print("VALIDATED_DYADIC_VERIFY_PASS=true")


if __name__ == "__main__":
    main()
