#!/usr/bin/env python3
"""Independently replay the exact target-23 Picard-step certificate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from fractions import Fraction
from pathlib import Path


ONE = 1 << 96
DOMAIN_LIMIT = 1 << 111
STEP_DENOMINATOR = 256
INFLATION = 1 << 64
Interval = tuple[int, int]


def fail(message: str) -> None:
    raise ValueError(f"target-23 Picard verify error: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_kv(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        fail(f"missing regular file: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.count("=") != 1:
            fail(f"malformed line: {path.name}")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or key in fields:
            fail(f"invalid field: {path.name}:{key}")
        fields[key] = value
    return fields


def floor_q(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_q(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def enclosure(value: Fraction) -> Interval:
    return floor_q(value * ONE), ceil_q(value * ONE)


def plus(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def minus(a: Interval, b: Interval) -> Interval:
    return a[0] - b[1], a[1] - b[0]


def times(a: Interval, b: Interval) -> Interval:
    products = [Fraction(x * y, ONE) for x in a for y in b]
    return floor_q(min(products)), ceil_q(max(products))


def half(a: Interval) -> Interval:
    return floor_q(Fraction(a[0], 2)), ceil_q(Fraction(a[1], 2))


def magnitude(a: Interval) -> int:
    return max(abs(a[0]), abs(a[1]))


def vector_field(state: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    x, y, w, _ell = state
    xy = times(x, y)
    yy = times(y, y)
    wzs = plus(w, zs)
    return (
        minus((2 * yy[0], 2 * yy[1]), xy),
        minus(xy, half(times(y, wzs))),
        minus(minus(xy, w), zs),
        minus(minus(minus(x, y), half(wzs)), (ONE, ONE)),
    )


def picard_image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    time = (0, ONE // STEP_DENOMINATOR)
    return tuple(plus(x0, times(time, derivative)) for x0, derivative in zip(initial, vector_field(box, zs), strict=True))


def row_bounds(box: tuple[Interval, ...], zs: Interval) -> tuple[int, ...]:
    x, y, w, _ell = box
    four_y_minus_x = minus((4 * y[0], 4 * y[1]), x)
    diagonal_y = minus(x, half(plus(w, zs)))
    return (
        magnitude(y) + magnitude(four_y_minus_x),
        magnitude(y) + magnitude(diagonal_y) + ceil_q(Fraction(magnitude(y), 2)),
        magnitude(y) + magnitude(x) + ONE,
        5 * ONE // 2,
    )


def evaluate(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[int, ...]:
    all_intervals = (*initial, *box, zs)
    if any(a > b for a, b in all_intervals):
        return (0,) * 21 + (-1,)
    if any(not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for interval in all_intervals for endpoint in interval):
        return (0,) * 21 + (-2,)
    derivative = vector_field(box, zs)
    candidate = picard_image(initial, box, zs)
    if any(not (outer[0] < inner[0] and inner[1] < outer[1]) for outer, inner in zip(box, candidate, strict=True)):
        return (0,) * 21 + (-4,)
    rows = row_bounds(box, zs)
    contraction = ceil_q(Fraction(max(rows), STEP_DENOMINATOR))
    if contraction >= ONE:
        return (0,) * 21 + (-5,)
    return tuple(value for interval in (*derivative, *candidate) for value in interval) + rows + (contraction, 1)


def decode(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 16:
        fail(f"unaligned binary: {path.name}")
    return [int.from_bytes(raw[index:index + 16], "little", signed=True) for index in range(0, len(raw), 16)]


def expected_initial() -> tuple[tuple[Interval, ...], Interval]:
    q = Fraction
    u = -q("0.004") + q(447, 2) * q("0.008") / 256
    s = -q("0.3") + q(651, 2) * q("0.6") / 512
    x = q("15.186446520640786") + q("-0.67430316214199759") * u + q("-0.94170446778164518") * s
    y = q("10.908543194765466") + q("-0.73845463335624273") * u + q("0.33644122125579123") * s
    return (enclosure(x), enclosure(y), (0, 0), (0, 0)), enclosure(q("22.3274637391"))


def reconstruct_box(initial: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    box = initial
    for _ in range(64):
        candidate = picard_image(initial, box, zs)
        widened = tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(box, candidate, strict=True))
        if widened == box:
            return tuple((lower - INFLATION, upper + INFLATION) for lower, upper in box)
        box = widened
    fail("independent box reconstruction did not stabilize")


def verify(receipt: Path, require_csim: bool) -> None:
    root = Path.cwd()
    summary = read_kv(receipt / "summary.txt")
    required = {
        "SCHEMA": "sounio.cs6.u250-target23-picard-step-vectors.v1",
        "CONTRACT_SHA256": digest(root / "scripts/research/cs6_u250_target23_picard_step_contract_v1.txt"),
        "GENERATOR_SHA256": digest(root / "scripts/research/cs6_u250_target23_picard_step_generate.py"),
        "CASES": "4",
        "ACCEPTED_CASES": "1",
        "REFUSED_CASES": "3",
        "INPUT_WORDS": "72",
        "OUTPUT_WORDS": "88",
        "CASES_SHA256": digest(receipt / "cases.tsv"),
        "INPUTS_SHA256": digest(receipt / "inputs.bin"),
        "EXPECTED_SHA256": digest(receipt / "expected.bin"),
        "EXACT_RATIONAL_PICARD_RECONSTRUCTION": "true",
        "PHYSICAL_FPGA_EXECUTION": "false",
        "FULL_ORBIT_CERTIFICATE": "false",
        "LEAF_WIDE_CERTIFICATE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false",
        "OPEN_PROBLEM_SOLVED": "false",
    }
    for key, value in required.items():
        if summary.get(key) != value:
            fail(f"summary mismatch: {key}")
    inputs = decode(receipt / "inputs.bin")
    expected = decode(receipt / "expected.bin")
    if len(inputs) != 72 or len(expected) != 88:
        fail("word cardinality mismatch")
    initial0, zs0 = expected_initial()
    box0 = reconstruct_box(initial0, zs0)
    rows = list(csv.DictReader((receipt / "cases.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(rows) != 4:
        fail("case cardinality mismatch")
    statuses: list[int] = []
    for index, row in enumerate(rows):
        words = inputs[index * 18:(index + 1) * 18]
        intervals = tuple((words[offset], words[offset + 1]) for offset in range(0, 18, 2))
        initial = intervals[:4]
        box = intervals[4:8]
        zs = intervals[8]
        if index == 0 and (initial != initial0 or box != box0 or zs != zs0):
            fail("accepted case is not the independently reconstructed target-23 candidate")
        actual = evaluate(initial, box, zs)
        if tuple(expected[index * 22:(index + 1) * 22]) != actual:
            fail(f"exact transcript mismatch at case {index + 1}")
        if row["CASE_INDEX"] != str(index + 1) or row["STATUS"] != str(actual[-1]):
            fail(f"case row mismatch at case {index + 1}")
        if actual[-1] == 1:
            contraction = actual[-2]
            candidate = picard_image(initial, box, zs)
            margin = min(min(inner[0] - outer[0], outer[1] - inner[1]) for outer, inner in zip(box, candidate, strict=True))
            if row["CONTRACTION_UPPER_RAW"] != str(contraction) or row["MIN_SELF_MAP_MARGIN_RAW"] != str(margin):
                fail("accepted obligation metrics mismatch")
            if not (margin > 0 and contraction < ONE):
                fail("accepted mathematical obligation failed")
        statuses.append(actual[-1])
    if statuses != [1, -4, -1, -2]:
        fail("refusal population drift")
    if require_csim:
        csim = read_kv(receipt / "csim-summary.txt")
        csim_required = {
            "SCHEMA": "sounio.cs6.u250-target23-picard-step-csim.v1",
            "KERNEL_SHA256": digest(root / "hardware/fpga/u250_target23_picard_step/kernel.cpp"),
            "TESTBENCH_SHA256": digest(root / "hardware/fpga/u250_target23_picard_step/testbench.cpp"),
            "TCL_SHA256": digest(root / "hardware/fpga/u250_target23_picard_step/run_hls_csim.tcl"),
            "INPUTS_SHA256": digest(receipt / "inputs.bin"),
            "EXPECTED_SHA256": digest(receipt / "expected.bin"),
            "CSIM_LOG_SHA256": digest(receipt / "csim.log"),
            "CSIM_CASES": "4",
            "CSIM_WORDS": "88",
            "CSIM_MISMATCHES": "0",
            "TARGET23_PICARD_CSIM_PASS": "true",
            "PHYSICAL_FPGA_EXECUTION": "false",
        }
        for key, value in csim_required.items():
            if csim.get(key) != value:
                fail(f"CSim mismatch: {key}")
        csynth = read_kv(receipt / "csynth-summary.txt")
        csynth_required = {
            "SCHEMA": "sounio.cs6.u250-target23-picard-step-csynth.v1",
            "VITIS_VERSION": "2025.1",
            "PART": "xcu250-figd2104-2L-e",
            "KERNEL_SHA256": digest(root / "hardware/fpga/u250_target23_picard_step/kernel.cpp"),
            "TCL_SHA256": digest(root / "hardware/fpga/u250_target23_picard_step/run_hls_csynth.tcl"),
            "CSYNTH_LOG_SHA256": digest(receipt / "csynth.log"),
            "CSYNTH_REPORT_SHA256": digest(receipt / "csynth.rpt"),
            "TARGET_CLOCK_NS": "4.00",
            "ESTIMATED_CLOCK_NS": "2.920",
            "ESTIMATED_FMAX_MHZ": "342.47",
            "LATENCY_CYCLES": "267",
            "LATENCY_US_AT_TARGET_CLOCK": "1.068",
            "INTERVAL_CYCLES": "268",
            "BRAM_18K": "19",
            "DSP": "50",
            "FF": "22508",
            "LUT": "29269",
            "URAM": "0",
            "ALL_LOOP_CONSTRAINTS_SATISFIED": "false",
            "RTL_GENERATED": "true",
            "HLS_SYNTHESIS_PASS": "true",
            "PHYSICAL_FPGA_EXECUTION": "false",
        }
        for key, value in csynth_required.items():
            if csynth.get(key) != value:
                fail(f"CSynth mismatch: {key}")
        report = (receipt / "csynth.rpt").read_text(encoding="utf-8")
        for witness in (
            "Target device:  xcu250-figd2104-2L-e",
            "|      267|      267|  1.068 us|  1.068 us|  268|  268|       no|",
            "|Total                |       19|     50|    22508|    29269|     0|",
        ):
            if witness not in report:
                fail("CSynth report witness missing")
    manifest = receipt / "artifact-files.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="ascii").splitlines():
            sha, relative = line.split("  ", 1)
            target = receipt / relative
            if target == manifest or not target.is_file() or target.is_symlink() or digest(target) != sha:
                fail(f"artifact manifest mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--vectors-only", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.receipt, not args.vectors_only)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("VERIFY_SCHEMA=sounio.cs6.u250-target23-picard-step-verification.v1")
    print("VERIFIED_CASES=4")
    print("VERIFIED_ACCEPTED_CASES=1")
    print("VERIFIED_REFUSED_CASES=3")
    print("BOUNDED_ONE_STEP_PICARD_CERTIFICATE=true")
    print(f"VERIFIED_HLS_CSIM={str(not args.vectors_only).lower()}")
    print(f"VERIFIED_HLS_SYNTHESIS={str(not args.vectors_only).lower()}")
    print("TARGET23_PICARD_STEP_VERIFY_PASS=true")


if __name__ == "__main__":
    main()
