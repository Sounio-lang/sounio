#!/usr/bin/env python3
"""Generate the exact step-scaled Taylor-16 target-23 transcript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs6_u250_target23_picard_step_generate as picard


ONE = 1 << 96
H = (ONE >> 8, ONE >> 8)
ORDER = 16
CONTRACT = Path("scripts/research/cs6_u250_target23_scaled_taylor16_contract_v1.txt")
PREDECESSOR_INPUTS = Path("scripts/research/receipts/cs6_u250_target23_picard_step_v1/inputs.bin")
Interval = tuple[int, int]


def floor_q(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_q(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def divn(value: Interval, divisor: int) -> Interval:
    return floor_q(Fraction(value[0], divisor)), ceil_q(Fraction(value[1], divisor))


def total(values: list[Interval]) -> Interval:
    result = (0, 0)
    for value in values:
        result = picard.add(result, value)
    return result


def step_div(value: Interval, divisor: int) -> Interval:
    return divn(picard.mul(value, H), divisor)


def coefficients(state: tuple[Interval, ...], zs: Interval, order: int) -> list[list[Interval]]:
    coeff = [[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] for axis in range(4)]
    for degree in range(order):
        xy = total([picard.mul(coeff[0][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yy = total([picard.mul(coeff[1][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yw = total([picard.mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])
        coeff[0][degree + 1] = step_div(picard.sub(picard.scale2(yy), xy), degree + 1)
        coeff[1][degree + 1] = step_div(picard.sub(xy, picard.div2(picard.add(yw, picard.mul(zs, coeff[1][degree])))), degree + 1)
        coeff[2][degree + 1] = step_div(picard.sub(xy, picard.add(coeff[2][degree], zs if degree == 0 else (0, 0))), degree + 1)
        constant = picard.add(picard.div2(zs), (ONE, ONE)) if degree == 0 else (0, 0)
        coeff[3][degree + 1] = step_div(picard.sub(picard.sub(picard.sub(coeff[0][degree], coeff[1][degree]), picard.div2(coeff[2][degree])), constant), degree + 1)
    return coeff


def decode(path: Path) -> list[int]:
    raw = path.read_bytes()
    return [int.from_bytes(raw[index:index + 16], "little", signed=True) for index in range(0, len(raw), 16)]


def positive_input() -> tuple[Interval, ...]:
    words = decode(PREDECESSOR_INPUTS)[:18]
    return tuple((words[index], words[index + 1]) for index in range(0, 18, 2))


def evaluate(intervals: tuple[Interval, ...]) -> tuple[int, ...]:
    initial, box, zs = intervals[:4], intervals[4:8], intervals[8]
    if any(lower > upper for lower, upper in intervals):
        return (0,) * 152 + (-1,)
    if picard.status(initial, box, zs) != 1:
        return (0,) * 152 + (-4,)
    center_coeff = coefficients(initial, zs, ORDER - 1)
    box_coeff = coefficients(box, zs, ORDER)
    polynomial = [total(center_coeff[axis]) for axis in range(4)]
    # Taylor's componentwise Lagrange remainder evaluates the normalized
    # order-16 flow derivative at some trajectory state inside the Picard box.
    lagrange_remainder = [box_coeff[axis][ORDER] for axis in range(4)]
    next_state = [picard.add(polynomial[axis], lagrange_remainder[axis]) for axis in range(4)]
    transcript = [endpoint for degree in range(ORDER) for axis in range(4) for endpoint in center_coeff[axis][degree]]
    transcript += [endpoint for value in lagrange_remainder + polynomial + next_state for endpoint in value]
    return tuple(transcript) + (1,)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(words: list[int]) -> bytes:
    return b"".join(word.to_bytes(16, "little", signed=True) for word in words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted = positive_input()
    non_strict = (*accepted[:4], *tuple((a + (1 << 64), b - (1 << 64)) for a, b in accepted[4:8]), accepted[8])
    reversed_case = ((accepted[0][1], accepted[0][0]), *accepted[1:])
    cases = [("leaf331_scaled_taylor16", accepted), ("picard_precondition_refusal", non_strict), ("reversed_refusal", reversed_case)]
    inputs = [endpoint for _, case in cases for interval in case for endpoint in interval]
    results = [evaluate(case) for _, case in cases]
    outputs = [word for result in results for word in result]
    (args.out_dir / "inputs.bin").write_bytes(encode(inputs))
    (args.out_dir / "expected.bin").write_bytes(encode(outputs))
    with (args.out_dir / "cases.tsv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["CASE_INDEX", "CASE_ID", "STATUS", "MAX_REMAINDER_ABS_RAW", "MAX_NEXT_WIDTH_RAW"])
        for index, ((case_id, _case), result) in enumerate(zip(cases, results, strict=True), 1):
            remainder_words = result[128:136] if result[-1] == 1 else (0,) * 8
            next_words = result[144:152] if result[-1] == 1 else (0,) * 8
            max_remainder = max(abs(word) for word in remainder_words)
            max_width = max(next_words[2 * axis + 1] - next_words[2 * axis] for axis in range(4))
            writer.writerow([index, case_id, result[-1], max_remainder, max_width])
    summary = [
        "SCHEMA=sounio.cs6.u250-target23-scaled-taylor16-vectors.v1",
        f"CONTRACT_SHA256={digest(CONTRACT)}",
        f"GENERATOR_SHA256={digest(Path(__file__))}",
        f"PREDECESSOR_INPUTS_SHA256={digest(PREDECESSOR_INPUTS)}",
        "CASES=3", "ACCEPTED_CASES=1", "REFUSED_CASES=2",
        f"INPUT_WORDS={len(inputs)}", f"OUTPUT_WORDS={len(outputs)}",
        f"CASES_SHA256={digest(args.out_dir / 'cases.tsv')}",
        f"INPUTS_SHA256={digest(args.out_dir / 'inputs.bin')}",
        f"EXPECTED_SHA256={digest(args.out_dir / 'expected.bin')}",
        "EXACT_RATIONAL_SCALED_TAYLOR_RECONSTRUCTION=true",
        "HLS_CSIM=false", "HLS_SYNTHESIS=false", "PHYSICAL_FPGA_EXECUTION=false",
        "FULL_ORBIT_CERTIFICATE=false", "LEAF_WIDE_CERTIFICATE=false", "GLOBAL_HPG_CERTIFICATE=false",
        "NOVELTY_OR_PRIORITY_CLAIMED=false", "OPEN_PROBLEM_SOLVED=false",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
