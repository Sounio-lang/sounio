#!/usr/bin/env python3
"""Generate exact adversarial vectors for the U250 S1.I31.F96 interval nucleus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


BITS = 128
FRAC_BITS = 96
ONE = 1 << FRAC_BITS
DOMAIN_LIMIT = 1 << 111
DIVISORS = (2, 3, 6, 41)
VALID_CASES = 80
REFUSAL_CASES = 16


@dataclass(frozen=True)
class Case:
    identity: str
    a_lo: int
    a_hi: int
    b_lo: int
    b_hi: int
    divisor: int


def fail(message: str) -> None:
    raise SystemExit(f"validated dyadic generator error: {message}")


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def decimal_interval(token: str) -> tuple[int, int]:
    scaled = Fraction(token) * ONE
    return floor_fraction(scaled), ceil_fraction(scaled)


def status(case: Case) -> int:
    if case.a_lo > case.a_hi or case.b_lo > case.b_hi:
        return -1
    values = (case.a_lo, case.a_hi, case.b_lo, case.b_hi)
    if any(not -DOMAIN_LIMIT < value < DOMAIN_LIMIT for value in values):
        return -2
    if case.divisor not in DIVISORS:
        return -3
    return 1


def expected(case: Case) -> tuple[int, ...]:
    case_status = status(case)
    if case_status != 1:
        return (0, 0, 0, 0, 0, 0, 0, 0, case_status)
    products = (
        case.a_lo * case.b_lo,
        case.a_lo * case.b_hi,
        case.a_hi * case.b_lo,
        case.a_hi * case.b_hi,
    )
    result = (
        case.a_lo + case.b_lo,
        case.a_hi + case.b_hi,
        case.a_lo - case.b_hi,
        case.a_hi - case.b_lo,
        min(product // ONE for product in products),
        max(-((-product) // ONE) for product in products),
        case.a_lo // case.divisor,
        -((-case.a_hi) // case.divisor),
        case_status,
    )
    exact_a = (Fraction(case.a_lo, ONE), Fraction(case.a_hi, ONE))
    exact_b = (Fraction(case.b_lo, ONE), Fraction(case.b_hi, ONE))
    exact_products = tuple(left * right for left in exact_a for right in exact_b)
    checks = (
        (Fraction(result[0], ONE), exact_a[0] + exact_b[0], Fraction(result[1], ONE)),
        (Fraction(result[2], ONE), exact_a[0] - exact_b[1], Fraction(result[3], ONE)),
        (Fraction(result[4], ONE), min(exact_products), Fraction(result[5], ONE)),
        (Fraction(result[6], ONE), exact_a[0] / case.divisor, Fraction(result[7], ONE)),
        (Fraction(result[6], ONE), exact_a[1] / case.divisor, Fraction(result[7], ONE)),
    )
    if any(not lower <= value <= upper for lower, value, upper in checks):
        fail(f"independent containment failed for {case.identity}")
    return result


def valid_cases() -> list[Case]:
    tx_lo, tx_hi = decimal_interval("15.186446520640786")
    ty_lo, ty_hi = decimal_interval("10.908543194765466")
    boundary = DOMAIN_LIMIT - 1
    manual = [
        Case("zero", 0, 0, 0, 0, 2),
        Case("one_exact", ONE, ONE, ONE, ONE, 3),
        Case("minus_one_exact", -ONE, -ONE, -ONE, -ONE, 6),
        Case("mixed_unit", -ONE, -ONE, ONE, ONE, 41),
        Case("zero_crossing_a", -ONE, ONE, ONE, ONE, 2),
        Case("zero_crossing_b", ONE, ONE, -ONE, ONE, 3),
        Case("both_cross_zero", -ONE, ONE, -ONE, ONE, 6),
        Case("ulp_above_one", ONE + 1, ONE + 1, ONE + 1, ONE + 1, 41),
        Case("ulp_below_one", ONE - 1, ONE - 1, ONE - 1, ONE - 1, 2),
        Case("negative_ulp_boundary", -ONE - 1, -ONE + 1, ONE - 1, ONE + 1, 3),
        Case("target23_xy", tx_lo, tx_hi, ty_lo, ty_hi, 6),
        Case("target23_yx", ty_lo, ty_hi, tx_lo, tx_hi, 41),
        Case("narrow_positive", 17 * ONE + 1, 17 * ONE + 3, 23 * ONE - 2, 23 * ONE + 2, 2),
        Case("narrow_negative", -23 * ONE - 2, -23 * ONE + 2, 17 * ONE + 1, 17 * ONE + 3, 3),
        Case("wide_sign_mix", -31 * ONE, 29 * ONE, -7 * ONE, 11 * ONE, 6),
        Case("domain_positive_edge", boundary - 17, boundary, 1, ONE + 1, 41),
        Case("domain_negative_edge", -boundary, -boundary + 17, ONE - 1, ONE + 1, 2),
        Case("rounding_remainder_one", ONE + 1, ONE + 1, 1, 1, 3),
        Case("rounding_remainder_minus_one", -ONE - 1, -ONE - 1, 1, 1, 6),
        Case("division_negative_remainder", -17, -1, ONE, ONE, 41),
    ]
    rng = random.Random(0xC506250)
    cases = list(manual)
    while len(cases) < VALID_CASES:
        index = len(cases)
        center_a = rng.randrange(-(1 << 104), 1 << 104)
        center_b = rng.randrange(-(1 << 104), 1 << 104)
        width_a = rng.randrange(0, 1 << (20 + index % 61))
        width_b = rng.randrange(0, 1 << (20 + (3 * index) % 61))
        a_lo, a_hi = center_a - width_a, center_a + width_a
        b_lo, b_hi = center_b - width_b, center_b + width_b
        cases.append(Case(f"seeded_{index:03d}", a_lo, a_hi, b_lo, b_hi, DIVISORS[index % 4]))
    if len(cases) != VALID_CASES or any(status(case) != 1 for case in cases):
        fail("valid corpus construction drifted")
    return cases


def refusal_cases() -> list[Case]:
    cases = [
        Case("reversed_a_0", 1, 0, 0, 0, 2),
        Case("reversed_a_1", ONE, -ONE, 0, 0, 3),
        Case("reversed_a_2", 5, 4, -2, 2, 6),
        Case("reversed_b_0", 0, 0, 1, 0, 41),
        Case("reversed_b_1", 0, 1, ONE, -ONE, 2),
        Case("reversed_b_2", -2, 2, 5, 4, 3),
        Case("domain_a_lo", -DOMAIN_LIMIT, 0, 0, 0, 6),
        Case("domain_a_hi", 0, DOMAIN_LIMIT, 0, 0, 41),
        Case("domain_b_lo", 0, 0, -DOMAIN_LIMIT, 0, 2),
        Case("domain_b_hi", 0, 0, 0, DOMAIN_LIMIT, 3),
        Case("domain_both", -DOMAIN_LIMIT, DOMAIN_LIMIT, 0, 0, 6),
        Case("divisor_zero", 0, 0, 0, 0, 0),
        Case("divisor_negative", 0, 0, 0, 0, -1),
        Case("divisor_five", 0, 0, 0, 0, 5),
        Case("divisor_forty", 0, 0, 0, 0, 40),
        Case("divisor_forty_two", 0, 0, 0, 0, 42),
    ]
    if len(cases) != REFUSAL_CASES or sum(status(case) != 1 for case in cases) != REFUSAL_CASES:
        fail("refusal corpus construction drifted")
    return cases


def encode_word(value: int) -> bytes:
    if not -(1 << (BITS - 1)) <= value < (1 << (BITS - 1)):
        fail(f"word outside signed {BITS}-bit range: {value}")
    return value.to_bytes(16, "little", signed=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = valid_cases() + refusal_cases()
    inputs = [value for case in cases for value in (case.a_lo, case.a_hi, case.b_lo, case.b_hi, case.divisor)]
    outputs = [value for case in cases for value in expected(case)]
    (args.out_dir / "inputs.bin").write_bytes(b"".join(map(encode_word, inputs)))
    (args.out_dir / "expected.bin").write_bytes(b"".join(map(encode_word, outputs)))
    with (args.out_dir / "cases.tsv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("CASE_INDEX", "CASE_ID", "A_LO_RAW", "A_HI_RAW", "B_LO_RAW", "B_HI_RAW", "DIVISOR", "STATUS", "MUL_LO_RAW", "MUL_HI_RAW"))
        for index, case in enumerate(cases, 1):
            result = expected(case)
            writer.writerow((index, case.identity, case.a_lo, case.a_hi, case.b_lo, case.b_hi, case.divisor, result[8], result[4], result[5]))
    root = Path(__file__).resolve().parents[2]
    contract = root / "scripts/research/cs6_u250_validated_dyadic_contract_v1.txt"
    summary = (
        "SCHEMA=sounio.cs6.u250-validated-dyadic-vectors.v1\n"
        f"CONTRACT_SHA256={digest(contract)}\n"
        f"GENERATOR_SHA256={digest(Path(__file__))}\n"
        f"CASES={len(cases)}\nVALID_CASES={VALID_CASES}\nREFUSAL_CASES={REFUSAL_CASES}\n"
        f"INPUT_WORDS={len(inputs)}\nOUTPUT_WORDS={len(outputs)}\n"
        f"CASES_SHA256={digest(args.out_dir / 'cases.tsv')}\n"
        f"INPUTS_SHA256={digest(args.out_dir / 'inputs.bin')}\n"
        f"EXPECTED_SHA256={digest(args.out_dir / 'expected.bin')}\n"
        "EXACT_RATIONAL_CONTAINMENT_PASS=true\nFPGA_EXECUTION=false\n"
        "DYADIC_ARITHMETIC_CERTIFICATE=false\nPICARD_STEP_CERTIFICATE=false\n"
        "NOVELTY_OR_PRIORITY_CLAIMED=false\nOPEN_PROBLEM_SOLVED=false\n"
    )
    (args.out_dir / "summary.txt").write_text(summary, encoding="ascii")
    print(summary, end="")


if __name__ == "__main__":
    main()
