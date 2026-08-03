#!/usr/bin/env python3
"""Generate exact dyadic target-23 Picard-step known-answer vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


BITS = 128
FRACTION_BITS = 96
ONE = 1 << FRACTION_BITS
DOMAIN_LIMIT = 1 << 111
STEP_DENOMINATOR = 256
BOX_INFLATION_RAW = 1 << 64
CONTRACT = Path("scripts/research/cs6_u250_target23_picard_step_contract_v1.txt")

Interval = tuple[int, int]


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def enclose(value: Fraction) -> Interval:
    return floor_fraction(value * ONE), ceil_fraction(value * ONE)


def add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def neg(value: Interval) -> Interval:
    return -value[1], -value[0]


def sub(left: Interval, right: Interval) -> Interval:
    return add(left, neg(right))


def mul(left: Interval, right: Interval) -> Interval:
    corners = [Fraction(a * b, ONE) for a in left for b in right]
    return floor_fraction(min(corners)), ceil_fraction(max(corners))


def div2(value: Interval) -> Interval:
    return floor_fraction(Fraction(value[0], 2)), ceil_fraction(Fraction(value[1], 2))


def scale2(value: Interval) -> Interval:
    return value[0] * 2, value[1] * 2


def absolute_upper(value: Interval) -> int:
    return max(abs(value[0]), abs(value[1]))


def field(state: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    x, y, w, _ell = state
    yy = mul(y, y)
    xy = mul(x, y)
    wzs = add(w, zs)
    return (
        sub(scale2(yy), xy),
        sub(xy, div2(mul(y, wzs))),
        sub(sub(xy, w), zs),
        sub(sub(sub(x, y), div2(wzs)), (ONE, ONE)),
    )


def image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    time = (0, ONE // STEP_DENOMINATOR)
    return tuple(add(component, mul(time, derivative)) for component, derivative in zip(initial, field(box, zs), strict=True))


def lipschitz_rows(box: tuple[Interval, ...], zs: Interval) -> tuple[int, ...]:
    x, y, w, _ell = box
    wzs = add(w, zs)
    row0 = absolute_upper(y) + absolute_upper(sub(scale2(scale2(y)), x))
    row1 = absolute_upper(y) + absolute_upper(sub(x, div2(wzs))) + ceil_fraction(Fraction(absolute_upper(y), 2))
    row2 = absolute_upper(y) + absolute_upper(x) + ONE
    row3 = 2 * ONE + ONE // 2
    return row0, row1, row2, row3


def initial_state() -> tuple[tuple[Interval, ...], Interval]:
    decimal = Fraction
    u = -decimal("0.004") + Fraction(447, 2) * decimal("0.008") / 256
    s = -decimal("0.3") + Fraction(651, 2) * decimal("0.6") / 512
    x = decimal("15.186446520640786") + decimal("-0.67430316214199759") * u + decimal("-0.94170446778164518") * s
    y = decimal("10.908543194765466") + decimal("-0.73845463335624273") * u + decimal("0.33644122125579123") * s
    return (enclose(x), enclose(y), (0, 0), (0, 0)), enclose(decimal("22.3274637391"))


def fixed_box(initial: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    box = initial
    for _ in range(64):
        candidate = image(initial, box, zs)
        widened = tuple((min(old[0], new[0]), max(old[1], new[1])) for old, new in zip(box, candidate, strict=True))
        if widened == box:
            return tuple((lower - BOX_INFLATION_RAW, upper + BOX_INFLATION_RAW) for lower, upper in box)
        box = widened
    raise ValueError("Picard candidate construction did not stabilize")


def status(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> int:
    intervals = (*initial, *box, zs)
    if any(lower > upper for lower, upper in intervals):
        return -1
    if any(not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for interval in intervals for endpoint in interval):
        return -2
    candidate = image(initial, box, zs)
    if any(not (container[0] < value[0] and value[1] < container[1]) for container, value in zip(box, candidate, strict=True)):
        return -4
    contraction = ceil_fraction(Fraction(max(lipschitz_rows(box, zs)), STEP_DENOMINATOR))
    if contraction >= ONE:
        return -5
    return 1


def outputs(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval) -> tuple[int, ...]:
    result_status = status(initial, box, zs)
    if result_status != 1:
        return (0,) * 21 + (result_status,)
    derivatives = field(box, zs)
    candidate = image(initial, box, zs)
    rows = lipschitz_rows(box, zs)
    contraction = ceil_fraction(Fraction(max(rows), STEP_DENOMINATOR))
    return tuple(endpoint for interval in (*derivatives, *candidate) for endpoint in interval) + rows + (contraction, result_status)


@dataclass(frozen=True)
class Case:
    case_id: str
    initial: tuple[Interval, ...]
    box: tuple[Interval, ...]
    zs: Interval

    def input_words(self) -> tuple[int, ...]:
        return tuple(endpoint for interval in (*self.initial, *self.box, self.zs) for endpoint in interval)


def cases() -> list[Case]:
    initial, zs = initial_state()
    box = fixed_box(initial, zs)
    uninflated = tuple((lower + BOX_INFLATION_RAW, upper - BOX_INFLATION_RAW) for lower, upper in box)
    reversed_initial = ((initial[0][1], initial[0][0]), *initial[1:])
    outside_initial = (((DOMAIN_LIMIT, DOMAIN_LIMIT)), *initial[1:])
    return [
        Case("leaf331_center_valid", initial, box, zs),
        Case("leaf331_center_non_strict_box", initial, uninflated, zs),
        Case("reversed_initial_refusal", reversed_initial, box, zs),
        Case("outside_domain_refusal", outside_initial, box, zs),
    ]


def encode(words: list[int]) -> bytes:
    return b"".join(word.to_bytes(16, "little", signed=True) for word in words)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = cases()
    input_words = [word for case in selected for word in case.input_words()]
    output_words = [word for case in selected for word in outputs(case.initial, case.box, case.zs)]
    (args.out_dir / "inputs.bin").write_bytes(encode(input_words))
    (args.out_dir / "expected.bin").write_bytes(encode(output_words))
    with (args.out_dir / "cases.tsv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["CASE_INDEX", "CASE_ID", "STATUS", "CONTRACTION_UPPER_RAW", "MIN_SELF_MAP_MARGIN_RAW"])
        for index, case in enumerate(selected, 1):
            result = outputs(case.initial, case.box, case.zs)
            if result[-1] == 1:
                candidate = image(case.initial, case.box, case.zs)
                margin = min(min(value[0] - container[0], container[1] - value[1]) for container, value in zip(case.box, candidate, strict=True))
                contraction = result[-2]
            else:
                margin = 0
                contraction = 0
            writer.writerow([index, case.case_id, result[-1], contraction, margin])
    summary = [
        "SCHEMA=sounio.cs6.u250-target23-picard-step-vectors.v1",
        f"CONTRACT_SHA256={digest(CONTRACT)}",
        f"GENERATOR_SHA256={digest(Path(__file__))}",
        "CASES=4",
        "ACCEPTED_CASES=1",
        "REFUSED_CASES=3",
        f"INPUT_WORDS={len(input_words)}",
        f"OUTPUT_WORDS={len(output_words)}",
        f"CASES_SHA256={digest(args.out_dir / 'cases.tsv')}",
        f"INPUTS_SHA256={digest(args.out_dir / 'inputs.bin')}",
        f"EXPECTED_SHA256={digest(args.out_dir / 'expected.bin')}",
        "EXACT_RATIONAL_PICARD_RECONSTRUCTION=true",
        "HLS_CSIM=false",
        "PHYSICAL_FPGA_EXECUTION=false",
        "FULL_ORBIT_CERTIFICATE=false",
        "LEAF_WIDE_CERTIFICATE=false",
        "GLOBAL_HPG_CERTIFICATE=false",
        "NOVELTY_OR_PRIORITY_CLAIMED=false",
        "OPEN_PROBLEM_SOLVED=false",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
