#!/usr/bin/env python3
"""Generate Q24.40 inputs and an independent integer reference transcript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import struct
from fractions import Fraction
from pathlib import Path


FRAC_BITS = 40
ONE = 1 << FRAC_BITS
STEP = ONE >> 10
MAX_STEPS = 8000
EVENT_BISECTIONS = 24
WORDS_PER_OUTPUT = 8
ZS_TEXT = "22.3274637391"
ZS = 24549305999887
POST_ZS = float.fromhex("0x1.653d4a9e20f75p+4")
Q0_AREA = float.fromhex("-0x1.221ef15087f44p-10")
ORIGIN_X = Fraction("15.186446520640786")
ORIGIN_Y = Fraction("10.908543194765466")
UNSTABLE_X = Fraction("-0.67430316214199759")
UNSTABLE_Y = Fraction("-0.73845463335624273")
STABLE_X = Fraction("-0.94170446778164518")
STABLE_Y = Fraction("0.33644122125579123")
RADIUS_U = Fraction("0.004")
RADIUS_S = Fraction("0.3")
COORDINATES = Path("scripts/research/cs6_v7b_target23_prospective_epistemic_replay_coordinates_v1.tsv")
DECIMAL_RESULTS = Path("scripts/research/receipts/cs6_v7b_target23_decimal_center_replay_v1/results.tsv")
LEAF_RE = re.compile(r"U(?P<ud>[0-9]{2})-(?P<ui>[0-9]{10})_S(?P<sd>[0-9]{2})-(?P<si>[0-9]{10})")


def fail(message: str) -> None:
    raise SystemExit(f"target-23 U250 generator error: {message}")


def trunc_fraction(value: Fraction) -> int:
    magnitude = abs(value.numerator) // value.denominator
    return -magnitude if value < 0 else magnitude


def quantize(value: Fraction) -> int:
    return trunc_fraction(value * ONE)


def qmul(a: int, b: int) -> int:
    return (a * b) >> FRAC_BITS


def qdiv(value: int, divisor: int) -> int:
    magnitude = abs(value) // divisor
    return -magnitude if value < 0 else magnitude


def field(state: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, _ell = state
    xy = qmul(x, y)
    half_w_zs = qdiv(w + ZS, 2)
    return (
        2 * qmul(y, y) - xy,
        xy - qmul(y, half_w_zs),
        xy - w - ZS,
        x - y - half_w_zs - ONE,
    )


def add_scaled(base: tuple[int, ...], delta: tuple[int, ...], scale: int) -> tuple[int, ...]:
    return tuple(base[i] + qmul(delta[i], scale) for i in range(4))


def rk4(state: tuple[int, int, int, int], step: int) -> tuple[int, int, int, int]:
    k1 = field(state)
    k2 = field(add_scaled(state, k1, qdiv(step, 2)))
    k3 = field(add_scaled(state, k2, qdiv(step, 2)))
    k4 = field(add_scaled(state, k3, step))
    weighted = tuple(k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i] for i in range(4))
    return tuple(state[i] + qdiv(qmul(weighted[i], step), 6) for i in range(4))


def localize(left: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int, int]]:
    low, high = 0, STEP
    high_state = rk4(left, high)
    for _ in range(EVENT_BISECTIONS):
        middle = (low + high) >> 1
        middle_state = rk4(left, middle)
        if middle_state[2] < 0:
            low = middle
        else:
            high, high_state = middle, middle_state
    return high, high_state


def propagate(x: int, y: int) -> tuple[int, ...]:
    state = (x, y, 0, 0)
    event2_state = state
    time = 0
    event1_time = event2_time = 0
    events = 0
    armed = False
    steps = 0
    for step_index in range(MAX_STEPS):
        following = rk4(state, STEP)
        if following[2] < 0:
            armed = True
        if armed and state[2] < 0 <= following[2]:
            local_time, localized = localize(state)
            if events == 0:
                event1_time = time + local_time
            elif events == 1:
                event2_time = time + local_time
                event2_state = localized
            events += 1
            armed = False
        state = following
        time += STEP
        steps = step_index + 1
        if events == 2:
            break
    initial_normal = qmul(x, y) - ZS
    final_normal = qmul(event2_state[0], event2_state[1]) - ZS
    flags = (1 if events == 2 else 0) | (2 if initial_normal > 0 else 0) | (4 if final_normal > 0 else 0)
    return (steps, events, event1_time, event2_time, event2_state[0], event2_state[1], event2_state[3], flags)


def load_coordinates(root: Path) -> list[tuple[str, int, int, int, int]]:
    rows = list(csv.DictReader((root / COORDINATES).read_text(encoding="ascii").splitlines(), delimiter="\t"))
    result = []
    for row in rows:
        match = LEAF_RE.fullmatch(row["LEAF_ID"])
        if not match:
            fail(f"bad leaf id: {row['LEAF_ID']}")
        result.append((row["LEAF_ID"], *(int(match.group(name)) for name in ("ud", "ui", "sd", "si"))))
    if len(result) != 331:
        fail("coordinate cardinality drifted")
    return result


def initial_xy(ud: int, ui: int, sd: int, si: int) -> tuple[int, int]:
    u = -RADIUS_U + (Fraction(ui) + Fraction(1, 2)) * 2 * RADIUS_U / (1 << ud)
    s = -RADIUS_S + (Fraction(si) + Fraction(1, 2)) * 2 * RADIUS_S / (1 << sd)
    return quantize(ORIGIN_X + UNSTABLE_X * u + STABLE_X * s), quantize(ORIGIN_Y + UNSTABLE_Y * u + STABLE_Y * s)


def qfloat(value: int) -> float:
    return math.ldexp(value, -FRAC_BITS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coordinates = load_coordinates(root)
    decimal_rows = list(csv.DictReader((root / DECIMAL_RESULTS).read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(decimal_rows) != 331:
        fail("Decimal result cardinality drifted")
    inputs: list[int] = []
    outputs: list[int] = []
    reference_rows: list[dict[str, str]] = []
    max_delta = 0.0
    minimum_margin = math.inf
    passes = 0
    for leaf_index, ((leaf_id, ud, ui, sd, si), decimal) in enumerate(zip(coordinates, decimal_rows), 1):
        if decimal["LEAF_INDEX"] != str(leaf_index) or decimal["LEAF_ID"] != leaf_id:
            fail("Decimal ordering drifted")
        x0, y0 = initial_xy(ud, ui, sd, si)
        result = propagate(x0, y0)
        inputs.extend((x0, y0)); outputs.extend(result)
        x2, y2, ell2 = (qfloat(result[index]) for index in (4, 5, 6))
        determinant = math.exp(ell2) * (qfloat(x0) * qfloat(y0) - POST_ZS) / (x2 * y2 - POST_ZS) * Q0_AREA
        decimal_det = float(decimal["FINE_DETERMINANT"])
        delta = abs(determinant - decimal_det)
        max_delta = max(max_delta, delta)
        endpoints = [float.fromhex(decimal[key]) for key in ("C0HORECT2_LOWER", "C0HORECT2_UPPER", "C0RECT2_LOWER", "C0RECT2_UPPER")]
        inside = endpoints[0] < determinant < endpoints[1] and endpoints[2] < determinant < endpoints[3]
        minimum_margin = min(minimum_margin, determinant - endpoints[0], endpoints[1] - determinant,
                             determinant - endpoints[2], endpoints[3] - determinant)
        passed = result[1] == 2 and result[7] == 7 and determinant < 0 and inside
        passes += passed
        reference_rows.append({
            "LEAF_INDEX": str(leaf_index), "LEAF_ID": leaf_id, "X0_Q": str(x0), "Y0_Q": str(y0),
            "STEPS": str(result[0]), "EVENT1_TIME_Q": str(result[2]), "EVENT2_TIME_Q": str(result[3]),
            "X2_Q": str(result[4]), "Y2_Q": str(result[5]), "ELL2_Q": str(result[6]), "FLAGS": str(result[7]),
            "DETERMINANT": f"{determinant:.17g}", "DECIMAL_DETERMINANT": f"{decimal_det:.17g}",
            "ABS_DELTA": f"{delta:.17g}", "INSIDE_BOTH_CAPD": str(inside).lower(), "REFERENCE_PASS": str(passed).lower(),
        })
    input_path = args.out_dir / "inputs.bin"
    expected_path = args.out_dir / "expected.bin"
    input_path.write_bytes(struct.pack(f"<{len(inputs)}q", *inputs))
    expected_path.write_bytes(struct.pack(f"<{len(outputs)}q", *outputs))
    columns = tuple(reference_rows[0])
    with (args.out_dir / "reference.tsv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, columns, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(reference_rows)
    summary = (
        "SCHEMA=sounio.cs6.v7b-target23-u250-reference.v1\n"
        f"LEAVES={len(reference_rows)}\nOUTPUT_WORDS={len(outputs)}\nREFERENCE_PASSES={passes}\n"
        f"MAX_ABS_DELTA_VS_DECIMAL={max_delta:.17g}\nMIN_CAPD_MARGIN={minimum_margin:.17g}\n"
        f"INPUTS_SHA256={sha256(input_path)}\nEXPECTED_SHA256={sha256(expected_path)}\n"
        f"REFERENCE_PASS={str(passes == 331).lower()}\nFPGA_EXECUTION=false\n"
        "RIGOROUS_INTERVAL_CERTIFICATE=false\nLEAF_WIDE_CERTIFICATE=false\nGLOBAL_HPG_CERTIFICATE=false\n"
    )
    (args.out_dir / "reference-summary.txt").write_text(summary, encoding="ascii")
    print(summary, end="")
    if passes != 331:
        fail(f"only {passes}/331 reference leaves passed")


if __name__ == "__main__":
    main()
