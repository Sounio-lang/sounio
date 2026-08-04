#!/usr/bin/env python3
"""Independently replay the exact F192 Taylor-41 two-return chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from fractions import Fraction
from pathlib import Path


F = 192
U = 1 << F
H = U >> 8
N = 41
PAD = 1 << (F - 96)
LIMIT = U << 15
RADIUS_LIMIT = U >> 16
Interval = tuple[int, int]


def fail(message: str) -> None:
    raise ValueError(f"chained Taylor-41 verify error: {message}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.count("=") != 1:
            fail(f"malformed summary line in {path.name}")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or key in result:
            fail(f"invalid summary field {key}")
        result[key] = value
    return result


def down(numerator: int, denominator: int) -> int:
    return numerator // denominator


def up(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def plus(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def minus(a: Interval, b: Interval) -> Interval:
    return a[0] - b[1], a[1] - b[0]


def product(a: Interval, b: Interval) -> Interval:
    raw = [x * y for x in a for y in b]
    return down(min(raw), U), up(max(raw), U)


def halve(a: Interval) -> Interval:
    return down(a[0], 2), up(a[1], 2)


def absolute(a: Interval) -> int:
    return max(abs(a[0]), abs(a[1]))


def add_all(items: list[Interval]) -> Interval:
    lower = sum(item[0] for item in items)
    upper = sum(item[1] for item in items)
    return lower, upper


def initial_values() -> tuple[tuple[Interval, ...], Interval]:
    q = Fraction
    u = -q("0.004") + q(447, 2) * q("0.008") / 256
    s = -q("0.3") + q(651, 2) * q("0.6") / 512
    x = q("15.186446520640786") + q("-0.67430316214199759") * u + q("-0.94170446778164518") * s
    y = q("10.908543194765466") + q("-0.73845463335624273") * u + q("0.33644122125579123") * s

    def enclose(value: Fraction) -> Interval:
        scaled = value * U
        return scaled.numerator // scaled.denominator, -((-scaled.numerator) // scaled.denominator)

    return (enclose(x), enclose(y), (0, 0), (0, 0)), enclose(q("22.3274637391"))


def rhs(state: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    x, y, w, _ell = state
    xy = product(x, y)
    yy = product(y, y)
    wz = plus(w, zs)
    return (
        minus((2 * yy[0], 2 * yy[1]), xy),
        minus(xy, halve(product(y, wz))),
        minus(minus(xy, w), zs),
        minus(minus(minus(x, y), halve(wz)), (U, U)),
    )


def image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval, h: int) -> tuple[Interval, ...]:
    return tuple(plus(value, product((0, h), derivative)) for value, derivative in zip(initial, rhs(box, zs), strict=True))


def union(a: tuple[Interval, ...], b: tuple[Interval, ...]) -> tuple[Interval, ...]:
    return tuple((min(x[0], y[0]), max(x[1], y[1])) for x, y in zip(a, b, strict=True))


def lipschitz(box: tuple[Interval, ...], zs: Interval) -> int:
    x, y, w, _ell = box
    return max(
        absolute(y) + absolute(minus((4 * y[0], 4 * y[1]), x)),
        absolute(y) + absolute(minus(x, halve(plus(w, zs)))) + up(absolute(y), 2),
        absolute(y) + absolute(x) + U,
        5 * U // 2,
    )


def mu_infinity(box: tuple[Interval, ...], zs: Interval) -> int:
    x, y, w, _ell = box
    return max(
        -y[0] + absolute(minus((4 * y[0], 4 * y[1]), x)),
        minus(x, halve(plus(w, zs)))[1] + absolute(y) + up(absolute(y), 2),
        -U + absolute(y) + absolute(x),
        5 * U // 2,
    )


def close_box(initial: tuple[Interval, ...], zs: Interval, h: int) -> tuple[tuple[Interval, ...], int, int]:
    if any(a > b or not -LIMIT < endpoint < LIMIT for a, b in (*initial, zs) for endpoint in (a, b)):
        fail("state outside the frozen fixed-point domain")
    box = initial
    for iteration in range(1, 513):
        widened = union(box, image(initial, box, zs, h))
        if widened == box:
            padded = tuple((a - PAD, b + PAD) for a, b in box)
            mapped = image(initial, padded, zs, h)
            if not all(a[0] < b[0] and b[1] < a[1] for a, b in zip(padded, mapped, strict=True)):
                fail("padded Picard hull is not a strict self-map")
            contraction = up(lipschitz(padded, zs) * h, U)
            if contraction >= U:
                fail("Picard contraction is not strict")
            return padded, iteration, contraction
        box = widened
    fail("Picard hull did not stabilize")


def series(state: tuple[Interval, ...], zs: Interval, h: int, order: int) -> list[list[Interval]]:
    values = [[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] for axis in range(4)]
    for degree in range(order):
        xy = add_all([product(values[0][j], values[1][degree - j]) for j in range(degree + 1)])
        yy = add_all([product(values[1][j], values[1][degree - j]) for j in range(degree + 1)])
        yw = add_all([product(values[1][j], values[2][degree - j]) for j in range(degree + 1)])

        def scale(value: Interval) -> Interval:
            stepped = product(value, (h, h))
            return down(stepped[0], degree + 1), up(stepped[1], degree + 1)

        values[0][degree + 1] = scale(minus((2 * yy[0], 2 * yy[1]), xy))
        values[1][degree + 1] = scale(minus(xy, halve(plus(yw, product(zs, values[1][degree])))))
        values[2][degree + 1] = scale(minus(xy, plus(values[2][degree], zs if degree == 0 else (0, 0))))
        forcing = plus(halve(zs), (U, U)) if degree == 0 else (0, 0)
        values[3][degree + 1] = scale(minus(minus(minus(values[0][degree], values[1][degree]), halve(values[2][degree])), forcing))
    return values


def exp_majorant(x: int) -> int:
    if x <= 0:
        return U
    term = U
    answer = U
    for k in range(1, 33):
        term = up(term * x, U * k)
        answer += term
    following = up(term * x, U * 33)
    ratio = up(x, 34)
    return answer + up(following * U, U - ratio)


def section_sign(center: int, radius: int) -> int:
    return -1 if center + radius < 0 else 1 if center - radius > 0 else 0


def replay_step(center: tuple[int, ...], radius: int, zs: Interval, h: int) -> dict[str, object]:
    initial = tuple((value - radius, value + radius) for value in center)
    box, iterations, contraction = close_box(initial, zs, h)
    point = tuple((value, value) for value in center)
    point_box, _, _ = close_box(point, zs, h)
    polynomial_terms = series(point, zs, h, N - 1)
    remainder_terms = series(point_box, zs, h, N)
    next_center = []
    local = 0
    for axis in range(4):
        enclosure = plus(add_all(polynomial_terms[axis]), remainder_terms[axis][N])
        midpoint = (enclosure[0] + enclosure[1]) // 2
        next_center.append(midpoint)
        local = max(local, midpoint - enclosure[0], enclosure[1] - midpoint)
    mu = mu_infinity(box, zs)
    mu_h = up(max(mu, 0) * h, U)
    amplification = exp_majorant(mu_h)
    next_radius = up(radius * amplification, U) + local
    if next_radius >= RADIUS_LIMIT:
        fail("radius refusal threshold reached")
    return {"center": tuple(next_center), "radius": next_radius, "local": local, "mu": mu, "mu_h": mu_h, "amplification": amplification, "iterations": iterations, "contraction": contraction, "box": box}


def replay_event(center: tuple[int, ...], radius: int, zs: Interval, h: int) -> dict[str, object]:
    low, high = 0, h
    low_result: dict[str, object] = {"center": center, "radius": radius}
    high_result = replay_step(center, radius, zs, high)
    decisive = 0
    for _ in range(42):
        middle = (low + high) // 2
        candidate = replay_step(center, radius, zs, middle)
        sign = section_sign(candidate["center"][2], int(candidate["radius"]))
        if sign < 0:
            low, low_result = middle, candidate
        elif sign > 0:
            high, high_result = middle, candidate
        else:
            break
        decisive += 1
    if section_sign(low_result["center"][2], int(low_result["radius"])) != -1 or section_sign(high_result["center"][2], int(high_result["radius"])) != 1:
        fail("event bracket endpoint signs are not strict")
    if high - low > U >> 50:
        fail("event bracket is wider than 2^-50")
    event_initial = tuple((value - int(low_result["radius"]), value + int(low_result["radius"])) for value in low_result["center"])
    event_box, iterations, contraction = close_box(event_initial, zs, high - low)
    normal = minus(product(event_box[0], event_box[1]), zs)
    if normal[0] <= 0:
        fail("event transversality is not strictly positive")
    return {"low": low, "high": high, "normal": normal, "bisections": decisive, "iterations": iterations, "contraction": contraction}


def decode(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 28:
        fail(f"unaligned F192 binary {path.name}")
    return [int.from_bytes(raw[i:i + 28], "little", signed=True) for i in range(0, len(raw), 28)]


def verify(receipt: Path) -> None:
    root = Path.cwd()
    summary = read_summary(receipt / "summary.txt")
    frozen = {
        "SCHEMA": "sounio.cs6.u250-target23-chained-taylor41-vectors.v1",
        "CONTRACT_SHA256": sha(root / "scripts/research/cs6_u250_target23_chained_taylor41_contract_v1.txt"),
        "GENERATOR_SHA256": sha(root / "scripts/research/cs6_u250_target23_chained_taylor41_generate.py"),
        "FRACTION_BITS": "192", "TAYLOR_ORDER": "41", "STEPS": "1686",
        "FINAL_TIME_RAW": "41340599710398217843074769404406740537299106560024852299776",
        "SMALLEST_STEP_BITS": "8", "EVENTS": "2", "EVENT_BISECTIONS": "42",
        "MAX_RADIUS_RAW": "1738370293121837432722397315748401651833148",
        "MAX_LOCAL_RADIUS_RAW": "98577913497", "MAX_PICARD_ITERATIONS": "100",
        "MAX_CONTRACTION_RAW": "1301403511132979253557791172114621424919482302109361604391",
        "CHAIN_SHA256": sha(receipt / "chain.tsv"), "EVENTS_SHA256": sha(receipt / "events.tsv"),
        "INPUTS_SHA256": sha(receipt / "inputs.bin"), "EXPECTED_SHA256": sha(receipt / "expected.bin"),
        "HARDWARE_PARTITIONS": "2", "HARDWARE_INPUT_WORDS": "26", "HARDWARE_OUTPUT_WORDS": "16860",
        "HARDWARE_INPUTS_SHA256": sha(receipt / "hardware_inputs.bin"),
        "PARTITIONS_SHA256": sha(receipt / "partitions.tsv"),
        "TWO_RETURN_CHAIN_CERTIFICATE": "true", "HLS_CSIM": "false", "HLS_SYNTHESIS": "false",
        "PHYSICAL_FPGA_EXECUTION": "false", "DUAL_U250_EXECUTION": "false",
        "FULL_ORBIT_CERTIFICATE": "false", "LEAF_WIDE_CERTIFICATE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "NOVELTY_OR_PRIORITY_CLAIMED": "false", "OPEN_PROBLEM_SOLVED": "false",
    }
    for key, value in frozen.items():
        if summary.get(key) != value:
            fail(f"summary mismatch {key}")
    raw_initial, zs = initial_values()
    center = []
    radius = 0
    for lower, upper in raw_initial:
        midpoint = (lower + upper) // 2
        center.append(midpoint)
        radius = max(radius, midpoint - lower, upper - midpoint)
    current = tuple(center)
    if decode(receipt / "inputs.bin") != [*current, radius, zs[0], zs[1]]:
        fail("input binary does not bind the exact initial condition")
    chain = list(csv.DictReader((receipt / "chain.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    events = list(csv.DictReader((receipt / "events.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(chain) != 1686 or len(events) != 2:
        fail("chain or event cardinality mismatch")
    time = 0
    expected_words: list[int] = []
    observed_events = 0
    armed = False
    partition_starts = [(0, 0, 843, 0, current, radius, False, 0)]
    for index, row in enumerate(chain, 1):
        h = int(row["STEP_RAW"])
        before = section_sign(current[2], radius)
        result = replay_step(current, radius, zs, h)
        after = section_sign(result["center"][2], int(result["radius"]))
        scalar_expected = {
            "STEP": index, "TIME_START_RAW": time, "TIME_END_RAW": time + h,
            "STEP_BITS": 8, "RADIUS_RAW": result["radius"], "LOCAL_RADIUS_RAW": result["local"],
            "MU_RAW": result["mu"], "MU_H_RAW": result["mu_h"], "AMPLIFICATION_RAW": result["amplification"],
            "PICARD_ITERATIONS": result["iterations"], "CONTRACTION_RAW": result["contraction"],
            "BEFORE_SIGN": before, "AFTER_SIGN": after,
        }
        for key, value in scalar_expected.items():
            if row[key] != str(value):
                fail(f"step {index} mismatch {key}")
        for axis, name in enumerate(("X", "Y", "W", "ELL")):
            if row[f"CENTER_{name}_RAW"] != str(result["center"][axis]):
                fail(f"step {index} center mismatch {name}")
        event_index = int(row["EVENT_INDEX"])
        if after < 0:
            armed = True
        if event_index:
            observed_events += 1
            event = replay_event(current, radius, zs, h)
            record = events[event_index - 1]
            base = time
            event_expected = {
                "EVENT_INDEX": event_index, "BASE_STEP": index - 1,
                "LOCAL_LOW_RAW": event["low"], "LOCAL_HIGH_RAW": event["high"],
                "GLOBAL_LOW_RAW": base + int(event["low"]), "GLOBAL_HIGH_RAW": base + int(event["high"]),
                "NORMAL_LOWER_RAW": event["normal"][0], "NORMAL_UPPER_RAW": event["normal"][1],
                "BISECTIONS": event["bisections"], "PICARD_ITERATIONS": event["iterations"],
                "CONTRACTION_RAW": event["contraction"],
            }
            for key, value in event_expected.items():
                if record[key] != str(value):
                    fail(f"event {event_index} mismatch {key}")
            armed = False
        expected_words.extend([time, time + h, h, 8, *result["center"], result["radius"], event_index])
        current, radius, time = result["center"], int(result["radius"]), time + h
        if index == 843:
            partition_starts.append((1, 843, 843, time, current, radius, armed, observed_events))
    if observed_events != 2 or decode(receipt / "expected.bin") != expected_words:
        fail("event count or expected binary transcript mismatch")
    partitions = list(csv.DictReader((receipt / "partitions.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(partitions) != 2:
        fail("hardware partition cardinality mismatch")
    hardware_words: list[int] = []
    for record, start in zip(partitions, partition_starts, strict=True):
        partition_id, start_step, count, start_time, start_center, start_radius, start_armed, prior_events = start
        expected_partition = {
            "PARTITION_ID": partition_id, "START_STEP": start_step, "STEPS": count,
            "OUTPUT_WORD_OFFSET": start_step * 10, "OUTPUT_WORDS": count * 10,
            "ARMED": int(start_armed), "PRIOR_EVENTS": prior_events,
        }
        for key, value in expected_partition.items():
            if record[key] != str(value):
                fail(f"partition {partition_id} mismatch {key}")
        hardware_words.extend([
            partition_id, start_step, count, start_time, *start_center, start_radius,
            zs[0], zs[1], int(start_armed), prior_events,
        ])
    if decode(receipt / "hardware_inputs.bin") != hardware_words:
        fail("hardware partition input binding mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        verify(args.receipt)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("VERIFY_SCHEMA=sounio.cs6.u250-target23-chained-taylor41-verification.v1")
    print("VERIFIED_STEPS=1686")
    print("VERIFIED_EVENTS=2")
    print("EVENT_BRACKET_WIDTH=2^-50")
    print("F192_CORRECTION_LANE_VERIFIED=true")
    print("TWO_RETURN_CHAIN_CERTIFICATE_VERIFIED=true")
    print("TARGET23_CHAINED_TAYLOR41_VERIFY_PASS=true")


if __name__ == "__main__":
    main()
