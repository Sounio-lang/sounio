#!/usr/bin/env python3
"""Validated Arb/Taylor enclosure for the frozen target-23 center orbit."""

from __future__ import annotations

import argparse
import hashlib
import platform
from pathlib import Path
from fractions import Fraction

import flint
from flint import arb, ctx


PRECISION_BITS = 256
TAYLOR_ORDER = 40
MAX_STEPS = 2000
EVENT_BISECTIONS = 60
LEAF_ID = "U08-0000000223_S09-0000000325"
U_DEPTH, U_INDEX, S_DEPTH, S_INDEX = 8, 223, 9, 325

ctx.prec = PRECISION_BITS
ctx.threads = 1


def exact_decimal(token: str) -> arb:
    value = Fraction(token)
    return arb(value.numerator) / arb(value.denominator)


ZS = exact_decimal("22.3274637391")
ORIGIN_X = exact_decimal("15.186446520640786")
ORIGIN_Y = exact_decimal("10.908543194765466")
UNSTABLE_X = exact_decimal("-0.67430316214199759")
UNSTABLE_Y = exact_decimal("-0.73845463335624273")
STABLE_X = exact_decimal("-0.94170446778164518")
STABLE_Y = exact_decimal("0.33644122125579123")
RADIUS_U = exact_decimal("0.004")
RADIUS_S = exact_decimal("0.3")
STEP = arb(1) / arb(2**8)


def fail(message: str) -> None:
    raise SystemExit(f"Arb validated center worker error: {message}")


def exact_fraction(value: arb) -> str:
    rational = value.fmpq()
    return str(rational)


def lower_fraction(value: arb) -> str:
    return exact_fraction(value.lower())


def upper_fraction(value: arb) -> str:
    return exact_fraction(value.upper())


def upper_abs(value: arb) -> arb:
    return abs(value).upper()


def max_upper(values: list[arb]) -> arb:
    if not values:
        fail("empty maximum")
    result = values[0].upper()
    for value in values[1:]:
        candidate = value.upper()
        if candidate > result:
            result = candidate
    return result


def field(state: list[arb]) -> list[arb]:
    x, y, w, _ell = state
    return [
        2 * y * y - x * y,
        x * y - y * (w + ZS) / 2,
        x * y - w - ZS,
        x - y - (w + ZS) / 2 - 1,
    ]


def flow_coefficients(initial: list[arb], degree: int) -> list[list[arb]]:
    coefficients = [
        [initial[index]] + [arb(0) for _ in range(degree)]
        for index in range(4)
    ]
    for n in range(degree):
        xy = sum(
            (coefficients[0][j] * coefficients[1][n - j] for j in range(n + 1)),
            arb(0),
        )
        yy = sum(
            (coefficients[1][j] * coefficients[1][n - j] for j in range(n + 1)),
            arb(0),
        )
        yw = sum(
            (coefficients[1][j] * coefficients[2][n - j] for j in range(n + 1)),
            arb(0),
        )
        divisor = arb(n + 1)
        coefficients[0][n + 1] = (2 * yy - xy) / divisor
        coefficients[1][n + 1] = (
            xy - (yw + ZS * coefficients[1][n]) / 2
        ) / divisor
        coefficients[2][n + 1] = (
            xy - coefficients[2][n] - (ZS if n == 0 else 0)
        ) / divisor
        coefficients[3][n + 1] = (
            coefficients[0][n]
            - coefficients[1][n]
            - coefficients[2][n] / 2
            - ((ZS / 2 + 1) if n == 0 else 0)
        ) / divisor
    return coefficients


def inflate(value: arb) -> arb:
    factor = arb(17) / arb(16)
    epsilon = arb(1) / (arb(2) ** 220)
    return arb(value.mid(), value.rad() * factor + epsilon)


class Statistics:
    def __init__(self) -> None:
        self.picard_calls = 0
        self.picard_containments = 0
        self.advance_calls = 0
        self.max_picard_iterations = 0
        self.ambiguous_event_stops = 0
        self.max_global_radius = arb(0)
        self.max_local_remainder = arb(0)
        self.accumulated_mu_h = arb(0)
        self.max_picard_contraction = arb(0)


STATS = Statistics()


def picard_box(center: list[arb], radius: arb, step: arb) -> list[arb]:
    STATS.picard_calls += 1
    initial = [arb(component, radius) for component in center]
    time_interval = arb(step / 2, step / 2)
    box = [
        inflate(component.union(component + time_interval * derivative))
        for component, derivative in zip(initial, field(initial), strict=True)
    ]
    for iteration in range(1, 51):
        image = [
            component + time_interval * derivative
            for component, derivative in zip(initial, field(box), strict=True)
        ]
        if all(container.contains(candidate) for container, candidate in zip(box, image, strict=True)):
            contraction = ordinary_lipschitz_bound(box) * step
            if contraction.upper() >= 1:
                fail("Picard enclosure is not a strict contraction")
            STATS.picard_containments += 1
            STATS.max_picard_iterations = max(STATS.max_picard_iterations, iteration)
            STATS.max_picard_contraction = max_upper([
                STATS.max_picard_contraction, contraction,
            ])
            return box
        box = [
            inflate(container.union(candidate))
            for container, candidate in zip(box, image, strict=True)
        ]
    fail("Picard enclosure did not close")


def logarithmic_norm_bound(box: list[arb]) -> arb:
    x, y, w, _ell = box
    rows = [
        -y + abs(4 * y - x),
        x - (w + ZS) / 2 + abs(y) + abs(y) / 2,
        -1 + abs(y) + abs(x),
        arb(5) / 2,
    ]
    return max_upper(rows)


def ordinary_lipschitz_bound(box: list[arb]) -> arb:
    x, y, w, _ell = box
    rows = [
        abs(y) + abs(4 * y - x),
        abs(y) + abs(x - (w + ZS) / 2) + abs(y) / 2,
        abs(y) + abs(x) + 1,
        arb(5) / 2,
    ]
    return max_upper(rows)


def advance(center: list[arb], radius: arb, step: arb) -> tuple[list[arb], arb]:
    STATS.advance_calls += 1
    box = picard_box(center, radius, step)
    coefficients = flow_coefficients(center, TAYLOR_ORDER)
    polynomial = [
        sum(
            (coefficients[index][power] * step**power for power in range(TAYLOR_ORDER + 1)),
            arb(0),
        )
        for index in range(4)
    ]
    remainder_coefficients = flow_coefficients(box, TAYLOR_ORDER + 1)
    local_remainder = max_upper([
        upper_abs(remainder_coefficients[index][TAYLOR_ORDER + 1])
        for index in range(4)
    ]) * step ** (TAYLOR_ORDER + 1)
    rounding_radius = max_upper([component.rad() for component in polynomial])
    mu = logarithmic_norm_bound(box)
    amplification = (mu * step).exp().upper()
    next_radius = (amplification * radius + local_remainder + rounding_radius).upper()
    STATS.max_global_radius = max_upper([STATS.max_global_radius, next_radius])
    STATS.max_local_remainder = max_upper([STATS.max_local_remainder, local_remainder])
    STATS.accumulated_mu_h += mu * step
    return [component.mid() for component in polynomial], next_radius


def sign(component: arb, radius: arb) -> int:
    enclosure = arb(component, radius)
    if enclosure.upper() < 0:
        return -1
    if enclosure.lower() > 0:
        return 1
    return 0


def locate_event(center: list[arb], radius: arb) -> tuple[arb, arb, list[arb]]:
    low, high = arb(0), STEP
    low_center, low_radius = center, radius
    for _ in range(EVENT_BISECTIONS):
        middle = (low + high) / 2
        middle_center, middle_radius = advance(center, radius, middle)
        middle_sign = sign(middle_center[2], middle_radius)
        if middle_sign < 0:
            low, low_center, low_radius = middle, middle_center, middle_radius
        elif middle_sign > 0:
            high = middle
        else:
            STATS.ambiguous_event_stops += 1
            break
    if sign(low_center[2], low_radius) >= 0:
        fail("event lower endpoint is not strictly negative")
    high_center, high_radius = advance(center, radius, high)
    if sign(high_center[2], high_radius) <= 0:
        fail("event upper endpoint is not strictly positive")
    event_box = picard_box(low_center, low_radius, high - low)
    return low, high, event_box


def validate_hex(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        fail(f"invalid {name}")


def emit_interval(prefix: str, value: arb) -> None:
    print(f"{prefix}_LOWER_Q={lower_fraction(value)}")
    print(f"{prefix}_UPPER_Q={upper_fraction(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("challenge")
    parser.add_argument("attempt_binding")
    args = parser.parse_args()
    validate_hex(args.challenge, "challenge")
    validate_hex(args.attempt_binding, "attempt binding")

    u = -RADIUS_U + (arb(U_INDEX) + arb("0.5")) * (2 * RADIUS_U) / arb(2**U_DEPTH)
    s = -RADIUS_S + (arb(S_INDEX) + arb("0.5")) * (2 * RADIUS_S) / arb(2**S_DEPTH)
    raw_initial = [
        ORIGIN_X + UNSTABLE_X * u + STABLE_X * s,
        ORIGIN_Y + UNSTABLE_Y * u + STABLE_Y * s,
        arb(0),
        arb(0),
    ]
    radius = max_upper([component.rad() for component in raw_initial])
    center = [component.mid() for component in raw_initial]
    initial_box = [arb(component, radius) for component in center]
    initial_containment = all(
        enclosure.contains(raw)
        for enclosure, raw in zip(initial_box, raw_initial, strict=True)
    )
    if not initial_containment:
        fail("initial center-radius box does not contain the exact rational input enclosure")
    initial_normal = initial_box[0] * initial_box[1] - ZS
    q0_area = (
        UNSTABLE_X * STABLE_Y - STABLE_X * UNSTABLE_Y
    ) * RADIUS_U * RADIUS_S

    armed = False
    events: list[tuple[arb, arb, list[arb]]] = []
    steps = 0
    while steps < MAX_STEPS and len(events) < 2:
        next_center, next_radius = advance(center, radius, STEP)
        before_sign = sign(center[2], radius)
        after_sign = sign(next_center[2], next_radius)
        if after_sign < 0:
            armed = True
        if armed and before_sign < 0 and after_sign > 0:
            low, high, event_box = locate_event(center, radius)
            base_time = arb(steps) * STEP
            events.append((base_time + low, base_time + high, event_box))
            armed = False
        center, radius = next_center, next_radius
        steps += 1
    if len(events) != 2:
        fail(f"expected two validated events, got {len(events)}")

    second_event = events[1][2]
    final_normal = second_event[0] * second_event[1] - ZS
    determinant = second_event[3].exp() * initial_normal / final_normal * q0_area
    certificate = (
        STATS.picard_calls == STATS.picard_containments
        and STATS.max_picard_contraction.upper() < 1
        and initial_containment
        and initial_normal.lower() > 0
        and final_normal.lower() > 0
        and determinant.upper() < 0
    )
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    print("SCHEMA=sounio.cs6.v7b-target23-arb-validated-minmargin-worker.v1")
    print(f"WORKER_SOURCE_SHA256={source_sha}")
    print(f"PYTHON_VERSION={platform.python_version()}")
    print(f"PYTHON_FLINT_VERSION={flint.__version__}")
    print(f"RUN_CHALLENGE={args.challenge}")
    print(f"ATTEMPT_BINDING={args.attempt_binding}")
    print(f"LEAF_ID={LEAF_ID}")
    print(f"U_DEPTH={U_DEPTH}")
    print(f"U_INDEX={U_INDEX}")
    print(f"S_DEPTH={S_DEPTH}")
    print(f"S_INDEX={S_INDEX}")
    print(f"ARB_PRECISION_BITS={PRECISION_BITS}")
    print("ARB_THREADS=1")
    print(f"TAYLOR_ORDER={TAYLOR_ORDER}")
    print("TIME_STEP_POWER=-8")
    print(f"STEPS_COMPLETED={steps}")
    print(f"ADVANCE_CALLS={STATS.advance_calls}")
    print(f"PICARD_CALLS={STATS.picard_calls}")
    print(f"PICARD_CONTAINMENTS={STATS.picard_containments}")
    print(f"MAX_PICARD_ITERATIONS={STATS.max_picard_iterations}")
    print(f"AMBIGUOUS_EVENT_STOPS={STATS.ambiguous_event_stops}")
    print(f"EVENTS_VALIDATED={len(events)}")
    print(f"INITIAL_STATE_CONTAINMENT={str(initial_containment).lower()}")
    for index, (low, high, _box) in enumerate(events, 1):
        print(f"EVENT{index}_TIME_LOWER_Q={lower_fraction(low)}")
        print(f"EVENT{index}_TIME_UPPER_Q={upper_fraction(high)}")
    emit_interval("INITIAL_NORMAL", initial_normal)
    emit_interval("FINAL_NORMAL", final_normal)
    emit_interval("EVENT2_ELL", second_event[3])
    emit_interval("Q0_AREA", q0_area)
    emit_interval("DETERMINANT", determinant)
    print(f"MAX_GLOBAL_RADIUS_UPPER_Q={upper_fraction(STATS.max_global_radius)}")
    print(f"MAX_LOCAL_REMAINDER_UPPER_Q={upper_fraction(STATS.max_local_remainder)}")
    print(f"ACCUMULATED_MU_H_UPPER_Q={upper_fraction(STATS.accumulated_mu_h)}")
    print(f"MAX_PICARD_CONTRACTION_UPPER_Q={upper_fraction(STATS.max_picard_contraction)}")
    print(f"INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE={str(certificate).lower()}")
    print(f"PICARD_CONTRACTION_OBLIGATION={str(STATS.max_picard_contraction.upper() < 1).lower()}")
    print(f"EVENT2_TRANSVERSALITY={str(final_normal.lower() > 0).lower()}")
    print(f"DETERMINANT_STRICT_NEGATIVE={str(determinant.upper() < 0).lower()}")
    print("CAPD_USED_BY_WORKER=false")
    print("LEAF_WIDE_CERTIFICATE=false")
    print("INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("PROMOTION_ELIGIBLE=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("NOVELTY_OR_PRIORITY_CLAIMED=false")
    print("FPGA_EXECUTION=false")
    if not certificate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
