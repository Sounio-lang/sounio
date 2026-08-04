#!/usr/bin/env python3
"""Generate an exact F192 center-radius Taylor-41 chain for target 23."""

from __future__ import annotations

import argparse
import csv
import hashlib
from fractions import Fraction
from pathlib import Path


FRACTION_BITS = 192
ONE = 1 << FRACTION_BITS
MAX_STEP_BITS = 8
MIN_STEP_BITS = 16
STEP_RAW = ONE >> MAX_STEP_BITS
ORDER = 41
MAX_STEPS = 10000
EVENT_BISECTIONS = 42
RADIUS_LIMIT = ONE >> 16
DOMAIN_LIMIT = ONE << 15
PICARD_INFLATION_RAW = 1 << (FRACTION_BITS - 96)
CONTRACT = Path("scripts/research/cs6_u250_target23_chained_taylor41_contract_v1.txt")
Interval = tuple[int, int]


def floor_q(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_q(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def enclose(value: Fraction) -> Interval:
    return floor_q(value * ONE), ceil_q(value * ONE)


def add(left: Interval, right: Interval) -> Interval:
    return left[0] + right[0], left[1] + right[1]


def sub(left: Interval, right: Interval) -> Interval:
    return left[0] - right[1], left[1] - right[0]


def mul(left: Interval, right: Interval) -> Interval:
    corners = [a * b for a in left for b in right]
    return min(corners) // ONE, -((-max(corners)) // ONE)


def half(value: Interval) -> Interval:
    return floor_q(Fraction(value[0], 2)), ceil_q(Fraction(value[1], 2))


def magnitude(value: Interval) -> int:
    return max(abs(value[0]), abs(value[1]))


def total(values: list[Interval]) -> Interval:
    result = (0, 0)
    for value in values:
        result = add(result, value)
    return result


def directed_divide(value: Interval, divisor: int) -> Interval:
    return value[0] // divisor, -((-value[1]) // divisor)


def scaled_divide(value: Interval, step_raw: int, divisor: int) -> Interval:
    return directed_divide(mul(value, (step_raw, step_raw)), divisor)


def initial_state() -> tuple[tuple[Interval, ...], Interval]:
    q = Fraction
    u = -q("0.004") + q(447, 2) * q("0.008") / 256
    s = -q("0.3") + q(651, 2) * q("0.6") / 512
    x = q("15.186446520640786") + q("-0.67430316214199759") * u + q("-0.94170446778164518") * s
    y = q("10.908543194765466") + q("-0.73845463335624273") * u + q("0.33644122125579123") * s
    return (enclose(x), enclose(y), (0, 0), (0, 0)), enclose(q("22.3274637391"))


def vector_field(state: tuple[Interval, ...], zs: Interval) -> tuple[Interval, ...]:
    x, y, w, _ell = state
    xy = mul(x, y)
    yy = mul(y, y)
    wzs = add(w, zs)
    return (
        sub((2 * yy[0], 2 * yy[1]), xy),
        sub(xy, half(mul(y, wzs))),
        sub(sub(xy, w), zs),
        sub(sub(sub(x, y), half(wzs)), (ONE, ONE)),
    )


def picard_image(initial: tuple[Interval, ...], box: tuple[Interval, ...], zs: Interval, step_raw: int) -> tuple[Interval, ...]:
    time = (0, step_raw)
    return tuple(add(component, mul(time, derivative)) for component, derivative in zip(initial, vector_field(box, zs), strict=True))


def hull(left: tuple[Interval, ...], right: tuple[Interval, ...]) -> tuple[Interval, ...]:
    return tuple((min(a[0], b[0]), max(a[1], b[1])) for a, b in zip(left, right, strict=True))


def ordinary_lipschitz(box: tuple[Interval, ...], zs: Interval) -> int:
    x, y, w, _ell = box
    return max(
        magnitude(y) + magnitude(sub((4 * y[0], 4 * y[1]), x)),
        magnitude(y) + magnitude(sub(x, half(add(w, zs)))) + ceil_q(Fraction(magnitude(y), 2)),
        magnitude(y) + magnitude(x) + ONE,
        5 * ONE // 2,
    )


def logarithmic_norm(box: tuple[Interval, ...], zs: Interval) -> int:
    x, y, w, _ell = box
    return max(
        -y[0] + magnitude(sub((4 * y[0], 4 * y[1]), x)),
        sub(x, half(add(w, zs)))[1] + magnitude(y) + ceil_q(Fraction(magnitude(y), 2)),
        -ONE + magnitude(y) + magnitude(x),
        5 * ONE // 2,
    )


def picard_box(initial: tuple[Interval, ...], zs: Interval, step_raw: int) -> tuple[tuple[Interval, ...], int, int]:
    if any(lower > upper or not -DOMAIN_LIMIT < endpoint < DOMAIN_LIMIT for lower, upper in (*initial, zs) for endpoint in (lower, upper)):
        raise ValueError("initial interval outside frozen domain")
    box = initial
    for iteration in range(1, 513):
        image = picard_image(initial, box, zs, step_raw)
        widened = hull(box, image)
        if widened == box:
            candidate = tuple((lower - PICARD_INFLATION_RAW, upper + PICARD_INFLATION_RAW) for lower, upper in box)
            candidate_image = picard_image(initial, candidate, zs, step_raw)
            if not all(outer[0] < inner[0] and inner[1] < outer[1] for outer, inner in zip(candidate, candidate_image, strict=True)):
                raise ValueError("inflated Picard box is not a strict self-map")
            contraction = -((-ordinary_lipschitz(candidate, zs) * step_raw) // ONE)
            if contraction >= ONE:
                raise ValueError("Picard box is not a strict contraction")
            return candidate, iteration, contraction
        box = widened
    raise ValueError("Picard box did not close")


def coefficients(state: tuple[Interval, ...], zs: Interval, step_raw: int, order: int) -> list[list[Interval]]:
    coeff = [[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] for axis in range(4)]
    for degree in range(order):
        xy = total([mul(coeff[0][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yy = total([mul(coeff[1][j], coeff[1][degree - j]) for j in range(degree + 1)])
        yw = total([mul(coeff[1][j], coeff[2][degree - j]) for j in range(degree + 1)])
        coeff[0][degree + 1] = scaled_divide(sub((2 * yy[0], 2 * yy[1]), xy), step_raw, degree + 1)
        coeff[1][degree + 1] = scaled_divide(sub(xy, half(add(yw, mul(zs, coeff[1][degree])))), step_raw, degree + 1)
        coeff[2][degree + 1] = scaled_divide(sub(xy, add(coeff[2][degree], zs if degree == 0 else (0, 0))), step_raw, degree + 1)
        constant = add(half(zs), (ONE, ONE)) if degree == 0 else (0, 0)
        coeff[3][degree + 1] = scaled_divide(sub(sub(sub(coeff[0][degree], coeff[1][degree]), half(coeff[2][degree])), constant), step_raw, degree + 1)
    return coeff


def sign(center: int, radius: int) -> int:
    if center + radius < 0:
        return -1
    if center - radius > 0:
        return 1
    return 0


def exp_upper_raw(argument: int) -> int:
    if argument <= 0:
        return ONE
    term = ONE
    result = ONE
    for degree in range(1, 33):
        term = -((-(term * argument)) // (ONE * degree))
        result += term
    next_term = -((-(term * argument)) // (ONE * 33))
    ratio = -((-argument) // 34)
    tail = -((-(next_term * ONE)) // (ONE - ratio))
    return result + tail


def advance(center: tuple[int, ...], radius: int, zs: Interval, step_raw: int) -> dict[str, object]:
    initial = tuple((component - radius, component + radius) for component in center)
    box, iterations, contraction = picard_box(initial, zs, step_raw)
    center_intervals = tuple((component, component) for component in center)
    center_box, _, _ = picard_box(center_intervals, zs, step_raw)
    center_coeff = coefficients(center_intervals, zs, step_raw, ORDER - 1)
    wide_coeff = coefficients(center_box, zs, step_raw, ORDER)
    next_center = []
    local_radius = 0
    for axis in range(4):
        enclosure = add(total(center_coeff[axis]), wide_coeff[axis][ORDER])
        midpoint = (enclosure[0] + enclosure[1]) // 2
        next_center.append(midpoint)
        local_radius = max(local_radius, midpoint - enclosure[0], enclosure[1] - midpoint)
    mu = logarithmic_norm(box, zs)
    mu_h = -((-(max(mu, 0) * step_raw)) // ONE)
    if mu_h >= ONE:
        raise ValueError("logarithmic-norm amplification denominator is nonpositive")
    amplification = exp_upper_raw(mu_h)
    propagated = -((-(radius * amplification)) // ONE)
    next_radius = propagated + local_radius
    if next_radius >= RADIUS_LIMIT:
        raise ValueError("arithmetic radius reached the frozen refusal limit")
    return {
        "center": tuple(next_center), "radius": next_radius, "local_radius": local_radius,
        "mu": mu, "mu_h": mu_h, "amplification": amplification,
        "picard_iterations": iterations,
        "contraction": contraction, "box": box,
    }


def locate_event(center: tuple[int, ...], radius: int, zs: Interval, step_raw: int) -> dict[str, object]:
    low, high = 0, step_raw
    low_result: dict[str, object] = {"center": center, "radius": radius}
    high_result = advance(center, radius, zs, high)
    if sign(high_result["center"][2], int(high_result["radius"])) != 1:
        raise ValueError("event initial upper bracket is not strictly positive")
    bisections = 0
    for _ in range(EVENT_BISECTIONS):
        middle = (low + high) // 2
        result = advance(center, radius, zs, middle)
        middle_center = result["center"]
        middle_radius = int(result["radius"])
        middle_sign = sign(middle_center[2], middle_radius)
        if middle_sign < 0:
            low, low_result = middle, result
        elif middle_sign > 0:
            high, high_result = middle, result
        else:
            break
        bisections += 1
    if sign(low_result["center"][2], int(low_result["radius"])) != -1:
        raise ValueError("event lower bracket is not strictly negative")
    if sign(high_result["center"][2], int(high_result["radius"])) != 1:
        raise ValueError("event upper bracket is not strictly positive")
    if high - low > ONE >> 50:
        raise ValueError(f"event bracket exceeds 2^-50 after {bisections} decisive bisections: width_raw={high - low}")
    event_initial = tuple((component - int(low_result["radius"]), component + int(low_result["radius"])) for component in low_result["center"])
    event_box, iterations, contraction = picard_box(event_initial, zs, high - low)
    normal = sub(mul(event_box[0], event_box[1]), zs)
    if normal[0] <= 0:
        raise ValueError("event normal velocity is not strictly positive")
    return {
        "low": low, "high": high, "normal": normal,
        "bisections": bisections, "picard_iterations": iterations, "contraction": contraction,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(words: list[int]) -> bytes:
    return b"".join(word.to_bytes(28, "little", signed=True) for word in words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_initial, zs = initial_state()
    center = []
    radius = 0
    for lower, upper in raw_initial:
        midpoint = (lower + upper) // 2
        center.append(midpoint)
        radius = max(radius, midpoint - lower, upper - midpoint)
    current_center = tuple(center)
    initial_center = current_center
    initial_radius = radius
    armed = False
    events: list[dict[str, object]] = []
    rows: list[list[object]] = []
    max_radius = radius
    max_local_radius = 0
    max_picard_iterations = 0
    max_contraction = 0
    minimum_step_bits = MAX_STEP_BITS
    time_raw = 0
    partition_starts: list[tuple[int, int, int, tuple[int, ...], int, bool, int]] = [
        (0, 0, 843, current_center, radius, False, 0),
    ]
    for step in range(1, MAX_STEPS + 1):
        before_sign = sign(current_center[2], radius)
        last_error: ValueError | None = None
        result: dict[str, object] | None = None
        accepted_step_bits = MAX_STEP_BITS
        for step_bits in range(MAX_STEP_BITS, MIN_STEP_BITS + 1):
            candidate_step = ONE >> step_bits
            try:
                result = advance(current_center, radius, zs, candidate_step)
                accepted_step_bits = step_bits
                break
            except ValueError as error:
                last_error = error
        if result is None:
            raise ValueError(f"chain step {step}: all adaptive steps refused: {last_error}") from last_error
        accepted_step_raw = ONE >> accepted_step_bits
        next_center = result["center"]
        next_radius = int(result["radius"])
        after_sign = sign(next_center[2], next_radius)
        event_index = 0
        if after_sign < 0:
            armed = True
        if armed and before_sign < 0 and after_sign > 0:
            try:
                event = locate_event(current_center, radius, zs, accepted_step_raw)
            except ValueError as error:
                raise ValueError(f"event at chain step {step}, radius_raw={radius}: {error}") from error
            event_index = len(events) + 1
            event["index"] = event_index
            event["base_step"] = step - 1
            event["base_time"] = time_raw
            events.append(event)
            armed = False
        rows.append([
            step, time_raw, time_raw + accepted_step_raw, accepted_step_raw, accepted_step_bits,
            *next_center, next_radius, result["local_radius"], result["mu"],
            result["mu_h"], result["amplification"], result["picard_iterations"], result["contraction"],
            before_sign, after_sign, event_index,
        ])
        max_radius = max(max_radius, next_radius)
        max_local_radius = max(max_local_radius, int(result["local_radius"]))
        max_picard_iterations = max(max_picard_iterations, int(result["picard_iterations"]))
        max_contraction = max(max_contraction, int(result["contraction"]))
        minimum_step_bits = max(minimum_step_bits, accepted_step_bits)
        time_raw += accepted_step_raw
        current_center, radius = next_center, next_radius
        if step == 843:
            partition_starts.append((1, step, 843, current_center, radius, armed, len(events)))
        if len(events) == 2:
            break
    if len(events) != 2:
        raise SystemExit(f"expected two events, found {len(events)} after {len(rows)} steps")
    with (args.out_dir / "chain.tsv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["STEP", "TIME_START_RAW", "TIME_END_RAW", "STEP_RAW", "STEP_BITS", "CENTER_X_RAW", "CENTER_Y_RAW", "CENTER_W_RAW", "CENTER_ELL_RAW", "RADIUS_RAW", "LOCAL_RADIUS_RAW", "MU_RAW", "MU_H_RAW", "AMPLIFICATION_RAW", "PICARD_ITERATIONS", "CONTRACTION_RAW", "BEFORE_SIGN", "AFTER_SIGN", "EVENT_INDEX"])
        writer.writerows(rows)
    with (args.out_dir / "events.tsv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["EVENT_INDEX", "BASE_STEP", "LOCAL_LOW_RAW", "LOCAL_HIGH_RAW", "GLOBAL_LOW_RAW", "GLOBAL_HIGH_RAW", "NORMAL_LOWER_RAW", "NORMAL_UPPER_RAW", "BISECTIONS", "PICARD_ITERATIONS", "CONTRACTION_RAW"])
        for event in events:
            base = int(event["base_time"])
            normal = event["normal"]
            writer.writerow([event["index"], event["base_step"], event["low"], event["high"], base + int(event["low"]), base + int(event["high"]), normal[0], normal[1], event["bisections"], event["picard_iterations"], event["contraction"]])
    input_words = [*initial_center, initial_radius, zs[0], zs[1]]
    output_words = [word for row in rows for word in (*row[1:10], row[-1])]
    hardware_input_words = [
        word
        for partition_id, start_step, count, start_center, start_radius, start_armed, prior_events in partition_starts
        for word in (
            partition_id, start_step, count, rows[start_step][1] if start_step else 0,
            *start_center, start_radius, zs[0], zs[1], int(start_armed), prior_events,
        )
    ]
    (args.out_dir / "inputs.bin").write_bytes(encode(input_words))
    (args.out_dir / "expected.bin").write_bytes(encode(output_words))
    (args.out_dir / "hardware_inputs.bin").write_bytes(encode(hardware_input_words))
    with (args.out_dir / "partitions.tsv").open("w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["PARTITION_ID", "START_STEP", "STEPS", "OUTPUT_WORD_OFFSET", "OUTPUT_WORDS", "ARMED", "PRIOR_EVENTS"])
        for partition_id, start_step, count, _center, _radius, start_armed, prior_events in partition_starts:
            writer.writerow([partition_id, start_step, count, start_step * 10, count * 10, int(start_armed), prior_events])
    summary = [
        "SCHEMA=sounio.cs6.u250-target23-chained-taylor41-vectors.v1",
        f"CONTRACT_SHA256={digest(CONTRACT)}", f"GENERATOR_SHA256={digest(Path(__file__))}",
        f"FRACTION_BITS={FRACTION_BITS}", f"TAYLOR_ORDER={ORDER}", f"STEPS={len(rows)}",
        f"FINAL_TIME_RAW={time_raw}", f"SMALLEST_STEP_BITS={minimum_step_bits}",
        "EVENTS=2", f"EVENT_BISECTIONS={EVENT_BISECTIONS}",
        f"MAX_RADIUS_RAW={max_radius}", f"MAX_LOCAL_RADIUS_RAW={max_local_radius}",
        f"MAX_PICARD_ITERATIONS={max_picard_iterations}", f"MAX_CONTRACTION_RAW={max_contraction}",
        f"CHAIN_SHA256={digest(args.out_dir / 'chain.tsv')}", f"EVENTS_SHA256={digest(args.out_dir / 'events.tsv')}",
        f"INPUTS_SHA256={digest(args.out_dir / 'inputs.bin')}", f"EXPECTED_SHA256={digest(args.out_dir / 'expected.bin')}",
        "HARDWARE_PARTITIONS=2", "HARDWARE_INPUT_WORDS=26", "HARDWARE_OUTPUT_WORDS=16860",
        f"HARDWARE_INPUTS_SHA256={digest(args.out_dir / 'hardware_inputs.bin')}",
        f"PARTITIONS_SHA256={digest(args.out_dir / 'partitions.tsv')}",
        "TWO_RETURN_CHAIN_CERTIFICATE=true", "HLS_CSIM=false", "HLS_SYNTHESIS=false",
        "PHYSICAL_FPGA_EXECUTION=false", "DUAL_U250_EXECUTION=false",
        "FULL_ORBIT_CERTIFICATE=false", "LEAF_WIDE_CERTIFICATE=false", "GLOBAL_HPG_CERTIFICATE=false",
        "NOVELTY_OR_PRIORITY_CLAIMED=false", "OPEN_PROBLEM_SOLVED=false",
    ]
    (args.out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
