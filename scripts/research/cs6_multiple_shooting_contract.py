#!/usr/bin/env python3
"""Directed-decimal local flow enclosures for the CS6 periodic-orbit target.

This script is intentionally narrower than a periodic-orbit proof.  It uses
directed Decimal arithmetic, a Picard self-inclusion tube, and an order-p
Taylor expansion with a Lagrange remainder bound to enclose short pieces of
the polynomial CS6 flow.  A multiple-shooting run resets the coordinate box at
each declared node to avoid the wrapping explosion of a single long box.

What a green full run means:
  * every short flow segment has a proved local existence tube;
  * every Taylor endpoint contains the exact endpoint for its declared node;
  * the residual vector of the declared multiple-shooting witness is bounded.

What it does not mean:
  * the local segments have been glued by interval Newton/Krawczyk;
  * a periodic orbit, hyperbolicity, a homoclinic intersection, or chaos is
    proved.

The full run has no third-party dependency.  Python's decimal.Context is the
arithmetic TCB for directed floor/ceiling operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Context, Decimal, Overflow as DecimalOverflow
from decimal import ROUND_CEILING, ROUND_FLOOR
from pathlib import Path


PRECISION = 70
LOWER = Context(
    prec=PRECISION,
    rounding=ROUND_FLOOR,
    Emin=-999999,
    Emax=999999,
)
UPPER = Context(
    prec=PRECISION,
    rounding=ROUND_CEILING,
    Emin=-999999,
    Emax=999999,
)

D0 = Decimal(0)
D1 = Decimal(1)
D2 = Decimal(2)

SYSTEM = {
    "x_dot": "2*y^2-x*y",
    "y_dot": "x*y-(1/2)*y*z",
    "z_dot": "x*y-z",
}

FULL_CONFIG = {
    "seed": (
        "15.186446520640915",
        "10.90854319476487",
        "22.3274637391",
    ),
    "horizon": "29.510309219673534",
    "segment_span": "0.1",
    "base_dt": "0.005",
    "taylor_order": 18,
}


class ProofRefused(RuntimeError):
    """Raised when a local enclosure cannot be established."""


class Interval:
    __slots__ = ("lo", "hi")

    def __init__(self, lo: Decimal | str | int, hi=None):
        self.lo = lo if isinstance(lo, Decimal) else Decimal(lo)
        raw_hi = lo if hi is None else hi
        self.hi = raw_hi if isinstance(raw_hi, Decimal) else Decimal(raw_hi)
        if self.lo > self.hi:
            raise ValueError((self.lo, self.hi))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(
            LOWER.add(self.lo, other.lo),
            UPPER.add(self.hi, other.hi),
        )

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(
            LOWER.subtract(self.lo, other.hi),
            UPPER.subtract(self.hi, other.lo),
        )

    def __neg__(self) -> "Interval":
        return Interval(LOWER.minus(self.hi), UPPER.minus(self.lo))

    def __mul__(self, other: "Interval") -> "Interval":
        lower_products = (
            LOWER.multiply(self.lo, other.lo),
            LOWER.multiply(self.lo, other.hi),
            LOWER.multiply(self.hi, other.lo),
            LOWER.multiply(self.hi, other.hi),
        )
        upper_products = (
            UPPER.multiply(self.lo, other.lo),
            UPPER.multiply(self.lo, other.hi),
            UPPER.multiply(self.hi, other.lo),
            UPPER.multiply(self.hi, other.hi),
        )
        return Interval(min(lower_products), max(upper_products))

    def divide_by_positive_int(self, value: int) -> "Interval":
        if value <= 0:
            raise ValueError(value)
        denominator = Decimal(value)
        return Interval(
            LOWER.divide(self.lo, denominator),
            UPPER.divide(self.hi, denominator),
        )

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def inflated(self, fraction: Decimal, absolute: Decimal) -> "Interval":
        magnitude = max(abs(self.lo), abs(self.hi), D1)
        radius = max(UPPER.multiply(magnitude, fraction), absolute)
        return Interval(
            LOWER.subtract(self.lo, radius),
            UPPER.add(self.hi, radius),
        )

    def strict_subset(self, other: "Interval") -> bool:
        return self.lo > other.lo and self.hi < other.hi

    def subset(self, other: "Interval") -> bool:
        return self.lo >= other.lo and self.hi <= other.hi

    def contains(self, value: Decimal) -> bool:
        return self.lo <= value <= self.hi

    def contains_zero(self) -> bool:
        return self.lo <= D0 <= self.hi

    def width(self) -> Decimal:
        return UPPER.subtract(self.hi, self.lo)

    def max_abs(self) -> Decimal:
        return max(abs(self.lo), abs(self.hi))

    def pair(self) -> list[str]:
        return [str(self.lo), str(self.hi)]


ZERO = Interval(D0)
ONE = Interval(D1)
TWO = Interval(D2)
HALF = Interval(Decimal("0.5"))


def interval_sum(values) -> Interval:
    result = ZERO
    for value in values:
        result = result + value
    return result


def field(state: tuple[Interval, Interval, Interval]):
    x, y, z = state
    xy = x * y
    return TWO * (y * y) - xy, xy - HALF * (y * z), xy - z


def taylor_coefficients(state, order: int):
    """Return autonomous-flow Taylor coefficients through ``order``.

    If x(t)=sum x_k t^k (and similarly for y,z), coefficient matching in
    x'=2y^2-xy, y'=xy-yz/2, z'=xy-z gives the recurrence below.  Applying it
    to interval state inputs encloses the derivative coefficient polynomial at
    every state in that box.
    """

    xs = [state[0]]
    ys = [state[1]]
    zs = [state[2]]
    for k in range(order):
        yy = interval_sum(ys[i] * ys[k - i] for i in range(k + 1))
        xy = interval_sum(xs[i] * ys[k - i] for i in range(k + 1))
        yz = interval_sum(ys[i] * zs[k - i] for i in range(k + 1))
        xs.append((TWO * yy - xy).divide_by_positive_int(k + 1))
        ys.append((xy - HALF * yz).divide_by_positive_int(k + 1))
        zs.append((xy - zs[k]).divide_by_positive_int(k + 1))
    return xs, ys, zs


def interval_power(value: Interval, exponent: int) -> Interval:
    result = ONE
    for _ in range(exponent):
        result = result * value
    return result


def evaluate_polynomial(coefficients: list[Interval], h: Interval) -> Interval:
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = coefficient + h * result
    return result


def picard_tube(state, h: Interval):
    """Find B with X0+[0,h]f(B) strictly inside B."""

    time = Interval(D0, h.hi)
    initial_flow = field(state)
    candidate = tuple(
        state[i].hull(state[i] + time * initial_flow[i]).inflated(
            Decimal("1e-5"), Decimal("1e-60")
        )
        for i in range(3)
    )
    for iteration in range(1, 31):
        flow_box = field(candidate)
        image = tuple(state[i] + time * flow_box[i] for i in range(3))
        if all(image[i].strict_subset(candidate[i]) for i in range(3)):
            margins = []
            for axis in range(3):
                margins.append(LOWER.subtract(image[axis].lo, candidate[axis].lo))
                margins.append(LOWER.subtract(candidate[axis].hi, image[axis].hi))
            return candidate, iteration, min(margins)
        candidate = tuple(
            candidate[i].hull(image[i]).inflated(
                Decimal("1e-5"), Decimal("1e-60")
            )
            for i in range(3)
        )
    raise ProofRefused(f"Picard self-inclusion failed for h={h.hi}")


def taylor_step(state, h_decimal: Decimal, order: int):
    h = Interval(h_decimal)
    tube, iterations, margin = picard_tube(state, h)
    local_coefficients = taylor_coefficients(state, order)
    remainder_coefficients = taylor_coefficients(tube, order + 1)
    remainder_scale = interval_power(h, order + 1)
    endpoint = []
    for axis in range(3):
        polynomial = evaluate_polynomial(local_coefficients[axis], h)
        remainder = remainder_coefficients[axis][order + 1] * remainder_scale
        endpoint.append(polynomial + remainder)
    result = tuple(endpoint)
    if not all(result[i].subset(tube[i]) for i in range(3)):
        raise ProofRefused("Taylor endpoint escaped the proved Picard tube")
    return result, {
        "picard_iterations": iterations,
        "picard_margin": margin,
        "tube_lo": tuple(axis.lo for axis in tube),
        "tube_hi": tuple(axis.hi for axis in tube),
    }


def advance(state, duration: Decimal, base_dt: Decimal, order: int):
    elapsed = D0
    stats = {
        "steps": 0,
        "max_picard_iterations": 0,
        "rejected_steps": 0,
        "smallest_dt": base_dt,
        "min_picard_margin": None,
        "tube_lo_min": [axis.lo for axis in state],
        "tube_hi_max": [axis.hi for axis in state],
    }
    while elapsed < duration:
        h = min(base_dt, LOWER.subtract(duration, elapsed))
        while True:
            try:
                next_state, step_stats = taylor_step(state, h, order)
                break
            except (ProofRefused, DecimalOverflow) as error:
                h = LOWER.divide(h, D2)
                stats["rejected_steps"] += 1
                if h < Decimal("1e-8"):
                    raise ProofRefused(
                        f"adaptive step underflow at local t={elapsed}"
                    ) from error
        state = next_state
        elapsed = UPPER.add(elapsed, h)
        stats["steps"] += 1
        stats["max_picard_iterations"] = max(
            stats["max_picard_iterations"], step_stats["picard_iterations"]
        )
        stats["smallest_dt"] = min(stats["smallest_dt"], h)
        margin = step_stats["picard_margin"]
        current_margin = stats["min_picard_margin"]
        stats["min_picard_margin"] = (
            margin if current_margin is None else min(current_margin, margin)
        )
        for axis in range(3):
            stats["tube_lo_min"][axis] = min(
                stats["tube_lo_min"][axis], step_stats["tube_lo"][axis]
            )
            stats["tube_hi_max"][axis] = max(
                stats["tube_hi_max"][axis], step_stats["tube_hi"][axis]
            )
    return state, stats


def interval_midpoint(axis: Interval) -> Decimal:
    value = LOWER.divide(LOWER.add(axis.lo, axis.hi), D2)
    return min(max(value, axis.lo), axis.hi)


def canonical_records_digest(records) -> str:
    encoded = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def run_multiple_shooting(progress: bool = True):
    initial_point = tuple(Decimal(value) for value in FULL_CONFIG["seed"])
    node = initial_point
    elapsed = D0
    stop = Decimal(FULL_CONFIG["horizon"])
    segment_span = Decimal(FULL_CONFIG["segment_span"])
    base_dt = Decimal(FULL_CONFIG["base_dt"])
    order = FULL_CONFIG["taylor_order"]
    records = []
    aggregate = {
        "total_duration": D0,
        "total_integrator_steps": 0,
        "total_rejected_steps": 0,
        "max_endpoint_width": D0,
        "min_picard_margin": None,
        "tube_lo_min": list(initial_point),
        "tube_hi_max": list(initial_point),
        "intermediate_zero_residuals": 0,
    }

    while elapsed < stop:
        segment_start = elapsed
        duration = min(segment_span, LOWER.subtract(stop, elapsed))
        start_box = tuple(Interval(value) for value in node)
        endpoint, stats = advance(start_box, duration, base_dt, order)
        aggregate["total_integrator_steps"] += stats["steps"]
        aggregate["total_rejected_steps"] += stats["rejected_steps"]
        aggregate["max_endpoint_width"] = max(
            aggregate["max_endpoint_width"],
            *(axis.width() for axis in endpoint),
        )
        margin = stats["min_picard_margin"]
        prior_margin = aggregate["min_picard_margin"]
        aggregate["min_picard_margin"] = (
            margin if prior_margin is None else min(prior_margin, margin)
        )
        for axis in range(3):
            aggregate["tube_lo_min"][axis] = min(
                aggregate["tube_lo_min"][axis], stats["tube_lo_min"][axis]
            )
            aggregate["tube_hi_max"][axis] = max(
                aggregate["tube_hi_max"][axis], stats["tube_hi_max"][axis]
            )

        is_final = UPPER.add(elapsed, duration) == stop
        target = initial_point if is_final else tuple(
            interval_midpoint(axis) for axis in endpoint
        )
        residual = tuple(endpoint[i] - Interval(target[i]) for i in range(3))
        if not is_final:
            if not all(axis.contains_zero() for axis in residual):
                raise ProofRefused("an intermediate midpoint residual missed zero")
            aggregate["intermediate_zero_residuals"] += 1

        records.append(
            {
                "index": len(records),
                "time_start": str(segment_start),
                "time_end": str(UPPER.add(segment_start, duration)),
                "duration": str(duration),
                "start": [str(value) for value in node],
                "endpoint": [axis.pair() for axis in endpoint],
                "target": [str(value) for value in target],
                "residual": [axis.pair() for axis in residual],
                "residual_abs_upper": [str(axis.max_abs()) for axis in residual],
                "integrator_steps": stats["steps"],
                "rejected_steps": stats["rejected_steps"],
                "smallest_dt": str(stats["smallest_dt"]),
                "min_picard_margin": str(stats["min_picard_margin"]),
            }
        )
        elapsed = UPPER.add(elapsed, duration)
        aggregate["total_duration"] = UPPER.add(
            aggregate["total_duration"], duration
        )
        node = target
        if progress and (len(records) % 25 == 0 or is_final):
            print(
                "CS6_FULL_PROGRESS "
                f"segments={len(records)} t={elapsed} "
                f"steps={aggregate['total_integrator_steps']} "
                f"max_width={aggregate['max_endpoint_width']}",
                flush=True,
            )

    if elapsed != stop or aggregate["total_duration"] != stop:
        raise ProofRefused(
            "temporal ledger does not sum exactly to the declared horizon"
        )
    for index, record in enumerate(records):
        expected_start = D0 if index == 0 else Decimal(records[index - 1]["time_end"])
        if Decimal(record["time_start"]) != expected_start:
            raise ProofRefused(f"temporal ledger gap or overlap at segment {index}")
        computed_end = LOWER.add(
            Decimal(record["time_start"]), Decimal(record["duration"])
        )
        if computed_end != Decimal(record["time_end"]):
            raise ProofRefused(f"temporal ledger duration mismatch at segment {index}")

    initial_box = tuple(Interval(value) for value in initial_point)
    initial_normal_velocity = field(initial_box)[2]
    final_residual = records[-1]["residual"]
    final_residual_abs = records[-1]["residual_abs_upper"]
    summary = {
        "schema": "sounio.cs6.multiple-shooting-local-enclosures.v1",
        "evidence_label": "EXECUTABLE_LOCAL_INTERVAL_ENCLOSURES",
        "system": SYSTEM,
        "section": {
            "equation": "z=22.3274637391",
            "direction": "positive",
            "initial_normal_velocity": initial_normal_velocity.pair(),
        },
        "configuration": {
            "decimal_precision": PRECISION,
            **FULL_CONFIG,
        },
        "arithmetic_tcb": {
            "implementation": "Python decimal.Context",
            "lower_rounding": "ROUND_FLOOR",
            "upper_rounding": "ROUND_CEILING",
            "formal_verification": False,
        },
        "segments": len(records),
        "temporal_ledger": {
            "full_span_segments": sum(
                record["duration"] == FULL_CONFIG["segment_span"]
                for record in records
            ),
            "remainder_span": records[-1]["duration"],
            "total_duration": str(aggregate["total_duration"]),
            "no_gaps_or_overlaps": True,
        },
        "total_integrator_steps": aggregate["total_integrator_steps"],
        "total_rejected_steps": aggregate["total_rejected_steps"],
        "intermediate_zero_residuals": aggregate["intermediate_zero_residuals"],
        "max_endpoint_width": str(aggregate["max_endpoint_width"]),
        "min_picard_margin": str(aggregate["min_picard_margin"]),
        "tube_lo_min": [str(value) for value in aggregate["tube_lo_min"]],
        "tube_hi_max": [str(value) for value in aggregate["tube_hi_max"]],
        "final_closure_residual": final_residual,
        "final_closure_residual_abs_upper": final_residual_abs,
        "records_sha256": canonical_records_digest(records),
        "local_enclosures_glued": False,
        "periodic_orbit_proved": False,
        "hyperbolicity_proved": False,
        "homoclinic_or_covering_proved": False,
        "chaos_proved": False,
        "remaining_obligation": (
            "global multiple-shooting interval Newton/Krawczyk inclusion"
        ),
    }
    return summary, records


def smoke_tests():
    third = ONE.divide_by_positive_int(3)
    reconstructed = third * Interval(3)
    if not reconstructed.contains(D1):
        raise ProofRefused("directed division/multiplication lost 1")
    print("CS6_DIRECTED_DECIMAL_INTERVAL PASS")

    equilibrium = (Interval(4), Interval(2), Interval(8))
    enclosed, _ = advance(equilibrium, Decimal("0.1"), Decimal("0.005"), 12)
    equilibrium_values = (Decimal(4), Decimal(2), Decimal(8))
    if not all(
        enclosed[i].contains(value)
        for i, value in enumerate(equilibrium_values)
    ):
        raise ProofRefused("known CS6 equilibrium escaped its enclosure")
    print("CS6_EQUILIBRIUM_ENCLOSURE PASS")

    seed = tuple(Interval(value) for value in FULL_CONFIG["seed"])
    endpoint, stats = advance(seed, Decimal("0.1"), Decimal("0.005"), 18)
    width = max(axis.width() for axis in endpoint)
    if width > Decimal("3e-16"):
        raise ProofRefused(f"local segment width regression: {width}")
    if stats["min_picard_margin"] <= D0:
        raise ProofRefused("Picard inclusion margin is not positive")
    print(f"CS6_LOCAL_TAYLOR_PICARD PASS width={width}")
    print("CS6_SMOKE_VERDICT PASS")


def check_witness(path: Path):
    witness = json.loads(path.read_text(encoding="ascii"))
    if witness.get("schema") != "sounio.cs6.multiple-shooting-local-enclosures.v1":
        raise ProofRefused("wrong witness schema")
    print("CS6_WITNESS_SCHEMA PASS")

    forbidden = (
        "local_enclosures_glued",
        "periodic_orbit_proved",
        "hyperbolicity_proved",
        "homoclinic_or_covering_proved",
        "chaos_proved",
    )
    if any(witness.get(field) is not False for field in forbidden):
        raise ProofRefused("an unproved promotion bit is set")
    print("CS6_WITNESS_NONPROMOTION PASS")

    if witness.get("segments") != 296:
        raise ProofRefused("unexpected segment count")
    ledger = witness.get("temporal_ledger", {})
    if ledger.get("full_span_segments") != 295:
        raise ProofRefused("unexpected full-span segment count")
    if ledger.get("remainder_span") != "0.010309219673534":
        raise ProofRefused("unexpected final remainder span")
    if Decimal(ledger.get("total_duration", "NaN")) != Decimal(
        FULL_CONFIG["horizon"]
    ):
        raise ProofRefused("witness temporal ledger misses the horizon")
    if ledger.get("no_gaps_or_overlaps") is not True:
        raise ProofRefused("witness does not assert a contiguous temporal ledger")
    if witness.get("total_integrator_steps") != 5903:
        raise ProofRefused("unexpected local integrator step count")
    if witness.get("intermediate_zero_residuals") != 295:
        raise ProofRefused("not all intermediate shooting residuals contain zero")
    if Decimal(witness["max_endpoint_width"]) > Decimal("3e-16"):
        raise ProofRefused("witness local width exceeds the declared cap")
    if max(Decimal(v) for v in witness["final_closure_residual_abs_upper"]) > Decimal("1e-12"):
        raise ProofRefused("witness closure residual exceeds the declared cap")
    if Decimal(witness["section"]["initial_normal_velocity"][0]) <= D0:
        raise ProofRefused("phase-section normal velocity is not positive")
    print("CS6_WITNESS_NUMERICS PASS")
    print("CS6_WITNESS_VERDICT PASS")


def compare_replay(summary, witness_path: Path):
    witness = json.loads(witness_path.read_text(encoding="ascii"))
    normalized_summary = json.loads(
        json.dumps(summary, ensure_ascii=True, sort_keys=True)
    )
    mismatches = [
        key
        for key, value in normalized_summary.items()
        if witness.get(key) != value
    ]
    if mismatches:
        raise ProofRefused(
            "full replay differs from the compact witness in: "
            + ", ".join(mismatches)
        )
    print("CS6_FULL_REPLAY_WITNESS_MATCH PASS")


def write_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--check-witness", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--full-output", type=Path)
    parser.add_argument("--replay-witness", type=Path)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    if args.check_witness:
        check_witness(args.check_witness)
        return
    if args.mode == "smoke":
        smoke_tests()
        return

    summary, records = run_multiple_shooting(progress=not args.quiet_progress)
    if args.replay_witness:
        compare_replay(summary, args.replay_witness)
    if args.summary_output:
        write_json(args.summary_output, summary)
    if args.full_output:
        write_json(args.full_output, {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    print("CS6_FULL_LOCAL_ENCLOSURE_VERDICT PASS")


if __name__ == "__main__":
    main()
