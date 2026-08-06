#!/usr/bin/env python3
"""Derive exact quantitative margins from the event-local receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-local-diagnostic.v2"


def fail(message: str) -> None:
    raise SystemExit(f"event-local analysis error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 30
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def power(scan: dict[str, object], exponent: int) -> dict[str, object]:
    matches = [item for item in scan["scales"] if item["power"] == exponent]
    if len(matches) != 1:
        fail(f"expected exactly one 2^-{exponent} scale")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))
    if payload.get("schema") != SCHEMA:
        fail("wrong receipt schema")
    if payload.get("classification") != "UNRESOLVED_ENCLOSURE":
        fail("receipt is not the unresolved-enclosure outcome")
    if payload.get("implementation_checks_passed") is not True:
        fail("implementation controls did not all pass")
    if payload.get("endpoint_delay_from_crossing_q") != "0":
        fail("captured crossing and production endpoint times differ")

    final = payload["prefix_diagnostics"][-1]
    raw = power(final["raw_symmetric_slab"], 7)
    reconditioned = power(final["reconditioned_symmetric_slab"], 7)
    crossing = power(
        payload["crossing_event_diagnostic"]["raw_symmetric_slab"], 7
    )
    if not (
        raw["predictor"]
        == reconditioned["predictor"]
        == crossing["predictor"]
    ):
        fail("the three predictor intervals are not exactly identical")
    if not all(
        item["status"] == "PREDICTOR_ESCAPED"
        for item in (raw, reconditioned, crossing)
    ):
        fail("a 2^-7 test did not refuse with PREDICTOR_ESCAPED")

    lower, upper = (Fraction(value) for value in raw["predictor"])
    radius = Fraction(1, 128)
    width = upper - lower
    center = (lower + upper) / 2
    half_width = width / 2
    lower_boundary = -radius
    boundary_deficit = lower_boundary - lower
    center_clearance = center - lower_boundary
    if not lower < lower_boundary < upper:
        fail("predictor does not straddle the lower slab boundary")
    if boundary_deficit <= 0 or center_clearance <= 0:
        fail("predictor boundary geometry has the wrong sign")
    minimum_width_divisor = half_width / center_clearance
    maximum_width_multiplier = 1 / minimum_width_divisor
    if minimum_width_divisor <= 1:
        fail("the current predictor would not require width reduction")

    print(f"SCHEMA={SCHEMA}")
    print(f"RECEIPT_SHA256={sha256(args.receipt)}")
    print(f"PREDICTOR_LOWER_Q={lower}")
    print(f"PREDICTOR_UPPER_Q={upper}")
    print(f"PREDICTOR_WIDTH_Q={width}")
    print(f"PREDICTOR_CENTER_Q={center}")
    print(f"SLAB_LOWER_BOUNDARY_Q={lower_boundary}")
    print(f"LOWER_BOUNDARY_DEFICIT_Q={boundary_deficit}")
    print(f"CENTER_CLEARANCE_Q={center_clearance}")
    print(f"MINIMUM_WIDTH_DIVISOR_Q={minimum_width_divisor}")
    print(f"MAXIMUM_WIDTH_MULTIPLIER_Q={maximum_width_multiplier}")
    print(f"PREDICTOR_LOWER_DEC={decimal(lower)}")
    print(f"PREDICTOR_UPPER_DEC={decimal(upper)}")
    print(f"PREDICTOR_WIDTH_DEC={decimal(width)}")
    print(f"PREDICTOR_CENTER_DEC={decimal(center)}")
    print(f"LOWER_BOUNDARY_DEFICIT_DEC={decimal(boundary_deficit)}")
    print(
        "MINIMUM_WIDTH_DIVISOR_DEC="
        f"{decimal(minimum_width_divisor)}"
    )
    print(
        "MAXIMUM_WIDTH_MULTIPLIER_DEC="
        f"{decimal(maximum_width_multiplier)}"
    )
    print("RAW_RECONDITIONED_CROSSING_PREDICTORS_IDENTICAL=true")
    print("IMPLEMENTATION_CHECKS_PASSED=true")
    print("FULL_TRANSPORT_ATTEMPTED=false")
    print("ANALYZED=true")


if __name__ == "__main__":
    main()
