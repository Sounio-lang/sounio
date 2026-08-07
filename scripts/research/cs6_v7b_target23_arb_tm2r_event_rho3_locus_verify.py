#!/usr/bin/env python3
"""Fail-closed verifier for the residual rho3 locus diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-rho3-locus.v1"
EXPECTED_CENTER = Fraction(
    -229481176335118857750998868969216857385473740087351435274360730211439744304527,
    29642774844752946028434172162224104410437116074403984394101141506025761187823616,
)
EXPECTED_PATH = [
    "DOWN_RHO0L",
    "DOWN_ETAH",
    "DOWN_RHO0L",
    "DOWN_ETAL",
    "DOWN_RHO0L",
    "DOWN_ETAL",
    "DOWN_RHO0L",
    "DOWN_RHO1H",
    "DOWN_ETAH",
    "DOWN_RHO0L",
    "DOWN_RHO1H",
    "DOWN_ETAH",
]
EXPECTED_EVENT_LOCAL = (
    "0a47c711f8442bfe4bb3ce844cb247c74aaf9922f5dcf224411f58acd2bb3146"
)
EXPECTED_EVENT_CENTERED = (
    "37464bf6b240e9dc621aed3b4fcdf02b8276ce44a21623001aea0d62f6ce29f7"
)
ALLOWED_CLASSIFICATIONS = {
    "RECONDITION_COLLAPSES_RESIDUAL_RANK",
    "FLOW_ERASES_RHO3",
    "PROJECTION_ERASES_RHO3",
    "RHO3_PRESERVED_ALL_STAGES",
    "MIXED_RHO3_LOCI",
    "NO_SCALE_REACHED_PROJECTION",
    "IMPLEMENTATION_INCONSISTENCY",
}
SCALE_LOCI = {
    "FLOW_ERASES_RHO3",
    "PROJECTION_ERASES_RHO3",
    "RECONDITION_COLLAPSES_RESIDUAL_RANK",
    "RHO3_PRESERVED_ALL_STAGES",
}


def fail(message: str) -> None:
    raise SystemExit(f"rho3-locus verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def interval(value: object, label: str) -> tuple[Fraction, Fraction]:
    if not isinstance(value, list) or len(value) != 2:
        fail(f"{label} is not an interval")
    lower, upper = (Fraction(str(item)) for item in value)
    if lower > upper:
        fail(f"{label} has reversed endpoints")
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--centered-worker", type=Path, required=True)
    parser.add_argument("--prior-worker", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--event-local-receipt", type=Path, required=True)
    parser.add_argument("--event-centered-receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    require(payload, "worker_source_sha256", sha256(args.worker))
    require(payload, "centered_worker_source_sha256", sha256(args.centered_worker))
    require(payload, "prior_worker_source_sha256", sha256(args.prior_worker))
    require(payload, "carrier_source_sha256", sha256(args.carrier))
    require(payload, "chain_source_sha256", sha256(args.chain))
    require(payload, "adaptive_source_sha256", sha256(args.adaptive))
    require(payload, "event_source_sha256", sha256(args.event))
    require(payload, "base_source_sha256", sha256(args.base))
    require(
        payload,
        "prior_event_local_receipt_sha256",
        sha256(args.event_local_receipt),
    )
    require(
        payload,
        "prior_event_centered_receipt_sha256",
        sha256(args.event_centered_receipt),
    )
    require(payload, "prior_event_local_receipt_sha256", EXPECTED_EVENT_LOCAL)
    require(payload, "prior_event_centered_receipt_sha256", EXPECTED_EVENT_CENTERED)
    require(payload, "tile_id", "XLEL")
    require(payload, "critical_path", EXPECTED_PATH)
    require(payload, "critical_depth", len(EXPECTED_PATH))
    require(payload, "source_degree", 2)
    require(payload, "time_taylor_order", 12)
    require(payload, "arb_precision_bits", 256)
    require(payload, "lost_variable_index", 5)
    require(payload, "lost_variable_name", "rho3")
    require(
        payload,
        "reconditioner",
        "cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker."
        "point_coefficient_recondition",
    )
    require(payload, "diagnostic_complete", True)
    require(payload, "point_fallback_used", False)
    require(payload, "box_flattening_used", False)
    require(payload, "full_transport_attempted", False)
    require(payload, "covering_relation_certified", False)
    require(payload, "recurrent_graph_certified", False)
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    if Fraction(str(payload.get("predictor_center_q"))) != EXPECTED_CENTER:
        fail("predictor center does not match the frozen exact center")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks]
    if len(names) != len(set(names)):
        fail("implementation check names are not unique")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    chart = payload.get("rho3_locus_chart")
    if not isinstance(chart, dict):
        fail("rho3 locus chart is absent")
    if Fraction(str(chart.get("fixed_shift_q"))) != EXPECTED_CENTER:
        fail("chart fixed shift is wrong")

    scales = chart.get("scales")
    if not isinstance(scales, list) or not scales:
        fail("chart scales are absent")
    projection_scales = [
        scale
        for scale in scales
        if isinstance(scale, dict) and "raw_projection_rho3_positive" in scale
    ]
    require(payload, "classification", chart.get("status") if checks_passed else "IMPLEMENTATION_INCONSISTENCY")
    if not checks_passed:
        if payload.get("classification") != "IMPLEMENTATION_INCONSISTENCY":
            fail("failed checks must classify as IMPLEMENTATION_INCONSISTENCY")
    else:
        if payload.get("classification") not in ALLOWED_CLASSIFICATIONS:
            fail("classification is not an allowed locus class")

    locus_counts: dict[str, int] = {}
    for scale in projection_scales:
        status = scale.get("status")
        if status not in SCALE_LOCI:
            fail(f"projection scale has unknown locus status {status!r}")
        locus_counts[str(status)] = locus_counts.get(str(status), 0) + 1
        raw = scale.get("raw_projection_rho3_positive")
        recond = scale.get("reconditioned_rho3_positive")
        if not isinstance(raw, bool) or not isinstance(recond, bool):
            fail("projection scale lacks boolean rho3 flags")
        if status == "RECONDITION_COLLAPSES_RESIDUAL_RANK" and not (raw and not recond):
            fail("recondition-collapse status contradicts rho3 flags")
        if status == "FLOW_ERASES_RHO3":
            event_weights = scale.get("event_state_variable_weights_section")
            if not isinstance(event_weights, list) or len(event_weights) != 6:
                fail("flow-erasure scale lacks event section weights")
            _lower, upper = interval(event_weights[5], "event rho3 weight")
            if upper > 0:
                fail("flow-erasure status but event rho3 weight is positive")
        if status == "PROJECTION_ERASES_RHO3" and raw:
            fail("projection-erasure status but raw rho3 is positive")
        rank = scale.get("residual_pure_direction_rank")
        if not isinstance(rank, int) or rank < 0 or rank > 4:
            fail("residual pure-direction rank is malformed")
        if status == "RECONDITION_COLLAPSES_RESIDUAL_RANK" and rank >= 4:
            fail("recondition collapse claimed but residual rank is full")

    if checks_passed:
        if chart.get("locus_counts") != locus_counts:
            fail("locus_counts summary does not match projection scales")
        if chart.get("projection_scales") != len(projection_scales):
            fail("projection_scales count is wrong")
        if chart.get("status") != payload.get("classification"):
            fail("chart status disagrees with classification")

    print(f"SCHEMA={SCHEMA}")
    print(f"CLASSIFICATION={payload.get('classification')}")
    print(f"IMPLEMENTATION_CHECKS_PASSED={str(checks_passed).lower()}")
    print(f"PROJECTION_SCALES={len(projection_scales)}")
    print(f"LOCUS_COUNTS={json.dumps(locus_counts, sort_keys=True)}")
    print("FULL_TRANSPORT_ATTEMPTED=false")
    print("COVERING_RELATION_CERTIFIED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
