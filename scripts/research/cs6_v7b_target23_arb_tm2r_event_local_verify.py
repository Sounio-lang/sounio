#!/usr/bin/env python3
"""Fail-closed verifier for the event-local diagnostic receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-event-local-diagnostic.v1"
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
CLASSIFICATIONS = {
    "IMPLEMENTATION_INCONSISTENCY",
    "CURRENT_CRITERION_ACCEPTS",
    "MIXED_CHART_AND_EVENT_CRITERION",
    "CHART_DRIFT",
    "EVENT_CRITERION",
    "UNRESOLVED_ENCLOSURE",
}


def fail(message: str) -> None:
    raise SystemExit(f"event-local diagnostic verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(payload: dict[str, object], key: str, expected: object) -> None:
    if payload.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {payload.get(key)!r}")


def expected_classification(payload: dict[str, object]) -> str:
    implementation_ok = payload.get("implementation_checks_passed") is True
    raw = payload.get("final_raw_accepted") is True
    reconditioned = payload.get("final_reconditioned_accepted") is True
    anchored = payload.get("final_anchored_accepted") is True
    if not implementation_ok:
        return "IMPLEMENTATION_INCONSISTENCY"
    if raw:
        return "CURRENT_CRITERION_ACCEPTS"
    if reconditioned and anchored:
        return "MIXED_CHART_AND_EVENT_CRITERION"
    if reconditioned:
        return "CHART_DRIFT"
    if anchored:
        return "EVENT_CRITERION"
    return "UNRESOLVED_ENCLOSURE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="ascii"))

    require(payload, "schema", SCHEMA)
    require(payload, "worker_source_sha256", sha256(args.worker))
    require(payload, "carrier_source_sha256", sha256(args.carrier))
    require(payload, "chain_source_sha256", sha256(args.chain))
    require(payload, "adaptive_source_sha256", sha256(args.adaptive))
    require(payload, "event_source_sha256", sha256(args.event))
    require(payload, "base_source_sha256", sha256(args.base))
    require(payload, "tile_id", "XLEL")
    require(payload, "critical_path", EXPECTED_PATH)
    require(payload, "critical_depth", len(EXPECTED_PATH))
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
    require(payload, "chaos_certified", False)
    require(payload, "open_problem_solved", False)

    prefixes = payload.get("prefix_diagnostics")
    if not isinstance(prefixes, list) or len(prefixes) != len(EXPECTED_PATH) + 1:
        fail("prefix diagnostics do not cover every critical split depth")
    for depth, prefix in enumerate(prefixes):
        if prefix.get("depth") != depth:
            fail(f"prefix {depth} has the wrong depth")
        if prefix.get("path") != EXPECTED_PATH[:depth]:
            fail(f"prefix {depth} has the wrong path")
        for mode in ("raw_symmetric_slab", "reconditioned_symmetric_slab"):
            scan = prefix.get(mode)
            if not isinstance(scan, dict) or not isinstance(scan.get("scales"), list):
                fail(f"prefix {depth} lacks {mode}")
            powers = [item.get("power") for item in scan["scales"]]
            expected_powers = list(range(18, 18 - len(powers), -1))
            if powers != expected_powers or not powers or powers[-1] < 7:
                fail(f"prefix {depth} has a malformed slab scale sequence")
            if scan.get("accepted") is True:
                if scan.get("accepted_power") != powers[-1]:
                    fail(f"prefix {depth} has an inconsistent accepted power")
                if scan["scales"][-1].get("status") != "ACCEPTED":
                    fail(f"prefix {depth} accepted without an accepted scale")
            elif powers != list(range(18, 6, -1)):
                fail(f"prefix {depth} refusal did not exhaust every scale")

    checks = payload.get("implementation_checks")
    if not isinstance(checks, list) or not checks:
        fail("implementation checks are absent")
    names = [item.get("name") for item in checks]
    if len(names) != len(set(names)):
        fail("implementation check names are not unique")
    checks_passed = all(item.get("passed") is True for item in checks)
    require(payload, "implementation_checks_passed", checks_passed)

    classification = payload.get("classification")
    if classification not in CLASSIFICATIONS:
        fail(f"unknown classification {classification!r}")
    expected = expected_classification(payload)
    if classification != expected:
        fail(f"classification mismatch: expected {expected}, got {classification}")

    anchored = payload.get("anchored_crossing_step_newton")
    if not isinstance(anchored, dict):
        fail("anchored Newton diagnostic is absent")
    if anchored.get("accepted") is not payload.get("final_anchored_accepted"):
        fail("anchored acceptance summary mismatch")
    if (
        prefixes[-1]["raw_symmetric_slab"].get("accepted")
        is not payload.get("final_raw_accepted")
    ):
        fail("raw acceptance summary mismatch")
    if (
        prefixes[-1]["reconditioned_symmetric_slab"].get("accepted")
        is not payload.get("final_reconditioned_accepted")
    ):
        fail("reconditioned acceptance summary mismatch")

    print(f"SCHEMA={SCHEMA}")
    print(f"CLASSIFICATION={classification}")
    print(f"IMPLEMENTATION_CHECKS_PASSED={str(checks_passed).lower()}")
    print(f"FINAL_RAW_ACCEPTED={str(payload['final_raw_accepted']).lower()}")
    print(
        "FINAL_RECONDITIONED_ACCEPTED="
        f"{str(payload['final_reconditioned_accepted']).lower()}"
    )
    print(f"FINAL_ANCHORED_ACCEPTED={str(payload['final_anchored_accepted']).lower()}")
    print("FULL_TRANSPORT_ATTEMPTED=false")
    print("COVERING_RELATION_CERTIFIED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
