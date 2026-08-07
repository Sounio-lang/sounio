#!/usr/bin/env python3
"""Negative mutations for the residual rho3 locus verifier."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path


Mutation = Callable[[dict[str, object]], None]


def set_value(key: str, value: object) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        payload[key] = value

    return mutate


def flip_implementation(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload[
        "implementation_checks_passed"
    ]


def corrupt_locus(payload: dict[str, object]) -> None:
    payload["classification"] = "mutated"
    payload["rho3_locus_chart"]["status"] = "mutated"


def flip_raw_rho3(payload: dict[str, object]) -> None:
    for scale in payload["rho3_locus_chart"]["scales"]:
        if "raw_projection_rho3_positive" in scale:
            scale["raw_projection_rho3_positive"] = not scale[
                "raw_projection_rho3_positive"
            ]
            break


def corrupt_rank(payload: dict[str, object]) -> None:
    for scale in payload["rho3_locus_chart"]["scales"]:
        if scale.get("status") == "RECONDITION_COLLAPSES_RESIDUAL_RANK":
            scale["residual_pure_direction_rank"] = 4
            break


def delete_projection_scale(payload: dict[str, object]) -> None:
    chart = payload["rho3_locus_chart"]
    chart["scales"] = [
        scale
        for scale in chart["scales"]
        if "raw_projection_rho3_positive" not in scale
    ]
    chart["projection_scales"] = 0
    chart["locus_counts"] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
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
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "prior_centered_hash": set_value(
            "prior_event_centered_receipt_sha256", "0" * 64
        ),
        "predictor_center": set_value("predictor_center_q", "0"),
        "implementation_summary": flip_implementation,
        "classification": corrupt_locus,
        "raw_rho3_flag": flip_raw_rho3,
        "residual_rank": corrupt_rank,
        "projection_scales": delete_projection_scale,
        "full_transport": set_value("full_transport_attempted", True),
        "chaos": set_value("chaos_certified", True),
        "open_problem": set_value("open_problem_solved", True),
    }
    command = [
        sys.executable,
        "-B",
        str(args.verifier),
        "RECEIPT",
        "--worker",
        str(args.worker),
        "--centered-worker",
        str(args.centered_worker),
        "--prior-worker",
        str(args.prior_worker),
        "--carrier",
        str(args.carrier),
        "--chain",
        str(args.chain),
        "--adaptive",
        str(args.adaptive),
        "--event",
        str(args.event),
        "--base",
        str(args.base),
        "--event-local-receipt",
        str(args.event_local_receipt),
        "--event-centered-receipt",
        str(args.event_centered_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-rho3-locus-mutations-") as directory:
        root = Path(directory)
        for name, mutate in mutations.items():
            payload = copy.deepcopy(original)
            mutate(payload)
            candidate = root / f"{name}.json"
            candidate.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            candidate_command = [
                str(candidate) if item == "RECEIPT" else item for item in command
            ]
            result = subprocess.run(
                candidate_command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.returncode == 0:
                raise SystemExit(f"mutation unexpectedly accepted: {name}")
            print(f"MUTATION_REJECTED={name}")
            passed += 1
    print(f"MUTATIONS_REJECTED={passed}/{len(mutations)}")


if __name__ == "__main__":
    main()
