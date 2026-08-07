#!/usr/bin/env python3
"""Negative mutations for the pre-QR residual gate verifier."""

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


def flip_acceptance(payload: dict[str, object]) -> None:
    payload["predictor_centered_prerecond_event_accepted"] = not payload[
        "predictor_centered_prerecond_event_accepted"
    ]


def flip_implementation(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload[
        "implementation_checks_passed"
    ]


def erase_raw_rho3(payload: dict[str, object]) -> None:
    chart = payload["prerecond_event_chart"]
    for scale in chart["scales"]:
        if scale.get("status") == "ACCEPTED":
            scale["raw_projection_variable_weights"][5] = ["0", "0"]
            scale["raw_projection_rho3_positive"] = False
            scale["raw_projection_variables_preserved"] = False
            break


def force_post_qr_symbolic(payload: dict[str, object]) -> None:
    chart = payload["prerecond_event_chart"]
    for scale in chart["scales"]:
        if scale.get("status") == "ACCEPTED":
            scale["symbolic_gate_uses_raw_projection"] = False
            break


def corrupt_policy(payload: dict[str, object]) -> None:
    payload["symbolic_gate_policy"] = "post_qr"
    payload["prerecond_event_chart"]["symbolic_policy"] = "post_qr"


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
    parser.add_argument("--rho3-locus-receipt", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "rho3_locus_hash": set_value("prior_rho3_locus_receipt_sha256", "0" * 64),
        "predictor_center": set_value("predictor_center_q", "0"),
        "implementation_summary": flip_implementation,
        "acceptance_summary": flip_acceptance,
        "raw_rho3": erase_raw_rho3,
        "post_qr_gate": force_post_qr_symbolic,
        "policy": corrupt_policy,
        "classification": set_value("classification", "mutated"),
        "full_transport": set_value("full_transport_attempted", True),
        "chaos": set_value("chaos_certified", True),
        "open_problem": set_value("open_problem_solved", True),
        "covering": set_value("covering_relation_certified", True),
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
        "--rho3-locus-receipt",
        str(args.rho3_locus_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-prerecond-mutations-") as directory:
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
