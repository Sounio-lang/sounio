#!/usr/bin/env python3
"""Negative mutations for the event-local receipt verifier."""

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


def delete_last_path(payload: dict[str, object]) -> None:
    payload["critical_path"] = payload["critical_path"][:-1]


def corrupt_scale_sequence(payload: dict[str, object]) -> None:
    payload["prefix_diagnostics"][-1]["raw_symmetric_slab"]["scales"][0][
        "power"
    ] = 17


def duplicate_check_name(payload: dict[str, object]) -> None:
    checks = payload["implementation_checks"]
    checks[1]["name"] = checks[0]["name"]


def flip_raw_summary(payload: dict[str, object]) -> None:
    payload["final_raw_accepted"] = not payload["final_raw_accepted"]


def flip_implementation_summary(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload[
        "implementation_checks_passed"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "critical_path": delete_last_path,
        "reconditioner": set_value("reconditioner", "mutated"),
        "full_transport": set_value("full_transport_attempted", True),
        "chaos": set_value("chaos_certified", True),
        "raw_summary": flip_raw_summary,
        "implementation_summary": flip_implementation_summary,
        "duplicate_check": duplicate_check_name,
        "slab_sequence": corrupt_scale_sequence,
        "classification": set_value("classification", "mutated"),
    }
    command = [
        sys.executable,
        "-B",
        str(args.verifier),
        "RECEIPT",
        "--worker",
        str(args.worker),
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
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-event-local-mutations-") as directory:
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
