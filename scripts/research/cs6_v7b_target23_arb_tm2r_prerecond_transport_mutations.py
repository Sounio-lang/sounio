#!/usr/bin/env python3
"""Negative mutations for the pre-QR lineage transport verifier."""

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


def set_transport(key: str, value: object) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        payload["transport"][key] = value

    return mutate


def flip_implementation(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload[
        "implementation_checks_passed"
    ]


def duplicate_check(payload: dict[str, object]) -> None:
    payload["implementation_checks"].append(
        copy.deepcopy(payload["implementation_checks"][0])
    )


def erase_raw_rho3(payload: dict[str, object]) -> None:
    payload["raw_projection_variable_weights"][5] = ["0", "0"]


def drift_raw_section(payload: dict[str, object]) -> None:
    payload["raw_projection_components"][2]["remainder"] = ["0", "1/1048576"]


def corrupt_domain(payload: dict[str, object]) -> None:
    del payload["critical_domain"]["bounds"]["rho3"]


def corrupt_terminal(payload: dict[str, object]) -> None:
    transport = payload["transport"]
    if transport["carriers"]:
        transport["carriers"][0]["event_normal"] = ["-1", "1"]
    elif transport["unresolved"]:
        transport["unresolved"] = []
    else:
        transport["status"] = "COMPLETE"


def expose_partial_hull(payload: dict[str, object]) -> None:
    transport = payload["transport"]
    if transport["status"] == "TRANSPORT_REFUSED":
        transport["event_time"] = ["0", "1"]
    elif transport["carriers"]:
        transport["event_derivative"] = ["-1", "1"]
    else:
        transport["event_time"] = ["0", "1"]


def corrupt_aggregate_hull(payload: dict[str, object]) -> None:
    transport = payload["transport"]
    if transport["carriers"] and not transport["unresolved"]:
        transport["event_time"] = ["0", "0"]
    else:
        transport["event_normal"] = ["1", "2"]


def flip_complete(payload: dict[str, object]) -> None:
    payload["next_return_complete"] = not payload["next_return_complete"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--prerecond-worker", type=Path, required=True)
    parser.add_argument("--centered-worker", type=Path, required=True)
    parser.add_argument("--composability", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prerecond-receipt", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "prior_hash": set_value("prior_prerecond_receipt_sha256", "0" * 64),
        "critical_path": set_value("critical_path", []),
        "policy": set_value("symbolic_transport_policy", "post_qr_renumbered"),
        "raw_rho3": erase_raw_rho3,
        "raw_section": drift_raw_section,
        "domain": corrupt_domain,
        "implementation": flip_implementation,
        "duplicate_check": duplicate_check,
        "reconditioner": set_transport("reconditioner", "point_qr"),
        "split_limit": set_transport("split_node_limit", 256),
        "stop_policy": set_transport("stop_after_first_unresolved", False),
        "terminal": corrupt_terminal,
        "partial_hull": expose_partial_hull,
        "aggregate_hull": corrupt_aggregate_hull,
        "complete": flip_complete,
        "classification": set_value("classification", "mutated"),
        "full_transport": set_value("full_transport_attempted", False),
        "covering": set_value("covering_relation_certified", True),
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
        "--prerecond-worker",
        str(args.prerecond_worker),
        "--centered-worker",
        str(args.centered_worker),
        "--composability",
        str(args.composability),
        "--transport",
        str(args.transport),
        "--chain",
        str(args.chain),
        "--adaptive",
        str(args.adaptive),
        "--event",
        str(args.event),
        "--base",
        str(args.base),
        "--prerecond-receipt",
        str(args.prerecond_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-prerecond-transport-mutations-") as directory:
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
