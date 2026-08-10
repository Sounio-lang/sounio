#!/usr/bin/env python3
"""Negative mutations for the witness-local event verifier."""

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


def set_diag(key: str, value: object) -> Mutation:
    def mutate(payload: dict[str, object]) -> None:
        payload["diagnostic"][key] = value
    return mutate


def erase_weight(payload: dict[str, object]) -> None:
    payload["reconstruction"]["raw_projection_weights"][5] = ["0", "0"]


def corrupt_domain(payload: dict[str, object]) -> None:
    payload["witness_domain"]["bounds"]["eta"] = ["-1", "1"]


def corrupt_last_negative(payload: dict[str, object]) -> None:
    payload["diagnostic"]["last_strict_negative"]["w"] = ["0", "1"]


def corrupt_first_tube(payload: dict[str, object]) -> None:
    payload["diagnostic"]["first_ambiguous"]["tube"]["w"] = ["1", "2"]


def corrupt_result_surface(payload: dict[str, object]) -> None:
    diagnostic = payload["diagnostic"]
    if diagnostic["accepted"]:
        diagnostic["accepted_projection"]["carriers"][0]["event_normal"] = ["-1", "1"]
    else:
        diagnostic["terminal_ambiguous"]["tube"]["strictly_upward"] = not diagnostic["terminal_ambiguous"]["tube"]["strictly_upward"]


def flip_check(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload["implementation_checks_passed"]


def duplicate_check(payload: dict[str, object]) -> None:
    payload["implementation_checks"].append(copy.deepcopy(payload["implementation_checks"][0]))


def remove_lineage_control(payload: dict[str, object]) -> None:
    payload["implementation_checks"] = [
        item
        for item in payload["implementation_checks"]
        if item.get("name") != "lineage_reconditioner_active_for_witness_event"
    ]


def remove_production_control(payload: dict[str, object]) -> None:
    payload["implementation_checks"] = [
        item
        for item in payload["implementation_checks"]
        if item.get("name") != "production_reconditioner_active_before_replay"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--prior-worker", type=Path, required=True)
    parser.add_argument("--prerecond-worker", type=Path, required=True)
    parser.add_argument("--centered-worker", type=Path, required=True)
    parser.add_argument("--composability", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prerecond-receipt", type=Path, required=True)
    parser.add_argument("--transport-receipt", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "prior_hash": set_value("transport_receipt_sha256", "0" * 64),
        "path": set_value("witness_path", []),
        "domain": corrupt_domain,
        "time_depth": set_value("diagnostic_time_refinement_depth", 10),
        "policy": set_value("symbolic_policy", "renumbered"),
        "raw_weight": erase_weight,
        "check": flip_check,
        "duplicate_check": duplicate_check,
        "lineage_control": remove_lineage_control,
        "production_control": remove_production_control,
        "boundary": set_diag("production_boundary_reproduced", False),
        "last_negative": corrupt_last_negative,
        "first_tube": corrupt_first_tube,
        "result_surface": corrupt_result_surface,
        "classification": set_value("classification", "mutated"),
        "full_transport": set_value("full_transport_attempted", True),
        "covering": set_value("covering_relation_certified", True),
        "recurrence": set_value("recurrent_graph_certified", True),
        "chaos": set_value("chaos_certified", True),
        "open_problem": set_value("open_problem_solved", True),
    }
    command = [
        sys.executable, "-B", str(args.verifier), "RECEIPT",
        "--worker", str(args.worker),
        "--prior-worker", str(args.prior_worker),
        "--prerecond-worker", str(args.prerecond_worker),
        "--centered-worker", str(args.centered_worker),
        "--composability", str(args.composability),
        "--transport", str(args.transport),
        "--chain", str(args.chain),
        "--adaptive", str(args.adaptive),
        "--event", str(args.event),
        "--base", str(args.base),
        "--prerecond-receipt", str(args.prerecond_receipt),
        "--transport-receipt", str(args.transport_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-witness-event-mutations-") as directory:
        root = Path(directory)
        for name, mutate in mutations.items():
            payload = copy.deepcopy(original)
            mutate(payload)
            candidate = root / f"{name}.json"
            candidate.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            candidate_command = [str(candidate) if item == "RECEIPT" else item for item in command]
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
