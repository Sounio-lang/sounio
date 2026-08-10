#!/usr/bin/env python3
"""Negative mutations for the exact witness derivative budget verifier."""

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


def terminal_budget(payload: dict[str, object]) -> dict[str, object]:
    return payload["analyses"]["terminal_before"]["derivative_budget"]


def reverse_range(payload: dict[str, object]) -> None:
    terminal_budget(payload)["range"] = ["1", "-1"]


def corrupt_width(payload: dict[str, object]) -> None:
    terminal_budget(payload)["width_q"] = "1"


def corrupt_term(payload: dict[str, object]) -> None:
    terminal_budget(payload)["terms"][0]["range_contribution"] = ["0", "0"]


def duplicate_term(payload: dict[str, object]) -> None:
    terminal_budget(payload)["terms"].append(
        copy.deepcopy(terminal_budget(payload)["terms"][0])
    )


def corrupt_group(payload: dict[str, object]) -> None:
    terminal_budget(payload)["group_widths_q"]["linear"] = "0"


def corrupt_attribution(payload: dict[str, object]) -> None:
    terminal_budget(payload)["variable_attributed_widths_q"]["rho0"] = "0"


def corrupt_rank(payload: dict[str, object]) -> None:
    ranked = terminal_budget(payload)["ranked_variables"]
    ranked[0], ranked[-1] = ranked[-1], ranked[0]


def erase_remainder_part(payload: dict[str, object]) -> None:
    terminal_budget(payload)["remainder_parts"].pop()


def corrupt_remainder_part(payload: dict[str, object]) -> None:
    terminal_budget(payload)["remainder_parts"][0]["interval"] = ["0", "0"]


def corrupt_remainder_fraction(payload: dict[str, object]) -> None:
    terminal_budget(payload)["remainder_parts"][0]["fraction_of_total_remainder_q"] = "0"


def corrupt_split_domain(payload: dict[str, object]) -> None:
    payload["analyses"]["terminal_before"]["one_level_split_scan"][0]["children"][0]["domain"]["rho3"] = ["-1", "1"]


def corrupt_split_radius(payload: dict[str, object]) -> None:
    payload["analyses"]["terminal_before"]["one_level_split_scan"][0]["worst_child_radius_q"] = "1"


def corrupt_split_factor(payload: dict[str, object]) -> None:
    payload["analyses"]["terminal_before"]["best_split_contraction_factor_q"] = "18"


def corrupt_best_variable(payload: dict[str, object]) -> None:
    payload["analyses"]["terminal_before"]["best_split_variable"] = "xi"


def corrupt_split_positive(payload: dict[str, object]) -> None:
    payload["analyses"]["terminal_before"]["one_split_certifies_transversality"] = True


def flip_check(payload: dict[str, object]) -> None:
    payload["implementation_checks_passed"] = not payload["implementation_checks_passed"]


def duplicate_check(payload: dict[str, object]) -> None:
    payload["implementation_checks"].append(copy.deepcopy(payload["implementation_checks"][0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--witness-receipt", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "witness_hash": set_value("witness_receipt_sha256", "0" * 64),
        "variables": set_value("variables", []),
        "domain": set_value("witness_domain", {}),
        "range": reverse_range,
        "width": corrupt_width,
        "term": corrupt_term,
        "duplicate_term": duplicate_term,
        "group": corrupt_group,
        "attribution": corrupt_attribution,
        "rank": corrupt_rank,
        "remainder_part_missing": erase_remainder_part,
        "remainder_part_interval": corrupt_remainder_part,
        "remainder_part_fraction": corrupt_remainder_fraction,
        "split_domain": corrupt_split_domain,
        "split_radius": corrupt_split_radius,
        "split_factor": corrupt_split_factor,
        "best_variable": corrupt_best_variable,
        "split_positive": corrupt_split_positive,
        "check": flip_check,
        "duplicate_check": duplicate_check,
        "classification": set_value("classification", "mutated"),
        "full_transport": set_value("full_transport_attempted", True),
        "newton": set_value("interval_newton_attempted", True),
        "covering": set_value("covering_relation_certified", True),
        "recurrence": set_value("recurrent_graph_certified", True),
        "chaos": set_value("chaos_certified", True),
        "open_problem": set_value("open_problem_solved", True),
    }
    command = [
        sys.executable, "-B", str(args.verifier), "RECEIPT",
        "--worker", str(args.worker),
        "--witness-receipt", str(args.witness_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-derivative-budget-mutations-") as directory:
        root = Path(directory)
        for name, mutate in mutations.items():
            payload = copy.deepcopy(original)
            mutate(payload)
            candidate = root / f"{name}.json"
            candidate.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            result = subprocess.run(
                [str(candidate) if item == "RECEIPT" else item for item in command],
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
