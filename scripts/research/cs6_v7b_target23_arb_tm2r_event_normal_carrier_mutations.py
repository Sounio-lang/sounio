#!/usr/bin/env python3
"""Negative mutations for the event-normal carrier verifier."""

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


def first_mode(payload: dict[str, object]) -> dict[str, object]:
    return payload["result"]["modes"][0]


def initial(payload: dict[str, object]) -> dict[str, object]:
    return first_mode(payload)["initial_witness_analysis"]


def first_basis(payload: dict[str, object]) -> dict[str, object]:
    return initial(payload)["stats"]["basis_history"][0]


def corrupt_basis(payload: dict[str, object]) -> None:
    first_basis(payload)["basis"][0][0] = "0"


def corrupt_inverse(payload: dict[str, object]) -> None:
    first_basis(payload)["inverse"][0][0] = "0"


def corrupt_kernel_pairing(payload: dict[str, object]) -> None:
    first_basis(payload)["kernel_pairings_q"][0] = "1"


def corrupt_covector(payload: dict[str, object]) -> None:
    first_basis(payload)["event_covector"][2] = "0"


def corrupt_radius(payload: dict[str, object]) -> None:
    first_basis(payload)["coordinate_radii"][0] = ["-1", "1"]


def corrupt_reconstruction_count(payload: dict[str, object]) -> None:
    initial(payload)["stats"]["reconstruction_checks"] = 0


def corrupt_kernel_count(payload: dict[str, object]) -> None:
    initial(payload)["stats"]["kernel_orthogonality_checks"] = 0


def corrupt_normal_form_count(payload: dict[str, object]) -> None:
    initial(payload)["stats"]["normal_form_checks"] = 0


def corrupt_history(payload: dict[str, object]) -> None:
    initial(payload)["stats"]["basis_history"].pop()


def corrupt_component_variables(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_components"][0]["coefficients"][0]["monomial"] = [0] * 6


def corrupt_component_degree(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_components"][0]["coefficients"][0]["monomial"] = [3] + [0] * 9


def corrupt_component_normal_form(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_components"][0]["coefficients"][0]["monomial"] = [1] + [0] * 5 + [1] + [0] * 3


def corrupt_component_remainder(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_components"][0]["remainder"] = ["0", "1"]


def corrupt_budget_range(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_derivative_budget"]["range"] = ["1", "-1"]


def corrupt_budget_width(payload: dict[str, object]) -> None:
    initial(payload)["carrier_one_step_derivative_budget"]["width_q"] = "1"


def corrupt_improvement(payload: dict[str, object]) -> None:
    initial(payload)["one_step_derivative_width_improvement_factor_q"] = "18"


def corrupt_margin(payload: dict[str, object]) -> None:
    initial(payload)["one_step_derivative_width_margin_q"] = "1"


def corrupt_improvement_flag(payload: dict[str, object]) -> None:
    initial(payload)["one_step_improves_lineage"] = False


def corrupt_certificate(payload: dict[str, object]) -> None:
    initial(payload)["generator_reconstruction_certificate"] = False


def corrupt_mode_classification(payload: dict[str, object]) -> None:
    first_mode(payload)["classification"] = "mutated"


def corrupt_candidate(payload: dict[str, object]) -> None:
    payload["result"]["transport_candidates"] = []


def corrupt_posthoc_boundary(payload: dict[str, object]) -> None:
    first_mode(payload)["post_hoc_endpoint_recovery_is_control_only"] = False


def duplicate_check(payload: dict[str, object]) -> None:
    payload["implementation_checks"].append(copy.deepcopy(payload["implementation_checks"][0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--witness-worker", type=Path, required=True)
    parser.add_argument("--budget-worker", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--witness-receipt", type=Path, required=True)
    parser.add_argument("--budget-receipt", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.receipt.read_text(encoding="ascii"))
    mutations: dict[str, Mutation] = {
        "schema": set_value("schema", "mutated"),
        "worker_hash": set_value("worker_source_sha256", "0" * 64),
        "witness_hash": set_value("witness_receipt_sha256", "0" * 64),
        "budget_hash": set_value("budget_receipt_sha256", "0" * 64),
        "primary_variables": set_value("primary_variables", []),
        "carrier_variables": set_value("carrier_variables", []),
        "basis": corrupt_basis,
        "inverse": corrupt_inverse,
        "kernel_pairing": corrupt_kernel_pairing,
        "covector": corrupt_covector,
        "radius": corrupt_radius,
        "reconstruction_count": corrupt_reconstruction_count,
        "kernel_count": corrupt_kernel_count,
        "normal_form_count": corrupt_normal_form_count,
        "history": corrupt_history,
        "component_variables": corrupt_component_variables,
        "component_degree": corrupt_component_degree,
        "component_normal_form": corrupt_component_normal_form,
        "component_remainder": corrupt_component_remainder,
        "budget_range": corrupt_budget_range,
        "budget_width": corrupt_budget_width,
        "improvement": corrupt_improvement,
        "margin": corrupt_margin,
        "improvement_flag": corrupt_improvement_flag,
        "certificate": corrupt_certificate,
        "mode_classification": corrupt_mode_classification,
        "candidate": corrupt_candidate,
        "posthoc_boundary": corrupt_posthoc_boundary,
        "duplicate_check": duplicate_check,
        "top_classification": lambda p: p["result"].update(classification="mutated"),
        "newton": set_value("interval_newton_attempted", True),
        "covering": set_value("covering_relation_certified", True),
        "recurrence": set_value("recurrent_graph_certified", True),
        "chaos": set_value("chaos_certified", True),
        "open_problem": set_value("open_problem_solved", True),
    }
    command = [
        sys.executable, "-B", str(args.verifier), "RECEIPT",
        "--worker", str(args.worker),
        "--witness-worker", str(args.witness_worker),
        "--budget-worker", str(args.budget_worker),
        "--base", str(args.base),
        "--adaptive", str(args.adaptive),
        "--witness-receipt", str(args.witness_receipt),
        "--budget-receipt", str(args.budget_receipt),
    ]
    passed = 0
    with tempfile.TemporaryDirectory(prefix="cs6-event-normal-mutations-") as directory:
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
