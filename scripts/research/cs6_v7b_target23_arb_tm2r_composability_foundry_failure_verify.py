#!/usr/bin/env python3
"""Verify the fail-closed Foundry outcome of the composability experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SCHEMA = "sounio.cs6.v7b-target23-arb-tm2r-composability-foundry-execution.v1"


def fail(message: str) -> None:
    raise SystemExit(f"composability Foundry failure verify error: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        if "=" not in line:
            fail(f"invalid context line {number}")
        key, value = line.split("=", 1)
        if key in result:
            fail(f"duplicate context field {key}")
        result[key] = value
    return result


def require(values: dict[str, str], key: str, expected: str) -> None:
    if values.get(key) != expected:
        fail(f"{key}: expected {expected!r}, got {values.get(key)!r}")


def require_tokens(value: str, tokens: tuple[str, ...], label: str) -> None:
    for token in tokens:
        if token not in value:
            fail(f"{label} is missing {token!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    context = fields(args.receipts / "foundry_execution_context.txt")
    base_log_path = args.receipts / "foundry_XLEL_base.incomplete.stderr.txt"
    retry_log_path = args.receipts / "foundry_XLEL_retry.incomplete.stderr.txt"
    base_log = base_log_path.read_text(encoding="ascii")
    retry_log = retry_log_path.read_text(encoding="ascii")

    require(context, "SCHEMA", SCHEMA)
    require(context, "BASE_XLEL_INCOMPLETE_SHA256", sha256(base_log_path))
    require(context, "RETRY_XLEL_INCOMPLETE_SHA256", sha256(retry_log_path))
    require(context, "BASE_SPLIT_DEPTH_MAX", "8")
    require(context, "BASE_SPLIT_NODES_MAX_PER_TILE", "63")
    require(context, "RETRY_SPLIT_DEPTH_MAX", "12")
    require(context, "RETRY_SPLIT_NODES_MAX_PER_TILE", "255")
    require(context, "RETRY_XLEL_FAILURE_CLASS", "EVENT_SLAB_UNRESOLVED")
    require(
        context,
        "RETRY_XLEL_SLAB_DIAGNOSTIC",
        "2^-18_THROUGH_2^-7_PREDICTOR_ESCAPED",
    )
    require(context, "COMPLETE_TILE_RECEIPT_COUNT", "0")

    require_tokens(
        base_log,
        (
            "downward-event-split tile=XLEL depth=8 variable=RHO1",
            "refusal=EVENT_SLAB_UNRESOLVED",
            "DOWN_RHO1H') refusal=EVENT_SLAB_UNRESOLVED",
        ),
        "base refusal log",
    )
    require_tokens(
        retry_log,
        (
            "downward-event-split tile=XLEL depth=12 variable=ETA",
            "depth=10 variable=RHO0",
            "carriers=1",
            "refusal=EVENT_SLAB_UNRESOLVED",
            "2^-18:PREDICTOR_ESCAPED",
            "2^-7:PREDICTOR_ESCAPED",
        ),
        "retry refusal log",
    )
    for key in (
        "FULL_SUPPORT_CERTIFICATE",
        "HSET_C_DERIVED",
        "EXIT_FACE_INEQUALITIES_CERTIFICATE",
        "ENTRY_BOUNDARY_AVOIDANCE_CERTIFICATE",
        "COVERING_DEGREE_CERTIFICATE",
        "RETURN_MAP_DETERMINANT_CERTIFICATE",
        "LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE",
        "RECURRENT_COVERING_GRAPH_CERTIFICATE",
        "CHAOS_PROVED",
        "OPEN_PROBLEM_SOLVED",
    ):
        require(context, key, "false")

    print(f"SCHEMA={SCHEMA}")
    print("BASE_XLEL_SPLIT_DEPTH_FALSIFIED=8")
    print("RETRY_XLEL_SPLIT_DEPTH_FALSIFIED=12")
    print("RETRY_XLEL_FAILURE_CLASS=EVENT_SLAB_UNRESOLVED")
    print("COMPLETE_TILE_RECEIPT_COUNT=0")
    print("FULL_SUPPORT_CERTIFICATE=false")
    print("HSET_C_DERIVED=false")
    print("LOCAL_HSET_COVERING_RELATION_B_TO_C_CERTIFICATE=false")
    print("CHAOS_PROVED=false")
    print("VERIFIED=true")


if __name__ == "__main__":
    main()
