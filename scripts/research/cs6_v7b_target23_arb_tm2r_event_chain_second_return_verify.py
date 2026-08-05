#!/usr/bin/env python3
"""Verify and aggregate the four isolated CS6 TM2R event-chain receipts."""

from __future__ import annotations

import argparse
import hashlib
import sys
from fractions import Fraction
from pathlib import Path


TILES = ("XLEL", "XLEH", "XHEL", "XHEH")
WORKER_SCHEMA = (
    "sounio.cs6.v7b-target23-arb-tm2r-event-chain-second-return-worker.v7"
)
WORKER_SHA256 = "edde152f2be29f37eefaf1dd859b74f1984714f0b0aceeff53162fa1336d4fb5"
ADAPTIVE_SHA256 = "1a3a4a73897794c525dbae2a82edb38bffefb2b4ba985cb886d7692758101793"
EVENT_SHA256 = "b08dc19d593bbe4056ed3277ab664d1aa4a868949dbaa556d8cbb73803317d7e"


class VerificationError(RuntimeError):
    pass


def parse_receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise VerificationError(f"missing receipt: {path}")
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        if not raw_line or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        if key in result:
            raise VerificationError(
                f"duplicate key {key} in {path.name}:{line_number}"
            )
        result[key] = value
    return result


def require_equal(
    receipt: dict[str, str], key: str, expected: str, tile: str
) -> None:
    actual = receipt.get(key)
    if actual != expected:
        raise VerificationError(
            f"{tile}: {key} expected {expected!r}, got {actual!r}"
        )


def require_positive_fraction(
    receipt: dict[str, str], key: str, tile: str
) -> Fraction:
    raw = receipt.get(key)
    if raw is None:
        raise VerificationError(f"{tile}: missing {key}")
    value = Fraction(raw)
    if value <= 0:
        raise VerificationError(f"{tile}: {key} was not positive: {value}")
    return value


def receipt_fraction(receipt: dict[str, str], key: str, tile: str) -> Fraction:
    raw = receipt.get(key)
    if raw is None:
        raise VerificationError(f"{tile}: missing {key}")
    return Fraction(raw)


def verify_sources(script_dir: Path) -> None:
    expected = {
        "cs6_v7b_target23_arb_tm2r_event_chain_second_return_worker.py": WORKER_SHA256,
        "cs6_v7b_target23_arb_tm2r_subdivided_second_return_worker.py": ADAPTIVE_SHA256,
        "cs6_v7b_target23_arb_tm2r_second_return_worker.py": EVENT_SHA256,
    }
    for name, digest in expected.items():
        path = script_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise VerificationError(
                f"source hash mismatch for {name}: expected {digest}, got {actual}"
            )


def verify_tile(receipt: dict[str, str], tile: str) -> None:
    upward_profiles = {
        "XLEL": ("1", "0", "0", "922", "924", "1", "-9"),
        "XLEH": ("1", "0", "0", "922", "924", "1", "-9"),
        "XHEL": ("2", "1", "8", "924", "926", "3", "-11"),
        "XHEH": ("1", "0", "0", "922", "924", "1", "-9"),
    }
    (
        projected_leaves,
        split_nodes,
        split_reconstructions,
        zero_free_tubes,
        accepted_substeps,
        time_bisections,
        minimum_step_power,
    ) = upward_profiles[tile]
    exact = {
        "SCHEMA": WORKER_SCHEMA,
        "WORKER_SOURCE_SHA256": WORKER_SHA256,
        "ADAPTIVE_DEPENDENCY_SHA256": ADAPTIVE_SHA256,
        "EVENT_PROJECTION_DEPENDENCY_SHA256": EVENT_SHA256,
        "LEAF_ID": "U08-0000000223_S09-0000000325",
        "ARB_PRECISION_BITS": "256",
        "SOURCE_DEGREE": "2",
        "SOURCE_VARIABLES": "2",
        "RESIDUAL_VARIABLES": "4",
        "TIME_TAYLOR_ORDER": "12",
        "RECONDITIONING": "QR_POINT_COEFFICIENT_CARRIER",
        "EVENT_CHAIN": "UPWARD_PROJECT_DOWNWARD_PROJECT_UPWARD_RETURN",
        "INITIAL_SOURCE_TILES": "1",
        "INITIAL_SOURCE_SPLIT_RECONSTRUCTIONS": "24",
        "SOURCE_TILE_FILTER": tile,
        "INITIAL_SOURCE_COVERAGE": "false",
        "FIRST_EVENT_PROJECTED_TILES": "1",
        "FIRST_RETURN_END_STEP_MIN": "617",
        "FIRST_RETURN_END_STEP_MAX": "617",
        "DOWNWARD_EVENT_SPLIT_NODES": "0",
        f"DOWNWARD_EVENT_SPLIT_NODES_{tile}": "0",
        "DOWNWARD_EVENT_SPLIT_RECONSTRUCTIONS": "0",
        "DOWNWARD_PROJECTED_LEAVES": "1",
        "UNRESOLVED_EVENT_LEAVES": "0",
        "DOWNWARD_PROJECTED_W_EXACTLY_ZERO": "true",
        "SECOND_EVENT_PROJECTED_LEAVES_TOTAL": projected_leaves,
        "SECOND_EVENT_SPLIT_NODES_TOTAL": split_nodes,
        "SECOND_EVENT_SPLIT_RECONSTRUCTIONS_TOTAL": split_reconstructions,
        "SECOND_EVENT_PROJECTED_W_EXACTLY_ZERO": "true",
        "DOWNWARD_INITIAL_DEPARTURE_TUBES_MIN": "1",
        "UPWARD_ZERO_FREE_PRIOR_TUBES_MIN": zero_free_tubes,
        "UPWARD_ACCEPTED_SUBSTEPS_MAX": accepted_substeps,
        "UPWARD_TIME_BISECTIONS_MAX": time_bisections,
        "UPWARD_MINIMUM_TIME_STEP_POWER": minimum_step_power,
        "FIRST_UNRESOLVED_BRANCH": "NONE",
        "FIRST_UNRESOLVED_FAILURE_CLASS": "NONE",
        "OUTWARD_STABILIZATION_CHECKS": "39",
        "BOUNDED_METHOD_RESULT": "true",
        "SELECTED_SOURCE_FIRST_EVENT_PROJECTION_CERTIFICATE": "true",
        "SELECTED_SOURCE_CHAIN_CERTIFICATE": "true",
        "FULL_LEAF_FIRST_RETURN_CERTIFICATE": "false",
        "FIRST_INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE": "false",
        "FULL_LEAF_DOWNWARD_EVENT_PROJECTION_CERTIFICATE": "false",
        "FULL_LEAF_SECOND_RETURN_CERTIFICATE": "false",
        "RETURN_MAP_DETERMINANT_CERTIFICATE": "false",
        "COVERING_RELATION_CERTIFICATE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false",
        "V7_B_ELIGIBILITY": "false",
        "CHAOS_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false",
        "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false",
        "CAPD_USED_BY_WORKER": "false",
        "POINT_FALLBACK_USED": "false",
        "FPGA_EXECUTION": "false",
    }
    for key, value in exact.items():
        require_equal(receipt, key, value, tile)

    for key in (
        "DOWNWARD_PURE_SOURCE_MONOMIALS_MIN",
        "DOWNWARD_EVENT_TIME_PURE_SOURCE_MONOMIALS_MIN",
        "SECOND_EVENT_PURE_SOURCE_MONOMIALS_MIN",
    ):
        require_positive_fraction(receipt, key, tile)
    require_positive_fraction(receipt, "SECOND_RETURN_DERIVATIVE_HULL_LOWER_Q", tile)
    require_positive_fraction(receipt, "SECOND_RETURN_NORMAL_HULL_LOWER_Q", tile)

    for prefix in ("DOWNWARD_EVENT_TIME_HULL", "FULL_SECOND_RETURN_TIME_HULL"):
        lower = require_positive_fraction(receipt, f"{prefix}_LOWER_Q", tile)
        upper = require_positive_fraction(receipt, f"{prefix}_UPPER_Q", tile)
        if lower > upper:
            raise VerificationError(f"{tile}: inverted {prefix}")


def aggregate(receipts: dict[str, dict[str, str]]) -> list[str]:
    def bounds(prefix: str) -> tuple[Fraction, Fraction]:
        lowers = [
            receipt_fraction(receipts[tile], f"{prefix}_LOWER_Q", tile)
            for tile in TILES
        ]
        uppers = [
            receipt_fraction(receipts[tile], f"{prefix}_UPPER_Q", tile)
            for tile in TILES
        ]
        return min(lowers), max(uppers)

    downward_lower, downward_upper = bounds("DOWNWARD_EVENT_TIME_HULL")
    second_lower, second_upper = bounds("FULL_SECOND_RETURN_TIME_HULL")
    derivative_lower, derivative_upper = bounds("SECOND_RETURN_DERIVATIVE_HULL")
    normal_lower, normal_upper = bounds("SECOND_RETURN_NORMAL_HULL")
    return [
        "SCHEMA=sounio.cs6.v7b-target23-arb-tm2r-event-chain-second-return-aggregate.v1",
        f"WORKER_SOURCE_SHA256={WORKER_SHA256}",
        f"SOURCE_TILES={','.join(TILES)}",
        "SOURCE_TILE_COUNT=4",
        "SECOND_EVENT_PROJECTED_LEAVES_TOTAL=5",
        "SECOND_EVENT_SPLIT_NODES_TOTAL=1",
        "SECOND_EVENT_SPLIT_RECONSTRUCTIONS_TOTAL=8",
        "INITIAL_SOURCE_SPLIT_RECONSTRUCTIONS=24",
        f"DOWNWARD_EVENT_TIME_HULL_LOWER_Q={downward_lower}",
        f"DOWNWARD_EVENT_TIME_HULL_UPPER_Q={downward_upper}",
        f"FULL_SECOND_RETURN_TIME_HULL_LOWER_Q={second_lower}",
        f"FULL_SECOND_RETURN_TIME_HULL_UPPER_Q={second_upper}",
        f"SECOND_RETURN_DERIVATIVE_HULL_LOWER_Q={derivative_lower}",
        f"SECOND_RETURN_DERIVATIVE_HULL_UPPER_Q={derivative_upper}",
        f"SECOND_RETURN_NORMAL_HULL_LOWER_Q={normal_lower}",
        f"SECOND_RETURN_NORMAL_HULL_UPPER_Q={normal_upper}",
        "FULL_LEAF_SOURCE_COVERAGE_CERTIFICATE=true",
        "FULL_LEAF_FIRST_RETURN_CERTIFICATE=true",
        "FIRST_INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true",
        "FULL_LEAF_DOWNWARD_EVENT_PROJECTION_CERTIFICATE=true",
        "FULL_LEAF_SECOND_RETURN_CERTIFICATE=true",
        "RETURN_MAP_DETERMINANT_CERTIFICATE=false",
        "COVERING_RELATION_CERTIFICATE=false",
        "GLOBAL_HPG_CERTIFICATE=false",
        "V7_B_ELIGIBILITY=false",
        "CHAOS_PROVED=false",
        "CHAOTIC_ATTRACTOR_PROVED=false",
        "OPEN_PROBLEM_SOLVED=false",
        "NOVELTY_OR_PRIORITY_CLAIMED=false",
        "VERIFIED=true",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    default_receipts = (
        Path(__file__).resolve().parent
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_chain_second_return_v1"
    )
    parser.add_argument("--receipts", type=Path, default=default_receipts)
    args = parser.parse_args()
    try:
        script_dir = Path(__file__).resolve().parent
        verify_sources(script_dir)
        receipts = {
            tile: parse_receipt(args.receipts / f"{tile}.stdout.txt")
            for tile in TILES
        }
        observed_tiles = {receipt.get("SOURCE_TILE_FILTER") for receipt in receipts.values()}
        if observed_tiles != set(TILES):
            raise VerificationError(
                f"tile cover mismatch: expected {set(TILES)}, got {observed_tiles}"
            )
        for tile in TILES:
            verify_tile(receipts[tile], tile)
        print("\n".join(aggregate(receipts)))
        return 0
    except (OSError, ValueError, VerificationError) as error:
        print(f"VERIFY_ERROR={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
