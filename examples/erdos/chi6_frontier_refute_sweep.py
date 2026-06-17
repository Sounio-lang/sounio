#!/usr/bin/env python3
"""Record a bounded rational-frontier campaign through the chi6 refute pipeline.

This composes the local search loop:

  campaign -> cube preflight batch -> refute attempt ledger

The result is still a search ledger, not a chromatic-number certificate. Even a
successful leaf refutation only says that repo-local CNF/DRAT/LRAT artifacts
were emitted for selected cubes. Promotion still requires checked cube cover
LRAT/Lean replay plus exact Euclidean geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CAMPAIGN = Path(__file__).with_name("chi6_rational_frontier_campaign.py")
PREFLIGHT_BATCH = Path(__file__).with_name("chi6_frontier_campaign_preflight_batch.py")
REFUTE_ATTEMPT = Path(__file__).with_name("chi6_frontier_refute_attempt.py")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_kv_output(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key[0].isdigit():
            continue
        if not all(c == "_" or ("0" <= c <= "9") or ("A" <= c <= "Z") or ("a" <= c <= "z") for c in key):
            continue
        if key in fields:
            raise ValueError(f"duplicate key in tool output: {key}")
        fields[key] = value
    return fields


def parse_positive_int_list(raw: str, name: str) -> list[int]:
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    out: list[int] = []
    for token in raw.split(","):
        if not token.isdigit() or int(token) <= 0:
            raise ValueError(f"bad {name} token: {token!r}")
        out.append(int(token))
    return out


def run_command(cmd: list[str], out_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(proc.stdout, encoding="utf-8")
    out_path.with_name(out_path.name + ".stderr").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}: {detail}")
    return proc.stdout


def load_json(path: Path, expected_schema: str, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema") != expected_schema:
        raise ValueError(f"{label} schema must be {expected_schema}")
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if key in meta and meta[key] != "none":
            raise ValueError(f"{label} must carry {key}=none")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} must remain non-promotable")
    return meta


def require_field(fields: dict[str, str], key: str, label: str) -> str:
    if key not in fields or not fields[key]:
        raise RuntimeError(f"{label}: missing field {key}; present={','.join(sorted(fields))}")
    return fields[key]


def recommendation_from_counts(attempt_count: int, refute_success_count: int, status_counts: dict[str, int]) -> str:
    if attempt_count == 0:
        if status_counts.get("CAMPAIGN_NO_VIABLE_SCOUTS", 0):
            return "adjust_split_parameters_or_expand_frontier"
        return "preflight_produced_no_refute_ready_cubes"
    if refute_success_count > 0:
        return "package_leaf_lrat_with_cover_lrat_and_geometry_next"
    if status_counts.get("REFUTE_NORESULT_MUTATE_FRONTIER", 0) or status_counts.get("REFUTE_SAT_MUTATE_FRONTIER", 0):
        return "mutate_or_expand_frontier"
    return "inspect_refute_infra_failure"


def merge_status_counts(total: dict[str, int], row: dict[str, Any]) -> None:
    for key, raw in row.items():
        value = int(raw)
        if value < 0:
            raise ValueError(f"negative status count for {key}: {raw}")
        total[key] = total.get(key, 0) + value


def run_cell(
    *,
    cell_dir: Path,
    cell_index: int,
    candidate_prefix: str,
    max_den_list: str,
    layers_list: str,
    max_points_list: str,
    min_vertices: int,
    min_edges: int,
    split_depth: int,
    min_split_degree: int,
    dsatur_node_limit: int,
    preflight_limit: int,
    max_cubes: int,
    sample_hard_cubes: int,
    refute_limit: int,
    timeout_seconds: int,
    skip_coloring_found: bool,
) -> dict[str, Any]:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_prefix = f"{candidate_prefix}_c{cell_index:03d}"
    campaign_out = cell_dir / "campaign.out"
    parameters = {
        "max_den_list": max_den_list,
        "layers_list": layers_list,
        "max_points_list": max_points_list,
        "min_vertices": min_vertices,
        "min_edges": min_edges,
        "split_depth": split_depth,
        "min_split_degree": min_split_degree,
        "dsatur_node_limit": dsatur_node_limit,
        "preflight_limit": preflight_limit,
        "max_cubes": max_cubes,
        "sample_hard_cubes": sample_hard_cubes,
        "refute_limit": refute_limit,
        "timeout_seconds": timeout_seconds,
        "skip_coloring_found": 1 if skip_coloring_found else 0,
    }
    campaign_cmd = [
        sys.executable,
        str(CAMPAIGN),
        str(cell_dir / "campaign"),
        "--candidate-prefix",
        cell_prefix,
        "--max-den-list",
        max_den_list,
        "--layers-list",
        layers_list,
        "--max-points-list",
        max_points_list,
        "--min-vertices",
        str(min_vertices),
        "--min-edges",
        str(min_edges),
        "--split-depth",
        str(split_depth),
        "--min-split-degree",
        str(min_split_degree),
        "--dsatur-node-limit",
        str(dsatur_node_limit),
    ]
    try:
        campaign_stdout = run_command(campaign_cmd, campaign_out)
    except RuntimeError as exc:
        campaign_stderr = campaign_out.with_name(campaign_out.name + ".stderr")
        stderr_text = campaign_stderr.read_text(encoding="utf-8") if campaign_stderr.is_file() else ""
        stdout_text = campaign_out.read_text(encoding="utf-8") if campaign_out.is_file() else ""
        if "campaign produced no viable scout candidates" not in f"{stdout_text}\n{stderr_text}\n{exc}":
            raise
        return {
            "cell_index": cell_index,
            "cell_dir": str(cell_dir),
            "candidate_prefix": cell_prefix,
            "parameters": parameters,
            "campaign_stdout": str(campaign_out),
            "campaign_stderr": str(campaign_stderr),
            "campaign_json": "NONE",
            "campaign_json_sha256": "NONE",
            "campaign_count": 0,
            "preflight_batch_stdout": "NONE",
            "preflight_batch_stderr": "NONE",
            "preflight_batch_json": "NONE",
            "preflight_batch_sha256": "NONE",
            "preflight_selected_count": 0,
            "preflight_refute_ready_count": 0,
            "preflight_status": "CAMPAIGN_NO_VIABLE_SCOUTS",
            "refute_attempt_stdout": "NONE",
            "refute_attempt_stderr": "NONE",
            "refute_attempt_json": "NONE",
            "refute_attempt_sha256": "NONE",
            "attempt_count": 0,
            "refute_success_count": 0,
            "refute_failed_count": 0,
            "first_success_candidate": "NONE",
            "first_failed_candidate": "NONE",
            "status_counts": {"CAMPAIGN_NO_VIABLE_SCOUTS": 1},
            "recommended_next_action": "adjust_split_parameters_or_expand_frontier",
        }
    campaign_fields = parse_kv_output(campaign_stdout)
    campaign_json = Path(require_field(campaign_fields, "campaign_json", str(campaign_out)))
    campaign = load_json(campaign_json, "chi6_rational_frontier_campaign.v1", "campaign")

    preflight_cmd = [
        sys.executable,
        str(PREFLIGHT_BATCH),
        str(campaign_json),
        str(cell_dir / "preflight-batch"),
        "--limit",
        str(preflight_limit),
        "--max-cubes",
        str(max_cubes),
        "--sample-hard-cubes",
        str(sample_hard_cubes),
    ]
    if skip_coloring_found:
        preflight_cmd.append("--skip-coloring-found")
    preflight_out = cell_dir / "preflight-batch.out"
    preflight_stdout = run_command(preflight_cmd, preflight_out)
    preflight_fields = parse_kv_output(preflight_stdout)
    preflight_batch_json = Path(require_field(preflight_fields, "preflight_batch_json", str(preflight_out)))
    preflight_batch = load_json(
        preflight_batch_json,
        "chi6_frontier_campaign_preflight_batch.v1",
        "preflight batch",
    )
    if int(preflight_batch["refute_ready_count"]) == 0:
        return {
            "cell_index": cell_index,
            "cell_dir": str(cell_dir),
            "candidate_prefix": cell_prefix,
            "parameters": parameters,
            "campaign_stdout": str(campaign_out),
            "campaign_stderr": str(campaign_out.with_name(campaign_out.name + ".stderr")),
            "campaign_json": str(campaign_json),
            "campaign_json_sha256": sha256_file(campaign_json),
            "campaign_count": int(campaign["campaign_count"]),
            "preflight_batch_stdout": str(preflight_out),
            "preflight_batch_stderr": str(preflight_out.with_name(preflight_out.name + ".stderr")),
            "preflight_batch_json": str(preflight_batch_json),
            "preflight_batch_sha256": sha256_file(preflight_batch_json),
            "preflight_selected_count": int(preflight_batch["selected_count"]),
            "preflight_refute_ready_count": 0,
            "preflight_status": "PREFLIGHT_ZERO_REFUTE_READY",
            "refute_attempt_stdout": "NONE",
            "refute_attempt_stderr": "NONE",
            "refute_attempt_json": "NONE",
            "refute_attempt_sha256": "NONE",
            "attempt_count": 0,
            "refute_success_count": 0,
            "refute_failed_count": 0,
            "first_success_candidate": "NONE",
            "first_failed_candidate": "NONE",
            "status_counts": {"PREFLIGHT_ZERO_REFUTE_READY": 1},
            "recommended_next_action": "preflight_produced_no_refute_ready_cubes",
        }

    refute_cmd = [
        sys.executable,
        str(REFUTE_ATTEMPT),
        str(preflight_batch_json),
        str(cell_dir / "refute-attempt"),
        "--limit",
        str(refute_limit),
    ]
    refute_cmd.extend(["--timeout-seconds", str(timeout_seconds)])
    refute_out = cell_dir / "refute-attempt.out"
    refute_stdout = run_command(refute_cmd, refute_out)
    refute_fields = parse_kv_output(refute_stdout)
    refute_attempt_json = Path(require_field(refute_fields, "refute_attempt_json", str(refute_out)))
    refute_attempt = load_json(
        refute_attempt_json,
        "chi6_frontier_refute_attempt.v1",
        "refute attempt",
    )

    status_counts = {str(k): int(v) for k, v in refute_attempt["status_counts"].items()}
    return {
        "cell_index": cell_index,
        "cell_dir": str(cell_dir),
        "candidate_prefix": cell_prefix,
        "parameters": parameters,
        "campaign_stdout": str(campaign_out),
        "campaign_stderr": str(campaign_out.with_name(campaign_out.name + ".stderr")),
        "campaign_json": str(campaign_json),
        "campaign_json_sha256": sha256_file(campaign_json),
        "campaign_count": int(campaign["campaign_count"]),
        "preflight_batch_stdout": str(preflight_out),
        "preflight_batch_stderr": str(preflight_out.with_name(preflight_out.name + ".stderr")),
        "preflight_batch_json": str(preflight_batch_json),
        "preflight_batch_sha256": sha256_file(preflight_batch_json),
        "preflight_selected_count": int(preflight_batch["selected_count"]),
        "preflight_refute_ready_count": int(preflight_batch["refute_ready_count"]),
        "preflight_status": "PREFLIGHT_REFUTE_READY",
        "refute_attempt_stdout": str(refute_out),
        "refute_attempt_stderr": str(refute_out.with_name(refute_out.name + ".stderr")),
        "refute_attempt_json": str(refute_attempt_json),
        "refute_attempt_sha256": sha256_file(refute_attempt_json),
        "attempt_count": int(refute_attempt["attempt_count"]),
        "refute_success_count": int(refute_attempt["refute_success_count"]),
        "refute_failed_count": int(refute_attempt["refute_failed_count"]),
        "first_success_candidate": refute_attempt["first_success_candidate"],
        "first_failed_candidate": refute_attempt["first_failed_candidate"],
        "status_counts": status_counts,
        "recommended_next_action": recommendation_from_counts(
            int(refute_attempt["attempt_count"]),
            int(refute_attempt["refute_success_count"]),
            status_counts,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--candidate-prefix", default="sweep")
    parser.add_argument("--max-den-list", default="1,5")
    parser.add_argument("--layers-list", default="1")
    parser.add_argument("--max-points-list", default="16")
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--split-depth-list")
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--min-split-degree-list")
    parser.add_argument("--dsatur-node-limit", type=int, default=1)
    parser.add_argument("--dsatur-node-limit-list")
    parser.add_argument("--preflight-limit", type=int, default=1)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    parser.add_argument("--refute-limit", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--skip-coloring-found", action="store_true")
    parser.add_argument("--stop-after-first-success", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        for script in (CAMPAIGN, PREFLIGHT_BATCH, REFUTE_ATTEMPT):
            if not script.is_file():
                raise ValueError(f"missing script: {script}")
        if args.min_vertices < 2:
            raise ValueError("--min-vertices must be at least 2")
        if args.preflight_limit <= 0:
            raise ValueError("--preflight-limit must be positive")
        if args.max_cubes <= 0:
            raise ValueError("--max-cubes must be positive")
        if args.sample_hard_cubes < 0:
            raise ValueError("--sample-hard-cubes must be non-negative")
        if args.sample_hard_cubes > args.max_cubes:
            raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")
        if args.refute_limit < 0:
            raise ValueError("--refute-limit must be non-negative")
        if args.timeout_seconds < 0:
            raise ValueError("--timeout-seconds must be non-negative")
        split_depths = (
            parse_positive_int_list(args.split_depth_list, "--split-depth-list")
            if args.split_depth_list
            else [args.split_depth]
        )
        min_split_degrees = (
            parse_positive_int_list(args.min_split_degree_list, "--min-split-degree-list")
            if args.min_split_degree_list
            else [args.min_split_degree]
        )
        dsatur_node_limits = (
            parse_positive_int_list(args.dsatur_node_limit_list, "--dsatur-node-limit-list")
            if args.dsatur_node_limit_list
            else [args.dsatur_node_limit]
        )
        if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
            raise ValueError("out_dir already exists and is non-empty; pass --resume to reuse it")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        cells: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}
        cell_index = 0
        for split_depth, min_split_degree, dsatur_node_limit in itertools.product(
            split_depths,
            min_split_degrees,
            dsatur_node_limits,
        ):
            cell = run_cell(
                cell_dir=args.out_dir / f"cell_{cell_index:03d}",
                cell_index=cell_index,
                candidate_prefix=args.candidate_prefix,
                max_den_list=args.max_den_list,
                layers_list=args.layers_list,
                max_points_list=args.max_points_list,
                min_vertices=args.min_vertices,
                min_edges=args.min_edges,
                split_depth=split_depth,
                min_split_degree=min_split_degree,
                dsatur_node_limit=dsatur_node_limit,
                preflight_limit=args.preflight_limit,
                max_cubes=args.max_cubes,
                sample_hard_cubes=args.sample_hard_cubes,
                refute_limit=args.refute_limit,
                timeout_seconds=args.timeout_seconds,
                skip_coloring_found=args.skip_coloring_found,
            )
            cells.append(cell)
            merge_status_counts(status_counts, cell["status_counts"])
            cell_index += 1
            if args.stop_after_first_success and int(cell["refute_success_count"]) > 0:
                break

        attempt_count = sum(int(cell["attempt_count"]) for cell in cells)
        refute_success_count = sum(int(cell["refute_success_count"]) for cell in cells)
        refute_failed_count = sum(int(cell["refute_failed_count"]) for cell in cells)
        first_success_candidate = next(
            (cell["first_success_candidate"] for cell in cells if cell["first_success_candidate"] != "NONE"),
            "NONE",
        )
        first_failed_candidate = next(
            (cell["first_failed_candidate"] for cell in cells if cell["first_failed_candidate"] != "NONE"),
            "NONE",
        )
        next_action = recommendation_from_counts(attempt_count, refute_success_count, status_counts)
        manifest = {
            "schema": "chi6_frontier_refute_sweep.v1",
            "cell_count": len(cells),
            "campaign_manifest_count": sum(1 for cell in cells if cell["campaign_json"] != "NONE"),
            "preflight_batch_count": sum(1 for cell in cells if cell["preflight_batch_json"] != "NONE"),
            "attempt_manifest_count": sum(1 for cell in cells if cell["refute_attempt_json"] != "NONE"),
            "campaign_total_count": sum(int(cell["campaign_count"]) for cell in cells),
            "preflight_selected_count": sum(int(cell["preflight_selected_count"]) for cell in cells),
            "preflight_refute_ready_count": sum(int(cell["preflight_refute_ready_count"]) for cell in cells),
            "attempt_count": attempt_count,
            "refute_success_count": refute_success_count,
            "refute_failed_count": refute_failed_count,
            "first_success_candidate": first_success_candidate,
            "first_failed_candidate": first_failed_candidate,
            "status_counts": status_counts,
            "recommended_next_action": next_action,
            "stop_after_first_success": 1 if args.stop_after_first_success else 0,
            "next_parameter_hint": {
                "if_noresult": "increase --max-points-list or --layers-list, then rerun this sweep",
                "if_success": "package leaf LRATs with a checked cube-cover certificate before any SAT claim",
            },
            "claim_scope": "frontier_refute_sweep_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_checked_cube_cover_lrat_lean_exact_geometry_real_bridge",
            "cells": cells,
        }
        manifest_path = args.out_dir / "frontier_refute_sweep.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_refute_sweep v1")
    print(f"sweep_json={manifest_path}")
    print(f"sweep_json_sha256={sha256_file(manifest_path)}")
    print(f"cell_count={manifest['cell_count']}")
    print(f"campaign_manifest_count={manifest['campaign_manifest_count']}")
    print(f"preflight_batch_count={manifest['preflight_batch_count']}")
    print(f"attempt_manifest_count={manifest['attempt_manifest_count']}")
    print(f"campaign_total_count={manifest['campaign_total_count']}")
    print(f"preflight_selected_count={manifest['preflight_selected_count']}")
    print(f"preflight_refute_ready_count={manifest['preflight_refute_ready_count']}")
    print(f"attempt_count={manifest['attempt_count']}")
    print(f"refute_success_count={manifest['refute_success_count']}")
    print(f"refute_failed_count={manifest['refute_failed_count']}")
    print(f"first_success_candidate={manifest['first_success_candidate']}")
    print(f"first_failed_candidate={manifest['first_failed_candidate']}")
    print(f"recommended_next_action={manifest['recommended_next_action']}")
    print("claim_scope=frontier_refute_sweep_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=FRONTIER_REFUTE_SWEEP_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
