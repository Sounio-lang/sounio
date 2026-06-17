#!/usr/bin/env python3
"""Shardable persistent campaign runner for colour-guided chi6 beam search.

This is an operational wrapper around `chi6_colour_guided_beam.py`.  It turns a
single beam run into a deterministic parameter-grid campaign with an append-only
JSONL ledger, shard selection, and resume semantics suitable for OrangeFS/Slurm
payload runs.

The campaign is still search-only.  Every row preserves the non-promotable
contract: no SAT, chromatic, global UNSAT, or verified mathematical claim is
made by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BEAM = Path(__file__).with_name("chi6_colour_guided_beam.py")


@dataclass(frozen=True, order=True)
class CellParams:
    generations: int
    beam_width: int
    branch_width: int
    mutation_max_den: int
    mutation_top_points: int
    dsatur_node_limit: int

    def key(self) -> str:
        return (
            f"g{self.generations}_bw{self.beam_width}_br{self.branch_width}"
            f"_d{self.mutation_max_den}_tp{self.mutation_top_points}"
            f"_ds{self.dsatur_node_limit}"
        )

    def as_manifest(self) -> dict[str, int]:
        return {
            "generations": self.generations,
            "beam_width": self.beam_width,
            "branch_width": self.branch_width,
            "mutation_max_den": self.mutation_max_den,
            "mutation_top_points": self.mutation_top_points,
            "dsatur_node_limit": self.dsatur_node_limit,
        }


def campaign_cell_key(cell_index: int, params: CellParams) -> str:
    return f"i{cell_index:06d}_{params.key()}"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)


def parse_positive_int_list(raw: str, name: str) -> list[int]:
    if raw == "":
        raise ValueError(f"{name} cannot be empty")
    values: list[int] = []
    for token in raw.split(","):
        if not token.isdigit() or int(token) <= 0:
            raise ValueError(f"bad {name} token: {token!r}")
        values.append(int(token))
    return values


def parse_nonnegative_int_list(raw: str, name: str) -> list[int]:
    if raw == "":
        raise ValueError(f"{name} cannot be empty")
    values: list[int] = []
    for token in raw.split(","):
        if not token.isdigit():
            raise ValueError(f"bad {name} token: {token!r}")
        values.append(int(token))
    return values


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
        if not all(
            c == "_" or ("0" <= c <= "9") or ("A" <= c <= "Z") or ("a" <= c <= "z")
            for c in key
        ):
            continue
        if key in fields:
            raise ValueError(f"duplicate key in tool output: {key}")
        fields[key] = value
    return fields


def require_no_claims(meta: dict[str, Any], label: str) -> None:
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if key in meta and meta[key] != "none":
            raise ValueError(f"{label} leaked {key}={meta[key]}")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} leaked promotable={meta.get('promotable')}")


def command_with_optional_path(cmd: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_command(cmd: list[str], out_path: Path) -> tuple[int, str, str]:
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
    stderr_path = out_path.with_name(out_path.name + ".stderr")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return proc.returncode, proc.stdout, proc.stderr


def iter_cells(args: argparse.Namespace) -> list[tuple[int, CellParams]]:
    generations = parse_positive_int_list(args.generations_list, "--generations-list")
    beam_widths = parse_positive_int_list(args.beam_width_list, "--beam-width-list")
    branch_widths = parse_positive_int_list(args.branch_width_list, "--branch-width-list")
    mutation_max_dens = parse_positive_int_list(
        args.mutation_max_den_list,
        "--mutation-max-den-list",
    )
    mutation_top_points = parse_nonnegative_int_list(
        args.mutation_top_points_list,
        "--mutation-top-points-list",
    )
    dsatur_node_limits = parse_nonnegative_int_list(
        args.dsatur_node_limit_list,
        "--dsatur-node-limit-list",
    )
    cells: list[tuple[int, CellParams]] = []
    for index, values in enumerate(
        itertools.product(
            generations,
            beam_widths,
            branch_widths,
            mutation_max_dens,
            mutation_top_points,
            dsatur_node_limits,
        )
    ):
        cells.append((index, CellParams(*values)))
    return cells


def selected_cells(args: argparse.Namespace) -> list[tuple[int, CellParams]]:
    cells = [
        (index, params)
        for index, params in iter_cells(args)
        if index % args.shard_count == args.shard_index
    ]
    return cells[: args.cell_budget]


def load_completed_ledger_rows(ledger_path: Path) -> dict[str, dict[str, Any]]:
    if not ledger_path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with ledger_path.open("r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ledger JSONL parse failed at line {lineno}: {exc}") from exc
            if row.get("schema") != "chi6_colour_guided_beam_campaign_cell.v1":
                raise ValueError(f"ledger row {lineno} has wrong schema")
            require_no_claims(row, f"ledger row {lineno}")
            key = str(row.get("campaign_cell_key", ""))
            if row.get("status") == "BEAM_CAMPAIGN_CELL_RECORDED":
                completed[key] = row
    return completed


def append_ledger_row(ledger_path: Path, row: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="ascii") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def failed_cell_row(
    *,
    args: argparse.Namespace,
    cell_index: int,
    params: CellParams,
    cell_dir: Path,
    stdout_path: Path,
    returncode: int,
    stderr: str,
) -> dict[str, Any]:
    return {
        "schema": "chi6_colour_guided_beam_campaign_cell.v1",
        "campaign_cell_index": cell_index,
        "campaign_cell_key": campaign_cell_key(cell_index, params),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cell_dir": str(cell_dir),
        "parameters": params.as_manifest(),
        "mutation_min_neighbor_count": args.mutation_min_neighbor_count,
        "mutation_edge_gain_pool_points": args.mutation_edge_gain_pool_points,
        "mutation_edge_gain_max_combinations": args.mutation_edge_gain_max_combinations,
        "mutation_edge_gain_combination_offset": args.mutation_edge_gain_combination_offset,
        "mutation_edge_gain_combination_stride": args.mutation_edge_gain_combination_stride,
        "mutation_edge_gain_emit_mutations": args.mutation_edge_gain_emit_mutations,
        "beam_stdout": str(stdout_path),
        "beam_stderr": str(stdout_path.with_name(stdout_path.name + ".stderr")),
        "beam_json": "NONE",
        "beam_json_sha256": "NONE",
        "returncode": returncode,
        "failure_summary": (stderr.strip() or "beam_command_failed_without_stderr")[:4000],
        "completed_generation_count": 0,
        "evaluated_child_count": 0,
        "coloured_child_count": 0,
        "refute_attempt_child_count": 0,
        "refute_success_count": 0,
        "refute_failed_count": 0,
        "best_child_candidate_id": "NONE",
        "best_child_score": 0,
        "best_child_dsatur_status": "NONE",
        "best_child_recommended_next_action": "NONE",
        "claim_scope": "colour_guided_beam_campaign_cell_only",
        "sat_claim": "none",
        "chromatic_claim": "none",
        "global_unsat_claim": "none",
        "verified_claim": "none",
        "promotable": 0,
        "status": "BEAM_CAMPAIGN_CELL_FAILED_INFRA",
    }


def run_cell(
    *,
    args: argparse.Namespace,
    cell_index: int,
    params: CellParams,
    out_dir: Path,
) -> dict[str, Any]:
    cell_dir = out_dir / f"cell_{cell_index:06d}_{params.key()}"
    stdout_path = cell_dir / "beam.out"
    cmd = [
        sys.executable,
        str(BEAM),
        str(cell_dir / "beam"),
        "--candidate-prefix",
        f"{args.candidate_prefix}_s{args.shard_index:03d}_c{cell_index:06d}",
        "--generations",
        str(params.generations),
        "--beam-width",
        str(params.beam_width),
        "--branch-width",
        str(params.branch_width),
        "--mutation-max-den",
        str(params.mutation_max_den),
        "--mutation-max-candidates",
        str(args.mutation_max_candidates),
        "--mutation-top-points",
        str(params.mutation_top_points),
        "--mutation-emit-mutations",
        str(args.mutation_emit_mutations),
        "--mutation-add-points",
        str(args.mutation_add_points),
        "--mutation-min-neighbor-count",
        str(args.mutation_min_neighbor_count),
        "--mutation-edge-gain-pool-points",
        str(args.mutation_edge_gain_pool_points),
        "--mutation-edge-gain-max-combinations",
        str(args.mutation_edge_gain_max_combinations),
        "--mutation-edge-gain-combination-offset",
        str(args.mutation_edge_gain_combination_offset),
        "--mutation-edge-gain-combination-stride",
        str(args.mutation_edge_gain_combination_stride),
        "--mutation-edge-gain-emit-mutations",
        str(args.mutation_edge_gain_emit_mutations),
        "--max-vertices",
        str(args.max_vertices),
        "--min-vertices",
        str(args.min_vertices),
        "--min-edges",
        str(args.min_edges),
        "--split-depth",
        str(args.split_depth),
        "--min-split-degree",
        str(args.min_split_degree),
        "--dsatur-node-limit",
        str(params.dsatur_node_limit),
        "--max-cubes",
        str(args.max_cubes),
        "--sample-hard-cubes",
        str(args.sample_hard_cubes),
        "--refute-limit",
        str(args.refute_limit),
        "--refute-timeout-seconds",
        str(args.refute_timeout_seconds),
        "--max-carried-colourings",
        str(args.max_carried_colourings),
    ]
    command_with_optional_path(cmd, "--satfanout-json", args.satfanout_json)
    command_with_optional_path(cmd, "--coords-csv", args.coords_csv)
    command_with_optional_path(cmd, "--colourings-file", args.colourings_file)
    if args.run_refute_ready:
        cmd.append("--run-refute-ready")

    returncode, stdout, stderr = run_command(cmd, stdout_path)
    if returncode != 0:
        return failed_cell_row(
            args=args,
            cell_index=cell_index,
            params=params,
            cell_dir=cell_dir,
            stdout_path=stdout_path,
            returncode=returncode,
            stderr=stderr,
        )

    fields = parse_kv_output(stdout)
    beam_json = Path(fields.get("beam_json", ""))
    if not beam_json.is_file():
        raise RuntimeError(f"beam did not emit beam_json: {beam_json}")
    beam = load_json(beam_json)
    if beam.get("schema") != "chi6_colour_guided_beam.v1":
        raise RuntimeError("beam schema mismatch")
    require_no_claims(beam, "beam manifest")
    row = {
        "schema": "chi6_colour_guided_beam_campaign_cell.v1",
        "campaign_cell_index": cell_index,
        "campaign_cell_key": campaign_cell_key(cell_index, params),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cell_dir": str(cell_dir),
        "parameters": params.as_manifest(),
        "mutation_min_neighbor_count": args.mutation_min_neighbor_count,
        "mutation_edge_gain_pool_points": args.mutation_edge_gain_pool_points,
        "mutation_edge_gain_max_combinations": args.mutation_edge_gain_max_combinations,
        "mutation_edge_gain_combination_offset": args.mutation_edge_gain_combination_offset,
        "mutation_edge_gain_combination_stride": args.mutation_edge_gain_combination_stride,
        "mutation_edge_gain_emit_mutations": args.mutation_edge_gain_emit_mutations,
        "beam_stdout": str(stdout_path),
        "beam_stderr": str(stdout_path.with_name(stdout_path.name + ".stderr")),
        "beam_json": str(beam_json),
        "beam_json_sha256": sha256_file(beam_json),
        "returncode": 0,
        "completed_generation_count": int(beam["completed_generation_count"]),
        "evaluated_child_count": int(beam["evaluated_child_count"]),
        "coloured_child_count": int(beam["coloured_child_count"]),
        "refute_attempt_child_count": int(beam["refute_attempt_child_count"]),
        "refute_success_count": int(beam["refute_success_count"]),
        "refute_failed_count": int(beam["refute_failed_count"]),
        "best_child_candidate_id": beam["best_child_candidate_id"],
        "best_child_score": int(beam["best_child_score"]),
        "best_child_dsatur_status": beam["best_child_dsatur_status"],
        "best_child_recommended_next_action": beam["best_child_recommended_next_action"],
        "claim_scope": "colour_guided_beam_campaign_cell_only",
        "sat_claim": "none",
        "chromatic_claim": "none",
        "global_unsat_claim": "none",
        "verified_claim": "none",
        "promotable": 0,
        "status": "BEAM_CAMPAIGN_CELL_RECORDED",
    }
    require_no_claims(row, "campaign cell")
    return row


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def best_cell(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    recorded = [row for row in rows if row["status"] == "BEAM_CAMPAIGN_CELL_RECORDED"]
    if not recorded:
        return None
    return max(
        recorded,
        key=lambda row: (
            int(row["best_child_score"]),
            int(row["evaluated_child_count"]),
            str(row["campaign_cell_key"]),
        ),
    )


def validate_args(args: argparse.Namespace) -> None:
    if not BEAM.is_file():
        raise ValueError(f"missing beam runner: {BEAM}")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.shard_index < 0:
        raise ValueError("--shard-index must be non-negative")
    if args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be smaller than --shard-count")
    if args.cell_budget <= 0:
        raise ValueError("--cell-budget must be positive")
    if args.mutation_max_candidates <= 0:
        raise ValueError("--mutation-max-candidates must be positive")
    if args.mutation_emit_mutations <= 0:
        raise ValueError("--mutation-emit-mutations must be positive")
    if args.mutation_add_points <= 0:
        raise ValueError("--mutation-add-points must be positive")
    if args.mutation_min_neighbor_count <= 0:
        raise ValueError("--mutation-min-neighbor-count must be positive")
    if args.mutation_edge_gain_pool_points < 0:
        raise ValueError("--mutation-edge-gain-pool-points must be non-negative")
    if args.mutation_edge_gain_max_combinations <= 0:
        raise ValueError("--mutation-edge-gain-max-combinations must be positive")
    if args.mutation_edge_gain_combination_offset < 0:
        raise ValueError("--mutation-edge-gain-combination-offset must be non-negative")
    if args.mutation_edge_gain_combination_stride <= 0:
        raise ValueError("--mutation-edge-gain-combination-stride must be positive")
    if args.mutation_edge_gain_emit_mutations < 0:
        raise ValueError("--mutation-edge-gain-emit-mutations must be non-negative")
    if args.max_vertices < 2:
        raise ValueError("--max-vertices must be at least 2")
    if args.min_vertices < 2:
        raise ValueError("--min-vertices must be at least 2")
    if args.max_vertices < args.min_vertices:
        raise ValueError("--max-vertices cannot be smaller than --min-vertices")
    if args.min_edges < 0:
        raise ValueError("--min-edges must be non-negative")
    if args.split_depth < 0:
        raise ValueError("--split-depth must be non-negative")
    if args.min_split_degree <= 0:
        raise ValueError("--min-split-degree must be positive")
    if args.max_cubes <= 0:
        raise ValueError("--max-cubes must be positive")
    if args.sample_hard_cubes < 0:
        raise ValueError("--sample-hard-cubes must be non-negative")
    if args.sample_hard_cubes > args.max_cubes:
        raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")
    if args.refute_limit < 0:
        raise ValueError("--refute-limit must be non-negative")
    if args.refute_timeout_seconds < 0:
        raise ValueError("--refute-timeout-seconds must be non-negative")
    if args.max_carried_colourings <= 0:
        raise ValueError("--max-carried-colourings must be positive")
    if args.satfanout_json is None and (
        args.coords_csv is None or args.colourings_file is None
    ):
        raise ValueError("provide --satfanout-json or both --coords-csv and --colourings-file")
    for path in (args.satfanout_json, args.coords_csv, args.colourings_file):
        if path is not None and not path.is_file():
            raise ValueError(f"input file does not exist: {path}")
    branch_widths = parse_positive_int_list(args.branch_width_list, "--branch-width-list")
    if max(branch_widths) > args.mutation_emit_mutations:
        raise ValueError("--branch-width-list cannot exceed --mutation-emit-mutations")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--satfanout-json", type=Path)
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--colourings-file", type=Path)
    parser.add_argument("--candidate-prefix", default="cgcampaign")
    parser.add_argument("--generations-list", default="2")
    parser.add_argument("--beam-width-list", default="2")
    parser.add_argument("--branch-width-list", default="2")
    parser.add_argument("--mutation-max-den-list", default="5")
    parser.add_argument("--mutation-top-points-list", default="50")
    parser.add_argument("--dsatur-node-limit-list", default="100000")
    parser.add_argument("--mutation-max-candidates", type=int, default=20_000)
    parser.add_argument("--mutation-emit-mutations", type=int, default=4)
    parser.add_argument("--mutation-add-points", type=int, default=4)
    parser.add_argument("--mutation-min-neighbor-count", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-pool-points", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-max-combinations", type=int, default=250_000)
    parser.add_argument("--mutation-edge-gain-combination-offset", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-combination-stride", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-emit-mutations", type=int, default=1)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    parser.add_argument("--run-refute-ready", action="store_true")
    parser.add_argument("--refute-limit", type=int, default=1)
    parser.add_argument("--refute-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-carried-colourings", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--cell-budget", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        validate_args(args)
        selected = selected_cells(args)
        if not selected:
            raise ValueError("this shard selected zero campaign cells")
        if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
            raise ValueError("out_dir already exists and is non-empty; pass --resume to reuse it")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = args.out_dir / "colour_guided_beam_campaign.ledger.jsonl"
        completed = load_completed_ledger_rows(ledger_path) if args.resume else {}
        rows: list[dict[str, Any]] = []
        skipped_resume_cell_count = 0

        for cell_index, params in selected:
            existing = completed.get(campaign_cell_key(cell_index, params))
            if existing is not None:
                rows.append(existing)
                skipped_resume_cell_count += 1
                continue
            row = run_cell(
                args=args,
                cell_index=cell_index,
                params=params,
                out_dir=args.out_dir,
            )
            append_ledger_row(ledger_path, row)
            rows.append(row)

        best = best_cell(rows)
        manifest = {
            "schema": "chi6_colour_guided_beam_campaign.v1",
            "ledger_jsonl": str(ledger_path),
            "ledger_jsonl_sha256": sha256_file(ledger_path),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "selected_cell_count": len(selected),
            "executed_cell_count": len(rows) - skipped_resume_cell_count,
            "skipped_resume_cell_count": skipped_resume_cell_count,
            "mutation_min_neighbor_count": args.mutation_min_neighbor_count,
            "mutation_edge_gain_pool_points": args.mutation_edge_gain_pool_points,
            "mutation_edge_gain_max_combinations": args.mutation_edge_gain_max_combinations,
            "mutation_edge_gain_combination_offset": (
                args.mutation_edge_gain_combination_offset
            ),
            "mutation_edge_gain_combination_stride": (
                args.mutation_edge_gain_combination_stride
            ),
            "mutation_edge_gain_emit_mutations": args.mutation_edge_gain_emit_mutations,
            "completed_cell_count": sum(
                1 for row in rows if row["status"] == "BEAM_CAMPAIGN_CELL_RECORDED"
            ),
            "failed_cell_count": sum(
                1 for row in rows if row["status"] == "BEAM_CAMPAIGN_CELL_FAILED_INFRA"
            ),
            "status_counts": status_counts(rows),
            "evaluated_child_count": sum(int(row["evaluated_child_count"]) for row in rows),
            "coloured_child_count": sum(int(row["coloured_child_count"]) for row in rows),
            "refute_attempt_child_count": sum(
                int(row["refute_attempt_child_count"]) for row in rows
            ),
            "refute_success_count": sum(int(row["refute_success_count"]) for row in rows),
            "refute_failed_count": sum(int(row["refute_failed_count"]) for row in rows),
            "best_cell_key": best["campaign_cell_key"] if best else "NONE",
            "best_child_candidate_id": best["best_child_candidate_id"] if best else "NONE",
            "best_child_score": int(best["best_child_score"]) if best else 0,
            "best_child_dsatur_status": (
                best["best_child_dsatur_status"] if best else "NONE"
            ),
            "best_child_recommended_next_action": (
                best["best_child_recommended_next_action"] if best else "NONE"
            ),
            "claim_scope": "colour_guided_beam_campaign_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_exact_geometry_plus_checked_sat_lrat_after_campaign",
            "cells": rows,
        }
        require_no_claims(manifest, "campaign manifest")
        manifest_path = args.out_dir / "colour_guided_beam_campaign.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_colour_guided_beam_campaign v1")
    print(f"campaign_json={manifest_path}")
    print(f"campaign_json_sha256={sha256_file(manifest_path)}")
    print(f"ledger_jsonl={ledger_path}")
    print(f"ledger_jsonl_sha256={sha256_file(ledger_path)}")
    print(f"shard_index={manifest['shard_index']}")
    print(f"shard_count={manifest['shard_count']}")
    print(f"selected_cell_count={manifest['selected_cell_count']}")
    print(f"executed_cell_count={manifest['executed_cell_count']}")
    print(f"skipped_resume_cell_count={manifest['skipped_resume_cell_count']}")
    print(f"mutation_min_neighbor_count={manifest['mutation_min_neighbor_count']}")
    print(f"mutation_edge_gain_pool_points={manifest['mutation_edge_gain_pool_points']}")
    print(
        "mutation_edge_gain_max_combinations="
        f"{manifest['mutation_edge_gain_max_combinations']}"
    )
    print(
        "mutation_edge_gain_combination_offset="
        f"{manifest['mutation_edge_gain_combination_offset']}"
    )
    print(
        "mutation_edge_gain_combination_stride="
        f"{manifest['mutation_edge_gain_combination_stride']}"
    )
    print(f"mutation_edge_gain_emit_mutations={manifest['mutation_edge_gain_emit_mutations']}")
    print(f"completed_cell_count={manifest['completed_cell_count']}")
    print(f"failed_cell_count={manifest['failed_cell_count']}")
    print(f"evaluated_child_count={manifest['evaluated_child_count']}")
    print(f"coloured_child_count={manifest['coloured_child_count']}")
    print(f"refute_attempt_child_count={manifest['refute_attempt_child_count']}")
    print(f"refute_success_count={manifest['refute_success_count']}")
    print(f"refute_failed_count={manifest['refute_failed_count']}")
    print(f"best_cell_key={manifest['best_cell_key']}")
    print(f"best_child_candidate_id={manifest['best_child_candidate_id']}")
    print(f"best_child_score={manifest['best_child_score']}")
    print(f"best_child_dsatur_status={manifest['best_child_dsatur_status']}")
    print(f"best_child_recommended_next_action={manifest['best_child_recommended_next_action']}")
    print("claim_scope=colour_guided_beam_campaign_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOUR_GUIDED_BEAM_CAMPAIGN_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
