#!/usr/bin/env python3
"""Adaptive recursive frontier sweep for the chi6 search lane.

This driver turns the flat `mutate_or_expand_frontier` recommendation from a
bounded sweep into a concrete next search generation.  The metaphor is
"fractal" recursion plus an LC-style attention gain: hard, geometry-ready,
NORESULT branches receive more expansion pressure.  It is only a scheduling
heuristic.  It emits no SAT/chromatic/global UNSAT claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from chi6_frontier_refute_sweep import parse_positive_int_list, run_cell


COLOUR_MUTATOR = Path(__file__).with_name("chi6_colour_guided_mutation.py")
ATTENTION_POLICY = (
    "recursive_fractal_frontier_locus_coeruleus_gain_heuristic_not_chromatic_evidence"
)
EXPAND_ACTIONS = {
    "mutate_or_expand_frontier",
    "preflight_produced_no_refute_ready_cubes",
    "inspect_refute_infra_failure",
}
DSATUR_ATTENTION = {
    "NO_K_COLORING_FOUND_BY_CPU_PROBE_NONCERTIFYING": 80_000,
    "UNKNOWN_NODE_LIMIT": 40_000,
    "K_COLORING_FOUND": 0,
}


@dataclasses.dataclass(frozen=True, order=True)
class CellParams:
    max_den: int
    layers: int
    max_points: int
    min_vertices: int
    min_edges: int
    split_depth: int
    min_split_degree: int
    dsatur_node_limit: int
    max_cubes: int
    sample_hard_cubes: int
    refute_limit: int
    timeout_seconds: int

    def key(self) -> str:
        return (
            f"d{self.max_den}_l{self.layers}_p{self.max_points}"
            f"_v{self.min_vertices}_e{self.min_edges}"
            f"_s{self.split_depth}_msd{self.min_split_degree}"
            f"_ds{self.dsatur_node_limit}_c{self.max_cubes}"
            f"_h{self.sample_hard_cubes}_r{self.refute_limit}_t{self.timeout_seconds}"
        )

    def as_manifest(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="ascii") as f:
        return json.load(f)


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


def parse_cell_campaign(
    cell: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    campaign_json = cell.get("campaign_json")
    if not campaign_json or campaign_json == "NONE":
        return None, None
    campaign = load_json(campaign_json)
    ranking = campaign.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return campaign, None
    top = ranking[0]
    if not isinstance(top, dict):
        return campaign, None
    return campaign, top


def as_int(raw: Any, default: int = 0) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value


def attention_score(cell: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    _campaign, top = parse_cell_campaign(cell)
    top = top or {}
    status = str(top.get("dsatur_status", ""))
    n = as_int(top.get("n"))
    m = as_int(top.get("m"))
    max_degree = as_int(top.get("max_degree"))
    dsatur_nodes = min(as_int(top.get("dsatur_nodes")), 100_000)
    ready = as_int(cell.get("preflight_refute_ready_count"))
    attempts = as_int(cell.get("attempt_count"))
    successes = as_int(cell.get("refute_success_count"))
    noresult = as_int(cell.get("status_counts", {}).get("REFUTE_NORESULT_MUTATE_FRONTIER", 0))
    sat_mutate = as_int(cell.get("status_counts", {}).get("REFUTE_SAT_MUTATE_FRONTIER", 0))
    geometry_bonus = 25_000 if top.get("geometry_status") == "PASS" else 0
    ready_bonus = 30_000 * ready
    noresult_bonus = 20_000 * noresult
    success_bonus = 2_000_000 * successes
    score = (
        success_bonus
        + ready_bonus
        + noresult_bonus
        + geometry_bonus
        + DSATUR_ATTENTION.get(status, 0)
        + m * 250
        + n * 60
        + max_degree * 120
        + dsatur_nodes
        + attempts * 500
        + sat_mutate * 5_000
    )
    signal = {
        "candidate_id": top.get("candidate_id", "NONE"),
        "n": n,
        "m": m,
        "max_degree": max_degree,
        "dsatur_status": status or "NONE",
        "dsatur_nodes_capped": dsatur_nodes,
        "geometry_status": top.get("geometry_status", "NONE"),
        "preflight_refute_ready_count": ready,
        "attempt_count": attempts,
        "refute_success_count": successes,
        "noresult_count": noresult,
        "attention_score": score,
    }
    return score, signal


def clamp(value: int, cap: int) -> int:
    return min(value, cap)


def append_child(
    out: list[tuple[str, CellParams, int]],
    seen: set[CellParams],
    branch: str,
    params: CellParams,
    hint: int,
) -> None:
    if params in seen:
        return
    seen.add(params)
    out.append((branch, params, hint))


def expand_params(
    *,
    parent: CellParams,
    parent_score: int,
    max_den_cap: int,
    layers_cap: int,
    max_points_cap: int,
    split_depth_cap: int,
    max_cubes_cap: int,
    sample_hard_cubes_cap: int,
    point_growth: int,
    layer_growth: int,
    den_growth: int,
    sample_growth: int,
    children_per_cell: int,
) -> list[tuple[str, CellParams, int]]:
    if children_per_cell <= 0:
        return []
    candidates: list[tuple[str, CellParams, int]] = []
    seen: set[CellParams] = {parent}

    split_child = dataclasses.replace(
        parent,
        split_depth=clamp(parent.split_depth + 1, split_depth_cap),
        max_cubes=clamp(
            max(parent.max_cubes, 5 ** min(parent.split_depth + 1, split_depth_cap)),
            max_cubes_cap,
        ),
        sample_hard_cubes=clamp(parent.sample_hard_cubes + sample_growth, sample_hard_cubes_cap),
    )
    append_child(candidates, seen, "recursive_split_gain", split_child, parent_score + 7_000)

    layer_child = dataclasses.replace(
        parent,
        layers=clamp(parent.layers + layer_growth, layers_cap),
        max_points=clamp(parent.max_points + point_growth, max_points_cap),
    )
    append_child(candidates, seen, "fractal_layer_growth", layer_child, parent_score + 5_000)

    point_child = dataclasses.replace(
        parent,
        max_points=clamp(parent.max_points + point_growth, max_points_cap),
    )
    append_child(candidates, seen, "frontier_point_growth", point_child, parent_score + 3_000)

    den_child = dataclasses.replace(
        parent,
        max_den=clamp(parent.max_den + den_growth, max_den_cap),
        max_points=clamp(parent.max_points + point_growth, max_points_cap),
    )
    append_child(candidates, seen, "denominator_locus_probe", den_child, parent_score + 2_000)

    return candidates[:children_per_cell]


def skipped_colour_guided_mutation(reason: str) -> dict[str, Any]:
    return {
        "status": "COLOUR_GUIDED_MUTATION_SKIPPED",
        "reason": reason,
        "stdout": "NONE",
        "stderr": "NONE",
        "mutation_json": "NONE",
        "mutation_json_sha256": "NONE",
        "candidate_point_count_scored": 0,
        "single_point_full_blocker_count": 0,
        "mutation_count": 0,
        "first_mutation_coords_csv": "NONE",
        "first_mutation_new_n": 0,
        "first_mutation_new_m": 0,
        "first_mutation_killed_colouring_count_by_unit_neighborhood": 0,
    }


def run_colour_guided_mutation(
    *,
    cell_dir: Path,
    cell_index: int,
    refute_attempt_json: str,
    candidate_prefix: str,
    max_den: int,
    max_candidates: int,
    top_points: int,
    emit_mutations: int,
    add_points: int,
) -> dict[str, Any]:
    if not COLOUR_MUTATOR.is_file():
        raise ValueError(f"missing colour-guided mutator: {COLOUR_MUTATOR}")
    if not refute_attempt_json or refute_attempt_json == "NONE":
        return skipped_colour_guided_mutation("no_refute_attempt_manifest")
    mutation_dir = cell_dir / "colour-guided-mutation"
    stdout_path = cell_dir / "colour-guided-mutation.out"
    cmd = [
        sys.executable,
        str(COLOUR_MUTATOR),
        str(mutation_dir),
        "--satfanout-json",
        refute_attempt_json,
        "--candidate-prefix",
        f"{candidate_prefix}_cg{cell_index:03d}",
        "--max-den",
        str(max_den),
        "--max-candidates",
        str(max_candidates),
        "--top-points",
        str(top_points),
        "--emit-mutations",
        str(emit_mutations),
        "--add-points",
        str(add_points),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path = stdout_path.with_name(stdout_path.name + ".stderr")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"colour-guided mutation failed with exit {proc.returncode}: {detail}")
    fields = parse_kv_output(proc.stdout)
    mutation_json = Path(fields.get("mutation_json", ""))
    if not mutation_json.is_file():
        raise RuntimeError(f"colour-guided mutation did not emit mutation_json: {mutation_json}")
    mutation = load_json(mutation_json)
    if mutation.get("schema") != "chi6_colour_guided_mutation.v1":
        raise RuntimeError("colour-guided mutation schema mismatch")
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if mutation.get(key) != "none":
            raise RuntimeError(f"colour-guided mutation leaked {key}={mutation.get(key)}")
    if mutation.get("promotable") != 0:
        raise RuntimeError("colour-guided mutation leaked promotable claim")
    return {
        "status": fields.get("status", "COLOUR_GUIDED_MUTATION_RECORDED"),
        "reason": "sat_colourings_consumed",
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "mutation_json": str(mutation_json),
        "mutation_json_sha256": sha256_file(mutation_json),
        "candidate_point_count_scored": int(mutation["candidate_point_count_scored"]),
        "single_point_full_blocker_count": int(mutation["single_point_full_blocker_count"]),
        "mutation_count": int(mutation["mutation_count"]),
        "first_mutation_coords_csv": fields.get("first_mutation_coords_csv", "NONE"),
        "first_mutation_new_n": int(fields.get("first_mutation_new_n", "0")),
        "first_mutation_new_m": int(fields.get("first_mutation_new_m", "0")),
        "first_mutation_killed_colouring_count_by_unit_neighborhood": int(
            fields.get("first_mutation_killed_colouring_count_by_unit_neighborhood", "0")
        ),
    }


def run_fractal_cell(
    *,
    out_dir: Path,
    global_index: int,
    generation: int,
    parent_index: int | None,
    branch: str,
    candidate_prefix: str,
    params: CellParams,
    preflight_limit: int,
    skip_coloring_found: bool,
    colour_guided_mutations: bool,
    colour_max_den: int,
    colour_max_candidates: int,
    colour_top_points: int,
    colour_emit_mutations: int,
    colour_add_points: int,
) -> dict[str, Any]:
    cell_dir = out_dir / f"generation_{generation:02d}" / f"cell_{global_index:03d}_{branch}"
    cell = run_cell(
        cell_dir=cell_dir,
        cell_index=global_index,
        candidate_prefix=f"{candidate_prefix}_g{generation:02d}",
        max_den_list=str(params.max_den),
        layers_list=str(params.layers),
        max_points_list=str(params.max_points),
        min_vertices=params.min_vertices,
        min_edges=params.min_edges,
        split_depth=params.split_depth,
        min_split_degree=params.min_split_degree,
        dsatur_node_limit=params.dsatur_node_limit,
        preflight_limit=preflight_limit,
        max_cubes=params.max_cubes,
        sample_hard_cubes=params.sample_hard_cubes,
        refute_limit=params.refute_limit,
        timeout_seconds=params.timeout_seconds,
        skip_coloring_found=skip_coloring_found,
    )
    score, signal = attention_score(cell)
    cell.update(
        {
            "generation": generation,
            "parent_cell_index": parent_index if parent_index is not None else "NONE",
            "branch": branch,
            "fractal_params": params.as_manifest(),
            "attention_policy": ATTENTION_POLICY,
            "attention_score": score,
            "attention_signal": signal,
        }
    )
    if colour_guided_mutations and as_int(cell["status_counts"].get("REFUTE_SAT_MUTATE_FRONTIER")):
        cell["colour_guided_mutation"] = run_colour_guided_mutation(
            cell_dir=cell_dir,
            cell_index=global_index,
            refute_attempt_json=str(cell["refute_attempt_json"]),
            candidate_prefix=candidate_prefix,
            max_den=colour_max_den,
            max_candidates=colour_max_candidates,
            top_points=colour_top_points,
            emit_mutations=colour_emit_mutations,
            add_points=colour_add_points,
        )
    else:
        reason = "disabled" if not colour_guided_mutations else "no_sat_colourings"
        cell["colour_guided_mutation"] = skipped_colour_guided_mutation(reason)
    return cell


def failed_fractal_cell(
    *,
    out_dir: Path,
    global_index: int,
    generation: int,
    parent_index: int | None,
    branch: str,
    params: CellParams,
    priority_hint: int,
    exc: BaseException,
) -> dict[str, Any]:
    cell_dir = out_dir / f"generation_{generation:02d}" / f"cell_{global_index:03d}_{branch}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    failure_path = cell_dir / "fractal_cell_failure.txt"
    failure_text = f"{type(exc).__name__}: {exc}\n"
    failure_path.write_text(failure_text, encoding="utf-8")
    return {
        "cell_index": global_index,
        "cell_dir": str(cell_dir),
        "generation": generation,
        "parent_cell_index": parent_index if parent_index is not None else "NONE",
        "branch": branch,
        "incoming_priority_hint": priority_hint,
        "fractal_params": params.as_manifest(),
        "attention_policy": ATTENTION_POLICY,
        "attention_score": 0,
        "attention_signal": {
            "candidate_id": "NONE",
            "n": 0,
            "m": 0,
            "max_degree": 0,
            "dsatur_status": "NONE",
            "dsatur_nodes_capped": 0,
            "geometry_status": "NONE",
            "preflight_refute_ready_count": 0,
            "attempt_count": 0,
            "refute_success_count": 0,
            "noresult_count": 0,
            "attention_score": 0,
        },
        "failure_path": str(failure_path),
        "failure_message": failure_text.strip(),
        "campaign_stdout": "NONE",
        "campaign_stderr": "NONE",
        "campaign_json": "NONE",
        "campaign_json_sha256": "NONE",
        "campaign_count": 0,
        "preflight_batch_stdout": "NONE",
        "preflight_batch_stderr": "NONE",
        "preflight_batch_json": "NONE",
        "preflight_batch_sha256": "NONE",
        "preflight_selected_count": 0,
        "preflight_refute_ready_count": 0,
        "preflight_status": "FRACTAL_CELL_FAILED_INFRA",
        "refute_attempt_stdout": "NONE",
        "refute_attempt_stderr": "NONE",
        "refute_attempt_json": "NONE",
        "refute_attempt_sha256": "NONE",
        "attempt_count": 0,
        "refute_success_count": 0,
        "refute_failed_count": 0,
        "first_success_candidate": "NONE",
        "first_failed_candidate": "NONE",
        "status_counts": {"FRACTAL_CELL_FAILED_INFRA": 1},
        "recommended_next_action": "discard_invalid_branch",
        "colour_guided_mutation": skipped_colour_guided_mutation(
            "cell_failed_before_refute_attempt"
        ),
        "children": [],
    }


def initial_params(args: argparse.Namespace) -> list[CellParams]:
    max_dens = parse_positive_int_list(args.max_den_list, "--max-den-list")
    layers = parse_positive_int_list(args.layers_list, "--layers-list")
    max_points = parse_positive_int_list(args.max_points_list, "--max-points-list")
    return [
        CellParams(
            max_den=max_den,
            layers=layer,
            max_points=points,
            min_vertices=args.min_vertices,
            min_edges=args.min_edges,
            split_depth=args.split_depth,
            min_split_degree=args.min_split_degree,
            dsatur_node_limit=args.dsatur_node_limit,
            max_cubes=args.max_cubes,
            sample_hard_cubes=args.sample_hard_cubes,
            refute_limit=args.refute_limit,
            timeout_seconds=args.timeout_seconds,
        )
        for max_den, layer, points in itertools.product(max_dens, layers, max_points)
    ]


def validate_args(args: argparse.Namespace) -> None:
    if args.max_generations < 1:
        raise ValueError("--max-generations must be positive")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be positive")
    if args.children_per_cell < 0:
        raise ValueError("--children-per-cell must be non-negative")
    if args.min_vertices < 2:
        raise ValueError("--min-vertices must be at least 2")
    if args.min_edges < 1:
        raise ValueError("--min-edges must be positive")
    if args.split_depth < 1:
        raise ValueError("--split-depth must be positive")
    if args.min_split_degree < 1:
        raise ValueError("--min-split-degree must be positive")
    if args.dsatur_node_limit < 1:
        raise ValueError("--dsatur-node-limit must be positive")
    if args.preflight_limit < 1:
        raise ValueError("--preflight-limit must be positive")
    if args.max_cubes < 1:
        raise ValueError("--max-cubes must be positive")
    if args.sample_hard_cubes < 0:
        raise ValueError("--sample-hard-cubes must be non-negative")
    if args.sample_hard_cubes > args.max_cubes:
        raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")
    if args.refute_limit < 0:
        raise ValueError("--refute-limit must be non-negative")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout-seconds must be non-negative")
    if args.max_den_cap < 1:
        raise ValueError("--max-den-cap must be positive")
    if args.layers_cap < 1:
        raise ValueError("--layers-cap must be positive")
    if args.max_points_cap < 1:
        raise ValueError("--max-points-cap must be positive")
    if args.split_depth_cap < 1:
        raise ValueError("--split-depth-cap must be positive")
    if args.max_cubes_cap < 1:
        raise ValueError("--max-cubes-cap must be positive")
    if args.sample_hard_cubes_cap < 0:
        raise ValueError("--sample-hard-cubes-cap must be non-negative")
    if args.point_growth < 1:
        raise ValueError("--point-growth must be positive")
    if args.layer_growth < 1:
        raise ValueError("--layer-growth must be positive")
    if args.den_growth < 1:
        raise ValueError("--den-growth must be positive")
    if args.sample_growth < 0:
        raise ValueError("--sample-growth must be non-negative")
    if args.colour_max_den < 1:
        raise ValueError("--colour-max-den must be positive")
    if args.colour_max_candidates < 1:
        raise ValueError("--colour-max-candidates must be positive")
    if args.colour_top_points < 1:
        raise ValueError("--colour-top-points must be positive")
    if args.colour_emit_mutations < 1:
        raise ValueError("--colour-emit-mutations must be positive")
    if args.colour_add_points < 1:
        raise ValueError("--colour-add-points must be positive")


def recommendation_from_totals(
    attempt_count: int,
    refute_success_count: int,
    expanded_cell_count: int,
    frontier_leaf_count: int,
) -> str:
    if refute_success_count > 0:
        return "package_leaf_lrat_with_cover_lrat_and_geometry_next"
    if expanded_cell_count > 0:
        return "continue_fractal_attention_or_raise_refuter_budget"
    if attempt_count == 0:
        return "expand_frontier_parameters_or_relax_selection"
    if frontier_leaf_count > 0:
        return "raise_caps_or_refuter_budget"
    return "inspect_fractal_sweep_infra"


def status_counts_from_cells(cells: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cell in cells:
        for key, raw in cell["status_counts"].items():
            counts[str(key)] = counts.get(str(key), 0) + int(raw)
    return counts


def colour_guided_mutation_count(cells: list[dict[str, Any]]) -> int:
    return sum(
        1
        for cell in cells
        if cell.get("colour_guided_mutation", {}).get("status") == "COLOUR_GUIDED_MUTATION_RECORDED"
    )


def colour_guided_single_point_full_blocker_count(cells: list[dict[str, Any]]) -> int:
    return sum(
        as_int(cell.get("colour_guided_mutation", {}).get("single_point_full_blocker_count"))
        for cell in cells
    )


def write_checkpoint(
    *,
    out_dir: Path,
    requested_max_generations: int,
    beam_width: int,
    children_per_cell: int,
    expanded_cell_count: int,
    frontier_leaf_count: int,
    next_frontier: list[tuple[int, int | None, str, CellParams]],
    cells: list[dict[str, Any]],
) -> None:
    if not cells:
        return
    checkpoint = {
        "schema": "chi6_fractal_frontier_sweep_checkpoint.v1",
        "requested_max_generations": requested_max_generations,
        "beam_width": beam_width,
        "children_per_cell": children_per_cell,
        "completed_cell_count": len(cells),
        "expanded_cell_count": expanded_cell_count,
        "frontier_leaf_count": frontier_leaf_count,
        "cell_failure_count": sum(
            1 for cell in cells if cell["preflight_status"] == "FRACTAL_CELL_FAILED_INFRA"
        ),
        "status_counts": status_counts_from_cells(cells),
        "attempt_count": sum(int(cell["attempt_count"]) for cell in cells),
        "refute_success_count": sum(int(cell["refute_success_count"]) for cell in cells),
        "refute_failed_count": sum(int(cell["refute_failed_count"]) for cell in cells),
        "attention_policy": ATTENTION_POLICY,
        "colour_guided_mutation_count": colour_guided_mutation_count(cells),
        "colour_guided_single_point_full_blocker_count": (
            colour_guided_single_point_full_blocker_count(cells)
        ),
        "next_frontier_count": len(next_frontier),
        "next_frontier": [
            {
                "priority_hint": priority_hint,
                "parent_cell_index": parent_index if parent_index is not None else "NONE",
                "branch": branch,
                "parameters": params.as_manifest(),
            }
            for priority_hint, parent_index, branch, params in next_frontier
        ],
        "claim_scope": "fractal_frontier_sweep_checkpoint_only",
        "sat_claim": "none",
        "chromatic_claim": "none",
        "global_unsat_claim": "none",
        "verified_claim": "none",
        "promotable": 0,
        "cells": cells,
    }
    checkpoint_path = out_dir / "fractal_frontier_sweep.checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--candidate-prefix", default="fractal")
    parser.add_argument("--max-den-list", default="5")
    parser.add_argument("--layers-list", default="2")
    parser.add_argument("--max-points-list", default="64")
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=2)
    parser.add_argument("--min-split-degree", type=int, default=4)
    parser.add_argument("--dsatur-node-limit", type=int, default=1)
    parser.add_argument("--preflight-limit", type=int, default=1)
    parser.add_argument("--max-cubes", type=int, default=20_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=8)
    parser.add_argument("--refute-limit", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--children-per-cell", type=int, default=2)
    parser.add_argument("--max-den-cap", type=int, default=7)
    parser.add_argument("--layers-cap", type=int, default=4)
    parser.add_argument("--max-points-cap", type=int, default=160)
    parser.add_argument("--split-depth-cap", type=int, default=4)
    parser.add_argument("--max-cubes-cap", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes-cap", type=int, default=64)
    parser.add_argument("--point-growth", type=int, default=32)
    parser.add_argument("--layer-growth", type=int, default=1)
    parser.add_argument("--den-growth", type=int, default=1)
    parser.add_argument("--sample-growth", type=int, default=8)
    parser.add_argument("--skip-coloring-found", action="store_true")
    parser.add_argument("--colour-guided-mutations", action="store_true")
    parser.add_argument("--colour-max-den", type=int, default=5)
    parser.add_argument("--colour-max-candidates", type=int, default=20_000)
    parser.add_argument("--colour-top-points", type=int, default=50)
    parser.add_argument("--colour-emit-mutations", type=int, default=4)
    parser.add_argument("--colour-add-points", type=int, default=4)
    parser.add_argument("--stop-after-first-success", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        validate_args(args)
        if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
            raise ValueError("out_dir already exists and is non-empty; pass --resume to reuse it")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        frontier: list[tuple[int, int | None, str, CellParams]] = [
            (0, None, "seed", params) for params in initial_params(args)
        ]
        seen_params = {params for _hint, _parent, _branch, params in frontier}
        cells: list[dict[str, Any]] = []
        expanded_cell_count = 0
        frontier_leaf_count = 0
        global_index = 0
        stopped_after_success = False

        for generation in range(args.max_generations):
            if not frontier:
                break
            frontier.sort(key=lambda item: (-item[0], item[3].key()))
            active = frontier[: args.beam_width]
            overflow = frontier[args.beam_width :]
            for priority_hint, parent_index, branch, params in active:
                try:
                    cell = run_fractal_cell(
                        out_dir=args.out_dir,
                        global_index=global_index,
                        generation=generation,
                        parent_index=parent_index,
                        branch=branch,
                        candidate_prefix=args.candidate_prefix,
                        params=params,
                        preflight_limit=args.preflight_limit,
                        skip_coloring_found=args.skip_coloring_found,
                        colour_guided_mutations=args.colour_guided_mutations,
                        colour_max_den=args.colour_max_den,
                        colour_max_candidates=args.colour_max_candidates,
                        colour_top_points=args.colour_top_points,
                        colour_emit_mutations=args.colour_emit_mutations,
                        colour_add_points=args.colour_add_points,
                    )
                    cell["incoming_priority_hint"] = priority_hint
                except (
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    cell = failed_fractal_cell(
                        out_dir=args.out_dir,
                        global_index=global_index,
                        generation=generation,
                        parent_index=parent_index,
                        branch=branch,
                        params=params,
                        priority_hint=priority_hint,
                        exc=exc,
                    )
                cells.append(cell)
                current_index = global_index
                global_index += 1
                if int(cell["refute_success_count"]) > 0 and args.stop_after_first_success:
                    stopped_after_success = True
                    write_checkpoint(
                        out_dir=args.out_dir,
                        requested_max_generations=args.max_generations,
                        beam_width=args.beam_width,
                        children_per_cell=args.children_per_cell,
                        expanded_cell_count=expanded_cell_count,
                        frontier_leaf_count=frontier_leaf_count,
                        next_frontier=overflow,
                        cells=cells,
                    )
                    break
                if (
                    generation + 1 >= args.max_generations
                    or cell["recommended_next_action"] not in EXPAND_ACTIONS
                ):
                    frontier_leaf_count += 1
                    cell["children"] = []
                    write_checkpoint(
                        out_dir=args.out_dir,
                        requested_max_generations=args.max_generations,
                        beam_width=args.beam_width,
                        children_per_cell=args.children_per_cell,
                        expanded_cell_count=expanded_cell_count,
                        frontier_leaf_count=frontier_leaf_count,
                        next_frontier=overflow,
                        cells=cells,
                    )
                    continue
                score = int(cell["attention_score"])
                children = expand_params(
                    parent=params,
                    parent_score=score,
                    max_den_cap=args.max_den_cap,
                    layers_cap=args.layers_cap,
                    max_points_cap=args.max_points_cap,
                    split_depth_cap=args.split_depth_cap,
                    max_cubes_cap=args.max_cubes_cap,
                    sample_hard_cubes_cap=args.sample_hard_cubes_cap,
                    point_growth=args.point_growth,
                    layer_growth=args.layer_growth,
                    den_growth=args.den_growth,
                    sample_growth=args.sample_growth,
                    children_per_cell=args.children_per_cell,
                )
                accepted_children = []
                for child_branch, child_params, child_hint in children:
                    if child_params in seen_params:
                        continue
                    seen_params.add(child_params)
                    accepted_children.append(
                        (child_hint, current_index, child_branch, child_params)
                    )
                if accepted_children:
                    expanded_cell_count += 1
                    cell["children"] = [
                        {
                            "branch": child_branch,
                            "priority_hint": child_hint,
                            "parameters": child_params.as_manifest(),
                        }
                        for child_hint, _parent, child_branch, child_params in accepted_children
                    ]
                    overflow.extend(accepted_children)
                else:
                    cell["children"] = []
                    frontier_leaf_count += 1
                write_checkpoint(
                    out_dir=args.out_dir,
                    requested_max_generations=args.max_generations,
                    beam_width=args.beam_width,
                    children_per_cell=args.children_per_cell,
                    expanded_cell_count=expanded_cell_count,
                    frontier_leaf_count=frontier_leaf_count,
                    next_frontier=overflow,
                    cells=cells,
                )
            if stopped_after_success:
                break
            frontier = overflow

        if not cells:
            raise RuntimeError("fractal sweep executed zero cells")

        status_counts = status_counts_from_cells(cells)
        attempt_count = sum(int(cell["attempt_count"]) for cell in cells)
        refute_success_count = sum(int(cell["refute_success_count"]) for cell in cells)
        refute_failed_count = sum(int(cell["refute_failed_count"]) for cell in cells)
        first_success_candidate = next(
            (
                cell["first_success_candidate"]
                for cell in cells
                if cell["first_success_candidate"] != "NONE"
            ),
            "NONE",
        )
        first_failed_candidate = next(
            (
                cell["first_failed_candidate"]
                for cell in cells
                if cell["first_failed_candidate"] != "NONE"
            ),
            "NONE",
        )
        manifest = {
            "schema": "chi6_fractal_frontier_sweep.v1",
            "generation_count": max(cell["generation"] for cell in cells) + 1,
            "requested_max_generations": args.max_generations,
            "beam_width": args.beam_width,
            "children_per_cell": args.children_per_cell,
            "cell_count": len(cells),
            "expanded_cell_count": expanded_cell_count,
            "frontier_leaf_count": frontier_leaf_count,
            "cell_failure_count": sum(
                1
                for cell in cells
                if cell["preflight_status"] == "FRACTAL_CELL_FAILED_INFRA"
            ),
            "status_counts": status_counts,
            "attempt_count": attempt_count,
            "refute_success_count": refute_success_count,
            "refute_failed_count": refute_failed_count,
            "first_success_candidate": first_success_candidate,
            "first_failed_candidate": first_failed_candidate,
            "best_attention_cell_index": max(
                range(len(cells)),
                key=lambda idx: int(cells[idx]["attention_score"]),
            ),
            "best_attention_score": max(int(cell["attention_score"]) for cell in cells),
            "attention_policy": ATTENTION_POLICY,
            "colour_guided_mutations_enabled": 1 if args.colour_guided_mutations else 0,
            "colour_guided_mutation_count": colour_guided_mutation_count(cells),
            "colour_guided_single_point_full_blocker_count": (
                colour_guided_single_point_full_blocker_count(cells)
            ),
            "recommended_next_action": recommendation_from_totals(
                attempt_count,
                refute_success_count,
                expanded_cell_count,
                frontier_leaf_count,
            ),
            "claim_scope": "fractal_frontier_sweep_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_checked_cube_cover_lrat_lean_exact_geometry_real_bridge",
            "cells": cells,
        }
        manifest_path = args.out_dir / "fractal_frontier_sweep.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_fractal_frontier_sweep v1")
    print(f"fractal_json={manifest_path}")
    print(f"fractal_json_sha256={sha256_file(manifest_path)}")
    print(f"generation_count={manifest['generation_count']}")
    print(f"cell_count={manifest['cell_count']}")
    print(f"expanded_cell_count={manifest['expanded_cell_count']}")
    print(f"frontier_leaf_count={manifest['frontier_leaf_count']}")
    print(f"cell_failure_count={manifest['cell_failure_count']}")
    print(f"attempt_count={manifest['attempt_count']}")
    print(f"refute_success_count={manifest['refute_success_count']}")
    print(f"refute_failed_count={manifest['refute_failed_count']}")
    print(f"first_success_candidate={manifest['first_success_candidate']}")
    print(f"first_failed_candidate={manifest['first_failed_candidate']}")
    print(f"best_attention_cell_index={manifest['best_attention_cell_index']}")
    print(f"best_attention_score={manifest['best_attention_score']}")
    print(f"attention_policy={ATTENTION_POLICY}")
    print(f"colour_guided_mutation_count={manifest['colour_guided_mutation_count']}")
    print(
        "colour_guided_single_point_full_blocker_count="
        f"{manifest['colour_guided_single_point_full_blocker_count']}"
    )
    print(f"recommended_next_action={manifest['recommended_next_action']}")
    print("claim_scope=fractal_frontier_sweep_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=FRACTAL_FRONTIER_SWEEP_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
