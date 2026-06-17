#!/usr/bin/env python3
"""Beam-search SAT-colouring-guided exact-rational frontier evolution.

This is the bolder search scheduler over `chi6_colour_guided_evolution.py`.
Instead of following one selected mutation per generation, it evaluates several
mutation children from each active parent, scouts/preflights every evaluated
child, scores them as search signals, and keeps the highest-scoring coloured
children as the next beam.

No ranking here is a proof.  A child without a bounded DSATUR colouring is an
escalation/refuter signal only; promotion still requires exact geometry plus
checked SAT/LRAT/Lean artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chi6_colour_guided_evolution import (
    colouring_line,
    load_json,
    run_mutation,
    run_preflight,
    run_scout,
    run_command,
    sha256_file,
    validate_args as validate_evolution_args,
)
from chi6_colour_guided_mutation import load_colourings_file, load_from_satfanout
from make_chi6_rational_unit_graph_source_package import parse_coord_table, unit_edges


PREFLIGHT_TO_BATCH = Path(__file__).with_name("chi6_campaign_preflight_to_refute_batch.py")
REFUTE_ATTEMPT = Path(__file__).with_name("chi6_frontier_refute_attempt.py")
DSATUR_SCORE = {
    "NO_K_COLORING_FOUND_BY_CPU_PROBE_NONCERTIFYING": 2_000_000,
    "UNKNOWN_NODE_LIMIT": 1_000_000,
    "K_COLORING_FOUND": 0,
}
ACTION_SCORE = {
    "prepare_cube_refute_batch": 700_000,
    "propagation_conflicts_require_lrat_lean_upgrade": 400_000,
    "reject_or_mutate_frontier": 0,
}


@dataclass(frozen=True)
class Parent:
    parent_id: str
    coords_csv: Path | None
    colourings_file: Path | None
    satfanout_json: Path | None
    lineage: tuple[str, ...]


def child_score(row: dict[str, Any]) -> int:
    return (
        DSATUR_SCORE.get(str(row["dsatur_status"]), 0)
        + ACTION_SCORE.get(str(row["recommended_next_action"]), 0)
        + int(row["selected_killed_colouring_count_by_unit_neighborhood"]) * 120_000
        + int(row["single_point_full_blocker_count"]) * 30_000
        + int(row["selected_edge_gain_after_mutation"]) * 2_000
        + int(row["selected_existing_neighbor_count_total"]) * 600
        + int(row["propagation_hard_count"]) * 100
        + int(row["selected_new_m"]) * 450
        + int(row["selected_new_n"]) * 60
        + min(int(row["dsatur_nodes"]), 100_000)
    )


def require_no_claims(meta: dict[str, Any], label: str) -> None:
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if key in meta and meta[key] != "none":
            raise ValueError(f"{label} leaked {key}={meta[key]}")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} leaked promotable={meta.get('promotable')}")


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


def parent_colouring_lines(parent: Parent) -> list[str]:
    if parent.colourings_file is not None:
        return load_colourings_file(parent.colourings_file)
    if parent.satfanout_json is not None:
        _coords_csv, _candidate_id, raw_colourings = load_from_satfanout(parent.satfanout_json)
        return raw_colourings
    return []


def parse_indexed_colouring(raw: str, n: int) -> list[int] | None:
    colours = [-1] * n
    for token in raw.split(","):
        if ":" not in token:
            return None
        v_raw, c_raw = token.split(":", 1)
        if not v_raw.isdigit() or not c_raw.isdigit():
            return None
        v = int(v_raw)
        c = int(c_raw)
        if not (0 <= v < n and 0 <= c < 5) or colours[v] >= 0:
            return None
        colours[v] = c
    if any(c < 0 for c in colours):
        return None
    return colours


def extend_colouring(
    *,
    old_line: str,
    old_n: int,
    new_n: int,
    edges: list[tuple[int, int]],
) -> list[int] | None:
    colours = parse_indexed_colouring(old_line, old_n)
    if colours is None:
        return None
    colours.extend([-1] * (new_n - old_n))
    adj = [set() for _ in range(new_n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if u < old_n and v < old_n and colours[u] == colours[v]:
            return None

    def choose_vertex() -> int | None:
        uncoloured = [v for v in range(old_n, new_n) if colours[v] < 0]
        if not uncoloured:
            return None
        return min(
            uncoloured,
            key=lambda v: (
                len([c for c in range(5) if c not in {colours[u] for u in adj[v] if colours[u] >= 0}]),
                -len(adj[v]),
                v,
            ),
        )

    def search() -> bool:
        v = choose_vertex()
        if v is None:
            return True
        used = {colours[u] for u in adj[v] if colours[u] >= 0}
        for c in range(5):
            if c in used:
                continue
            colours[v] = c
            if search():
                return True
            colours[v] = -1
        return False

    if not search():
        return None
    return colours


def write_next_colourings_file(
    *,
    path: Path,
    parent: Parent,
    selected_coords: Path,
    old_n: int,
    scout_colouring: list[int],
    max_carried_colourings: int,
) -> dict[str, int]:
    new_n = len(scout_colouring)
    coords = parse_coord_table(selected_coords, 1_000_000)
    if len(coords) != new_n:
        raise RuntimeError("selected coords and scout colouring length disagree")
    edges = unit_edges(coords)
    lines: list[str] = []
    extended_count = 0
    for raw in parent_colouring_lines(parent):
        extended = extend_colouring(
            old_line=raw,
            old_n=old_n,
            new_n=new_n,
            edges=edges,
        )
        if extended is None:
            continue
        lines.append(colouring_line(extended, new_n, "extended_parent_colouring"))
        extended_count += 1
    lines.append(colouring_line(scout_colouring, new_n, "scout_colouring"))

    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    deduped = deduped[-max_carried_colourings:]
    path.write_text("\n".join(deduped) + "\n", encoding="ascii")
    return {
        "next_colouring_history_count": len(deduped),
        "next_extended_parent_colouring_count": extended_count,
    }


def skipped_refute_attempt(reason: str) -> dict[str, Any]:
    return {
        "status": "REFUTE_ATTEMPT_SKIPPED",
        "reason": reason,
        "preflight_batch_json": "NONE",
        "preflight_batch_sha256": "NONE",
        "refute_attempt_json": "NONE",
        "refute_attempt_sha256": "NONE",
        "attempt_count": 0,
        "refute_success_count": 0,
        "refute_failed_count": 0,
        "status_counts": {},
    }


def run_refute_attempt_if_ready(
    *,
    child_dir: Path,
    preflight_json: Path,
    preflight: dict[str, Any],
    run_refute_ready: bool,
    refute_limit: int,
    refute_timeout_seconds: int,
) -> dict[str, Any]:
    if not run_refute_ready:
        return skipped_refute_attempt("disabled")
    if preflight.get("recommended_next_action") != "prepare_cube_refute_batch":
        return skipped_refute_attempt("not_refute_ready")
    if not PREFLIGHT_TO_BATCH.is_file() or not REFUTE_ATTEMPT.is_file():
        raise ValueError("missing refute bridge or refute attempt tool")
    batch_stdout = child_dir / "refute-batch.out"
    batch_text = run_command(
        [
            sys.executable,
            str(PREFLIGHT_TO_BATCH),
            str(preflight_json),
            str(child_dir / "refute-batch"),
        ],
        batch_stdout,
    )
    batch_fields = parse_kv_output(batch_text)
    batch_json = Path(batch_fields.get("preflight_batch_json", ""))
    if not batch_json.is_file():
        raise RuntimeError(f"refute bridge did not emit preflight_batch_json: {batch_json}")
    attempt_cmd = [
        sys.executable,
        str(REFUTE_ATTEMPT),
        str(batch_json),
        str(child_dir / "refute-attempt"),
    ]
    if refute_limit > 0:
        attempt_cmd.extend(["--limit", str(refute_limit)])
    if refute_timeout_seconds > 0:
        attempt_cmd.extend(["--timeout-seconds", str(refute_timeout_seconds)])
    attempt_stdout = child_dir / "refute-attempt.out"
    attempt_text = run_command(attempt_cmd, attempt_stdout)
    attempt_fields = parse_kv_output(attempt_text)
    attempt_json = Path(attempt_fields.get("refute_attempt_json", ""))
    if not attempt_json.is_file():
        raise RuntimeError(f"refute attempt did not emit refute_attempt_json: {attempt_json}")
    attempt = load_json(attempt_json)
    if attempt.get("schema") != "chi6_frontier_refute_attempt.v1":
        raise RuntimeError("refute attempt schema mismatch")
    require_no_claims(attempt, "refute attempt")
    return {
        "status": "REFUTE_ATTEMPT_RECORDED",
        "reason": "refute_ready_child",
        "preflight_batch_stdout": str(batch_stdout),
        "preflight_batch_json": str(batch_json),
        "preflight_batch_sha256": sha256_file(batch_json),
        "refute_attempt_stdout": str(attempt_stdout),
        "refute_attempt_json": str(attempt_json),
        "refute_attempt_sha256": sha256_file(attempt_json),
        "attempt_count": int(attempt["attempt_count"]),
        "refute_success_count": int(attempt["refute_success_count"]),
        "refute_failed_count": int(attempt["refute_failed_count"]),
        "status_counts": attempt["status_counts"],
    }


def evaluate_child(
    *,
    generation_dir: Path,
    generation: int,
    parent: Parent,
    parent_rank: int,
    mutation: dict[str, Any],
    selected_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Parent | None]:
    selected = mutation["mutations"][selected_index]
    selected_coords = Path(str(selected["coords_csv"]))
    if not selected_coords.is_file():
        raise RuntimeError(f"selected mutation coords_csv missing: {selected_coords}")
    child_dir = generation_dir / f"parent_{parent_rank:03d}" / f"child_{selected_index:03d}"
    candidate_id = str(
        selected.get("candidate_id_hint")
        or f"{args.candidate_prefix}_g{generation:02d}_p{parent_rank:03d}_m{selected_index:03d}"
    )
    scout_fields, scout, scout_stdout = run_scout(
        generation_dir=child_dir,
        candidate_id=candidate_id,
        coords_csv=selected_coords,
        max_vertices=args.max_vertices,
        min_vertices=args.min_vertices,
        min_edges=args.min_edges,
        split_depth=args.split_depth,
        min_split_degree=args.min_split_degree,
        dsatur_node_limit=args.dsatur_node_limit,
    )
    scout_json = Path(scout_fields["frontier_scout"])
    preflight_fields, preflight, preflight_stdout = run_preflight(
        generation_dir=child_dir,
        scout_json=scout_json,
        max_cubes=args.max_cubes,
        sample_hard_cubes=args.sample_hard_cubes,
    )
    require_no_claims(scout, "scout")
    require_no_claims(preflight, "preflight")
    preflight_json = Path(preflight_fields["campaign_preflight_json"])
    refute_attempt = run_refute_attempt_if_ready(
        child_dir=child_dir,
        preflight_json=preflight_json,
        preflight=preflight,
        run_refute_ready=args.run_refute_ready,
        refute_limit=args.refute_limit,
        refute_timeout_seconds=args.refute_timeout_seconds,
    )

    next_parent: Parent | None = None
    next_colourings_file = "NONE"
    next_colouring_vertex_count = 0
    next_colouring_history_count = 0
    next_extended_parent_colouring_count = 0
    if isinstance(scout.get("colouring"), list):
        next_colourings_path = child_dir / "next-colourings.txt"
        history = write_next_colourings_file(
            path=next_colourings_path,
            parent=parent,
            selected_coords=selected_coords,
            old_n=int(selected["old_n"]),
            scout_colouring=scout["colouring"],
            max_carried_colourings=args.max_carried_colourings,
        )
        next_colourings_file = str(next_colourings_path)
        next_colouring_vertex_count = int(scout["n"])
        next_colouring_history_count = history["next_colouring_history_count"]
        next_extended_parent_colouring_count = history["next_extended_parent_colouring_count"]
        next_parent = Parent(
            parent_id=f"g{generation:02d}_p{parent_rank:03d}_m{selected_index:03d}",
            coords_csv=selected_coords,
            colourings_file=next_colourings_path,
            satfanout_json=None,
            lineage=parent.lineage + (candidate_id,),
        )

    row = {
        "generation": generation,
        "parent_id": parent.parent_id,
        "parent_rank": parent_rank,
        "selected_mutation_index": selected_index,
        "candidate_id": candidate_id,
        "lineage": list(parent.lineage + (candidate_id,)),
        "selected_coords_csv": str(selected_coords),
        "selected_coords_sha256": sha256_file(selected_coords),
        "selected_old_n": int(selected["old_n"]),
        "selected_new_n": int(selected["new_n"]),
        "selected_new_m": int(selected["new_m"]),
        "selected_edge_gain_after_mutation": int(selected["edge_gain_after_mutation"]),
        "selected_existing_neighbor_count_total": int(
            selected["selected_existing_neighbor_count_total"]
        ),
        "selected_existing_neighbor_count_min": int(
            selected["selected_existing_neighbor_count_min"]
        ),
        "selected_existing_neighbor_count_max": int(
            selected["selected_existing_neighbor_count_max"]
        ),
        "selected_added_point_count": int(selected["added_point_count"]),
        "selected_killed_colouring_count_by_unit_neighborhood": int(
            selected["killed_colouring_count_by_unit_neighborhood"]
        ),
        "selected_mutation_strategy": selected["selection_strategy"],
        "mutation_min_neighbor_count": int(mutation["min_neighbor_count"]),
        "mutation_edge_gain_pool_points": int(mutation["edge_gain_pool_points"]),
        "mutation_edge_gain_emit_mutations": int(mutation["edge_gain_emit_mutations"]),
        "mutation_edge_gain_combination_offset": int(
            mutation["edge_gain_combination_offset"]
        ),
        "mutation_edge_gain_combination_stride": int(
            mutation["edge_gain_combination_stride"]
        ),
        "mutation_edge_gain_considered_combination_count": int(
            mutation["edge_gain_considered_combination_count"]
        ),
        "mutation_edge_gain_combination_truncated": bool(
            mutation["edge_gain_combination_truncated"]
        ),
        "single_point_full_blocker_count": int(mutation["single_point_full_blocker_count"]),
        "candidate_point_count_scored": int(mutation["candidate_point_count_scored"]),
        "frontier_scout": scout_fields["frontier_scout"],
        "frontier_scout_sha256": sha256_file(scout_json),
        "scout_stdout": str(scout_stdout),
        "dsatur_status": scout["dsatur_status"],
        "dsatur_nodes": int(scout["dsatur_nodes"]),
        "campaign_preflight_json": preflight_fields["campaign_preflight_json"],
        "campaign_preflight_sha256": sha256_file(
            Path(preflight_fields["campaign_preflight_json"])
        ),
        "preflight_stdout": str(preflight_stdout),
        "recommended_next_action": preflight["recommended_next_action"],
        "refute_attempt": refute_attempt,
        "cube_count": int(preflight["cube_count"]),
        "propagation_conflict_count": int(preflight["propagation_conflict_count"]),
        "propagation_hard_count": int(preflight["propagation_hard_count"]),
        "next_colourings_file": next_colourings_file,
        "next_colouring_vertex_count": next_colouring_vertex_count,
        "next_colouring_history_count": next_colouring_history_count,
        "next_extended_parent_colouring_count": next_extended_parent_colouring_count,
        "claim_scope": "colour_guided_beam_child_only",
        "sat_claim": "none",
        "chromatic_claim": "none",
        "global_unsat_claim": "none",
        "verified_claim": "none",
        "promotable": 0,
    }
    row["beam_score"] = child_score(row)
    return row, next_parent


def validate_args(args: argparse.Namespace) -> None:
    validate_evolution_args(args)
    if args.beam_width < 1:
        raise ValueError("--beam-width must be positive")
    if args.branch_width < 1:
        raise ValueError("--branch-width must be positive")
    if args.branch_width > args.mutation_emit_mutations:
        raise ValueError("--branch-width cannot exceed --mutation-emit-mutations")
    if args.refute_limit < 0:
        raise ValueError("--refute-limit must be non-negative")
    if args.refute_timeout_seconds < 0:
        raise ValueError("--refute-timeout-seconds must be non-negative")
    if args.max_carried_colourings < 1:
        raise ValueError("--max-carried-colourings must be positive")
    if args.mutation_min_neighbor_count < 1:
        raise ValueError("--mutation-min-neighbor-count must be positive")
    if args.mutation_edge_gain_pool_points < 0:
        raise ValueError("--mutation-edge-gain-pool-points must be non-negative")
    if args.mutation_edge_gain_max_combinations < 1:
        raise ValueError("--mutation-edge-gain-max-combinations must be positive")
    if args.mutation_edge_gain_combination_offset < 0:
        raise ValueError("--mutation-edge-gain-combination-offset must be non-negative")
    if args.mutation_edge_gain_combination_stride < 1:
        raise ValueError("--mutation-edge-gain-combination-stride must be positive")
    if args.mutation_edge_gain_emit_mutations < 0:
        raise ValueError("--mutation-edge-gain-emit-mutations must be non-negative")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--satfanout-json", type=Path)
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--colourings-file", type=Path)
    parser.add_argument("--candidate-prefix", default="cgbeam")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-width", type=int, default=2)
    parser.add_argument("--mutation-max-den", type=int, default=5)
    parser.add_argument("--mutation-max-candidates", type=int, default=20_000)
    parser.add_argument("--mutation-top-points", type=int, default=50)
    parser.add_argument("--mutation-emit-mutations", type=int, default=4)
    parser.add_argument("--mutation-add-points", type=int, default=4)
    parser.add_argument("--mutation-min-neighbor-count", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-pool-points", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-max-combinations", type=int, default=250_000)
    parser.add_argument("--mutation-edge-gain-combination-offset", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-combination-stride", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-emit-mutations", type=int, default=1)
    parser.add_argument("--select-mutation-index", type=int, default=0)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--dsatur-node-limit", type=int, default=100_000)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    parser.add_argument("--run-refute-ready", action="store_true")
    parser.add_argument("--refute-limit", type=int, default=1)
    parser.add_argument("--refute-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-carried-colourings", type=int, default=8)
    args = parser.parse_args()

    try:
        validate_args(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        beam = [
            Parent(
                parent_id="seed",
                coords_csv=args.coords_csv,
                colourings_file=args.colourings_file,
                satfanout_json=args.satfanout_json,
                lineage=("seed",),
            )
        ]
        generations: list[dict[str, Any]] = []
        stopped_reason = "generation_budget_exhausted"

        for generation in range(args.generations):
            generation_dir = args.out_dir / f"generation_{generation:02d}"
            generation_dir.mkdir(parents=True, exist_ok=True)
            active_parent_count = len(beam)
            evaluated: list[dict[str, Any]] = []
            next_candidates: list[tuple[int, dict[str, Any], Parent]] = []
            for parent_rank, parent in enumerate(beam):
                parent_dir = generation_dir / f"parent_{parent_rank:03d}"
                mutation_fields, mutation, mutation_stdout = run_mutation(
                    generation_dir=parent_dir,
                    generation=generation,
                    candidate_prefix=f"{args.candidate_prefix}_p{parent_rank:03d}",
                    satfanout_json=parent.satfanout_json,
                    coords_csv=parent.coords_csv,
                    colourings_file=parent.colourings_file,
                    max_den=args.mutation_max_den,
                    max_candidates=args.mutation_max_candidates,
                    top_points=args.mutation_top_points,
                    emit_mutations=args.mutation_emit_mutations,
                    add_points=args.mutation_add_points,
                    min_neighbor_count=args.mutation_min_neighbor_count,
                    edge_gain_pool_points=args.mutation_edge_gain_pool_points,
                    edge_gain_max_combinations=args.mutation_edge_gain_max_combinations,
                    edge_gain_combination_offset=(
                        args.mutation_edge_gain_combination_offset
                    ),
                    edge_gain_combination_stride=(
                        args.mutation_edge_gain_combination_stride
                    ),
                    edge_gain_emit_mutations=args.mutation_edge_gain_emit_mutations,
                )
                require_no_claims(mutation, "mutation")
                mutations = mutation.get("mutations")
                if not isinstance(mutations, list) or not mutations:
                    raise RuntimeError("mutation emitted no mutation candidates")
                for selected_index in range(min(args.branch_width, len(mutations))):
                    row, next_parent = evaluate_child(
                        generation_dir=generation_dir,
                        generation=generation,
                        parent=parent,
                        parent_rank=parent_rank,
                        mutation=mutation,
                        selected_index=selected_index,
                        args=args,
                    )
                    row["mutation_stdout"] = str(mutation_stdout)
                    row["mutation_json"] = mutation_fields["mutation_json"]
                    row["mutation_json_sha256"] = sha256_file(
                        Path(mutation_fields["mutation_json"])
                    )
                    row["mutation_count"] = int(mutation["mutation_count"])
                    evaluated.append(row)
                    if next_parent is not None:
                        next_candidates.append((int(row["beam_score"]), row, next_parent))

            if not evaluated:
                stopped_reason = "beam_evaluated_zero_children"
                break
            evaluated.sort(key=lambda row: (-int(row["beam_score"]), str(row["candidate_id"])))
            next_candidates.sort(key=lambda item: (-item[0], str(item[1]["candidate_id"])))
            beam = [parent for _score, _row, parent in next_candidates[: args.beam_width]]
            generations.append(
                {
                    "generation": generation,
                    "input_parent_count": active_parent_count,
                    "evaluated_child_count": len(evaluated),
                    "next_beam_count": len(beam),
                    "best_child_score": int(evaluated[0]["beam_score"]),
                    "best_child_candidate_id": evaluated[0]["candidate_id"],
                    "children": evaluated,
                }
            )
            if not beam and generation + 1 < args.generations:
                stopped_reason = "no_coloured_children_for_next_beam"
                break

        if not generations:
            raise RuntimeError("beam evolution executed zero generations")
        all_children = [child for gen in generations for child in gen["children"]]
        best_child = max(
            all_children,
            key=lambda row: (
                int(row["beam_score"]),
                int(row["selected_new_m"]),
                str(row["candidate_id"]),
            ),
        )
        manifest = {
            "schema": "chi6_colour_guided_beam.v1",
            "requested_generations": args.generations,
            "completed_generation_count": len(generations),
            "beam_width": args.beam_width,
            "branch_width": args.branch_width,
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
            "stopped_reason": stopped_reason,
            "evaluated_child_count": len(all_children),
            "coloured_child_count": sum(
                1 for row in all_children if row["next_colourings_file"] != "NONE"
            ),
            "refute_attempt_child_count": sum(
                1
                for row in all_children
                if row["refute_attempt"]["status"] == "REFUTE_ATTEMPT_RECORDED"
            ),
            "refute_success_count": sum(
                int(row["refute_attempt"]["refute_success_count"]) for row in all_children
            ),
            "refute_failed_count": sum(
                int(row["refute_attempt"]["refute_failed_count"]) for row in all_children
            ),
            "best_child_candidate_id": best_child["candidate_id"],
            "best_child_score": int(best_child["beam_score"]),
            "best_child_new_n": int(best_child["selected_new_n"]),
            "best_child_new_m": int(best_child["selected_new_m"]),
            "best_child_dsatur_status": best_child["dsatur_status"],
            "best_child_recommended_next_action": best_child["recommended_next_action"],
            "total_selected_killed_colouring_count_by_unit_neighborhood": sum(
                int(row["selected_killed_colouring_count_by_unit_neighborhood"])
                for row in all_children
            ),
            "claim_scope": "colour_guided_frontier_beam_search_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_exact_geometry_plus_checked_sat_lrat_after_beam",
            "generations": generations,
        }
        manifest_path = args.out_dir / "colour_guided_beam.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2

    print("chi6_colour_guided_beam v1")
    print(f"beam_json={manifest_path}")
    print(f"beam_json_sha256={sha256_file(manifest_path)}")
    print(f"requested_generations={manifest['requested_generations']}")
    print(f"completed_generation_count={manifest['completed_generation_count']}")
    print(f"beam_width={manifest['beam_width']}")
    print(f"branch_width={manifest['branch_width']}")
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
    print(f"stopped_reason={manifest['stopped_reason']}")
    print(f"evaluated_child_count={manifest['evaluated_child_count']}")
    print(f"coloured_child_count={manifest['coloured_child_count']}")
    print(f"refute_attempt_child_count={manifest['refute_attempt_child_count']}")
    print(f"refute_success_count={manifest['refute_success_count']}")
    print(f"refute_failed_count={manifest['refute_failed_count']}")
    print(f"best_child_candidate_id={manifest['best_child_candidate_id']}")
    print(f"best_child_score={manifest['best_child_score']}")
    print(f"best_child_new_n={manifest['best_child_new_n']}")
    print(f"best_child_new_m={manifest['best_child_new_m']}")
    print(f"best_child_dsatur_status={manifest['best_child_dsatur_status']}")
    print(f"best_child_recommended_next_action={manifest['best_child_recommended_next_action']}")
    print(
        "total_selected_killed_colouring_count_by_unit_neighborhood="
        f"{manifest['total_selected_killed_colouring_count_by_unit_neighborhood']}"
    )
    print("claim_scope=colour_guided_frontier_beam_search_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOUR_GUIDED_BEAM_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
