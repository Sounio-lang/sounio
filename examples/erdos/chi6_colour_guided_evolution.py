#!/usr/bin/env python3
"""Iterate SAT-colouring-guided exact-rational frontier mutations.

This is a search/evolution ledger, not a chromatic-number proof.  Each
generation consumes one observed 5-colouring of the current exact-rational
unit-distance graph, proposes rational unit-neighbour mutations, scouts the
selected mutation, runs the deterministic campaign preflight, and feeds the next
DSATUR colouring back into the next generation when one is available.

The loop stops when the selected mutated frontier no longer has a bounded
DSATUR colouring to feed back.  That is a refuter/search signal only; promotion
still requires exact geometry plus checked SAT/LRAT/Lean artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MUTATOR = Path(__file__).with_name("chi6_colour_guided_mutation.py")
SCOUT = Path(__file__).with_name("chi6_rational_frontier_scout.py")
PREFLIGHT = Path(__file__).with_name("chi6_frontier_campaign_preflight.py")


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
        if not all(
            c == "_" or ("0" <= c <= "9") or ("A" <= c <= "Z") or ("a" <= c <= "z")
            for c in key
        ):
            continue
        if key in fields:
            raise ValueError(f"duplicate key in tool output: {key}")
        fields[key] = value
    return fields


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)


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
        raise RuntimeError(
            f"command failed with exit {proc.returncode}: {' '.join(cmd)}: {detail}"
        )
    return proc.stdout


def require_no_claims(meta: dict[str, Any], label: str) -> None:
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if key in meta and meta[key] != "none":
            raise ValueError(f"{label} leaked {key}={meta[key]}")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} leaked promotable={meta.get('promotable')}")


def colouring_line(colouring: Any, n: int, label: str) -> str:
    if not isinstance(colouring, list):
        raise ValueError(f"{label}: colouring must be a list")
    if len(colouring) != n:
        raise ValueError(f"{label}: colouring length {len(colouring)} does not match n={n}")
    out: list[str] = []
    for v, raw in enumerate(colouring):
        if not isinstance(raw, int) or raw < 0 or raw >= 5:
            raise ValueError(f"{label}: bad colour at vertex {v}: {raw!r}")
        out.append(f"{v}:{raw}")
    return ",".join(out)


def write_colourings_file(path: Path, colouring: list[int], n: int, label: str) -> str:
    line = colouring_line(colouring, n, label)
    path.write_text(line + "\n", encoding="ascii")
    return line


def run_mutation(
    *,
    generation_dir: Path,
    generation: int,
    candidate_prefix: str,
    satfanout_json: Path | None,
    coords_csv: Path | None,
    colourings_file: Path | None,
    max_den: int,
    max_candidates: int,
    top_points: int,
    emit_mutations: int,
    add_points: int,
    min_neighbor_count: int,
    edge_gain_pool_points: int,
    edge_gain_max_combinations: int,
    edge_gain_combination_offset: int,
    edge_gain_combination_stride: int,
    edge_gain_emit_mutations: int,
) -> tuple[dict[str, str], dict[str, Any], Path]:
    mutation_dir = generation_dir / "mutation"
    stdout_path = generation_dir / "mutation.out"
    cmd = [
        sys.executable,
        str(MUTATOR),
        str(mutation_dir),
        "--candidate-prefix",
        f"{candidate_prefix}_g{generation:02d}",
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
        "--min-neighbor-count",
        str(min_neighbor_count),
        "--edge-gain-pool-points",
        str(edge_gain_pool_points),
        "--edge-gain-max-combinations",
        str(edge_gain_max_combinations),
        "--edge-gain-combination-offset",
        str(edge_gain_combination_offset),
        "--edge-gain-combination-stride",
        str(edge_gain_combination_stride),
        "--edge-gain-emit-mutations",
        str(edge_gain_emit_mutations),
    ]
    if satfanout_json is not None:
        cmd.extend(["--satfanout-json", str(satfanout_json)])
    else:
        if coords_csv is None or colourings_file is None:
            raise ValueError("explicit evolution input requires coords_csv and colourings_file")
        cmd.extend(["--coords-csv", str(coords_csv), "--colourings-file", str(colourings_file)])
    stdout = run_command(cmd, stdout_path)
    fields = parse_kv_output(stdout)
    mutation_json = Path(fields.get("mutation_json", ""))
    if not mutation_json.is_file():
        raise RuntimeError(f"mutation did not emit mutation_json: {mutation_json}")
    mutation = load_json(mutation_json)
    if mutation.get("schema") != "chi6_colour_guided_mutation.v1":
        raise RuntimeError("mutation schema mismatch")
    require_no_claims(mutation, "mutation")
    return fields, mutation, stdout_path


def run_scout(
    *,
    generation_dir: Path,
    candidate_id: str,
    coords_csv: Path,
    max_vertices: int,
    min_vertices: int,
    min_edges: int,
    split_depth: int,
    min_split_degree: int,
    dsatur_node_limit: int,
) -> tuple[dict[str, str], dict[str, Any], Path]:
    scout_dir = generation_dir / "scout"
    stdout_path = generation_dir / "scout.out"
    stdout = run_command(
        [
            sys.executable,
            str(SCOUT),
            "--coords-csv",
            str(coords_csv),
            "--candidate-id",
            candidate_id,
            "--out-dir",
            str(scout_dir),
            "--max-vertices",
            str(max_vertices),
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
        ],
        stdout_path,
    )
    fields = parse_kv_output(stdout)
    scout_json = Path(fields.get("frontier_scout", ""))
    if not scout_json.is_file():
        raise RuntimeError(f"scout did not emit frontier_scout: {scout_json}")
    scout = load_json(scout_json)
    if scout.get("schema") != "chi6_rational_frontier_scout.v1":
        raise RuntimeError("scout schema mismatch")
    require_no_claims(scout, "scout")
    return fields, scout, stdout_path


def run_preflight(
    *,
    generation_dir: Path,
    scout_json: Path,
    max_cubes: int,
    sample_hard_cubes: int,
) -> tuple[dict[str, str], dict[str, Any], Path]:
    preflight_dir = generation_dir / "preflight"
    stdout_path = generation_dir / "preflight.out"
    stdout = run_command(
        [
            sys.executable,
            str(PREFLIGHT),
            str(scout_json),
            str(preflight_dir),
            "--max-cubes",
            str(max_cubes),
            "--sample-hard-cubes",
            str(sample_hard_cubes),
        ],
        stdout_path,
    )
    fields = parse_kv_output(stdout)
    preflight_json = Path(fields.get("campaign_preflight_json", ""))
    if not preflight_json.is_file():
        raise RuntimeError(f"preflight did not emit campaign_preflight_json: {preflight_json}")
    preflight = load_json(preflight_json)
    if preflight.get("schema") != "chi6_frontier_campaign_preflight.v1":
        raise RuntimeError("preflight schema mismatch")
    require_no_claims(preflight, "preflight")
    return fields, preflight, stdout_path


def validate_args(args: argparse.Namespace) -> None:
    for tool in (MUTATOR, SCOUT, PREFLIGHT):
        if not tool.is_file():
            raise ValueError(f"missing tool: {tool}")
    if args.generations < 1:
        raise ValueError("--generations must be positive")
    if args.mutation_max_den < 1:
        raise ValueError("--mutation-max-den must be positive")
    if args.mutation_max_candidates < 1:
        raise ValueError("--mutation-max-candidates must be positive")
    if args.mutation_top_points < 1:
        raise ValueError("--mutation-top-points must be positive")
    if args.mutation_emit_mutations < 1:
        raise ValueError("--mutation-emit-mutations must be positive")
    if args.mutation_add_points < 1:
        raise ValueError("--mutation-add-points must be positive")
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
    if args.select_mutation_index < 0:
        raise ValueError("--select-mutation-index must be non-negative")
    if args.max_vertices < 2:
        raise ValueError("--max-vertices must be at least 2")
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
    if args.max_cubes < 1:
        raise ValueError("--max-cubes must be positive")
    if args.sample_hard_cubes < 0:
        raise ValueError("--sample-hard-cubes must be non-negative")
    if args.sample_hard_cubes > args.max_cubes:
        raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")
    has_satfanout = args.satfanout_json is not None
    has_explicit = args.coords_csv is not None or args.colourings_file is not None
    if has_satfanout and has_explicit:
        raise ValueError("pass either --satfanout-json or explicit coords/colourings, not both")
    if not has_satfanout and (args.coords_csv is None or args.colourings_file is None):
        raise ValueError("pass --satfanout-json or both --coords-csv and --colourings-file")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("out_dir already exists and is non-empty")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--satfanout-json", type=Path)
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--colourings-file", type=Path)
    parser.add_argument("--candidate-prefix", default="cgevolve")
    parser.add_argument("--generations", type=int, default=2)
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
    args = parser.parse_args()

    try:
        validate_args(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        current_coords = args.coords_csv
        current_colourings = args.colourings_file
        first_satfanout = args.satfanout_json
        generations: list[dict[str, Any]] = []
        stopped_reason = "generation_budget_exhausted"

        for generation in range(args.generations):
            generation_dir = args.out_dir / f"generation_{generation:02d}"
            generation_dir.mkdir(parents=True, exist_ok=True)
            mutation_fields, mutation, mutation_stdout = run_mutation(
                generation_dir=generation_dir,
                generation=generation,
                candidate_prefix=args.candidate_prefix,
                satfanout_json=first_satfanout if generation == 0 else None,
                coords_csv=current_coords,
                colourings_file=current_colourings,
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
            first_satfanout = None
            mutations = mutation.get("mutations")
            if not isinstance(mutations, list) or not mutations:
                raise RuntimeError("mutation emitted no mutation candidates")
            if args.select_mutation_index >= len(mutations):
                raise RuntimeError(
                    f"selected mutation index {args.select_mutation_index} "
                    f"but only {len(mutations)} mutations were emitted"
                )
            selected = mutations[args.select_mutation_index]
            selected_coords = Path(str(selected["coords_csv"]))
            if not selected_coords.is_file():
                raise RuntimeError(f"selected mutation coords_csv missing: {selected_coords}")
            candidate_id = str(
                selected.get("candidate_id_hint")
                or f"{args.candidate_prefix}_g{generation:02d}_mut{args.select_mutation_index:03d}"
            )

            scout_fields, scout, scout_stdout = run_scout(
                generation_dir=generation_dir,
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
                generation_dir=generation_dir,
                scout_json=scout_json,
                max_cubes=args.max_cubes,
                sample_hard_cubes=args.sample_hard_cubes,
            )
            next_colourings_file = "NONE"
            next_colouring_vertex_count = 0
            if isinstance(scout.get("colouring"), list):
                next_colourings_path = generation_dir / "next-colourings.txt"
                write_colourings_file(
                    next_colourings_path,
                    scout["colouring"],
                    int(scout["n"]),
                    f"generation_{generation:02d}",
                )
                next_colourings_file = str(next_colourings_path)
                next_colouring_vertex_count = int(scout["n"])
                current_coords = selected_coords
                current_colourings = next_colourings_path
            else:
                stopped_reason = "no_bounded_dsatur_colouring_for_selected_mutation"

            row = {
                "generation": generation,
                "input_mode": mutation["input_mode"],
                "mutation_stdout": str(mutation_stdout),
                "mutation_json": mutation_fields["mutation_json"],
                "mutation_json_sha256": sha256_file(Path(mutation_fields["mutation_json"])),
                "mutation_count": int(mutation["mutation_count"]),
                "candidate_point_count_scored": int(mutation["candidate_point_count_scored"]),
                "mutation_min_neighbor_count": int(mutation["min_neighbor_count"]),
                "mutation_edge_gain_pool_points": int(mutation["edge_gain_pool_points"]),
                "mutation_edge_gain_emit_mutations": int(
                    mutation["edge_gain_emit_mutations"]
                ),
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
                "selected_mutation_index": args.select_mutation_index,
                "selected_mutation_strategy": selected["selection_strategy"],
                "selected_coords_csv": str(selected_coords),
                "selected_coords_sha256": sha256_file(selected_coords),
                "selected_old_n": int(selected["old_n"]),
                "selected_new_n": int(selected["new_n"]),
                "selected_new_m": int(selected["new_m"]),
                "selected_edge_gain_after_mutation": int(
                    selected["edge_gain_after_mutation"]
                ),
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
                "scout_stdout": str(scout_stdout),
                "frontier_scout": scout_fields["frontier_scout"],
                "frontier_scout_sha256": sha256_file(scout_json),
                "dsatur_status": scout["dsatur_status"],
                "dsatur_nodes": int(scout["dsatur_nodes"]),
                "preflight_stdout": str(preflight_stdout),
                "campaign_preflight_json": preflight_fields["campaign_preflight_json"],
                "campaign_preflight_sha256": sha256_file(
                    Path(preflight_fields["campaign_preflight_json"])
                ),
                "recommended_next_action": preflight["recommended_next_action"],
                "cube_count": int(preflight["cube_count"]),
                "propagation_conflict_count": int(preflight["propagation_conflict_count"]),
                "propagation_hard_count": int(preflight["propagation_hard_count"]),
                "next_colourings_file": next_colourings_file,
                "next_colouring_vertex_count": next_colouring_vertex_count,
                "claim_scope": "colour_guided_evolution_generation_only",
                "sat_claim": "none",
                "chromatic_claim": "none",
                "global_unsat_claim": "none",
                "verified_claim": "none",
                "promotable": 0,
            }
            generations.append(row)
            if next_colourings_file == "NONE":
                break

        if not generations:
            raise RuntimeError("evolution executed zero generations")
        last = generations[-1]
        manifest = {
            "schema": "chi6_colour_guided_evolution.v1",
            "requested_generations": args.generations,
            "completed_generation_count": len(generations),
            "stopped_reason": stopped_reason,
            "candidate_prefix": args.candidate_prefix,
            "last_selected_coords_csv": last["selected_coords_csv"],
            "last_selected_new_n": last["selected_new_n"],
            "last_selected_new_m": last["selected_new_m"],
            "last_dsatur_status": last["dsatur_status"],
            "last_recommended_next_action": last["recommended_next_action"],
            "colouring_feedback_count": sum(
                1 for row in generations if row["next_colourings_file"] != "NONE"
            ),
            "total_single_point_full_blocker_count": sum(
                int(row["single_point_full_blocker_count"]) for row in generations
            ),
            "total_selected_killed_colouring_count_by_unit_neighborhood": sum(
                int(row["selected_killed_colouring_count_by_unit_neighborhood"])
                for row in generations
            ),
            "claim_scope": "colour_guided_frontier_evolution_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_exact_geometry_plus_checked_sat_lrat_after_evolution",
            "generations": generations,
        }
        manifest_path = args.out_dir / "colour_guided_evolution.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_colour_guided_evolution v1")
    print(f"evolution_json={manifest_path}")
    print(f"evolution_json_sha256={sha256_file(manifest_path)}")
    print(f"requested_generations={manifest['requested_generations']}")
    print(f"completed_generation_count={manifest['completed_generation_count']}")
    print(f"stopped_reason={manifest['stopped_reason']}")
    print(f"last_selected_new_n={manifest['last_selected_new_n']}")
    print(f"last_selected_new_m={manifest['last_selected_new_m']}")
    print(f"last_dsatur_status={manifest['last_dsatur_status']}")
    print(f"last_recommended_next_action={manifest['last_recommended_next_action']}")
    print(f"colouring_feedback_count={manifest['colouring_feedback_count']}")
    print(
        "total_selected_killed_colouring_count_by_unit_neighborhood="
        f"{manifest['total_selected_killed_colouring_count_by_unit_neighborhood']}"
    )
    print("claim_scope=colour_guided_frontier_evolution_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOUR_GUIDED_EVOLUTION_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
