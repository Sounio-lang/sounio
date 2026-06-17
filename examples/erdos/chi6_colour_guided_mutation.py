#!/usr/bin/env python3
"""Use SAT colourings to propose exact-rational frontier mutations.

This is search steering, not a chromatic-number proof.  Given a sat-fanout
ledger (or an explicit coordinate CSV), it looks for new rational points at unit
distance from existing vertices.  A candidate point is interesting when, under
observed SAT colourings, its unit-neighbour colours cover many of the five
colours; if all five are covered, that single point blocks that observed
colouring.  The emitted mutated coordinate CSVs remain non-promotable inputs for
the ordinary exact-geometry/SAT/LRAT/Lean pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from chi6_rational_frontier_scout import rational_unit_directions, write_coord_csv
from make_chi6_rational_unit_graph_source_package import (
    parse_coord_table,
    sha256_file,
    unit_edges,
)


K = 5
COLOURING_RE = re.compile(r"^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$")


@dataclass(frozen=True)
class Colouring:
    label: str
    colours: tuple[int, ...]


@dataclass(frozen=True)
class PointScore:
    point: tuple[Fraction, Fraction]
    neighbor_vertices: tuple[int, ...]
    killed: tuple[str, ...]
    covered_colour_total: int
    max_covered_colours: int

    def score_key(self) -> tuple[int, int, int, int, tuple[Fraction, Fraction, Fraction]]:
        x, y = self.point
        return (
            len(self.killed),
            self.covered_colour_total,
            self.max_covered_colours,
            len(self.neighbor_vertices),
            (-(x * x + y * y), -x, -y),
        )


def point_key(p: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    x, y = p
    return (x * x + y * y, x, y)


def fraction_text(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def parse_colouring(raw: str, n: int, label: str) -> Colouring:
    if raw == "NONE" or not COLOURING_RE.fullmatch(raw):
        raise ValueError(f"{label}: bad SAT colouring payload")
    colours = [-1] * n
    for token in raw.split(","):
        v_raw, c_raw = token.split(":", 1)
        v = int(v_raw)
        c = int(c_raw)
        if not (0 <= v < n):
            raise ValueError(f"{label}: colouring vertex out of range: {v}")
        if not (0 <= c < K):
            raise ValueError(f"{label}: colouring colour out of range: {c}")
        if colours[v] >= 0:
            raise ValueError(f"{label}: duplicate vertex in colouring: {v}")
        colours[v] = c
    missing = [str(i) for i, c in enumerate(colours) if c < 0]
    if missing:
        raise ValueError(f"{label}: missing coloured vertices: {','.join(missing[:10])}")
    return Colouring(label, tuple(colours))


def validate_colouring_respects_edges(colouring: Colouring, edges: list[tuple[int, int]]) -> None:
    for u, v in edges:
        if colouring.colours[u] == colouring.colours[v]:
            raise ValueError(
                f"{colouring.label}: observed SAT colouring violates existing edge {u}-{v}"
            )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)


def load_from_satfanout(path: Path) -> tuple[Path, str, list[str]]:
    meta = load_json(path)
    if meta.get("sat_claim") != "none" or meta.get("chromatic_claim") != "none":
        raise ValueError("satfanout input must carry no SAT/chromatic claim")
    if meta.get("promotable") != 0:
        raise ValueError("satfanout input must be non-promotable")
    preflight_raw = meta.get("source_preflight_batch") or meta.get("preflight_batch_json")
    preflight_path = Path(str(preflight_raw or ""))
    if not preflight_path.is_file():
        raise ValueError(f"satfanout source preflight batch not found: {preflight_path}")
    preflight = load_json(preflight_path)
    rows = preflight.get("preflights")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source preflight batch has no preflight rows")
    row = rows[0]
    scout_path = Path(str(row.get("frontier_scout", "")))
    if not scout_path.is_file():
        raise ValueError(f"frontier_scout not found: {scout_path}")
    scout = load_json(scout_path)
    coords_csv = Path(str(scout.get("coords_csv", "")))
    if not coords_csv.is_file():
        raise ValueError(f"frontier coords_csv not found: {coords_csv}")
    records = meta.get("records")
    status_key = "status"
    if records is None:
        records = meta.get("attempts")
        status_key = "classified_status"
    if not isinstance(records, list) or not records:
        raise ValueError("satfanout has no records/attempts")
    colourings: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        raw = rec.get("sat_colouring")
        status = rec.get(status_key)
        if isinstance(raw, str) and raw != "NONE" and status == "REFUTE_SAT_MUTATE_FRONTIER":
            colourings.append(raw)
    if not colourings:
        raise ValueError("satfanout has no REFUTE_SAT_MUTATE_FRONTIER colourings")
    candidate_id = str(
        meta.get("candidate_id")
        or meta.get("source_candidate_id")
        or row.get("candidate_id")
        or "colour_guided"
    )
    return coords_csv, candidate_id, colourings


def load_colourings_file(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    if not out:
        raise ValueError(f"{path}: no colourings found")
    return out


def candidate_points(
    coords: list[tuple[Fraction, Fraction]],
    *,
    max_den: int,
    max_candidates: int,
) -> list[tuple[Fraction, Fraction]]:
    directions = rational_unit_directions(max_den)
    existing = set(coords)
    proposed: set[tuple[Fraction, Fraction]] = set()
    for x, y in coords:
        for dx, dy in directions:
            p = (x + dx, y + dy)
            if p not in existing:
                proposed.add(p)
    return sorted(proposed, key=point_key)[:max_candidates]


def score_point(
    point: tuple[Fraction, Fraction],
    coords: list[tuple[Fraction, Fraction]],
    colourings: list[Colouring],
) -> PointScore | None:
    tmp_coords = coords + [point]
    edges = unit_edges(tmp_coords)
    new_vertex = len(coords)
    neighbors = sorted(u if v == new_vertex else v for u, v in edges if new_vertex in (u, v))
    if not neighbors:
        return None
    killed: list[str] = []
    covered_total = 0
    max_covered = 0
    for colouring in colourings:
        covered = {colouring.colours[v] for v in neighbors}
        covered_total += len(covered)
        max_covered = max(max_covered, len(covered))
        if len(covered) >= K:
            killed.append(colouring.label)
    return PointScore(point, tuple(neighbors), tuple(killed), covered_total, max_covered)


def score_candidates(
    coords: list[tuple[Fraction, Fraction]],
    colourings: list[Colouring],
    *,
    max_den: int,
    max_candidates: int,
    min_neighbor_count: int,
) -> list[PointScore]:
    scored = [
        score
        for p in candidate_points(coords, max_den=max_den, max_candidates=max_candidates)
        if (score := score_point(p, coords, colourings)) is not None
        and len(score.neighbor_vertices) >= min_neighbor_count
    ]
    scored.sort(key=lambda s: s.score_key(), reverse=True)
    return scored


def greedy_mutation(
    coords: list[tuple[Fraction, Fraction]],
    scored: list[PointScore],
    *,
    add_points: int,
) -> list[PointScore]:
    chosen: list[PointScore] = []
    used_points: set[tuple[Fraction, Fraction]] = set()
    killed: set[str] = set()
    for _ in range(add_points):
        best: PointScore | None = None
        best_key: tuple[
            int,
            int,
            int,
            int,
            int,
            tuple[Fraction, Fraction, Fraction],
        ] | None = None
        for score in scored:
            if score.point in used_points:
                continue
            incremental = len(set(score.killed) - killed)
            key = (
                incremental,
                len(score.killed),
                score.covered_colour_total,
                score.max_covered_colours,
                len(score.neighbor_vertices),
                score.score_key()[4],
            )
            if best_key is None or key > best_key:
                best = score
                best_key = key
        if best is None:
            break
        chosen.append(best)
        used_points.add(best.point)
        killed.update(best.killed)
    return chosen


def points_unit_distance(
    a: tuple[Fraction, Fraction],
    b: tuple[Fraction, Fraction],
) -> bool:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy == 1


def edge_gain_combo_key(combo: tuple[PointScore, ...]) -> tuple[int, int, int, int, int]:
    existing_edge_gain = sum(len(score.neighbor_vertices) for score in combo)
    new_new_edge_gain = sum(
        1
        for a, b in itertools.combinations(combo, 2)
        if points_unit_distance(a.point, b.point)
    )
    killed = {label for score in combo for label in score.killed}
    return (
        existing_edge_gain + new_new_edge_gain,
        new_new_edge_gain,
        len(killed),
        sum(score.covered_colour_total for score in combo),
        max((score.max_covered_colours for score in combo), default=0),
    )


def point_set_key(scores: list[PointScore] | tuple[PointScore, ...]) -> tuple[tuple[Fraction, Fraction], ...]:
    return tuple(sorted((score.point for score in scores), key=point_key))


def edge_gain_mutations(
    scored: list[PointScore],
    *,
    add_points: int,
    pool_points: int,
    max_combinations: int,
    combination_offset: int,
    combination_stride: int,
    emit_count: int,
    skip_point_sets: set[tuple[tuple[Fraction, Fraction], ...]],
) -> tuple[list[list[PointScore]], int, bool]:
    pool = scored[:pool_points]
    if not pool or emit_count <= 0:
        return [], 0, False
    size = min(add_points, len(pool))
    top: list[
        tuple[
            tuple[
                int,
                int,
                int,
                int,
                int,
                tuple[tuple[int, int, int, int, tuple[Fraction, Fraction, Fraction]], ...],
            ],
            tuple[PointScore, ...],
        ]
    ] = []
    considered = 0
    truncated = False
    for combo_index, combo in enumerate(itertools.combinations(pool, size)):
        if combo_index < combination_offset:
            continue
        if (combo_index - combination_offset) % combination_stride != 0:
            continue
        if considered >= max_combinations:
            truncated = True
            break
        considered += 1
        if point_set_key(combo) in skip_point_sets:
            continue
        combo_score_keys = tuple(score.score_key() for score in combo)
        key = edge_gain_combo_key(combo) + (combo_score_keys,)
        if len(top) < emit_count:
            top.append((key, combo))
            continue
        min_index, min_item = min(enumerate(top), key=lambda item: item[1][0])
        if key > min_item[0]:
            top[min_index] = (key, combo)
    top.sort(key=lambda item: item[0], reverse=True)
    return [list(combo) for _key, combo in top], considered, truncated


def write_mutation(
    *,
    out_dir: Path,
    index: int,
    base_coords: list[tuple[Fraction, Fraction]],
    chosen: list[PointScore],
    candidate_prefix: str,
    selection_strategy: str,
) -> dict[str, Any]:
    mutation_dir = out_dir / f"mutation_{index:03d}"
    mutation_dir.mkdir(parents=True, exist_ok=True)
    coords = base_coords + [score.point for score in chosen]
    coords_csv = mutation_dir / f"{candidate_prefix}_mutation_{index:03d}.coords.csv"
    write_coord_csv(coords_csv, coords)
    base_edges = unit_edges(base_coords)
    edges = unit_edges(coords)
    killed = sorted({label for score in chosen for label in score.killed})
    neighbor_counts = [len(score.neighbor_vertices) for score in chosen]
    payload = {
        "index": index,
        "selection_strategy": selection_strategy,
        "coords_csv": str(coords_csv),
        "coords_sha256": sha256_file(coords_csv),
        "old_n": len(base_coords),
        "old_m": len(base_edges),
        "new_n": len(coords),
        "new_m": len(edges),
        "edge_gain_after_mutation": len(edges) - len(base_edges),
        "added_point_count": len(chosen),
        "selected_existing_neighbor_count_total": sum(neighbor_counts),
        "selected_existing_neighbor_count_min": min(neighbor_counts) if neighbor_counts else 0,
        "selected_existing_neighbor_count_max": max(neighbor_counts) if neighbor_counts else 0,
        "killed_colouring_count_by_unit_neighborhood": len(killed),
        "killed_colouring_labels": killed,
        "candidate_id_hint": f"{candidate_prefix}_mut{index:03d}",
        "points": [
            {
                "x": fraction_text(score.point[0]),
                "y": fraction_text(score.point[1]),
                "neighbor_vertices": list(score.neighbor_vertices),
                "neighbor_count": len(score.neighbor_vertices),
                "killed_colouring_count": len(score.killed),
                "killed_colouring_labels": list(score.killed),
                "covered_colour_total": score.covered_colour_total,
                "max_covered_colours": score.max_covered_colours,
            }
            for score in chosen
        ],
    }
    (mutation_dir / "mutation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--satfanout-json", type=Path)
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--colourings-file", type=Path)
    parser.add_argument("--candidate-prefix")
    parser.add_argument("--max-den", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--top-points", type=int, default=50)
    parser.add_argument("--emit-mutations", type=int, default=4)
    parser.add_argument("--add-points", type=int, default=4)
    parser.add_argument("--min-neighbor-count", type=int, default=1)
    parser.add_argument("--edge-gain-pool-points", type=int, default=0)
    parser.add_argument("--edge-gain-max-combinations", type=int, default=250000)
    parser.add_argument("--edge-gain-combination-offset", type=int, default=0)
    parser.add_argument("--edge-gain-combination-stride", type=int, default=1)
    parser.add_argument("--edge-gain-emit-mutations", type=int, default=1)
    args = parser.parse_args()

    try:
        if args.max_den < 1:
            raise ValueError("--max-den must be positive")
        if args.max_candidates < 1:
            raise ValueError("--max-candidates must be positive")
        if args.top_points < 1:
            raise ValueError("--top-points must be positive")
        if args.emit_mutations < 1:
            raise ValueError("--emit-mutations must be positive")
        if args.add_points < 1:
            raise ValueError("--add-points must be positive")
        if args.min_neighbor_count < 1:
            raise ValueError("--min-neighbor-count must be positive")
        if args.edge_gain_pool_points < 0:
            raise ValueError("--edge-gain-pool-points must be non-negative")
        if args.edge_gain_max_combinations < 1:
            raise ValueError("--edge-gain-max-combinations must be positive")
        if args.edge_gain_combination_offset < 0:
            raise ValueError("--edge-gain-combination-offset must be non-negative")
        if args.edge_gain_combination_stride < 1:
            raise ValueError("--edge-gain-combination-stride must be positive")
        if args.edge_gain_emit_mutations < 0:
            raise ValueError("--edge-gain-emit-mutations must be non-negative")
        if args.satfanout_json is None and (
            args.coords_csv is None or args.colourings_file is None
        ):
            raise ValueError("pass --satfanout-json or both --coords-csv and --colourings-file")
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError("out_dir already exists and is non-empty")
        args.out_dir.mkdir(parents=True, exist_ok=True)

        if args.satfanout_json is not None:
            coords_csv, source_candidate_id, raw_colourings = load_from_satfanout(
                args.satfanout_json
            )
            input_path = args.satfanout_json
            input_mode = "satfanout"
        else:
            assert args.coords_csv is not None
            assert args.colourings_file is not None
            coords_csv = args.coords_csv
            source_candidate_id = args.candidate_prefix or "colour_guided"
            raw_colourings = load_colourings_file(args.colourings_file)
            input_path = args.colourings_file
            input_mode = "explicit_colourings"
        candidate_prefix = args.candidate_prefix or f"{source_candidate_id}_cg"

        coords = parse_coord_table(coords_csv, 1000000)
        base_edges = unit_edges(coords)
        colourings = [
            parse_colouring(raw, len(coords), f"colouring_{i:03d}")
            for i, raw in enumerate(raw_colourings)
        ]
        for colouring in colourings:
            validate_colouring_respects_edges(colouring, base_edges)

        scored = score_candidates(
            coords,
            colourings,
            max_den=args.max_den,
            max_candidates=args.max_candidates,
            min_neighbor_count=args.min_neighbor_count,
        )
        if not scored:
            raise ValueError(
                "no adjacent rational unit-step mutation candidates found "
                f"with min_neighbor_count={args.min_neighbor_count}"
            )
        top = scored[: args.top_points]
        single_blockers = [score for score in scored if score.killed]

        mutations: list[dict[str, Any]] = []
        emitted_point_sets: set[tuple[tuple[Fraction, Fraction], ...]] = set()
        edge_gain_considered = 0
        edge_gain_truncated = False
        greedy = greedy_mutation(coords, scored, add_points=args.add_points)
        if greedy:
            emitted_point_sets.add(point_set_key(greedy))
            mutations.append(
                write_mutation(
                    out_dir=args.out_dir,
                    index=0,
                    base_coords=coords,
                    chosen=greedy,
                    candidate_prefix=candidate_prefix,
                    selection_strategy="colour_greedy",
                )
            )
        if args.edge_gain_pool_points > 0 and len(mutations) < args.emit_mutations:
            edge_gain_batches, edge_gain_considered, edge_gain_truncated = edge_gain_mutations(
                scored,
                add_points=args.add_points,
                pool_points=args.edge_gain_pool_points,
                max_combinations=args.edge_gain_max_combinations,
                combination_offset=args.edge_gain_combination_offset,
                combination_stride=args.edge_gain_combination_stride,
                emit_count=min(
                    args.edge_gain_emit_mutations,
                    args.emit_mutations - len(mutations),
                ),
                skip_point_sets=emitted_point_sets,
            )
            for edge_gain in edge_gain_batches:
                emitted_point_sets.add(point_set_key(edge_gain))
                mutations.append(
                    write_mutation(
                        out_dir=args.out_dir,
                        index=len(mutations),
                        base_coords=coords,
                        chosen=edge_gain,
                        candidate_prefix=candidate_prefix,
                        selection_strategy="edge_gain_batch",
                    )
                )
                if len(mutations) >= args.emit_mutations:
                    break
        for score in top:
            if len(mutations) >= args.emit_mutations:
                break
            if point_set_key([score]) in emitted_point_sets:
                continue
            emitted_point_sets.add(point_set_key([score]))
            mutations.append(
                write_mutation(
                    out_dir=args.out_dir,
                    index=len(mutations),
                    base_coords=coords,
                    chosen=[score],
                    candidate_prefix=candidate_prefix,
                    selection_strategy="single_point_rank",
                )
            )

        manifest = {
            "schema": "chi6_colour_guided_mutation.v1",
            "input_mode": input_mode,
            "input_path": str(input_path),
            "coords_csv": str(coords_csv),
            "coords_sha256": sha256_file(coords_csv),
            "source_candidate_id": source_candidate_id,
            "candidate_prefix": candidate_prefix,
            "n": len(coords),
            "m": len(base_edges),
            "k": K,
            "observed_colouring_count": len(colourings),
            "max_den": args.max_den,
            "min_neighbor_count": args.min_neighbor_count,
            "edge_gain_pool_points": args.edge_gain_pool_points,
            "edge_gain_max_combinations": args.edge_gain_max_combinations,
            "edge_gain_combination_offset": args.edge_gain_combination_offset,
            "edge_gain_combination_stride": args.edge_gain_combination_stride,
            "edge_gain_emit_mutations": args.edge_gain_emit_mutations,
            "edge_gain_considered_combination_count": edge_gain_considered,
            "edge_gain_combination_truncated": edge_gain_truncated,
            "candidate_point_count_scored": len(scored),
            "single_point_full_blocker_count": len(single_blockers),
            "top_points": [
                {
                    "rank": i,
                    "x": fraction_text(score.point[0]),
                    "y": fraction_text(score.point[1]),
                    "neighbor_vertices": list(score.neighbor_vertices),
                    "neighbor_count": len(score.neighbor_vertices),
                    "killed_colouring_count": len(score.killed),
                    "killed_colouring_labels": list(score.killed),
                    "covered_colour_total": score.covered_colour_total,
                    "max_covered_colours": score.max_covered_colours,
                }
                for i, score in enumerate(top)
            ],
            "mutation_count": len(mutations),
            "mutations": mutations,
            "claim_scope": "colour_guided_frontier_mutation_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_exact_geometry_plus_checked_sat_lrat_after_mutation",
        }
        manifest_path = args.out_dir / "colour_guided_mutation.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2

    print("chi6_colour_guided_mutation v1")
    print(f"mutation_json={manifest_path}")
    print(f"mutation_json_sha256={sha256_file(manifest_path)}")
    print(f"source_candidate_id={manifest['source_candidate_id']}")
    print(f"n={manifest['n']}")
    print(f"m={manifest['m']}")
    print(f"observed_colouring_count={manifest['observed_colouring_count']}")
    print(f"candidate_point_count_scored={manifest['candidate_point_count_scored']}")
    print(f"single_point_full_blocker_count={manifest['single_point_full_blocker_count']}")
    print(f"mutation_count={manifest['mutation_count']}")
    if mutations:
        print(f"first_mutation_coords_csv={mutations[0]['coords_csv']}")
        print(f"first_mutation_new_n={mutations[0]['new_n']}")
        print(f"first_mutation_new_m={mutations[0]['new_m']}")
        print(
            "first_mutation_killed_colouring_count_by_unit_neighborhood="
            f"{mutations[0]['killed_colouring_count_by_unit_neighborhood']}"
        )
    print("claim_scope=colour_guided_frontier_mutation_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOUR_GUIDED_MUTATION_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
