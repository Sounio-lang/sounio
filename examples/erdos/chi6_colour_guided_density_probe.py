#!/usr/bin/env python3
"""Probe exact-rational colour-guided mutation density envelopes.

This is search instrumentation, not a chromatic-number proof.  It reuses the
colour-guided mutator's exact unit-neighbour scoring, but instead of emitting
mutated coordinate files it records how many candidate rational unit steps exist
at each denominator budget and neighbour-count threshold.  The resulting ledger
helps decide whether a frontier is genuinely exhausted under a bounded search
envelope or merely needs a wider denominator campaign.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from chi6_colour_guided_mutation import (
    K,
    PointScore,
    candidate_points,
    fraction_text,
    load_colourings_file,
    load_from_satfanout,
    parse_colouring,
    score_point,
    validate_colouring_respects_edges,
)
from make_chi6_rational_unit_graph_source_package import (
    parse_coord_table,
    sha256_file,
    unit_edges,
)


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


def score_row(score: PointScore, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "x": fraction_text(score.point[0]),
        "y": fraction_text(score.point[1]),
        "neighbor_vertices": list(score.neighbor_vertices),
        "neighbor_count": len(score.neighbor_vertices),
        "killed_colouring_count": len(score.killed),
        "killed_colouring_labels": list(score.killed),
        "covered_colour_total": score.covered_colour_total,
        "max_covered_colours": score.max_covered_colours,
    }


def threshold_summary(scored: list[PointScore], threshold: int) -> dict[str, Any]:
    eligible = [score for score in scored if len(score.neighbor_vertices) >= threshold]
    if not eligible:
        return {
            "min_neighbor_count": threshold,
            "available_count": 0,
            "best_neighbor_count": 0,
            "best_killed_colouring_count": 0,
            "best_covered_colour_total": 0,
            "best_max_covered_colours": 0,
            "best_point": "NONE",
        }
    best = eligible[0]
    return {
        "min_neighbor_count": threshold,
        "available_count": len(eligible),
        "best_neighbor_count": len(best.neighbor_vertices),
        "best_killed_colouring_count": len(best.killed),
        "best_covered_colour_total": best.covered_colour_total,
        "best_max_covered_colours": best.max_covered_colours,
        "best_point": {
            "x": fraction_text(best.point[0]),
            "y": fraction_text(best.point[1]),
        },
    }


def probe_denominator(
    *,
    coords: list[Any],
    colourings: list[Any],
    max_den: int,
    max_candidates: int,
    top_points: int,
    thresholds: list[int],
) -> dict[str, Any]:
    points = candidate_points(coords, max_den=max_den, max_candidates=max_candidates)
    scored = [
        score
        for point in points
        if (score := score_point(point, coords, colourings)) is not None
    ]
    scored.sort(key=lambda score: score.score_key(), reverse=True)
    histogram = Counter(len(score.neighbor_vertices) for score in scored)
    max_neighbor_count = max(histogram) if histogram else 0
    return {
        "max_den": max_den,
        "candidate_point_count_considered": len(points),
        "candidate_point_prefix_may_be_truncated": len(points) >= max_candidates,
        "candidate_point_count_scored": len(scored),
        "max_neighbor_count": max_neighbor_count,
        "single_point_full_blocker_count": sum(1 for score in scored if score.killed),
        "neighbor_count_histogram": [
            {"neighbor_count": key, "count": histogram[key]} for key in sorted(histogram)
        ],
        "thresholds": [threshold_summary(scored, threshold) for threshold in thresholds],
        "top_points": [
            score_row(score, rank) for rank, score in enumerate(scored[:top_points])
        ],
    }


def first_den_by_threshold(probes: list[dict[str, Any]], thresholds: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for threshold in thresholds:
        first: int | str = "NONE"
        for probe in probes:
            rows = {
                int(row["min_neighbor_count"]): int(row["available_count"])
                for row in probe["thresholds"]
            }
            if rows.get(threshold, 0) > 0:
                first = int(probe["max_den"])
                break
        out[str(threshold)] = first
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--satfanout-json", type=Path)
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--colourings-file", type=Path)
    parser.add_argument("--candidate-prefix", default="cgdensity")
    parser.add_argument("--max-den-list", default="5")
    parser.add_argument("--min-neighbor-count-list", default="1,2,3")
    parser.add_argument("--max-candidates", type=int, default=20_000)
    parser.add_argument("--top-points", type=int, default=20)
    args = parser.parse_args()

    try:
        if args.max_candidates <= 0:
            raise ValueError("--max-candidates must be positive")
        if args.top_points <= 0:
            raise ValueError("--top-points must be positive")
        max_den_list = parse_positive_int_list(args.max_den_list, "--max-den-list")
        thresholds = parse_positive_int_list(
            args.min_neighbor_count_list,
            "--min-neighbor-count-list",
        )
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError("out_dir already exists and is non-empty")
        has_satfanout = args.satfanout_json is not None
        has_explicit = args.coords_csv is not None or args.colourings_file is not None
        if has_satfanout and has_explicit:
            raise ValueError("pass either --satfanout-json or explicit coords/colourings, not both")
        if not has_satfanout and (args.coords_csv is None or args.colourings_file is None):
            raise ValueError("pass --satfanout-json or both --coords-csv and --colourings-file")
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
            source_candidate_id = args.candidate_prefix
            raw_colourings = load_colourings_file(args.colourings_file)
            input_path = args.colourings_file
            input_mode = "explicit_colourings"

        coords = parse_coord_table(coords_csv, 1_000_000)
        base_edges = unit_edges(coords)
        colourings = [
            parse_colouring(raw, len(coords), f"colouring_{i:03d}")
            for i, raw in enumerate(raw_colourings)
        ]
        for colouring in colourings:
            validate_colouring_respects_edges(colouring, base_edges)

        probes = [
            probe_denominator(
                coords=coords,
                colourings=colourings,
                max_den=max_den,
                max_candidates=args.max_candidates,
                top_points=args.top_points,
                thresholds=thresholds,
            )
            for max_den in max_den_list
        ]
        max_probe = max(
            probes,
            key=lambda row: (
                int(row["max_neighbor_count"]),
                int(row["candidate_point_count_scored"]),
                -int(row["max_den"]),
            ),
        )
        manifest = {
            "schema": "chi6_colour_guided_density_probe.v1",
            "input_mode": input_mode,
            "input_path": str(input_path),
            "coords_csv": str(coords_csv),
            "coords_sha256": sha256_file(coords_csv),
            "source_candidate_id": source_candidate_id,
            "candidate_prefix": args.candidate_prefix,
            "n": len(coords),
            "m": len(base_edges),
            "k": K,
            "observed_colouring_count": len(colourings),
            "max_den_list": max_den_list,
            "min_neighbor_count_list": thresholds,
            "max_candidates": args.max_candidates,
            "top_points_requested": args.top_points,
            "max_observed_neighbor_count": int(max_probe["max_neighbor_count"]),
            "max_observed_neighbor_count_den": int(max_probe["max_den"]),
            "first_den_by_min_neighbor_count": first_den_by_threshold(probes, thresholds),
            "probes": probes,
            "claim_scope": "colour_guided_density_envelope_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_exact_geometry_plus_checked_sat_lrat_after_search",
        }
        manifest_path = args.out_dir / "colour_guided_density_probe.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2

    print("chi6_colour_guided_density_probe v1")
    print(f"density_probe_json={manifest_path}")
    print(f"density_probe_json_sha256={sha256_file(manifest_path)}")
    print(f"source_candidate_id={manifest['source_candidate_id']}")
    print(f"n={manifest['n']}")
    print(f"m={manifest['m']}")
    print(f"observed_colouring_count={manifest['observed_colouring_count']}")
    print(f"max_observed_neighbor_count={manifest['max_observed_neighbor_count']}")
    print(f"max_observed_neighbor_count_den={manifest['max_observed_neighbor_count_den']}")
    for threshold in thresholds:
        first_den = manifest["first_den_by_min_neighbor_count"][str(threshold)]
        print(f"first_den_with_min_neighbor_count_{threshold}={first_den}")
    print("claim_scope=colour_guided_density_envelope_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOUR_GUIDED_DENSITY_PROBE_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
