#!/usr/bin/env python3
"""Sample bounded DSATUR 5-colourings for colour-guided frontier search.

This is search instrumentation only.  It enumerates a small deterministic set of
proper colourings for an exact-rational unit-distance graph so downstream
colour-guided mutation can score candidate points against more than one
observed colouring.  Finding colourings is not a chromatic lower-bound claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chi6_colour_guided_evolution import colouring_line
from make_chi6_rational_unit_graph_source_package import (
    parse_coord_table,
    sha256_file,
    unit_edges,
)


K = 5


def adjacency(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def canonical_colouring(colours: list[int]) -> tuple[int, ...]:
    renaming: dict[int, int] = {}
    out: list[int] = []
    for colour in colours:
        if colour not in renaming:
            renaming[colour] = len(renaming)
        out.append(renaming[colour])
    return tuple(out)


def enumerate_colourings(
    *,
    n: int,
    edges: list[tuple[int, int]],
    max_colourings: int,
    node_limit: int,
) -> tuple[list[list[int]], int, str]:
    if max_colourings < 1:
        raise ValueError("--max-colourings must be positive")
    if node_limit < 1:
        raise ValueError("--node-limit must be positive")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 1000))
    adj = adjacency(n, edges)
    degree = [len(neighbours) for neighbours in adj]
    colours = [-1] * n
    seen: set[tuple[int, ...]] = set()
    found: list[list[int]] = []
    nodes = 0

    def available(v: int) -> list[int]:
        used = {colours[u] for u in adj[v] if colours[u] >= 0}
        return [c for c in range(K) if c not in used]

    def choose_vertex() -> int:
        candidates = [v for v in range(n) if colours[v] < 0]
        return max(
            candidates,
            key=lambda v: (
                len({colours[u] for u in adj[v] if colours[u] >= 0}),
                degree[v],
                -v,
            ),
        )

    def symmetry_reduced_colours(v: int) -> list[int]:
        used_colours = {colour for colour in colours if colour >= 0}
        next_new = (max(used_colours) + 1) if used_colours else 0
        allowed = []
        for colour in available(v):
            if colour in used_colours or colour == next_new:
                allowed.append(colour)
        return allowed

    def search() -> str | None:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return "NODE_LIMIT_REACHED"
        if len(found) >= max_colourings:
            return "MAX_COLOURINGS_REACHED"
        if all(colour >= 0 for colour in colours):
            key = canonical_colouring(colours)
            if key not in seen:
                seen.add(key)
                found.append(list(key))
                if len(found) >= max_colourings:
                    return "MAX_COLOURINGS_REACHED"
            return None
        for v in range(n):
            if colours[v] < 0 and not available(v):
                return None
        v = choose_vertex()
        for colour in symmetry_reduced_colours(v):
            colours[v] = colour
            status = search()
            colours[v] = -1
            if status in {"MAX_COLOURINGS_REACHED", "NODE_LIMIT_REACHED"}:
                return status
        return None

    status = search() or "COLOURING_SPACE_EXHAUSTED"
    return found, nodes, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--coords-csv", type=Path, required=True)
    parser.add_argument("--candidate-id", default="colouring_sample")
    parser.add_argument("--max-colourings", type=int, default=8)
    parser.add_argument("--node-limit", type=int, default=100_000)
    parser.add_argument("--max-vertices", type=int, default=4096)
    args = parser.parse_args()

    try:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise ValueError("out_dir already exists and is non-empty")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        coords = parse_coord_table(args.coords_csv, args.max_vertices)
        edges = unit_edges(coords)
        colourings, nodes, status = enumerate_colourings(
            n=len(coords),
            edges=edges,
            max_colourings=args.max_colourings,
            node_limit=args.node_limit,
        )
        colourings_path = args.out_dir / "colourings.txt"
        colourings_path.write_text(
            "\n".join(
                colouring_line(colouring, len(coords), f"sampled_colouring_{i:03d}")
                for i, colouring in enumerate(colourings)
            )
            + ("\n" if colourings else ""),
            encoding="ascii",
        )
        manifest = {
            "schema": "chi6_colouring_sampler.v1",
            "candidate_id": args.candidate_id,
            "coords_csv": str(args.coords_csv),
            "coords_sha256": sha256_file(args.coords_csv),
            "colourings_file": str(colourings_path),
            "colourings_sha256": sha256_file(colourings_path),
            "n": len(coords),
            "m": len(edges),
            "k": K,
            "max_colourings": args.max_colourings,
            "node_limit": args.node_limit,
            "search_status": status,
            "search_nodes": nodes,
            "colouring_count": len(colourings),
            "claim_scope": "bounded_colouring_sampling_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
        }
        manifest_path = args.out_dir / "colouring_sampler.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_colouring_sampler v1")
    print(f"colouring_sampler_json={manifest_path}")
    print(f"colouring_sampler_json_sha256={sha256_file(manifest_path)}")
    print(f"candidate_id={manifest['candidate_id']}")
    print(f"n={manifest['n']}")
    print(f"m={manifest['m']}")
    print(f"k={manifest['k']}")
    print(f"colourings_file={colourings_path}")
    print(f"colouring_count={manifest['colouring_count']}")
    print(f"search_nodes={manifest['search_nodes']}")
    print(f"search_status={manifest['search_status']}")
    print("claim_scope=bounded_colouring_sampling_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=COLOURING_SAMPLE_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
