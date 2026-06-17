#!/usr/bin/env python3
"""Bounded finite-graph producer for the chi>=6 search lane.

This is intentionally not a Euclidean theorem prover. It enumerates a declared
finite graph family, looks for graphs that are not `k`-colourable, and emits the
same DIMACS edge/cube-batch shape consumed by the existing SAT/LRAT/Lean
pipeline. A result here is an untrusted finite-graph handoff only; promotion
still requires exact Euclidean geometry and checked certificates on the same
edge list.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

from cube_sieve_batch_manifest import sha256_file
from cube_sieve_propagation_manifest import parse_edge_file
from cube_split_batch import (
    checked_cube_count,
    cube_line,
    parse_split_vertices,
)


Edge = tuple[int, int]


def all_edges(n: int) -> list[Edge]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def is_k_colourable(n: int, k: int, edges: tuple[Edge, ...]) -> bool:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    colour = [-1 for _ in range(n)]

    def go(pos: int) -> bool:
        if pos == len(order):
            return True
        v = order[pos]
        used = {colour[w] for w in adj[v] if colour[w] >= 0}
        for c in range(k):
            if c in used:
                continue
            colour[v] = c
            if go(pos + 1):
                return True
            colour[v] = -1
        return False

    return go(0)


def write_edge_file(path: Path, n: int, edges: tuple[Edge, ...]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u + 1} {v + 1}\n")


def copy_edge_file(src: Path, dst: Path) -> tuple[int, int]:
    n, m, _edges = parse_edge_file(src)
    shutil.copyfile(src, dst)
    return n, m


def write_external_meta(
    path: Path,
    *,
    source_edge: Path,
    packaged_edge: Path,
    candidate_id: str,
    n: int,
    m: int,
    k: int,
    split_vertices: list[int],
) -> None:
    source_sha = sha256_file(source_edge)
    packaged_sha = sha256_file(packaged_edge)
    if source_sha != packaged_sha:
        raise ValueError(
            f"packaged edge hash mismatch: source {source_sha}, packaged {packaged_sha}"
        )
    meta = {
        "schema": "chi6_external_dimacs_edge_package.v1",
        "candidate_id": candidate_id,
        "source_edge_path": str(source_edge),
        "source_edge_sha256": source_sha,
        "packaged_edge_path": str(packaged_edge),
        "packaged_edge_sha256": packaged_sha,
        "n": n,
        "m": m,
        "k": k,
        "split_vertices": split_vertices,
        "argv": sys.argv,
        "packaged_at_unix": int(time.time()),
        "provenance_scope": "edge_packaging_only",
        "promotion_gate": "requires_lrat_lean_and_exact_euclidean_geometry",
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_split_batch(
    path: Path,
    edge_path: Path,
    n: int,
    m: int,
    k: int,
    split_vertices: list[int],
    prefix: str,
    max_cubes: int,
) -> int:
    count = checked_cube_count(k, len(split_vertices), max_cubes)
    with path.open("w", encoding="ascii") as f:
        f.write("# split-product cube batch: zero-based vertex:colour assignments\n")
        f.write(
            f"# graph={edge_path} n={n} m={m} k={k} "
            f"split_vertices={','.join(str(v) for v in split_vertices)}\n"
        )
        for colours in itertools.product(range(k), repeat=len(split_vertices)):
            f.write(cube_line(split_vertices, colours, prefix))
    return count


def search_graphs(
    n: int,
    k: int,
    min_edges: int,
    max_edges: int,
    max_graphs: int,
    max_candidates: int,
) -> tuple[int, bool, list[tuple[int, tuple[Edge, ...]]]]:
    universe = all_edges(n)
    examined = 0
    truncated = False
    candidates: list[tuple[int, tuple[Edge, ...]]] = []
    for m in range(max_edges, min_edges - 1, -1):
        for combo in itertools.combinations(universe, m):
            if max_graphs > 0 and examined >= max_graphs:
                truncated = True
                return examined, truncated, candidates
            examined += 1
            if not is_k_colourable(n, k, combo):
                candidates.append((m, combo))
                if len(candidates) >= max_candidates:
                    return examined, truncated, candidates
    return examined, truncated, candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--edge-file",
        type=Path,
        default=None,
        help="package an existing DIMACS p edge graph instead of enumerating all_simple_graphs",
    )
    parser.add_argument(
        "--candidate-id",
        default="",
        help="optional id for --edge-file packaging; defaults to external_n<N>_m<M>",
    )
    parser.add_argument("--min-edges", type=int, default=0)
    parser.add_argument("--max-edges", type=int, default=-1)
    parser.add_argument("--max-graphs", type=int, default=100_000)
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument(
        "--split-vertices",
        default="",
        help="optional comma-separated zero-based vertices for emitted cube batches",
    )
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    args = parser.parse_args()

    try:
        if args.k <= 0 or args.k >= 62:
            raise ValueError("--k must satisfy 0 < k < 62")
        if args.max_graphs < 0:
            raise ValueError("--max-graphs must be non-negative")
        if args.max_candidates <= 0:
            raise ValueError("--max-candidates must be positive")
        if args.max_cubes < 0:
            raise ValueError("--max-cubes must be non-negative")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[str, Path, Path | None, Path | None, int, int]] = []
        if args.edge_file is not None:
            if args.n != 0:
                raise ValueError(
                    "--n must not be provided in external mode; vertex count is read from "
                    "the DIMACS p edge header"
                )
            n, m = parse_edge_file(args.edge_file)[:2]
            if m == 0:
                raise ValueError("--edge-file must contain at least one edge")
            max_edges = m
            examined = 1
            truncated = False
            candidates = []
            raw_id = args.candidate_id or f"external_n{n}_k{args.k}_m{m}"
            if not raw_id.replace(".", "_").replace("-", "_").replace("_", "").isalnum():
                raise ValueError("--candidate-id must use only letters, digits, '.', '_', or '-'")
            split_vertices = parse_split_vertices(args.split_vertices, n) if args.split_vertices else []
            edge_path = args.out_dir / f"{raw_id}.edge"
            copy_edge_file(args.edge_file, edge_path)
            meta_path = args.out_dir / f"{raw_id}.meta.json"
            write_external_meta(
                meta_path,
                source_edge=args.edge_file,
                packaged_edge=edge_path,
                candidate_id=raw_id,
                n=n,
                m=m,
                k=args.k,
                split_vertices=split_vertices,
            )
            cube_path: Path | None = None
            cube_count = 0
            if split_vertices:
                cube_path = args.out_dir / f"{raw_id}.cubes"
                cube_count = write_split_batch(
                    cube_path,
                    edge_path,
                    n,
                    m,
                    args.k,
                    split_vertices,
                    raw_id,
                    args.max_cubes,
                )
            rows.append((raw_id, edge_path, cube_path, meta_path, m, cube_count))
            family = "external_dimacs_edge"
            finite_search_claim = "none_external_graph_packaging_only"
            status = "EXTERNAL_GRAPH_PACKAGED_UNPROMOTABLE"
        else:
            if args.n <= 0:
                raise ValueError("--n must be positive unless --edge-file is supplied")
            n = args.n
            total_edges = n * (n - 1) // 2
            max_edges = total_edges if args.max_edges < 0 else args.max_edges
            if not (0 <= args.min_edges <= max_edges <= total_edges):
                raise ValueError("edge bounds must satisfy 0 <= min_edges <= max_edges <= n*(n-1)/2")
            split_vertices = parse_split_vertices(args.split_vertices, n) if args.split_vertices else []
            examined, truncated, candidates = search_graphs(
                n, args.k, args.min_edges, max_edges, args.max_graphs, args.max_candidates
            )
            for idx, (m, edges) in enumerate(candidates):
                cid = f"candidate_{idx:04d}_n{n}_k{args.k}_m{m}"
                edge_path = args.out_dir / f"{cid}.edge"
                write_edge_file(edge_path, n, edges)
                cube_path = None
                cube_count = 0
                if split_vertices:
                    cube_path = args.out_dir / f"{cid}.cubes"
                    cube_count = write_split_batch(
                        cube_path,
                        edge_path,
                        n,
                        m,
                        args.k,
                        split_vertices,
                        cid,
                        args.max_cubes,
                    )
                rows.append((cid, edge_path, cube_path, None, m, cube_count))
            family = "all_simple_graphs"
            finite_search_claim = "untrusted_backtracking_only"
            if candidates:
                status = "FINITE_GRAPH_CANDIDATE_EMITTED_UNPROMOTABLE"
            elif truncated:
                status = "FINITE_GRAPH_SEARCH_TRUNCATED_NO_CANDIDATE"
            else:
                status = "FINITE_GRAPH_CANDIDATE_ABSENT_WITHIN_BOUND"
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_candidate_search_manifest v1")
    print("trust_boundary=finite_graph_search_untrusted__geometry_lrat_lean_required")
    print(f"family={family}")
    print(f"n={n}")
    print(f"k={args.k}")
    print(f"min_edges={args.min_edges}")
    print(f"max_edges={max_edges}")
    print(f"max_graphs={args.max_graphs}")
    print(f"graphs_examined={examined}")
    print(f"search_truncated={1 if truncated else 0}")
    print(f"candidate_count={len(rows)}")
    print(f"out_dir={args.out_dir}")
    print(f"split_vertices={','.join(str(v) for v in split_vertices) if split_vertices else 'none'}")
    print(f"finite_graph_search_claim={finite_search_claim}")
    print("verified_claim=none")
    print("global_unsat_claim=none")
    print("geometry_claim=none")
    print("promotable=0")
    for idx, (cid, edge_path, cube_path, meta_path, m, cube_count) in enumerate(rows):
        parts = [
            "candidate",
            f"index={idx}",
            f"id={cid}",
            f"n={n}",
            f"m={m}",
            f"edge_path={edge_path}",
            f"edge_sha256={sha256_file(edge_path)}",
        ]
        if family == "all_simple_graphs":
            parts.append("not_k_colourable_by_untrusted_search=1")
        else:
            parts.append("not_k_colourable_claim=none")
        parts.append("geometry_claim=none")
        if meta_path is not None:
            parts.extend(
                [
                    f"source_meta_path={meta_path}",
                    f"source_meta_sha256={sha256_file(meta_path)}",
                ]
            )
        if cube_path is not None:
            parts.extend(
                [
                    f"cube_batch_path={cube_path}",
                    f"cube_batch_sha256={sha256_file(cube_path)}",
                    f"cube_count={cube_count}",
                    "cover_route=split_vertices_atleast_one_product",
                ]
            )
        print(" ".join(parts))
    print(f"status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
