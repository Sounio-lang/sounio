#!/usr/bin/env python3
"""CPU scout for exact rational unit-distance graph frontiers.

This is a search front door, not a chromatic-number proof. It either ingests an
exact rational coordinate CSV or generates a small deterministic rational
unit-step cloud, derives the unit-distance graph exactly, runs a bounded DSATUR
5-colourability probe, chooses high-degree split vertices, and emits the
existing chi6 solver-candidate source package for downstream SAT/LRAT/Lean work.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from make_chi6_rational_unit_graph_source_package import (
    parse_coord_table,
    sha256_file,
    unit_edges,
    validate_candidate_id,
)


K = 5
MAKER = Path(__file__).with_name("make_chi6_rational_unit_graph_source_package.py")


@dataclass(frozen=True)
class ColourProbe:
    status: str
    nodes: int
    colouring: list[int] | None


def point_key(p: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction, Fraction]:
    x, y = p
    return (x * x + y * y, x, y)


def rational_unit_directions(max_den: int) -> list[tuple[Fraction, Fraction]]:
    if max_den < 1:
        raise ValueError("--max-den must be positive")
    out: set[tuple[Fraction, Fraction]] = set()
    for c in range(1, max_den + 1):
        for a in range(0, c + 1):
            b2 = c * c - a * a
            b = math.isqrt(b2)
            if b * b != b2:
                continue
            if math.gcd(math.gcd(a, b), c) != 1:
                continue
            base = (Fraction(a, c), Fraction(b, c))
            variants = {
                base,
                (base[1], base[0]),
                (-base[0], base[1]),
                (base[0], -base[1]),
                (-base[0], -base[1]),
                (-base[1], base[0]),
                (base[1], -base[0]),
                (-base[1], -base[0]),
            }
            out.update(variants)
    return sorted(out, key=point_key)


def fib_rational_unit_directions(max_fib: int) -> list[tuple[Fraction, Fraction]]:
    """Fibonacci-biased unit directions (valid alternative for 5-fold/golden symmetry in unit-distance graphs).
    Uses Fib numbers as possible c in a^2+b^2=c^2 (known triples like 3-4-5,5-12-13,39-80-89 appear).
    Good for pentagonal configs relevant to k=5 boundary. 1/137 is valid via --max-den 137 (includes 88-105-137 triple).
    """
    if max_fib < 1:
        raise ValueError("--max-den (as max_fib) must be positive")
    fs = [1, 1]
    while fs[-1] <= max_fib:
        fs.append(fs[-1] + fs[-2])
    fs = [f for f in fs if f <= max_fib]
    out: set[tuple[Fraction, Fraction]] = set()
    for c in fs:
        for a in range(0, c + 1):
            b2 = c * c - a * a
            b = math.isqrt(b2)
            if b * b != b2:
                continue
            if math.gcd(math.gcd(a, b), c) != 1:
                continue
            base = (Fraction(a, c), Fraction(b, c))
            variants = {
                base,
                (base[1], base[0]),
                (-base[0], base[1]),
                (base[0], -base[1]),
                (-base[0], -base[1]),
                (-base[1], base[0]),
                (base[1], -base[0]),
                (-base[1], -base[0]),
            }
            out.update(variants)
    return sorted(out, key=point_key)


def generate_unit_step_cloud(
    max_den: int,
    layers: int,
    max_points: int,
    denom_mode: str = "pythag",
) -> list[tuple[Fraction, Fraction]]:
    """Generate a deterministic seed cloud; unit edges are still all-pairs exact.
    denom_mode: "pythag" (default) or "fib" (Fib-biased dens for golden/5-fold; valid alt).
    1/137 valid via --max-den 137 in pythag mode.
    """
    if layers < 0:
        raise ValueError("--layers must be non-negative")
    if max_points < 2:
        raise ValueError("--max-points must be at least 2")
    if denom_mode == "fib":
        directions = fib_rational_unit_directions(max_den)
    else:
        directions = rational_unit_directions(max_den)
    points: set[tuple[Fraction, Fraction]] = {(Fraction(0), Fraction(0))}
    frontier = set(points)
    for _ in range(layers):
        new_points = {
            (p[0] + d[0], p[1] + d[1])
            for p in frontier
            for d in directions
        }
        points.update(new_points)
        if len(points) > max_points:
            points = set(sorted(points, key=point_key)[:max_points])
        # Keep expanding only from generated points retained after the cap.
        frontier = new_points & points
    return sorted(points, key=point_key)


def write_coord_csv(path: Path, coords: list[tuple[Fraction, Fraction]]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("id,x,y\n")
        for i, (x, y) in enumerate(coords):
            f.write(f"{i},{x},{y}\n")


def adjacency(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def dsatur_probe(n: int, edges: list[tuple[int, int]], k: int, node_limit: int) -> ColourProbe:
    if node_limit < 1:
        raise ValueError("--dsatur-node-limit must be positive")
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + 1000))
    adj = adjacency(n, edges)
    degree = [len(a) for a in adj]
    colours = [-1] * n
    nodes = 0
    best: list[int] | None = None

    def available(v: int) -> list[int]:
        used = {colours[u] for u in adj[v] if colours[u] >= 0}
        return [c for c in range(k) if c not in used]

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

    def search() -> bool | None:
        nonlocal nodes, best
        nodes += 1
        if nodes > node_limit:
            return None
        if all(c >= 0 for c in colours):
            best = colours[:]
            return True
        for v in range(n):
            if colours[v] < 0 and not available(v):
                return False
        v = choose_vertex()
        for c in available(v):
            colours[v] = c
            result = search()
            colours[v] = -1
            if result is True:
                return True
            if result is None:
                return None
        return False

    result = search()
    if result is True:
        return ColourProbe("K_COLORING_FOUND", nodes, best)
    if result is False:
        return ColourProbe("NO_K_COLORING_FOUND_BY_CPU_PROBE_NONCERTIFYING", nodes, None)
    return ColourProbe("UNKNOWN_NODE_LIMIT", nodes, None)


def choose_split_vertices(
    n: int,
    edges: list[tuple[int, int]],
    split_depth: int,
    min_split_degree: int,
) -> list[int]:
    """Pick high-degree split vertices; adjacency is allowed and handled by SAT."""
    if split_depth < 1:
        raise ValueError("--split-depth must be positive")
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    ranked = [v for v in range(n) if deg[v] >= min_split_degree]
    ranked.sort(key=lambda v: (-deg[v], v))
    if len(ranked) < split_depth:
        raise ValueError(
            f"only {len(ranked)} vertices have degree >= {min_split_degree}; "
            f"cannot choose split depth {split_depth}"
        )
    return ranked[:split_depth]


def run_source_maker(
    coords_csv: Path,
    candidate_id: str,
    split_vertices: list[int],
    out_dir: Path,
    min_edges: int,
    max_vertices: int,
    min_split_degree: int,
    expected_n: int,
    expected_m: int,
) -> tuple[Path, Path]:
    if not MAKER.is_file():
        raise RuntimeError(f"missing source maker: {MAKER}")
    package_dir = out_dir / "source"
    split_csv = ",".join(str(v) for v in split_vertices)
    cmd = [
        sys.executable,
        str(MAKER),
        "--min-edges",
        str(min_edges),
        "--max-vertices",
        str(max_vertices),
        "--min-split-degree",
        str(min_split_degree),
        str(coords_csv),
        candidate_id,
        split_csv,
        str(package_dir),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    maker_out = out_dir / "source_maker.out"
    source_json = package_dir / f"{candidate_id}.candidate-source.json"
    maker_out.write_text(proc.stdout + proc.stderr, encoding="ascii")
    if proc.returncode != 0:
        raise RuntimeError(f"source maker failed with exit {proc.returncode}; see {maker_out}")
    if not source_json.is_file():
        raise RuntimeError(f"source maker did not emit expected JSON: {source_json}")
    with source_json.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if meta.get("schema") != "chi6_solver_candidate_package.v1":
        raise RuntimeError(f"source maker emitted unexpected schema in {source_json}")
    if meta.get("candidate_id") != candidate_id:
        raise RuntimeError(f"source maker emitted mismatched candidate_id in {source_json}")
    if meta.get("n") != expected_n or meta.get("m") != expected_m:
        raise RuntimeError(
            f"source maker emitted n={meta.get('n')} m={meta.get('m')}, "
            f"expected n={expected_n} m={expected_m}"
        )
    return source_json, maker_out


def load_or_generate_coords(args: argparse.Namespace) -> tuple[Path, list[tuple[Fraction, Fraction]], str]:
    if args.coords_csv is not None:
        coords = parse_coord_table(args.coords_csv, args.max_vertices)
        copied = args.out_dir / "frontier.coords.csv"
        shutil.copyfile(args.coords_csv, copied)
        return copied, coords, "ingest_csv"
    coords = generate_unit_step_cloud(args.max_den, args.layers, args.max_points, args.denom_mode)
    generated = args.out_dir / "frontier.coords.csv"
    write_coord_csv(generated, coords)
    return generated, coords, "generated_unit_step_cloud"


def selected_split_adjacencies(
    split_vertices: list[int],
    edges: list[tuple[int, int]],
) -> list[list[int]]:
    selected = set(split_vertices)
    return [[u, v] for u, v in edges if u in selected and v in selected]


def split_incident_edge_count(split_vertices: list[int], edges: list[tuple[int, int]]) -> int:
    selected = set(split_vertices)
    return sum(1 for u, v in edges if u in selected or v in selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords-csv", type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--denom-mode", choices=["pythag", "fib"], default="pythag", help="pythag: standard Pyth triples up to max-den; fib: Fib numbers as dens for golden/5-fold bias (valid alt)")
    parser.add_argument("--max-den", type=int, default=5)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=64)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--dsatur-node-limit", type=int, default=100_000)
    args = parser.parse_args()

    try:
        validate_candidate_id(args.candidate_id)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        coords_csv, coords, mode = load_or_generate_coords(args)
        n = len(coords)
        if n < args.min_vertices:
            raise ValueError(
                f"derived unit graph has {n} vertices, below required minimum {args.min_vertices}"
            )
        edges = unit_edges(coords)
        if len(edges) < args.min_edges:
            raise ValueError(
                f"derived unit graph has {len(edges)} edges, below required minimum {args.min_edges}"
            )
        split_vertices = choose_split_vertices(n, edges, args.split_depth, args.min_split_degree)
        probe = dsatur_probe(n, edges, K, args.dsatur_node_limit)
        split_adjacencies = selected_split_adjacencies(split_vertices, edges)
        source_json, maker_out = run_source_maker(
            coords_csv,
            args.candidate_id,
            split_vertices,
            args.out_dir,
            args.min_edges,
            args.max_vertices,
            args.min_split_degree,
            n,
            len(edges),
        )
        degrees = [0] * n
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        sidecar = args.out_dir / f"{args.candidate_id}.frontier-scout.json"
        payload = {
            "schema": "chi6_rational_frontier_scout.v1",
            "candidate_id": args.candidate_id,
            "mode": mode,
            "coords_csv": str(coords_csv),
            "coords_sha256": sha256_file(coords_csv),
            "candidate_source": str(source_json),
            "source_maker_out": str(maker_out),
            "n": n,
            "m": len(edges),
            "k": K,
            "min_vertices": args.min_vertices,
            "min_edges": args.min_edges,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "split_vertices": split_vertices,
            "split_vertex_degrees": {str(v): degrees[v] for v in split_vertices},
            "split_vertices_adjacent_pairs": split_adjacencies,
            "split_vertices_induced_edge_count": len(split_adjacencies),
            "split_vertices_induced_is_clique": len(split_adjacencies)
            == len(split_vertices) * (len(split_vertices) - 1) // 2,
            "split_vertices_incident_edge_count": split_incident_edge_count(split_vertices, edges),
            "split_cover_note": "split product enumerates assignments; SAT handles adjacent split conflicts",
            "dsatur_status": probe.status,
            "dsatur_nodes": probe.nodes,
            "dsatur_claim_scope": "bounded_cpu_search_probe_only",
            "dsatur_warning": "negative DSATUR probe statuses are not SAT/LRAT/Lean certificates",
            "claim_scope": "solver_candidate_frontier_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "promotable": 0,
        }
        if probe.colouring is not None:
            payload["colouring"] = probe.colouring
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_rational_frontier_scout v1")
    print(f"candidate_id={args.candidate_id}")
    print(f"mode={mode}")
    print(f"coords_csv={coords_csv}")
    print(f"coords_sha256={sha256_file(coords_csv)}")
    print(f"n={n}")
    print(f"m={len(edges)}")
    print(f"k={K}")
    print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
    print(f"dsatur_status={probe.status}")
    print(f"dsatur_nodes={probe.nodes}")
    print(f"candidate_source={source_json}")
    print(f"source_maker_out={maker_out}")
    print(f"frontier_scout={sidecar}")
    print("claim_scope=solver_candidate_frontier_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("promotable=0")
    print("status=SCOUT_SOURCE_PACKAGE_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
