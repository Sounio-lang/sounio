#!/usr/bin/env python3
"""Emit an unpromotable cube-propagation manifest from DIMACS edges plus a cube.

The producer is search plumbing, not a theorem prover. It runs deterministic
unit-domain propagation for graph colouring and emits enough metadata for
validate_cube_sieve_manifest.py to replay the domain transitions.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


def dimacs_lit(v: int, c: int, k: int) -> int:
    return v * k + c + 1


def parse_edge_file(path: Path) -> tuple[int, int, list[tuple[int, int]]]:
    n = -1
    m = -1
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[:2] == ["p", "edge"] and len(parts) == 4:
            if n != -1:
                raise ValueError(f"{path}:{lineno}: duplicate p edge header")
            n = int(parts[2])
            m = int(parts[3])
            if n <= 0 or m < 0:
                raise ValueError(f"{path}:{lineno}: invalid p edge header")
            continue
        if parts and parts[0] == "e" and len(parts) == 3:
            if n == -1:
                raise ValueError(f"{path}:{lineno}: edge before p edge header")
            u = int(parts[1]) - 1
            v = int(parts[2]) - 1
            if not (0 <= u < n) or not (0 <= v < n) or u == v:
                raise ValueError(f"{path}:{lineno}: malformed edge {parts[1]} {parts[2]}")
            key = (u, v) if u < v else (v, u)
            if key in seen:
                raise ValueError(f"{path}:{lineno}: duplicate unordered edge {key}")
            seen.add(key)
            edges.append(key)
            continue
        raise ValueError(f"{path}:{lineno}: unsupported DIMACS line: {raw!r}")
    if n == -1:
        raise ValueError(f"{path}: missing p edge header")
    if len(edges) != m:
        raise ValueError(f"{path}: p edge declares {m} edges but found {len(edges)}")
    return n, m, edges


def parse_cube(path: Path, n: int, k: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    seen_vertices: set[int] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(?:v(?:ertex)?=)?(\d+)\s+(?:c(?:olou?r)?=)?(\d+)", line)
        if not m:
            raise ValueError(f"{path}:{lineno}: expected zero-based '<vertex> <colour>'")
        v = int(m.group(1))
        c = int(m.group(2))
        if not (0 <= v < n) or not (0 <= c < k):
            raise ValueError(f"{path}:{lineno}: cube assignment out of range: {v} {c}")
        if v in seen_vertices:
            raise ValueError(f"{path}:{lineno}: duplicate cube assignment for vertex {v}")
        seen_vertices.add(v)
        rows.append((v, c))
    if not rows:
        raise ValueError(f"{path}: cube must contain at least one assignment")
    return rows


def singleton_colour(mask: int) -> int | None:
    if mask == 0 or mask & (mask - 1):
        return None
    c = 0
    while mask & 1 == 0:
        mask >>= 1
        c += 1
    return c


def emit_assignment(v: int, c: int, k: int) -> None:
    lit = dimacs_lit(v, c, k)
    print(f"    precolour vertex={v} colour={c} colour_valid=1 bounded_encoding_supported=1 "
          f"lean_var={lit - 1} dimacs_lit={lit}")
    print(f"      rup_fact_clause={lit} 0")


def propagate(
    n: int, k: int, edges: list[tuple[int, int]], cube: list[tuple[int, int]]
) -> tuple[int, int, int, int, list[int]]:
    all_colours = (1 << k) - 1
    domains = [all_colours for _ in range(n)]
    for v, c in cube:
        domains[v] = 1 << c

    changed = True
    conflict_vertex = -1
    trail = 0
    passes = 0
    max_passes = max(1, n * k + 1)
    while changed and conflict_vertex < 0 and passes < max_passes:
        changed = False
        passes += 1
        for u, v in edges:
            for src, dst in ((u, v), (v, u)):
                colour = singleton_colour(domains[src])
                if colour is None:
                    continue
                before = domains[dst]
                after = before & (all_colours ^ (1 << colour))
                if after == before:
                    continue
                domains[dst] = after
                trail += 1
                changed = True
                print(f"    trail_step={trail} op=remove reason=edge({src},{dst}) "
                      f"source_singleton_colour={colour} target_vertex={dst} "
                      f"before_domain={before} after_domain={after} "
                      f"removed_dimacs_lit={dimacs_lit(dst, colour, k)}")
                print(f"      rup_reason_clause=-{dimacs_lit(src, colour, k)} "
                      f"-{dimacs_lit(dst, colour, k)} 0")
                if after == 0:
                    conflict_vertex = dst
                    break
            if conflict_vertex >= 0:
                break
    guard = 1 if conflict_vertex < 0 and changed and passes >= max_passes else 0
    return passes, guard, trail, conflict_vertex, domains


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_file", type=Path)
    args = parser.parse_args()

    if args.k <= 0 or args.k >= 62:
        raise SystemExit("error: k must satisfy 0 < k < 62")
    try:
        n, m, edges = parse_edge_file(args.edge_file)
        cube = parse_cube(args.cube_file, n, args.k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    edge_sha = hashlib.sha256(args.edge_file.read_bytes()).hexdigest()
    print("cube_sieve_propagation_manifest v1")
    print("trust_boundary=search_untrusted__drat_lrat_lean_verified_required")
    print("output=dimacs_cube_propagation_manifest\n")
    print("section=dimacs_cube_propagation")
    print("  purpose=data-driven producer smoke for domain propagation plus RUP-trail metadata")
    print("  graph_family=dimacs_edge")
    print(f"  edge_path={args.edge_file}")
    print(f"  edge_sha256={edge_sha}")
    print(f"  n={n}")
    print(f"  m={m}")
    print(f"  edge_count={len(edges)}")
    print(f"  k={args.k}")
    print(f"  cube_assignment_count={len(cube)}")
    print("  claim=domain_propagation_result_only")
    print("  verified_claim=none")
    print("  geometry_claim=none")
    print("  proof_artifact_sha256=NONE")
    print("  proof_required=compose_this_trail_to_DRAT_OR_LRAT_AND_CHECK_IN_LEAN")
    for u, v in edges:
        print(f"    edge {u} {v}")
    for idx, (v, c) in enumerate(cube):
        print(f"    cube_assignment index={idx} vertex={v} colour={c}")
        emit_assignment(v, c, args.k)
    negated = " ".join(f"-{dimacs_lit(v, c, args.k)}" for v, c in cube)
    print("  rup_clause_scope=declared_cube_assignments")
    print(f"  rup_clause_negated_cube={negated} 0")
    passes, guard, trail_len, conflict_vertex, domains = propagate(n, args.k, edges, cube)
    print(f"  propagation_passes={passes}")
    print(f"  termination_guard_tripped={guard}")
    print(f"  trail_len={trail_len}")
    print(f"  conflict={1 if conflict_vertex >= 0 else 0}")
    print(f"  conflict_vertex={conflict_vertex}")
    print(f"  hard_cube={0 if conflict_vertex >= 0 else 1}")
    print("  final_domains=" + ",".join(str(d) for d in domains) + "\n")
    print("promotion_gate=REJECT_NONE_PROOF_ARTIFACT")
    print("promotable=0")
    print("status=manifest_emitted_unpromotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
