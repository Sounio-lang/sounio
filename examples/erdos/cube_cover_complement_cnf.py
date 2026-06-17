#!/usr/bin/env python3
"""Emit the DIMACS complement-cover CNF for a cube family.

The CNF shape matches `SounioSatCubeCover.cubeCoverComplementCNF`:

    colourCNF(n,k,edges) ∧ ⋀ cube, block(cube)

where `block(cube)` is the disjunction of the negated positive colour literals
in that cube. An UNSAT LRAT proof for this CNF is a Lean-checkable certificate
that the cube list covers every satisfying assignment of the base colouring CNF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cube_sieve_batch_manifest import parse_batch
from cube_sieve_propagation_manifest import parse_edge_file


def dimacs_lit(v: int, c: int, k: int) -> int:
    return v * k + c + 1


def validate_cubes(cubes: list[tuple[str, list[tuple[int, int]]]], n: int, k: int) -> None:
    seen_assignments: set[str] = set()
    for cube_id, cube in cubes:
        assignment = ",".join(f"{v}:{c}" for v, c in cube)
        if assignment in seen_assignments:
            raise ValueError(f"duplicate cube assignment: {assignment}")
        seen_assignments.add(assignment)
        for v, c in cube:
            if not (0 <= v < n):
                raise ValueError(f"cube {cube_id} vertex out of range: {v}")
            if not (0 <= c < k):
                raise ValueError(f"cube {cube_id} colour out of range: {c}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_batch", type=Path)
    parser.add_argument("out_cnf", type=Path)
    args = parser.parse_args()

    try:
        if args.k <= 0:
            raise ValueError("k must be positive")
        n, m, edges = parse_edge_file(args.edge_file)
        cubes = parse_batch(args.cube_batch)
        validate_cubes(cubes, n, args.k)

        clauses: list[list[int]] = []
        for v in range(n):
            clauses.append([dimacs_lit(v, c, args.k) for c in range(args.k)])
        for u, v in edges:
            for c in range(args.k):
                clauses.append([-dimacs_lit(u, c, args.k), -dimacs_lit(v, c, args.k)])
        for _cube_id, cube in cubes:
            clauses.append([-dimacs_lit(v, c, args.k) for v, c in cube])

        expected = n + m * args.k + len(cubes)
        if len(clauses) != expected:
            raise AssertionError(f"internal clause count mismatch: {len(clauses)} != {expected}")

        args.out_cnf.parent.mkdir(parents=True, exist_ok=True)
        with args.out_cnf.open("w", encoding="ascii") as f:
            f.write(f"p cnf {n * args.k} {len(clauses)}\n")
            for clause in clauses:
                f.write(" ".join(str(lit) for lit in clause))
                f.write(" 0\n")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("cube_cover_complement_cnf v1")
    print(f"output={args.out_cnf}")
    print(f"n={n}")
    print(f"m={m}")
    print(f"k={args.k}")
    print(f"cube_count={len(cubes)}")
    print(f"var_count={n * args.k}")
    print(f"clause_count={len(clauses)}")
    print("claim=base_plus_cube_blockers_dimacs_only")
    print("status=complement_cnf_emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
