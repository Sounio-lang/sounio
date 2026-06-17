#!/usr/bin/env python3
"""Emit a split-product cube batch for graph-colouring cube-and-conquer.

This is the canonical front door from a DIMACS graph plus chosen split vertices
to the batch format consumed by cube_sieve_refute_batch.py and the Lean cube-cover
generators. It is producer/search plumbing only: generated cubes are not a SAT
proof, a cover proof, or geometry evidence until the downstream LRAT/Lean gates
consume them.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

from cube_sieve_batch_manifest import sha256_file
from cube_sieve_propagation_manifest import parse_edge_file


def parse_split_vertices(raw: str, n: int) -> list[int]:
    if not raw:
        raise ValueError("--split-vertices cannot be empty")
    out: list[int] = []
    for token in raw.split(","):
        if not re.fullmatch(r"[0-9]+", token):
            raise ValueError(f"bad split vertex token: {token!r}")
        v = int(token)
        if not (0 <= v < n):
            raise ValueError(f"split vertex out of range: {v}")
        if v in out:
            raise ValueError(f"duplicate split vertex: {v}")
        out.append(v)
    return out


def cube_id(split_vertices: list[int], colours: tuple[int, ...], prefix: str) -> str:
    body = "_".join(f"v{v}_c{c}" for v, c in zip(split_vertices, colours, strict=True))
    if not prefix:
        return body
    return f"{prefix}_{body}"


def cube_line(split_vertices: list[int], colours: tuple[int, ...], prefix: str) -> str:
    name = cube_id(split_vertices, colours, prefix)
    assignments = " ".join(
        f"{v}:{c}" for v, c in zip(split_vertices, colours, strict=True)
    )
    return f"{name}: {assignments}\n"


def checked_cube_count(k: int, depth: int, max_cubes: int) -> int:
    count = k**depth
    if max_cubes > 0 and count > max_cubes:
        raise ValueError(
            f"split product would emit {count} cubes; pass --max-cubes 0 or a larger cap"
        )
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("out_file", type=Path)
    parser.add_argument(
        "--split-vertices",
        required=True,
        help="comma-separated zero-based vertices to split over all colours",
    )
    parser.add_argument(
        "--id-prefix",
        default="",
        help="optional cube-id prefix, e.g. chi6 gives chi6_v0_c0...",
    )
    parser.add_argument(
        "--max-cubes",
        type=int,
        default=1_000_000,
        help="fail closed above this many cubes; 0 disables the cap",
    )
    args = parser.parse_args()

    try:
        if args.k <= 0 or args.k >= 62:
            raise ValueError("k must satisfy 0 < k < 62")
        if args.max_cubes < 0:
            raise ValueError("--max-cubes must be non-negative")
        if args.id_prefix and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.id_prefix):
            raise ValueError("--id-prefix must use only letters, digits, '.', '_', or '-'")
        n, m, _edges = parse_edge_file(args.edge_file)
        split_vertices = parse_split_vertices(args.split_vertices, n)
        count = checked_cube_count(args.k, len(split_vertices), args.max_cubes)
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        with args.out_file.open("w", encoding="ascii") as f:
            f.write("# split-product cube batch: zero-based vertex:colour assignments\n")
            f.write(
                f"# graph={args.edge_file} n={n} m={m} k={args.k} "
                f"split_vertices={','.join(str(v) for v in split_vertices)}\n"
            )
            for colours in itertools.product(range(args.k), repeat=len(split_vertices)):
                f.write(cube_line(split_vertices, colours, args.id_prefix))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    first_colours = tuple(0 for _ in split_vertices)
    last_colours = tuple(args.k - 1 for _ in split_vertices)
    print("cube_split_batch v1")
    print("trust_boundary=search_untrusted__drat_lrat_lean_verified_required")
    print("output=split_product_cube_batch")
    print(f"edge_path={args.edge_file}")
    print(f"edge_sha256={sha256_file(args.edge_file)}")
    print(f"n={n}")
    print(f"m={m}")
    print(f"k={args.k}")
    print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
    print(f"split_depth={len(split_vertices)}")
    print(f"cube_count={count}")
    print(f"cube_batch_path={args.out_file}")
    print(f"cube_batch_sha256={sha256_file(args.out_file)}")
    print(f"first_cube_id={cube_id(split_vertices, first_colours, args.id_prefix)}")
    print(f"last_cube_id={cube_id(split_vertices, last_colours, args.id_prefix)}")
    print("cover_route=split_vertices_atleast_one_product")
    print("verified_claim=none")
    print("global_unsat_claim=none")
    print("geometry_claim=none")
    print("promotable=0")
    print("status=cube_batch_emitted_unpromotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
