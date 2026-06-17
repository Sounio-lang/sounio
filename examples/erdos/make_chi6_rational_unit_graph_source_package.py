#!/usr/bin/env python3
"""Derive a rational unit-distance graph and package it as a chi6 source.

This is geometry/search plumbing, not a chromatic-number proof. The producer
reads exact rational coordinates, derives every pair at squared distance 1, and
then emits the same `chi6_solver_candidate_package.v1` JSON consumed by the
existing integrated preflight. The derived edge list is therefore bound to the
coordinate table by exact rational arithmetic before any SAT/LRAT work starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from gen_lean_rational_geometry import dist2, parse_rat, validate_geometry


ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
UNIT_DIST_SQUARED = Fraction(1, 1)
DEFAULT_MAX_VERTICES = 4096
DEFAULT_MIN_SPLIT_DEGREE = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_split_vertices(raw: str, n: int) -> list[int]:
    if not raw:
        raise ValueError("split_vertices cannot be empty")
    out: list[int] = []
    for token in raw.split(","):
        if not re.fullmatch(r"[0-9]+", token):
            raise ValueError(f"bad split vertex token: {token!r}")
        if len(token) > 1 and token.startswith("0"):
            raise ValueError(f"split vertex token must not have leading zeros: {token!r}")
        v = int(token)
        if not (0 <= v < n):
            raise ValueError(f"split vertex out of range: {v}")
        if v in out:
            raise ValueError(f"duplicate split vertex: {v}")
        out.append(v)
    return out


def parse_coord_table(path: Path, max_vertices: int) -> list[tuple[Fraction, Fraction]]:
    rows: dict[int, tuple[Fraction, Fraction]] = {}
    with path.open("r", encoding="ascii", newline="") as f:
        first = f.readline()
        f.seek(0)
        has_header = "id" in first.lower()
        if has_header:
            reader = csv.DictReader(f)
            for lineno, row in enumerate(reader, 2):
                if row is None:
                    continue
                try:
                    add_coord(path, lineno, rows, row["id"], row["x"], row["y"], max_vertices)
                except KeyError as exc:
                    raise ValueError(f"{path}:{lineno}: expected id,x,y header") from exc
        else:
            reader = csv.reader(f)
            for lineno, row in enumerate(reader, 1):
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) != 3:
                    raise ValueError(f"{path}:{lineno}: expected id,x,y row")
                add_coord(path, lineno, rows, row[0], row[1], row[2], max_vertices)
    if not rows:
        raise ValueError(f"{path}: coordinate table is empty")
    n = max(rows) + 1
    missing = [str(i) for i in range(n) if i not in rows]
    if missing:
        raise ValueError(
            f"{path}: vertex ids must be consecutive from 0; "
            f"missing coordinate rows for vertices {','.join(missing)}"
        )
    coords = [rows[i] for i in range(n)]
    seen: dict[tuple[Fraction, Fraction], int] = {}
    for i, xy in enumerate(coords):
        if xy in seen:
            raise ValueError(f"duplicate coordinates {xy} for vertices {seen[xy]} and {i}")
        seen[xy] = i
    return coords


def add_coord(
    path: Path,
    lineno: int,
    rows: dict[int, tuple[Fraction, Fraction]],
    raw_id: str,
    raw_x: str,
    raw_y: str,
    max_vertices: int,
) -> None:
    raw_id = raw_id.strip()
    if not re.fullmatch(r"[0-9]+", raw_id):
        raise ValueError(f"{path}:{lineno}: bad vertex id {raw_id!r}")
    if len(raw_id) > 1 and raw_id.startswith("0"):
        raise ValueError(f"{path}:{lineno}: vertex id must not have leading zeros: {raw_id!r}")
    vid = int(raw_id)
    if vid >= max_vertices:
        raise ValueError(
            f"{path}:{lineno}: vertex id {vid} exceeds --max-vertices limit {max_vertices}"
        )
    if vid in rows:
        raise ValueError(f"{path}:{lineno}: duplicate vertex id: {vid}")
    rows[vid] = (parse_rat(raw_x), parse_rat(raw_y))


def unit_edges(coords: list[tuple[Fraction, Fraction]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = dist2(coords[i], coords[j])
            if not isinstance(d, Fraction):
                raise TypeError("dist2 must return fractions.Fraction for exact rational derivation")
            if d == UNIT_DIST_SQUARED:
                edges.append((i, j))
    return edges


def require_split_vertex_degree(
    split_vertices: list[int],
    n: int,
    edges: list[tuple[int, int]],
    min_degree: int,
) -> None:
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    for v in split_vertices:
        if v < 0 or v >= n:
            raise ValueError(f"split vertex out of range: {v}")
        if degrees[v] < min_degree:
            raise ValueError(
                f"split vertex {v} has degree {degrees[v]}, "
                f"below required minimum {min_degree}"
            )


def write_edge_file(path: Path, n: int, edges: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write(f"p edge {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u + 1} {v + 1}\n")


def run_validator(source_json: Path, out_path: Path) -> None:
    validator = Path(__file__).with_name("validate_chi6_solver_candidate_package.py")
    proc = subprocess.run(
        [sys.executable, str(validator), str(source_json)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="ascii",
    )
    out_path.write_text(proc.stdout, encoding="ascii")
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"source validator rejected generated package with exit {proc.returncode}{suffix}"
        )
    if "status=VALID_SOLVER_CANDIDATE_PACKAGE" not in proc.stdout.splitlines():
        raise RuntimeError("source validator returned success without the v1 valid-package status")


def validate_candidate_id(candidate_id: str) -> None:
    if "/" in candidate_id or "\\" in candidate_id:
        raise ValueError("candidate-id must not contain path separators")
    if not ID_RE.fullmatch(candidate_id):
        raise ValueError("candidate-id must use only letters, digits, '.', '_', or '-'")
    if candidate_id == "." or ".." in candidate_id:
        raise ValueError("candidate-id must not be '.' or contain '..'")


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def render_producer_command(args: argparse.Namespace) -> str:
    return shell_join(
        [
            "make_chi6_rational_unit_graph_source_package.py",
            "--min-edges",
            str(args.min_edges),
            "--max-vertices",
            str(args.max_vertices),
            "--min-split-degree",
            str(args.min_split_degree),
            str(args.coords_csv),
            str(args.candidate_id),
            str(args.split_vertices),
            str(args.out_dir),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_csv", type=Path)
    parser.add_argument("candidate_id")
    parser.add_argument("split_vertices")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--max-vertices", type=int, default=DEFAULT_MAX_VERTICES)
    parser.add_argument("--min-split-degree", type=int, default=DEFAULT_MIN_SPLIT_DEGREE)
    args = parser.parse_args()

    try:
        if not args.coords_csv.is_file():
            raise ValueError(f"missing coordinate CSV: {args.coords_csv}")
        validate_candidate_id(args.candidate_id)
        if args.min_edges < 1:
            raise ValueError("--min-edges must be positive")
        if args.max_vertices < 2:
            raise ValueError("--max-vertices must be at least 2")
        if args.min_split_degree < 1:
            raise ValueError("--min-split-degree must be positive")

        coords = parse_coord_table(args.coords_csv, args.max_vertices)
        n = len(coords)
        edges = unit_edges(coords)
        if len(edges) < args.min_edges:
            raise ValueError(
                f"derived unit graph has {len(edges)} edges, below required minimum {args.min_edges}"
            )
        split_vertices = parse_split_vertices(args.split_vertices, n)
        require_split_vertex_degree(split_vertices, n, edges, args.min_split_degree)
        validate_geometry(coords, edges)

        package_dir = args.out_dir / "package"
        package_dir.mkdir(parents=True, exist_ok=True)
        edge_path = package_dir / f"{args.candidate_id}.edge"
        coords_path = package_dir / f"{args.candidate_id}.coords.csv"
        source_json = args.out_dir / f"{args.candidate_id}.candidate-source.json"
        validator_out = args.out_dir / "source_validator.out"

        write_edge_file(edge_path, n, edges)
        shutil.copyfile(args.coords_csv, coords_path)
        meta = {
            "schema": "chi6_solver_candidate_package.v1",
            "candidate_id": args.candidate_id,
            "edge_path": f"package/{args.candidate_id}.edge",
            "edge_sha256": sha256_file(edge_path),
            "coords_path": f"package/{args.candidate_id}.coords.csv",
            "coords_sha256": sha256_file(coords_path),
            "coordinate_domain": "rational_xy",
            "n": n,
            "m": len(edges),
            "k": 5,
            "split_vertices": split_vertices,
            "producer_command": render_producer_command(args),
            "claim_scope": "solver_candidate_source_only",
            "promotion_gate": "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge",
        }
        source_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
        run_validator(source_json, validator_out)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_rational_unit_graph_source_package v1")
    print("trust_boundary=exact_rational_geometry_handoff__lrat_lean_required")
    print(f"candidate_id={args.candidate_id}")
    print(f"n={n}")
    print(f"m={len(edges)}")
    print("k=5")
    print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
    print("split_vertex_indexing=zero_based")
    print("edge_vertex_indexing=one_based_dimacs")
    print(f"split_vertex_min_degree={args.min_split_degree}")
    print(f"max_vertices={args.max_vertices}")
    print(f"coordinate_source={args.coords_csv}")
    print(f"coordinate_source_sha256={sha256_file(args.coords_csv)}")
    print(f"candidate_source={source_json}")
    print(f"edge={edge_path}")
    print(f"edge_sha256={meta['edge_sha256']}")
    print(f"coords={coords_path}")
    print(f"coords_sha256={meta['coords_sha256']}")
    print(f"source_validator={validator_out}")
    print("edge_derivation=all_pairs_exact_rational_dist2_eq_1")
    print("claim_scope=solver_candidate_source_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("promotable=0")
    print("status=FORMAT_VALID_RATIONAL_UNIT_GRAPH_SOURCE_PACKAGE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
