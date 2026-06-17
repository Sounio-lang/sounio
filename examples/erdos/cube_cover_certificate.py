#!/usr/bin/env python3
"""Emit small cube-cover certificates for colourCNF split leaves.

The first supported cover is deliberately tiny: split one vertex into all `k`
positive colour literals. The next supported family is the product cover from
splitting a finite list of vertices over all `k` colours. Both cover arguments
are tied to the `colourCNF` shape in `SounioSatColouringBridge`: each vertex
contributes one at-least-one colour clause, followed by `k` edge clauses per
graph edge. This does not assume at-most-one colours, so leaves may overlap. The
emitted certificate is still unpromotable until composed with checked per-cube
LRAT proofs in Lean.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from cube_sieve_batch_manifest import parse_batch, sha256_file
from cube_sieve_propagation_manifest import parse_edge_file


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_kv_lines(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    top: dict[str, str] = {}
    cubes: list[dict[str, str]] = []
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "cube_sieve_refute_batch v1":
        raise ValueError(f"{path}: expected cube_sieve_refute_batch v1 header")
    for lineno, line in enumerate(lines[1:], 2):
        if not line:
            continue
        if line.startswith("cube "):
            row: dict[str, str] = {}
            for token in line.split()[1:]:
                if "=" not in token:
                    raise ValueError(f"{path}:{lineno}: bad cube token {token!r}")
                k, v = token.split("=", 1)
                row[k] = v
            cubes.append(row)
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            top[k] = v
            continue
        raise ValueError(f"{path}:{lineno}: unexpected line {line!r}")
    return top, cubes


def checked_sha(label: str, value: str) -> None:
    if not HEX64_RE.fullmatch(value):
        raise ValueError(f"{label} is not a sha256 hex digest: {value!r}")


def checked_file_sha(root: Path, rel: str, expected: str, label: str) -> None:
    checked_sha(label, expected)
    path = root / rel
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: got {actual}, expected {expected}")


def lrat_has_empty_clause(path: Path) -> bool:
    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if re.match(r"^\d+\s+0(\s+|$)", line):
                return True
    return False


def assignment_string(cube: list[tuple[int, int]]) -> str:
    return ",".join(f"{v}:{c}" for v, c in cube)


def product_colour_tuples(k: int, depth: int) -> list[tuple[int, ...]]:
    if depth == 0:
        return [()]
    tails = product_colour_tuples(k, depth - 1)
    return [(c, *tail) for c in range(k) for tail in tails]


def split_product_order(split_vertices: list[int], k: int) -> list[list[tuple[int, int]]]:
    return [
        list(zip(split_vertices, colours, strict=True))
        for colours in product_colour_tuples(k, len(split_vertices))
    ]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_batch", type=Path)
    parser.add_argument("refutation_batch", type=Path)
    parser.add_argument(
        "--cover-rule",
        choices=("single_vertex_atleast_one_split", "split_vertices_atleast_one_product"),
        default="single_vertex_atleast_one_split",
    )
    parser.add_argument("--split-vertex", type=int, default=0)
    parser.add_argument(
        "--split-vertices",
        default="",
        help="comma-separated zero-based vertices for split_vertices_atleast_one_product",
    )
    args = parser.parse_args()

    try:
        n, m, _edges = parse_edge_file(args.edge_file)
        cubes = parse_batch(args.cube_batch)
        summary, rows = parse_kv_lines(args.refutation_batch)
        if args.k <= 0:
            raise ValueError("k must be positive")
        if not (0 <= args.split_vertex < n):
            raise ValueError(f"split vertex out of range: {args.split_vertex}")
        if summary.get("formula_kind") != "colourCNF":
            raise ValueError("only formula_kind=colourCNF is supported")
        if summary.get("n") != str(n):
            raise ValueError("refutation summary n mismatch")
        if summary.get("m") != str(m):
            raise ValueError("refutation summary m mismatch")
        if summary.get("expected_vars") != str(n * args.k):
            raise ValueError("refutation summary expected_vars mismatch")
        if summary.get("base_clause_count") != str(n + m * args.k):
            raise ValueError("refutation summary base_clause_count mismatch")
        if summary.get("edge_sha256") != sha256_file(args.edge_file):
            raise ValueError("refutation summary edge_sha256 mismatch")
        if summary.get("cube_batch_sha256") != sha256_file(args.cube_batch):
            raise ValueError("refutation summary cube_batch_sha256 mismatch")
        if summary.get("k") != str(args.k):
            raise ValueError("refutation summary k mismatch")
        if summary.get("failed_count") != "0":
            raise ValueError("refutation summary has failed cube subproblems")
        if summary.get("sb_mode") != "0":
            raise ValueError("cover certificate currently requires sb_mode=0")
        if summary.get("promotable") != "0":
            raise ValueError("cover input must remain unpromotable")
        if "out_dir" not in summary or not summary["out_dir"]:
            raise ValueError("refutation summary out_dir is missing")
        refute_root = Path(summary["out_dir"])
        if not refute_root.is_dir():
            raise ValueError("refutation summary out_dir is missing or not a directory")

        if args.cover_rule == "single_vertex_atleast_one_split":
            split_vertices = [args.split_vertex]
            expected_cubes = [[(args.split_vertex, c)] for c in range(args.k)]
            if len(cubes) != args.k:
                raise ValueError(f"single-vertex cover needs exactly k={args.k} cubes")
        else:
            if args.split_vertices:
                split_vertices = parse_split_vertices(args.split_vertices, n)
            else:
                if not cubes or not cubes[0][1]:
                    raise ValueError("split product cover needs at least one split vertex")
                split_vertices = [v for v, _c in cubes[0][1]]
                parse_split_vertices(",".join(str(v) for v in split_vertices), n)
            expected_cubes = split_product_order(split_vertices, args.k)
            if len(cubes) != len(expected_cubes):
                raise ValueError(
                    f"split product cover needs exactly {len(expected_cubes)} cubes, got {len(cubes)}"
                )
        if len(rows) != len(expected_cubes):
            raise ValueError(
                f"refutation summary has {len(rows)} cube rows, expected {len(expected_cubes)}"
            )

        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("duplicate cube ids in refutation summary")
        by_assignment: dict[str, tuple[str, list[tuple[int, int]]]] = {}
        for cube_id, cube in cubes:
            vertices = [v for v, _c in cube]
            if vertices != split_vertices:
                raise ValueError(f"cube {cube_id} has vertices {vertices}, expected {split_vertices}")
            for _v, c in cube:
                if not (0 <= c < args.k):
                    raise ValueError(f"cube {cube_id} colour out of range: {c}")
            assignment = assignment_string(cube)
            if assignment in by_assignment:
                raise ValueError(f"duplicate leaf assignment: {assignment}")
            by_assignment[assignment] = (cube_id, cube)
        expected_assignments = [assignment_string(cube) for cube in expected_cubes]
        missing = [assn for assn in expected_assignments if assn not in by_assignment]
        if missing:
            raise ValueError("missing split-product leaves: " + ",".join(missing[:10]))

        leaves: list[tuple[int, str, str, int, str, str, str]] = []
        for index, assignment in enumerate(expected_assignments):
            cube_id, cube = by_assignment[assignment]
            row = by_id.get(cube_id)
            if row is None:
                raise ValueError(f"missing refutation row for cube {cube_id}")
            if row.get("assignments") != assignment:
                raise ValueError(f"refutation row for {cube_id} has wrong assignments")
            if row.get("cube_assignment_count") != str(len(cube)):
                raise ValueError(f"refutation row for {cube_id} has wrong cube_assignment_count")
            if row.get("drat_deletions") != "0":
                raise ValueError(f"refutation row for {cube_id} has deletion records")
            if row.get("cnf_clauses") != row.get("expected_cnf_clauses"):
                raise ValueError(f"refutation row for {cube_id} has unvalidated CNF clause count")
            checked_file_sha(refute_root, row.get("cube", ""), row.get("cube_sha256", ""), f"{cube_id}.cube_sha256")
            checked_file_sha(refute_root, row.get("cnf", ""), row.get("cnf_sha256", ""), f"{cube_id}.cnf_sha256")
            checked_file_sha(refute_root, row.get("lrat", ""), row.get("lrat_sha256", ""), f"{cube_id}.lrat_sha256")
            lrat_path = refute_root / row["lrat"]
            if not lrat_has_empty_clause(lrat_path):
                raise ValueError(f"LRAT artifact for {cube_id} has no empty-clause row")
            leaves.append(
                (
                    index,
                    cube_id,
                    assignment,
                    len(cube),
                    row["cube_sha256"],
                    row["cnf_sha256"],
                    row["lrat_sha256"],
                )
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("cube_cover_certificate v1")
    print("trust_boundary=cover_checker_untrusted__lean_composition_required_for_promotion")
    print(f"cover_rule={args.cover_rule}")
    print("formula_kind=colourCNF")
    print(f"edge_path={args.edge_file}")
    print(f"edge_sha256={sha256_file(args.edge_file)}")
    print(f"n={n}")
    print(f"m={m}")
    print(f"k={args.k}")
    print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
    if args.cover_rule == "single_vertex_atleast_one_split":
        print(f"split_vertex={args.split_vertex}")
        print("base_clause=atleast_one_colour_for_split_vertex")
    else:
        print("base_clause=atleast_one_colour_for_each_split_vertex")
    print(f"cube_batch_path={args.cube_batch}")
    print(f"cube_batch_sha256={sha256_file(args.cube_batch)}")
    print(f"refutation_batch_path={args.refutation_batch}")
    print(f"refutation_batch_sha256={sha256_file(args.refutation_batch)}")
    print(f"leaf_count={len(leaves)}")
    print(f"covered_cube_count={len(leaves)}")
    print(f"lrat_artifact_count={len(leaves)}")
    if args.cover_rule == "single_vertex_atleast_one_split":
        print("cover_complete_for_split_vertex=1")
        print("cover_claim=atleast_one_cover_for_split_vertex")
        print("promotion_gate=REJECT_LEAN_CHECKED_LEAF_UNSAT_NOT_ATTACHED")
    else:
        print("cover_complete_for_split_vertices=1")
        print("cover_claim=atleast_one_product_cover_for_split_vertices")
        print("lean_cover_obligation=CubeCover n k edges splitVerticesCubes")
        print("promotion_gate=REJECT_LEAN_CUBECOVER_PROOF_NOT_ATTACHED")
    print("verified_claim=none")
    print("global_unsat_claim=none")
    print("geometry_claim=none")
    print("promotable=0")
    for index, cube_id, assignment, assignment_count, cube_sha, cnf_sha, lrat_sha in leaves:
        if args.cover_rule == "single_vertex_atleast_one_split":
            colour = assignment.split(":", 1)[1]
            print(
                "leaf "
                f"index={index} colour={colour} cube_id={cube_id} "
                f"assignment={assignment} cube_sha256={cube_sha} "
                f"lrat_sha256={lrat_sha}"
            )
        else:
            print(
                "leaf "
                f"index={index} cube_id={cube_id} assignment_count={assignment_count} "
                f"assignments={assignment} cube_sha256={cube_sha} cnf_sha256={cnf_sha} "
                f"lrat_sha256={lrat_sha}"
            )
    print("status=cover_certificate_emitted_unpromotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
