#!/usr/bin/env python3
"""Generate a Lean smoke module for cube-cover refutations.

This is deliberately narrow proof plumbing for the no-5-colouring SAT component
of the chi>=6 search lane. It consumes the existing cube refutation batch summary,
rechecks the cube leaves, embeds each leaf LRAT as a String, and emits a Lean
file where the leaf `Unsat` proofs are composed either with the legacy
`SounioSatCubeCover.unsat_of_split_vertex5` theorem or with the generic
`SounioSatCubeCover.unsat_of_cube_cover` theorem.

The generic composition mode supports product covers obtained by splitting a
finite list of vertices over all colours. This is more general than the first
single-vertex K6 smoke, but it is still a checked cover family, not an arbitrary
unverified cube list.

The arbitrary composition mode accepts any finite positive colour-cube list,
provided the producer also supplies a Lean-checkable LRAT refutation of the
complement-cover CNF (`base ∧ ⋀ cube, block(cube)`). That complement proof is
what Lean trusts as the `CubeCover` certificate.

The output is a calibration module, not a Euclidean chromatic-number witness.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path

from cube_sieve_batch_manifest import parse_batch, sha256_file
from cube_sieve_propagation_manifest import parse_edge_file


@dataclass(frozen=True)
class Leaf:
    index: int
    cube_id: str
    cube: list[tuple[int, int]]
    cnf_path: Path
    lrat_path: Path


def parse_cnf_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if not (len(parts) == 4 and parts[:2] == ["p", "cnf"]):
                raise RuntimeError(f"{path}: expected p cnf header, got {line!r}")
            return int(parts[2]), int(parts[3])
    raise RuntimeError(f"{path}: missing p cnf header")


def parse_key_value_line(line: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    if not parts:
        return "", {}
    fields: dict[str, str] = {}
    for tok in parts[1:]:
        if "=" not in tok:
            raise RuntimeError(f"malformed field in summary row: {tok!r}")
        key, value = tok.split("=", 1)
        fields[key] = value
    return parts[0], fields


def parse_refute_summary(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    meta: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "cube_sieve_refute_batch v1":
                meta["format"] = line
                continue
            if line.startswith("cube "):
                tag, fields = parse_key_value_line(line)
                if tag != "cube":
                    raise RuntimeError(f"bad cube row: {line}")
                rows.append(fields)
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                meta[key] = value
                continue
            raise RuntimeError(f"{path}: unknown summary line: {line!r}")
    return meta, rows


def parse_lrat_line(line: str) -> tuple[int, bool, list[str], list[int]]:
    line = line.strip()
    toks = line.strip().split()
    if not toks:
        raise ValueError("empty")
    old_id = int(toks[0])
    if len(toks) >= 2 and toks[1] == "d":
        ids: list[int] = []
        for tok in toks[2:]:
            val = int(tok)
            if val == 0:
                break
            ids.append(val)
        return old_id, True, [], ids
    lits: list[str] = []
    i = 1
    while i < len(toks):
        if int(toks[i]) == 0:
            break
        lits.append(toks[i])
        i += 1
    i += 1
    hints: list[int] = []
    while i < len(toks):
        val = int(toks[i])
        if val == 0:
            break
        hints.append(val)
        i += 1
    return old_id, False, lits, hints


def renumber_lrat(path: Path, original_clause_count: int) -> str:
    mapping: dict[int, int] = {}
    addition_count = 0
    out: list[str] = []

    def remap_ref(ref: int) -> int:
        if ref <= original_clause_count:
            return ref
        if ref not in mapping:
            raise RuntimeError(f"{path}: LRAT reference {ref} appears before its addition")
        return mapping[ref]

    with path.open("r", encoding="ascii") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            old_id, deletion, lits, hints = parse_lrat_line(line)
            if deletion:
                mapped = [str(remap_ref(h)) for h in hints]
                delete_id = remap_ref(old_id) if old_id > original_clause_count else old_id
                out.append(f"{delete_id} d {' '.join(mapped)} 0")
                continue
            if old_id <= original_clause_count:
                raise RuntimeError(f"{path}: LRAT addition id {old_id} overlaps original clauses")
            if old_id in mapping:
                raise RuntimeError(f"{path}: duplicate LRAT addition id {old_id}")
            addition_count += 1
            new_id = original_clause_count + addition_count
            mapping[old_id] = new_id
            mapped_hints = [str(remap_ref(h)) for h in hints]
            lit_part = " ".join(lits)
            hint_part = " ".join(mapped_hints)
            if lit_part and hint_part:
                out.append(f"{new_id} {lit_part} 0 {hint_part} 0")
            elif lit_part:
                out.append(f"{new_id} {lit_part} 0 0")
            elif hint_part:
                out.append(f"{new_id} 0 {hint_part} 0")
            else:
                out.append(f"{new_id} 0 0")
    return "\n".join(out) + "\n"


def lean_name(raw: str) -> str:
    reserved = {
        "def", "theorem", "lemma", "example", "import", "open", "namespace",
        "section", "end", "by", "let", "have", "show", "if", "then", "else",
        "match", "with", "fun", "forall", "exists", "true", "false", "not",
    }
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not name or name[0].isdigit():
        name = f"g_{name}"
    if name in reserved:
        name = f"{name}_generated"
    return name


def rel_or_abs(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def assignment_string(cube: list[tuple[int, int]]) -> str:
    return ",".join(f"{v}:{c}" for v, c in cube)


def lean_cube_term(cube: list[tuple[int, int]], k: int) -> str:
    if not cube:
        return "[]"
    return "[" + ", ".join(f"({v} * {k} + {c}, true)" for v, c in cube) + "]"


def lean_nat_list(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def lean_cube_list_term(cubes: list[list[tuple[int, int]]], k: int) -> str:
    if not cubes:
        return "[]"
    return "[\n" + ",\n".join(f"  {lean_cube_term(cube, k)}" for cube in cubes) + "\n]"


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


def validate_single_vertex_split(
    *,
    cubes: list[tuple[str, list[tuple[int, int]]]],
    k: int,
    n: int,
) -> tuple[int, list[tuple[str, list[tuple[int, int]]]]]:
    if len(cubes) != k:
        raise RuntimeError(f"expected exactly {k} cube leaves")
    split_vertices = {cube[0][0] for _cube_id, cube in cubes if len(cube) == 1}
    if len(split_vertices) != 1:
        raise RuntimeError("all cube leaves must be one-literal cubes over one split vertex")
    split_vertex = next(iter(split_vertices))
    if not (0 <= split_vertex < n):
        raise RuntimeError(f"split vertex out of range: {split_vertex}")

    by_assignment: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    for cube_id, cube in cubes:
        if len(cube) != 1:
            raise RuntimeError(f"cube {cube_id} is not one-literal")
        v, c = cube[0]
        if v != split_vertex:
            raise RuntimeError(f"cube {cube_id} splits vertex {v}, expected {split_vertex}")
        if not (0 <= c < k):
            raise RuntimeError(f"cube {cube_id} colour out of range: {c}")
        assignment = f"{v}:{c}"
        if assignment in by_assignment:
            raise RuntimeError(f"duplicate split leaf assignment: {assignment}")
        by_assignment[assignment] = (cube_id, cube)
    expected = [f"{split_vertex}:{colour}" for colour in range(k)]
    missing = [assn for assn in expected if assn not in by_assignment]
    if missing:
        raise RuntimeError(f"missing split leaves: {','.join(missing)}")
    return split_vertex, [by_assignment[assn] for assn in expected]


def validate_split_product(
    *,
    cubes: list[tuple[str, list[tuple[int, int]]]],
    k: int,
    n: int,
) -> tuple[list[int], list[tuple[str, list[tuple[int, int]]]]]:
    if not cubes:
        raise RuntimeError("expected at least one cube leaf")
    split_vertices = [v for v, _c in cubes[0][1]]
    if not split_vertices:
        raise RuntimeError("split product cover needs at least one split vertex")
    if len(set(split_vertices)) != len(split_vertices):
        raise RuntimeError("split product cover has duplicate split vertices")
    for v in split_vertices:
        if not (0 <= v < n):
            raise RuntimeError(f"split vertex out of range: {v}")

    by_assignment: dict[str, tuple[str, list[tuple[int, int]]]] = {}
    for cube_id, cube in cubes:
        vertices = [v for v, _c in cube]
        if vertices != split_vertices:
            raise RuntimeError(
                f"cube {cube_id} has vertices {vertices}, expected {split_vertices}"
            )
        for _v, c in cube:
            if not (0 <= c < k):
                raise RuntimeError(f"cube {cube_id} colour out of range: {c}")
        assignment = assignment_string(cube)
        if assignment in by_assignment:
            raise RuntimeError(f"duplicate split-product leaf assignment: {assignment}")
        by_assignment[assignment] = (cube_id, cube)

    expected_cubes = split_product_order(split_vertices, k)
    expected_assignments = [assignment_string(cube) for cube in expected_cubes]
    if len(cubes) != len(expected_assignments):
        raise RuntimeError(
            f"split product cover needs exactly {len(expected_assignments)} cubes, got {len(cubes)}"
        )
    missing = [assn for assn in expected_assignments if assn not in by_assignment]
    if missing:
        raise RuntimeError("missing split-product leaves: " + ";".join(missing[:10]))
    return split_vertices, [by_assignment[assn] for assn in expected_assignments]


def validate_arbitrary_cubes(
    *,
    cubes: list[tuple[str, list[tuple[int, int]]]],
    k: int,
    n: int,
) -> tuple[list[int], list[tuple[str, list[tuple[int, int]]]]]:
    if not cubes:
        raise RuntimeError("arbitrary cube cover needs at least one cube leaf")
    seen_assignments: set[str] = set()
    touched_vertices: list[int] = []
    for cube_id, cube in cubes:
        assignment = assignment_string(cube)
        if assignment in seen_assignments:
            raise RuntimeError(f"duplicate arbitrary cube assignment: {assignment}")
        seen_assignments.add(assignment)
        for v, c in cube:
            if not (0 <= v < n):
                raise RuntimeError(f"cube {cube_id} vertex out of range: {v}")
            if not (0 <= c < k):
                raise RuntimeError(f"cube {cube_id} colour out of range: {c}")
            if v not in touched_vertices:
                touched_vertices.append(v)
    return touched_vertices, cubes


def emit_membership_cases(lines: list[str], theorem_names: list[str], indent: str) -> None:
    if not theorem_names:
        lines.append(f"{indent}cases hcube")
        return
    lines.append(f"{indent}cases hcube with")
    lines.append(f"{indent}| head =>")
    lines.append(f"{indent}    exact {theorem_names[0]}")
    lines.append(f"{indent}| tail _ hcube =>")
    emit_membership_cases(lines, theorem_names[1:], indent + "    ")


def emit_lean(
    *,
    module_name: str,
    prefix: str,
    edge_file: Path,
    n: int,
    k: int,
    edges: list[tuple[int, int]],
    split_vertex: int,
    split_vertices: list[int],
    leaves: list[Leaf],
    composition: str,
    cover_lrat_text: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append("/-")
    lines.append(f"{module_name} - autogenerated by examples/erdos/gen_lean_cube_cover_reflect.py.")
    lines.append("")
    lines.append("Cube-unit LRAT leaves are checked by Lean core and then composed")
    if composition == "arbitrary":
        lines.append(
            "with SounioSatCubeCover.unsat_of_cube_cover via a complement-LRAT CubeCover proof."
        )
        lines.append("Arbitrary cover mode: the proof path does not assume a split product.")
        lines.append("Lean checks the complement LRAT against cubeCoverComplementCNF.")
    elif composition == "generic":
        lines.append("with SounioSatCubeCover.unsat_of_cube_cover via an explicit CubeCover proof.")
    else:
        lines.append("with SounioSatCubeCover.unsat_of_split_vertex5.")
    lines.append("This is a finite SAT calibration artifact, not a Euclidean chi >= 6 witness.")
    lines.append("The LRAT checks use Std.Tactic.BVDecide.LRAT.check via native_decide,")
    lines.append("matching the repo-local reflected LRAT modules.")
    lines.append("Python generation is not trusted; the gate must run this file through Lean.")
    lines.append(
        f"graph: {edge_file.name}, n={n}, k={k}, split_vertices={','.join(str(v) for v in split_vertices)}"
    )
    lines.append("-/")
    lines.append("import SounioSatCubeCover")
    lines.append("import SounioSatReflect")
    lines.append("")
    lines.append("open Std.Sat")
    lines.append("open Std.Tactic.BVDecide.LRAT")
    lines.append("open SounioSatColouring")
    lines.append("open SounioSatCubeCover")
    lines.append("open SounioSatReflect")
    lines.append("")
    lines.append("set_option maxRecDepth 1000000")
    lines.append("set_option maxHeartbeats 0")
    lines.append("")
    lines.append(f"/-- Edge list (0-based), generated from `{edge_file.name}`. -/")
    lines.append(f"def {prefix}_edges : List (Prod Nat Nat) := [")
    for u, v in edges:
        lines.append(f"  ({u}, {v}),")
    lines.append("]")
    lines.append("")

    generic_theorems: list[str] = []
    for leaf in leaves:
        cnf_path = leaf.cnf_path
        lrat_path = leaf.lrat_path
        cnf_vars, cnf_clauses = parse_cnf_header(cnf_path)
        lrat_text = renumber_lrat(lrat_path, cnf_clauses)
        assignment = assignment_string(leaf.cube)
        lines.append(f"/-- Cube leaf `{assignment}` CNF, vars={cnf_vars}, clauses={cnf_clauses}. -/")
        if composition in {"generic", "arbitrary"}:
            leaf_name = f"{prefix}_leaf{leaf.index}"
            cube_term = lean_cube_term(leaf.cube, k)
            cnf_term = (
                f"SounioSatCubeCover.colourCNFWithCube {n} {k} {prefix}_edges "
                f"{cube_term}"
            )
        else:
            colour = leaf.cube[0][1]
            leaf_name = f"{prefix}_v{split_vertex}_c{colour}"
            cnf_term = (
                f"SounioSatCubeCover.colourCNFWithUnit {n} {k} {prefix}_edges {split_vertex} {colour}"
            )
        lines.append(
            f"def {leaf_name}_cnf : CNF Nat := {cnf_term}"
        )
        lines.append("")
        lines.append(f"def {leaf_name}_lrat : String := \"")
        lines.append(lrat_text.rstrip("\n"))
        lines.append("\"")
        lines.append("")
        lines.append(
            f"theorem {leaf_name}_check : "
            f"Std.Tactic.BVDecide.LRAT.check (parseLRAT {leaf_name}_lrat) "
            f"{leaf_name}_cnf = true := by native_decide"
        )
        lines.append("")
        lines.append(
            f"theorem {leaf_name}_unsat : "
            f"({cnf_term}).Unsat := "
            f"Std.Tactic.BVDecide.LRAT.check_sound _ {leaf_name}_cnf "
            f"{leaf_name}_check"
        )
        lines.append("")
        if composition in {"generic", "arbitrary"}:
            generic_theorems.append(f"{leaf_name}_unsat")

    if composition == "arbitrary":
        if cover_lrat_text is None:
            raise RuntimeError("internal error: arbitrary composition requires cover LRAT text")
        lines.append(f"def {prefix}_cubes : List SounioSatCubeCover.Cube :=")
        lines.append(lean_cube_list_term([leaf.cube for leaf in leaves], k))
        lines.append("")
        lines.append(f"def {prefix}_cover_complement_lrat : String := \"")
        lines.append(cover_lrat_text.rstrip("\n"))
        lines.append("\"")
        lines.append("")
        lines.append(f"theorem {prefix}_cover_complement_check :")
        lines.append(
            f"    Std.Tactic.BVDecide.LRAT.check (parseLRAT {prefix}_cover_complement_lrat)"
        )
        lines.append(
            f"      (SounioSatCubeCover.cubeCoverComplementCNF {n} {k} {prefix}_edges {prefix}_cubes)"
        )
        lines.append("      = true := by native_decide")
        lines.append("")
        lines.append(f"theorem {prefix}_cube_cover :")
        lines.append(f"    SounioSatCubeCover.CubeCover {n} {k} {prefix}_edges {prefix}_cubes :=")
        lines.append("  SounioSatCubeCover.cube_cover_of_complement_unsat")
        lines.append(
            f"    (Std.Tactic.BVDecide.LRAT.check_sound _ "
            f"(SounioSatCubeCover.cubeCoverComplementCNF {n} {k} {prefix}_edges {prefix}_cubes)"
        )
        lines.append(f"      {prefix}_cover_complement_check)")
        lines.append("")
        lines.append(f"theorem {prefix}_cube_unsat :")
        lines.append(f"    forall cube, List.Mem cube {prefix}_cubes ->")
        lines.append(f"      (SounioSatCubeCover.colourCNFWithCube {n} {k} {prefix}_edges cube).Unsat := by")
        lines.append("  intro cube hcube")
        lines.append(f"  simp only [{prefix}_cubes] at hcube")
        emit_membership_cases(lines, generic_theorems, "  ")
        lines.append("")
        lines.append(f"theorem {prefix}_unsat_from_arbitrary_cube_cover :")
        lines.append(f"    (colourCNF {n} {k} {prefix}_edges).Unsat :=")
        lines.append("  SounioSatCubeCover.unsat_of_cube_cover")
        lines.append(f"    (n := {n}) (k := {k}) (edges := {prefix}_edges) (cubes := {prefix}_cubes)")
        lines.append(f"    {prefix}_cube_cover")
        lines.append(f"    {prefix}_cube_unsat")
        lines.append("")
        lines.append(f"#print axioms {prefix}_unsat_from_arbitrary_cube_cover")
    elif composition == "generic":
        lines.append(f"def {prefix}_cubes : List SounioSatCubeCover.Cube :=")
        lines.append(
            f"  SounioSatCubeCover.splitVerticesCubes {k} {lean_nat_list(split_vertices)}"
        )
        lines.append("")
        lines.append(f"theorem {prefix}_cube_cover :")
        lines.append(f"    SounioSatCubeCover.CubeCover {n} {k} {prefix}_edges {prefix}_cubes :=")
        lines.append("  SounioSatCubeCover.split_vertices_cubes_cover")
        lines.append(
            f"    (n := {n}) (k := {k}) (edges := {prefix}_edges) "
            f"(vs := {lean_nat_list(split_vertices)}) (by decide)"
        )
        lines.append("")
        lines.append(f"theorem {prefix}_cube_unsat :")
        lines.append(f"    forall cube, List.Mem cube {prefix}_cubes ->")
        lines.append(f"      (SounioSatCubeCover.colourCNFWithCube {n} {k} {prefix}_edges cube).Unsat := by")
        lines.append("  intro cube hcube")
        lines.append(
            f"  simp only [{prefix}_cubes, SounioSatCubeCover.splitVerticesCubes] at hcube"
        )
        emit_membership_cases(lines, generic_theorems, "  ")
        lines.append("")
        lines.append(f"theorem {prefix}_unsat_from_generic_cube_cover :")
        lines.append(f"    (colourCNF {n} {k} {prefix}_edges).Unsat :=")
        lines.append("  SounioSatCubeCover.unsat_of_cube_cover")
        lines.append(f"    (n := {n}) (k := {k}) (edges := {prefix}_edges) (cubes := {prefix}_cubes)")
        lines.append(f"    {prefix}_cube_cover")
        lines.append(f"    {prefix}_cube_unsat")
        lines.append("")
        lines.append(f"#print axioms {prefix}_unsat_from_generic_cube_cover")
    else:
        lines.append(f"theorem {prefix}_unsat_from_v{split_vertex}_split :")
        lines.append(f"    (colourCNF {n} {k} {prefix}_edges).Unsat :=")
        lines.append("  SounioSatCubeCover.unsat_of_split_vertex5")
        lines.append(f"    (n := {n}) (edges := {prefix}_edges) (v := {split_vertex})")
        lines.append("    (by decide)")
        for leaf in leaves:
            colour = leaf.cube[0][1]
            lines.append(f"    {prefix}_v{split_vertex}_c{colour}_unsat")
        lines.append("")
        lines.append(f"#print axioms {prefix}_unsat_from_v{split_vertex}_split")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int, help="must be 5; this generator targets no-5-colouring")
    parser.add_argument("cube_batch", type=Path)
    parser.add_argument("refute_summary", type=Path)
    parser.add_argument("out_lean", type=Path)
    parser.add_argument("--module", default="SounioSatCubeCoverGenerated")
    parser.add_argument("--prefix", default="cubeCover")
    parser.add_argument(
        "--composition",
        choices=("split5", "generic", "arbitrary"),
        default="split5",
        help="compose with the legacy split theorem or the generic CubeCover theorem",
    )
    parser.add_argument(
        "--cover-cnf",
        type=Path,
        help="DIMACS CNF for base plus cube-blocking clauses; required for arbitrary mode",
    )
    parser.add_argument(
        "--cover-lrat",
        type=Path,
        help="LRAT refutation of --cover-cnf; required for arbitrary mode",
    )
    parser.add_argument(
        "--max-lrat-bytes",
        type=int,
        default=50_000_000,
        help="reject leaf LRAT files larger than this native_decide smoke limit; 0 disables",
    )
    args = parser.parse_args()

    try:
        if args.k != 5:
            raise RuntimeError("this Lean cover generator currently supports only k=5; use k=5")
        n, m, edges = parse_edge_file(args.edge_file)
        cubes = parse_batch(args.cube_batch)
        meta, rows = parse_refute_summary(args.refute_summary)
        required_meta = {
            "format", "formula_kind", "edge_sha256", "cube_batch_sha256",
            "n", "m", "k", "expected_vars", "base_clause_count", "out_dir",
        }
        missing_meta = sorted(required_meta.difference(meta))
        if missing_meta:
            raise RuntimeError(f"refutation summary missing metadata: {','.join(missing_meta)}")
        if meta.get("format") != "cube_sieve_refute_batch v1":
            raise RuntimeError("refutation summary has wrong or missing format marker")
        if meta.get("formula_kind") != "colourCNF":
            raise RuntimeError("only plain colourCNF refutation batches are composable")
        if meta.get("n") != str(n) or meta.get("m") != str(m) or meta.get("k") != str(args.k):
            raise RuntimeError("refutation summary graph metadata does not match inputs")
        if meta.get("edge_sha256") != sha256_file(args.edge_file):
            raise RuntimeError("refutation summary edge_sha256 mismatch")
        if meta.get("cube_batch_sha256") != sha256_file(args.cube_batch):
            raise RuntimeError("refutation summary cube_batch_sha256 mismatch")
        expected_vars = n * args.k
        base_clause_count = int(meta["base_clause_count"])
        if meta.get("expected_vars") != str(expected_vars):
            raise RuntimeError("refutation summary expected_vars mismatch")
        if args.composition == "split5":
            split_vertex, ordered_cubes = validate_single_vertex_split(cubes=cubes, k=args.k, n=n)
            split_vertices = [split_vertex]
        elif args.composition == "generic":
            split_vertices, ordered_cubes = validate_split_product(cubes=cubes, k=args.k, n=n)
            split_vertex = split_vertices[0]
        else:
            split_vertices, ordered_cubes = validate_arbitrary_cubes(cubes=cubes, k=args.k, n=n)
            split_vertex = split_vertices[0] if split_vertices else 0
            if args.cover_cnf is None or args.cover_lrat is None:
                raise RuntimeError("--composition arbitrary requires --cover-cnf and --cover-lrat")
        if len(rows) != len(ordered_cubes):
            raise RuntimeError(
                f"refutation summary has {len(rows)} cube rows, expected {len(ordered_cubes)}"
            )

        out_dir = Path(meta["out_dir"])
        if not out_dir.is_dir():
            raise RuntimeError(f"refutation summary out_dir does not exist: {out_dir}")
        row_by_id = {row.get("id", ""): row for row in rows}
        if len(row_by_id) != len(rows):
            raise RuntimeError("duplicate cube ids in refutation summary")
        leaves: list[Leaf] = []
        expected_ids = {cube_id for cube_id, _cube in ordered_cubes}
        extra_ids = sorted(set(row_by_id).difference(expected_ids))
        if extra_ids:
            raise RuntimeError("unexpected refutation rows: " + ",".join(extra_ids))
        for index, (cube_id, cube) in enumerate(ordered_cubes):
            row = row_by_id.get(cube_id)
            if row is None:
                raise RuntimeError(f"missing refutation row for cube {cube_id}")
            assignment = assignment_string(cube)
            if row.get("assignments") != assignment:
                raise RuntimeError(
                    f"row for {cube_id} has assignments {row.get('assignments')}, expected {assignment}"
                )
            if row.get("cube_assignment_count") != str(len(cube)):
                raise RuntimeError(
                    f"row {assignment} assignment count {row.get('cube_assignment_count')} "
                    f"!= expected {len(cube)}"
                )
            if row.get("cnf_clauses") != row.get("expected_cnf_clauses"):
                raise RuntimeError(f"row {assignment} CNF clause count was not validated")
            cnf_path = rel_or_abs(out_dir, row["cnf"])
            lrat_path = rel_or_abs(out_dir, row["lrat"])
            cnf_vars, cnf_clauses = parse_cnf_header(cnf_path)
            if cnf_vars != expected_vars:
                raise RuntimeError(f"row {assignment} CNF vars {cnf_vars} != expected {expected_vars}")
            if cnf_clauses != base_clause_count + len(cube):
                raise RuntimeError(
                    f"row {assignment} CNF clauses {cnf_clauses} != expected {base_clause_count + len(cube)}"
                )
            if args.max_lrat_bytes > 0 and lrat_path.stat().st_size > args.max_lrat_bytes:
                raise RuntimeError(
                    f"row {assignment} LRAT exceeds native_decide smoke limit "
                    f"({lrat_path.stat().st_size} > {args.max_lrat_bytes} bytes)"
                )
            if sha256_file(cnf_path) != row.get("cnf_sha256"):
                raise RuntimeError(f"row {assignment} cnf_sha256 mismatch")
            if sha256_file(lrat_path) != row.get("lrat_sha256"):
                raise RuntimeError(f"row {assignment} lrat_sha256 mismatch")
            leaves.append(
                Leaf(index=index, cube_id=cube_id, cube=cube, cnf_path=cnf_path, lrat_path=lrat_path)
            )

        cover_lrat_text: str | None = None
        if args.composition == "arbitrary":
            assert args.cover_cnf is not None
            assert args.cover_lrat is not None
            cover_vars, cover_clauses = parse_cnf_header(args.cover_cnf)
            if cover_vars != expected_vars:
                raise RuntimeError(f"cover CNF vars {cover_vars} != expected {expected_vars}")
            expected_cover_clauses = base_clause_count + len(ordered_cubes)
            if cover_clauses != expected_cover_clauses:
                raise RuntimeError(
                    f"cover CNF clauses {cover_clauses} != expected {expected_cover_clauses}"
                )
            if args.max_lrat_bytes > 0 and args.cover_lrat.stat().st_size > args.max_lrat_bytes:
                raise RuntimeError(
                    f"cover LRAT exceeds native_decide smoke limit "
                    f"({args.cover_lrat.stat().st_size} > {args.max_lrat_bytes} bytes)"
                )
            cover_lrat_text = renumber_lrat(args.cover_lrat, cover_clauses)

        module_name = lean_name(args.module)
        prefix = lean_name(args.prefix)
        args.out_lean.parent.mkdir(parents=True, exist_ok=True)
        args.out_lean.write_text(
            emit_lean(
                module_name=module_name,
                prefix=prefix,
                edge_file=args.edge_file,
                n=n,
                k=args.k,
                edges=edges,
                split_vertex=split_vertex,
                split_vertices=split_vertices,
                leaves=leaves,
                composition=args.composition,
                cover_lrat_text=cover_lrat_text,
            ),
            encoding="ascii",
        )
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("lean_cube_cover_reflect v1")
    print(f"output={args.out_lean}")
    print(f"module={lean_name(args.module)}")
    print(f"prefix={lean_name(args.prefix)}")
    print(f"n={n}")
    print(f"k={args.k}")
    print(f"split_vertex={split_vertex}")
    print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
    print(f"leaf_count={len(leaves)}")
    print(f"composition={args.composition}")
    if args.composition == "arbitrary":
        print(f"cover_cnf={args.cover_cnf}")
        print(f"cover_lrat={args.cover_lrat}")
        print("cover_claim=base_plus_cube_blockers_unsat")
    print("claim=finite_colourCNF_unsat_from_checked_cube_lrat_leaves")
    print("geometry_claim=none")
    print("status=lean_cube_cover_reflect_emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
