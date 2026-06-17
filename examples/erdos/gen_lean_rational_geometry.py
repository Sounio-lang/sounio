#!/usr/bin/env python3
"""Generate a Lean exact-Euclidean geometry module from rational coordinates.

Input contract: a DIMACS `p edge` graph and a zero-based coordinate CSV
(`id,x,y`) with one rational point per vertex. The generator rejects malformed
ids, duplicate/collapsed points, missing rows, and every listed edge whose exact
rational squared distance is not `1`.

Output contract: a Lean module that type-checks a concrete
`EuclideanNatEdgeExactGeometry` over `Rat^2`, an ordered edge-list sync theorem,
and a geometry-only `Real × Real` bridge for the same rational points via `qR`.
The Real bridge proves the standard expanded squared-distance unit relation and
listed Real unit edges. The output attaches no SAT/LRAT proof and claims no
chromatic lower bound; downstream promotion must add the no-5 certificate.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from fractions import Fraction
from pathlib import Path

from cube_sieve_propagation_manifest import parse_edge_file


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*$")


def lean_rat(q: Fraction) -> str:
    if q.denominator == 1:
        return f"({q.numerator} : Rat)"
    return f"(({q.numerator} : Rat) / ({q.denominator} : Rat))"


def parse_rat(raw: str) -> Fraction:
    try:
        return Fraction(raw.strip())
    except ValueError as exc:
        raise ValueError(f"bad rational literal {raw!r}") from exc


def parse_coords(path: Path, n: int) -> list[tuple[Fraction, Fraction]]:
    rows: dict[int, tuple[Fraction, Fraction]] = {}
    with path.open("r", encoding="ascii", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "id" in sample.splitlines()[0].lower() if sample.splitlines() else False
        if has_header:
            reader = csv.DictReader(f)
            for lineno, row in enumerate(reader, 2):
                if row is None:
                    continue
                try:
                    raw_id = row["id"]
                    raw_x = row["x"]
                    raw_y = row["y"]
                except KeyError as exc:
                    raise ValueError(f"{path}:{lineno}: expected id,x,y header") from exc
                add_coord(path, lineno, rows, raw_id, raw_x, raw_y, n)
        else:
            reader = csv.reader(f)
            for lineno, row in enumerate(reader, 1):
                if not row or (row[0].strip().startswith("#")):
                    continue
                if len(row) != 3:
                    raise ValueError(f"{path}:{lineno}: expected id,x,y row")
                add_coord(path, lineno, rows, row[0], row[1], row[2], n)
    missing = [str(i) for i in range(n) if i not in rows]
    if missing:
        raise ValueError(f"{path}: missing coordinate rows for vertices {','.join(missing)}")
    return [rows[i] for i in range(n)]


def add_coord(
    path: Path,
    lineno: int,
    rows: dict[int, tuple[Fraction, Fraction]],
    raw_id: str,
    raw_x: str,
    raw_y: str,
    n: int,
) -> None:
    if not re.fullmatch(r"[0-9]+", raw_id.strip()):
        raise ValueError(f"{path}:{lineno}: bad vertex id {raw_id!r}")
    vid = int(raw_id)
    if not (0 <= vid < n):
        raise ValueError(f"{path}:{lineno}: vertex id out of range: {vid}")
    if vid in rows:
        raise ValueError(f"{path}:{lineno}: duplicate vertex id: {vid}")
    rows[vid] = (parse_rat(raw_x), parse_rat(raw_y))


def dist2(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> Fraction:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def validate_geometry(
    coords: list[tuple[Fraction, Fraction]], edges: list[tuple[int, int]]
) -> None:
    seen_coords: dict[tuple[Fraction, Fraction], int] = {}
    for idx, xy in enumerate(coords):
        if xy in seen_coords:
            raise ValueError(f"duplicate coordinates for vertices {seen_coords[xy]} and {idx}")
        seen_coords[xy] = idx
    for u, v in edges:
        d = dist2(coords[u], coords[v])
        if d != 1:
            raise ValueError(f"edge {u},{v} has dist2={d}, expected 1")


def disjunction(var: str, n: int) -> str:
    return " ∨ ".join(f"{var} = {i}" for i in range(n))


def rcases(n: int, name: str) -> str:
    return f"rcases {name} with " + " | ".join("rfl" for _ in range(n))


def lean_list_edges(edges: list[tuple[int, int]]) -> str:
    return "[" + ", ".join(f"({u}, {v})" for u, v in edges) + "]"


def if_chain(name: str, coords: list[tuple[Fraction, Fraction]], index: int) -> str:
    parts: list[str] = []
    for i, xy in enumerate(coords[:-1]):
        parts.append(f"if {name}.val = {i} then {lean_rat(xy[index])} else ")
    return "".join(parts) + lean_rat(coords[-1][index])


def real_if_chain(name: str, coords: list[tuple[Fraction, Fraction]], index: int) -> str:
    parts: list[str] = []
    for i, xy in enumerate(coords[:-1]):
        parts.append(f"if {name}.val = {i} then qR {lean_rat(xy[index])} else ")
    return "".join(parts) + f"qR {lean_rat(coords[-1][index])}"


def emit_cases_theorem(
    name: str,
    n: int,
    statement: str,
    body_kind: str,
    simp_defs: str,
) -> list[str]:
    lines = [f"theorem {name} {statement} := by"]
    if body_kind == "binary":
        lines.extend(
            [
                "  cases p with",
                "  | mk pv hp =>",
                "    cases q with",
                "    | mk qv hq =>",
                f"      have hp_cases : {disjunction('pv', n)} := by omega",
                f"      have hq_cases : {disjunction('qv', n)} := by omega",
                f"      {rcases(n, 'hp_cases')} <;>",
                f"        {rcases(n, 'hq_cases')} <;>",
                f"        simp [{simp_defs}] <;> native_decide",
            ]
        )
    elif body_kind == "unary":
        lines.extend(
            [
                "  cases p with",
                "  | mk pv hp =>",
                f"    have hp_cases : {disjunction('pv', n)} := by omega",
                f"    {rcases(n, 'hp_cases')} <;>",
                f"      simp [{simp_defs}] <;> native_decide",
            ]
        )
    else:
        raise AssertionError(body_kind)
    return lines


def generate(
    *,
    module: str,
    namespace: str,
    prefix: str,
    n: int,
    edges: list[tuple[int, int]],
    coords: list[tuple[Fraction, Fraction]],
    sat_module: str,
    sat_edges_term: str,
) -> str:
    if not MODULE_RE.fullmatch(module):
        raise ValueError(f"invalid Lean module name: {module}")
    if not IDENT_RE.fullmatch(namespace):
        raise ValueError(f"invalid Lean namespace component: {namespace}")
    if not IDENT_RE.fullmatch(prefix):
        raise ValueError(f"invalid Lean prefix: {prefix}")
    if sat_module and not MODULE_RE.fullmatch(sat_module):
        raise ValueError(f"invalid Lean SAT module name: {sat_module}")
    if sat_edges_term and not MODULE_RE.fullmatch(sat_edges_term):
        raise ValueError(f"invalid Lean SAT edge term: {sat_edges_term}")
    if bool(sat_module) != bool(sat_edges_term):
        raise ValueError("--sat-module and --sat-edges-term must be supplied together")

    imports = [
        "import SounioFiniteUnitDistanceWitness",
        "import SounioMultiquadIndep",
        "import SounioRealPlaneGeometry",
        "import Init.Data.Rat.Lemmas",
    ]
    if sat_module:
        imports.append(f"import {sat_module}")

    lines: list[str] = []
    lines.extend(imports)
    lines.extend(
        [
            "",
            "set_option maxHeartbeats 0",
            "set_option maxRecDepth 1000000",
            "",
            "/-!",
            f"# {module}",
            "",
            "Generated exact rational-coordinate Euclidean geometry.",
            "This module is geometry-only: it contains no SAT/LRAT proof and makes",
            "no chromatic-number claim by itself.",
            "-/",
            "",
            "namespace UnitDistanceChromatic",
            "open SounioSqrt.RealCauchyField",
            f"namespace {namespace}",
            "",
            "def ratExactFieldLike : ExactFieldLike Rat where",
            "  zero := 0",
            "  one := 1",
            "  add := (· + ·)",
            "  neg := Neg.neg",
            "  sub := (· - ·)",
            "  mul := (· * ·)",
            "  inv := Inv.inv",
            "  ofNat := fun n => (n : Rat)",
            "  add_assoc := Rat.add_assoc",
            "  add_comm := Rat.add_comm",
            "  zero_add := Rat.zero_add",
            "  add_zero := Rat.add_zero",
            "  add_left_neg := by intro a; grind",
            "  sub_eq_add_neg := Rat.sub_eq_add_neg",
            "  mul_assoc := Rat.mul_assoc",
            "  mul_comm := Rat.mul_comm",
            "  one_mul := Rat.one_mul",
            "  mul_one := Rat.mul_one",
            "  left_distrib := Rat.mul_add",
            "  right_distrib := Rat.add_mul",
            "  zero_ne_one := by native_decide",
            "  inv_mul_cancel := Rat.inv_mul_cancel",
            "  ofNat_zero := rfl",
            "  ofNat_one := rfl",
            "  ofNat_add := Rat.natCast_add",
            "  ofNat_mul := Rat.natCast_mul",
            "  ofNat_inj := by",
            "    intro m n h",
            "    exact Rat.natCast_inj.mp h",
            "",
            f"def pointX (p : Fin {n}) : Rat :=",
            f"  {if_chain('p', coords, 0)}",
            "",
            f"def pointY (p : Fin {n}) : Rat :=",
            f"  {if_chain('p', coords, 1)}",
            "",
            f"def dist2 (p q : Fin {n}) : Rat :=",
            "  ((pointX p - pointX q) * (pointX p - pointX q)) +",
            "    ((pointY p - pointY q) * (pointY p - pointY q))",
            "",
            f"def unit (p q : Fin {n}) : Prop := dist2 p q = 1",
            "",
            f"instance unitDecidable (p q : Fin {n}) : Decidable (unit p q) := by",
            "  unfold unit dist2 pointX pointY",
            "  infer_instance",
            "",
        ]
    )
    lines.extend(
        [
            f"theorem dist2_zero_iff_eq : ∀ p q : Fin {n}, dist2 p q = 0 ↔ p = q := by",
            "  native_decide",
            "",
            f"theorem unit_symm : ∀ p q : Fin {n}, unit p q → unit q p := by",
            "  native_decide",
            "",
            f"theorem unit_irrefl : ∀ p : Fin {n}, ¬ unit p p := by",
            "  native_decide",
        ]
    )
    lines.extend(
        [
            "",
            f"def plane : ExactSquaredDistancePlane (Fin {n}) unit where",
            "  Scalar := Rat",
            "  scalar := ratExactFieldLike",
            "  x := pointX",
            "  y := pointY",
            "  dist2 := dist2",
            "  dist2_formula := by intro _p _q; rfl",
            "  unit_iff_dist2_eq_one := by intro _p _q; rfl",
            "  dist2_zero_iff_eq := dist2_zero_iff_eq",
            "  unit_symm := unit_symm",
            "  unit_irrefl := unit_irrefl",
            "",
            f"def emb (v : Nat) : Fin {n} := if h : v < {n} then ⟨v, h⟩ else ⟨0, by decide⟩",
            "",
            f"theorem emb_injective : ∀ {{i j}}, i < {n} → j < {n} → emb i = emb j → i = j := by",
            "  intro _i _j hi hj h",
            "  simp [emb, hi, hj] at h",
            "  exact h",
            "",
            f"def edges : List (Nat × Nat) := {lean_list_edges(edges)}",
            "",
            f"theorem endpoints : ∀ e ∈ edges, e.1 < {n} ∧ e.2 < {n} := by",
            "  native_decide",
            "",
            "theorem unit_edges : ∀ e ∈ edges, unit (emb e.1) (emb e.2) := by",
            "  native_decide",
            "",
            f"def exactGeometry : NatEdgeExactGeometry {n} (Fin {n}) unit where",
            "  edges := edges",
            "  emb := emb",
            "  emb_injective := emb_injective",
            "  endpoints := endpoints",
            "  unit_edges := unit_edges",
            "",
            f"def euclideanGeometry : EuclideanNatEdgeExactGeometry {n} (Fin {n}) unit where",
            "  exact := exactGeometry",
            "  plane := plane",
            "",
            f"theorem geometryHasEuclideanContract :",
            f"    ∃ G : EuclideanNatEdgeExactGeometry {n} (Fin {n}) unit, G.exact.edges = edges :=",
            "  ⟨euclideanGeometry, rfl⟩",
            "",
        ]
    )
    if sat_edges_term:
        lines.extend(
            [
                f"theorem edgesSync : euclideanGeometry.exact.edges = {sat_edges_term} := by",
                "  native_decide",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "theorem edgesSyncSelf : euclideanGeometry.exact.edges = edges := rfl",
                "",
            ]
        )
    lines.extend(
        [
            "/-! ## Real-plane geometry bridge -/",
            "",
            "abbrev realDist2 : Real × Real → Real × Real → Real := standardRealPlaneDist2",
            "",
            "abbrev realUnit : Real × Real → Real × Real → Prop := standardRealPlaneUnit",
            "",
            "theorem standardSubR_qR (x y : Rat) : standardSubR (qR x) (qR y) = qR (x - y) := by",
            "  unfold standardSubR",
            "  rw [qR_neg, qR_add]",
            "  congr 1",
            "  exact (Rat.sub_eq_add_neg x y).symm",
            "",
            "theorem standardSqR_qR (x : Rat) : standardSqR (qR x) = qR (x * x) := by",
            "  unfold standardSqR",
            "  rw [qR_mul]",
            "",
            "theorem standardRealPlaneDist2_qR (x1 y1 x2 y2 : Rat) :",
            "    standardRealPlaneDist2 (qR x1, qR y1) (qR x2, qR y2) =",
            "      qR ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) := by",
            "  unfold standardRealPlaneDist2",
            "  rw [standardSubR_qR, standardSubR_qR, standardSqR_qR, standardSqR_qR, qR_add]",
            "",
            "theorem realDist2_qR (x1 y1 x2 y2 : Rat) :",
            "    realDist2 (qR x1, qR y1) (qR x2, qR y2) =",
            "      qR ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) :=",
            "  standardRealPlaneDist2_qR x1 y1 x2 y2",
            "",
            "theorem realUnit_iff_standard :",
            "    ∀ p q : Real × Real, realUnit p q ↔ standardRealPlaneDist2 p q = qR (1 : Rat) := by",
            "  intro _p _q",
            "  rfl",
            "",
            f"def realPointX (p : Fin {n}) : Real :=",
            f"  {real_if_chain('p', coords, 0)}",
            "",
            f"def realPointY (p : Fin {n}) : Real :=",
            f"  {real_if_chain('p', coords, 1)}",
            "",
            f"def realPoint (p : Fin {n}) : Real × Real :=",
            "  (realPointX p, realPointY p)",
            "",
            f"def realEmb (v : Nat) : Real × Real := if h : v < {n} then realPoint ⟨v, h⟩ else realPoint ⟨0, by decide⟩",
            "",
            "theorem realUnitEdges : ∀ e ∈ edges, realUnit (realEmb e.1) (realEmb e.2) := by",
            "  intro e he",
            "  simp [edges] at he",
            f"  {rcases(len(edges), 'he')} <;>",
            "    unfold realUnit standardRealPlaneUnit realEmb realPoint realPointX realPointY <;>",
            "    simp <;>",
            "    rw [standardRealPlaneDist2_qR] <;>",
            "    congr 1 <;>",
            "    native_decide",
            "",
            f"abbrev {prefix}_point_type := Fin {n}",
            f"abbrev {prefix}_unit := unit",
            f"abbrev {prefix}_geometry := euclideanGeometry",
            f"abbrev {prefix}_edges := edges",
            f"abbrev {prefix}_real_unit := realUnit",
            f"abbrev {prefix}_real_unit_iff_standard := realUnit_iff_standard",
            f"abbrev {prefix}_real_emb := realEmb",
            f"abbrev {prefix}_real_unit_edges := realUnitEdges",
            "",
            f"end {namespace}",
            "end UnitDistanceChromatic",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("coord_csv", type=Path)
    parser.add_argument("out_lean", type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--prefix", default="ratgeom")
    parser.add_argument("--sat-module", default="")
    parser.add_argument("--sat-edges-term", default="")
    args = parser.parse_args()

    try:
        n, _m, edges = parse_edge_file(args.edge_file)
        if n < 2:
            raise ValueError("rational geometry generator expects at least two vertices")
        if not edges:
            raise ValueError("rational geometry generator expects at least one edge")
        coords = parse_coords(args.coord_csv, n)
        validate_geometry(coords, edges)
        namespace = args.namespace or args.module.split(".")[-1]
        text = generate(
            module=args.module,
            namespace=namespace,
            prefix=args.prefix,
            n=n,
            edges=edges,
            coords=coords,
            sat_module=args.sat_module,
            sat_edges_term=args.sat_edges_term,
        )
        args.out_lean.parent.mkdir(parents=True, exist_ok=True)
        args.out_lean.write_text(text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("gen_lean_rational_geometry v1")
    print(f"edge_file={args.edge_file}")
    print(f"coord_csv={args.coord_csv}")
    print(f"out_lean={args.out_lean}")
    print(f"module={args.module}")
    print(f"n={n}")
    print(f"m={len(edges)}")
    print("geometry_claim=exact_rational_squared_distance_edges_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("promotable=0")
    print("status=lean_rational_geometry_emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
