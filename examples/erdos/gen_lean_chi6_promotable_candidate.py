#!/usr/bin/env python3
"""Generate the candidate-owned Lean join module for a promotable chi6 package.

This script does not search and does not prove a new witness by itself. It only
joins two already-produced manifests:

* a rational exact-geometry manifest with Real-plane bridge terms, and
* a cube-cover SAT manifest with a Lean-checked no-5-colouring proof.

The output Lean module imports those artifacts and exposes the term surface
required by `validate_chi6_promotable_candidate.sh`: edge sync, no-five witness,
finite no-five theorem, standard `Real x Real` unit relation, and Real-plane
no-five theorem. If the manifests do not describe the same graph, the generator
fails before emitting Lean.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys


LEAN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_']*$")


class ManifestError(RuntimeError):
    pass


def read_manifest(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ManifestError(f"{path}:{lineno}: line lacks '='")
        key, value = line.split("=", 1)
        if key in fields:
            raise ManifestError(f"{path}:{lineno}: duplicate key {key}")
        fields[key] = value
    return fields


def need(fields: dict[str, str], key: str, label: str) -> str:
    value = fields.get(key, "")
    if not value or value == "NONE":
        raise ManifestError(f"{label} manifest missing {key}")
    return value


def need_optional_name(fields: dict[str, str], key: str, label: str) -> str:
    value = need(fields, key, label)
    if not LEAN_NAME_RE.fullmatch(value):
        raise ManifestError(f"{label} manifest has invalid Lean name in {key}: {value}")
    return value


def resolve_manifest_path(manifest: Path, raw: str) -> Path:
    if raw == "NONE":
        raise ManifestError("cannot resolve NONE artifact path")
    path = Path(raw)
    if path.is_absolute():
        return path
    return manifest.parent / path


def module_name_from_path(path: Path) -> str:
    if path.suffix != ".lean":
        raise ManifestError(f"Lean module path must end in .lean: {path}")
    return path.stem


@dataclass(frozen=True)
class SatTerms:
    route: str
    edges: str
    plain_cnf: str = ""
    plain_unsat: str = ""
    triangle: tuple[str, str, str] = ()
    triangle_n: str = ""
    sb_cnf: str = ""
    sb_unsat: str = ""
    split_vertex: str = ""
    split_unsats: tuple[str, ...] = ()
    cubes: str = ""
    cube_cover: str = ""
    cube_unsat: str = ""


def single_decl(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    matches = [m[0] if isinstance(m, tuple) else m for m in matches]
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise ManifestError(f"expected exactly one {label} declaration, found {unique}")
    return unique[0]


def parse_cube_cover_sat_terms(path: Path) -> SatTerms:
    text = path.read_text(encoding="utf-8")
    if "unsat_of_cube_cover" not in text:
        raise ManifestError(f"{path}: SAT module lacks unsat_of_cube_cover route")
    return SatTerms(
        route="cube_cover_generic",
        edges=single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_edges)\b", "SAT edges"),
        cubes=single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_cubes)\b", "SAT cubes"),
        cube_cover=single_decl(
            text, r"^theorem\s+([A-Za-z_][A-Za-z0-9_']*_cube_cover)\b", "cube cover"
        ),
        cube_unsat=single_decl(
            text, r"^theorem\s+([A-Za-z_][A-Za-z0-9_']*_cube_unsat)\b", "cube unsat"
        ),
    )


def parse_plain_lrat_sat_terms(path: Path) -> SatTerms:
    text = path.read_text(encoding="utf-8")
    if "colourCNFsb5" in text or "colourCNFWithUnit" in text or "colourCNFWithCube" in text:
        raise ManifestError(f"{path}: plain_lrat route must use plain colourCNF only")
    if "colourCNF" not in text:
        raise ManifestError(f"{path}: plain_lrat route lacks colourCNF")
    return SatTerms(
        route="plain_lrat",
        edges=single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_edges)\b", "SAT edges"),
        plain_cnf=single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_cnf)\b", "SAT CNF"),
        plain_unsat=single_decl(text, r"^theorem\s+([A-Za-z_][A-Za-z0-9_']*_unsat)\b", "plain SAT unsat"),
    )


def parse_triangle_sb5_sat_terms(path: Path) -> SatTerms:
    text = path.read_text(encoding="utf-8")
    if "colourCNFsb5" not in text:
        raise ManifestError(f"{path}: triangle_sb5_lrat route lacks colourCNFsb5")
    edges = single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_edges)\b", "SAT edges")
    cnf_matches = re.findall(
        rf"^def\s+([A-Za-z_][A-Za-z0-9_']*_cnf)\b[^\n]*:=\s*"
        rf"colourCNFsb5\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+{edges}\b",
        text,
        flags=re.MULTILINE,
    )
    if len(cnf_matches) != 1:
        raise ManifestError(f"{path}: expected exactly one colourCNFsb5 CNF declaration")
    cnf, a, b, c, n = cnf_matches[0]
    unsat = single_decl(
        text, rf"^theorem\s+([A-Za-z_][A-Za-z0-9_']*_unsat)\s*:\s*{cnf}\.Unsat\b",
        "SB5 SAT unsat",
    )
    return SatTerms(
        route="triangle_sb5_lrat",
        edges=edges,
        triangle=(a, b, c),
        triangle_n=n,
        sb_cnf=cnf,
        sb_unsat=unsat,
    )


def parse_split5_sat_terms(path: Path) -> SatTerms:
    text = path.read_text(encoding="utf-8")
    if "colourCNFWithUnit" not in text:
        raise ManifestError(f"{path}: cube_cover_split5 route lacks colourCNFWithUnit")
    if "unsat_of_split_vertex5" not in text:
        raise ManifestError(f"{path}: cube_cover_split5 route lacks unsat_of_split_vertex5")
    if "colourCNFWithCube" in text or "unsat_of_cube_cover" in text:
        raise ManifestError(f"{path}: cube_cover_split5 route must not use generic cube-cover composition")
    edges = single_decl(text, r"^def\s+([A-Za-z_][A-Za-z0-9_']*_edges)\b", "SAT edges")
    leaf_matches = re.findall(
        r"^theorem\s+([A-Za-z_][A-Za-z0-9_']*_v([0-9]+)_c([0-4])_unsat)\b",
        text,
        flags=re.MULTILINE,
    )
    if len(leaf_matches) != 5:
        raise ManifestError(f"{path}: cube_cover_split5 route expected five leaf UNSAT theorems")
    split_vertices = {v for _name, v, _colour in leaf_matches}
    if len(split_vertices) != 1:
        raise ManifestError(f"{path}: cube_cover_split5 leaf theorems mix split vertices")
    split_vertex = split_vertices.pop()
    by_colour: dict[str, str] = {}
    for name, _v, colour in leaf_matches:
        if colour in by_colour:
            raise ManifestError(f"{path}: duplicate cube_cover_split5 leaf colour {colour}")
        by_colour[colour] = name
    missing = [str(c) for c in range(5) if str(c) not in by_colour]
    if missing:
        raise ManifestError(f"{path}: cube_cover_split5 missing leaf colours {','.join(missing)}")
    return SatTerms(
        route="cube_cover_split5",
        edges=edges,
        split_vertex=split_vertex,
        split_unsats=tuple(by_colour[str(c)] for c in range(5)),
    )


def parse_sat_terms(path: Path, route: str) -> SatTerms:
    if route == "plain_lrat":
        return parse_plain_lrat_sat_terms(path)
    if route == "triangle_sb5_lrat":
        return parse_triangle_sb5_sat_terms(path)
    if route == "cube_cover_split5":
        return parse_split5_sat_terms(path)
    if route == "cube_cover_generic":
        return parse_cube_cover_sat_terms(path)
    raise ManifestError(
        "unsupported sat_proof_route for this assembler: "
        f"{route} (supported: plain_lrat, triangle_sb5_lrat, cube_cover_split5, "
        "cube_cover_generic)"
    )


def check_same_candidate(geom: dict[str, str], sat: dict[str, str]) -> None:
    for key in ("candidate_id", "n", "m", "k", "edge_sha256"):
        if geom.get(key) != sat.get(key):
            raise ManifestError(
                f"manifest metadata mismatch for {key}: geometry={geom.get(key)} sat={sat.get(key)}"
            )
    if geom.get("geometry_proof_type") != "euclidean":
        raise ManifestError("geometry manifest must have geometry_proof_type=euclidean")
    if geom.get("sat_proof_route") != "none":
        raise ManifestError("geometry manifest must have sat_proof_route=none")
    if sat.get("sat_proof_route") not in {
        "plain_lrat",
        "triangle_sb5_lrat",
        "cube_cover_split5",
        "cube_cover_generic",
    }:
        raise ManifestError(
            "only sat_proof_route=plain_lrat, triangle_sb5_lrat, cube_cover_split5, "
            "or cube_cover_generic is supported for this assembler"
        )
    if geom.get("promotable") != "0" or sat.get("promotable") != "0":
        raise ManifestError("input manifests must be non-promotable components")


def generate(
    *,
    module: str,
    namespace: str,
    sat_import: str,
    geom_import: str,
    n: str,
    sat_terms: SatTerms,
    point_type: str,
    unit_term: str,
    geometry_term: str,
    real_unit_term: str,
    real_unit_iff: str,
    real_emb_term: str,
    real_unit_edges_term: str,
) -> str:
    for label, name in {
        "module": module,
        "namespace": namespace,
        "sat import": sat_import,
        "geometry import": geom_import,
    }.items():
        if not LEAN_NAME_RE.fullmatch(name):
            raise ManifestError(f"invalid Lean {label}: {name}")
    if not IDENT_RE.fullmatch(namespace.split(".")[-1]):
        raise ManifestError(f"invalid namespace leaf: {namespace}")
    for label, name in {
        "point type": point_type,
        "unit": unit_term,
        "geometry": geometry_term,
        "real unit": real_unit_term,
        "real unit iff": real_unit_iff,
        "real emb": real_emb_term,
        "real unit edges": real_unit_edges_term,
    }.items():
        if not LEAN_NAME_RE.fullmatch(name):
            raise ManifestError(f"invalid Lean {label} term: {name}")

    if not LEAN_NAME_RE.fullmatch(sat_terms.edges):
        raise ManifestError(f"invalid Lean SAT edges term: {sat_terms.edges}")

    if sat_terms.route == "plain_lrat":
        for label, name in {
            "plain CNF": sat_terms.plain_cnf,
            "plain UNSAT": sat_terms.plain_unsat,
        }.items():
            if not LEAN_NAME_RE.fullmatch(name):
                raise ManifestError(f"invalid Lean {label} term: {name}")
        route_marker = "Route-specific no-five witness adapter appears in the definitions below."
        route_block = f"""
theorem plainUnsatOnGeometry :
    (SounioSatColouring.colourCNF {n} 5 geometry.exact.edges).Unsat := by
  simpa [satEdges, edgesSync, {sat_terms.plain_cnf}] using {sat_terms.plain_unsat}

def noFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} point_type unit :=
  geometry.noFiveWitnessOfColourCNFUnsat plainUnsatOnGeometry

theorem finalTheorem :
    ¬ Nonempty (PlaneColouring point_type unit 5) :=
  EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract
    geometry noFiveWitness rfl rfl

def realNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} (Real × Real) realUnit :=
  NatEdgeUnitDistanceCertificate.ofColourCNFUnsat
    (n := {n}) (k := 5) (P := Real × Real) (unit := realUnit)
    (by decide) geometry.exact.edges realEmb
    geometry.exact.endpoints realUnitEdges
    plainUnsatOnGeometry
"""
    elif sat_terms.route == "triangle_sb5_lrat":
        if len(sat_terms.triangle) != 3:
            raise ManifestError("triangle_sb5_lrat requires a parsed triangle")
        for label, name in {
            "SB5 CNF": sat_terms.sb_cnf,
            "SB5 UNSAT": sat_terms.sb_unsat,
        }.items():
            if not LEAN_NAME_RE.fullmatch(name):
                raise ManifestError(f"invalid Lean {label} term: {name}")
        a, b, c = sat_terms.triangle
        route_marker = "Route-specific no-five witness adapter appears in the definitions below."
        route_block = f"""
theorem triangleUnsatOnGeometry :
    (SounioSatColouringSB.colourCNFsb5 {a} {b} {c} {n} geometry.exact.edges).Unsat := by
  simpa [satEdges, edgesSync, {sat_terms.sb_cnf}] using {sat_terms.sb_unsat}

def noFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} point_type unit :=
  geometry.noFiveWitnessOfColourCNFsb5UnsatTri
    {a} {b} {c}
    (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    triangleUnsatOnGeometry

theorem finalTheorem :
    ¬ Nonempty (PlaneColouring point_type unit 5) :=
  EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract
    geometry noFiveWitness rfl rfl

def realNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} (Real × Real) realUnit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfColourCNFsb5UnsatTri
    (n := {n}) (P := Real × Real) (unit := realUnit)
    geometry.exact.edges realEmb
    {a} {b} {c}
    (by decide) (by decide) (by decide)
    (by native_decide) (by native_decide) (by native_decide)
    geometry.exact.endpoints
    realUnitEdges
    triangleUnsatOnGeometry
"""
    elif sat_terms.route == "cube_cover_split5":
        if not sat_terms.split_vertex.isdecimal():
            raise ManifestError(f"invalid split vertex: {sat_terms.split_vertex}")
        if len(sat_terms.split_unsats) != 5:
            raise ManifestError("cube_cover_split5 requires exactly five leaf UNSAT terms")
        for idx, name in enumerate(sat_terms.split_unsats):
            if not LEAN_NAME_RE.fullmatch(name):
                raise ManifestError(f"invalid Lean split colour {idx} UNSAT term: {name}")
        route_marker = "Route-specific no-five witness adapter appears in the definitions below."
        split_theorems = "\n\n".join(
            f"""theorem splitUnsat{idx}OnGeometry :
    (SounioSatCubeCover.colourCNFWithUnit {n} 5 geometry.exact.edges {sat_terms.split_vertex} {idx}).Unsat := by
  simpa [satEdges, edgesSync] using {name}"""
            for idx, name in enumerate(sat_terms.split_unsats)
        )
        split_args = "\n".join(f"    splitUnsat{idx}OnGeometry" for idx in range(5))
        route_block = f"""
{split_theorems}

def noFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} point_type unit :=
  geometry.noFiveWitnessOfSplitVertex5Unsat
    {sat_terms.split_vertex}
    (by decide)
{split_args}

theorem finalTheorem :
    ¬ Nonempty (PlaneColouring point_type unit 5) :=
  EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract
    geometry noFiveWitness rfl rfl

def realNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} (Real × Real) realUnit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfSplitVertex5Unsat
    (n := {n}) (P := Real × Real) (unit := realUnit)
    geometry.exact.edges realEmb
    {sat_terms.split_vertex}
    (by decide)
    geometry.exact.endpoints realUnitEdges
{split_args}
"""
    elif sat_terms.route == "cube_cover_generic":
        for label, name in {
            "SAT cubes": sat_terms.cubes,
            "cube cover": sat_terms.cube_cover,
            "cube UNSAT": sat_terms.cube_unsat,
        }.items():
            if not LEAN_NAME_RE.fullmatch(name):
                raise ManifestError(f"invalid Lean {label} term: {name}")
        route_marker = "Route-specific no-five witness adapter appears in the definitions below."
        route_block = f"""
theorem cubeCoverOnGeometry :
    SounioSatCubeCover.CubeCover {n} 5 geometry.exact.edges {sat_terms.cubes} := by
  simpa [satEdges, edgesSync] using {sat_terms.cube_cover}

theorem cubeUnsatOnGeometry :
    ∀ cube, cube ∈ {sat_terms.cubes} →
      (SounioSatCubeCover.colourCNFWithCube {n} 5 geometry.exact.edges cube).Unsat := by
  intro cube hcube
  simpa [satEdges, edgesSync] using {sat_terms.cube_unsat} cube hcube

def noFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} point_type unit :=
  geometry.noFiveWitnessOfCubeCoverUnsat
    {sat_terms.cubes}
    cubeCoverOnGeometry
    cubeUnsatOnGeometry

theorem finalTheorem :
    ¬ Nonempty (PlaneColouring point_type unit 5) :=
  EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract
    geometry noFiveWitness rfl rfl

def realNoFiveWitness :
    NatEdgeUnitDistanceCertificate.NoFiveColourWitness {n} (Real × Real) realUnit :=
  NatEdgeUnitDistanceCertificate.noFiveWitnessOfCubeCoverUnsat
    (n := {n}) (P := Real × Real) (unit := realUnit)
    geometry.exact.edges realEmb {sat_terms.cubes}
    cubeCoverOnGeometry
    geometry.exact.endpoints
    realUnitEdges
    cubeUnsatOnGeometry
"""
    else:
        raise ManifestError(f"internal error: unsupported route {sat_terms.route}")

    return f"""import SounioFiniteUnitDistanceWitness
import SounioRootedFieldReal
import SounioMultiquadIndep
import SounioRealPlaneGeometry
import {sat_import}
import {geom_import}

/-!
# {module}

Candidate-owned promotion join module.

This file contains no search. It joins an exact Euclidean rational geometry
artifact to a Lean-checked cube-cover no-5 SAT artifact for the same ordered
edge list. The surrounding manifest/offload gates decide whether the package is
eligible for promotion. SAT route: `{sat_terms.route}`.

Contract markers for the manifest validator:
`EuclideanNatEdgeExactGeometry`, `ExactFieldLike`,
{route_marker}
-/

namespace UnitDistanceChromatic
open SounioSqrt.RealCauchyField

namespace {namespace}

abbrev point_type := {point_type}
abbrev unit := {unit_term}
abbrev geometry : EuclideanNatEdgeExactGeometry {n} point_type unit := {geometry_term}
abbrev satEdges : List (Nat × Nat) := {sat_terms.edges}

theorem edgesSync : geometry.exact.edges = satEdges := by
  native_decide

abbrev realUnit : Real × Real → Real × Real → Prop := {real_unit_term}
abbrev realUnitIffStandard := {real_unit_iff}
abbrev realEmb : Nat → Real × Real := {real_emb_term}

theorem realUnitEdges : ∀ e ∈ geometry.exact.edges, realUnit (realEmb e.1) (realEmb e.2) := by
  simpa [geometry] using {real_unit_edges_term}

{route_block}

theorem realFinalTheorem :
    ¬ Nonempty (PlaneColouring (Real × Real) realUnit 5) :=
  NatEdgeUnitDistanceCertificate.generic_no_five_colour_obstruction realNoFiveWitness

#print axioms edgesSync
#print axioms noFiveWitness
#print axioms finalTheorem
#print axioms realUnitIffStandard
#print axioms realFinalTheorem

end {namespace}
end UnitDistanceChromatic
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry_manifest", type=Path)
    parser.add_argument("sat_manifest", type=Path)
    parser.add_argument("out_lean", type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--namespace", default="")
    parser.add_argument("--sat-import", default="")
    parser.add_argument("--geometry-import", default="")
    args = parser.parse_args()

    try:
        geom = read_manifest(args.geometry_manifest)
        sat = read_manifest(args.sat_manifest)

        real_unit = need_optional_name(geom, "lean_real_unit_term", "geometry")
        real_unit_iff = need_optional_name(geom, "lean_real_unit_iff_standard", "geometry")
        real_emb = need_optional_name(geom, "lean_real_emb_term", "geometry")
        real_unit_edges = need_optional_name(geom, "lean_real_unit_edges_term", "geometry")

        need_optional_name(geom, "lean_point_type", "geometry")
        need_optional_name(geom, "lean_unit_term", "geometry")
        need_optional_name(geom, "lean_geometry_term", "geometry")
        check_same_candidate(geom, sat)

        sat_path = resolve_manifest_path(args.sat_manifest, need(sat, "lean_sat_module_path", "sat"))
        geom_path = resolve_manifest_path(
            args.geometry_manifest, need(geom, "geometry_module_path", "geometry")
        )
        if not sat_path.is_file():
            raise ManifestError(f"SAT Lean module not found: {sat_path}")
        if not geom_path.is_file():
            raise ManifestError(f"geometry Lean module not found: {geom_path}")

        sat_route = need(sat, "sat_proof_route", "sat")
        sat_terms = parse_sat_terms(sat_path, sat_route)
        if sat_terms.route == "triangle_sb5_lrat":
            expected_triangle = need(sat, "triangle_sb", "sat")
            parsed_triangle = ",".join(sat_terms.triangle)
            if expected_triangle != parsed_triangle:
                raise ManifestError(
                    "sat manifest triangle_sb does not match Lean SAT module: "
                    f"manifest={expected_triangle} lean={parsed_triangle}"
                )
            expected_n = need(sat, "n", "sat")
            if sat_terms.triangle_n != expected_n:
                raise ManifestError(
                    "sat manifest n does not match Lean SB5 CNF declaration: "
                    f"manifest={expected_n} lean={sat_terms.triangle_n}"
                )
        sat_import = args.sat_import or module_name_from_path(sat_path)
        geom_import = args.geometry_import or need_optional_name(geom, "lean_module", "geometry")
        namespace = args.namespace or args.module.split(".")[-1]

        text = generate(
            module=args.module,
            namespace=namespace,
            sat_import=sat_import,
            geom_import=geom_import,
            n=need(sat, "n", "sat"),
            sat_terms=sat_terms,
            point_type=geom["lean_point_type"],
            unit_term=geom["lean_unit_term"],
            geometry_term=geom["lean_geometry_term"],
            real_unit_term=real_unit,
            real_unit_iff=real_unit_iff,
            real_emb_term=real_emb,
            real_unit_edges_term=real_unit_edges,
        )
        args.out_lean.parent.mkdir(parents=True, exist_ok=True)
        args.out_lean.write_text(text, encoding="utf-8")
    except (OSError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("gen_lean_chi6_promotable_candidate v1")
    print(f"geometry_manifest={args.geometry_manifest}")
    print(f"sat_manifest={args.sat_manifest}")
    print(f"out_lean={args.out_lean}")
    print(f"module={args.module}")
    print(f"namespace={namespace}")
    print(f"sat_proof_route={sat_terms.route}")
    print(f"sat_edges_term={sat_terms.edges}")
    if sat_terms.route == "plain_lrat":
        print(f"sat_plain_cnf_term={sat_terms.plain_cnf}")
        print(f"sat_plain_unsat_term={sat_terms.plain_unsat}")
    elif sat_terms.route == "triangle_sb5_lrat":
        print(f"sat_triangle_sb={','.join(sat_terms.triangle)}")
        print(f"sat_sb5_cnf_term={sat_terms.sb_cnf}")
        print(f"sat_sb5_unsat_term={sat_terms.sb_unsat}")
    elif sat_terms.route == "cube_cover_split5":
        print(f"sat_split_vertex={sat_terms.split_vertex}")
        for idx, term in enumerate(sat_terms.split_unsats):
            print(f"sat_split_unsat_c{idx}_term={term}")
    elif sat_terms.route == "cube_cover_generic":
        print(f"sat_cubes_term={sat_terms.cubes}")
        print(f"sat_cube_cover_term={sat_terms.cube_cover}")
        print(f"sat_cube_unsat_term={sat_terms.cube_unsat}")
    print("claim=promotable_join_module_surface_only")
    print("status=lean_chi6_promotable_candidate_emitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
