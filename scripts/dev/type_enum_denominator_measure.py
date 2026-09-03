#!/usr/bin/env python3
"""
Type-enum denominator measurement (PROTOCOLO v3 companion).

REFUTATION CRITERIA (written BEFORE measurement; fail closed):

R1. COUNT ≠ claimed 8
    If the number of *included* type-kind enums is not 8, the report MUST
    state the measured N and the inclusion/exclusion of every candidate with
    path:line. Silent rounding to 8 is a FAIL.

R2. ORDINAL IDENTITY FORBIDDEN
    Two variants must NEVER be declared "the same type" because their
    source-order positions or raw integer codes coincide. This tree has at
    least two token/type numberings (TokenKind enum vs MC_TK_* lean constants;
    TypeKind ordinal vs lean_single EXPR_TY / ETY_KIND / stdlib TYPE_* codes).
    Comparing code-n with enum-cast is a FAIL of the ruler.

R3. STEM-ONLY IDENTITY INSUFFICIENT
    A shared normalized stem (e.g. "Octonion", "Knowledge") is a *candidate*
    for same-type, not a proof. Classification:
      - same_type: stem match AND an explicit cross-layer conversion/use site
        names both sides (e.g. HlirTypeKind::HlirTypeX in convert path, or
        TypeExprKind::TypeX lowered to TypeKind::TyX in check.sio).
      - homonym: stem match AND no conversion/use bridge found.
      - singleton: stem appears in exactly one included enum.

R4. IMPOSSIBLE POSITION → RULER
    If a derived "coverage" cell would require a kind that does not exist in
    the enum body, or a fixture path that is not a file, mark RULER_SUSPECT
    and do not invent the row. (Reserved was found this way.)

R5. FIXTURE COVERAGE IS FILE EXISTENCE + NON-EMPTY PATH
    "Has fixture" means index row has non-empty pass OR refuse path that is
    not '-' AND the file exists. Ghosts (*.ghost_*.sio) are attempts, not
    fixtures. Position is NOT derived here — only denominator/coverage.

R6. SHADOW COPIES
    bootstrap/, *_test.sio, *_lean.sio duplicates of a primary enum are
    listed as shadows, not additional denominators.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(os.environ.get("SOUNIO_ROOT", os.getcwd())).resolve()

ENUM_RE = re.compile(r"\benum\s+(\w+)\s*\{")
# TYPE_*() kind constants in stdlib
TYPE_CONST_RE = re.compile(r"fn\s+(TYPE_\w+)\(\)\s*->\s*i32\s*\{\s*(\d+)\s*\}")

# Comment-documented lean kind tables (not enums — second numbering)
LEAN_KIND_COMMENT = re.compile(
    r"Type kinds:\s*(.+?)(?:\n//\s{0,3}\n|\n\nvar |\nfn )",
    re.S,
)


@dataclass
class Variant:
    name: str
    line: int  # 1-based declaration line
    # position_in_source is recorded for audit but NEVER used as identity
    source_index: int


@dataclass
class EnumBody:
    name: str
    path: str  # repo-relative
    line: int
    variants: List[Variant] = field(default_factory=list)
    kind: str = "enum"  # enum | const_table | lean_comment_table
    notes: str = ""


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def extract_enum_variants(path: Path, start_line: int) -> List[Variant]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = start_line - 1
    if i < 0 or i >= len(lines):
        return []
    depth = 0
    started = False
    body_lines: List[Tuple[int, str]] = []
    for j in range(i, min(i + 500, len(lines))):
        line = lines[j]
        if not started:
            if "{" in line:
                started = True
                depth += line.count("{") - line.count("}")
                after = line.split("{", 1)[1]
                body_lines.append((j + 1, after))
                if depth <= 0:
                    break
            continue
        depth += line.count("{") - line.count("}")
        body_lines.append((j + 1, line))
        if depth <= 0:
            break
    variants: List[Variant] = []
    seen_idx = 0
    for lineno, raw in body_lines:
        s = raw.split("//")[0].strip().rstrip(",").strip()
        if not s or s in ("{", "}", ","):
            continue
        # skip nested-looking junk
        tok = re.split(r"[\s,\(\{]", s)[0]
        if not tok or not tok[0].isalpha():
            continue
        # Sounio enums: bare variant names
        variants.append(Variant(name=tok, line=lineno, source_index=seen_idx))
        seen_idx += 1
    return variants


def discover_enum_candidates() -> List[EnumBody]:
    """Walk self-hosted + stdlib/compiler for type-kind enums."""
    roots = [ROOT / "self-hosted", ROOT / "stdlib" / "compiler"]
    hits: List[EnumBody] = []
    name_hint = re.compile(
        r"(TypeKind|TypeExprKind|HlirTypeKind|HlirTypeDefKind|LayTypeKind|"
        r"OwnTypeKind|GpuType|LsphTypeCategory|HirType|TypeCategory|"
        r"TypeTag|ValueType)$"
    )
    for base in roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*.sio"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                m = ENUM_RE.search(line)
                if not m:
                    continue
                name = m.group(1)
                # Include if name matches type-kind pattern OR contains Type+Kind
                if not (
                    name_hint.search(name)
                    or (name.endswith("Type") and "Op" not in name)
                    or ("Type" in name and "Kind" in name)
                ):
                    continue
                # Exclude obvious non-value classifiers later; still collect
                variants = extract_enum_variants(path, i)
                hits.append(
                    EnumBody(
                        name=name,
                        path=rel(path),
                        line=i,
                        variants=variants,
                    )
                )
    return hits


def classify_inclusion(e: EnumBody) -> Tuple[str, str]:
    """
    Returns (bucket, reason).
    bucket: primary | shadow | excluded | adjacent_const_table
    """
    p = e.path.replace("\\", "/")
    n = e.name

    # Shadows / duplicates
    if "/bootstrap/" in p or p.endswith("bootstrap_v0.sio"):
        return "shadow", "bootstrap copy of primary parser/checker surface"
    if p.endswith("_test.sio") or "/test" in p:
        return "shadow", "test double"
    if "wmma_lean" in p or p.endswith("_lean.sio"):
        return "shadow", "lean/wmma duplicate of GpuType"

    # Explicit primary type-kind enums (value or definition classification in pipeline)
    primary_keys = {
        ("self-hosted/parser/ast.sio", "TypeExprKind"): "parser AST type-expr kind",
        ("self-hosted/check/types.sio", "TypeKind"): "checker semantic TypeKind",
        ("self-hosted/check/layout.sio", "LayTypeKind"): "layout ABI type kind",
        ("self-hosted/check/ownership.sio", "OwnTypeKind"): "ownership class of a type",
        ("self-hosted/hlir/ir.sio", "HlirTypeKind"): "HLIR value type kind",
        ("self-hosted/hlir/ir.sio", "HlirTypeDefKind"): "HLIR type-definition kind (struct/enum)",
        ("self-hosted/gpu/kernel_ir.sio", "GpuType"): "GPU kernel IR type",
        ("self-hosted/compiler/parser.sio", "TypeKind"): "legacy lean-path thin TypeKind (name collision)",
        ("self-hosted/lsp/hover.sio", "LsphTypeCategory"): "LSP hover type category",
        ("stdlib/compiler/transform/type_annotation.sio", "HirType"): "stdlib transform HirType sketch",
    }
    key = (p, n)
    if key in primary_keys:
        # Still may exclude below
        reason = primary_keys[key]
    else:
        # Other Type* enums not in the curated map
        if n in ("EffectKind",) or "Effect" in n:
            return "excluded", "effect taxonomy, not type taxonomy"
        return "excluded", f"not in curated type-kind map ({n})"

    # Sub-buckets for the curated set — all reported; "included" decided below
    return "candidate", reason


def load_stdlib_type_const_table() -> EnumBody:
    path = ROOT / "stdlib/compiler/types/type.sio"
    text = path.read_text(encoding="utf-8", errors="replace")
    variants: List[Variant] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = TYPE_CONST_RE.search(line)
        if not m:
            continue
        variants.append(Variant(name=m.group(1), line=i, source_index=len(variants)))
    return EnumBody(
        name="TYPE_const_table",
        path=rel(path),
        line=variants[0].line if variants else 0,
        variants=variants,
        kind="const_table",
        notes="stdlib kind codes (INTEGER numbering — not an enum; second ruler)",
    )


def load_lean_ety_comment_table() -> EnumBody:
    """Documented ETY_KIND codes from lean_single comment — not an enum."""
    path = ROOT / "self-hosted/compiler/lean_single.sio"
    text = path.read_text(encoding="utf-8", errors="replace")
    # Find the Type kinds comment near ETY_KIND
    variants: List[Variant] = []
    # Known from comment + ety_mk_* / type_name_kind / ty_eq arms — DO NOT use ordinals as identity
    # Extract only from explicit comment lines and ety_mk_* function names
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if "Type kinds:" in line or (line.strip().startswith("//") and "0=unknown" in line):
            # parse "0=unknown, 1=i64, ..."
            blob = line
            # also pull continuation lines
            j = i
            while j < len(lines) and (lines[j - 1].strip().startswith("//") or j == i):
                blob += " " + lines[j - 1]
                j += 1
                if j > i + 5:
                    break
            for m in re.finditer(r"(\d+)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", blob):
                variants.append(
                    Variant(name=f"ETY_{m.group(2)}", line=i, source_index=int(m.group(1)))
                )
            break
    # Also collect ety_mk_* as named constructors (identity by name, not code)
    for i, line in enumerate(lines, 1):
        m = re.search(r"fn\s+(ety_mk_\w+)\s*\(", line)
        if m:
            nm = m.group(1)
            if not any(v.name == nm for v in variants):
                variants.append(Variant(name=nm, line=i, source_index=len(variants)))
    return EnumBody(
        name="lean_single_ETY_kinds",
        path=rel(path),
        line=variants[0].line if variants else 0,
        variants=variants,
        kind="lean_comment_table",
        notes="seed EXPR_TY/ETY_KIND integer table — MUST NOT be cast against TypeKind ordinals",
    )


def normalize_stem(name: str) -> str:
    """Normalize variant name to a comparable stem. NEVER uses ordinals."""
    s = name
    # const table
    if s.startswith("TYPE_"):
        s = s[len("TYPE_") :]
    # lean
    if s.startswith("ety_mk_"):
        s = s[len("ety_mk_") :]
    if s.startswith("ETY_"):
        s = s[len("ETY_") :]
    # enum prefixes (longest first)
    for pref in (
        "HlirType",
        "Hlir",
        "TypeExpr",
        "Type",
        "Ty",
        "Tk",
        "Gpu",
        "Tc",
        "Lsph",
    ):
        if s.startswith(pref) and len(s) > len(pref):
            s = s[len(pref) :]
            break
    # trailing Type
    if s.endswith("Type") and len(s) > 4:
        s = s[: -len("Type")]
    s = s.lower()
    # alias map (semantic, not ordinal)
    aliases = {
        "void": "unit",
        "mutref": "refmut",
        "ref_mut": "refmut",
        "rawpointer": "rawptr",
        "raw_pointer": "rawptr",
        "ptr": "rawptr",  # layout/gpu ptr ≈ raw pointer — marked alias, bridge must confirm
        "str": "string",
        "function": "fn",
        "float": "f64",
        "int": "i64",
        "inttype": "i64",
        "floattype": "f64",
        "stringtype": "string",
        "booltype": "bool",
        "unittype": "unit",
        "structtype": "struct",
        "enumtype": "enum",
        "arraytype": "array",
        "functiontype": "fn",
        "unknowntype": "unknown",
        "polymorphictype": "var",
        "selfupper": "self",
        "infer": "unknown",
        "error": "error",
        "named": "named",
        "struct": "struct",
        "opaque": "opaque",
        "linear": "linear",
        "affine": "affine",
        "copy": "copy",
        "drop": "drop",
        "primitive": "primitive",
        "generic": "generic",
        "slicemut": "slicemut",
        "slice_mut": "slicemut",
    }
    return aliases.get(s, s)


def load_indices() -> Tuple[List[dict], Set[str], Set[str]]:
    """Load all tests/typekind/**/index.tsv. Returns rows, kinds_in_index, kinds_with_fixture."""
    rows = []
    kinds: Set[str] = set()
    with_fix: Set[str] = set()
    for idx in sorted((ROOT / "tests" / "typekind").rglob("index.tsv")):
        text = idx.read_text(encoding="utf-8", errors="replace")
        header = None
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            parts = line.split("\t")
            if header is None:
                header = [p.strip() for p in parts]
                continue
            # map flexible headers
            rec = {}
            for i, h in enumerate(header):
                rec[h] = parts[i].strip() if i < len(parts) else ""
            kind = rec.get("kind") or rec.get("Kind") or ""
            if not kind or kind == "kind":
                continue
            pass_f = rec.get("pass_fixture") or rec.get("pass") or ""
            refuse_f = rec.get("refuse_fixture") or rec.get("refuse") or ""
            if pass_f == "-":
                pass_f = ""
            if refuse_f == "-":
                refuse_f = ""
            kinds.add(kind)
            has = False
            for f in (pass_f, refuse_f):
                if not f:
                    continue
                fp = ROOT / f
                if fp.is_file() and ".ghost_" not in fp.name:
                    has = True
            if has:
                with_fix.add(kind)
            rows.append(
                {
                    "index": rel(idx),
                    "kind": kind,
                    "pass": pass_f,
                    "refuse": refuse_f,
                    "has_fixture": has,
                }
            )
    return rows, kinds, with_fix


def index_matches_variant(kind: str, variant: str) -> bool:
    """Match index kind name to enum variant without ordinals."""
    if kind == variant:
        return True
    # TyI64 vs I64 vs i64
    ks = normalize_stem(kind)
    vs = normalize_stem(variant)
    if ks == vs:
        return True
    # index often uses TypeKind names without Ty: Distribution vs TyDistribution
    if normalize_stem("Ty" + kind) == vs:
        return True
    if normalize_stem(kind) == normalize_stem("Ty" + variant):
        return True
    return False


def find_bridges(stem: str, enums: List[EnumBody]) -> List[str]:
    """
    Search source for explicit conversion/use that bridges layers for this stem.
    Returns list of evidence strings. Empty => homonym if multi-enum.
    Does NOT use ordinal equality.
    """
    evidence: List[str] = []
    # Build variant names per enum for this stem
    names_by_enum: Dict[str, List[str]] = {}
    for e in enums:
        for v in e.variants:
            if normalize_stem(v.name) == stem:
                names_by_enum.setdefault(e.name + "@" + e.path, []).append(v.name)

    if len(names_by_enum) < 2:
        return evidence

    # Search conversion and lowering files for co-occurrence of two different layer names
    search_files = [
        ROOT / "self-hosted/check/check.sio",
        ROOT / "self-hosted/check/types.sio",
        ROOT / "self-hosted/check/epistemic.sio",
        ROOT / "self-hosted/ir/lower.sio",
        ROOT / "self-hosted/hlir/builder.sio",
        ROOT / "self-hosted/llvm/type_convert.sio",
        ROOT / "self-hosted/check/layout.sio",
        ROOT / "self-hosted/parser/parser.sio",
    ]
    # All variant spellings
    spellings = sorted({n for ns in names_by_enum.values() for n in ns})
    # Also surface syntax tokens for common stems
    surface = {
        "i64": ["i64", "TyI64", "HlirTypeI64", "TkI64", "TYPE_I64"],
        "f64": ["f64", "TyF64", "HlirTypeF64", "TkF64", "TYPE_F64"],
        "bool": ["bool", "TyBool", "HlirTypeBool", "TkBool", "TYPE_BOOL"],
        "knowledge": ["Knowledge", "TyKnowledge", "HlirTypeKnowledge", "TypeKnowledge", "TYPE_KNOWLEDGE"],
        "octonion": ["Octonion", "HlirTypeOctonion", "octonion"],
        "sedenion": ["Sedenion", "HlirTypeSedenion", "sedenion"],
        "quat": ["Quat", "HlirTypeQuat", "quaternion"],
        "array": ["TyArray", "TypeArray", "HlirTypeArray", "TkArray", "TYPE_ARRAY"],
        "rawptr": ["TyRawPtr", "TypeRawPtr", "HlirTypePtr", "TkPtr", "GpuPtr", "TYPE_RAW_POINTER"],
        "unit": ["TyUnit", "TypeUnit", "HlirTypeVoid", "TkUnit", "TYPE_UNIT"],
        "contest": ["TyContest", "TypeContest", "HlirTypeContest"],
        "robust": ["TyRobust", "TypeRobust", "HlirTypeRobust"],
        "intervention": ["TyIntervention", "TypeIntervention", "HlirTypeIntervention"],
        "counterfactual": ["TyCounterfactual", "TypeCounterfactual", "HlirTypeCounterfactual"],
        "validated": ["TyValidated", "TypeValidated", "HlirTypeValidated"],
        "i128": ["TyI128", "HlirTypeI128", "TYPE_I128", "i128"],
        "u128": ["TyU128", "HlirTypeU128", "TYPE_U128", "u128"],
        "f128": ["TyF128", "f128", "TYPE_"],  # may be weak
        "refmut": ["TyRefMut", "TypeRefMut", "TkMutRef"],
        "fn": ["TyFn", "TypeFn", "HlirTypeFunction", "TYPE_FUNCTION"],
    }
    keys = set(spellings)
    keys.update(surface.get(stem, []))

    for fp in search_files:
        if not fp.is_file():
            continue
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Count how many distinct layer spellings appear in this file
        present = [k for k in keys if k in txt]
        # Need at least two different normalized forms from different enums
        layers_hit = set()
        for e in enums:
            for v in e.variants:
                if normalize_stem(v.name) != stem:
                    continue
                if v.name in txt or any(
                    s in txt for s in surface.get(stem, []) if normalize_stem(s) == stem or s == v.name
                ):
                    layers_hit.add(e.path)
        if len(layers_hit) >= 2 and len(present) >= 2:
            evidence.append(f"{rel(fp)}: co-mentions {sorted(present)[:8]} layers={sorted(layers_hit)}")
    return evidence


def main() -> int:
    out_dir = Path(os.environ.get("DENOM_OUT", str(ROOT / "docs/audit/type_enum_denominator")))
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_enum_candidates()
    # Dedupe exact path:line
    uniq = {(e.path, e.line, e.name): e for e in candidates}
    candidates = list(uniq.values())

    classified = []
    for e in candidates:
        bucket, reason = classify_inclusion(e)
        classified.append((bucket, reason, e))

    # Adjacent non-enum tables (always report; not "enums")
    const_table = load_stdlib_type_const_table()
    lean_table = load_lean_ety_comment_table()

    # Inclusion decision for "the N type enums"
    # Defended set: value-type taxonomies in the Madaros pipeline + ownership + layout + gpu
    # Explicit include list (path, name) — if discovery misses one, RULER_SUSPECT
    INCLUDE = [
        ("self-hosted/parser/ast.sio", "TypeExprKind"),
        ("self-hosted/check/types.sio", "TypeKind"),
        ("self-hosted/check/layout.sio", "LayTypeKind"),
        ("self-hosted/check/ownership.sio", "OwnTypeKind"),
        ("self-hosted/hlir/ir.sio", "HlirTypeKind"),
        ("self-hosted/hlir/ir.sio", "HlirTypeDefKind"),
        ("self-hosted/gpu/kernel_ir.sio", "GpuType"),
        ("self-hosted/compiler/parser.sio", "TypeKind"),
    ]
    # LsphTypeCategory and HirType are real enums but off the compile pipeline core —
    # reported as adjacent, not in the 8.

    included: List[EnumBody] = []
    missing_include = []
    by_key = {(e.path, e.name): e for _, _, e in classified}
    for key in INCLUDE:
        if key in by_key:
            included.append(by_key[key])
        else:
            # try discover by reading file directly
            path = ROOT / key[0]
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                found = None
                for i, line in enumerate(text.splitlines(), 1):
                    m = ENUM_RE.search(line)
                    if m and m.group(1) == key[1]:
                        found = EnumBody(
                            name=key[1],
                            path=key[0],
                            line=i,
                            variants=extract_enum_variants(path, i),
                        )
                        break
                if found:
                    included.append(found)
                else:
                    missing_include.append(key)
            else:
                missing_include.append(key)

    n_included = len(included)
    ruler_flags = []
    if n_included != 8:
        ruler_flags.append(f"R1_COUNT N={n_included} expected_claim=8")
    if missing_include:
        ruler_flags.append(f"R4_MISSING_INCLUDE {missing_include}")

    # Deduplicate variants inside an enum (HlirTypeContest appears twice!)
    for e in included:
        seen = set()
        uniq_v = []
        dups = []
        for v in e.variants:
            if v.name in seen:
                dups.append(v.name)
            else:
                seen.add(v.name)
                uniq_v.append(v)
        if dups:
            ruler_flags.append(
                f"R4_DUP_VARIANTS enum={e.name}@{e.path} dups={dups} "
                f"(source lists same name twice; unique count used for denominator)"
            )
            e.notes = (e.notes + f" duplicate_source_names={dups}").strip()
        e.variants = uniq_v

    index_rows, index_kinds, index_with_fix = load_indices()

    # Coverage per enum
    coverage_rows = []
    for e in included:
        covered = []
        for v in e.variants:
            hit = None
            for k in index_kinds:
                if index_matches_variant(k, v.name):
                    hit = k
                    break
            in_idx = hit is not None
            has_fix = bool(hit and hit in index_with_fix)
            covered.append((v, in_idx, has_fix, hit))
        n = len(e.variants)
        n_idx = sum(1 for _, i, _, _ in covered if i)
        n_fix = sum(1 for _, _, f, _ in covered if f)
        coverage_rows.append(
            {
                "enum": e.name,
                "path": e.path,
                "line": e.line,
                "n_variants_unique": n,
                "n_in_index": n_idx,
                "n_with_fixture": n_fix,
                "detail": covered,
            }
        )

    # Cross-layer stems
    stem_to_enums: Dict[str, List[Tuple[EnumBody, Variant]]] = defaultdict(list)
    for e in included:
        for v in e.variants:
            stem_to_enums[normalize_stem(v.name)].append((e, v))

    multi = {s: occ for s, occ in stem_to_enums.items() if len({e.path for e, _ in occ}) > 1}
    cross_rows = []
    for stem in sorted(multi.keys()):
        occ = multi[stem]
        enums_hit = sorted({f"{e.name}@{e.path}:{v.name}" for e, v in occ})
        bridges = find_bridges(stem, included)
        # Also search if conversion file mentions the stem literally for HLIR/checker
        if bridges:
            verdict = "same_type"
        else:
            verdict = "homonym"
        cross_rows.append(
            {
                "stem": stem,
                "verdict": verdict,
                "n_enums": len({e.path for e, _ in occ}),
                "occurrences": enums_hit,
                "bridges": bridges,
            }
        )

    # Write outputs
    sha = os.environ.get("DENOM_SHA", "")
    if not sha:
        try:
            import subprocess

            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
            ).strip()
        except Exception:
            sha = "unknown"

    # TSV: enum coverage
    cov_tsv = out_dir / "enum_coverage.tsv"
    with cov_tsv.open("w") as f:
        f.write(
            "enum\tpath\tline\tn_variants_unique\tn_in_index\tn_with_fixture\tnotes\n"
        )
        for row, e in zip(coverage_rows, included):
            f.write(
                f"{row['enum']}\t{row['path']}\t{row['line']}\t"
                f"{row['n_variants_unique']}\t{row['n_in_index']}\t{row['n_with_fixture']}\t"
                f"{e.notes}\n"
            )

    # TSV: every candidate discovery
    cand_tsv = out_dir / "candidates.tsv"
    with cand_tsv.open("w") as f:
        f.write("bucket\treason\tname\tpath\tline\tn_variants\n")
        for bucket, reason, e in sorted(classified, key=lambda x: (x[0], x[2].path, x[2].line)):
            f.write(
                f"{bucket}\t{reason}\t{e.name}\t{e.path}\t{e.line}\t{len(e.variants)}\n"
            )
        f.write(
            f"adjacent_const_table\t{const_table.notes}\t{const_table.name}\t"
            f"{const_table.path}\t{const_table.line}\t{len(const_table.variants)}\n"
        )
        f.write(
            f"adjacent_lean_table\t{lean_table.notes}\t{lean_table.name}\t"
            f"{lean_table.path}\t{lean_table.line}\t{len(lean_table.variants)}\n"
        )

    # TSV: cross-layer
    cross_tsv = out_dir / "cross_layer.tsv"
    with cross_tsv.open("w") as f:
        f.write("stem\tverdict\tn_enums\toccurrences\tbridge_evidence\n")
        for r in cross_rows:
            br = " | ".join(r["bridges"][:3]) if r["bridges"] else ""
            occ = ";".join(r["occurrences"])
            f.write(f"{r['stem']}\t{r['verdict']}\t{r['n_enums']}\t{occ}\t{br}\n")

    # Per-variant detail for included enums
    det_tsv = out_dir / "variant_index_coverage.tsv"
    with det_tsv.open("w") as f:
        f.write(
            "enum\tpath\tvariant\tvariant_line\tsource_index_AUDIT_ONLY\tin_index\thas_fixture\tindex_kind\n"
        )
        for row in coverage_rows:
            for v, in_idx, has_fix, hit in row["detail"]:
                f.write(
                    f"{row['enum']}\t{row['path']}\t{v.name}\t{v.line}\t{v.source_index}\t"
                    f"{int(in_idx)}\t{int(has_fix)}\t{hit or ''}\n"
                )

    # Summary markdown fragment
    summary = out_dir / "MEASUREMENT_RECEIPT.txt"
    total_unique_stems = len(stem_to_enums)
    multi_same = sum(1 for r in cross_rows if r["verdict"] == "same_type")
    multi_hom = sum(1 for r in cross_rows if r["verdict"] == "homonym")
    sum_variants = sum(len(e.variants) for e in included)
    # Denominator options
    denom_sum = sum_variants
    denom_union_stems = total_unique_stems

    lines = []
    lines.append(f"sha={sha}")
    lines.append(f"root={ROOT}")
    lines.append(f"n_included_enums={n_included}")
    lines.append(f"claimed_eight=8")
    lines.append(f"r1_match={n_included == 8}")
    lines.append(f"ruler_flags={ruler_flags}")
    lines.append(f"sum_variants_across_included={sum_variants}")
    lines.append(f"union_normalized_stems={total_unique_stems}")
    lines.append(f"cross_layer_stems={len(cross_rows)} same_type={multi_same} homonym={multi_hom}")
    lines.append(f"index_kinds={len(index_kinds)} with_fixture={len(index_with_fix)}")
    lines.append(f"index_rows={len(index_rows)}")
    lines.append("--- included ---")
    for row in coverage_rows:
        lines.append(
            f"{row['enum']}\t{row['path']}:{row['line']}\t"
            f"variants={row['n_variants_unique']}\tin_index={row['n_in_index']}\t"
            f"fixture={row['n_with_fixture']}"
        )
    lines.append("--- adjacent non-enum tables ---")
    lines.append(
        f"{const_table.name}\t{const_table.path}\tvariants={len(const_table.variants)}"
    )
    lines.append(
        f"{lean_table.name}\t{lean_table.path}\tvariants={len(lean_table.variants)}"
    )
    summary.write_text("\n".join(lines) + "\n")

    # Machine-readable JSON-ish key file for the doc
    print("\n".join(lines))
    print(f"WROTE {cov_tsv}")
    print(f"WROTE {cand_tsv}")
    print(f"WROTE {cross_tsv}")
    print(f"WROTE {det_tsv}")
    print(f"WROTE {summary}")

    # Exit nonzero only on hard ruler failure that voids the measurement
    if missing_include:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
