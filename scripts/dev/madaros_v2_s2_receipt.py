#!/usr/bin/env python3
"""Emit a Madaros v2 S2 contract/audit scaffold receipt witness.

S2 is the typed front-end boundary. The current compiler does not yet expose a
full compiler-native typed HIR/THIR serializer, so this receipt is explicit
about its scope: it links to a deterministic S1 receipt and records stable
source-scan tables for public symbols, imports, effects, refinements,
epistemic declarations, and structured compiler diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = "madaros.v2.s2.receipt/0.1"
CLAIM_LEVEL = "s2_contract_scaffold"
S2_STATUS = "no_current_madaros_typed_hir_serializer"
TYPED_HIR_STATUS = "not_emitted_by_current_madaros"
TYPED_HIR_ROUNDTRIP_STATUS = "not_available"
EFFECT_RE = re.compile(r"\bwith\s+([A-Za-z0-9_,\s]+)")
FN_RE = re.compile(r"^\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*([A-Za-z_][A-Za-z0-9_<>,:\[\]\s]*))?")
STRUCT_RE = re.compile(r"^\s*(pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)")
ENUM_RE = re.compile(r"^\s*(pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)")
TYPE_TOKEN_RE = re.compile(r"\b(?:Knowledge|Epistemic|Validated|Contest|Robust|PBox|GUM|Interval|Fuzzy|Ekan|EKAN|KAN)<[^>\n]+>|\b(?:f32|f64|i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|bool|str|String)\b")
REFINEMENT_RE = re.compile(r"\b(?:requires|ensures|invariant|assert|prove|proof|Validated|Refined|within|epsilon|confidence)\b", re.I)
EPISTEMIC_RE = re.compile(r"\b(?:Knowledge|Epistemic|Validated|Contest|Robust|PBox|GUM|Knightian|Walley|aleatoric|epistemic|uncertainty|epsilon|provenance|Ekan|EKAN|KAN)\b")
DIAG_CODE_RE = re.compile(r"\b(E[0-9]+)\b")
SEVERITY_RE = re.compile(r"\b(error|warning|panic|segmentation fault|SIGSEGV)\b", re.I)


def load_s1_module(root: Path):
    path = root / "scripts" / "dev" / "madaros_v2_s1_receipt.py"
    spec = importlib.util.spec_from_file_location("madaros_v2_s1_receipt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load S1 receipt module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def split_effects(text: str) -> list[str]:
    effects: list[str] = []
    for match in EFFECT_RE.finditer(text):
        for item in match.group(1).split(","):
            effect = item.strip()
            if effect and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", effect) and effect not in effects:
                effects.append(effect)
    return sorted(effects)


def extract_type_tokens(text: str) -> list[str]:
    return sorted(set(match.group(0).strip() for match in TYPE_TOKEN_RE.finditer(text)))


def parse_symbols(s1, module: dict, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    path = root / module["path"]
    text = s1.read_text(path)
    public_symbols: list[dict] = []
    effect_rows: list[dict] = []
    refinement_rows: list[dict] = []
    epistemic_rows: list[dict] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("//", 1)[0].rstrip()
        for match in FN_RE.finditer(line):
            name = match.group(2)
            is_public = bool(match.group(1))
            effects = split_effects(line)
            ret = (match.group(4) or "").strip()
            params = match.group(3).strip()
            type_tokens = extract_type_tokens(params + " " + ret)
            row = {
                "module_path": module["path"],
                "line": line_no,
                "kind": "fn",
                "name": name,
                "public": is_public,
                "return_type": ret,
                "type_tokens": type_tokens,
            }
            if is_public:
                public_symbols.append(row)
            effect_rows.append({
                "module_path": module["path"],
                "line": line_no,
                "symbol": name,
                "effects": effects,
            })
        for regex, kind in ((STRUCT_RE, "struct"), (ENUM_RE, "enum")):
            match = regex.match(line)
            if match and match.group(1):
                public_symbols.append({
                    "module_path": module["path"],
                    "line": line_no,
                    "kind": kind,
                    "name": match.group(2),
                    "public": True,
                    "return_type": "",
                    "type_tokens": [],
                })
        if REFINEMENT_RE.search(line):
            refinement_rows.append({
                "module_path": module["path"],
                "line": line_no,
                "tokens": sorted(set(match.group(0) for match in REFINEMENT_RE.finditer(line))),
                "line_sha256": s1.sha256_text(line.strip()),
            })
        if EPISTEMIC_RE.search(line):
            epistemic_rows.append({
                "module_path": module["path"],
                "line": line_no,
                "tokens": sorted(set(match.group(0) for match in EPISTEMIC_RE.finditer(line))),
                "line_sha256": s1.sha256_text(line.strip()),
            })

    return public_symbols, effect_rows, refinement_rows, epistemic_rows


def structured_diagnostics(s1, compiler: Path, source: Path, root: Path, timeout_s: int) -> dict:
    proc = subprocess.run(
        [str(compiler), "check", str(source)],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    out = proc.stdout or ""
    entries: list[dict] = []
    by_code: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for line in out.splitlines():
        severity_match = SEVERITY_RE.search(line)
        code_match = DIAG_CODE_RE.search(line)
        if not severity_match and not code_match:
            continue
        severity = (severity_match.group(1).lower() if severity_match else "diagnostic").replace("segmentation fault", "segfault")
        code = code_match.group(1) if code_match else "NO_CODE"
        by_code[code] = by_code.get(code, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1
        entries.append({
            "severity": severity,
            "code": code,
            "line_sha256": s1.sha256_text(line.strip()),
        })
    return {
        "compiler_check_rc": proc.returncode,
        "compiler_check_status": "check_ok" if proc.returncode == 0 else "check_failed",
        "raw_output_sha256": s1.sha256_text(out),
        "diagnostic_count": len(entries),
        "by_code": dict(sorted(by_code.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "entries": entries,
    }


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(item) for item in row) + "\n")


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    compiler = Path(args.compiler).resolve() if args.compiler else root / "bin" / "madaros"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1 = load_s1_module(root)

    stem = args.case_id or source.stem
    s1_dir = out_dir / "s1"
    s1_args = argparse.Namespace(
        root=str(root),
        source=str(source),
        out_dir=str(s1_dir),
        compiler=str(compiler),
        compiler_route_kind=args.compiler_route_kind,
        case_id=stem,
        parser_sha=args.parser_sha,
        max_modules=args.max_modules,
        timeout_s=args.timeout_s,
        deterministic_time=True,
    )
    s1.emit(s1_args)
    s1_receipt_path = s1_dir / f"{stem}.s1.receipt.json"
    s1_ast_path = s1_dir / f"{stem}.s1.ast.json"
    s1_receipt = json.loads(s1_receipt_path.read_text(encoding="utf-8"))

    public_symbols: list[dict] = []
    effect_rows: list[dict] = []
    refinement_rows: list[dict] = []
    epistemic_rows: list[dict] = []
    for module in s1_receipt["module_graph"]:
        pub, eff, refn, epi = parse_symbols(s1, module, root)
        public_symbols.extend(pub)
        effect_rows.extend(eff)
        refinement_rows.extend(refn)
        epistemic_rows.extend(epi)

    public_symbols.sort(key=lambda row: (row["module_path"], row["line"], row["kind"], row["name"]))
    effect_rows.sort(key=lambda row: (row["module_path"], row["line"], row["symbol"]))
    refinement_rows.sort(key=lambda row: (row["module_path"], row["line"], row["line_sha256"]))
    epistemic_rows.sort(key=lambda row: (row["module_path"], row["line"], row["line_sha256"]))
    diagnostics = structured_diagnostics(s1, compiler, source, root, args.timeout_s)

    public_symbols_path = out_dir / f"{stem}.s2.public_symbols.tsv"
    import_audit_path = out_dir / f"{stem}.s2.import_audit.tsv"
    effects_path = out_dir / f"{stem}.s2.effects.tsv"
    refinements_path = out_dir / f"{stem}.s2.refinements.tsv"
    epistemic_decls_path = out_dir / f"{stem}.s2.epistemic_decls.tsv"
    diagnostics_path = out_dir / f"{stem}.s2.diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    import_rows = []
    for module in s1_receipt["module_graph"]:
        for item in module["imports"]:
            import_rows.append([module["path"], item])
    for item in s1_receipt["unresolved_imports"]:
        import_rows.append([item["from"], f"{item['module']} ({item['reason']})"])
    import_rows.sort()

    write_tsv(
        public_symbols_path,
        ["module_path", "line", "kind", "name", "return_type", "type_tokens"],
        [[row["module_path"], row["line"], row["kind"], row["name"], row["return_type"], ",".join(row["type_tokens"])] for row in public_symbols],
    )
    write_tsv(import_audit_path, ["from_path", "import"], import_rows)
    write_tsv(effects_path, ["module_path", "line", "symbol", "effects"], [[row["module_path"], row["line"], row["symbol"], ",".join(row["effects"])] for row in effect_rows])
    write_tsv(refinements_path, ["module_path", "line", "tokens", "line_sha256"], [[row["module_path"], row["line"], ",".join(row["tokens"]), row["line_sha256"]] for row in refinement_rows])
    write_tsv(epistemic_decls_path, ["module_path", "line", "tokens", "line_sha256"], [[row["module_path"], row["line"], ",".join(row["tokens"]), row["line_sha256"]] for row in epistemic_rows])

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "claim_level": CLAIM_LEVEL,
        "s2_complete": False,
        "s2_status": S2_STATUS,
        "case_id": stem,
        "source_path": s1.relpath(source, root),
        "source_sha256": s1.sha256_bytes(source.read_bytes()),
        "parser_sha": args.parser_sha,
        "compiler_route": str(compiler),
        "compiler_route_kind": args.compiler_route_kind,
        "upstream_s1": {
            "schema_version": s1_receipt["schema_version"],
            "receipt_sha256": s1.sha256_bytes(s1_receipt_path.read_bytes()),
            "receipt_relpath": f"s1/{stem}.s1.receipt.json",
            "canonical_ast_sha256": s1.sha256_bytes(s1_ast_path.read_bytes()),
            "module_graph_sha256": s1_receipt["module_graph_sha256"],
            "ast_serializer_version": s1_receipt["ast_serializer_version"],
        },
        "typed_hir_sha256": None,
        "typed_hir_status": TYPED_HIR_STATUS,
        "typed_hir_roundtrip_status": TYPED_HIR_ROUNDTRIP_STATUS,
        "effect_table_sha256": s1.stable_json_sha(effect_rows),
        "import_audit_table_sha256": s1.sha256_bytes(import_audit_path.read_bytes()),
        "public_symbol_table_sha256": s1.stable_json_sha(public_symbols),
        "refinement_table_sha256": s1.stable_json_sha(refinement_rows),
        "epistemic_declaration_table_sha256": s1.stable_json_sha(epistemic_rows),
        "diagnostic_table_sha256": s1.sha256_bytes(diagnostics_path.read_bytes()),
        "table_status": {
            "public_symbols": "source_scan_surrogate",
            "imports": "s1_module_graph_plus_source_scan",
            "effects": "source_scan_surrogate_or_empty",
            "refinements": "source_scan_surrogate_or_empty",
            "epistemic_declarations": "source_scan_surrogate_or_empty",
        },
        "structured_diagnostics": {
            "compiler_check_rc": diagnostics["compiler_check_rc"],
            "compiler_check_status": diagnostics["compiler_check_status"],
            "diagnostic_count": diagnostics["diagnostic_count"],
            "by_code": diagnostics["by_code"],
            "by_severity": diagnostics["by_severity"],
            "raw_output_sha256": diagnostics["raw_output_sha256"],
        },
        "visibility_audit": {
            "observed_modules": len(s1_receipt["module_graph"]),
            "public_symbol_count": len(public_symbols),
            "effectful_symbol_count": sum(1 for row in effect_rows if row["effects"]),
            "refinement_row_count": len(refinement_rows),
            "epistemic_declaration_count": len(epistemic_rows),
        },
        "phase_caps": {
            "max_modules": args.max_modules,
            "observed_modules": len(s1_receipt["module_graph"]),
            "unresolved_imports": len(s1_receipt["unresolved_imports"]),
            "typed_hir_surface": "none_until_compiler_native_thir_serializer",
        },
        "generated_at_utc": "1970-01-01T00:00:00Z" if args.deterministic_time else _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    receipt_for_hash = dict(receipt)
    receipt_for_hash["receipt_sha256"] = ""
    receipt["receipt_sha256"] = s1.stable_json_sha(receipt_for_hash)
    receipt_path = out_dir / f"{stem}.s2.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"receipt={receipt_path}")
    print(f"typed_hir_status={receipt['typed_hir_status']}")
    print(f"public_symbols={public_symbols_path}")
    print(f"import_audit={import_audit_path}")
    print(f"diagnostics={diagnostics_path}")
    print(f"receipt_sha256={receipt['receipt_sha256']}")
    print(f"compiler_check={diagnostics['compiler_check_status']} rc={diagnostics['compiler_check_rc']}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emit", nargs="?")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--compiler", default=str(Path(__file__).resolve().parents[2] / "bin" / "madaros"))
    parser.add_argument("--compiler-route-kind", default="madaros-wrapper")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--parser-sha", default="unknown")
    parser.add_argument("--max-modules", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--deterministic-time", action="store_true")
    args = parser.parse_args(argv)
    if args.emit not in (None, "emit"):
        parser.error("only the 'emit' subcommand is supported")
    return emit(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
