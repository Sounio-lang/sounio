#!/usr/bin/env python3
"""Emit a Madaros v2 S1 AST/source/module-graph receipt.

S1 is intentionally before type checking and lowering. This tool records a
deterministic compiler-native Stage1 AST sidecar, source/import graph, and
compiler check witness for the source being observed. The source graph remains
as a secondary L1 witness; canonical_ast_sha256 is the S1 completion witness.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = "madaros.v2.s1.receipt/0.2"
AST_BOUNDARY = "stage1_parser_top_level_ast"
AST_SERIALIZER_VERSION = "madaros.stage1.ast/0.1"
SOURCE_GRAPH_BOUNDARY = "s1_l1_text_import_public_symbol_surrogate"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
IMPORT_RE = re.compile(r"^\s*use\s+(.+?)(?:\s*;)?\s*$")
MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_:]*)")
PUB_SYMBOL_RE = re.compile(r"^\s*pub\s+(?:fn|struct|enum|type|const|let)\s+([A-Za-z_][A-Za-z0-9_]*)")
ITEM_RE = re.compile(r"^\s*(?:pub\s+)?(?:fn|struct|enum|type|const|let|extern)\b")
DIAG_RE = re.compile(r"(?:^|\b)(?:error(?:\[E[0-9]+\])?|warning|panic|segmentation fault|SIGSEGV)\b", re.I)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_source(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n") + "\n"


def parse_imports(text: str) -> list[str]:
    imports: list[str] = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        match = IMPORT_RE.match(line)
        if not match:
            continue
        module = match.group(1).strip()
        if " as " in module:
            module = module.split(" as ", 1)[0].strip()
        if "{" in module:
            module = module.split("{", 1)[0].rstrip(":").strip()
        if module.endswith("::*"):
            module = module[:-3]
        module = module.rstrip(":").strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_:]*$", module):
            continue
        if module not in imports:
            imports.append(module)
    return imports


def parse_module_name(text: str) -> str:
    for line in text.splitlines():
        match = MODULE_RE.match(line.split("//", 1)[0])
        if match:
            return match.group(1)
    return ""


def count_public_symbols(text: str) -> int:
    return sum(1 for line in text.splitlines() if PUB_SYMBOL_RE.match(line.split("//", 1)[0]))


def count_items(text: str) -> int:
    return sum(1 for line in text.splitlines() if ITEM_RE.match(line.split("//", 1)[0]))


def module_candidates(module: str, source: Path, root: Path) -> list[Path]:
    parts = [p for p in module.split("::") if p and p != "*"]
    if not parts:
        return []
    rel = Path(*parts)
    roots = [
        source.parent,
        root / "self-hosted",
        root / "stdlib",
        root / "tests",
        root,
    ]
    candidates: list[Path] = []
    for base in roots:
        candidates.append(base / rel.with_suffix(".sio"))
        candidates.append(base / rel / "mod.sio")
        if len(parts) > 1:
            candidates.append(base / parts[0] / Path(*parts[1:]).with_suffix(".sio"))
            candidates.append(base / parts[0] / Path(*parts[1:]) / "mod.sio")
    seen: set[Path] = set()
    unique: list[Path] = []
    for cand in candidates:
        resolved = cand.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(cand)
    return unique


def resolve_module(module: str, source: Path, root: Path) -> Path | None:
    for candidate in module_candidates(module, source, root):
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_graph(source: Path, root: Path, max_modules: int) -> tuple[list[dict], list[dict]]:
    modules: list[dict] = []
    unresolved: list[dict] = []
    path_to_id: dict[Path, int] = {}
    queue: list[Path] = [source.resolve()]

    while queue:
        path = queue.pop(0)
        if path in path_to_id:
            continue
        if len(modules) >= max_modules:
            unresolved.append({"from": relpath(path, root), "module": "<phase-cap>", "reason": "max_modules"})
            continue
        text = read_text(path)
        raw = path.read_bytes()
        module_id = len(modules)
        path_to_id[path] = module_id
        imports = parse_imports(text)
        module_entry = {
            "module_id": module_id,
            "path": relpath(path, root),
            "module_name": parse_module_name(text),
            "sha256": sha256_bytes(raw),
            "normalized_source_sha256": sha256_text(normalize_source(text)),
            "imports": imports,
            "import_module_ids": [],
            "public_symbol_count": count_public_symbols(text),
            "item_count": count_items(text),
        }
        modules.append(module_entry)
        for module in imports:
            resolved = resolve_module(module, path, root)
            if resolved is None:
                unresolved.append({"from": relpath(path, root), "module": module, "reason": "not_found"})
                continue
            if resolved not in path_to_id and resolved not in queue:
                queue.append(resolved)

    # Fill import ids after all reachable modules have IDs.
    for entry in modules:
        path = (root / entry["path"]).resolve() if not os.path.isabs(entry["path"]) else Path(entry["path"]).resolve()
        ids: list[int] = []
        for module in entry["imports"]:
            resolved = resolve_module(module, path, root)
            if resolved is not None and resolved in path_to_id:
                ids.append(path_to_id[resolved])
        entry["import_module_ids"] = sorted(set(ids))

    modules.sort(key=lambda item: (item["module_id"], item["path"]))
    unresolved.sort(key=lambda item: (item["from"], item["module"], item["reason"]))
    return modules, unresolved


def run_compiler_check(compiler: Path, source: Path, timeout_s: int) -> dict:
    if not compiler:
        return {
            "compiler_check_rc": -1,
            "compiler_check_status": "not_run",
            "compiler_check_output_sha256": EMPTY_SHA256,
            "compiler_check_diagnostic_count": 0,
            "compiler_check_output_tail": "",
        }
    proc = subprocess.run(
        [str(compiler), "check", str(source)],
        cwd=str(repo_root_from_script()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    out = proc.stdout or ""
    diagnostics = sum(1 for line in out.splitlines() if DIAG_RE.search(line))
    return {
        "compiler_check_rc": proc.returncode,
        "compiler_check_status": "check_ok" if proc.returncode == 0 else "check_failed",
        "compiler_check_output_sha256": sha256_text(out),
        "compiler_check_diagnostic_count": diagnostics,
        "compiler_check_output_tail": "\n".join(out.splitlines()[-12:]),
    }


def run_compiler_emit_ast(compiler: Path, source: Path, root: Path, timeout_s: int, ast_path: Path) -> dict:
    source_arg = relpath(source, root)
    proc = subprocess.run(
        [str(compiler), "--emit-ast", source_arg],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    out = proc.stdout or ""
    ast_lines = [
        line
        for line in out.splitlines()
        if line.startswith("{\"schema\":\"" + AST_SERIALIZER_VERSION + "\"")
    ]
    if proc.returncode != 0 or len(ast_lines) != 1:
        ast_path.write_text("", encoding="utf-8")
        return {
            "ast_emit_rc": proc.returncode,
            "ast_emit_status": "emit_ast_failed",
            "ast_emit_output_sha256": sha256_text(out),
            "ast_emit_output_tail": "\n".join(out.splitlines()[-12:]),
        }

    ast_bytes = (ast_lines[0] + "\n").encode("utf-8")
    ast_path.write_bytes(ast_bytes)
    return {
        "ast_emit_rc": proc.returncode,
        "ast_emit_status": "emit_ast_ok",
        "ast_emit_output_sha256": sha256_text(out),
        "ast_emit_output_tail": "\n".join(out.splitlines()[-12:]),
    }


def stable_json_sha(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(data)


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()
    compiler = Path(args.compiler).resolve() if args.compiler else root / "bin" / "madaros"
    out_dir.mkdir(parents=True, exist_ok=True)

    modules, unresolved = build_graph(source, root, args.max_modules)
    module_graph_payload = {"modules": modules, "unresolved_imports": unresolved}
    module_graph_sha = stable_json_sha(module_graph_payload)
    canonical_source_graph_payload = {
        "boundary": SOURCE_GRAPH_BOUNDARY,
        "source": relpath(source, root),
        "modules": [
            {
                "path": item["path"],
                "module_name": item["module_name"],
                "normalized_source_sha256": item["normalized_source_sha256"],
                "imports": item["imports"],
                "public_symbol_count": item["public_symbol_count"],
                "item_count": item["item_count"],
            }
            for item in modules
        ],
        "unresolved_imports": unresolved,
    }
    stem = args.case_id or source.stem
    ast_path = out_dir / f"{stem}.s1.ast.json"
    ast_emit = run_compiler_emit_ast(compiler, source, root, args.timeout_s, ast_path)
    ast_bytes = ast_path.read_bytes()
    ast_sha = sha256_bytes(ast_bytes)
    compiler_witness = run_compiler_check(compiler, source, args.timeout_s)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "case_id": args.case_id or source.stem,
        "source_path": relpath(source, root),
        "source_sha256": sha256_bytes(source.read_bytes()),
        "parser_sha": args.parser_sha,
        "compiler_route": str(compiler),
        "compiler_route_kind": args.compiler_route_kind,
        "module_graph": modules,
        "module_graph_sha256": module_graph_sha,
        "unresolved_imports": unresolved,
        "canonical_ast_sha256": ast_sha,
        "canonical_ast_relpath": ast_path.name,
        "canonical_ast_status": "stable_stage1_ast_serializer",
        "ast_surface_kind": "compiler_native_top_level_ast_json",
        "ast_boundary": AST_BOUNDARY,
        "ast_serializer_version": AST_SERIALIZER_VERSION,
        "canonical_source_graph_sha256": stable_json_sha(canonical_source_graph_payload),
        "canonical_source_graph_status": "stable_l1_source_import_public_symbol_surrogate",
        "public_symbol_count": sum(item["public_symbol_count"] for item in modules),
        "item_count": sum(item["item_count"] for item in modules),
        "diagnostic_count": compiler_witness["compiler_check_diagnostic_count"],
        "diagnostics_sha256": compiler_witness["compiler_check_output_sha256"],
        "phase_caps": {
            "max_modules": args.max_modules,
            "observed_modules": len(modules),
            "unresolved_imports": len(unresolved),
        },
        "ast_emit": ast_emit,
        "compiler_check": compiler_witness,
        "generated_at_utc": "1970-01-01T00:00:00Z" if args.deterministic_time else _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    receipt_for_hash = dict(receipt)
    receipt_for_hash["receipt_sha256"] = ""
    receipt["receipt_sha256"] = stable_json_sha(receipt_for_hash)

    receipt_path = out_dir / f"{stem}.s1.receipt.json"
    edges_path = out_dir / f"{stem}.s1.module_edges.tsv"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with edges_path.open("w", encoding="utf-8") as fh:
        fh.write("from_module_id\tfrom_path\tto_module_id\tto_path\timport\n")
        for entry in modules:
            for module in entry["imports"]:
                resolved = resolve_module(module, (root / entry["path"]).resolve(), root)
                to_id = ""
                to_path = ""
                if resolved is not None:
                    for target in modules:
                        if (root / target["path"]).resolve() == resolved:
                            to_id = str(target["module_id"])
                            to_path = target["path"]
                            break
                fh.write(f"{entry['module_id']}\t{entry['path']}\t{to_id}\t{to_path}\t{module}\n")

    print(f"receipt={receipt_path}")
    print(f"canonical_ast={ast_path}")
    print(f"canonical_ast_sha256={receipt['canonical_ast_sha256']}")
    print(f"module_edges={edges_path}")
    print(f"receipt_sha256={receipt['receipt_sha256']}")
    print(f"module_graph_sha256={receipt['module_graph_sha256']}")
    print(f"compiler_check={compiler_witness['compiler_check_status']} rc={compiler_witness['compiler_check_rc']}")
    if unresolved:
        print(f"unresolved_imports={len(unresolved)}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("emit", nargs="?")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--root", default=str(repo_root_from_script()))
    parser.add_argument("--compiler", default=str(repo_root_from_script() / "bin" / "madaros"))
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
