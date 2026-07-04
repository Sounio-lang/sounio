#!/usr/bin/env python3
"""Emit a Madaros v2 S3 compiler-native HLIR receipt.

S3 is the first SOTA+++ plan step that exposes a compiler-native lowered IR
artifact: source -> parser/typecheck -> hlir_lower_module -> canonical JSON.
The receipt proves the emitted HLIR is clean JSON, deterministic byte-for-byte,
parseable, and roundtrippable through a canonical JSON hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = "madaros.v2.s3.receipt/0.1"
HLIR_SCHEMA = "madaros.hlir.module/0.2"
HLIR_SOURCE_TO_HLIR = "compiler_native_hlir_lower_module"


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


def run_emit_hlir(compiler: Path, source_arg: str, root: Path, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        [str(compiler), "--emit-hlir", source_arg],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def validate_hlir(text: str, source_rel: str) -> tuple[dict, dict]:
    if not text.startswith('{"schema":"madaros.hlir.module/0.2"'):
        raise SystemExit("HLIR output is not clean JSON; banner or diagnostics leaked to stdout")
    if "\n" in text.rstrip("\n"):
        raise SystemExit("HLIR output must be one canonical JSON object line")
    data = json.loads(text)
    if data.get("schema") != HLIR_SCHEMA:
        raise SystemExit(f"bad HLIR schema: {data.get('schema')!r}")
    if data.get("stage") != "S3":
        raise SystemExit(f"bad stage: {data.get('stage')!r}")
    if data.get("source") != source_rel:
        raise SystemExit(f"bad source field: {data.get('source')!r} != {source_rel!r}")
    if data.get("source_to_hlir") != HLIR_SOURCE_TO_HLIR:
        raise SystemExit("HLIR was not produced by compiler_native_hlir_lower_module")
    if data.get("ownership_effect_normalization") != "hlir_lower_module_v0":
        raise SystemExit("missing ownership/effect normalization marker")
    if data.get("normalized_ids") is not True:
        raise SystemExit("normalized_ids must be true")

    module = data.get("module")
    if not isinstance(module, dict):
        raise SystemExit("module must be an object")
    functions = module.get("functions")
    globals_ = module.get("globals")
    typedefs = module.get("typedefs")
    if module.get("function_count") != len(functions):
        raise SystemExit("function_count mismatch")
    if module.get("global_count") != len(globals_):
        raise SystemExit("global_count mismatch")
    if module.get("typedef_count") != len(typedefs):
        raise SystemExit("typedef_count mismatch")
    if not functions:
        raise SystemExit("S3 HLIR must contain at least one function")

    ops: set[str] = set()
    terminators: set[str] = set()
    calls: set[str] = set()
    const_kinds: set[str] = set()
    instr_total = 0
    for func in functions:
        if func["param_count"] != len(func["params"]):
            raise SystemExit(f"param_count mismatch in {func['name']}")
        if func["effect_count"] != len(func["effects"]):
            raise SystemExit(f"effect_count mismatch in {func['name']}")
        if func["block_count"] != len(func["blocks"]):
            raise SystemExit(f"block_count mismatch in {func['name']}")
        for block in func["blocks"]:
            if block["param_count"] != len(block["params"]):
                raise SystemExit(f"block param_count mismatch in {func['name']}")
            if block["instr_count"] != len(block["instrs"]):
                raise SystemExit(f"instr_count mismatch in {func['name']}/{block['label']}")
            instr_total += block["instr_count"]
            terminators.add(block["terminator"]["kind"])
            for instr in block["instrs"]:
                ops.add(instr["op"])
                call = instr.get("call_name", "")
                if call:
                    calls.add(call)
                const_kinds.add(instr.get("constant", {}).get("kind", ""))
    if instr_total <= 0:
        raise SystemExit("S3 HLIR must contain real instructions")

    facts = {
        "function_count": len(functions),
        "global_count": len(globals_),
        "typedef_count": len(typedefs),
        "instruction_count": instr_total,
        "ops": sorted(op for op in ops if op),
        "terminators": sorted(term for term in terminators if term),
        "calls": sorted(calls),
        "const_kinds": sorted(kind for kind in const_kinds if kind),
    }
    return data, facts


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_id = args.case_id or source.stem
    source_rel = relpath(source, root)
    source_arg = source_rel if not source_rel.startswith("/") else str(source)
    rc_a, out_a, err_a = run_emit_hlir(compiler, source_arg, root, args.timeout)
    rc_b, out_b, err_b = run_emit_hlir(compiler, source_arg, root, args.timeout)
    if rc_a != 0 or rc_b != 0:
        raise SystemExit(f"--emit-hlir failed: rc_a={rc_a} rc_b={rc_b}\n{err_a}{err_b}")
    if err_a or err_b:
        raise SystemExit(f"--emit-hlir wrote stderr:\n{err_a}{err_b}")
    if out_a != out_b:
        raise SystemExit("--emit-hlir is not byte-deterministic")

    data, facts = validate_hlir(out_a, source_rel)
    canonical = stable_json(data)
    hlir_path = out_dir / f"{case_id}.s3.hlir.json"
    receipt_path = out_dir / f"{case_id}.s3.receipt.json"
    hlir_path.write_text(out_a, encoding="utf-8")
    receipt = {
        "schema": SCHEMA_VERSION,
        "case_id": case_id,
        "source": source_rel,
        "source_sha256": sha256_bytes(source.read_bytes()),
        "compiler": str(compiler),
        "compiler_route_kind": args.compiler_route_kind,
        "parser_sha": args.parser_sha,
        "hlir_schema": data["schema"],
        "source_to_hlir": data["source_to_hlir"],
        "ownership_effect_normalization": data["ownership_effect_normalization"],
        "normalized_ids": data["normalized_ids"],
        "facts": facts,
        "hlir_json_path": hlir_path.name,
        "hlir_byte_sha256": sha256_text(out_a),
        "hlir_canonical_roundtrip_sha256": sha256_text(canonical),
        "deterministic_reemit": True,
        "s4_ready": True,
        "s4_ready_contract": "egraph_ekan_optimizer_can_consume_hlir_json_and_hash",
    }
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt["receipt_sha256"] = sha256_text(payload)
    payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    receipt_path.write_text(payload, encoding="utf-8")
    print(f"madaros-v2-s3: wrote {receipt_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--source", required=True)
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--compiler", default=str(repo_root_from_script() / "bin" / "madaros"))
    emit_p.add_argument("--compiler-route-kind", default="madaros-wrapper")
    emit_p.add_argument("--parser-sha", default="unknown")
    emit_p.add_argument("--case-id", default="")
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--timeout", type=int, default=120)
    emit_p.set_defaults(func=emit)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
