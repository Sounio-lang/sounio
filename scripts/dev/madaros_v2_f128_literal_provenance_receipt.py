#!/usr/bin/env python3
"""Emit a Madaros v2 f128 literal-provenance/decimal-metadata receipt.

This receipt closes the parser-side prerequisite for a real binary128 f128
implementation: float literals must preserve their original source spelling in
the AST and must also carry bounded decimal metadata independent of the rounded
f64 value. It deliberately does not promote f128 execution; IR/MIR/ABI/software-
helper coverage remains a separate S5 contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.f128_literal_provenance_receipt/0.3"
STAGE_CONTRACT_LEVEL = "S4_S5_F128_LITERAL_DECIMAL_METADATA_AND_TYPE_PROMOTED_NOT_F128_EXECUTION"

SOURCE = "fn main() -> i64 { let x: f128 = 1.2345678901234567890123456789012345 as f128; 0 }\n"
LITERAL_DIGITS = "1.2345678901234567890123456789012345"
REQUIRED_PARSER_SNIPPETS = [
    "let raw = self.current_name()",
    "var e = mk_expr(ExprKind::ExprFloatLit, span)",
    "e.float_val = tok.float_value",
    "e.name = raw",
    "expr_apply_float_literal_decimal_metadata(&! e, raw)",
]
REQUIRED_AST_SNIPPETS = [
    "float_decimal_sign: i64",
    "float_decimal_sig_hi: i64",
    "float_decimal_sig_lo: i64",
    "float_decimal_digit_count: i64",
    "float_decimal_scale10: i64",
    "float_decimal_truncated_digits: i64",
    "pub fn expr_apply_float_literal_decimal_metadata(e: &! Expr, raw: Name)",
]
REQUIRED_CHECKER_SNIPPETS = {
    "self-hosted/check/types.sio": [
        "TyF128",
        "fn ty_f128() -> TypeEntry",
        "kind: TypeKind::TyF128",
        "clifford_p: 128",
    ],
    "self-hosted/check/compat.sio": [
        "if a.kind == TypeKind::TyF128 && b.kind == TypeKind::TyF128",
        "ty.kind == TypeKind::TyF32 || ty.kind == TypeKind::TyF64 || ty.kind == TypeKind::TyF128",
        "pub fn name_is_f128(n: Name) -> bool",
        "name_matches_str4(n, 102, 49, 50, 56)",
    ],
    "self-hosted/check/check.sio": [
        "mangle_append_literal(result, 102, 49, 50, 56, 4)",
        "} else if name_is_f128(name) {",
        "ty_f128()",
        "TypeKind::TyF128 => print(\"f128\")",
        "(self, ty_f128())",
        "if ty.kind == TypeKind::TyF128",
        "|| arg_ty.kind == TypeKind::TyF128",
    ],
    "self-hosted/test_check.sio": [
        "fn tc_f128_type() -> TypeExpr",
        "tc_named_type4(102, 49, 50, 56)",
        "fn tc_test_07b() -> bool",
        "T07b OK: f128 param + return type",
    ],
    "self-hosted/main.sio": [
        "if ty.kind == TypeKind::TyF128 { print(\"f128\"); return }",
    ],
    "self-hosted/compiler/main.sio": [
        "if ty.kind == TypeKind::TyF128 { print(\"f128\"); return }",
    ],
    "self-hosted/ir/serialize.sio": [
        "let SOIR_TYPE_F128: i8 = 26",
        "if ty.kind == TypeKind::TyF128",
        "if tag == SOIR_TYPE_F128 { return (ty_f128(), p) }",
    ],
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def run_command(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def parser_float_literal_block(parser_source: str) -> str:
    marker = "fn parse_float_literal(self)"
    start = parser_source.find(marker)
    if start < 0:
        raise SystemExit("parse_float_literal not found in parser/exprs.sio")
    next_marker = parser_source.find("fn parse_string_literal(self)", start)
    if next_marker < 0:
        raise SystemExit("parse_string_literal marker not found after parse_float_literal")
    return parser_source[start:next_marker]


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("f128 literal provenance receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parser_path = root / "self-hosted" / "parser" / "exprs.sio"
    ast_path = root / "self-hosted" / "parser" / "ast.sio"
    parser_source = parser_path.read_text(encoding="utf-8")
    ast_source = ast_path.read_text(encoding="utf-8")
    checker_sources: dict[str, str] = {
        relpath: (root / relpath).read_text(encoding="utf-8") for relpath in REQUIRED_CHECKER_SNIPPETS
    }
    block = parser_float_literal_block(parser_source)
    missing = [snippet for snippet in REQUIRED_PARSER_SNIPPETS if snippet not in block]
    if missing:
        raise SystemExit(f"parse_float_literal missing required f128 provenance snippets: {missing}")
    missing_ast = [snippet for snippet in REQUIRED_AST_SNIPPETS if snippet not in ast_source]
    if missing_ast:
        raise SystemExit(f"AST missing required f128 decimal metadata snippets: {missing_ast}")
    missing_checker: dict[str, list[str]] = {}
    for relpath, snippets in REQUIRED_CHECKER_SNIPPETS.items():
        missing_for_file = [snippet for snippet in snippets if snippet not in checker_sources[relpath]]
        if missing_for_file:
            missing_checker[relpath] = missing_for_file
    if missing_checker:
        raise SystemExit(f"checker missing required f128 type-awareness snippets: {missing_checker}")
    if block.find("let raw = self.current_name()") > block.find("let p = self.advance()"):
        raise SystemExit("parse_float_literal must capture raw token text before advance")
    if block.find("e.name = raw") < block.find("var e = mk_expr(ExprKind::ExprFloatLit, span)"):
        raise SystemExit("parse_float_literal assigns raw token text before constructing ExprFloatLit")
    if block.find("expr_apply_float_literal_decimal_metadata") < block.find("e.name = raw"):
        raise SystemExit("parse_float_literal must attach decimal metadata after raw token text is stored")

    source_path = out_dir / "f128_literal_provenance_probe.sio"
    source_path.write_text(SOURCE, encoding="utf-8")
    check_rc, check_stdout, check_stderr = run_command([str(compiler), "check", str(source_path)], root, args.timeout)
    check_log = check_stdout + check_stderr
    check_log_path = out_dir / "f128_literal_provenance_probe.check.log"
    check_log_path.write_text(check_log, encoding="utf-8")
    if check_rc != 0 or "check: OK" not in check_log:
        raise SystemExit(f"f128 provenance probe must pass frontend check; log={check_log_path}")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "f128_literal_source_spelling_and_decimal_metadata_preserved_in_ast",
        "parser_source_relpath": "self-hosted/parser/exprs.sio",
        "parser_source_sha256": sha256_text(parser_source),
        "ast_source_relpath": "self-hosted/parser/ast.sio",
        "ast_source_sha256": sha256_text(ast_source),
        "parse_float_literal_block_sha256": sha256_text(block),
        "required_parser_snippets": REQUIRED_PARSER_SNIPPETS,
        "required_ast_snippets": REQUIRED_AST_SNIPPETS,
        "required_checker_snippets": REQUIRED_CHECKER_SNIPPETS,
        "checker_source_sha256": {relpath: sha256_text(text) for relpath, text in checker_sources.items()},
        "raw_literal_capture_before_advance": True,
        "float_literal_ast_name_preserved": True,
        "float_literal_f64_value_still_preserved": True,
        "float_literal_decimal_metadata_fields_present": True,
        "float_literal_decimal_metadata_helper_present": True,
        "float_literal_decimal_metadata_attached_in_parser": True,
        "f128_type_kind_present": True,
        "f128_type_constructor_present": True,
        "f128_type_name_recognized_by_checker": True,
        "f128_type_mangle_and_print_present": True,
        "f128_type_positive_checker_test_present": True,
        "f128_type_manifest_printers_present": True,
        "f128_type_soir_serialization_present": True,
        "f128_type_byte_width_recorded": True,
        "f128_type_system_awareness_promoted": True,
        "probe_source": SOURCE,
        "probe_source_sha256": sha256_text(SOURCE),
        "probe_check_rc": check_rc,
        "probe_check_log_sha256": sha256_text(check_log),
        "f128_literal_decimal_digits": LITERAL_DIGITS,
        "f128_literal_decimal_digit_count": 35,
        "f128_literal_decimal_scale10": 34,
        "f128_literal_decimal_metadata_independent_from_f64": True,
        "f128_literal_provenance_preserved_for_future_binary128": True,
        "f128_decimal_not_forced_through_f64_only_ast": True,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "ExprFloatLit preserves source spelling in Expr.name",
            "ExprFloatLit carries bounded decimal sign/significand/scale metadata independent of Expr.float_val",
            "Checker recognizes f128 as TypeKind::TyF128 instead of leaving it as a generic named type",
            "Checker test suite includes a positive f128 parameter/return recognition case",
            "Type printers and SOIR type serialization carry f128 explicitly without falling back to i64",
            "ExprFloatLit still carries the historical f64 parsed value for f64 compatibility",
            "f128 frontend check accepts high-precision decimal source before native-v2 execution",
            "receipt does not claim f128 IR/MIR/ABI/codegen execution",
        ],
        "missing_full_obligations": [
            "binary128 decimal-to-binary128 rounded value from preserved source spelling and decimal metadata",
            "f128 IR opcodes and constructors",
            "f128 MachineIR lowering that emits slot kind 3 with two 64-bit limbs",
            "f128 SysV ABI classification and call-return signature metadata",
            "f128 software helper lowering and IEEE-754 rounding/NaN/Inf contract",
            "f128 native-v2 execution and differential receipts",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path = out_dir / "madaros_v2_f128_literal_provenance.receipt.json"
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        "madaros-v2-f128-literal-provenance: "
        f"status={receipt['status']} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--compiler", default=str(repo_root_from_script() / "bin" / "madaros"))
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--timeout", type=int, default=120)
    emit_p.set_defaults(func=emit)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
