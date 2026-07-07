#!/usr/bin/env python3
"""Emit the S5 extern-declaration front-half receipt.

This receipt promotes the parser/checker/lowerer split between real
`kernel fn` items and `extern "C"` declarations. It proves that extern
declarations no longer inherit kernel-only return diagnostics, including f128
signatures, while preserving E072 for an actual non-unit kernel. It does not
promote external SysV f128 runtime ABI, relocations, or linked C execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.extern_declaration_front_half_receipt/0.1"
STAGE = "S5_22_EXTERN_DECLARATION_IS_NOT_KERNEL_FRONT_HALF"
CASE_ID = "extern_declaration_is_not_kernel_front_half"
UNEXPECTED_EXTERN_DIAGNOSTICS = [
    "E072",
    "kernel function must return",
    "E008",
    "return value does not match",
]

POSITIVE_CASES: list[dict[str, str]] = [
    {
        "case_id": "extern_c_i64_return_decl_check",
        "return_kind": "i64",
        "source": 'extern "C" {\n  fn getpid() -> i64;\n}\nfn main() -> i64 { 0 }\n',
    },
    {
        "case_id": "extern_c_f64_arg_return_decl_check",
        "return_kind": "f64",
        "source": 'extern "C" {\n  fn sqrt(x: f64) -> f64;\n}\nfn main() -> i64 { 0 }\n',
    },
    {
        "case_id": "extern_c_f128_arg_return_decl_check",
        "return_kind": "f128",
        "source": 'extern "C" {\n  fn passthru_f128(x: f128) -> f128;\n}\nfn main() -> i64 { 0 }\n',
    },
]

NEGATIVE_CASE: dict[str, str] = {
    "case_id": "kernel_nonunit_return_still_rejected",
    "source": "kernel fn bad_kernel() -> i32 with GPU {\n  42\n}\nfn main() -> i32 with IO { 0 }\n",
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


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("extern declaration receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def run_command(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    env = os.environ.copy()
    raw = cwd / "artifacts" / "self-hosted" / "madaros"
    if raw.exists():
        env["MADAROS_RAW_BIN"] = str(raw)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def require_fragment(path: Path, fragment: str, label: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if fragment not in text:
        raise SystemExit(f"missing source fragment for {label}: {path}")
    return {
        "label": label,
        "path": str(path),
        "fragment_sha256": sha256_text(fragment),
        "file_sha256": sha256_text(text),
        "present": True,
    }


def collect_source_evidence(root: Path) -> list[dict[str, Any]]:
    return [
        require_fragment(
            root / "self-hosted" / "parser" / "ast.sio",
            "is_extern: bool",
            "fndef_has_explicit_is_extern_bit",
        ),
        require_fragment(
            root / "self-hosted" / "parser" / "items.sio",
            "is_kernel: is_kernel,\n            is_extern: false",
            "normal_fn_sets_is_extern_false",
        ),
        require_fragment(
            root / "self-hosted" / "parser" / "items.sio",
            "is_kernel: false, is_extern: true",
            "extern_fn_sets_is_kernel_false_is_extern_true",
        ),
        require_fragment(
            root / "self-hosted" / "check" / "check.sio",
            "if (*fd).is_extern {\n                return\n            }",
            "inplace_checker_skips_extern_body_return_check",
        ),
        require_fragment(
            root / "self-hosted" / "check" / "check.sio",
            "if (*fd).is_extern {\n                    return c\n                }",
            "byvalue_checker_skips_extern_body_return_check",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "if (*fd).is_extern {\n                                        (*(*lo).module).functions[fn_id_ps as usize].compile_strategy = IR_STRATEGY_EXTERN",
            "lowerer_preseeds_extern_strategy_before_kernel",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "if callee_strat_ext == IR_STRATEGY_EXTERN",
            "lowerer_uses_named_extern_strategy_constant",
        ),
    ]


def emit_positive_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, str], timeout_s: int) -> dict[str, Any]:
    case_id = case["case_id"]
    source = case["source"]
    source_path = out_dir / f"{case_id}.sio"
    log_path = out_dir / f"{case_id}.check.log"
    source_path.write_text(source, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    unexpected = [frag for frag in UNEXPECTED_EXTERN_DIAGNOSTICS if frag in log]
    if rc != 0:
        raise SystemExit(f"{case_id} expected check rc=0, got {rc}\n{log[-4000:]}")
    if unexpected:
        raise SystemExit(f"{case_id} emitted unexpected extern diagnostic fragments: {unexpected}\n{log[-4000:]}")
    return {
        "case_id": case_id,
        "class": "positive_extern_declaration_check",
        "return_kind": case["return_kind"],
        "source_sha256": sha256_text(source),
        "check_rc": rc,
        "check_log_sha256": sha256_text(log),
        "unexpected_diagnostic_fragments": unexpected,
        "extern_declaration_accepted": True,
        "kernel_e072_absent": True,
        "return_mismatch_e008_absent": True,
    }


def emit_negative_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = NEGATIVE_CASE["case_id"]
    source = NEGATIVE_CASE["source"]
    source_path = out_dir / f"{case_id}.sio"
    log_path = out_dir / f"{case_id}.check.log"
    source_path.write_text(source, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc == 0:
        raise SystemExit(f"{case_id} expected nonzero checker rc for non-unit kernel")
    if "E072" not in log or "kernel function must return" not in log:
        raise SystemExit(f"{case_id} expected E072 kernel return diagnostic\n{log[-4000:]}")
    return {
        "case_id": case_id,
        "class": "negative_real_kernel_nonunit_return",
        "source_sha256": sha256_text(source),
        "check_rc": rc,
        "check_log_sha256": sha256_text(log),
        "expected_diagnostic": "E072",
        "real_kernel_still_rejected": True,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_extern_declaration_front_half.receipt.json"

    source_evidence = collect_source_evidence(root)
    positive_cases = [
        emit_positive_case(root, compiler, out_dir, case, int(args.timeout))
        for case in POSITIVE_CASES
    ]
    negative_case = emit_negative_case(root, compiler, out_dir, int(args.timeout))
    cases = positive_cases + [negative_case]

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "target": "front-half",
        "case_id": CASE_ID,
        "case_count": len(cases),
        "positive_case_count": len(positive_cases),
        "negative_case_count": 1,
        "source_evidence": source_evidence,
        "cases": cases,
        "extern_decl_is_kernel_split_promoted": True,
        "extern_decl_no_kernel_e072_for_non_unit_returns": True,
        "extern_decl_no_empty_body_e008_for_non_unit_returns": True,
        "extern_decl_i64_typecheck_promoted": True,
        "extern_decl_f64_typecheck_promoted": True,
        "extern_decl_f128_typecheck_promoted": True,
        "real_kernel_e072_preserved": True,
        "ir_extern_strategy_no_longer_depends_on_is_kernel": True,
        "parser_has_explicit_is_extern_bit": True,
        "f128_external_sysv_abi_promoted": False,
        "f128_external_sysv_runtime_promoted": False,
        "native_v2_external_relocation_promoted": False,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "extern_C_declarations_are_not_kernel_functions",
            "extern_C_i64_f64_f128_nonunit_return_declarations_typecheck_without_E072",
            "extern_C_no_body_declarations_skip_return_body_check_without_E008",
            "real_kernel_nonunit_return_still_fails_closed_with_E072",
            "lowerer_uses_explicit_is_extern_for_IR_STRATEGY_EXTERN",
            "external_SysV_runtime_ABI_is_not_promoted_by_this_receipt",
        ],
        "missing_full_obligations": [
            "external C symbol/import representation in native-v2 MachineIR",
            "external SysV relocation/link path for C functions",
            "external SysV f128 argument and return execution oracle against C binary128",
            "external aggregate/SRET ABI oracle coverage",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-extern-declaration-front-half: cases={receipt['case_count']} "
        f"sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
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
