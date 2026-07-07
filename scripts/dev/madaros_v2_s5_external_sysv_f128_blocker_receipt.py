#!/usr/bin/env python3
"""Emit the S5 external SysV f128 blocker receipt.

This receipt makes the f128 extern boundary explicit without promoting it.
It proves that the parser/checker/lowerer front half accepts an extern C
binary128 signature, and records the precise remaining blocker: native-v2
still has no external-symbol MachineIR call shape, external relocation/link
path, or SysV binary128 runtime oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.external_sysv_f128_blocker_receipt/0.1"
STAGE = "S5_23_EXTERNAL_SYSV_F128_BLOCKED_WITH_CONCRETE_REASONS"
CASE_ID = "external_sysv_f128_abi_blocked_front_half_received"

PASSTHRU_SOURCE = """extern "C" {
  fn passthru_f128(x: f128) -> f128;
}
fn main() -> i64 { 0 }
"""

PASSTHRU_CALL_SOURCE = """extern "C" {
  fn passthru_f128(x: f128) -> f128;
}
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let y: f128 = passthru_f128(x)
  0
}
"""


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
        raise SystemExit("external SysV f128 blocker receipt canonical JSON roundtrip changed bytes")
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
            "is_kernel: false, is_extern: true",
            "extern_block_parser_sets_is_extern_true",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "IR_STRATEGY_EXTERN",
            "lowerer_preseeds_extern_strategy",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "lower.sio",
            "ir_call_extern(dst, callee_name, args, argc)",
            "lowerer_emits_symbolic_extern_call",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "ir.sio",
            "IrCallExtern",
            "ir_has_symbolic_extern_call_opcode",
        ),
        require_fragment(
            root / "self-hosted" / "ir" / "ir.sio",
            "pub fn ir_call_extern(dst: i64, symbol: Name",
            "ir_call_extern_records_symbol_name",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "lower_ir.sio",
            "lower_call_extern",
            "legacy_extern_lowerer_is_symbol_reloc_path",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "reloc.sio",
            "struct ExternReloc",
            "legacy_extern_reloc_shape_exists",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "MIR_OP_PSEUDO_CALL",
            "native_v2_internal_call_path_is_fn_id_shape",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "MIR_OP_PSEUDO_CALL_EXTERN",
            "native_v2_has_explicit_external_call_pseudo_shape",
        ),
        require_fragment(
            root / "self-hosted" / "native" / "machine_ir.sio",
            "external_sysv_abi_pending",
            "native_v2_external_sysv_boundary_fails_closed_with_named_detail",
        ),
    ]


def emit_passthru_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    source_path = out_dir / "extern_c_passthru_f128_decl_received.sio"
    log_path = out_dir / "extern_c_passthru_f128_decl_received.check.log"
    source_path.write_text(PASSTHRU_SOURCE, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"extern C passthru_f128 declaration must check, got rc={rc}\n{log[-4000:]}")
    for fragment in ["E072", "kernel function must return", "E008", "return value does not match"]:
        if fragment in log:
            raise SystemExit(f"extern C passthru_f128 declaration emitted unexpected diagnostic {fragment!r}")
    return {
        "case_id": "extern_c_passthru_f128_decl_received",
        "class": "positive_extern_declaration_check_blocker_boundary",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_SOURCE),
        "check_rc": rc,
        "check_log_sha256": sha256_text(log),
        "extern_declaration_accepted": True,
        "kernel_e072_absent": True,
        "return_mismatch_e008_absent": True,
        "lowerer_strategy_expected": "IR_STRATEGY_EXTERN",
        "ir_opcode_expected_if_called": "IrCallExtern",
        "native_v2_execution_attempted": False,
        "native_v2_external_sysv_f128_promoted": False,
    }


def emit_passthru_call_boundary_case(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    case_id = "extern_c_passthru_f128_call_reaches_machineir_boundary"
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    log_path = out_dir / f"{case_id}.native_v2.log"
    source_path.write_text(PASSTHRU_CALL_SOURCE, encoding="utf-8")
    rc, stdout, stderr = run_command(
        [
            str(compiler),
            "--native-v2-compile",
            str(source_path),
            "-o",
            str(elf_path),
            "--machine-module-json",
            str(mm_path),
        ],
        root,
        timeout_s,
    )
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id}: wrapper command must return rc=0 while reporting native-v2 failure; log={log_path}")
    if "native_v2_compile: FAIL to_file rc=12" not in log:
        raise SystemExit(f"{case_id}: expected native-v2 fail-closed rc=12; log={log_path}")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case_id}: crash or fallback detected; log={log_path}")
    if elf_path.exists() and elf_path.stat().st_size > 0:
        raise SystemExit(f"{case_id}: external SysV f128 blocker must not emit an ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: expected MachineModule JSON for classified external SysV blocker")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("schema") != "madaros.v2.s5.machine_module/0.1":
        raise SystemExit(f"{case_id}: bad MachineModule schema")
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id}: MachineModule used fallback")
    if module.get("supported") is not False:
        raise SystemExit(f"{case_id}: MachineModule must remain unsupported until external SysV ABI is implemented")
    if module.get("unsupported_detail") != "external_sysv_abi_pending":
        raise SystemExit(
            f"{case_id}: expected unsupported_detail external_sysv_abi_pending, "
            f"got {module.get('unsupported_detail')!r}"
        )
    return {
        "case_id": case_id,
        "class": "negative_native_v2_external_sysv_call_boundary",
        "symbol": "passthru_f128",
        "signature": "(f128)->f128",
        "source_sha256": sha256_text(PASSTHRU_CALL_SOURCE),
        "native_v2_compile_rc": rc,
        "native_v2_compile_log_sha256": sha256_text(log),
        "elf_emitted": False,
        "machine_module_json_emitted": True,
        "machine_module_json_sha256": sha256_text(stable_json(module)),
        "machine_module_supported": module.get("supported"),
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "legacy_fallback": False,
        "segfault": False,
        "native_v2_machineir_external_call_symbol_classified": True,
        "native_v2_external_sysv_f128_promoted": False,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_external_sysv_f128_blocker.receipt.json"

    source_evidence = collect_source_evidence(root)
    passthru_case = emit_passthru_case(root, compiler, out_dir, int(args.timeout))
    passthru_call_case = emit_passthru_call_boundary_case(root, compiler, out_dir, int(args.timeout))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "target": "x86_64-linux",
        "case_id": CASE_ID,
        "case_count": 2,
        "positive_case_count": 1,
        "negative_boundary_case_count": 1,
        "cases": [passthru_case, passthru_call_case],
        "source_evidence": source_evidence,
        "extern_decl_f128_typecheck_promoted": True,
        "parser_has_explicit_is_extern_bit": True,
        "ir_extern_strategy_promoted": True,
        "ir_call_extern_symbol_receipt_promoted": True,
        "native_v2_machineir_external_call_symbol_classified": True,
        "native_v2_machineir_external_call_symbol_promoted": False,
        "native_v2_external_relocation_promoted": False,
        "f128_external_sysv_abi_promoted": False,
        "f128_external_sysv_runtime_promoted": False,
        "f128_external_sysv_argument_oracle_promoted": False,
        "f128_external_sysv_return_oracle_promoted": False,
        "f128_internal_opaque_direct_call_abi_promoted_elsewhere": True,
        "f128_internal_opaque_return_abi_promoted_elsewhere": True,
        "f128_sysv_classes_recorded_as_metadata_only": True,
        "blocked": True,
        "blocked_reason": "extern_f128_declaration_reaches_IR_but_native_v2_has_no_external_symbol_call_shape_or_sysv_binary128_runtime_oracle",
        "roundtrip_contract": [
            "extern_C_f128_declaration_typechecks_without_kernel_diagnostics",
            "lowerer_has_IR_STRATEGY_EXTERN_and_IrCallExtern_symbol_path",
            "native_v2_external_symbol_call_shape_is_classified_as_external_sysv_abi_pending",
            "external_SysV_binary128_argument_and_return_oracle_not_promoted",
        ],
        "missing_full_obligations": [
            "external SysV relocation/link path for C functions in native-v2",
            "f128 SysV argument placement oracle against a C binary128 function",
            "f128 SysV return capture oracle against a C binary128 function",
            "external aggregate/SRET ABI oracle coverage",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-external-sysv-f128-blocker: cases={receipt['case_count']} "
        f"blocked={receipt['blocked']} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
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
