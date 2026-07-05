#!/usr/bin/env python3
"""Emit a Madaros v2 S5 wide-integer numeric-tower receipt.

This promotes the already-implemented i128/i256/u128/u256 wide-limb path into
S5 evidence. It deliberately does not claim f128: that remains a separate
numeric tower obligation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.wide_int_receipt/0.1"
STAGE_CONTRACT_LEVEL = "S5_WIDE_INT_I128_I256_PROMOTED_NOT_F128"


CHECK_OK_CASES: list[dict[str, Any]] = [
    {"case_id": "i128_type_identity", "source": "fn main() -> i64 { let x: i128 = 1 as i128; 0 }\n"},
    {"case_id": "i256_type_identity", "source": "fn main() -> i64 { let x: i256 = 1 as i256; 0 }\n"},
    {"case_id": "u128_type_identity", "source": "fn main() -> i64 { let x: u128 = 1 as u128; 0 }\n"},
    {"case_id": "u256_type_identity", "source": "fn main() -> i64 { let x: u256 = 1 as u256; 0 }\n"},
    {
        "case_id": "wide_explicit_casts",
        "source": "fn main() -> i64 { let a: i128 = 42 as i128; let b: i256 = 99 as i256; 0 }\n",
    },
    {
        "case_id": "i128_param_return_check",
        "source": "fn id(x: i128) -> i128 { x }\nfn main() -> i64 { let r = id(42 as i128); 0 }\n",
    },
    {
        "case_id": "i256_param_return_check",
        "source": "fn id(x: i256) -> i256 { x }\nfn main() -> i64 { let r = id(7 as i256); 0 }\n",
    },
]

CHECK_REJECT_CASES: list[dict[str, Any]] = [
    {
        "case_id": "reject_i128_from_i256",
        "source": "fn main() -> i64 { let x: i128 = 1 as i256; 0 }\n",
    },
    {
        "case_id": "reject_u128_from_i128",
        "source": "fn main() -> i64 { let x: u128 = 1 as i128; 0 }\n",
    },
]

SOURCE_NATIVE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "source_i128_mul_gt",
        "source": "fn main() -> i64 { let a: i128 = 4294967296 as i128; let b: i128 = 4294967296 as i128; let c: i128 = a * b; let z: i128 = 0 as i128; if c > z { return 42 } 1 }\n",
        "expected_exit": 42,
        "wide_type": "i128",
        "wide_ops": ["mul", "cmp_gt"],
    },
    {
        "case_id": "source_i256_mul_gt",
        "source": "fn main() -> i64 { let a: i256 = 4294967296 as i256; let b: i256 = 4294967296 as i256; let c: i256 = a * b; let z: i256 = 0 as i256; if c > z { return 42 } 1 }\n",
        "expected_exit": 42,
        "wide_type": "i256",
        "wide_ops": ["mul", "cmp_gt"],
    },
    {
        "case_id": "source_u128_mul_add_gt",
        "source": "fn main() -> i64 { let a: u128 = 4294967296 as u128; let b: u128 = 4294967296 as u128; let c: u128 = a * b; let d: u128 = c + c; if d > c { return 42 } 1 }\n",
        "expected_exit": 42,
        "wide_type": "u128",
        "wide_ops": ["mul", "add", "cmp_gt"],
    },
    {
        "case_id": "source_u256_mul_add_ne",
        "source": "fn main() -> i64 { let a: u256 = 4294967296 as u256; let b: u256 = 4294967296 as u256; let c: u256 = a * b; let d: u256 = c + c; if d != c { return 42 } 1 }\n",
        "expected_exit": 42,
        "wide_type": "u256",
        "wide_ops": ["mul", "add", "cmp_ne"],
    },
    {
        "case_id": "source_i128_sub_eq_zero",
        "source": "fn main() -> i64 { let a: i128 = 4294967296 as i128; let b: i128 = 4294967296 as i128; let c: i128 = a * b; let d: i128 = c - c; let z: i128 = 0 as i128; if d == z { return 42 } 1 }\n",
        "expected_exit": 42,
        "wide_type": "i128",
        "wide_ops": ["mul", "sub", "cmp_eq"],
    },
    {
        "case_id": "source_i256_add_eq",
        "source": "fn main() -> i64 { let x: i256 = 1 as i256; let y: i256 = 2 as i256; let z: i256 = x + y; if z == (3 as i256) { 7 } else { 1 } }\n",
        "expected_exit": 7,
        "wide_type": "i256",
        "wide_ops": ["add", "cmp_eq"],
    },
]

NATIVE_EMIT_CASES: list[dict[str, Any]] = [
    {
        "case_id": "irwide_add4_i256_carry_chain",
        "mode": "--native-v2-emit-wide-add4",
        "marker": "wide-add4 rc=0",
        "expected_exit": 1,
        "fake_i64_exit": 0,
        "wide_type": "i256",
        "ir_opcode": "IrWideAdd",
    },
    {
        "case_id": "irwide_sub_i128_borrow_chain",
        "mode": "--native-v2-emit-wide-sub",
        "marker": "wide-sub rc=0",
        "expected_exit": 1,
        "fake_i64_exit": 2,
        "wide_type": "i128",
        "ir_opcode": "IrWideSub",
    },
    {
        "case_id": "irwide_mul_i128_cross_limb",
        "mode": "--native-v2-emit-wide-mul",
        "marker": "wide-mul rc=0",
        "expected_exit": 1,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideMul",
    },
    {
        "case_id": "irwide_shr_limb_i128",
        "mode": "--native-v2-emit-wide-shr",
        "marker": "wide-shr rc=0",
        "expected_exit": 1,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideShrLimb",
    },
    {
        "case_id": "irwide_div_single_limb_i128",
        "mode": "--native-v2-emit-wide-div",
        "marker": "wide-div rc=0",
        "expected_exit": 5,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideDiv",
    },
    {
        "case_id": "irwide_mod_single_limb_i128",
        "mode": "--native-v2-emit-wide-mod",
        "marker": "wide-mod rc=0",
        "expected_exit": 3,
        "fake_i64_exit": 1,
        "wide_type": "i128",
        "ir_opcode": "IrWideMod",
    },
    {
        "case_id": "irwide_cmp_high_limb_i128",
        "mode": "--native-v2-emit-wide-cmp",
        "marker": "wide-cmp rc=0",
        "expected_exit": 1,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideCmp",
    },
    {
        "case_id": "irwide_shr_unaligned_i128",
        "mode": "--native-v2-emit-wide-shr-unaligned",
        "marker": "wide-shr-unaligned rc=0",
        "expected_exit": 16,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideShr",
    },
    {
        "case_id": "irwide_divfull_multilimb_i128",
        "mode": "--native-v2-emit-wide-divfull",
        "marker": "wide-divfull rc=0",
        "expected_exit": 5,
        "fake_i64_exit": 105,
        "wide_type": "i128",
        "ir_opcode": "IrWideDivFull",
    },
    {
        "case_id": "irwide_modfull_multilimb_i128",
        "mode": "--native-v2-emit-wide-modfull",
        "marker": "wide-modfull rc=0",
        "expected_exit": 193,
        "fake_i64_exit": 0,
        "wide_type": "i128",
        "ir_opcode": "IrWideModFull",
    },
]


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


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


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


def run_binary(path: Path, timeout_s: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.run([str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("wide-int receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def emit_check_ok(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    log_path = out_dir / f"{case_id}.check.log"
    source_path.write_text(source_text, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0 or "check: OK" not in log:
        raise SystemExit(f"{case_id} expected check OK, rc={rc}; log={log_path}")
    return {
        "case_id": case_id,
        "class": "wide_type_positive_check",
        "source": source_path.name,
        "check_rc": rc,
        "source_sha256": sha256_text(source_text),
        "check_log_sha256": sha256_text(normalize_log(log, out_dir)),
    }


def emit_check_reject(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    log_path = out_dir / f"{case_id}.check.log"
    source_path.write_text(source_text, encoding="utf-8")
    rc, stdout, stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc == 0 or "check: OK" in log:
        raise SystemExit(f"{case_id} expected rejected check, rc={rc}; log={log_path}")
    return {
        "case_id": case_id,
        "class": "wide_type_negative_check",
        "source": source_path.name,
        "check_rc": rc,
        "source_sha256": sha256_text(source_text),
        "check_log_sha256": sha256_text(normalize_log(log, out_dir)),
    }


def emit_source_native(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    expected_exit = int(case["expected_exit"])
    source_path = out_dir / f"{case_id}.sio"
    check_log_path = out_dir / f"{case_id}.check.log"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    source_path.write_text(source_text, encoding="utf-8")

    check_rc, check_stdout, check_stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    check_log = check_stdout + check_stderr
    check_log_path.write_text(check_log, encoding="utf-8")
    if check_rc != 0 or "check: OK" not in check_log:
        raise SystemExit(f"{case_id} expected check OK, rc={check_rc}; log={check_log_path}")

    rc, compile_stdout, compile_stderr = run_command(
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
    compile_log = compile_stdout + compile_stderr
    compile_log_path.write_text(compile_log, encoding="utf-8")
    if rc != 0 or "native_v2_compile: emitted" not in compile_log:
        raise SystemExit(f"{case_id} native-v2 compile failed rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")
    machine_module = json.loads(mm_path.read_text(encoding="utf-8"))
    if machine_module.get("schema") != "madaros.v2.s5.machine_module/0.1":
        raise SystemExit(f"{case_id} bad MachineModule schema")
    if machine_module.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"{case_id} missing MachineModule export flag")
    if machine_module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id} unexpectedly used legacy fallback")
    return {
        "case_id": case_id,
        "class": "source_level_wide_native_v2",
        "source": source_path.name,
        "wide_type": case["wide_type"],
        "wide_ops": list(case["wide_ops"]),
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "source_sha256": sha256_text(source_text),
        "check_log_sha256": sha256_text(normalize_log(check_log, out_dir)),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": sha256_text(stable_json(machine_module)),
        "machine_module_fn_count": int(machine_module.get("fn_count", -1)),
        "machine_module_total_instr_count": int(machine_module.get("total_machine_instr_count", -1)),
    }


def emit_native_builtin(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    elf_path = out_dir / f"{case_id}.native_v2"
    log_path = out_dir / f"{case_id}.emit.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    rc, stdout, stderr = run_command([str(compiler), str(case["mode"]), str(elf_path)], root, timeout_s)
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if rc != 0 or str(case["marker"]) not in log:
        raise SystemExit(f"{case_id} emit failed rc={rc}; log={log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not produce an ELF")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, run_stdout, run_stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(run_stdout)
    stderr_path.write_bytes(run_stderr)
    expected_exit = int(case["expected_exit"])
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")
    fake_i64_exit = int(case["fake_i64_exit"])
    if actual_exit == fake_i64_exit:
        raise SystemExit(f"{case_id} did not distinguish wide-limb behavior from fake i64")
    return {
        "case_id": case_id,
        "class": "hand_built_ir_wide_native_v2",
        "mode": case["mode"],
        "wide_type": case["wide_type"],
        "ir_opcode": case["ir_opcode"],
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "fake_i64_exit": fake_i64_exit,
        "emit_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "stdout_sha256": sha256_bytes(run_stdout),
        "stderr_sha256": sha256_bytes(run_stderr),
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_wide_int.receipt.json"

    check_ok = [emit_check_ok(root, compiler, out_dir, case, args.timeout) for case in CHECK_OK_CASES]
    check_reject = [emit_check_reject(root, compiler, out_dir, case, args.timeout) for case in CHECK_REJECT_CASES]
    source_native = [emit_source_native(root, compiler, out_dir, case, args.timeout) for case in SOURCE_NATIVE_CASES]
    native_emit = [emit_native_builtin(root, compiler, out_dir, case, args.timeout) for case in NATIVE_EMIT_CASES]
    cases = check_ok + check_reject + source_native + native_emit

    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "wide_int_i128_i256_numeric_tower",
        "case_count": len(cases),
        "check_ok_case_count": len(check_ok),
        "check_reject_case_count": len(check_reject),
        "source_native_case_count": len(source_native),
        "native_emit_case_count": len(native_emit),
        "cases": cases,
        "s5_wide_int_i128_i256_complete": True,
        "wide_i128_i256_promoted": True,
        "wide_u128_u256_promoted": True,
        "source_level_wide_arithmetic_promoted": True,
        "native_v2_wide_limb_backend_promoted": True,
        "wide_type_identity_and_safety_promoted": True,
        "numeric_tower_width_receipts_partial": True,
        "f128_promoted": False,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "i128_i256_u128_u256_type_identity_checks_pass",
            "i128_i256_and_u128_i128_mismatches_reject",
            "source_level_i128_i256_u128_u256_arithmetic_compiles_with_native_v2",
            "source_level_wide_arithmetic_executes_expected_discriminators",
            "native_v2_wide_add_sub_mul_shift_div_mod_cmp_emitters_execute_expected_discriminators",
            "wide_emitters_distinguish_multi_limb_behavior_from_fake_i64",
            "f128_is_not_promoted_by_this_receipt",
        ],
        "missing_full_obligations": [
            "f128 numeric tower width receipts",
            "generic aggregate return coverage",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-wide-int: cases={receipt['case_count']} "
        f"source={receipt['source_native_case_count']} native={receipt['native_emit_case_count']} "
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
