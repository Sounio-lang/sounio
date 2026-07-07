#!/usr/bin/env python3
"""Emit a Madaros v2 S5 diagnostics/fallback receipt.

This receipt closes the S5 unsupported-width and remaining f128 blocker
diagnostic slice. It deliberately does not promote full f128 execution. Instead
it proves that numeric widths outside the promoted native-v2 S5 set fail closed,
that the S5.18 bounded rounded-tenths add helper remains promoted, and that f128
operations beyond the current contract fail with specific MachineModule details
instead of silently emitting an ELF or falling through legacy code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.diagnostics_receipt/0.3"
STAGE_CONTRACT_LEVEL = "S5_1_UNSUPPORTED_NUMERIC_AND_F128_BLOCKER_DIAGNOSTICS_PROMOTED"
DIAGNOSTIC_FRAGMENT = "native-v2 S5 unsupported numeric width"

NEGATIVE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "reject_f128_overwide_arg_shape_native_v2",
        "class": "unsupported_f128_operation",
        "source": "fn too_many(a: f128, b: f128, c: f128, d: f128, e: f128, f: f128, g: f128, h: f128, i: f128) -> i64 { 9 }\nfn main() -> i64 { let x: f128 = 1.0 as f128; too_many(x, x, x, x, x, x, x, x, x) }\n",
        "unsupported_width": "f128",
        "expected_detail": "call_arity_gt_8",
        "expected_fragment": "call_arity_gt_8",
        "expect_machine_module_json": True,
        "expected_machine_module_supported": False,
    },
    {
        "case_id": "reject_i512_let_annotation_native_v2",
        "class": "unsupported_integer_width",
        "source": "fn main() -> i64 { let x: i512 = 1 as i512; 0 }\n",
        "unsupported_width": "i512",
        "expected_detail": "let annotation",
    },
    {
        "case_id": "reject_u512_cast_native_v2",
        "class": "unsupported_integer_width",
        "source": "fn main() -> i64 { let x = 1 as u512; 0 }\n",
        "unsupported_width": "u512",
        "expected_detail": "cast",
    },
]

POSITIVE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "preserve_f128_rounded_tenths_add_helper_native_v2",
        "class": "promoted_f128_helper_guard",
        "source": "fn main() -> i64 { let x: f128 = 0.1 as f128; let y: f128 = 0.2 as f128; let z: f128 = x + y; let w: f128 = z; 0 }\n",
        "expected_exit": 0,
        "promoted_width": "f128",
    },
    {
        "case_id": "preserve_i256_promoted_width_native_v2",
        "class": "promoted_integer_width_guard",
        "source": "fn main() -> i64 { let x: i256 = 1 as i256; let y: i256 = 2 as i256; let z: i256 = x + y; if z == (3 as i256) { 7 } else { 1 } }\n",
        "expected_exit": 7,
        "promoted_width": "i256",
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


def run_binary(path: Path, timeout_s: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.run([str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("diagnostics receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def emit_negative(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    check_log_path = out_dir / f"{case_id}.check.log"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    source_path.write_text(source_text, encoding="utf-8")

    check_rc, check_stdout, check_stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    check_log = check_stdout + check_stderr
    check_log_path.write_text(check_log, encoding="utf-8")
    expect_machine_module_json = bool(case.get("expect_machine_module_json", False))
    if not expect_machine_module_json and (check_rc != 0 or "check: OK" not in check_log):
        raise SystemExit(f"{case_id} expected syntax/typecheck acceptance before S5 native-v2 guard; log={check_log_path}")

    compile_rc, compile_stdout, compile_stderr = run_command(
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
    expected_fragment = str(case.get("expected_fragment", DIAGNOSTIC_FRAGMENT))
    if "expected_runtime_rc" in case:
        if "native_v2_compile: emitted" not in compile_log:
            raise SystemExit(f"{case_id} expected runtime fail-closed ELF emission; log={compile_log_path}")
        if not elf_path.exists() or elf_path.stat().st_size <= 0:
            raise SystemExit(f"{case_id} missing runtime fail-closed ELF: {elf_path}")
        os.chmod(elf_path, 0o755)
        run_rc, run_stdout, run_stderr = run_binary(elf_path, timeout_s)
        expected_runtime_rc = int(case["expected_runtime_rc"])
        if run_rc != expected_runtime_rc:
            raise SystemExit(f"{case_id} expected runtime rc={expected_runtime_rc}, got {run_rc}")
        if not mm_path.exists() or mm_path.stat().st_size <= 0:
            raise SystemExit(f"{case_id} did not emit expected runtime helper MachineModule JSON: {mm_path}")
        machine_module = json.loads(mm_path.read_text(encoding="utf-8"))
        if machine_module.get("schema") != "madaros.v2.s5.machine_module/0.1":
            raise SystemExit(f"{case_id} bad MachineModule schema")
        if machine_module.get("supported") is not True:
            raise SystemExit(f"{case_id} runtime helper MachineModule must be supported")
        expected_machine_opcode = int(case.get("expected_machine_opcode", 0) or 0)
        opcode_found = False
        for fn in machine_module.get("functions", []):
            for instr in fn.get("instrs", []):
                if isinstance(instr, list) and len(instr) > 0 and int(instr[0]) == expected_machine_opcode:
                    opcode_found = True
        if expected_machine_opcode != 0 and not opcode_found:
            raise SystemExit(f"{case_id} missing runtime helper MachineIR opcode")
        return {
            "case_id": case_id,
            "class": case["class"],
            "unsupported_width": case["unsupported_width"],
            "expected_detail": case["expected_detail"],
            "source": source_path.name,
            "source_sha256": sha256_text(source_text),
            "check_rc": check_rc,
            "native_v2_compile_rc": compile_rc,
            "run_rc": run_rc,
            "expected_runtime_rc": expected_runtime_rc,
            "check_log_sha256": sha256_text(normalize_log(check_log, out_dir)),
            "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
            "diagnostic_fragment": expected_fragment,
            "elf_emitted": True,
            "elf_sha256": sha256_bytes(elf_path.read_bytes()),
            "machine_module_json_emitted": True,
            "machine_module_supported": True,
            "machine_module_unsupported_detail": str(machine_module.get("unsupported_detail", "")),
            "machine_module_json_sha256": sha256_text(stable_json(machine_module)),
            "expected_machine_opcode": expected_machine_opcode,
            "expected_machine_opcode_found": opcode_found,
            "segfault": False,
            "legacy_fallback": False,
            "status": "runtime_fail_closed",
        }
    if expect_machine_module_json:
        if "native_v2_compile: emitted" in compile_log:
            raise SystemExit(f"{case_id} unexpectedly emitted f128 execution-pending ELF; log={compile_log_path}")
        if "native_v2_compile: FAIL" not in compile_log:
            raise SystemExit(f"{case_id} missing native-v2 fail-closed log line; log={compile_log_path}")
    elif compile_rc == 0:
        raise SystemExit(f"{case_id} unexpectedly compiled unsupported S5 width; log={compile_log_path}")
    if expected_fragment not in compile_log and not expect_machine_module_json:
        raise SystemExit(f"{case_id} missing stable diagnostic {expected_fragment!r}; log={compile_log_path}")
    if str(case["expected_detail"]) not in compile_log and not expect_machine_module_json:
        raise SystemExit(f"{case_id} missing expected diagnostic detail {case['expected_detail']!r}; log={compile_log_path}")
    forbidden = ["native_v2_compile: emitted", "Segmentation fault", "SIGSEGV", "legacy fallback"]
    for token in forbidden:
        if token in compile_log:
            raise SystemExit(f"{case_id} contains forbidden fallback/crash token {token!r}; log={compile_log_path}")
    if elf_path.exists() and elf_path.stat().st_size > 0:
        raise SystemExit(f"{case_id} emitted an ELF despite unsupported numeric width: {elf_path}")
    machine_module_json_emitted = mm_path.exists() and mm_path.stat().st_size > 0
    machine_module_supported = None
    machine_module_unsupported_detail = ""
    machine_module_json_sha256 = None
    if expect_machine_module_json:
        if not machine_module_json_emitted:
            raise SystemExit(f"{case_id} did not emit expected f128 MachineModule JSON: {mm_path}")
        machine_module = json.loads(mm_path.read_text(encoding="utf-8"))
        if machine_module.get("schema") != "madaros.v2.s5.machine_module/0.1":
            raise SystemExit(f"{case_id} bad MachineModule schema")
        expected_supported = bool(case.get("expected_machine_module_supported", True))
        machine_module_supported = bool(machine_module.get("supported") is True)
        if machine_module_supported != expected_supported:
            raise SystemExit(f"{case_id} MachineModule supported mismatch")
        machine_module_unsupported_detail = str(machine_module.get("unsupported_detail", ""))
        if str(case["expected_detail"]) and machine_module_unsupported_detail != str(case["expected_detail"]):
            raise SystemExit(f"{case_id} bad MachineModule unsupported detail")
        machine_module_json_sha256 = sha256_text(stable_json(machine_module))
    elif machine_module_json_emitted:
        raise SystemExit(f"{case_id} emitted MachineModule JSON despite front-half unsupported numeric width: {mm_path}")

    return {
        "case_id": case_id,
        "class": case["class"],
        "unsupported_width": case["unsupported_width"],
        "expected_detail": case["expected_detail"],
        "source": source_path.name,
        "source_sha256": sha256_text(source_text),
        "check_rc": check_rc,
        "native_v2_compile_rc": compile_rc,
        "check_log_sha256": sha256_text(normalize_log(check_log, out_dir)),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "diagnostic_fragment": expected_fragment,
        "elf_emitted": False,
        "machine_module_json_emitted": machine_module_json_emitted,
        "machine_module_supported": machine_module_supported,
        "machine_module_unsupported_detail": machine_module_unsupported_detail,
        "machine_module_json_sha256": machine_module_json_sha256,
        "segfault": False,
        "legacy_fallback": False,
        "status": "fail_closed",
    }


def emit_positive(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    source_path.write_text(source_text, encoding="utf-8")

    compile_rc, compile_stdout, compile_stderr = run_command(
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
    if compile_rc != 0 or "native_v2_compile: emitted" not in compile_log:
        raise SystemExit(f"{case_id} promoted width no longer compiles; log={compile_log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit an ELF")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    expected_exit = int(case["expected_exit"])
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")
    machine_module = json.loads(mm_path.read_text(encoding="utf-8"))
    if machine_module.get("schema") != "madaros.v2.s5.machine_module/0.1":
        raise SystemExit(f"{case_id} bad MachineModule schema")
    if machine_module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id} unexpectedly used legacy fallback")

    return {
        "case_id": case_id,
        "class": case["class"],
        "promoted_width": case["promoted_width"],
        "source": source_path.name,
        "source_sha256": sha256_text(source_text),
        "native_v2_compile_rc": compile_rc,
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": sha256_text(stable_json(machine_module)),
        "machine_module_supported": bool(machine_module.get("supported") is True),
        "machine_module_unsupported_detail": str(machine_module.get("unsupported_detail", "")),
        "legacy_fallback": False,
        "status": "promoted_width_preserved",
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_diagnostics.receipt.json"

    negative_cases = [emit_negative(root, compiler, out_dir, case, args.timeout) for case in NEGATIVE_CASES]
    positive_cases = [emit_positive(root, compiler, out_dir, case, args.timeout) for case in POSITIVE_CASES]
    cases = negative_cases + positive_cases

    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "unsupported_numeric_width_diagnostics",
        "case_count": len(cases),
        "negative_case_count": len(negative_cases),
        "positive_guard_case_count": len(positive_cases),
        "cases": cases,
        "s5_diagnostics_unsupported_numeric_complete": True,
        "unsupported_numeric_widths_fail_closed": True,
        "unsupported_widths_do_not_emit_elf": True,
        "front_half_unsupported_widths_do_not_emit_machine_module_json": True,
        "f128_blockers_emit_machine_module_json": True,
        "f128_machine_module_supported": "mixed",
        "f128_runtime_fail_closed_rc12": False,
        "f128_runtime_positive_rounded_tenths_add_helper_promoted_elsewhere": True,
        "f128_machine_module_unsupported_details": [
            "call_arity_gt_8",
        ],
        "unsupported_widths_do_not_segfault": True,
        "legacy_fallback_for_unsupported_widths": False,
        "f128_full_execution_not_promoted": True,
        "f128_opaque_direct_call_return_abi_promoted_elsewhere": True,
        "f128_direct_expanded_gpr_call_shape_promoted_elsewhere": True,
        "f128_direct_stack_call_shape_promoted_elsewhere": True,
        "f128_overwide_call_shape_promoted": False,
        "i512_u512_rejected_not_promoted": True,
        "promoted_i256_width_preserved": True,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "f128_native_v2_rounded_decimal_arithmetic_promoted_by_S5_18_helper",
            "f128_native_v2_overwide_arg_shape_fails_closed_after_MachineModule_metadata_export",
            "i512_native_v2_let_annotation_fails_closed_with_stable_diagnostic",
            "u512_native_v2_cast_fails_closed_with_stable_diagnostic",
            "front_half_unsupported_numeric_widths_emit_no_elf",
            "i512_u512_front_half_rejections_emit_no_machine_module_json",
            "f128_blocker_cases_emit_unsupported_machine_module_json",
            "unsupported_numeric_widths_do_not_segfault_or_use_legacy_fallback",
            "promoted_i256_native_v2_path_still_executes",
            "full_f128_execution_is_not_promoted_by_this_receipt",
        ],
        "missing_full_obligations": [
            "f128 IR/MIR/ABI/software-helper receipts",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-diagnostics: cases={receipt['case_count']} "
        f"negative={receipt['negative_case_count']} sha={receipt['receipt_sha256'][:12]} "
        f"receipt={receipt_path}"
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
