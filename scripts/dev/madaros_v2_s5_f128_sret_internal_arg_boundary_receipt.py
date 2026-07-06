#!/usr/bin/env python3
"""Emit the S5.5 f128 internal SRET argument-boundary receipt.

This receipt promotes a deliberately narrow native-v2 contract: internal calls
may pass opaque f128 values through the aggregate-return SRET path by expanding
each f128 explicit argument into two integer ABI words after the hidden
destination pointer. It also keeps a direct-call control case for the compact
vreg classifier bug that originally exposed this boundary.

It does not promote external SysV f128 ABI, full f128 SRET ABI, IEEE arithmetic,
software helpers, NaN/Inf behavior, or general f128 execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_sret_internal_arg_boundary_receipt/0.1"
STAGE = "S5_5_F128_INTERNAL_SRET_ARG_BOUNDARY_PROMOTED"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"

MIR_OP_LOAD_STACK = 100
MIR_OP_ARG_MOVE = 112
MIR_OP_STACK_ADJUST = 129
MIR_OP_STACK_ARG_PUSH = 130


CASES: list[dict[str, Any]] = [
    {
        "case_id": "direct_f128_then_i64_arithmetic_classifier_guard",
        "kind": "direct_control",
        "source": """fn make(v: f128, n: i64) -> i64 { n + 1 }
fn main() -> i64 { make(1.0 as f128, 40) }
""",
        "expected_exit": 41,
        "callee": "make",
        "expected_is_sret": False,
        "expected_arg_indices": [0, 1, 2],
        "expected_arg_source_slots": [1, 2, 3],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
        "expected_f128_param_count": 1,
    },
    {
        "case_id": "sret_f128_arg_then_i64_arithmetic",
        "kind": "sret_register",
        "source": """struct Big { a: i64, b: i64, c: i64 }
fn make(v: f128, n: i64) -> Big { Big { a: n, b: n + 1, c: n + 2 } }
fn main() -> i64 { let b = make(1.0 as f128, 40); b.a + b.b }
""",
        "expected_exit": 81,
        "callee": "make",
        "expected_is_sret": True,
        "expected_arg_indices": [0, 1, 2, 3],
        "expected_arg_source_slots": [4, 1, 2, 3],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
        "expected_f128_param_count": 1,
    },
    {
        "case_id": "sret_f128_arg_copied_to_f128_field_payload",
        "kind": "sret_register_f128_payload",
        "source": """struct BoxF128 { tag: i64, x: f128, tail: i64 }
fn make(v: f128, n: i64) -> BoxF128 { BoxF128 { tag: n, x: v, tail: n + 1 } }
fn main() -> i64 { let b = make(1.0 as f128, 40); b.tag + b.tail }
""",
        "expected_exit": 81,
        "callee": "make",
        "expected_is_sret": True,
        "expected_arg_indices": [0, 1, 2, 3],
        "expected_arg_source_slots": [4, 1, 2, 3],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
        "expected_f128_param_count": 1,
    },
    {
        "case_id": "sret_three_f128_args_crosses_stack_boundary",
        "kind": "sret_stack",
        "source": """struct Big { a: i64, b: i64, c: i64 }
fn make(a: f128, b: f128, c: f128, n: i64) -> Big { Big { a: n, b: n + 1, c: n + 2 } }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let b = make(x, x, x, 40)
  b.a + b.b
}
""",
        "expected_exit": 81,
        "callee": "make",
        "expected_is_sret": True,
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [4, 1, 2, 1, 2, 1],
        "expected_stack_arg_indices": [7, 6],
        "expected_stack_arg_source_slots": [3, 2],
        "expected_stack_adjusts": [16],
        "expected_f128_param_count": 3,
    },
]


def root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def run(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str]:
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
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


def load_machine(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit("MachineModule source mismatch")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit("MachineModule export flag missing")
    if payload.get("target") != "x86_64-linux":
        raise SystemExit("MachineModule target mismatch")
    if payload.get("active") is not True:
        raise SystemExit("MachineModule is not active")
    if payload.get("supported") is not True:
        raise SystemExit(f"MachineModule unsupported: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    return payload


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for fn in module.get("functions", []):
        name = str(fn.get("name", ""))
        if name:
            result[name] = fn
    return result


def opcodes(fn: dict[str, Any]) -> list[int]:
    return [int(instr[0]) for instr in fn.get("instrs", [])]


def instr_indices(fn: dict[str, Any], opcode: int) -> list[int]:
    return [int(instr[10]) for instr in fn.get("instrs", []) if int(instr[0]) == opcode]


def source_stack_slots_for(fn: dict[str, Any], opcode: int) -> list[int]:
    instrs = fn.get("instrs", [])
    slots: list[int] = []
    for idx, instr in enumerate(instrs):
        if int(instr[0]) != opcode:
            continue
        if idx == 0 or int(instrs[idx - 1][0]) != MIR_OP_LOAD_STACK:
            raise SystemExit(f"opcode {opcode} must be fed by immediate LOAD_STACK")
        slots.append(int(instrs[idx - 1][4]))
    return slots


def stack_adjust_immediates(fn: dict[str, Any]) -> list[int]:
    return [int(instr[4]) for instr in fn.get("instrs", []) if int(instr[0]) == MIR_OP_STACK_ADJUST]


def validate_shape(module: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    by_name = functions_by_name(module)
    if "main" not in by_name or case["callee"] not in by_name:
        raise SystemExit(f"{case['case_id']}: expected main and {case['callee']} in MachineModule")
    main = by_name["main"]
    callee = by_name[str(case["callee"])]
    main_ops = opcodes(main)
    callee_ops = opcodes(callee)
    arg_indices = instr_indices(main, MIR_OP_ARG_MOVE)
    arg_slots = source_stack_slots_for(main, MIR_OP_ARG_MOVE)
    stack_indices = instr_indices(main, MIR_OP_STACK_ARG_PUSH)
    stack_slots = source_stack_slots_for(main, MIR_OP_STACK_ARG_PUSH)
    stack_adjusts = stack_adjust_immediates(main)
    if arg_indices != list(case["expected_arg_indices"]):
        raise SystemExit(f"{case['case_id']}: ARG_MOVE indices mismatch: {arg_indices}")
    if arg_slots != list(case["expected_arg_source_slots"]):
        raise SystemExit(f"{case['case_id']}: ARG_MOVE source slots mismatch: {arg_slots}")
    if stack_indices != list(case["expected_stack_arg_indices"]):
        raise SystemExit(f"{case['case_id']}: STACK_ARG_PUSH indices mismatch: {stack_indices}")
    if stack_slots != list(case["expected_stack_arg_source_slots"]):
        raise SystemExit(f"{case['case_id']}: STACK_ARG_PUSH source slots mismatch: {stack_slots}")
    if stack_adjusts != list(case["expected_stack_adjusts"]):
        raise SystemExit(f"{case['case_id']}: stack-adjust immediates mismatch: {stack_adjusts}")
    if bool(callee.get("source_is_sret")) != bool(case["expected_is_sret"]):
        raise SystemExit(f"{case['case_id']}: callee SRET flag mismatch")
    if bool(case["expected_is_sret"]) and int(callee.get("source_sret_dest_reg", -1)) != 0:
        raise SystemExit(f"{case['case_id']}: SRET callee must report hidden dest reg 0")
    if int(callee.get("source_f128_param_count", -1)) != int(case["expected_f128_param_count"]):
        raise SystemExit(f"{case['case_id']}: f128 param count mismatch")
    if callee.get("source_f128_opaque_direct_call_return_promoted") is not True:
        raise SystemExit(f"{case['case_id']}: f128 internal call promotion flag missing")
    if MIR_OP_STACK_ARG_PUSH in callee_ops:
        raise SystemExit(f"{case['case_id']}: callee must not contain caller-side stack pushes")
    return {
        "main_arg_move_indices": arg_indices,
        "main_arg_move_source_stack_slots": arg_slots,
        "main_stack_arg_push_indices": stack_indices,
        "main_stack_arg_push_source_stack_slots": stack_slots,
        "main_stack_adjust_immediates": stack_adjusts,
        "main_opcodes": main_ops,
        "callee_opcodes": callee_ops,
        "callee_source_is_sret": bool(callee.get("source_is_sret")),
        "callee_source_sret_dest_reg": int(callee.get("source_sret_dest_reg", -1)),
        "callee_source_f128_param_count": int(callee.get("source_f128_param_count", 0)),
        "callee_internal_call_promoted": bool(callee.get("source_f128_opaque_direct_call_return_promoted")),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_dir = out_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    src = case_dir / f"{case['case_id']}.sio"
    elf = case_dir / "a.out"
    machine = case_dir / "machine.json"
    src.write_text(str(case["source"]), encoding="utf-8")
    rc, compile_log = run(
        [str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(machine)],
        root,
        timeout_s,
    )
    (case_dir / "compile.log").write_text(compile_log, encoding="utf-8")
    if rc != 0 or "native_v2_compile: emitted" not in compile_log:
        raise SystemExit(f"{case['case_id']}: expected native-v2 emission")
    os.chmod(elf, 0o755)
    run_rc, run_log = run([str(elf)], root, timeout_s)
    (case_dir / "run.log").write_text(run_log, encoding="utf-8")
    expected_exit = int(case["expected_exit"])
    if run_rc != expected_exit:
        raise SystemExit(f"{case['case_id']}: ELF exit {run_rc}, expected {expected_exit}")
    module = load_machine(machine)
    shape = validate_shape(module, case)
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "source_sha256": sha256_text(str(case["source"])),
        "compile_rc": rc,
        "run_rc": run_rc,
        "expected_exit": expected_exit,
        "machine_module_supported": True,
        "machine_module_sha256": sha256_text(stable_json(module)),
        "machine_shape": shape,
    }


def emit(args: argparse.Namespace) -> None:
    root = root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [emit_case(root, compiler, out_dir, case, args.timeout_s) for case in CASES]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "case_id": "s5_5_f128_sret_internal_arg_boundary",
        "case_count": len(cases),
        "direct_control_case_count": 1,
        "sret_case_count": 3,
        "sret_stack_case_count": 1,
        "f128_internal_sret_arg_boundary_promoted": True,
        "f128_internal_sret_arg_register_boundary_promoted": True,
        "f128_internal_sret_arg_stack_boundary_promoted": True,
        "f128_compact_vreg_classifier_base_only_promoted": True,
        "f128_external_sysv_abi_promoted": False,
        "f128_sret_abi_promoted": False,
        "f128_arithmetic_promoted": False,
        "f128_software_helpers_promoted": False,
        "f128_nan_inf_contract_promoted": False,
        "cases": cases,
    }
    canonical = stable_json(payload)
    payload["receipt_sha256"] = sha256_text(canonical)
    (out_dir / "madaros_v2_s5_f128_sret_internal_arg_boundary.receipt.json").write_text(
        pretty_json(payload),
        encoding="utf-8",
    )
    print(f"[f128-sret-internal-arg-boundary] PASS receipt_sha256={payload['receipt_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--compiler", default=str(root_from_script() / "bin/madaros"))
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()
    if args.cmd == "emit":
        emit(args)


if __name__ == "__main__":
    main()
