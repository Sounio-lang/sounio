#!/usr/bin/env python3
"""Emit a Madaros v2 S5 imported aggregate-return SRET receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.imported_sret_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_IMPORTED_SRET_MODULE_BOUNDARY_NOT_FULL"

MIR_OP_LOAD_STACK = 100
MIR_OP_ARG_MOVE = 112
MIR_OP_CALL = 113
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
MIR_OP_ALLOC = 116
MIR_OP_FIELD_LOAD = 117
MIR_OP_STACK_ADJUST = 129
MIR_OP_STACK_ARG_PUSH = 130

LIB_TEMPLATE = """pub struct Big {{
    f0: i64,
    f1: i64,
    f2: i64,
}}

pub fn make({params}) -> Big {{
    {body}
}}
"""

MAIN_TEMPLATE = """use {module_name}::{{Big, make}}

fn main() -> i64 {{
    let x = {call}
    x.f0 + x.f1 + x.f2
}}
"""

CASES = [
    {
        "case_id": "imported_sret_one_arg_return_29",
        "module_name": "imported_sret_one_lib",
        "params": "v: i64",
        "body": "Big { f0: v, f1: v * 2, f2: v + 1 }",
        "call": "make(7)",
        "expected_exit": 29,
        "explicit_arg_count": 1,
        "expected_arg_indices": [0, 1],
        "expected_arg_source_slots": [1, 0],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
    },
    {
        "case_id": "imported_sret_register_multi_arg_return_43",
        "module_name": "imported_sret_reg_lib",
        "params": "a: i64, b: i64, c: i64, d: i64, e: i64",
        "body": "Big { f0: a + c, f1: b * 2, f2: d + e }",
        "call": "make(3, 7, 4, 20, 2)",
        "expected_exit": 43,
        "explicit_arg_count": 5,
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [5, 0, 1, 2, 3, 4],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
    },
    {
        "case_id": "imported_sret_stack_two_arg_return_57",
        "module_name": "imported_sret_stack_lib",
        "params": "a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64",
        "body": "Big { f0: a + c + f, f1: b * 2, f2: d + e + g }",
        "call": "make(3, 7, 4, 20, 2, 6, 8)",
        "expected_exit": 57,
        "explicit_arg_count": 7,
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [7, 0, 1, 2, 3, 4],
        "expected_stack_arg_indices": [7, 6],
        "expected_stack_arg_source_slots": [6, 5],
        "expected_stack_adjusts": [16],
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
    proc = subprocess.run(
        [str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("imported SRET receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def load_machine_module(path: Path) -> dict[str, Any]:
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
    if int(payload.get("fn_count", -1)) != 3:
        raise SystemExit("imported SRET MachineModule must contain main + import stub + imported body")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def functions_by_name(module: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for fn in module["functions"]:
        name = str(fn.get("name", ""))
        if name:
            result.setdefault(name, []).append(fn)
    return result


def opcodes(function: dict[str, Any]) -> list[int]:
    return [int(instr[0]) for instr in function["instrs"]]


def indices(function: dict[str, Any], opcode: int) -> list[int]:
    return [int(instr[10]) for instr in function["instrs"] if int(instr[0]) == opcode]


def source_slots_for(function: dict[str, Any], opcode: int) -> list[int]:
    instrs = function["instrs"]
    slots: list[int] = []
    for idx, instr in enumerate(instrs):
        if int(instr[0]) != opcode:
            continue
        if idx == 0 or int(instrs[idx - 1][0]) != MIR_OP_LOAD_STACK:
            raise SystemExit("imported SRET move/push must be fed by immediate LOAD_STACK")
        slots.append(int(instrs[idx - 1][4]))
    return slots


def stack_adjusts(function: dict[str, Any]) -> list[int]:
    return [int(instr[4]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_STACK_ADJUST]


def field_load_indices(function: dict[str, Any]) -> list[int]:
    return [int(instr[8]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_FIELD_LOAD]


def validate_machine_shape(module: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    by_name = functions_by_name(module)
    if "main" not in by_name or "make" not in by_name:
        raise SystemExit(f"imported SRET functions missing: got {sorted(by_name)}")
    if len(by_name["make"]) < 2:
        raise SystemExit("imported SRET must preserve import stub plus imported make body")
    main_fn = by_name["main"][0]
    make_fns = by_name["make"]
    main_ops = opcodes(main_fn)
    required_main = {MIR_OP_ALLOC, MIR_OP_ARG_MOVE, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_FIELD_LOAD, MIR_OP_RET}
    missing = sorted(required_main - set(main_ops))
    if missing:
        raise SystemExit(f"imported SRET main MachineModule missing opcodes: {missing}")
    arg_indices = indices(main_fn, MIR_OP_ARG_MOVE)
    if arg_indices != case["expected_arg_indices"]:
        raise SystemExit(f"imported SRET register arg indices mismatch: expected {case['expected_arg_indices']}, got {arg_indices}")
    arg_slots = source_slots_for(main_fn, MIR_OP_ARG_MOVE)
    if arg_slots != case["expected_arg_source_slots"]:
        raise SystemExit(f"imported SRET register source slots mismatch: expected {case['expected_arg_source_slots']}, got {arg_slots}")
    stack_indices = indices(main_fn, MIR_OP_STACK_ARG_PUSH)
    if stack_indices != case["expected_stack_arg_indices"]:
        raise SystemExit(f"imported SRET stack arg indices mismatch: expected {case['expected_stack_arg_indices']}, got {stack_indices}")
    stack_slots = source_slots_for(main_fn, MIR_OP_STACK_ARG_PUSH)
    if stack_slots != case["expected_stack_arg_source_slots"]:
        raise SystemExit(f"imported SRET stack source slots mismatch: expected {case['expected_stack_arg_source_slots']}, got {stack_slots}")
    adjusts = stack_adjusts(main_fn)
    if adjusts != case["expected_stack_adjusts"]:
        raise SystemExit(f"imported SRET stack adjusts mismatch: expected {case['expected_stack_adjusts']}, got {adjusts}")
    field_indices = field_load_indices(main_fn)
    if field_indices != [0, 1, 2]:
        raise SystemExit(f"imported SRET imported aggregate field indices mismatch: expected [0, 1, 2], got {field_indices}")
    for make_fn in make_fns:
        if MIR_OP_ARG_MOVE in opcodes(make_fn) or MIR_OP_STACK_ARG_PUSH in opcodes(make_fn):
            raise SystemExit("imported SRET make body must not contain caller-side arg setup")
    return {
        "main_opcodes": main_ops,
        "make_function_count": len(make_fns),
        "make_opcodes": [opcodes(fn) for fn in make_fns],
        "main_arg_move_indices": arg_indices,
        "main_arg_move_source_stack_slots": arg_slots,
        "main_stack_arg_push_indices": stack_indices,
        "main_stack_arg_push_source_stack_slots": stack_slots,
        "main_stack_adjust_immediates": adjusts,
        "main_field_load_indices": field_indices,
        "main_instr_count": int(main_fn["instr_count"]),
        "make_instr_counts": [int(fn["instr_count"]) for fn in make_fns],
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    module_name = str(case["module_name"])
    lib_text = LIB_TEMPLATE.format(params=case["params"], body=case["body"])
    main_text = MAIN_TEMPLATE.format(module_name=module_name, call=case["call"])
    lib_path = out_dir / f"{module_name}.sio"
    main_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.compile.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    lib_path.write_text(lib_text, encoding="utf-8")
    main_path.write_text(main_text, encoding="utf-8")
    rc, compile_stdout, compile_stderr = run_command(
        [
            str(compiler),
            "--native-v2-compile",
            str(main_path),
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
    if rc != 0:
        raise SystemExit(f"imported SRET compile failed for {case_id} rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != int(case["expected_exit"]):
        raise SystemExit(f"imported SRET {case_id} expected exit {case['expected_exit']}, got {actual_exit}")
    module = load_machine_module(mm_path)
    machine_shape = validate_machine_shape(module, case)
    return {
        "case_id": case_id,
        "main_source": main_path.name,
        "lib_source": lib_path.name,
        "expected_exit": int(case["expected_exit"]),
        "actual_exit": actual_exit,
        "explicit_arg_count": int(case["explicit_arg_count"]),
        "main_source_sha256": sha256_text(main_text),
        "lib_source_sha256": sha256_text(lib_text),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_shape": machine_shape,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_imported_sret.receipt.json"
    case_results = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    receipt = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "imported_aggregate_sret_module_boundary",
        "case_count": len(case_results),
        "cases": case_results,
        "s5_imported_sret_module_boundary_complete": True,
        "source_frontend_lowers_imported_aggregate_return_to_IrCallSret": True,
        "compiler_machine_module_exported_for_imported_path": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "imported_one_arg_sret_executes_expected_exit",
            "imported_register_multi_arg_sret_executes_expected_exit",
            "imported_stack_arg_sret_executes_expected_exit",
            "imported_path_exports_machine_module_json",
            "imported_sret_hidden_dest_is_arg0",
            "imported_sret_explicit_args_shift_after_hidden_dest",
            "imported_sret_stack_args_use_STACK_ARG_PUSH",
        ],
        "missing_full_obligations": [
            "generic aggregate return coverage",
            "f64 XMM0 call/return receipt before f128 promotion",
            "numeric tower width receipts for f128/i256",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-imported-sret: case={receipt['case_id']} "
        f"cases={receipt['case_count']} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
