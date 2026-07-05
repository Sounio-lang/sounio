#!/usr/bin/env python3
"""Emit a Madaros v2 S5 method aggregate-return SRET receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.method_sret_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_METHOD_SRET_STACK_ARGS_NOT_FULL"

MIR_OP_LOAD_STACK = 100
MIR_OP_ARG_MOVE = 112
MIR_OP_CALL = 113
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
MIR_OP_ALLOC = 116
MIR_OP_FIELD_LOAD = 117
MIR_OP_STACK_ADJUST = 129
MIR_OP_STACK_ARG_PUSH = 130

PROGRAM_TEMPLATE = """struct Big {{
    f0: i64,
    f1: i64,
    f2: i64,
}}

struct Maker {{
    seed: i64,
}}

impl Maker {{
    fn make(self{params}) -> Big {{
        {body}
    }}
}}

fn main() -> i64 {{
    let m = Maker {{ seed: {seed} }}
    let x = m.make({call_args})
    x.f0 + x.f1 + x.f2
}}
"""

CASES = [
    {
        "case_id": "method_sret_receiver_only_return_24",
        "params": "",
        "body": "Big { f0: self.seed, f1: self.seed + 1, f2: self.seed + 2 }",
        "seed": 7,
        "call_args": "",
        "expected_exit": 24,
        "method_source_param_count": 1,
        "expected_arg_indices": [0, 1],
        "expected_arg_source_slots": [2, 0],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
    },
    {
        "case_id": "method_sret_receiver_register_args_return_43",
        "params": ", a: i64, b: i64, c: i64, d: i64",
        "body": "Big { f0: self.seed + a + c, f1: b * 2, f2: d + 1 }",
        "seed": 3,
        "call_args": "4, 7, 5, 16",
        "expected_exit": 43,
        "method_source_param_count": 5,
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [6, 0, 2, 3, 4, 5],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
    },
    {
        "case_id": "method_sret_receiver_stack_args_return_57",
        "params": ", a: i64, b: i64, c: i64, d: i64, e: i64, f: i64",
        "body": "Big { f0: self.seed + a + c + e, f1: b * 2, f2: d + f }",
        "seed": 3,
        "call_args": "4, 7, 5, 16, 6, 9",
        "expected_exit": 57,
        "method_source_param_count": 7,
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [8, 0, 2, 3, 4, 5],
        "expected_stack_arg_indices": [7, 6],
        "expected_stack_arg_source_slots": [7, 6],
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
        raise SystemExit("method SRET receipt canonical JSON roundtrip changed bytes")
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
    if int(payload.get("fn_count", -1)) != 2:
        raise SystemExit("method SRET MachineModule must contain Maker_make + main")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def opcodes(function: dict[str, Any]) -> list[int]:
    return [int(instr[0]) for instr in function["instrs"]]


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for fn in module["functions"]:
        name = str(fn.get("name", ""))
        if name:
            result[name] = fn
    return result


def indices(function: dict[str, Any], opcode: int) -> list[int]:
    return [int(instr[10]) for instr in function["instrs"] if int(instr[0]) == opcode]


def source_slots_for(function: dict[str, Any], opcode: int) -> list[int]:
    instrs = function["instrs"]
    slots: list[int] = []
    for idx, instr in enumerate(instrs):
        if int(instr[0]) != opcode:
            continue
        if idx == 0 or int(instrs[idx - 1][0]) != MIR_OP_LOAD_STACK:
            raise SystemExit("method SRET move/push must be fed by immediate LOAD_STACK")
        slots.append(int(instrs[idx - 1][4]))
    return slots


def stack_adjusts(function: dict[str, Any]) -> list[int]:
    return [int(instr[4]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_STACK_ADJUST]


def field_load_indices(function: dict[str, Any]) -> list[int]:
    return [int(instr[8]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_FIELD_LOAD]


def validate_machine_shape(module: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    by_name = functions_by_name(module)
    if "main" not in by_name or "Maker_make" not in by_name:
        raise SystemExit(f"method SRET functions missing: got {sorted(by_name)}")
    main_fn = by_name["main"]
    method_fn = by_name["Maker_make"]
    if int(method_fn.get("source_is_sret", 0)) != 1:
        raise SystemExit("method SRET callee must preserve source_is_sret=1")
    if int(method_fn.get("source_param_count", -1)) != int(case["method_source_param_count"]):
        raise SystemExit("method SRET callee source_param_count mismatch")
    main_ops = opcodes(main_fn)
    method_ops = opcodes(method_fn)
    required_main = {MIR_OP_ALLOC, MIR_OP_ARG_MOVE, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_FIELD_LOAD, MIR_OP_RET}
    missing = sorted(required_main - set(main_ops))
    if missing:
        raise SystemExit(f"method SRET main MachineModule missing opcodes: {missing}")
    if MIR_OP_ARG_MOVE in method_ops or MIR_OP_STACK_ARG_PUSH in method_ops:
        raise SystemExit("method SRET callee body must not contain caller-side arg setup")
    arg_indices = indices(main_fn, MIR_OP_ARG_MOVE)
    if arg_indices != case["expected_arg_indices"]:
        raise SystemExit(f"method SRET register arg indices mismatch: expected {case['expected_arg_indices']}, got {arg_indices}")
    arg_slots = source_slots_for(main_fn, MIR_OP_ARG_MOVE)
    if arg_slots != case["expected_arg_source_slots"]:
        raise SystemExit(f"method SRET register source slots mismatch: expected {case['expected_arg_source_slots']}, got {arg_slots}")
    stack_indices = indices(main_fn, MIR_OP_STACK_ARG_PUSH)
    if stack_indices != case["expected_stack_arg_indices"]:
        raise SystemExit(f"method SRET stack arg indices mismatch: expected {case['expected_stack_arg_indices']}, got {stack_indices}")
    stack_slots = source_slots_for(main_fn, MIR_OP_STACK_ARG_PUSH)
    if stack_slots != case["expected_stack_arg_source_slots"]:
        raise SystemExit(f"method SRET stack source slots mismatch: expected {case['expected_stack_arg_source_slots']}, got {stack_slots}")
    adjusts = stack_adjusts(main_fn)
    if adjusts != case["expected_stack_adjusts"]:
        raise SystemExit(f"method SRET stack adjusts mismatch: expected {case['expected_stack_adjusts']}, got {adjusts}")
    field_indices = field_load_indices(main_fn)
    if field_indices != [0, 1, 2]:
        raise SystemExit(f"method SRET aggregate field indices mismatch: expected [0, 1, 2], got {field_indices}")
    return {
        "main_opcodes": main_ops,
        "method_opcodes": method_ops,
        "main_arg_move_indices": arg_indices,
        "main_arg_move_source_stack_slots": arg_slots,
        "main_stack_arg_push_indices": stack_indices,
        "main_stack_arg_push_source_stack_slots": stack_slots,
        "main_stack_adjust_immediates": adjusts,
        "main_field_load_indices": field_indices,
        "main_instr_count": int(main_fn["instr_count"]),
        "method_instr_count": int(method_fn["instr_count"]),
        "method_source_is_sret": int(method_fn.get("source_is_sret", 0)),
        "method_source_param_count": int(method_fn.get("source_param_count", -1)),
        "method_source_sret_dest_reg": int(method_fn.get("source_sret_dest_reg", -1)),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    program_text = PROGRAM_TEMPLATE.format(
        params=case["params"],
        body=case["body"],
        seed=case["seed"],
        call_args=case["call_args"],
    )
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.compile.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    source_path.write_text(program_text, encoding="utf-8")
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
    if rc != 0:
        raise SystemExit(f"method SRET compile failed for {case_id} rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != int(case["expected_exit"]):
        raise SystemExit(f"method SRET {case_id} expected exit {case['expected_exit']}, got {actual_exit}")
    module = load_machine_module(mm_path)
    machine_shape = validate_machine_shape(module, case)
    return {
        "case_id": case_id,
        "source": source_path.name,
        "expected_exit": int(case["expected_exit"]),
        "actual_exit": actual_exit,
        "method_source_param_count": int(case["method_source_param_count"]),
        "source_sha256": sha256_text(program_text),
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
    receipt_path = out_dir / "madaros_v2_s5_method_sret.receipt.json"
    case_results = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    receipt = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "method_aggregate_sret",
        "case_count": len(case_results),
        "cases": case_results,
        "s5_method_sret_receiver_only_complete": True,
        "s5_method_sret_receiver_register_args_complete": True,
        "s5_method_sret_receiver_stack_args_complete": True,
        "source_frontend_lowers_method_aggregate_return_to_IrCallSret": True,
        "compiler_machine_module_exported_for_method_path": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "method_receiver_only_sret_executes_expected_exit",
            "method_register_arg_sret_executes_expected_exit",
            "method_stack_arg_sret_executes_expected_exit",
            "method_path_exports_machine_module_json",
            "method_sret_hidden_dest_is_arg0",
            "method_receiver_shifts_after_hidden_dest",
            "method_explicit_args_shift_after_receiver",
            "method_stack_args_use_STACK_ARG_PUSH",
            "method_callee_source_is_sret_recorded",
        ],
        "missing_full_obligations": [
            "generic aggregate return coverage",
            "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
            "f128 IR/MIR/ABI/software-helper receipts",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-method-sret: case={receipt['case_id']} "
        f"cases={receipt['case_count']} sha={receipt['receipt_sha256'][:12]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--compiler", default=str(repo_root_from_script() / "bin/madaros"))
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.cmd == "emit":
        return emit(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
