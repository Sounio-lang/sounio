#!/usr/bin/env python3
"""Emit a Madaros v2 S5 source front-end SRET receipt.

This receipt proves real source programs returning aggregates from local
functions are lowered to the native-v2 SRET ABI path. It is deliberately
narrower than S5 FULL: it covers local aggregate returns through register and stack integer arguments only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.source_sret_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_SOURCE_SRET_LOCAL_STACK_ARGS_NOT_FULL"

MIR_OP_LOAD_STACK = 100
MIR_OP_ARG_MOVE = 112
MIR_OP_CALL = 113
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
MIR_OP_ALLOC = 116
MIR_OP_FIELD_LOAD = 117
MIR_OP_FIELD_STORE = 118
MIR_OP_STACK_ADJUST = 129
MIR_OP_STACK_ARG_PUSH = 130

ONE_ARG_SOURCE = """struct Big {
    f0: i64,
    f1: i64,
    f2: i64,
}

fn make(v: i64) -> Big {
    Big { f0: v, f1: v * 2, f2: v + 1 }
}

fn main() -> i64 {
    let b = make(7)
    b.f1
}
"""

REGISTER_MULTI_ARG_SOURCE = """struct Big {
    f0: i64,
    f1: i64,
    f2: i64,
}

fn make(a: i64, b: i64, c: i64, d: i64, e: i64) -> Big {
    Big { f0: a + c, f1: b * 2, f2: d + e }
}

fn main() -> i64 {
    let x = make(3, 7, 4, 20, 2)
    x.f0 + x.f1 + x.f2
}
"""

STACK_ONE_ARG_SOURCE = """struct Big {
    f0: i64,
    f1: i64,
    f2: i64,
}

fn make(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> Big {
    Big { f0: a + c + f, f1: b * 2, f2: d + e }
}

fn main() -> i64 {
    let x = make(3, 7, 4, 20, 2, 6)
    x.f0 + x.f1 + x.f2
}
"""

STACK_TWO_ARG_SOURCE = """struct Big {
    f0: i64,
    f1: i64,
    f2: i64,
}

fn make(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64) -> Big {
    Big { f0: a + c + f, f1: b * 2, f2: d + e + g }
}

fn main() -> i64 {
    let x = make(3, 7, 4, 20, 2, 6, 8)
    x.f0 + x.f1 + x.f2
}
"""

CASES = [
    {
        "case_id": "source_sret_local_i64_triple_return_14",
        "source": ONE_ARG_SOURCE,
        "expected_exit": 14,
        "program_kind": "source_local_one_arg_aggregate_sret_return",
        "expected_arg_indices": [0, 1],
        "expected_arg_source_slots": [1, 0],
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
        "explicit_arg_count": 1,
    },
    {
        "case_id": "source_sret_local_register_multi_arg_return_43",
        "source": REGISTER_MULTI_ARG_SOURCE,
        "expected_exit": 43,
        "program_kind": "source_local_register_multi_arg_aggregate_sret_return",
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [5, 0, 1, 2, 3, 4],
        "explicit_arg_count": 5,
        "expected_stack_arg_indices": [],
        "expected_stack_arg_source_slots": [],
        "expected_stack_adjusts": [],
    },
    {
        "case_id": "source_sret_local_stack_one_arg_return_49",
        "source": STACK_ONE_ARG_SOURCE,
        "expected_exit": 49,
        "program_kind": "source_local_stack_one_arg_aggregate_sret_return",
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [6, 0, 1, 2, 3, 4],
        "expected_stack_arg_indices": [6],
        "expected_stack_arg_source_slots": [5],
        "expected_stack_adjusts": [-8, 16],
        "explicit_arg_count": 6,
    },
    {
        "case_id": "source_sret_local_stack_two_arg_return_57",
        "source": STACK_TWO_ARG_SOURCE,
        "expected_exit": 57,
        "program_kind": "source_local_stack_two_arg_aggregate_sret_return",
        "expected_arg_indices": [0, 1, 2, 3, 4, 5],
        "expected_arg_source_slots": [7, 0, 1, 2, 3, 4],
        "expected_stack_arg_indices": [7, 6],
        "expected_stack_arg_source_slots": [6, 5],
        "expected_stack_adjusts": [16],
        "explicit_arg_count": 7,
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
        raise SystemExit("source SRET receipt canonical JSON roundtrip changed bytes")
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
        raise SystemExit("source SRET MachineModule must contain make + main")
    functions = payload.get("functions")
    if not isinstance(functions, list) or len(functions) != 2:
        raise SystemExit("source SRET MachineModule functions array mismatch")
    instr_total = 0
    for fn in functions:
        instrs = fn.get("instrs")
        if not isinstance(instrs, list):
            raise SystemExit("MachineModule function lacks instrs")
        if int(fn.get("instr_count", -1)) != len(instrs):
            raise SystemExit("MachineModule instr_count mismatch")
        instr_total += len(instrs)
    if int(payload.get("total_machine_instr_count", -1)) != instr_total:
        raise SystemExit("MachineModule total_machine_instr_count mismatch")
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


def arg_move_indices(function: dict[str, Any]) -> list[int]:
    return [int(instr[10]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_ARG_MOVE]


def arg_move_source_stack_slots(function: dict[str, Any]) -> list[int]:
    instrs = function["instrs"]
    slots: list[int] = []
    for idx, instr in enumerate(instrs):
        if int(instr[0]) != MIR_OP_ARG_MOVE:
            continue
        if idx == 0 or int(instrs[idx - 1][0]) != MIR_OP_LOAD_STACK:
            raise SystemExit("source SRET ARG_MOVE must be fed by immediate LOAD_STACK")
        slots.append(int(instrs[idx - 1][4]))
    return slots


def stack_arg_push_indices(function: dict[str, Any]) -> list[int]:
    return [int(instr[10]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_STACK_ARG_PUSH]


def stack_arg_push_source_stack_slots(function: dict[str, Any]) -> list[int]:
    instrs = function["instrs"]
    slots: list[int] = []
    for idx, instr in enumerate(instrs):
        if int(instr[0]) != MIR_OP_STACK_ARG_PUSH:
            continue
        if idx == 0 or int(instrs[idx - 1][0]) != MIR_OP_LOAD_STACK:
            raise SystemExit("source SRET STACK_ARG_PUSH must be fed by immediate LOAD_STACK")
        slots.append(int(instrs[idx - 1][4]))
    return slots


def stack_adjust_immediates(function: dict[str, Any]) -> list[int]:
    return [int(instr[4]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_STACK_ADJUST]


def validate_source_sret_machine_module(
    module: dict[str, Any],
    expected_arg_indices: list[int],
    expected_arg_source_slots: list[int],
    expected_stack_arg_indices: list[int],
    expected_stack_arg_source_slots: list[int],
    expected_stack_adjusts: list[int],
) -> dict[str, Any]:
    by_name = functions_by_name(module)
    if set(by_name) != {"make", "main"}:
        raise SystemExit(f"source SRET functions must be make/main, got {sorted(by_name)}")
    make_fn = by_name["make"]
    main_fn = by_name["main"]
    make_ops = opcodes(make_fn)
    main_ops = opcodes(main_fn)
    required_main = {
        MIR_OP_ALLOC,
        MIR_OP_ARG_MOVE,
        MIR_OP_CALL,
        MIR_OP_CAPTURE_RET,
        MIR_OP_FIELD_LOAD,
        MIR_OP_RET,
    }
    required_make = {MIR_OP_FIELD_LOAD, MIR_OP_FIELD_STORE, MIR_OP_RET}
    missing_main = sorted(required_main - set(main_ops))
    missing_make = sorted(required_make - set(make_ops))
    if missing_main:
        raise SystemExit(f"source SRET main MachineModule missing opcodes: {missing_main}")
    if missing_make:
        raise SystemExit(f"source SRET make MachineModule missing opcodes: {missing_make}")
    indices = arg_move_indices(main_fn)
    if indices != expected_arg_indices:
        raise SystemExit(f"source SRET call arg indices mismatch: expected {expected_arg_indices}, got {indices}")
    stack_slots = arg_move_source_stack_slots(main_fn)
    if stack_slots != expected_arg_source_slots:
        raise SystemExit(
            "source SRET call source stack slots mismatch: "
            f"expected {expected_arg_source_slots}, got {stack_slots}"
        )
    stack_indices = stack_arg_push_indices(main_fn)
    if stack_indices != expected_stack_arg_indices:
        raise SystemExit(
            "source SRET stack arg indices mismatch: "
            f"expected {expected_stack_arg_indices}, got {stack_indices}"
        )
    stack_arg_slots = stack_arg_push_source_stack_slots(main_fn)
    if stack_arg_slots != expected_stack_arg_source_slots:
        raise SystemExit(
            "source SRET stack arg source slots mismatch: "
            f"expected {expected_stack_arg_source_slots}, got {stack_arg_slots}"
        )
    stack_adjusts = stack_adjust_immediates(main_fn)
    if stack_adjusts != expected_stack_adjusts:
        raise SystemExit(
            "source SRET stack adjust mismatch: "
            f"expected {expected_stack_adjusts}, got {stack_adjusts}"
        )
    if expected_stack_arg_indices and MIR_OP_STACK_ARG_PUSH not in main_ops:
        raise SystemExit("source SRET stack case must contain STACK_ARG_PUSH")
    if not expected_stack_arg_indices and MIR_OP_STACK_ARG_PUSH in main_ops:
        raise SystemExit("source SRET register-only case must not contain STACK_ARG_PUSH")
    if make_ops.count(MIR_OP_FIELD_STORE) < 6:
        raise SystemExit("source SRET make must store three local fields and copy three SRET fields")
    if make_ops.count(MIR_OP_FIELD_LOAD) < 3:
        raise SystemExit("source SRET make must field-load local aggregate fields before SRET copy")
    if main_ops.count(MIR_OP_FIELD_LOAD) < 1:
        raise SystemExit("source SRET main must load b.f1 from the returned aggregate")
    if make_ops.count(MIR_OP_ARG_MOVE) != 0:
        raise SystemExit("source SRET callee MachineModule should not contain caller-side ARG_MOVE")
    return {
        "make_opcodes": make_ops,
        "main_opcodes": main_ops,
        "main_arg_move_indices": indices,
        "main_arg_move_source_stack_slots": stack_slots,
        "main_stack_arg_push_indices": stack_indices,
        "main_stack_arg_push_source_stack_slots": stack_arg_slots,
        "main_stack_adjust_immediates": stack_adjusts,
        "make_field_store_count": make_ops.count(MIR_OP_FIELD_STORE),
        "make_field_load_count": make_ops.count(MIR_OP_FIELD_LOAD),
        "main_field_load_count": main_ops.count(MIR_OP_FIELD_LOAD),
        "main_instr_count": int(main_fn["instr_count"]),
        "make_instr_count": int(make_fn["instr_count"]),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    expected_exit = int(case["expected_exit"])
    explicit_arg_count = int(case["explicit_arg_count"])
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.compile.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"

    source_path.write_text(source_text, encoding="utf-8")
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
        raise SystemExit(f"source SRET compile failed for {case_id} rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)

    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != expected_exit:
        raise SystemExit(f"source SRET {case_id} expected exit {expected_exit}, got {actual_exit}")

    module = load_machine_module(mm_path)
    machine_shape = validate_source_sret_machine_module(
        module,
        list(case["expected_arg_indices"]),
        list(case["expected_arg_source_slots"]),
        list(case["expected_stack_arg_indices"]),
        list(case["expected_stack_arg_source_slots"]),
        list(case["expected_stack_adjusts"]),
    )
    register_param_count = min(explicit_arg_count, 5)
    stack_arg_count = max(explicit_arg_count - 5, 0)
    return {
        "case_id": case_id,
        "program_kind": case["program_kind"],
        "source": source_path.name,
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "explicit_arg_count": explicit_arg_count,
        "source_sha256": sha256_text(source_text),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "machine_module_schema": module["schema"],
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_shape": machine_shape,
        "abi_signature": {
            "aggregate_return": {"type": "Big", "flat_i64_fields": 3, "size_bytes": 24},
            "hidden_return_destination": {
                "register": "rdi",
                "machine_arg_index": 0,
                "caller_source_stack_slot": int(case["expected_arg_source_slots"][0]),
            },
            "params": [
                {
                    "type": "i64",
                    "class": "scalar_i64",
                    "register": reg,
                    "machine_arg_index": idx + 1,
                    "caller_source_stack_slot": idx,
                }
                for idx, reg in enumerate(["rsi", "rdx", "rcx", "r8", "r9"][:register_param_count])
            ],
            "stack_params": [
                {
                    "type": "i64",
                    "class": "scalar_i64",
                    "machine_arg_index": idx + 6,
                    "caller_source_stack_slot": idx + 5,
                }
                for idx in range(stack_arg_count)
            ],
            "return": {"type": "aggregate_pointer", "class": "sret_hidden_dest", "register": "rax"},
            "sret": True,
            "aggregate_layout": True,
            "stack_arg_count": stack_arg_count,
        },
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = out_dir / "madaros_v2_s5_source_sret.receipt.json"
    case_results = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    one_arg = case_results[0]
    multi_arg = case_results[1]
    stack_one_arg = case_results[2]
    stack_two_arg = case_results[3]
    receipt = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "source_sret_local_register_and_stack_args",
        "case_count": len(case_results),
        "cases": case_results,
        "abi_kind": "aggregate_sret_hidden_dest_call_return",
        "one_arg_case_id": one_arg["case_id"],
        "one_arg_actual_exit": one_arg["actual_exit"],
        "register_multi_arg_case_id": multi_arg["case_id"],
        "register_multi_arg_actual_exit": multi_arg["actual_exit"],
        "stack_one_arg_case_id": stack_one_arg["case_id"],
        "stack_one_arg_actual_exit": stack_one_arg["actual_exit"],
        "stack_two_arg_case_id": stack_two_arg["case_id"],
        "stack_two_arg_actual_exit": stack_two_arg["actual_exit"],
        "s5_source_sret_local_one_arg_complete": True,
        "s5_source_sret_local_register_multi_arg_complete": True,
        "s5_source_sret_local_stack_arg_complete": True,
        "source_frontend_lowers_local_aggregate_return_to_IrCallSret": True,
        "source_frontend_lowers_local_register_multi_arg_aggregate_return_to_IrCallSret": True,
        "source_frontend_lowers_local_stack_arg_aggregate_return_to_IrCallSret": True,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "s_full_contract": "blocked_until_multi_arg_imported_stack_f64_numeric_diagnostics_and_differential_gates_exist",
        "roundtrip_contract": [
            "real_source_program_compiles_with_native_v2_compile",
            "MachineModule_contains_make_and_main_without_legacy_fallback",
            "one_arg_caller_uses_IrCallSret_shape_with_two_ARG_MOVE_ops",
            "register_multi_arg_caller_uses_IrCallSret_shape_with_six_ARG_MOVE_ops",
            "stack_arg_callers_use_STACK_ARG_PUSH_not_ARG_MOVE_6",
            "one_stack_arg_case_records_alignment_padding_and_cleanup",
            "two_stack_arg_case_records_cleanup_without_padding",
            "hidden_dest_loaded_to_machine_arg_0",
            "explicit_register_args_loaded_to_machine_args_1_through_5",
            "callee_copies_three_fields_to_hidden_dest",
            "native_elves_return_expected_discriminators",
        ],
        "missing_full_obligations": [
            "imported aggregate/SRET receipt",
            "method/generic/module-boundary aggregate return coverage",
            "f64 XMM0 call/return receipt before f128 promotion",
            "numeric tower width receipts for f128/i256",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    canonical, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-source-sret: case={receipt['case_id']} "
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
