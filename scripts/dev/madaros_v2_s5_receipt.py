#!/usr/bin/env python3
"""Emit a Madaros v2 S5 scalar program MIR/ABI receipt.

This is a canonical per-source receipt for the current S5 scalar slice. It
proves that one native-v2 scalar witness compiles, runs with the expected exit
code, exposes the expected merged-IR shape in the compiler log, exports the
compiler's MachineModule JSON for that witness, and has native ELF
call/return/syscall evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.receipt/0.1"
PROGRAM_SCHEMA = "madaros.v2.s5.program_mir_abi_program/0.1"
PROGRAM_MIR_SCHEMA = "madaros.v2.s5.program_mir_shadow/0.1"
ABI_SCHEMA = "madaros.v2.s5.abi_scalar_call_return/0.1"
STAGE_CONTRACT_LEVEL = "S5_SCALAR_MACHINE_MODULE_EXPORT_WITH_ABI_SHADOW_NOT_FULL"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def parse_function_count(text: str) -> int:
    match = re.search(r"Merged IR:\s*(\d+)\s+functions", text)
    if not match:
        raise SystemExit("cannot parse Merged IR function count from native-v2 compile log")
    return int(match.group(1))


def run_compile(
    compiler: Path,
    source_arg: str,
    bin_path: Path,
    machine_module_path: Path,
    root: Path,
    timeout_s: int,
) -> tuple[int, str, str]:
    proc = subprocess.run(
        [
            str(compiler),
            "--native-v2-compile",
            source_arg,
            "-o",
            str(bin_path),
            "--machine-module-json",
            str(machine_module_path),
        ],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def run_binary(bin_path: Path, timeout_s: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.run(
        [str(bin_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def elf_exec_metrics(path: Path) -> dict[str, int]:
    blob = path.read_bytes()
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise SystemExit(f"not an ELF64 little-endian binary: {path}")

    phoff = struct.unpack_from("<Q", blob, 32)[0]
    phentsize = struct.unpack_from("<H", blob, 54)[0]
    phnum = struct.unpack_from("<H", blob, 56)[0]
    internal_call_count = 0
    ret_count = 0
    syscall_count = 0

    for index in range(phnum):
        off = phoff + index * phentsize
        if off + phentsize > len(blob):
            raise SystemExit(f"bad program header table in {path}")
        p_type, p_flags = struct.unpack_from("<II", blob, off)
        p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz, _p_align = struct.unpack_from("<QQQQQQ", blob, off + 8)
        if p_type != 1 or not (p_flags & 1):
            continue
        segment = blob[p_offset : p_offset + p_filesz]
        if len(segment) != p_filesz:
            raise SystemExit(f"truncated executable segment in {path}")
        ret_count += segment.count(b"\xc3")
        syscall_count += segment.count(b"\x0f\x05")
        for cursor in range(0, max(0, len(segment) - 4)):
            if segment[cursor] != 0xE8:
                continue
            rel = struct.unpack_from("<i", segment, cursor + 1)[0]
            target = p_vaddr + cursor + 5 + rel
            if p_vaddr <= target < p_vaddr + p_filesz:
                internal_call_count += 1

    return {
        "elf_internal_call_count": internal_call_count,
        "elf_ret_count": ret_count,
        "elf_syscall_count": syscall_count,
    }


def expected_from_case_id(case_id: str) -> int | None:
    known = {
        "scalar_i64_literal_return_42": 42,
        "scalar_i64_direct_call_return_42": 42,
        "scalar_bool_direct_call_return_1": 1,
    }
    return known.get(case_id)


def classify_program(case_id: str, source_text: str, expected_exit: int, function_count: int, metrics: dict[str, int]) -> dict[str, Any]:
    internal_calls = int(metrics["elf_internal_call_count"])
    if case_id == "scalar_i64_literal_return_42":
        kind = "scalar_i64_literal_return"
        abi_kind = "scalar_i64_return"
        expected_function_count = 1
        expected_internal_calls = 1
        entry_ops = ["MOV_IMM", "STORE_STACK", "LOAD_STACK", "RET"]
        boundary_ops = ["RET"]
        signature = {
            "params": [],
            "return": {"type": "i64", "class": "scalar_i64", "register": "rax"},
            "arg_registers_used": [],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        }
    elif case_id == "scalar_i64_direct_call_return_42":
        kind = "scalar_i64_direct_call_return"
        abi_kind = "scalar_i64_direct_call_return"
        expected_function_count = 2
        expected_internal_calls = 2
        entry_ops = ["LOAD_STACK", "ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "LOAD_STACK", "RET"]
        boundary_ops = ["ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"]
        signature = {
            "params": [{"type": "i64", "class": "scalar_i64", "register": "rdi"}],
            "return": {"type": "i64", "class": "scalar_i64", "register": "rax"},
            "arg_registers_used": ["rdi"],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        }
    elif case_id == "scalar_bool_direct_call_return_1":
        kind = "scalar_bool_direct_call_return"
        abi_kind = "scalar_bool_direct_call_return"
        expected_function_count = 2
        expected_internal_calls = 2
        entry_ops = ["LOAD_STACK", "ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "LOAD_STACK", "RET"]
        boundary_ops = ["ARG_MOVE", "CALL", "CAPTURE_RET", "STORE_STACK", "RET"]
        signature = {
            "params": [{"type": "i64", "class": "scalar_i64", "register": "rdi"}],
            "return": {"type": "bool", "class": "scalar_bool", "register": "rax", "canonical_values": [0, 1]},
            "arg_registers_used": ["rdi"],
            "stack_arg_count": 0,
            "sret": False,
            "aggregate_layout": False,
        }
        if "-> bool" not in source_text:
            raise SystemExit("bool scalar S5 receipt requires an explicit bool-return source witness")
    else:
        raise SystemExit(f"unsupported S5 scalar receipt case_id: {case_id}")

    if expected_exit not in {1, 42}:
        raise SystemExit(f"unsupported S5 scalar expected exit: {expected_exit}")
    if function_count != expected_function_count:
        raise SystemExit(f"{case_id}: function_count={function_count} expected={expected_function_count}")
    if internal_calls != expected_internal_calls:
        raise SystemExit(f"{case_id}: internal_calls={internal_calls} expected={expected_internal_calls}")

    return {
        "program_kind": kind,
        "abi_kind": abi_kind,
        "entry_function_legal_mir_ops": entry_ops,
        "call_boundary_ops": boundary_ops,
        "abi_signature": signature,
    }


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    parsed = json.loads(first)
    second = stable_json(parsed)
    if first != second:
        raise SystemExit("S5 canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def load_machine_module(path: Path, case_id: str, source_rel: str, function_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"{case_id}: compiler did not export MachineModule JSON at {path}; "
            "S5 requires --native-v2-compile --machine-module-json support"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "madaros.v2.s5.machine_module/0.1":
        raise SystemExit(f"{case_id}: bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit(f"{case_id}: MachineModule source is not native_v2_build_machine_module")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"{case_id}: MachineModule export flag is not true")
    if payload.get("target") != "x86_64-linux":
        raise SystemExit(f"{case_id}: MachineModule target mismatch")
    if payload.get("active") is not True:
        raise SystemExit(f"{case_id}: MachineModule is not active")
    if payload.get("supported") is not True:
        raise SystemExit(f"{case_id}: MachineModule unsupported: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id}: MachineModule must not use legacy fallback")
    if int(payload.get("fn_count", -1)) != function_count:
        raise SystemExit(f"{case_id}: MachineModule fn_count != merged IR function count")
    functions = payload.get("functions")
    if not isinstance(functions, list) or len(functions) != function_count:
        raise SystemExit(f"{case_id}: MachineModule functions array mismatch")
    instr_total = 0
    for fn in functions:
        instrs = fn.get("instrs")
        if not isinstance(instrs, list):
            raise SystemExit(f"{case_id}: MachineModule function lacks instrs")
        if int(fn.get("instr_count", -1)) != len(instrs):
            raise SystemExit(f"{case_id}: MachineModule instr_count mismatch")
        instr_total += len(instrs)
    if instr_total <= 0:
        raise SystemExit(f"{case_id}: MachineModule has no instructions")
    if int(payload.get("total_machine_instr_count", -1)) != instr_total:
        raise SystemExit(f"{case_id}: MachineModule total_machine_instr_count mismatch")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    source = Path(args.source).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_id = args.case_id or source.stem
    expected_exit = args.expected_exit
    if expected_exit is None:
        expected_exit = expected_from_case_id(case_id)
    if expected_exit is None:
        raise SystemExit("--expected-exit is required for non-canonical S5 scalar receipt cases")

    source_rel = relpath(source, root)
    source_arg = source_rel if not source_rel.startswith("/") else str(source)
    bin_path = out_dir / f"{case_id}.native_v2"
    machine_module_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.compile.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    program_path = out_dir / f"{case_id}.s5.program_mir_abi.json"
    receipt_path = out_dir / f"{case_id}.s5.receipt.json"

    rc, compile_stdout, compile_stderr = run_compile(compiler, source_arg, bin_path, machine_module_path, root, args.timeout)
    compile_log = compile_stdout + compile_stderr
    compile_log_path.write_text(compile_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"native-v2 compile failed for {case_id}; log={compile_log_path}")
    bin_path.chmod(bin_path.stat().st_mode | 0o111)

    actual_exit, stdout, stderr = run_binary(bin_path, args.timeout)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id}: expected_exit={expected_exit} actual_exit={actual_exit}")

    normalized_compile_log = normalize_log(compile_log, out_dir)
    function_count = parse_function_count(compile_log)
    machine_module = load_machine_module(machine_module_path, case_id, source_rel, function_count)
    metrics = elf_exec_metrics(bin_path)
    source_text = source.read_text(encoding="utf-8")
    classification = classify_program(case_id, source_text, expected_exit, function_count, metrics)

    program = {
        "schema": PROGRAM_SCHEMA,
        "case_id": case_id,
        "program": source_rel,
        "program_kind": classification["program_kind"],
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "merged_ir_function_count": function_count,
        "program_mir_schema": PROGRAM_MIR_SCHEMA,
        "program_mir_source": "compiler_exported_machine_module_json",
        "compiler_machine_module_exported": True,
        "machine_module_schema": machine_module["schema"],
        "machine_module_path": machine_module_path.name,
        "machine_module_json_sha256": machine_module["machine_module_json_sha256"],
        "entry_function_legal_mir_ops": classification["entry_function_legal_mir_ops"],
        "call_boundary_ops": classification["call_boundary_ops"],
        "machine_ir_contract_source": "self-hosted/native/machine_ir.sio:native_v2_lower_legal_function_from_ir_ref",
        "codegen_contract_source": "self-hosted/native/codegen_x86_linux.sio:native_v2_emit_machine_instr",
        "abi_schema": ABI_SCHEMA,
        "abi_signature": classification["abi_signature"],
        "native_v2_compile": {
            "elf_sha256": sha256_bytes(bin_path.read_bytes()),
            "compile_log_sha256": sha256_text(normalized_compile_log),
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
            **metrics,
        },
    }
    program["program_shadow_sha256"] = sha256_text(stable_json(program))
    program_canonical, program_sha = canonical_roundtrip(program)
    program["program_roundtrip_sha256"] = sha256_text(program_canonical)
    program_path.write_text(pretty_json(program), encoding="utf-8")

    receipt = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "case_id": case_id,
        "source": source_rel,
        "source_sha256": sha256_bytes(source.read_bytes()),
        "compiler": str(compiler),
        "compiler_route_kind": args.compiler_route_kind,
        "parser_sha": args.parser_sha,
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "program_kind": classification["program_kind"],
        "abi_kind": classification["abi_kind"],
        "abi_schema": ABI_SCHEMA,
        "abi_signature": classification["abi_signature"],
        "program_mir_abi_program_path": program_path.name,
        "program_mir_abi_program_sha256": program_sha,
        "program_shadow_sha256": program["program_shadow_sha256"],
        "machine_module_schema": machine_module["schema"],
        "machine_module_path": machine_module_path.name,
        "machine_module_json_sha256": machine_module["machine_module_json_sha256"],
        "merged_ir_function_count": function_count,
        "native_v2_compile": program["native_v2_compile"],
        "s5_receipt_ready": True,
        "s5_program_mir_abi_scalar_shadow_slice_complete": True,
        "s5_compiler_machine_module_export_slice_complete": True,
        "s5_mir_abi_boundary_complete": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "s_full_contract": "blocked_until_full_abi_numeric_differential_gates_exist",
        "program_mir_shadow_serialized": True,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": False,
        "roundtrip_contract": [
            "canonical_json_stable_after_parse_dump",
            "source_to_native_v2_compile_exit_matches_expected",
            "compiler_exported_machine_module_json_present",
            "machine_module_total_instr_count_matches_function_instrs",
            "merged_ir_function_count_matches_program_shape",
            "elf_internal_call_count_matches_program_shape",
            "scalar_abi_register_contract_recorded",
            "full_abi_numeric_differential_gates_still_required_before_s5_ready",
        ],
        "missing_full_obligations": [
            "ABI layout receipts for aggregate, SRET, imported call, stack-arg, and return paths",
            "f128 IR/MIR/ABI/software-helper receipts before f128 promotion",
            "f128 IR/MIR/ABI/software-helper receipts",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")

    print(
        f"madaros-v2-s5: case={case_id} kind={classification['program_kind']} "
        f"exit={actual_exit} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
    )
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
    emit_p.add_argument("--expected-exit", type=int, default=None)
    emit_p.add_argument("--root", default=str(repo_root_from_script()))
    emit_p.add_argument("--timeout", type=int, default=120)
    emit_p.set_defaults(func=emit)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
