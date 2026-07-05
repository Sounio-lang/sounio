#!/usr/bin/env python3
"""Emit a Madaros v2 S5 SRET ABI receipt.

This receipt covers the compiler-owned native-v2 SRET witness, not a source
front-end lowering claim. It proves the real IrCallSret/IrReturnSret module is
accepted by the MachineModule exporter, that the positive ELF returns 14, and
that the plain-call negative discriminator does not return 14.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.sret_abi_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_SRET_MACHINE_MODULE_ABI_DISCRIMINATOR_NOT_SOURCE_SRET_FULL"


MIR_OP_ARG_MOVE = 112
MIR_OP_CALL = 113
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
MIR_OP_ALLOC = 116
MIR_OP_FIELD_LOAD = 117
MIR_OP_FIELD_STORE = 118


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


def run_binary(path: Path, timeout_s: int) -> tuple[int, bytes, bytes]:
    proc = subprocess.run(
        [str(path)],
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
    if payload.get("supported") is not True:
        raise SystemExit(f"MachineModule unsupported: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    if int(payload.get("fn_count", -1)) != 2:
        raise SystemExit("SRET MachineModule must contain main + make")
    functions = payload.get("functions")
    if not isinstance(functions, list) or len(functions) != 2:
        raise SystemExit("SRET MachineModule functions array mismatch")
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


def arg_move_indices(function: dict[str, Any]) -> list[int]:
    return [int(instr[10]) for instr in function["instrs"] if int(instr[0]) == MIR_OP_ARG_MOVE]


def validate_sret_machine_module(module: dict[str, Any]) -> dict[str, Any]:
    main_fn = module["functions"][0]
    make_fn = module["functions"][1]
    main_ops = opcodes(main_fn)
    make_ops = opcodes(make_fn)
    required_main = {MIR_OP_ALLOC, MIR_OP_ARG_MOVE, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_FIELD_LOAD, MIR_OP_RET}
    required_make = {MIR_OP_FIELD_STORE, MIR_OP_RET}
    missing_main = sorted(required_main - set(main_ops))
    missing_make = sorted(required_make - set(make_ops))
    if missing_main:
        raise SystemExit(f"SRET main MachineModule missing opcodes: {missing_main}")
    if missing_make:
        raise SystemExit(f"SRET make MachineModule missing opcodes: {missing_make}")
    indices = arg_move_indices(main_fn)
    if indices[:2] != [0, 1]:
        raise SystemExit(f"SRET call must move hidden dest to arg0 and explicit v to arg1, got {indices}")
    if make_ops.count(MIR_OP_FIELD_STORE) < 3:
        raise SystemExit("SRET make must store three aggregate fields")
    if main_ops.count(MIR_OP_FIELD_LOAD) < 1:
        raise SystemExit("SRET main must load returned aggregate field")
    return {
        "main_opcodes": main_ops,
        "make_opcodes": make_ops,
        "sret_arg_move_indices": indices,
        "field_store_count": make_ops.count(MIR_OP_FIELD_STORE),
        "field_load_count": main_ops.count(MIR_OP_FIELD_LOAD),
    }


def validate_plaincall_negative_machine_module(module: dict[str, Any]) -> dict[str, Any]:
    main_fn = module["functions"][0]
    make_fn = module["functions"][1]
    main_ops = opcodes(main_fn)
    make_ops = opcodes(make_fn)
    required_main = {MIR_OP_ALLOC, MIR_OP_ARG_MOVE, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_FIELD_LOAD, MIR_OP_RET}
    required_make = {MIR_OP_FIELD_STORE, MIR_OP_RET}
    missing_main = sorted(required_main - set(main_ops))
    missing_make = sorted(required_make - set(make_ops))
    if missing_main:
        raise SystemExit(f"SRET plaincall negative main MachineModule missing opcodes: {missing_main}")
    if missing_make:
        raise SystemExit(f"SRET plaincall negative make MachineModule missing opcodes: {missing_make}")
    indices = arg_move_indices(main_fn)
    if indices != [0]:
        raise SystemExit(f"SRET plaincall negative must pass only one explicit arg at arg0, got {indices}")
    if make_ops.count(MIR_OP_FIELD_STORE) < 3:
        raise SystemExit("SRET plaincall negative make must still store three aggregate fields")
    if main_ops.count(MIR_OP_FIELD_LOAD) < 1:
        raise SystemExit("SRET plaincall negative main must load destination aggregate field")
    return {
        "main_opcodes": main_ops,
        "make_opcodes": make_ops,
        "plaincall_arg_move_indices": indices,
        "field_store_count": make_ops.count(MIR_OP_FIELD_STORE),
        "field_load_count": main_ops.count(MIR_OP_FIELD_LOAD),
    }


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("SRET receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_elf = out_dir / "sret_positive.native_v2"
    positive_mm = out_dir / "sret_positive.machine_module.json"
    positive_log = out_dir / "sret_positive.compile.log"
    negative_elf = out_dir / "sret_plaincall_negative.native_v2"
    negative_mm = out_dir / "sret_plaincall_negative.machine_module.json"
    negative_log = out_dir / "sret_plaincall_negative.compile.log"
    receipt_path = out_dir / "madaros_v2_s5_sret_abi.receipt.json"

    pos_rc, pos_stdout, pos_stderr = run_command(
        [
            str(compiler),
            "--native-v2-emit-sret",
            str(positive_elf),
            "--machine-module-json",
            str(positive_mm),
        ],
        root,
        args.timeout,
    )
    positive_log.write_text(pos_stdout + pos_stderr, encoding="utf-8")
    if pos_rc != 0:
        raise SystemExit(f"positive SRET compile failed rc={pos_rc}; log={positive_log}")
    positive_elf.chmod(positive_elf.stat().st_mode | 0o111)
    positive_exit, positive_stdout, positive_stderr = run_binary(positive_elf, args.timeout)
    if positive_exit != 14:
        raise SystemExit(f"positive SRET expected exit 14, got {positive_exit}")

    neg_rc, neg_stdout, neg_stderr = run_command(
        [
            str(compiler),
            "--native-v2-emit-sret-plaincall",
            str(negative_elf),
            "--machine-module-json",
            str(negative_mm),
        ],
        root,
        args.timeout,
    )
    negative_log.write_text(neg_stdout + neg_stderr, encoding="utf-8")
    if neg_rc != 0:
        raise SystemExit(f"negative SRET compile failed rc={neg_rc}; log={negative_log}")
    negative_elf.chmod(negative_elf.stat().st_mode | 0o111)
    negative_exit, negative_stdout, negative_stderr = run_binary(negative_elf, args.timeout)
    if negative_exit == 14:
        raise SystemExit("SRET plaincall negative unexpectedly returned the positive discriminator 14")

    positive_module = load_machine_module(positive_mm)
    negative_module = load_machine_module(negative_mm)
    positive_shape = validate_sret_machine_module(positive_module)
    negative_shape = validate_plaincall_negative_machine_module(negative_module)

    receipt = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "sret_i64_triple_return_14",
        "program_kind": "compiler_owned_ir_sret_i64_triple_return",
        "abi_kind": "aggregate_sret_hidden_dest_call_return",
        "source": "self-hosted/compiler/main.sio:compiler_main_make_native_v2_sret_module",
        "positive": {
            "expected_exit": 14,
            "actual_exit": positive_exit,
            "elf_sha256": sha256_bytes(positive_elf.read_bytes()),
            "compile_log_sha256": sha256_text((pos_stdout + pos_stderr).replace(str(out_dir), "<OUT_DIR>")),
            "stdout_sha256": sha256_bytes(positive_stdout),
            "stderr_sha256": sha256_bytes(positive_stderr),
            "machine_module_path": positive_mm.name,
            "machine_module_json_sha256": positive_module["machine_module_json_sha256"],
            "native_v2_compile": elf_exec_metrics(positive_elf),
            "machine_shape": positive_shape,
        },
        "negative_plaincall": {
            "expected_not_exit": 14,
            "actual_exit": negative_exit,
            "elf_sha256": sha256_bytes(negative_elf.read_bytes()),
            "compile_log_sha256": sha256_text((neg_stdout + neg_stderr).replace(str(out_dir), "<OUT_DIR>")),
            "stdout_sha256": sha256_bytes(negative_stdout),
            "stderr_sha256": sha256_bytes(negative_stderr),
            "machine_module_path": negative_mm.name,
            "machine_module_json_sha256": negative_module["machine_module_json_sha256"],
            "native_v2_compile": elf_exec_metrics(negative_elf),
            "machine_shape": negative_shape,
        },
        "abi_signature": {
            "aggregate_return": {"type": "Big", "flat_i64_fields": 3, "size_bytes": 24},
            "hidden_return_destination": {"register": "rdi", "machine_arg_index": 0},
            "params": [{"type": "i64", "class": "scalar_i64", "register": "rsi", "machine_arg_index": 1}],
            "return": {"type": "aggregate_pointer", "class": "sret_hidden_dest", "register": "rax"},
            "sret": True,
            "aggregate_layout": True,
            "stack_arg_count": 0,
        },
        "s5_sret_machine_module_abi_discriminator_complete": True,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "s_full_contract": "blocked_until_source_sret_lowering_imported_stack_f64_numeric_and_differential_gates_exist",
        "roundtrip_contract": [
            "compiler_owned_ir_uses_IrCallSret_and_IrReturnSret",
            "machine_module_accepts_sret_without_legacy_fallback",
            "hidden_dest_arg_index_0_and_explicit_param_arg_index_1",
            "positive_sret_elf_returns_14",
            "plaincall_negative_does_not_return_14",
            "aggregate_field_store_and_field_load_present",
        ],
        "missing_full_obligations": [
            "source front-end lowering to IrCallSret for by-value aggregate returns",
            "imported aggregate/SRET receipt",
            "stack-arg SRET receipt",
            "f64 XMM0 call/return receipt",
            "numeric tower width receipts for f128/i256",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    canonical, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")

    print(
        f"madaros-v2-s5-sret: case={receipt['case_id']} "
        f"positive={positive_exit} negative={negative_exit} "
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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
