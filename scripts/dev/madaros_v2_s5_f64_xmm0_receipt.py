#!/usr/bin/env python3
"""Emit a Madaros v2 S5 f64/XMM0 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.f64_xmm0_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_F64_XMM0_CALL_RETURN_PROMOTED"

MIR_OP_LOAD_STACK = 100
MIR_OP_STORE_STACK = 101
MIR_OP_ARG_MOVE = 112
MIR_OP_CALL = 113
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
MIR_OP_LOAD_FLOAT = 122
MIR_OP_FLOAT_BINOP = 123
MIR_OP_FLOAT_TO_INT = 125

X86_CVTTSD2SI_RAX_XMM0 = bytes.fromhex("f2480f2cc0")
X86_MOVQ_XMM0_RAX = bytes.fromhex("66480f6ec0")
X86_MOVQ_RAX_XMM0 = bytes.fromhex("66480f7ec0")
X86_F64_4_5_MOVABS = bytes.fromhex("48b80000000000001240")


CASES: list[dict[str, Any]] = [
    {
        "case_id": "f64_cast_literal_to_i64_return_4",
        "source": """fn main() -> i64 {
    let x = 4.5
    x as i64
}
""",
        "expected_exit": 4,
        "expected_stdout": b"",
        "expected_functions": ["main"],
        "required_ops": [MIR_OP_LOAD_FLOAT, MIR_OP_FLOAT_TO_INT, MIR_OP_RET],
        "required_bytes": ["cvttsd2si", "movq_xmm0_rax", "load_4_5"],
    },
    {
        "case_id": "f64_fractional_binop_cast_return_50",
        "source": """fn main() -> i64 {
    let x = 0.5 * 100.0
    x as i64
}
""",
        "expected_exit": 50,
        "expected_stdout": b"",
        "expected_functions": ["main"],
        "required_ops": [MIR_OP_LOAD_FLOAT, MIR_OP_FLOAT_BINOP, MIR_OP_FLOAT_TO_INT, MIR_OP_RET],
        "required_bytes": ["cvttsd2si", "movq_xmm0_rax", "movq_rax_xmm0"],
    },
    {
        "case_id": "f64_return_compare_exit_45",
        "source": """fn half(x: f64) -> f64 { x / 2.0 }
fn main() -> i64 {
    let y = half(9.0)
    if y > 4.4 { 45 } else { 1 }
}
""",
        "expected_exit": 45,
        "expected_stdout": b"",
        "expected_functions": ["half", "main"],
        "required_ops": [MIR_OP_LOAD_FLOAT, MIR_OP_FLOAT_BINOP, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_RET],
        "required_bytes": ["movq_xmm0_rax", "movq_rax_xmm0"],
    },
    {
        "case_id": "f64_mixed_args_return_compare_exit_55",
        "source": """fn mix(a: i64, x: f64, b: i64, y: f64) -> f64 { x + y + (a as f64) + (b as f64) }
fn main() -> i64 {
    let z = mix(10, 20.0, 5, 20.0)
    if z > 54.0 { 55 } else { 1 }
}
""",
        "expected_exit": 55,
        "expected_stdout": b"",
        "expected_functions": ["mix", "main"],
        "required_ops": [MIR_OP_ARG_MOVE, MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_RET],
        "required_bytes": ["movq_xmm0_rax", "movq_rax_xmm0"],
    },
    {
        "case_id": "f64_println_call_stdout_4_5",
        "source": """fn val() -> f64 { 4.5 }
fn main() -> i64 { println(val()) 0 }
""",
        "expected_exit": 0,
        "expected_stdout": b"4.500000\n",
        "expected_functions": ["val", "main", "print_f64", "print_char"],
        "required_ops": [MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_ARG_MOVE, MIR_OP_RET],
        "required_bytes": ["cvttsd2si", "movq_xmm0_rax", "movq_rax_xmm0", "load_4_5"],
        "must_call": ["print_f64", "print_char"],
        "must_not_call": ["print"],
    },
    {
        "case_id": "f64_print_call_stdout_4_5",
        "source": """fn val() -> f64 { 4.5 }
fn main() -> i64 { print(val()) 0 }
""",
        "expected_exit": 0,
        "expected_stdout": b"4.500000",
        "expected_functions": ["val", "main", "print_f64"],
        "required_ops": [MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_ARG_MOVE, MIR_OP_RET],
        "required_bytes": ["cvttsd2si", "movq_xmm0_rax", "movq_rax_xmm0", "load_4_5"],
        "must_call": ["print_f64"],
        "must_not_call": ["print"],
    },
    {
        "case_id": "f64_let_bound_println_stdout_4_5",
        "source": """fn val() -> f64 { 4.5 }
fn main() -> i64 {
    let y = val()
    println(y)
    0
}
""",
        "expected_exit": 0,
        "expected_stdout": b"4.500000\n",
        "expected_functions": ["val", "main", "print_f64", "print_char"],
        "required_ops": [MIR_OP_CALL, MIR_OP_CAPTURE_RET, MIR_OP_ARG_MOVE, MIR_OP_STORE_STACK, MIR_OP_RET],
        "required_bytes": ["cvttsd2si", "movq_xmm0_rax", "movq_rax_xmm0", "load_4_5"],
        "must_call": ["print_f64", "print_char"],
        "must_not_call": ["print"],
    },
]


BYTE_PATTERNS = {
    "cvttsd2si": X86_CVTTSD2SI_RAX_XMM0,
    "movq_xmm0_rax": X86_MOVQ_XMM0_RAX,
    "movq_rax_xmm0": X86_MOVQ_RAX_XMM0,
    "load_4_5": X86_F64_4_5_MOVABS,
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
        raise SystemExit("f64 XMM0 receipt canonical JSON roundtrip changed bytes")
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
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(fn.get("name", "")): fn for fn in module.get("functions", []) if fn.get("name")}


def all_opcodes(module: dict[str, Any]) -> list[int]:
    ops: list[int] = []
    for fn in module.get("functions", []):
        ops.extend(int(instr[0]) for instr in fn.get("instrs", []))
    return ops


def main_call_targets(module: dict[str, Any]) -> list[str]:
    by_id = [str(fn.get("name", "")) for fn in module.get("functions", [])]
    by_name = functions_by_name(module)
    main_fn = by_name.get("main")
    if not main_fn:
        raise SystemExit("MachineModule missing main")
    names: list[str] = []
    for instr in main_fn.get("instrs", []):
        if int(instr[0]) != MIR_OP_CALL:
            continue
        fn_id = int(instr[4])
        if fn_id < 0 or fn_id >= len(by_id):
            raise SystemExit(f"bad call fn_id in main: {fn_id}")
        names.append(by_id[fn_id])
    return names


def validate_case(module: dict[str, Any], elf: bytes, case: dict[str, Any]) -> dict[str, Any]:
    by_name = functions_by_name(module)
    missing_fns = [name for name in case["expected_functions"] if name not in by_name]
    if missing_fns:
        raise SystemExit(f"{case['case_id']} missing functions: {missing_fns}")
    if "print" in by_name and "print" in case.get("must_not_call", []):
        raise SystemExit(f"{case['case_id']} routed f64 print through string print")
    ops = all_opcodes(module)
    missing_ops = [op for op in case["required_ops"] if op not in ops]
    if missing_ops:
        raise SystemExit(f"{case['case_id']} missing MachineModule ops: {missing_ops}")
    missing_bytes = [name for name in case["required_bytes"] if BYTE_PATTERNS[name] not in elf]
    if missing_bytes:
        raise SystemExit(f"{case['case_id']} missing ELF byte patterns: {missing_bytes}")
    calls = main_call_targets(module)
    for name in case.get("must_call", []):
        if name not in calls:
            raise SystemExit(f"{case['case_id']} main does not call {name}; calls={calls}")
    for name in case.get("must_not_call", []):
        if name in calls:
            raise SystemExit(f"{case['case_id']} main unexpectedly calls {name}")
    return {
        "function_names": sorted(by_name),
        "main_call_targets": calls,
        "required_ops_present": sorted(set(case["required_ops"])),
        "required_bytes_present": sorted(case["required_bytes"]),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
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
        raise SystemExit(f"f64 XMM0 compile failed for {case_id} rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != int(case["expected_exit"]):
        raise SystemExit(f"{case_id} expected exit {case['expected_exit']}, got {actual_exit}")
    if stdout != case["expected_stdout"]:
        raise SystemExit(f"{case_id} expected stdout {case['expected_stdout']!r}, got {stdout!r}")
    module = load_machine_module(mm_path)
    machine_shape = validate_case(module, elf_path.read_bytes(), case)
    return {
        "case_id": case_id,
        "source": source_path.name,
        "expected_exit": int(case["expected_exit"]),
        "actual_exit": actual_exit,
        "expected_stdout": case["expected_stdout"].decode("utf-8"),
        "actual_stdout": stdout.decode("utf-8"),
        "source_sha256": sha256_text(source_text),
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
    receipt_path = out_dir / "madaros_v2_s5_f64_xmm0.receipt.json"
    case_results = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "f64_xmm0_call_return",
        "case_count": len(case_results),
        "cases": case_results,
        "s5_f64_xmm0_call_return_complete": True,
        "source_frontend_dispatches_print_to_print_f64": True,
        "source_frontend_dispatches_println_f64_to_print_f64": True,
        "source_frontend_tracks_let_bound_f64_identifiers": True,
        "ir_lowers_f64_literals_to_IrLoadFloat": True,
        "native_v2_lowers_f64_to_i64_cast": True,
        "native_v2_lowers_fractional_f64_binops": True,
        "native_v2_materializes_print_f64_fraction_scale_without_rodata_relocation": True,
        "native_v2_bridges_print_f64_arg0_to_xmm0": True,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "f64_xmm0_promoted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "f64_literal_uses_IrLoadFloat",
            "f64_to_i64_cast_emits_FLOAT_TO_INT_and_cvttsd2si",
            "fractional_f64_binop_survives_cast_to_i64",
            "f64_call_return_preserves_xmm0_result_bits",
            "mixed_i64_f64_args_preserve_expected_numeric_result",
            "print_and_println_f64_dispatch_to_print_f64_not_string_print",
            "let_bound_f64_identifier_routes_to_print_f64",
            "print_f64_fraction_scale_is_materialized_as_immediate_bits",
        ],
        "missing_full_obligations": [
            "generic aggregate return coverage",
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
        f"madaros-v2-s5-f64-xmm0: cases={receipt['case_count']} "
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
