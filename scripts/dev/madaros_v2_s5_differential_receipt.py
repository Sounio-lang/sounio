#!/usr/bin/env python3
"""Emit a Madaros v2 S5 native-v2 vs lean_single differential receipt.

This receipt is intentionally program-level: every promoted positive S5 source
surface that the legacy lean_single reference can execute is compiled through
the explicit native-v2 path, executed as an ELF, and compared against
SOUNIO_SOUC_ENGINE=lean_single. Cases with a known reference-output mismatch
are recorded as unavailable rather than counted as equivalence evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.differential_receipt/0.2"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_27_NATIVE_V2_LEAN_SINGLE_DIFFERENTIAL_WITH_F128_PROMOTED_SURFACES"

UNAVAILABLE_REFERENCE_CASES = {
    "f64_println_call_stdout_4_5": "lean_single prints f64 println without the trailing newline emitted by native-v2 print_char",
    "f64_let_bound_println_stdout_4_5": "lean_single prints let-bound f64 println without the trailing newline emitted by native-v2 print_char",
    "imported_f128_identity_arg_return": "lean_single returns rc=1 for imported f128 return while native-v2 executes the promoted imported f128 ABI surface",
    "imported_f128_return_only": "lean_single returns rc=1 for imported f128 return while native-v2 executes the promoted imported f128 ABI surface",
    "imported_f128_arg_i64_return": "lean_single returns rc=1 for imported f128 arg call while native-v2 executes the promoted imported f128 ABI surface",
    "imported_f128_plus_i64_arg_return": "lean_single returns rc=1 for imported mixed f128/i64 call while native-v2 executes the promoted imported f128 ABI surface",
    "imported_two_f128_args_return": "lean_single returns rc=1 for imported two-f128-arg call while native-v2 executes the promoted imported f128 ABI surface",
    "imported_two_f128_params_non_overlapping": "lean_single returns rc=1 for imported f128 param layout case while native-v2 executes the promoted imported f128 layout surface",
}

REQUIRED_CATEGORIES = {
    "scalar_i64",
    "scalar_bool",
    "normal_call_stack_args",
    "source_sret_local",
    "imported_sret_module_boundary",
    "method_sret",
    "f64_xmm0",
    "wide_int_source",
    "generic_aggregate_sret",
    "f128_arithmetic_value_contract",
    "f128_opaque_call_return_abi",
    "f128_sret_internal_arg_boundary",
    "f128_param_slot_layout",
    "f128_binary128_native_materialization",
    "f128_ieee_class_helper",
    "f128_ieee_predicate_helper",
}

EXPECTED_CASE_COUNT = 257
EXPECTED_MATCHED_CASE_COUNT = 219

F128_PARAM_EXPECTED_EXITS = {
    "local_two_f128_params_non_overlapping": 5,
    "local_f128_i64_f128_params_non_overlapping": 7,
    "imported_two_f128_params_non_overlapping": 12,
    "f128_callee_add_args_slot_layout_feeds_runtime_helper": 0,
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


def signed_i64(value: int) -> int:
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def u64_words_from_hex(hex_text: str) -> tuple[int, int]:
    bits = int(hex_text, 16)
    return (bits >> 64) & ((1 << 64) - 1), bits & ((1 << 64) - 1)


def mov_rax_imm_pattern(value: int) -> bytes:
    signed = signed_i64(value)
    if -(1 << 31) <= signed <= (1 << 31) - 1:
        return b"\x48\xc7\xc0" + (value & ((1 << 32) - 1)).to_bytes(4, "little", signed=False)
    return b"\x48\xb8" + (value & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)


def metadata_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    meta = module.get("f128_literal_metadata", {})
    for fn in meta.get("functions", []):
        for row in fn.get("rows", []):
            if isinstance(row, list) and len(row) >= 7:
                rows.append([int(x) for x in row])
    return rows


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("differential receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def load_module(root: Path, name: str) -> Any:
    path = root / "scripts" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import receipt helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
        env=merged_env,
    )
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def load_machine_module(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema in {path}: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit(f"MachineModule source mismatch in {path}")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"MachineModule export flag missing in {path}")
    if payload.get("active") is not True:
        raise SystemExit(f"MachineModule inactive in {path}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit(f"MachineModule used legacy fallback in {path}")
    return payload


def case_source_rows(root: Path) -> list[dict[str, Any]]:
    stack = load_module(root, "madaros_v2_s5_stack_call_receipt")
    source_sret = load_module(root, "madaros_v2_s5_source_sret_receipt")
    imported_sret = load_module(root, "madaros_v2_s5_imported_sret_receipt")
    method_sret = load_module(root, "madaros_v2_s5_method_sret_receipt")
    f64 = load_module(root, "madaros_v2_s5_f64_xmm0_receipt")
    wide = load_module(root, "madaros_v2_s5_wide_int_receipt")
    generic = load_module(root, "madaros_v2_s5_generic_aggregate_sret_receipt")
    f128_arith = load_module(root, "madaros_v2_s5_f128_arithmetic_value_contract_receipt")
    f128_call = load_module(root, "madaros_v2_s5_f128_opaque_call_return_abi_receipt")
    f128_sret = load_module(root, "madaros_v2_s5_f128_sret_internal_arg_boundary_receipt")
    f128_param = load_module(root, "madaros_v2_s5_f128_param_slot_layout_receipt")
    f128_binary128 = load_module(root, "madaros_v2_s5_f128_binary128_value_contract_native_receipt")
    f128_class = load_module(root, "madaros_v2_s5_f128_ieee_class_helper_receipt")

    rows: list[dict[str, Any]] = []
    for case_id, source_path, expected_exit, category in [
        (
            "scalar_i64_literal_return_42",
            root / "tests/madaros/v2_s5/scalar_i64_literal_return_42.sio",
            42,
            "scalar_i64",
        ),
        (
            "scalar_i64_direct_call_return_42",
            root / "tests/madaros/v2_s5/scalar_i64_direct_call_return_42.sio",
            42,
            "scalar_i64",
        ),
        (
            "scalar_bool_direct_call_return_1",
            root / "tests/madaros/v2_s5/scalar_bool_direct_call_return_1.sio",
            1,
            "scalar_bool",
        ),
    ]:
        rows.append(
            {
                "case_id": case_id,
                "category": category,
                "source": source_path.read_text(encoding="utf-8"),
                "expected_exit": expected_exit,
                "support_files": {},
            }
        )

    for case in stack.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "normal_call_stack_args",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in source_sret.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "source_sret_local",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in imported_sret.CASES:
        lib = imported_sret.LIB_TEMPLATE.format(params=case["params"], body=case["body"])
        main = imported_sret.MAIN_TEMPLATE.format(module_name=case["module_name"], call=case["call"])
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "imported_sret_module_boundary",
                "source": main,
                "expected_exit": int(case["expected_exit"]),
                "support_files": {f"{case['module_name']}.sio": lib},
            }
        )
    for case in method_sret.CASES:
        source = method_sret.PROGRAM_TEMPLATE.format(
            params=case["params"],
            body=case["body"],
            seed=case["seed"],
            call_args=case["call_args"],
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "method_sret",
                "source": source,
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in f64.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f64_xmm0",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
                "expected_stdout": case["expected_stdout"].decode("utf-8"),
            }
        )
    for case in wide.SOURCE_NATIVE_CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "wide_int_source",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
                "wide_type": case["wide_type"],
                "wide_ops": list(case["wide_ops"]),
            }
        )
    for case in generic.CASES:
        support_files = {}
        if case["path_kind"] == "imported":
            support_files["imported_generic_wide9_lib.sio"] = generic.IMPORTED_WIDE9_LIB
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "generic_aggregate_sret",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": support_files,
                "path_kind": case["path_kind"],
                "field_count": int(case["field_count"]),
                "declared_layout_bytes": int(case["declared_layout_bytes"]),
            }
        )
    for case in f128_arith.POSITIVE:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f128_arithmetic_value_contract",
                "source": case["source"],
                "expected_exit": 0,
                "support_files": {},
                "expected_binary128_hex": case["expected_hex"],
                "expected_f128_metadata": case.get("expected_metadata"),
                "expected_machine_opcode": int(case.get("expected_machine_opcode", 0) or 0),
            }
        )
    for case in f128_call.POSITIVE_CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f128_opaque_call_return_abi",
                "source": case["source"],
                "expected_exit": int(case.get("expected_exit", 0)),
                "support_files": dict(case.get("support_files", {})),
                "callee": case["callee"],
            }
        )
    for case in f128_sret.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f128_sret_internal_arg_boundary",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
                "callee": case["callee"],
                "f128_boundary_kind": case["kind"],
            }
        )
    for case in f128_param.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f128_param_slot_layout",
                "source": case["source"],
                "expected_exit": F128_PARAM_EXPECTED_EXITS[str(case["case_id"])],
                "support_files": dict(case.get("support_files", {})),
                "callee": case["callee"],
                "expected_source_f128_param_count": int(case["expected_source_f128_param_count"]),
                "expected_f128_slot_rows": list(case["expected_f128_rows"]),
            }
        )
    for case in f128_binary128.CASES:
        value_expr = f"{case.literal} as f128"
        rows.append(
            {
                "case_id": f"f128_binary128_materialize_{case.case_id}",
                "category": "f128_binary128_native_materialization",
                "source": f"""fn main() -> i64 {{
    let x: f128 = {value_expr}
    let y: f128 = x
    0
}}
""",
                "expected_exit": 0,
                "support_files": {},
                "expected_binary128_hex": case.expected_hex,
                "expected_f128_metadata": case.expected_metadata,
                "f128_literal": case.literal,
            }
        )
    for case in f128_class.CASES:
        case_id = f"f128_class_code_{case.case_id}"
        nan_decl = "fn f128_nan() -> f128 { 0.0 as f128 }\n" if case.literal == "f128_nan()" else ""
        value_expr = case.literal if case.literal == "f128_nan()" else f"{case.literal} as f128"
        if int(case.expected_class_code) != 0:
            UNAVAILABLE_REFERENCE_CASES[case_id] = (
                "lean_single executes the source stub for f128_class_code while native-v2 executes "
                "the promoted compiler-owned IEEE class-code builtin"
            )
        rows.append(
            {
                "case_id": case_id,
                "category": "f128_ieee_class_helper",
                "source": f"""{nan_decl}fn f128_class_code(x: f128) -> i64 {{ 0 }}
fn main() -> i64 {{
    let x: f128 = {value_expr}
    f128_class_code(x)
}}
""",
                "expected_exit": int(case.expected_class_code),
                "support_files": {},
                "expected_class_name": case.expected_class_name,
                "f128_literal": case.literal,
            }
        )
    for case in f128_class.CASES:
        for predicate_name, helper_name in f128_class.PREDICATE_HELPERS:
            expected_exit = int(f128_class.expected_predicate_value(case.expected_class_name, predicate_name))
            case_id = f"f128_predicate_{case.case_id}_{helper_name}"
            nan_decl = "fn f128_nan() -> f128 { 0.0 as f128 }\n" if case.literal == "f128_nan()" else ""
            value_expr = case.literal if case.literal == "f128_nan()" else f"{case.literal} as f128"
            if expected_exit != 0:
                UNAVAILABLE_REFERENCE_CASES[case_id] = (
                    "lean_single executes the source stub for IEEE f128 predicate helpers while "
                    "native-v2 executes the promoted compiler-owned predicate builtin"
                )
            rows.append(
                {
                    "case_id": case_id,
                    "category": "f128_ieee_predicate_helper",
                    "source": f"""{nan_decl}fn {helper_name}(x: f128) -> i64 {{ 0 }}
fn main() -> i64 {{
    let x: f128 = {value_expr}
    {helper_name}(x)
}}
""",
                    "expected_exit": expected_exit,
                    "support_files": {},
                    "expected_predicate": predicate_name,
                    "expected_class_name": case.expected_class_name,
                    "f128_literal": case.literal,
                }
            )

    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            raise SystemExit(f"duplicate differential case id: {case_id}")
        seen.add(case_id)
    return rows


def emit_case(
    root: Path,
    compiler: Path,
    reference_souc: Path,
    out_dir: Path,
    case: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / f"{case_id}.sio"
    source_text = str(case["source"])
    source_path.write_text(source_text, encoding="utf-8")
    for name, text in dict(case.get("support_files", {})).items():
        (case_dir / name).write_text(str(text), encoding="utf-8")

    elf_path = case_dir / f"{case_id}.native_v2"
    mm_path = case_dir / f"{case_id}.machine_module.json"
    compile_log = case_dir / f"{case_id}.native_v2.compile.log"
    native_stdout_path = case_dir / f"{case_id}.native.stdout"
    native_stderr_path = case_dir / f"{case_id}.native.stderr"
    lean_stdout_path = case_dir / f"{case_id}.lean_single.stdout"
    lean_stderr_path = case_dir / f"{case_id}.lean_single.stderr"

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
    compile_log.write_bytes(compile_stdout + compile_stderr)
    if compile_rc != 0:
        raise SystemExit(f"{case_id} native-v2 compile failed rc={compile_rc}; log={compile_log}")
    if not elf_path.is_file():
        raise SystemExit(f"{case_id} native-v2 compile did not produce ELF")
    elf_path.chmod(0o755)
    elf_bytes = elf_path.read_bytes()
    machine_module = load_machine_module(mm_path)

    native_rc, native_stdout, native_stderr = run_command([str(elf_path)], root, timeout_s)
    native_stdout_path.write_bytes(native_stdout)
    native_stderr_path.write_bytes(native_stderr)

    lean_rc, lean_stdout, lean_stderr = run_command(
        [str(reference_souc), "run", str(source_path)],
        root,
        timeout_s,
        {"SOUNIO_SOUC_ENGINE": "lean_single"},
    )
    lean_stdout_path.write_bytes(lean_stdout)
    lean_stderr_path.write_bytes(lean_stderr)

    expected_exit = int(case["expected_exit"])
    available = case_id not in UNAVAILABLE_REFERENCE_CASES
    status = "matched" if available else "reference_unavailable"
    mismatch_reason = ""
    if native_rc != expected_exit:
        raise SystemExit(f"{case_id} native-v2 exit mismatch: expected {expected_exit}, got {native_rc}")
    if available and lean_rc != expected_exit:
        raise SystemExit(f"{case_id} lean_single exit mismatch: expected {expected_exit}, got {lean_rc}")
    if available and native_stdout != lean_stdout:
        raise SystemExit(
            f"{case_id} stdout mismatch: native={native_stdout!r} lean_single={lean_stdout!r}"
        )
    if not available:
        mismatch_reason = UNAVAILABLE_REFERENCE_CASES[case_id]
        if lean_rc == expected_exit and native_stdout == lean_stdout:
            raise SystemExit(f"{case_id} was listed unavailable but now matches; update receipt contract")

    expected_hex = case.get("expected_binary128_hex")
    expected_metadata = case.get("expected_f128_metadata")
    expected_machine_opcode = int(case.get("expected_machine_opcode", 0) or 0)
    expected_f128_slot_rows = case.get("expected_f128_slot_rows")
    f128_contract: dict[str, Any] = {}
    if expected_hex is not None:
        hi, lo = u64_words_from_hex(str(expected_hex))
        hi_pattern = mov_rax_imm_pattern(hi)
        lo_pattern = mov_rax_imm_pattern(lo)
        hi_found = hi_pattern in elf_bytes
        lo_found = lo_pattern in elf_bytes
        if not hi_found:
            raise SystemExit(f"{case_id} missing expected f128 high-word immediate")
        if lo != 0 and not lo_found:
            raise SystemExit(f"{case_id} missing expected f128 low-word immediate")
        f128_contract.update(
            {
                "expected_binary128_hex": str(expected_hex),
                "expected_hi_u64": hi,
                "expected_lo_u64": lo,
                "hi_mov_imm_pattern_found": hi_found,
                "lo_mov_imm_pattern_found": lo_found,
            }
        )
    if expected_metadata is not None:
        rows = metadata_rows(machine_module)
        expected_row = [int(x) for x in expected_metadata]
        if expected_row not in [row[1:7] for row in rows]:
            raise SystemExit(f"{case_id} missing expected f128 metadata row {expected_row}")
        f128_contract["expected_f128_metadata"] = expected_row
        f128_contract["metadata_row_found"] = True
    if expected_machine_opcode != 0:
        opcode_found = any(
            isinstance(instr, list) and len(instr) > 0 and int(instr[0]) == expected_machine_opcode
            for fn in machine_module.get("functions", [])
            for instr in fn.get("instrs", [])
        )
        if not opcode_found:
            raise SystemExit(f"{case_id} missing expected MachineIR opcode {expected_machine_opcode}")
        f128_contract["expected_machine_opcode"] = expected_machine_opcode
        f128_contract["expected_machine_opcode_found"] = True
    if expected_f128_slot_rows is not None:
        callee = str(case.get("callee", ""))
        fn = next((fn for fn in machine_module.get("functions", []) if fn.get("name") == callee), None)
        if fn is None:
            raise SystemExit(f"{case_id} missing f128 slot-layout callee {callee}")
        actual_count = int(fn.get("source_f128_param_count", -1))
        expected_count = int(case["expected_source_f128_param_count"])
        if actual_count != expected_count:
            raise SystemExit(f"{case_id} f128 param count mismatch: {actual_count} != {expected_count}")
        slots = machine_module.get("slot_metadata", {})
        actual_rows: list[list[int]] = []
        callee_index = int(fn.get("index", -1))
        for row in slots.get("functions", []):
            if int(row.get("fn_index", -2)) == callee_index:
                for slot in row.get("slots", []):
                    if isinstance(slot, list) and len(slot) >= 3 and int(slot[1]) == 3:
                        actual_rows.append([int(slot[0]), int(slot[1]), int(slot[2])])
        expected_rows = [[int(x) for x in row] for row in expected_f128_slot_rows]
        if actual_rows != expected_rows:
            raise SystemExit(f"{case_id} f128 slot rows mismatch: {actual_rows} != {expected_rows}")
        f128_contract["expected_source_f128_param_count"] = expected_count
        f128_contract["expected_f128_slot_rows"] = expected_rows
        f128_contract["f128_slot_rows_match"] = True

    row = {
        "case_id": case_id,
        "category": case["category"],
        "status": status,
        "reference_available": available,
        "reference_unavailable_reason": mismatch_reason,
        "expected_exit": expected_exit,
        "native_v2_exit": native_rc,
        "lean_single_exit": lean_rc,
        "stdout_equal": native_stdout == lean_stdout,
        "stderr_equal": native_stderr == lean_stderr,
        "source_sha256": sha256_text(source_text),
        "support_file_count": len(dict(case.get("support_files", {}))),
        "support_file_sha256": {
            name: sha256_text(str(text)) for name, text in sorted(dict(case.get("support_files", {})).items())
        },
        "machine_module_schema": machine_module["schema"],
        "machine_module_json_sha256": sha256_text(stable_json(machine_module)),
        "machine_module_path": f"{case_id}/{mm_path.name}",
        "machine_module_supported": machine_module.get("supported"),
        "machine_module_unsupported_detail": machine_module.get("unsupported_detail"),
        "machine_module_legacy_fallback": machine_module["legacy_fallback"],
        "elf_sha256": sha256_bytes(elf_bytes),
        "native_stdout_sha256": sha256_bytes(native_stdout),
        "native_stderr_sha256": sha256_bytes(native_stderr),
        "lean_single_stdout_sha256": sha256_bytes(lean_stdout),
        "lean_single_stderr_sha256": sha256_bytes(lean_stderr),
        "native_stdout": native_stdout.decode("utf-8", errors="replace"),
        "lean_single_stdout": lean_stdout.decode("utf-8", errors="replace"),
    }
    if f128_contract:
        row["f128_value_contract_evidence"] = f128_contract
    for key in [
        "expected_stdout",
        "wide_type",
        "wide_ops",
        "path_kind",
        "field_count",
        "declared_layout_bytes",
        "callee",
        "f128_boundary_kind",
    ]:
        if key in case:
            row[key] = case[key]
    row["differential_case_sha256"] = sha256_text(stable_json(row))
    return row


def emit(args: argparse.Namespace) -> int:
    root = repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    reference_souc = Path(args.reference_souc).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = case_source_rows(root)
    results = [emit_case(root, compiler, reference_souc, out_dir, case, args.timeout) for case in cases]
    matched = [row for row in results if row["status"] == "matched"]
    unavailable = [row for row in results if row["status"] == "reference_unavailable"]
    if len(results) != EXPECTED_CASE_COUNT:
        raise SystemExit(f"differential receipt expected {EXPECTED_CASE_COUNT} cases, got {len(results)}")
    if len(matched) != EXPECTED_MATCHED_CASE_COUNT:
        raise SystemExit(
            f"differential receipt expected {EXPECTED_MATCHED_CASE_COUNT} matched cases, got {len(matched)}"
        )
    if {row["case_id"] for row in unavailable} != set(UNAVAILABLE_REFERENCE_CASES):
        raise SystemExit(f"unexpected unavailable differential cases: {[row['case_id'] for row in unavailable]}")
    matched_categories = {str(row["category"]) for row in matched}
    missing_categories = REQUIRED_CATEGORIES - matched_categories
    if missing_categories:
        raise SystemExit(f"differential matched cases missing categories: {sorted(missing_categories)}")

    payload = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "native_v2_vs_lean_single_promoted_surfaces",
        "target": "x86_64-linux",
        "native_compiler": str(compiler),
        "reference_souc": str(reference_souc),
        "reference_engine": "lean_single",
        "case_count": len(results),
        "matched_case_count": len(matched),
        "reference_unavailable_case_count": len(unavailable),
        "categories_compared": sorted(matched_categories),
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "cases": results,
        "native_v2_vs_lean_single_differential_complete": True,
        "s5_differential_native_v2_lean_single_complete": True,
        "differential_native_v2_vs_lean_single_promoted": True,
        "all_reference_available_cases_match_exit_and_stdout": True,
        "all_native_v2_cases_compile_without_legacy_fallback": True,
        "all_native_v2_cases_return_expected_exit": True,
        "all_reference_available_lean_single_cases_return_expected_exit": True,
        "known_reference_unavailable_cases_recorded": True,
        "f128_promoted": False,
        "f128_promoted_surface_differentials_complete": True,
        "f128_arithmetic_value_contract_differential_complete": True,
        "f128_opaque_call_return_abi_differential_complete": True,
        "f128_sret_internal_arg_boundary_differential_complete": True,
        "f128_param_slot_layout_differential_complete": True,
        "f128_binary128_native_materialization_differential_complete": True,
        "f128_ieee_class_helper_differential_recorded": True,
        "f128_ieee_predicate_helper_differential_recorded": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "missing_full_obligations": [
            "generic IEEE f128 software-helper semantics and differentials",
            "external SysV f128 ABI/SRET differentials",
            "arbitrary decimal f128 materialization beyond the finite value-contract case set",
        ],
    }
    canonical, payload_sha = canonical_roundtrip(payload)
    payload["receipt_sha256"] = payload_sha
    canonical_with_hash, _ = canonical_roundtrip(payload)
    payload["canonical_roundtrip_sha256"] = sha256_text(canonical_with_hash)
    receipt_path = out_dir / "madaros_v2_s5_differential.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    print(
        "madaros-v2-s5-differential: "
        f"matched={len(matched)}/{len(results)} unavailable={len(unavailable)} "
        f"sha={payload['receipt_sha256'][:12]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_parser = sub.add_parser("emit")
    emit_parser.add_argument("--compiler", required=True)
    emit_parser.add_argument("--reference-souc", required=True)
    emit_parser.add_argument("--out-dir", required=True)
    emit_parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.cmd == "emit":
        return emit(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
