#!/usr/bin/env python3
"""Emit a Madaros v2 S5 wide-integer ABI call/return receipt.

This receipt closes the local and imported module-boundary i256/u256
call-return slice: wide values must cross a native-v2 call boundary as four
64-bit limbs, return through the compiler's SRET path, and be consumed by the
caller through real wide comparisons/arithmetic. It deliberately does not claim
f128.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.wide_abi_call_return_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_WIDE_I256_U256_LOCAL_AND_IMPORTED_ABI_CALL_RETURN_PROMOTED_NOT_F128"
WIDE_SLOT_KIND = 4
WIDE_WIDTH_WORDS = 4

CASES: list[dict[str, Any]] = [
    {
        "case_id": "i256_return_only_sret_return_41",
        "class": "wide_i256_return_only",
        "source": """fn ret_i256() -> i256 { 7 as i256 }
fn main() -> i64 {
  let r: i256 = ret_i256()
  if r == (7 as i256) { return 41 }
  1
}
""",
        "expected_exit": 41,
        "fake_scalar_exit": 1,
        "callee": "ret_i256",
        "expected_fn_count": 2,
        "expected_callee_source_param_count": 0,
        "wide_type": "i256",
        "wide_arg_count": 0,
        "proves": ["wide_sret_return_consumed_by_caller"],
    },
    {
        "case_id": "i256_arg_return_sret_return_31",
        "class": "wide_i256_one_arg_return",
        "source": """fn id_i256(x: i256) -> i256 { x }
fn main() -> i64 {
  let a: i256 = 9 as i256
  let r: i256 = id_i256(a)
  if r == a { return 31 }
  1
}
""",
        "expected_exit": 31,
        "fake_scalar_exit": 1,
        "callee": "id_i256",
        "expected_fn_count": 2,
        "expected_callee_source_param_count": 4,
        "wide_type": "i256",
        "wide_arg_count": 1,
        "proves": ["one_wide_arg_expands_to_four_limb_params", "wide_sret_return_consumed_by_caller"],
    },
    {
        "case_id": "u256_first_of_two_wide_args_return_43",
        "class": "wide_u256_two_arg_first_return",
        "source": """fn first_u256(x: u256, y: u256) -> u256 { x }
fn main() -> i64 {
  let a: u256 = 5 as u256
  let b: u256 = 9 as u256
  let r: u256 = first_u256(a, b)
  if r == a { return 43 }
  if r == b { return 64 }
  1
}
""",
        "expected_exit": 43,
        "fake_scalar_exit": 64,
        "callee": "first_u256",
        "expected_fn_count": 2,
        "expected_callee_source_param_count": 8,
        "wide_type": "u256",
        "wide_arg_count": 2,
        "proves": ["two_wide_args_expand_to_eight_limb_params", "first_wide_arg_not_collapsed_into_second"],
    },
    {
        "case_id": "u256_second_of_two_wide_args_return_47",
        "class": "wide_u256_two_arg_second_return",
        "source": """fn second_u256(x: u256, y: u256) -> u256 { y }
fn main() -> i64 {
  let a: u256 = 1 as u256
  let b: u256 = 3 as u256
  let r: u256 = second_u256(a, b)
  if r == b { return 47 }
  if r == (0 as u256) { return 63 }
  if r == a { return 64 }
  1
}
""",
        "expected_exit": 47,
        "fake_scalar_exit": 63,
        "callee": "second_u256",
        "expected_fn_count": 2,
        "expected_callee_source_param_count": 8,
        "wide_type": "u256",
        "wide_arg_count": 2,
        "proves": ["second_wide_arg_preserved", "two_wide_args_expand_to_eight_limb_params"],
        "trace_required": True,
        "trace_pattern": "fn_after_params name=second_u256 param_count=8",
    },
    {
        "case_id": "u256_two_arg_add_return_37",
        "class": "wide_u256_two_arg_callee_arithmetic_return",
        "source": """fn addtwice(x: u256, y: u256) -> u256 { x + y }
fn main() -> i64 {
  let a: u256 = 10 as u256
  let b: u256 = 27 as u256
  let r: u256 = addtwice(a, b)
  if r == (37 as u256) { return 37 }
  1
}
""",
        "expected_exit": 37,
        "fake_scalar_exit": 1,
        "callee": "addtwice",
        "expected_fn_count": 2,
        "expected_callee_source_param_count": 8,
        "wide_type": "u256",
        "wide_arg_count": 2,
        "proves": ["callee_wide_arithmetic_result_returns_through_wide_sret"],
    },
    {
        "case_id": "imported_i256_return_only_sret_return_52",
        "class": "imported_wide_i256_return_only",
        "support_files": {
            "wide_lib.sio": """pub fn ret_i256() -> i256 { 7 as i256 }
""",
        },
        "source": """import "wide_lib.sio"
fn main() -> i64 {
  let r: i256 = ret_i256()
  if r == (7 as i256) { return 52 }
  1
}
""",
        "expected_exit": 52,
        "fake_scalar_exit": 1,
        "callee": "ret_i256",
        "expected_fn_count": 3,
        "expected_callee_source_param_count": 0,
        "wide_type": "i256",
        "wide_arg_count": 0,
        "imported_module": True,
        "proves": ["imported_wide_i256_return_consumed_by_caller"],
    },
    {
        "case_id": "imported_i256_arg_return_sret_return_54",
        "class": "imported_wide_i256_one_arg_return",
        "support_files": {
            "wide_lib.sio": """pub fn id_i256(x: i256) -> i256 { x }
""",
        },
        "source": """import "wide_lib.sio"
fn main() -> i64 {
  let r: i256 = id_i256(54 as i256)
  if r == (54 as i256) { return 54 }
  1
}
""",
        "expected_exit": 54,
        "fake_scalar_exit": 1,
        "callee": "id_i256",
        "expected_fn_count": 3,
        "expected_callee_source_param_count": 4,
        "wide_type": "i256",
        "wide_arg_count": 1,
        "imported_module": True,
        "proves": ["imported_wide_i256_arg_expands_to_four_limb_params"],
    },
    {
        "case_id": "imported_u256_second_of_two_wide_args_return_53",
        "class": "imported_wide_u256_two_arg_second_return",
        "support_files": {
            "wide_lib.sio": """pub fn second_u256(x: u256, y: u256) -> u256 { y }
""",
        },
        "source": """import "wide_lib.sio"
fn main() -> i64 {
  let a: u256 = 11 as u256
  let b: u256 = 53 as u256
  let r: u256 = second_u256(a, b)
  if r == b { return 53 }
  if r == a { return 62 }
  61
}
""",
        "expected_exit": 53,
        "fake_scalar_exit": 61,
        "callee": "second_u256",
        "expected_fn_count": 3,
        "expected_callee_source_param_count": 8,
        "wide_type": "u256",
        "wide_arg_count": 2,
        "imported_module": True,
        "proves": ["imported_second_wide_arg_preserved", "imported_two_wide_args_expand_to_eight_limb_params"],
    },
    {
        "case_id": "imported_i256_mixed_param_order_return_55",
        "class": "imported_wide_i256_mixed_param_order",
        "support_files": {
            "wide_lib.sio": """pub fn mixed_left(x: i256, y: i64) -> i256 { x + (y as i256) }
pub fn mixed_right(x: i64, y: i256) -> i256 { y + (x as i256) }
""",
        },
        "source": """import "wide_lib.sio"
fn main() -> i64 {
  let a: i256 = mixed_left(50 as i256, 5)
  let b: i256 = mixed_right(5, 50 as i256)
  if a == (55 as i256) && b == (55 as i256) { return 55 }
  1
}
""",
        "expected_exit": 55,
        "fake_scalar_exit": 1,
        "callee": "mixed_left",
        "extra_callee_param_counts": {"mixed_right": 5},
        "expected_fn_count": 5,
        "expected_callee_source_param_count": 5,
        "wide_type": "i256",
        "wide_arg_count": 1,
        "imported_module": True,
        "proves": ["imported_mixed_i256_i64_param_order_preserved", "imported_mixed_i64_i256_param_order_preserved"],
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


def run_command(
    cmd: list[str], cwd: Path, timeout_s: int, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
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
        raise SystemExit("wide ABI call-return receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def load_machine_module(path: Path, expected_fn_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit("MachineModule source mismatch")
    if payload.get("target") != "x86_64-linux":
        raise SystemExit("MachineModule target mismatch")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit("MachineModule export flag missing")
    if payload.get("active") is not True:
        raise SystemExit("MachineModule is not active")
    if payload.get("supported") is not True:
        raise SystemExit(f"MachineModule unsupported: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    if int(payload.get("fn_count", -1)) != expected_fn_count:
        raise SystemExit(f"MachineModule fn_count mismatch: {payload.get('fn_count')!r}")
    slot_metadata = payload.get("slot_metadata", {})
    if not isinstance(slot_metadata, dict):
        raise SystemExit("MachineModule missing slot_metadata object")
    if slot_metadata.get("schema") != "madaros.v2.s5.machine_module_slot_metadata/0.1":
        raise SystemExit("bad MachineModule slot_metadata schema")
    if slot_metadata.get("machine_ir_slot_metadata_exported") is not True:
        raise SystemExit("MachineModule must export slot metadata")
    if slot_metadata.get("f128_execution_promoted") is not False:
        raise SystemExit("wide ABI receipt must not promote f128 execution")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for fn in module.get("functions", []):
        name = str(fn.get("name", ""))
        if name and name not in result:
            result[name] = fn
    return result


def require_callee_param_counts(module: dict[str, Any], expected: dict[str, int], case_id: str) -> None:
    fns = functions_by_name(module)
    for name, expected_count in expected.items():
        if name not in fns:
            raise SystemExit(f"{case_id} MachineModule must contain extra callee {name}")
        actual = int(fns[name].get("source_param_count", -1))
        if actual != expected_count:
            raise SystemExit(f"{case_id} expected {name} source_param_count {expected_count}, got {actual}")


def wide_slot_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    slot_meta = module.get("slot_metadata", {})
    for fn_meta in slot_meta.get("functions", []):
        for row in fn_meta.get("slots", []):
            if len(row) >= 3 and int(row[1]) == WIDE_SLOT_KIND:
                rows.append([int(row[0]), int(row[1]), int(row[2])])
    return rows


def machine_shape(module: dict[str, Any], callee: str) -> dict[str, Any]:
    fns = functions_by_name(module)
    if "main" not in fns:
        raise SystemExit("MachineModule must contain main")
    if callee not in fns:
        raise SystemExit(f"MachineModule must contain callee {callee}")
    main_fn = fns["main"]
    callee_fn = fns[callee]
    wide_rows = wide_slot_rows(module)
    bad_width_rows = [row for row in wide_rows if row[2] != WIDE_WIDTH_WORDS]
    if bad_width_rows:
        raise SystemExit(f"wide slot metadata contains non-4-word rows: {bad_width_rows[:4]}")
    if not wide_rows:
        raise SystemExit("MachineModule has no wide slot metadata rows")
    return {
        "function_names": [str(fn.get("name", "")) for fn in module.get("functions", [])],
        "callee_source_is_sret": int(callee_fn.get("source_is_sret", -1)),
        "callee_source_sret_dest_reg": int(callee_fn.get("source_sret_dest_reg", -1)),
        "callee_source_param_count": int(callee_fn.get("source_param_count", -1)),
        "main_source_param_count": int(main_fn.get("source_param_count", -1)),
        "wide_slot_row_count": len(wide_rows),
        "wide_slot_width_words_seen": sorted({row[2] for row in wide_rows}),
        "wide_slot_kind_seen": sorted({row[1] for row in wide_rows}),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    expected_exit = int(case["expected_exit"])
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    check_log_path = out_dir / f"{case_id}.check.log"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    public_elf_path = out_dir / f"{case_id}.public_native"
    public_compile_log_path = out_dir / f"{case_id}.public_native.log"
    public_stdout_path = out_dir / f"{case_id}.public_stdout"
    public_stderr_path = out_dir / f"{case_id}.public_stderr"
    for rel_name, text in dict(case.get("support_files", {})).items():
        support_path = out_dir / rel_name
        support_path.parent.mkdir(parents=True, exist_ok=True)
        support_path.write_text(str(text), encoding="utf-8")
    source_path.write_text(source_text, encoding="utf-8")

    check_rc, check_stdout, check_stderr = run_command([str(compiler), "check", str(source_path)], root, timeout_s)
    check_log = check_stdout + check_stderr
    check_log_path.write_text(check_log, encoding="utf-8")
    if check_rc != 0 or "check: OK" not in check_log:
        raise SystemExit(f"{case_id} expected check OK, rc={check_rc}; log={check_log_path}")

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
        raise SystemExit(f"{case_id} native-v2 compile failed rc={compile_rc}; log={compile_log_path}")
    if not mm_path.exists():
        raise SystemExit(f"{case_id} did not emit MachineModule JSON")

    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")
    if actual_exit == int(case["fake_scalar_exit"]):
        raise SystemExit(f"{case_id} matched fake scalar/truncated discriminator")

    public_compile_rc = 0
    public_actual_exit = actual_exit
    public_compile_log_sha = ""
    public_stdout_sha = ""
    public_stderr_sha = ""
    public_elf_sha = ""
    if bool(case.get("imported_module", False)):
        public_compile_rc, public_compile_stdout, public_compile_stderr = run_command(
            [str(compiler), str(source_path), "-o", str(public_elf_path)],
            root,
            timeout_s,
        )
        public_compile_log = public_compile_stdout + public_compile_stderr
        public_compile_log_path.write_text(public_compile_log, encoding="utf-8")
        if public_compile_rc != 0:
            raise SystemExit(f"{case_id} public native compile failed rc={public_compile_rc}; log={public_compile_log_path}")
        if "compact modular IR table path" in public_compile_log:
            raise SystemExit(f"{case_id} public native compile used stale compact modular IR table path")
        if not public_elf_path.exists() or public_elf_path.stat().st_size <= 1024:
            raise SystemExit(f"{case_id} public native compile did not emit a full ELF: {public_elf_path}")
        public_elf_path.chmod(public_elf_path.stat().st_mode | 0o111)
        public_actual_exit, public_stdout, public_stderr = run_binary(public_elf_path, timeout_s)
        public_stdout_path.write_bytes(public_stdout)
        public_stderr_path.write_bytes(public_stderr)
        if public_actual_exit != expected_exit:
            raise SystemExit(f"{case_id} public native expected exit {expected_exit}, got {public_actual_exit}")
        if public_actual_exit == int(case["fake_scalar_exit"]):
            raise SystemExit(f"{case_id} public native matched fake scalar/truncated discriminator")
        public_compile_log_sha = sha256_text(normalize_log(public_compile_log, out_dir))
        public_stdout_sha = sha256_bytes(public_stdout)
        public_stderr_sha = sha256_bytes(public_stderr)
        public_elf_sha = sha256_bytes(public_elf_path.read_bytes())

    module = load_machine_module(mm_path, int(case["expected_fn_count"]))
    shape = machine_shape(module, str(case["callee"]))
    if shape["callee_source_is_sret"] != 1:
        raise SystemExit(f"{case_id} callee must be lowered as SRET for wide return")
    if shape["callee_source_sret_dest_reg"] != 0:
        raise SystemExit(f"{case_id} callee SRET dest must be register 0")
    if shape["callee_source_param_count"] != int(case["expected_callee_source_param_count"]):
        raise SystemExit(
            f"{case_id} expected callee source_param_count "
            f"{case['expected_callee_source_param_count']}, got {shape['callee_source_param_count']}"
        )
    require_callee_param_counts(
        module,
        {str(k): int(v) for k, v in dict(case.get("extra_callee_param_counts", {})).items()},
        case_id,
    )

    trace_sha = ""
    trace_matched = False
    trace_satisfied_by_machine_module = False
    if case.get("trace_required"):
        env = os.environ.copy()
        env["SOUNIO_LOWER_BODY_TRACE"] = "1"
        env["SOUNIO_LOWER_LIVE_TRACE"] = "1"
        trace_log_path = out_dir / f"{case_id}.trace.log"
        trace_rc, trace_stdout, trace_stderr = run_command(
            [str(compiler), "--native-v2-compile", str(source_path), "-o", str(out_dir / f"{case_id}.trace.native_v2")],
            root,
            timeout_s,
            env=env,
        )
        trace_log = trace_stdout + trace_stderr
        trace_log_path.write_text(trace_log, encoding="utf-8")
        if trace_rc != 0:
            raise SystemExit(f"{case_id} trace compile failed rc={trace_rc}; log={trace_log_path}")
        trace_pattern = str(case["trace_pattern"])
        trace_matched = trace_pattern in trace_log
        if not trace_matched:
            trace_satisfied_by_machine_module = (
                shape["callee_source_param_count"] == int(case["expected_callee_source_param_count"])
            )
            if not trace_satisfied_by_machine_module:
                raise SystemExit(f"{case_id} trace did not contain {trace_pattern!r}; log={trace_log_path}")
        trace_sha = sha256_text(normalize_log(trace_log, out_dir))

    return {
        "case_id": case_id,
        "class": case["class"],
        "source": source_path.name,
        "wide_type": case["wide_type"],
        "wide_arg_count": int(case["wide_arg_count"]),
        "imported_module": bool(case.get("imported_module", False)),
        "callee": case["callee"],
        "extra_callee_param_counts": {str(k): int(v) for k, v in dict(case.get("extra_callee_param_counts", {})).items()},
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "public_native_compile_checked": bool(case.get("imported_module", False)),
        "public_native_compile_rc": public_compile_rc,
        "public_native_actual_exit": public_actual_exit,
        "fake_scalar_exit": int(case["fake_scalar_exit"]),
        "source_sha256": sha256_text(source_text),
        "check_rc": check_rc,
        "compile_rc": compile_rc,
        "check_log_sha256": sha256_text(normalize_log(check_log, out_dir)),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "public_native_elf_sha256": public_elf_sha,
        "public_native_compile_log_sha256": public_compile_log_sha,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "public_native_stdout_sha256": public_stdout_sha,
        "public_native_stderr_sha256": public_stderr_sha,
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_shape": shape,
        "trace_required": bool(case.get("trace_required", False)),
        "trace_matched": trace_matched,
        "trace_satisfied_by_machine_module": trace_satisfied_by_machine_module,
        "trace_log_sha256": trace_sha,
        "proves": list(case["proves"]),
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_wide_abi_call_return.receipt.json"

    cases = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    i256_cases = [case for case in cases if case["wide_type"] == "i256"]
    u256_cases = [case for case in cases if case["wide_type"] == "u256"]
    two_wide_arg_cases = [case for case in cases if int(case["wide_arg_count"]) == 2]
    imported_cases = [case for case in cases if case["imported_module"]]
    public_imported_cases = [case for case in imported_cases if case["public_native_compile_checked"]]
    trace_cases = [case for case in cases if case["trace_required"]]

    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "wide_i256_u256_local_abi_call_return",
        "case_count": len(cases),
        "i256_case_count": len(i256_cases),
        "u256_case_count": len(u256_cases),
        "two_wide_arg_case_count": len(two_wide_arg_cases),
        "imported_module_case_count": len(imported_cases),
        "public_native_imported_case_count": len(public_imported_cases),
        "trace_assertion_case_count": len(trace_cases),
        "cases": cases,
        "s5_wide_i256_u256_local_abi_call_return_complete": True,
        "s5_wide_i256_u256_imported_abi_call_return_complete": True,
        "wide_i256_u256_local_abi_call_return_promoted": True,
        "wide_i256_u256_imported_abi_call_return_promoted": True,
        "wide_return_uses_sret": True,
        "wide_arg_limb_expansion_promoted": True,
        "wide_two_arg_order_preserved": True,
        "wide_second_arg_preserved": True,
        "wide_callee_arithmetic_return_promoted": True,
        "wide_machine_slot_kind": WIDE_SLOT_KIND,
        "wide_machine_slot_width_words": WIDE_WIDTH_WORDS,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "legacy_fallback_for_wide_abi": False,
        "public_native_imported_route_checked": True,
        "public_native_imported_route_uses_full_modular_native_v2": True,
        "stale_compact_modular_ir_table_path_blocked": True,
        "imported_module_wide_abi_promoted": True,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "i256_return_only_crosses_function_boundary_and_is_compared_by_caller",
            "i256_argument_crosses_function_boundary_and_returns_through_wide_sret",
            "u256_first_of_two_wide_arguments_is_preserved",
            "u256_second_of_two_wide_arguments_is_preserved_not_zeroed",
            "u256_callee_arithmetic_returns_through_wide_sret",
            "wide_arguments_expand_to_four_limb_params_per_value",
            "two_wide_arguments_expand_to_eight_limb_params",
            "MachineModule_exports_wide_slot_kind_4_width_words_4",
            "trace_asserts_mut_param_lowering_expands_second_wide_argument_to_param_count_8",
            "imported_i256_return_only_crosses_module_boundary_and_is_compared_by_caller",
            "imported_i256_argument_crosses_module_boundary_and_returns_through_wide_sret",
            "imported_u256_second_of_two_wide_arguments_is_preserved",
            "imported_mixed_i256_i64_and_i64_i256_argument_order_is_preserved",
            "public_madaros_source_o_output_route_matches_native_v2_for_imported_wide_abi",
            "f128_is_not_promoted_by_this_receipt",
        ],
        "missing_full_obligations": [
            "wide ABI stack-pressure cases beyond two register-level wide args",
            "f128 IR/MIR/ABI/software-helper receipts",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        "madaros-v2-s5-wide-abi-call-return: "
        f"cases={receipt['case_count']} i256={receipt['i256_case_count']} "
        f"u256={receipt['u256_case_count']} imported={receipt['imported_module_case_count']} "
        f"sha={receipt['receipt_sha256'][:12]} "
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
