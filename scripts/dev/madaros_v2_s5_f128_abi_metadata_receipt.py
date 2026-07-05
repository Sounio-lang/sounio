#!/usr/bin/env python3
"""Emit a Madaros v2 S5 f128 ABI metadata receipt.

This promotes the binary128 ABI signature metadata slice only: local/imported
f128 parameters and returns must be classified in the MachineModule export, and
native-v2 must still fail closed without emitting an executable until f128
software helpers and execution differentials are promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.f128_abi_metadata_receipt/0.2"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_METADATA_SCHEMA = "madaros.v2.s5.machine_module_slot_metadata/0.1"
STAGE_CONTRACT_LEVEL = "S5_1_F128_ABI_METADATA_PROMOTED_WITH_SPECIFIC_BLOCKERS_NOT_NATIVE_EXECUTION"

F128_SLOT_KIND = 3
F128_WIDTH_WORDS = 2
F128_UNSUPPORTED_DETAILS = {"f128_call_arg_pending", "f128_return_pending"}
F128_SYSV_CLASSES = "SSE,SSEUP"


CASES: list[dict[str, Any]] = [
    {
        "case_id": "local_f128_arg_return_metadata",
        "class": "local_f128_one_arg_return",
        "source": """fn id_f128(x: f128) -> f128 { x }
fn main() -> i64 {
  let x: f128 = id_f128(1.0 as f128)
  0
}
""",
        "callee": "id_f128",
        "expected_fn_count": 2,
        "expected_callee_f128_param_count": 1,
        "expected_callee_returns_f128": True,
        "imported_module": False,
        "proves": [
            "local_f128_param_slot_kind_3_width_2",
            "local_f128_return_signature_metadata",
            "caller_f128_call_result_slot_metadata",
        ],
    },
    {
        "case_id": "local_f128_return_only_metadata",
        "class": "local_f128_return_only",
        "source": """fn ret_f128() -> f128 { 1.0 as f128 }
fn main() -> i64 {
  let x: f128 = ret_f128()
  0
}
""",
        "callee": "ret_f128",
        "expected_fn_count": 2,
        "expected_callee_f128_param_count": 0,
        "expected_callee_returns_f128": True,
        "imported_module": False,
        "proves": [
            "local_f128_return_signature_metadata_without_args",
            "caller_f128_call_result_slot_metadata",
        ],
    },
    {
        "case_id": "imported_f128_arg_return_metadata",
        "class": "imported_f128_one_arg_return",
        "support_files": {
            "f128_lib.sio": """pub fn id_f128_imported(x: f128) -> f128 { x }
""",
        },
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let x: f128 = id_f128_imported(1.0 as f128)
  0
}
""",
        "callee": "id_f128_imported",
        "expected_fn_count": 3,
        "expected_callee_f128_param_count": 1,
        "expected_callee_returns_f128": True,
        "imported_module": True,
        "proves": [
            "imported_f128_param_slot_kind_3_width_2",
            "imported_f128_return_signature_metadata",
            "imported_caller_f128_call_result_slot_metadata",
        ],
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


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("f128 ABI metadata receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(fn.get("name", "")): fn for fn in module.get("functions", [])}


def slot_metadata_functions(module: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sm = module.get("slot_metadata")
    if not isinstance(sm, dict):
        raise SystemExit("MachineModule slot_metadata missing")
    if sm.get("schema") != SLOT_METADATA_SCHEMA:
        raise SystemExit(f"bad slot metadata schema: {sm.get('schema')!r}")
    if sm.get("f128_execution_promoted") is not False:
        raise SystemExit("f128 ABI metadata receipt must not promote f128 execution")
    return {int(fn.get("fn_index", -1)): fn for fn in sm.get("functions", [])}


def f128_slot_rows_for_fn(module: dict[str, Any], fn_index: int) -> list[list[int]]:
    sm_fn = slot_metadata_functions(module).get(fn_index)
    if not sm_fn:
        return []
    rows = sm_fn.get("slots", [])
    out: list[list[int]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise SystemExit(f"bad slot metadata row for fn_index={fn_index}: {row!r}")
        slot, kind, width = [int(v) for v in row]
        if kind == F128_SLOT_KIND:
            if width != F128_WIDTH_WORDS:
                raise SystemExit(f"f128 slot must have width_words=2, got row={row!r}")
            out.append([slot, kind, width])
    return out


def load_machine_module(path: Path, expected_fn_count: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit("MachineModule source mismatch")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit("MachineModule export flag missing")
    if payload.get("target") != "x86_64-linux":
        raise SystemExit(f"unexpected MachineModule target: {payload.get('target')!r}")
    if payload.get("supported") is not False:
        raise SystemExit("f128 ABI metadata cases must remain unsupported")
    if payload.get("unsupported_detail") not in F128_UNSUPPORTED_DETAILS:
        raise SystemExit(f"unexpected unsupported_detail: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    if int(payload.get("fn_count", -1)) != expected_fn_count:
        raise SystemExit(f"expected fn_count={expected_fn_count}, got {payload.get('fn_count')!r}")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def machine_shape(module: dict[str, Any], callee: str, case: dict[str, Any]) -> dict[str, Any]:
    fns = functions_by_name(module)
    if "main" not in fns:
        raise SystemExit("MachineModule must contain main")
    if callee not in fns:
        raise SystemExit(f"MachineModule must contain callee {callee}")
    main_fn = fns["main"]
    callee_fn = fns[callee]
    main_index = int(main_fn.get("index", -1))
    callee_index = int(callee_fn.get("index", -1))
    main_f128_rows = f128_slot_rows_for_fn(module, main_index)
    callee_f128_rows = f128_slot_rows_for_fn(module, callee_index)
    callee_param_count = int(callee_fn.get("source_f128_param_count", -1))
    callee_returns = bool(callee_fn.get("source_returns_f128", False))
    if callee_param_count != int(case["expected_callee_f128_param_count"]):
        raise SystemExit(
            f"{case['case_id']} expected callee source_f128_param_count="
            f"{case['expected_callee_f128_param_count']}, got {callee_param_count}"
        )
    if callee_returns is not bool(case["expected_callee_returns_f128"]):
        raise SystemExit(f"{case['case_id']} callee source_returns_f128 mismatch")
    if callee_fn.get("source_abi_metadata_schema") != "madaros.v2.s5.function_abi_metadata/0.1":
        raise SystemExit(f"{case['case_id']} missing function ABI metadata schema")
    if callee_fn.get("source_return_slot_kind") != F128_SLOT_KIND:
        raise SystemExit(f"{case['case_id']} callee return slot kind must be f128 kind 3")
    if callee_fn.get("source_return_width_words") != F128_WIDTH_WORDS:
        raise SystemExit(f"{case['case_id']} callee return width must be 2 words")
    if callee_fn.get("source_f128_sysv_classes") != F128_SYSV_CLASSES:
        raise SystemExit(f"{case['case_id']} callee f128 SysV classes mismatch")
    if callee_fn.get("source_f128_execution_pending") is not True:
        raise SystemExit(f"{case['case_id']} callee must keep f128 execution pending")
    if int(case["expected_callee_f128_param_count"]) > 0 and not callee_f128_rows:
        raise SystemExit(f"{case['case_id']} callee has no f128 parameter slot rows")
    if not main_f128_rows:
        raise SystemExit(f"{case['case_id']} main has no f128 call/local result slot rows")
    return {
        "function_names": [str(fn.get("name", "")) for fn in module.get("functions", [])],
        "callee_source_param_count": int(callee_fn.get("source_param_count", -1)),
        "callee_source_f128_param_count": callee_param_count,
        "callee_source_returns_f128": callee_returns,
        "callee_source_return_slot_kind": int(callee_fn.get("source_return_slot_kind", -1)),
        "callee_source_return_width_words": int(callee_fn.get("source_return_width_words", -1)),
        "callee_source_f128_sysv_classes": str(callee_fn.get("source_f128_sysv_classes", "")),
        "callee_f128_slot_row_count": len(callee_f128_rows),
        "main_f128_slot_row_count": len(main_f128_rows),
        "f128_slot_kind_seen": sorted({row[1] for row in callee_f128_rows + main_f128_rows}),
        "f128_slot_width_words_seen": sorted({row[2] for row in callee_f128_rows + main_f128_rows}),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    check_log_path = out_dir / f"{case_id}.check.log"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
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
    if not mm_path.exists():
        raise SystemExit(f"{case_id} did not emit MachineModule JSON")
    if "native_v2_compile:" not in compile_log or "FAIL" not in compile_log:
        raise SystemExit(f"{case_id} must fail closed in native-v2 compile; log={compile_log_path}")
    if elf_path.exists():
        raise SystemExit(f"{case_id} unexpectedly emitted an executable before f128 execution promotion")

    module = load_machine_module(mm_path, int(case["expected_fn_count"]))
    shape = machine_shape(module, str(case["callee"]), case)
    return {
        "case_id": case_id,
        "class": case["class"],
        "source": source_path.name,
        "imported_module": bool(case.get("imported_module", False)),
        "callee": case["callee"],
        "source_sha256": sha256_text(source_text),
        "check_rc": check_rc,
        "compile_rc": compile_rc,
        "check_log_sha256": sha256_text(normalize_log(check_log, out_dir)),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_supported": bool(module.get("supported", True)),
        "machine_unsupported_detail": str(module.get("unsupported_detail", "")),
        "elf_emitted": elf_path.exists(),
        "machine_shape": shape,
        "proves": list(case["proves"]),
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_f128_abi_metadata.receipt.json"

    cases = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    imported_cases = [case for case in cases if case["imported_module"]]
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "f128_local_and_imported_abi_metadata",
        "case_count": len(cases),
        "imported_module_case_count": len(imported_cases),
        "cases": cases,
        "compiler_machine_module_exported": True,
        "f128_binary128_slot_kind": F128_SLOT_KIND,
        "f128_binary128_width_words": F128_WIDTH_WORDS,
        "f128_sysv_classes": F128_SYSV_CLASSES,
        "f128_local_param_metadata_promoted": True,
        "f128_local_return_metadata_promoted": True,
        "f128_imported_param_metadata_promoted": True,
        "f128_imported_return_metadata_promoted": True,
        "f128_call_result_slot_metadata_promoted": True,
        "f128_abi_metadata_promoted": True,
        "f128_execution_promoted": False,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "legacy_fallback_for_f128_abi": False,
        "roundtrip_contract": [
            "local_f128_parameter_is_marked_as_slot_kind_3_width_words_2",
            "local_f128_return_signature_exports_kind_3_width_words_2",
            "local_f128_call_result_is_marked_in_caller_slot_metadata",
            "imported_f128_parameter_is_marked_as_slot_kind_3_width_words_2",
            "imported_f128_return_signature_exports_kind_3_width_words_2",
            "MachineModule_fails_closed_with_specific_f128_call_or_return_blocker",
            "no_f128_native_executable_is_emitted_before_helper_and_differential_receipts",
        ],
        "missing_full_obligations": [
            "f128 software-helper lowering with IEEE rounding and NaN/Inf contract",
            "f128 native-v2 execution and differential receipts",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        "madaros-v2-s5-f128-abi-metadata: "
        f"cases={receipt['case_count']} imported={receipt['imported_module_case_count']} "
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
    args = parser.parse_args()
    if args.cmd == "emit":
        return emit(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
