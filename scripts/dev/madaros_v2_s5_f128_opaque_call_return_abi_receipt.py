#!/usr/bin/env python3
"""Emit the S5.5 f128 opaque direct call/return ABI receipt.

This promotes an internal native-v2 contract only: f128 moves across direct
Madaros calls as two opaque binary128 words in the integer-register ABI,
including mixed i64/f128 orders and two-f128-arg shapes when the expanded
argument word count fits the direct-call register+stack window. It does not
promote external SysV ABI, SRET, IEEE helpers, NaN/Inf handling, or full f128
execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_opaque_call_return_abi_receipt/0.1"
STAGE = "S5_5_F128_OPAQUE_DIRECT_CALL_RETURN_ABI_PROMOTED"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
MIR_OP_CAPTURE_RET = 114
MIR_OP_RET = 115
INSTR_OPCODE = 0
INSTR_SRC2_KIND = 5
INSTR_COND = 9


POSITIVE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "local_f128_identity_arg_return",
        "source": """fn id_f128(x: f128) -> f128 { x }
fn main() -> i64 {
  let x: f128 = id_f128(1.0 as f128)
  let y: f128 = x
  0
}
""",
        "callee": "id_f128",
    },
    {
        "case_id": "local_f128_return_only",
        "source": """fn ret_f128() -> f128 { 1.0 as f128 }
fn main() -> i64 {
  let x: f128 = ret_f128()
  let y: f128 = x
  0
}
""",
        "callee": "ret_f128",
    },
    {
        "case_id": "local_f128_arg_i64_return",
        "source": """fn sink_f128(x: f128) -> i64 { 7 }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  sink_f128(x)
}
""",
        "callee": "sink_f128",
        "expected_exit": 7,
    },
    {
        "case_id": "imported_f128_identity_arg_return",
        "support_files": {"f128_lib.sio": "pub fn id_f128_imported(x: f128) -> f128 { x }\n"},
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let x: f128 = id_f128_imported(1.0 as f128)
  let y: f128 = x
  0
}
""",
        "callee": "id_f128_imported",
    },
    {
        "case_id": "imported_f128_return_only",
        "support_files": {"f128_lib.sio": "pub fn ret_f128_imported() -> f128 { 1.0 as f128 }\n"},
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let x: f128 = ret_f128_imported()
  let y: f128 = x
  0
}
""",
        "callee": "ret_f128_imported",
    },
    {
        "case_id": "imported_f128_arg_i64_return",
        "support_files": {"f128_lib.sio": "pub fn sink_f128_imported(x: f128) -> i64 { 11 }\n"},
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  sink_f128_imported(x)
}
""",
        "callee": "sink_f128_imported",
        "expected_exit": 11,
    },
    {
        "case_id": "imported_f128_plus_i64_arg_return",
        "support_files": {"f128_lib.sio": "pub fn pick_f128_imported(x: f128, y: i64) -> f128 { x }\n"},
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let z: f128 = pick_f128_imported(x, 3)
  0
}
""",
        "callee": "pick_f128_imported",
    },
    {
        "case_id": "imported_two_f128_args_return",
        "support_files": {"f128_lib.sio": "pub fn two_f128_imported(a: f128, b: f128) -> i64 { 12 }\n"},
        "source": """import "f128_lib.sio"
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 0.5 as f128
  two_f128_imported(a, b)
}
""",
        "callee": "two_f128_imported",
        "expected_exit": 12,
    },
    {
        "case_id": "local_f128_plus_i64_arg_return",
        "source": """fn mix(x: f128, y: i64) -> i64 { y }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  mix(x, 3)
}
""",
        "callee": "mix",
        "expected_exit": 3,
    },
    {
        "case_id": "local_i64_plus_f128_arg_return",
        "source": """fn mix(y: i64, x: f128) -> i64 { y }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  mix(4, x)
}
""",
        "callee": "mix",
        "expected_exit": 4,
    },
    {
        "case_id": "local_two_f128_args_return",
        "source": """fn first(x: f128, y: f128) -> i64 { 5 }
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 0.5 as f128
  first(a, b)
}
""",
        "callee": "first",
        "expected_exit": 5,
    },
    {
        "case_id": "local_mixed_arg_f128_return",
        "source": """fn pick(x: f128, y: i64) -> f128 { x }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let z: f128 = pick(x, 3)
  0
}
""",
        "callee": "pick",
    },
    {
        "case_id": "local_four_f128_args_stack_return",
        "source": """fn too_many(a: f128, b: f128, c: f128, d: f128) -> i64 { 9 }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  too_many(x, x, x, x)
}
""",
        "callee": "too_many",
        "expected_exit": 9,
    },
    {
        "case_id": "local_five_f128_args_deeper_stack_return",
        "source": """fn five(a: f128, b: f128, c: f128, d: f128, e: f128) -> i64 { 10 }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  five(x, x, x, x, x)
}
""",
        "callee": "five",
        "expected_exit": 10,
    },
]

NEGATIVE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "f128_rounded_decimal_arithmetic_still_blocked",
        "source": """fn main() -> i64 {
  let x: f128 = 0.1 as f128
  let y: f128 = 0.2 as f128
  let z: f128 = x + y
  0
}
""",
        "expected_detail": "f128_arithmetic_pending",
    },
    {
        "case_id": "f128_nine_arg_arity_still_blocked",
        "source": """fn too_many(a: f128, b: f128, c: f128, d: f128, e: f128, f: f128, g: f128, h: f128, i: f128) -> i64 { 9 }
fn main() -> i64 {
  let x: f128 = 1.0 as f128
  too_many(x, x, x, x, x, x, x, x, x)
}
""",
        "expected_detail": "call_arity_gt_8",
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
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout or ""


def write_case(case_dir: Path, case: dict[str, Any]) -> Path:
    for rel, text in case.get("support_files", {}).items():
        p = case_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text), encoding="utf-8")
    src = case_dir / f"{case['case_id']}.sio"
    src.write_text(str(case["source"]), encoding="utf-8")
    return src


def load_machine(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("legacy fallback must stay false")
    return payload


def fn_by_name(module: dict[str, Any], name: str) -> dict[str, Any]:
    for fn in module.get("functions", []):
        if fn.get("name") == name:
            return fn
    raise SystemExit(f"missing function in MachineModule: {name}")


def f128_slot_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    for fn in module.get("slot_metadata", {}).get("functions", []):
        for row in fn.get("slots", []):
            if isinstance(row, list) and len(row) == 3 and int(row[1]) == 3:
                if int(row[2]) != 2:
                    raise SystemExit(f"bad f128 slot width: {row!r}")
                rows.append([int(fn.get("fn_index", -1)), int(row[0]), int(row[1]), int(row[2])])
    if not rows:
        raise SystemExit("missing f128 slot metadata rows")
    return rows


def f128_literal_rows_for_fn(module: dict[str, Any], fn_index: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for fn in module.get("f128_literal_metadata", {}).get("functions", []):
        if int(fn.get("fn_index", -1)) == fn_index:
            for row in fn.get("rows", []):
                if isinstance(row, list) and len(row) >= 7:
                    rows.append([int(x) for x in row])
    return rows


def instr_rows(fn: dict[str, Any], opcode: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for raw in fn.get("instrs", []):
        if isinstance(raw, list) and len(raw) >= 12 and int(raw[INSTR_OPCODE]) == opcode:
            rows.append([int(v) for v in raw])
    return rows


def require_f128_return_word_flow(module: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    callee = fn_by_name(module, str(case["callee"]))
    main = fn_by_name(module, "main")
    callee_rets = instr_rows(callee, MIR_OP_RET)
    main_captures = instr_rows(main, MIR_OP_CAPTURE_RET)
    ret_words = sorted({row[INSTR_COND] for row in callee_rets if row[INSTR_COND] in (0, 1)})
    capture_words = sorted({row[INSTR_COND] for row in main_captures if row[INSTR_COND] in (0, 1)})
    callee_index = int(callee.get("index", -1))
    return_param_indices = module.get("f128_return_param_indices", [])
    callee_return_param_index = -1
    if 0 <= callee_index < len(return_param_indices):
        callee_return_param_index = int(return_param_indices[callee_index])
    if 1 not in ret_words:
        raise SystemExit(f"{case['case_id']}: f128 callee return must expose high word via RET cond=1")
    if [0, 1] != capture_words:
        if str(case["case_id"]).endswith("return_only"):
            main_index = int(main.get("index", -1))
            literal_rows = f128_literal_rows_for_fn(module, main_index)
            if literal_rows:
                return {
                    "callee_ret_word_selectors": ret_words,
                    "caller_capture_ret_word_selectors": capture_words,
                    "callee_ret_rows": callee_rets,
                    "caller_capture_ret_rows": main_captures,
                    "caller_literal_return_metadata_rows": literal_rows,
                    "literal_return_metadata_propagated": True,
                }
        if callee_return_param_index >= 0:
            main_index = int(main.get("index", -1))
            literal_rows = f128_literal_rows_for_fn(module, main_index)
            if literal_rows:
                return {
                    "callee_ret_word_selectors": ret_words,
                    "caller_capture_ret_word_selectors": capture_words,
                    "callee_ret_rows": callee_rets,
                    "caller_capture_ret_rows": main_captures,
                    "callee_return_param_index": callee_return_param_index,
                    "caller_literal_return_metadata_rows": literal_rows,
                    "param_return_literal_metadata_propagated": True,
                }
        raise SystemExit(f"{case['case_id']}: f128 caller must capture low/high words separately, got {capture_words}")
    for row in callee_rets:
        if row[INSTR_COND] == 1 and int(row[INSTR_SRC2_KIND]) != 1:
            raise SystemExit(f"{case['case_id']}: RET cond=1 must carry a high-word GPR source")
    return {
        "callee_ret_word_selectors": ret_words,
        "caller_capture_ret_word_selectors": capture_words,
        "callee_ret_rows": callee_rets,
        "caller_capture_ret_rows": main_captures,
    }


def emit_positive(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_dir = out_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    src = write_case(case_dir, case)
    elf = case_dir / "a.out"
    mm = case_dir / "machine.json"
    rc, log = run([str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)], root, timeout_s)
    (case_dir / "compile.log").write_text(log, encoding="utf-8")
    if rc != 0 or "native_v2_compile: emitted" not in log:
        raise SystemExit(f"{case['case_id']}: expected native-v2 emission")
    os.chmod(elf, 0o755)
    run_rc, run_log = run([str(elf)], root, timeout_s)
    (case_dir / "run.log").write_text(run_log, encoding="utf-8")
    expected_exit = int(case.get("expected_exit", 0))
    if run_rc != expected_exit:
        raise SystemExit(f"{case['case_id']}: ELF exit {run_rc}, expected {expected_exit}")
    module = load_machine(mm)
    if module.get("supported") is not True or module.get("unsupported_detail") not in ("", None):
        raise SystemExit(f"{case['case_id']}: MachineModule must be supported")
    callee = fn_by_name(module, str(case["callee"]))
    if callee.get("source_returns_f128") is not True and int(callee.get("source_f128_param_count", 0)) < 1:
        raise SystemExit(f"{case['case_id']}: callee must report f128 param or return")
    if callee.get("source_f128_opaque_direct_call_return_promoted") is not True:
        raise SystemExit(f"{case['case_id']}: direct f128 call/return promotion flag missing")
    f128_return_word_flow = None
    if callee.get("source_returns_f128") is True:
        f128_return_word_flow = require_f128_return_word_flow(module, case)
    return {
        "case_id": case["case_id"],
        "kind": "positive",
        "compile_rc": rc,
        "run_rc": run_rc,
        "expected_exit": expected_exit,
        "machine_module_supported": True,
        "callee": case["callee"],
        "callee_source_f128_param_count": callee.get("source_f128_param_count"),
        "callee_source_returns_f128": callee.get("source_returns_f128"),
        "callee_direct_promoted": callee.get("source_f128_opaque_direct_call_return_promoted"),
        "f128_return_word_flow": f128_return_word_flow,
        "f128_slot_rows": f128_slot_rows(module),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "source_sha256": sha256_text(str(case["source"])),
    }


def emit_negative(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_dir = out_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    src = write_case(case_dir, case)
    elf = case_dir / "a.out"
    mm = case_dir / "machine.json"
    rc, log = run([str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)], root, timeout_s)
    (case_dir / "compile.log").write_text(log, encoding="utf-8")
    detail = str(case["expected_detail"])
    if "native_v2_compile: emitted" in log:
        raise SystemExit(f"{case['case_id']}: unexpectedly emitted executable")
    module = load_machine(mm)
    if module.get("supported") is not False or module.get("unsupported_detail") != detail:
        raise SystemExit(f"{case['case_id']}: MachineModule detail mismatch")
    return {
        "case_id": case["case_id"],
        "kind": "negative",
        "compile_rc": rc,
        "expected_detail": detail,
        "machine_module_supported": False,
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "source_sha256": sha256_text(str(case["source"])),
    }


def emit(args: argparse.Namespace) -> None:
    root = root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [emit_positive(root, compiler, out_dir, c, args.timeout_s) for c in POSITIVE_CASES]
    cases.extend(emit_negative(root, compiler, out_dir, c, args.timeout_s) for c in NEGATIVE_CASES)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "case_id": "s5_5_f128_opaque_direct_call_return_abi",
        "case_count": len(cases),
        "positive_case_count": len(POSITIVE_CASES),
        "negative_case_count": len(NEGATIVE_CASES),
        "f128_opaque_direct_call_return_abi_promoted": True,
        "f128_opaque_direct_expanded_gpr_call_abi_promoted": True,
        "f128_opaque_direct_stack_call_abi_promoted": True,
        "f128_opaque_imported_direct_call_return_abi_promoted": True,
        "f128_native_internal_call_abi_promoted": True,
        "f128_native_internal_return_abi_promoted": True,
        "f128_machineir_return_high_word_capture_promoted": True,
        "f128_external_sysv_abi_promoted": False,
        "f128_sret_abi_promoted": False,
        "f128_arithmetic_promoted": False,
        "f128_software_helpers_promoted": False,
        "f128_nan_inf_contract_promoted": False,
        "cases": cases,
    }
    canonical = stable_json(payload)
    payload["receipt_sha256"] = sha256_text(canonical)
    (out_dir / "madaros_v2_s5_f128_opaque_call_return_abi.receipt.json").write_text(pretty_json(payload), encoding="utf-8")
    print(f"[f128-opaque-call-return-abi] PASS receipt_sha256={payload['receipt_sha256']}")


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
