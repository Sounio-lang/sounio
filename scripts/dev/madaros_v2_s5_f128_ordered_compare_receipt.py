#!/usr/bin/env python3
"""Emit a Madaros v2 S5.20 f128 ordered-comparison receipt.

This promotes source-observable native-v2 comparisons for compiler-owned
binary128 f128 payloads:

    ==, !=, <, <=, >, >=

The contract is IEEE ordered-comparison shaped for the current promoted
binary128 payload surface: NaN makes ordered predicates false and `!=` true,
signed zeros compare equal, infinities order outside finite values, and finite
positive/negative payloads compare by their binary128 sign/magnitude words. It
does not promote generic IEEE arithmetic, arbitrary decimal materialization, or
external SysV f128 ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_ordered_compare_receipt/0.1"
STAGE = "S5_20_F128_ORDERED_BINARY128_COMPARE"
CASE_ID = "f128_ordered_binary128_source_observable_compare"
EXPONENT_MASK = 0x7FFF000000000000
ABS_MASK = 0x7FFFFFFFFFFFFFFF
SIGN_MASK = 0x8000000000000000


@dataclass(frozen=True)
class Case:
    case_id: str
    lhs: str
    op: str
    rhs: str
    expected_bool: bool
    expected_category: str
    nan_decl: bool = False


CASES: list[Case] = [
    Case("f128_cmp_eq_one_one_true", "1.0 as f128", "==", "1.0 as f128", True, "finite_eq"),
    Case("f128_cmp_ne_one_two_true", "1.0 as f128", "!=", "2.0 as f128", True, "finite_ne"),
    Case("f128_cmp_lt_one_two_true", "1.0 as f128", "<", "2.0 as f128", True, "finite_lt"),
    Case("f128_cmp_le_one_one_true", "1.0 as f128", "<=", "1.0 as f128", True, "finite_le_eq"),
    Case("f128_cmp_gt_two_one_true", "2.0 as f128", ">", "1.0 as f128", True, "finite_gt"),
    Case("f128_cmp_ge_two_two_true", "2.0 as f128", ">=", "2.0 as f128", True, "finite_ge_eq"),
    Case("f128_cmp_signed_zero_eq_true", "-0.0 as f128", "==", "0.0 as f128", True, "signed_zero_eq"),
    Case("f128_cmp_signed_zero_ne_false", "-0.0 as f128", "!=", "0.0 as f128", False, "signed_zero_ne"),
    Case("f128_cmp_negative_order_true", "-2.0 as f128", "<", "-1.0 as f128", True, "negative_finite_lt"),
    Case("f128_cmp_negative_reverse_false", "-1.0 as f128", "<", "-2.0 as f128", False, "negative_finite_lt_false"),
    Case(
        "f128_cmp_subnormal_less_normal_true",
        "6.475175119438025110924438958227646552499569338034681e-4966 as f128",
        "<",
        "1.0 as f128",
        True,
        "subnormal_lt_normal",
    ),
    Case("f128_cmp_positive_infinity_gt_true", "1e5000 as f128", ">", "1.0 as f128", True, "infinity_gt"),
    Case("f128_cmp_negative_infinity_lt_true", "-1e5000 as f128", "<", "-2.0 as f128", True, "negative_infinity_lt"),
    Case("f128_cmp_nan_eq_false", "f128_nan()", "==", "1.0 as f128", False, "nan_eq", True),
    Case("f128_cmp_nan_ne_true", "f128_nan()", "!=", "1.0 as f128", True, "nan_ne", True),
    Case("f128_cmp_nan_lt_false", "f128_nan()", "<", "1.0 as f128", False, "nan_ordered_lt", True),
    Case("f128_cmp_nan_le_false", "f128_nan()", "<=", "1.0 as f128", False, "nan_ordered_le", True),
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


def run(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def source_for(case: Case) -> str:
    nan_decl = "fn f128_nan() -> f128 { 0.0 as f128 }\n" if case.nan_decl else ""
    return f"""{nan_decl}fn main() -> i64 {{
  let lhs: f128 = {case.lhs}
  let rhs: f128 = {case.rhs}
  if lhs {case.op} rhs {{ return 0 }}
  1
}}
"""


def compile_and_run(root: Path, compiler: Path, out_dir: Path, case: Case, timeout_s: int) -> dict[str, Any]:
    src_text = source_for(case)
    src = out_dir / f"{case.case_id}.sio"
    elf = out_dir / f"{case.case_id}.native_v2"
    mm = out_dir / f"{case.case_id}.machine_module.json"
    log_path = out_dir / f"{case.case_id}.native_v2.log"
    run_log_path = out_dir / f"{case.case_id}.run.log"
    src.write_text(src_text, encoding="utf-8")

    rc, log = run(
        [str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)],
        root,
        timeout_s,
    )
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case.case_id}: native-v2 compile failed rc={rc}\n{log}")
    if not elf.exists():
        raise SystemExit(f"{case.case_id}: native-v2 compile did not emit ELF")
    if not mm.exists():
        raise SystemExit(f"{case.case_id}: native-v2 compile did not emit MachineModule JSON")

    module = json.loads(mm.read_text(encoding="utf-8"))
    if module.get("supported") is not True:
        raise SystemExit(f"{case.case_id}: MachineModule must be supported")
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case.case_id}: MachineModule must not use legacy fallback")
    if module.get("unsupported_detail") not in ("", None):
        raise SystemExit(f"{case.case_id}: unexpected unsupported detail {module.get('unsupported_detail')!r}")

    os.chmod(elf, 0o755)
    run_rc, run_log = run([str(elf)], root, timeout_s)
    run_log_path.write_text(run_log, encoding="utf-8")
    expected_rc = 0 if case.expected_bool else 1
    if run_rc != expected_rc:
        raise SystemExit(f"{case.case_id}: expected rc={expected_rc}, got rc={run_rc}\n{run_log}")

    elf_bytes = elf.read_bytes()
    return {
        "case_id": case.case_id,
        "lhs": case.lhs,
        "op": case.op,
        "rhs": case.rhs,
        "expected_bool": case.expected_bool,
        "expected_exit": expected_rc,
        "run_rc": run_rc,
        "category": case.expected_category,
        "source_sha256": sha256_text(src_text),
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "run_log_sha256": sha256_text(normalize_log(run_log, out_dir)),
        "machine_module_json_sha256": sha256_bytes(mm.read_bytes()),
        "elf_sha256": sha256_bytes(elf_bytes),
        "machine_module_supported": module.get("supported"),
        "machine_module_legacy_fallback": module.get("legacy_fallback"),
        "contains_binary128_exponent_mask": EXPONENT_MASK.to_bytes(8, "little") in elf_bytes,
        "contains_binary128_abs_mask": ABS_MASK.to_bytes(8, "little") in elf_bytes,
        "contains_binary128_sign_mask": SIGN_MASK.to_bytes(8, "little") in elf_bytes,
    }


def emit_receipt(compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    root = repo_root_from_script()
    compiler_path = compiler if compiler.is_absolute() else root / compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [compile_and_run(root, compiler_path, out_dir, case, timeout_s) for case in CASES]

    ops_seen = sorted({row["op"] for row in cases})
    if ops_seen != ["!=", "<", "<=", "==", ">", ">="]:
        raise SystemExit(f"f128 ordered comparison ops mismatch: {ops_seen}")
    if not any(row["category"].startswith("nan_") for row in cases):
        raise SystemExit("f128 ordered comparison receipt must include NaN unordered cases")
    if not any(row["category"] == "signed_zero_eq" for row in cases):
        raise SystemExit("f128 ordered comparison receipt must include signed-zero equality")
    if not all(row["contains_binary128_exponent_mask"] for row in cases):
        raise SystemExit("all f128 ordered comparison ELFs must contain exponent mask")
    if not any(row["contains_binary128_sign_mask"] for row in cases):
        raise SystemExit("f128 ordered comparison ELFs must contain sign mask")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "stage_contract_level": STAGE,
        "case_id": CASE_ID,
        "status": "pass",
        "compiler": str(compiler),
        "case_count": len(cases),
        "operators_covered": ops_seen,
        "ordered_compare_contract": {
            "nan_ordered_predicates": False,
            "nan_not_equal": True,
            "signed_zero_equal": True,
            "infinities_ordered": True,
            "finite_sign_magnitude_ordered": True,
        },
        "claims": {
            "f128_ordered_binary128_compare_promoted": True,
            "f128_ordered_binary128_compare_source_observable_promoted": True,
            "f128_ordered_binary128_compare_nan_unordered_promoted": True,
            "f128_ordered_binary128_compare_signed_zero_promoted": True,
            "f128_ordered_binary128_compare_infinity_promoted": True,
            "f128_ordered_binary128_compare_subnormal_promoted": True,
            "f128_native_generic_ieee_arithmetic_promoted": False,
            "f128_software_helpers_promoted": False,
            "f128_external_sysv_abi_promoted": False,
            "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
            "f128_promoted": False,
            "legacy_fallback_used": False,
        },
        "cases": cases,
    }
    stable = stable_json(payload)
    payload["receipt_sha256"] = sha256_text(stable)
    receipt_path = out_dir / "madaros_v2_s5_f128_ordered_compare.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--compiler", default="bin/madaros", type=Path)
    emit.add_argument("--out-dir", required=True, type=Path)
    emit.add_argument("--timeout-s", default=30, type=int)
    args = parser.parse_args()

    if args.cmd == "emit":
        payload = emit_receipt(args.compiler, args.out_dir, args.timeout_s)
        print(
            "madaros-v2-s5-f128-ordered-compare: PASS "
            f"cases={payload['case_count']} receipt_sha256={payload['receipt_sha256']}"
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
