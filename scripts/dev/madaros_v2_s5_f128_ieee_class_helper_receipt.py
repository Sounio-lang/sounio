#!/usr/bin/env python3
"""Emit a Madaros v2 S5.14 f128 IEEE class-code helper receipt.

This promotes a narrow compiler-owned helper surface:

    f128_class_code(x: f128) -> i64

The native-v2 backend recognizes that name as a builtin and classifies the
binary128 payload into zero/subnormal/normal/infinity/NaN class codes. The
receipt promotes source-observable zero, subnormal, normal, infinity, and NaN
classification. NaN is constructed through the compiler-owned `f128_nan()`
canonical quiet-NaN builtin. It deliberately does not promote generic f128 IEEE
arithmetic, external SysV ABI, or arbitrary decimal materialization.
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


SCHEMA = "madaros.v2.s5.f128_ieee_class_helper_receipt/0.2"
STAGE_CONTRACT_LEVEL = "S5_14_F128_NATIVE_IEEE_CLASS_CODE_HELPER_WITH_NAN_SOURCE"
CASE_ID = "f128_ieee_class_code_source_observable_binary128_payloads"
EXPONENT_MASK = 0x7FFF000000000000
FRACTION_HIGH_MASK = 0x0000FFFFFFFFFFFF


@dataclass(frozen=True)
class Case:
    case_id: str
    literal: str
    expected_class_code: int
    expected_class_name: str
    expected_supported: bool = True
    expected_unsupported_detail: str = ""


CASES: list[Case] = [
    Case("zero_positive", "0.0", 0, "zero"),
    Case("zero_negative", "-0.0", 0, "zero"),
    Case("normal_one", "1.0", 2, "normal"),
    Case("normal_one_tenth", "0.1", 2, "normal"),
    Case("normal_negative_one_tenth", "-0.1", 2, "normal"),
    Case(
        "normal_smallest_binary128",
        "3.36210314311209350626267781732175260259807934484647e-4932",
        2,
        "normal",
    ),
    Case(
        "subnormal_min_positive",
        "6.475175119438025110924438958227646552499569338034681e-4966",
        1,
        "subnormal",
    ),
    Case("underflow_positive_zero", "1e-5000", 0, "zero"),
    Case("infinity_positive_overflow", "1e5000", 3, "infinity"),
    Case("infinity_negative_overflow", "-1e5000", 3, "infinity"),
    Case("nan_canonical_quiet_builtin", "f128_nan()", 4, "nan"),
]

NEGATIVE_CASES: list[Case] = [
    Case(
        "negative_min_subnormal_materialization_pending",
        "-6.475175119438025110924438958227646552499569338034681e-4966",
        1,
        "subnormal",
        expected_supported=False,
        expected_unsupported_detail="f128_decimal_materialization_pending",
    ),
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


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def extract_literal_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    meta = module.get("f128_literal_metadata", {})
    for fn in meta.get("functions", []):
        for row in fn.get("rows", []):
            if isinstance(row, list):
                rows.append([int(x) for x in row])
    return rows


def compile_case(root: Path, compiler: Path, out_dir: Path, case: Case, timeout_s: int) -> dict[str, Any]:
    nan_decl = "fn f128_nan() -> f128 { 0.0 as f128 }\n" if case.literal == "f128_nan()" else ""
    value_expr = case.literal if case.literal == "f128_nan()" else f"{case.literal} as f128"
    source = f"""{nan_decl}fn f128_class_code(x: f128) -> i64 {{ 0 }}
fn main() -> i64 {{
    let x: f128 = {value_expr}
    f128_class_code(x)
}}
"""
    source_path = out_dir / f"{case.case_id}.sio"
    elf_path = out_dir / f"{case.case_id}.native_v2"
    mm_path = out_dir / f"{case.case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case.case_id}.native_v2.log"
    run_log_path = out_dir / f"{case.case_id}.run.log"

    source_path.write_text(source, encoding="utf-8")
    rc, stdout, stderr = run_command(
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
    compile_log = stdout + stderr
    compile_log_path.write_text(compile_log, encoding="utf-8")

    module: dict[str, Any] = {}
    if mm_path.exists():
        module = json.loads(mm_path.read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "literal": case.literal,
        "expected_class_code": case.expected_class_code,
        "expected_class_name": case.expected_class_name,
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "machine_module_json_sha256": sha256_bytes(mm_path.read_bytes()) if mm_path.exists() else "",
        "machine_module_supported": module.get("supported"),
        "machine_module_legacy_fallback": module.get("legacy_fallback"),
        "machine_module_unsupported_detail": module.get("unsupported_detail", ""),
        "f128_literal_metadata_rows": extract_literal_rows(module),
    }

    if case.expected_supported:
        if rc != 0:
            raise SystemExit(f"{case.case_id}: native-v2 compile failed rc={rc}\n{compile_log}")
        if not elf_path.exists():
            raise SystemExit(f"{case.case_id}: native-v2 compile did not emit ELF")
        if module.get("supported") is not True:
            raise SystemExit(f"{case.case_id}: MachineModule must be supported, got {module.get('supported')}")
        if module.get("legacy_fallback") is not False:
            raise SystemExit(f"{case.case_id}: MachineModule must not use legacy fallback")
        if module.get("unsupported_detail") not in ("", None):
            raise SystemExit(f"{case.case_id}: unexpected unsupported detail {module.get('unsupported_detail')!r}")
        if case.expected_class_name != "nan" and not row["f128_literal_metadata_rows"]:
            raise SystemExit(f"{case.case_id}: expected f128 literal metadata row")
        os.chmod(elf_path, 0o755)
        run_rc, run_stdout, run_stderr = run_command([str(elf_path)], root, timeout_s)
        run_log = run_stdout + run_stderr
        run_log_path.write_text(run_log, encoding="utf-8")
        if run_rc != case.expected_class_code:
            raise SystemExit(
                f"{case.case_id}: expected class code {case.expected_class_code}, got {run_rc}\n{run_log}"
            )
        elf_bytes = elf_path.read_bytes()
        row.update(
            {
                "run_rc": run_rc,
                "run_stdout": run_stdout,
                "run_stderr_sha256": sha256_text(normalize_log(run_stderr, out_dir)),
                "elf_sha256": sha256_bytes(elf_bytes),
                "contains_exponent_mask_imm64": EXPONENT_MASK.to_bytes(8, "little") in elf_bytes,
                "contains_fraction_high_mask_imm64": FRACTION_HIGH_MASK.to_bytes(8, "little") in elf_bytes,
            }
        )
        if row["contains_exponent_mask_imm64"] is not True:
            raise SystemExit(f"{case.case_id}: emitted ELF missing binary128 exponent mask immediate")
        if row["contains_fraction_high_mask_imm64"] is not True:
            raise SystemExit(f"{case.case_id}: emitted ELF missing binary128 fraction-high mask immediate")
    else:
        if module.get("supported") is not False:
            raise SystemExit(f"{case.case_id}: expected fail-closed unsupported MachineModule")
        if module.get("legacy_fallback") is not False:
            raise SystemExit(f"{case.case_id}: negative case must not use legacy fallback")
        if module.get("unsupported_detail") != case.expected_unsupported_detail:
            raise SystemExit(
                f"{case.case_id}: expected unsupported detail {case.expected_unsupported_detail!r}, "
                f"got {module.get('unsupported_detail')!r}"
            )
        if elf_path.exists():
            raise SystemExit(f"{case.case_id}: negative case unexpectedly emitted ELF")
        row.update(
            {
                "run_rc": None,
                "run_stdout": "",
                "run_stderr_sha256": "",
                "elf_sha256": "",
                "contains_exponent_mask_imm64": False,
                "contains_fraction_high_mask_imm64": False,
            }
        )

    return row


def emit_receipt(compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    root = repo_root_from_script()
    compiler_path = compiler if compiler.is_absolute() else root / compiler
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [compile_case(root, compiler_path, out_dir, case, timeout_s) for case in CASES]
    negative_cases = [compile_case(root, compiler_path, out_dir, case, timeout_s) for case in NEGATIVE_CASES]

    class_codes_seen = sorted({int(row["run_rc"]) for row in cases})
    expected_class_codes = sorted({case.expected_class_code for case in CASES})
    if class_codes_seen != expected_class_codes:
        raise SystemExit(f"class codes mismatch: expected {expected_class_codes}, got {class_codes_seen}")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": CASE_ID,
        "status": "pass",
        "compiler": str(compiler),
        "case_count": len(cases),
        "negative_case_count": len(negative_cases),
        "class_code_contract": {
            "zero": 0,
            "subnormal": 1,
            "normal": 2,
            "infinity": 3,
            "nan": 4,
        },
        "claims": {
            "f128_native_ieee_class_code_helper_promoted": True,
            "f128_native_ieee_class_code_source_observable_zero_subnormal_normal_infinity_promoted": True,
            "f128_native_ieee_class_code_nan_branch_emitted": True,
            "f128_native_ieee_class_code_nan_source_surface_promoted": True,
            "f128_native_canonical_quiet_nan_constructor_promoted": True,
            "f128_native_generic_ieee_arithmetic_promoted": False,
            "f128_external_sysv_abi_promoted": False,
            "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
            "legacy_fallback_used": False,
        },
        "observed_class_codes": class_codes_seen,
        "cases": cases,
        "negative_cases": negative_cases,
    }
    stable = stable_json(payload)
    payload["receipt_sha256"] = sha256_text(stable)
    receipt_path = out_dir / "madaros_v2_s5_f128_ieee_class_helper.receipt.json"
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
            f"madaros-v2-s5-f128-ieee-class-helper: PASS "
            f"cases={payload['case_count']} negative={payload['negative_case_count']} "
            f"receipt_sha256={payload['receipt_sha256']}"
        )
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
