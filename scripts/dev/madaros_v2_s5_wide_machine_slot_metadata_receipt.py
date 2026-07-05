#!/usr/bin/env python3
"""Emit a Madaros v2 S5 wide-int MachineIR slot metadata receipt.

This receipt promotes MachineModule/MachineIR metadata for the already
executing i256/u256 wide-limb path. It proves that source-level wide programs
export supported MachineModule JSON with wide slots marked as kind 4 and
width_words=4. It deliberately does not promote f128 execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.wide_machine_slot_metadata_receipt/0.1"
STAGE_CONTRACT_LEVEL = "S5_WIDE_INT_MACHINE_SLOT_METADATA_PROMOTED_NOT_F128"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_KIND_WIDE_INT_LIMB = 4

CASES: list[dict[str, Any]] = [
    {
        "case_id": "i256_add_eq_machine_slots",
        "wide_type": "i256",
        "source": "fn main() -> i64 { let x: i256 = 1 as i256; let y: i256 = 2 as i256; let z: i256 = x + y; if z == (3 as i256) { 7 } else { 1 } }\n",
        "expected_exit": 7,
        "required_width_words": 4,
    },
    {
        "case_id": "u256_mul_add_ne_machine_slots",
        "wide_type": "u256",
        "source": "fn main() -> i64 { let a: u256 = 4294967296 as u256; let b: u256 = 4294967296 as u256; let c: u256 = a * b; let d: u256 = c + c; if d != c { return 42 } 1 }\n",
        "expected_exit": 42,
        "required_width_words": 4,
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
    proc = subprocess.run([str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s, check=False)
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("wide MachineIR slot metadata receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def slot_rows(module: dict[str, Any]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for fn in module.get("slot_metadata", {}).get("functions", []):
        fn_index = int(fn.get("fn_index", -1))
        for raw in fn.get("slots", []):
            if not isinstance(raw, list) or len(raw) != 3:
                raise SystemExit(f"bad slot tuple shape: {raw!r}")
            rows.append(
                {
                    "fn_index": fn_index,
                    "slot": int(raw[0]),
                    "kind": int(raw[1]),
                    "width_words": int(raw[2]),
                }
            )
    return rows


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / case_id
    mm_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.native_v2.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    source_path.write_text(source_text, encoding="utf-8")

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
    if rc != 0 or "native_v2_compile: emitted" not in compile_log:
        raise SystemExit(f"{case_id} native-v2 compile failed rc={rc}; log={compile_log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit MachineModule JSON")

    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, run_stdout, run_stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(run_stdout)
    stderr_path.write_bytes(run_stderr)
    expected_exit = int(case["expected_exit"])
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"{case_id} bad MachineModule schema: {module.get('schema')!r}")
    if module.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"{case_id} missing MachineModule export flag")
    if module.get("supported") is not True:
        raise SystemExit(f"{case_id} MachineModule must be supported, got detail={module.get('unsupported_detail')!r}")
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id} must not use legacy fallback")
    slot_metadata = module.get("slot_metadata", {})
    if slot_metadata.get("machine_ir_slot_metadata_exported") is not True:
        raise SystemExit(f"{case_id} missing slot metadata export flag")
    rows = slot_rows(module)
    wide_rows = [
        row for row in rows
        if row["kind"] == SLOT_KIND_WIDE_INT_LIMB and row["width_words"] == int(case["required_width_words"])
    ]
    if len(wide_rows) < int(case["required_width_words"]):
        raise SystemExit(f"{case_id} expected at least {case['required_width_words']} wide slot rows, got {wide_rows}")
    if any(row["kind"] == 3 for row in rows):
        raise SystemExit(f"{case_id} must not emit f128 slot kind 3")

    return {
        "case_id": case_id,
        "wide_type": case["wide_type"],
        "source": source_path.name,
        "source_sha256": sha256_text(source_text),
        "expected_exit": expected_exit,
        "actual_exit": actual_exit,
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "stdout_sha256": sha256_bytes(run_stdout),
        "stderr_sha256": sha256_bytes(run_stderr),
        "machine_module_json_sha256": sha256_text(stable_json(module)),
        "machine_module_supported": True,
        "slot_metadata_schema": slot_metadata.get("schema"),
        "wide_slot_kind": SLOT_KIND_WIDE_INT_LIMB,
        "wide_slot_width_words": int(case["required_width_words"]),
        "wide_slot_row_count": len(wide_rows),
        "slot_kinds_seen": sorted({row["kind"] for row in rows}),
        "status": "pass",
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "wide_int_machine_slot_metadata",
        "case_count": len(cases),
        "cases": cases,
        "wide_machine_slot_metadata_complete": True,
        "wide_i256_u256_machine_slots_promoted": True,
        "wide_slot_kind_encoding": "4=wide_int_limb",
        "wide_slot_width_words_exported": True,
        "machine_module_supported_for_wide_ints": True,
        "legacy_fallback_for_wide_ints": False,
        "f128_execution_slot_emitted": False,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "i256_u256_source_programs_compile_with_native_v2",
            "i256_u256_source_programs_execute_expected_discriminators",
            "MachineModule_JSON_reports_supported_true_for_wide_programs",
            "MachineModule_slot_metadata_marks_wide_limbs_as_kind_4",
            "MachineModule_slot_metadata_records_width_words_4",
            "f128_slot_kind_3_is_not_emitted_by_this_receipt",
        ],
        "missing_full_obligations": [
            "imported module-boundary i256/u256 ABI call-return receipts",
            "wide ABI stack-pressure cases beyond local two-wide-arg calls",
            "i512/u512 generalized 8-limb execution receipts",
            "f128 IR/MIR/ABI/software-helper receipts",
        ],
    }
    _, receipt_sha = canonical_roundtrip(receipt)
    receipt["receipt_sha256"] = receipt_sha
    receipt_path = out_dir / "madaros_v2_s5_wide_machine_slot_metadata.receipt.json"
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(f"madaros-v2-s5-wide-machine-slot-metadata: cases={len(cases)} sha={receipt_sha[:12]} receipt={receipt_path}")
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
