#!/usr/bin/env python3
"""Emit a Madaros v2 S5 f128 literal value bridge receipt.

This receipt promotes the decimal f128 literal bridge from parser/checker facts
into IR, MachineIR slot metadata, and supported MachineModule JSON. It
also records the S5.2 boundary: native-v2 may emit and execute local opaque
storage/copy cases, but this receipt still does not promote IEEE binary128
materialization, arithmetic, call ABI, or return ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.f128_literal_value_bridge_receipt/0.3"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_METADATA_SCHEMA = "madaros.v2.s5.machine_module_slot_metadata/0.1"
F128_LITERAL_METADATA_SCHEMA = "madaros.v2.s5.f128_literal_metadata/0.1"
STAGE_CONTRACT_LEVEL = "S5_2_F128_LITERAL_VALUE_BRIDGED_WITH_NATIVE_OPAQUE_LOCAL_STORAGE"

F128_SLOT_KIND = 3
F128_WIDTH_WORDS = 2


CASES: list[dict[str, Any]] = [
    {
        "case_id": "f128_literal_one_point_zero_bridge",
        "literal": "1.0",
        "expected": [1, 0, 10, 2, 1, 0],
    },
    {
        "case_id": "f128_literal_zero_point_five_bridge",
        "literal": "0.5",
        "expected": [1, 0, 5, 2, 1, 0],
    },
    {
        "case_id": "f128_literal_long_decimal_bridge",
        "literal": "1.2345678901234567890123456789012345",
        "expected": [1, 90123456789012345, 123456789012345678, 35, 34, 0],
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
        raise SystemExit("f128 literal value bridge receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def f128_slots(module: dict[str, Any]) -> list[dict[str, int]]:
    sm = module.get("slot_metadata")
    if not isinstance(sm, dict):
        raise SystemExit("MachineModule missing slot_metadata")
    if sm.get("schema") != SLOT_METADATA_SCHEMA:
        raise SystemExit(f"bad slot metadata schema: {sm.get('schema')!r}")
    if sm.get("f128_execution_promoted") is not False:
        raise SystemExit("slot metadata must not promote f128 execution")

    rows: list[dict[str, int]] = []
    for fn in sm.get("functions", []):
        fn_index = int(fn.get("fn_index", -1))
        for raw in fn.get("slots", []):
            if not isinstance(raw, list) or len(raw) != 3:
                raise SystemExit(f"bad slot metadata row: {raw!r}")
            slot, kind, width = [int(v) for v in raw]
            if kind == F128_SLOT_KIND:
                if width != F128_WIDTH_WORDS:
                    raise SystemExit(f"f128 slot must have width_words=2, got {raw!r}")
                rows.append({"fn_index": fn_index, "slot": slot, "kind": kind, "width_words": width})
    return rows


def f128_literal_rows(module: dict[str, Any]) -> list[dict[str, int]]:
    meta = module.get("f128_literal_metadata")
    if not isinstance(meta, dict):
        raise SystemExit("MachineModule missing f128_literal_metadata")
    if meta.get("schema") != F128_LITERAL_METADATA_SCHEMA:
        raise SystemExit(f"bad f128 literal metadata schema: {meta.get('schema')!r}")
    if meta.get("f128_literal_decimal_metadata_exported") is not True:
        raise SystemExit("f128 literal decimal metadata export flag missing")
    if meta.get("f128_execution_promoted") is not False:
        raise SystemExit("f128 literal metadata must not promote execution")

    rows: list[dict[str, int]] = []
    for fn in meta.get("functions", []):
        fn_index = int(fn.get("fn_index", -1))
        for raw in fn.get("rows", []):
            if not isinstance(raw, list) or len(raw) != 7:
                raise SystemExit(f"bad f128 literal metadata row: {raw!r}")
            slot, sign, sig_hi, sig_lo, digit_count, scale10, truncated = [int(v) for v in raw]
            rows.append(
                {
                    "fn_index": fn_index,
                    "slot": slot,
                    "decimal_sign": sign,
                    "sig_hi": sig_hi,
                    "sig_lo": sig_lo,
                    "digit_count": digit_count,
                    "scale10": scale10,
                    "truncated_digits": truncated,
                }
            )
    return rows


def load_machine_module(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit("MachineModule source mismatch")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit("MachineModule export flag missing")
    if payload.get("target") != "x86_64-linux":
        raise SystemExit(f"unexpected target: {payload.get('target')!r}")
    if payload.get("supported") is not True:
        raise SystemExit("f128 value bridge cases must now be supported at MachineIR level")
    if payload.get("unsupported_detail") not in ("", None):
        raise SystemExit(f"unexpected MachineModule unsupported detail: {payload.get('unsupported_detail')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    literal = str(case["literal"])
    source = f"""fn main() -> i64 {{
    let x: f128 = {literal} as f128
    0
}}
"""
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    log_path = out_dir / f"{case_id}.native_v2.log"
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
    log = stdout + stderr
    log_path.write_text(log, encoding="utf-8")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case_id} crashed or used fallback; log={log_path}")
    if rc != 0 or "native_v2_compile: emitted" not in log:
        raise SystemExit(f"{case_id} did not emit local opaque f128 ELF; log={log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} missing emitted local opaque f128 ELF")
    os.chmod(elf_path, 0o755)
    run_rc, run_stdout, run_stderr = run_command([str(elf_path)], root, timeout_s)
    run_log = run_stdout + run_stderr
    if run_rc != 0:
        raise SystemExit(f"{case_id} emitted ELF did not run rc=0; rc={run_rc} log={run_log!r}")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit MachineModule JSON")

    module = load_machine_module(mm_path)
    slots = f128_slots(module)
    literal_rows = f128_literal_rows(module)
    if not slots:
        raise SystemExit(f"{case_id} did not export an f128 slot row")
    if len(literal_rows) != 1:
        raise SystemExit(f"{case_id} expected exactly one f128 literal metadata row, got {literal_rows!r}")
    row = literal_rows[0]
    slot_keys = {(slot["fn_index"], slot["slot"]) for slot in slots}
    if (row["fn_index"], row["slot"]) not in slot_keys:
        raise SystemExit(f"{case_id} literal metadata slot is not also an f128 slot: row={row!r} slots={slots!r}")
    expected = [int(v) for v in case["expected"]]
    actual = [
        row["decimal_sign"],
        row["sig_hi"],
        row["sig_lo"],
        row["digit_count"],
        row["scale10"],
        row["truncated_digits"],
    ]
    if actual != expected:
        raise SystemExit(f"{case_id} f128 literal metadata mismatch: expected {expected!r}, got {actual!r}")

    return {
        "case_id": case_id,
        "literal": literal,
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_module_supported": module.get("supported"),
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "run_rc": run_rc,
        "run_log_sha256": sha256_text(run_log),
        "f128_slot_rows": slots,
        "f128_literal_metadata_rows": literal_rows,
        "expected_decimal_metadata": expected,
        "f128_native_opaque_local_storage_promoted": True,
        "f128_ieee_binary128_execution_promoted": False,
    }


def emit(root: Path, compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = [emit_case(root, compiler, out_dir, case, timeout_s) for case in CASES]
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage": "S5",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_count": len(case_rows),
        "cases": case_rows,
        "f128_literal_value_bridge_promoted": True,
        "f128_literal_decimal_metadata_bridged_to_ir": True,
        "f128_literal_decimal_metadata_bridged_to_machine_ir": True,
        "f128_literal_decimal_metadata_bridged_to_machine_module": True,
        "f128_literal_decimal_metadata_machine_module_supported": True,
        "f128_binary128_slot_metadata_emitted": True,
        "f128_machine_ir_opaque_literal_promoted": True,
        "f128_native_opaque_local_storage_promoted": True,
        "f128_native_v2_local_opaque_execution_promoted": True,
        "f128_native_ieee_binary128_materialization_promoted": False,
        "f128_native_arithmetic_promoted": False,
        "f128_native_call_abi_promoted": False,
        "f128_native_return_abi_promoted": False,
        "s5_ready": False,
        "remaining_missing_obligations": [
            "f128 software helper lowering with IEEE rounding and NaN/Inf contract",
            "f128 arithmetic, call ABI, return ABI, and differential receipts",
        ],
    }
    canonical, digest = canonical_roundtrip(payload)
    payload["receipt_sha256"] = digest
    payload["canonical_json_sha256"] = sha256_text(canonical)
    receipt_path = out_dir / "madaros_v2_s5_f128_literal_value_bridge.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["emit"])
    parser.add_argument("--compiler", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()

    root = repo_root_from_script()
    compiler = Path(args.compiler).resolve() if args.compiler else root / "artifacts/self-hosted/madaros"
    if not compiler.exists():
        raise SystemExit(f"compiler not found: {compiler}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "artifacts/madaros_v2_s5_f128_literal_value_bridge_receipt"
    payload = emit(root, compiler, out_dir, args.timeout_s)
    print(
        "f128_literal_value_bridge_receipt: "
        f"PASS cases={payload['case_count']} sha={payload['receipt_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
