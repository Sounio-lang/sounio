#!/usr/bin/env python3
"""Emit a Madaros v2 S5 MachineIR slot metadata receipt.

This receipt promotes the MachineModule slot-kind/width metadata contract used
by the future f128 binary128 ABI path. It deliberately does not promote f128
execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.machine_slot_metadata_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_METADATA_SCHEMA = "madaros.v2.s5.machine_module_slot_metadata/0.1"
STAGE_CONTRACT_LEVEL = "S5_MACHINE_SLOT_KIND_WIDTH_METADATA_PROMOTED_NOT_F128_EXECUTION"

MIR_SLOT_KIND_I64 = 1
MIR_SLOT_KIND_F64 = 2
MIR_SLOT_KIND_F128_BINARY128 = 3
MIR_SLOT_KIND_WIDE_INT_LIMB = 4


CASES: list[dict[str, Any]] = [
    {
        "case_id": "i64_slot_kind_width_metadata",
        "source": """fn main() -> i64 {
    let x = 1
    let y = x + 2
    if y == 3 { 5 } else { 1 }
}
""",
        "expected_exit": 5,
        "required_kinds": [MIR_SLOT_KIND_I64],
        "forbidden_kinds": [MIR_SLOT_KIND_F128_BINARY128],
    },
    {
        "case_id": "f64_slot_kind_width_metadata",
        "source": """fn main() -> i64 {
    let x = 1.5
    let y = x + 2.5
    if y > 3.5 { 7 } else { 1 }
}
""",
        "expected_exit": 7,
        "required_kinds": [MIR_SLOT_KIND_I64, MIR_SLOT_KIND_F64],
        "forbidden_kinds": [MIR_SLOT_KIND_F128_BINARY128],
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
        raise SystemExit("Machine slot metadata receipt canonical JSON roundtrip changed bytes")
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
    slot_metadata = payload.get("slot_metadata")
    if not isinstance(slot_metadata, dict):
        raise SystemExit("MachineModule missing top-level slot_metadata object")
    if slot_metadata.get("schema") != SLOT_METADATA_SCHEMA:
        raise SystemExit(f"bad slot_metadata schema: {slot_metadata.get('schema')!r}")
    if slot_metadata.get("machine_ir_slot_metadata_exported") is not True:
        raise SystemExit("slot_metadata export flag missing")
    if slot_metadata.get("slot_word_bits") != 64:
        raise SystemExit("slot_metadata must declare 64-bit words")
    if slot_metadata.get("f128_binary128_limb_count") != 2:
        raise SystemExit("slot_metadata must reserve two 64-bit limbs for binary128")
    if slot_metadata.get("f128_binary128_limb_bits") != 64:
        raise SystemExit("slot_metadata must declare binary128 limb width")
    if slot_metadata.get("f128_execution_promoted") is not False:
        raise SystemExit("slot_metadata must not promote f128 execution")
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def slot_rows(module: dict[str, Any]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for fn in module["slot_metadata"].get("functions", []):
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
        raise SystemExit(f"{case_id} failed native-v2 compile; log={compile_log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit an ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id} did not emit MachineModule JSON")

    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    expected_exit = int(case["expected_exit"])
    if actual_exit != expected_exit:
        raise SystemExit(f"{case_id} expected exit {expected_exit}, got {actual_exit}")

    module = load_machine_module(mm_path)
    rows = slot_rows(module)
    if not rows:
        raise SystemExit(f"{case_id} emitted no slot metadata rows")
    kinds = sorted({row["kind"] for row in rows})
    widths_by_kind: dict[int, list[int]] = {}
    for row in rows:
        widths_by_kind.setdefault(row["kind"], []).append(row["width_words"])
        if row["kind"] in (MIR_SLOT_KIND_I64, MIR_SLOT_KIND_F64) and row["width_words"] != 1:
            raise SystemExit(f"{case_id} scalar slot kind {row['kind']} must have width_words=1")
        if row["kind"] == MIR_SLOT_KIND_F128_BINARY128:
            raise SystemExit(f"{case_id} unexpectedly emitted f128 slot metadata")

    for required in case["required_kinds"]:
        if int(required) not in kinds:
            raise SystemExit(f"{case_id} missing required slot kind {required}; got {kinds}")
    for forbidden in case["forbidden_kinds"]:
        if int(forbidden) in kinds:
            raise SystemExit(f"{case_id} emitted forbidden slot kind {forbidden}; got {kinds}")

    return {
        "case_id": case_id,
        "source": source_path.name,
        "source_sha256": sha256_text(source_text),
        "compile_rc": compile_rc,
        "actual_exit": actual_exit,
        "expected_exit": expected_exit,
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "slot_metadata_schema": module["slot_metadata"]["schema"],
        "slot_metadata_function_count": len(module["slot_metadata"].get("functions", [])),
        "slot_metadata_row_count": len(rows),
        "slot_kinds_seen": kinds,
        "widths_by_kind": {str(k): sorted(set(v)) for k, v in widths_by_kind.items()},
        "status": "pass",
    }


def emit(args: argparse.Namespace) -> int:
    root = repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [emit_case(root, compiler, out_dir, case, int(args.timeout_s)) for case in CASES]

    all_kinds = sorted({kind for case in cases for kind in case["slot_kinds_seen"]})
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage": "S5",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "machine_slot_kind_width_metadata",
        "case_count": len(cases),
        "cases": cases,
        "machine_ir_slot_metadata_exported": True,
        "slot_metadata_schema": SLOT_METADATA_SCHEMA,
        "slot_kind_encoding_complete_for_current_scalars": True,
        "slot_width_words_exported": True,
        "i64_slot_kind_width_promoted": True,
        "f64_slot_kind_width_promoted": True,
        "f64_slot_kind_distinguished_from_i64": True,
        "f128_binary128_slot_kind_reserved": True,
        "f128_binary128_limb_contract_recorded": True,
        "f128_binary128_limb_count": 2,
        "f128_binary128_limb_bits": 64,
        "wide_int_limb_slot_kind_reserved": True,
        "slot_kinds_seen": all_kinds,
        "f128_execution_slot_emitted": False,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "missing_full_obligations": [
            "f128 decimal-to-binary128 rounded value from parser decimal metadata",
            "f128 IR opcodes and constructors",
            "f128 MachineIR lowering that emits slot kind 3 with two 64-bit limbs",
            "f128 SysV ABI classification and call-return signature metadata",
            "f128 software-helper lowering with IEEE rounding and NaN/Inf receipts",
            "native-v2 f128 execution differential receipts",
        ],
    }
    _, receipt_sha = canonical_roundtrip(receipt)
    receipt["receipt_sha256"] = receipt_sha
    receipt_path = out_dir / "madaros_v2_s5_machine_slot_metadata.receipt.json"
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(f"[madaros-v2-s5-machine-slot-metadata] ok cases={len(cases)} sha={receipt_sha[:12]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--compiler", required=True)
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args()
    if args.cmd == "emit":
        return emit(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
