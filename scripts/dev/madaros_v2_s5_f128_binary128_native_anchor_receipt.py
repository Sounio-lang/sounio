#!/usr/bin/env python3
"""Emit a Madaros v2 S5.3 f128 native binary128 anchor receipt.

S5.3 promotes native-v2 materialization of exact IEEE-754 binary128 payload
words for a deliberately bounded anchor class of f128 literals. It does not
promote general decimal-to-binary128 codegen, arithmetic, call ABI, or return
ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_binary128_native_anchor_receipt/0.1"
STAGE_CONTRACT_LEVEL = "S5_3_F128_NATIVE_BINARY128_ANCHOR_LITERAL_MATERIALIZATION"

CASES: list[dict[str, Any]] = [
    {
        "case_id": "binary128_anchor_half",
        "literal": "0.5",
        "expected_hex": "3ffe0000000000000000000000000000",
        "expected_hi": 0x3FFE000000000000,
        "expected_lo": 0,
    },
    {
        "case_id": "binary128_anchor_one",
        "literal": "1.0",
        "expected_hex": "3fff0000000000000000000000000000",
        "expected_hi": 0x3FFF000000000000,
        "expected_lo": 0,
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


def mov_rax_imm64_pattern(value: int) -> bytes:
    return b"\x48\xb8" + int(value).to_bytes(8, "little", signed=False)


def compile_anchor(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    literal = str(case["literal"])
    source = f"""fn main() -> i64 {{
    let x: f128 = {literal} as f128
    let y: f128 = x
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
    if rc != 0 or "native_v2_compile: emitted" not in log:
        raise SystemExit(f"{case_id}: expected native-v2 ELF emission; log={log_path}")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case_id}: crash or fallback detected; log={log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: missing emitted ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: missing MachineModule JSON")

    os.chmod(elf_path, 0o755)
    run_rc, run_stdout, run_stderr = run_command([str(elf_path)], root, timeout_s)
    run_log = run_stdout + run_stderr
    if run_rc != 0:
        raise SystemExit(f"{case_id}: emitted ELF must run rc=0, got {run_rc}")

    elf = elf_path.read_bytes()
    hi = int(case["expected_hi"])
    lo = int(case["expected_lo"])
    hi_pattern = mov_rax_imm64_pattern(hi)
    if hi_pattern not in elf:
        raise SystemExit(f"{case_id}: missing binary128 high-word mov immediate {hi:#x}")
    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case_id}: MachineModule used fallback")
    if module.get("supported") is not True:
        raise SystemExit(f"{case_id}: MachineModule should remain supported for local anchor metadata")
    return {
        "case_id": case_id,
        "literal": literal,
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "run_rc": run_rc,
        "run_log_sha256": sha256_text(run_log),
        "elf_sha256": sha256_bytes(elf),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "expected_binary128_hex": str(case["expected_hex"]),
        "expected_hi": hi,
        "expected_lo": lo,
        "hi_mov_imm64_pattern_hex": hi_pattern.hex(),
        "hi_mov_imm64_pattern_found": True,
    }


def emit_receipt(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve() if args.root else repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = [compile_anchor(root, compiler, out_dir, case, int(args.timeout_s)) for case in CASES]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "f128_binary128_native_anchor_literals",
        "case_count": len(case_rows),
        "cases": case_rows,
        "claims": {
            "f128_binary128_native_anchor_materialization_promoted": True,
            "f128_binary128_native_anchor_classes": ["positive finite exact 0.5", "positive finite exact 1.0"],
            "f128_native_payload_words": ["binary128_hi64", "binary128_lo64"],
            "f128_native_general_decimal_binary128_materialization_promoted": False,
            "f128_native_arithmetic_promoted": False,
            "f128_native_call_abi_promoted": False,
            "f128_native_return_abi_promoted": False,
            "legacy_fallback_used": False,
        },
        "roundtrip_contract": [
            "native_v2_emits_and_runs_anchor_literals",
            "elf_contains_expected_mov_rax_imm64_for_binary128_high_word",
            "machine_module_json_remains_supported_without_fallback",
            "receipt_does_not_promote_general_decimal_binary128_or_f128_arithmetic_or_abi",
        ],
    }
    canonical = stable_json(payload)
    if stable_json(json.loads(canonical)) != canonical:
        raise SystemExit("canonical JSON roundtrip changed bytes")
    payload["receipt_sha256"] = sha256_text(canonical)
    receipt_path = out_dir / "madaros_v2_s5_f128_binary128_native_anchor.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--compiler", required=True)
    emit.add_argument("--root")
    emit.add_argument("--out-dir", required=True)
    emit.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args()
    if args.cmd == "emit":
        receipt = emit_receipt(args)
        print(f"[madaros-v2-s5-f128-binary128-native-anchor] receipt={receipt}")


if __name__ == "__main__":
    main()
