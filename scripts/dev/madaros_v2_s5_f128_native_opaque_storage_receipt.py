#!/usr/bin/env python3
"""Emit a Madaros v2 S5.2 f128 native opaque-storage receipt.

S5.2 promotes local opaque storage and copy of f128 values as two 64-bit stack
words in native-v2 x86 ELF output. Later S5 receipts promote direct f128
call/return shapes; this receipt still guards arithmetic and over-wide f128
call shapes that exceed the promoted direct register+stack window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.f128_native_opaque_storage_receipt/0.1"
STAGE_CONTRACT_LEVEL = "S5_2_F128_NATIVE_OPAQUE_LOCAL_STORAGE_COPY"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_METADATA_SCHEMA = "madaros.v2.s5.machine_module_slot_metadata/0.1"
F128_LITERAL_METADATA_SCHEMA = "madaros.v2.s5.f128_literal_metadata/0.2"
F128_SLOT_KIND = 3
F128_WIDTH_WORDS = 2


CASES: list[dict[str, Any]] = [
    {
        "case_id": "local_literal_copy_executes",
        "kind": "exec",
        "source": """fn main() -> i64 {
    let x: f128 = 1.0 as f128
    let y: f128 = x
    0
}
""",
        "expected_exit": 0,
    },
    {
        "case_id": "f128_rounded_decimal_arithmetic_stays_blocked",
        "kind": "block",
        "expected_detail": "f128_arithmetic_pending",
        "source": """fn main() -> i64 {
    let x: f128 = 0.1 as f128
    let y: f128 = 0.2 as f128
    let z: f128 = x + y
    0
}
""",
    },
    {
        "case_id": "f128_overwide_arg_shape_stays_blocked",
        "kind": "block",
        "expected_detail": "call_arity_gt_8",
        "source": """fn too_many(a: f128, b: f128, c: f128, d: f128, e: f128, f: f128, g: f128, h: f128, i: f128) -> i64 {
    9
}

fn main() -> i64 {
    let x: f128 = 1.0 as f128
    too_many(x, x, x, x, x, x, x, x, x)
}
""",
    },
    {
        "case_id": "truncated_arbitrary_decimal_materialization_executes",
        "kind": "exec",
        "expected_exit": 0,
        "expected_truncated_tail_info": 71,
        "source": """fn main() -> i64 {
    let x: f128 = 1.23456789012345678901234567890123456789 as f128
    let y: f128 = x
    0
}
""",
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
    env = os.environ.copy()
    raw = cwd / "artifacts" / "self-hosted" / "madaros"
    if raw.exists():
        env["MADAROS_RAW_BIN"] = str(raw)
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


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def load_machine_module(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    return payload


def require_f128_slot_metadata(module: dict[str, Any], case_id: str) -> list[dict[str, int]]:
    sm = module.get("slot_metadata")
    if not isinstance(sm, dict) or sm.get("schema") != SLOT_METADATA_SCHEMA:
        raise SystemExit(f"{case_id}: missing or bad slot metadata")
    rows: list[dict[str, int]] = []
    for fn in sm.get("functions", []):
        for raw in fn.get("slots", []):
            if not isinstance(raw, list) or len(raw) != 3:
                raise SystemExit(f"{case_id}: bad slot row {raw!r}")
            slot, kind, width = [int(v) for v in raw]
            if kind == F128_SLOT_KIND:
                if width != F128_WIDTH_WORDS:
                    raise SystemExit(f"{case_id}: f128 width_words must be 2, got {raw!r}")
                rows.append({"fn_index": int(fn.get("fn_index", -1)), "slot": slot, "kind": kind, "width_words": width})
    if not rows:
        raise SystemExit(f"{case_id}: no f128 slot metadata rows")
    return rows


def require_f128_literal_metadata(module: dict[str, Any], case_id: str) -> list[dict[str, int]]:
    meta = module.get("f128_literal_metadata")
    if not isinstance(meta, dict) or meta.get("schema") != F128_LITERAL_METADATA_SCHEMA:
        raise SystemExit(f"{case_id}: missing or bad f128 literal metadata")
    rows: list[dict[str, int]] = []
    for fn in meta.get("functions", []):
        for raw in fn.get("rows", []):
            if not isinstance(raw, list) or len(raw) < 7:
                raise SystemExit(f"{case_id}: bad f128 literal row {raw!r}")
            slot, sign, sig_hi, sig_lo, digit_count, scale10, truncated = [int(v) for v in raw[:7]]
            tail_info = int(raw[7]) if len(raw) > 7 else 0
            rows.append(
                {
                    "fn_index": int(fn.get("fn_index", -1)),
                    "slot": slot,
                    "decimal_sign": sign,
                    "sig_hi": sig_hi,
                    "sig_lo": sig_lo,
                    "digit_count": digit_count,
                    "scale10": scale10,
                    "truncated_digits": truncated,
                    "truncated_tail_info": tail_info,
                }
            )
    if not rows:
        raise SystemExit(f"{case_id}: no f128 literal metadata rows")
    return rows


def compile_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_path = out_dir / f"{case_id}.sio"
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    log_path = out_dir / f"{case_id}.native_v2.log"
    source = str(case["source"])
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
        raise SystemExit(f"{case_id}: crash or fallback detected; log={log_path}")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case_id}: missing MachineModule JSON")

    module = load_machine_module(mm_path)
    slots = require_f128_slot_metadata(module, case_id)
    literals = require_f128_literal_metadata(module, case_id)
    expected_tail_info = case.get("expected_truncated_tail_info")
    if expected_tail_info is not None:
        if not any(row.get("truncated_tail_info") == int(expected_tail_info) for row in literals):
            raise SystemExit(f"{case_id}: expected truncated_tail_info={expected_tail_info}, got {literals!r}")
    result: dict[str, Any] = {
        "case_id": case_id,
        "kind": case["kind"],
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "f128_slot_rows": slots,
        "f128_literal_rows": literals,
    }

    if case["kind"] == "exec":
        if rc != 0 or "native_v2_compile: emitted" not in log:
            raise SystemExit(f"{case_id}: expected native-v2 ELF emission; log={log_path}")
        if not elf_path.exists() or elf_path.stat().st_size <= 0:
            raise SystemExit(f"{case_id}: missing emitted ELF")
        os.chmod(elf_path, 0o755)
        run_rc, run_stdout, run_stderr = run_command([str(elf_path)], root, timeout_s)
        run_log = run_stdout + run_stderr
        if run_rc != int(case["expected_exit"]):
            raise SystemExit(f"{case_id}: expected exit {case['expected_exit']}, got {run_rc}; run_log={run_log!r}")
        result.update(
            {
                "elf_sha256": sha256_bytes(elf_path.read_bytes()),
                "run_rc": run_rc,
                "run_log_sha256": sha256_text(run_log),
                "native_v2_emitted": True,
            }
        )
    else:
        detail = str(case["expected_detail"])
        if rc == 0 and elf_path.exists() and elf_path.stat().st_size > 0:
            raise SystemExit(f"{case_id}: unexpectedly emitted ELF for pending f128 operation")
        observed_detail = str(module.get("unsupported_detail") or "")
        log_fragment = str(case.get("expected_log_fragment", "native_v2_compile: FAIL"))
        if log_fragment not in log:
            raise SystemExit(f"{case_id}: expected fail-closed log fragment {log_fragment!r}; log={log_path}")
        if detail and observed_detail != detail:
            raise SystemExit(f"{case_id}: expected fail-closed detail {detail!r}; log={log_path}")
        result.update(
            {
                "expected_detail": detail,
                "expected_log_fragment": log_fragment,
                "machine_unsupported_detail": observed_detail,
                "native_v2_emitted": False,
                "blocked_fail_closed": True,
            }
        )
    return result


def emit_receipt(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve() if args.root else repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = int(args.timeout_s)

    case_results = [compile_case(root, compiler, out_dir, case, timeout_s) for case in CASES]
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "root": str(root),
        "compiler": str(compiler),
        "compiler_sha256": sha256_bytes(compiler.read_bytes()),
        "claims": {
            "f128_native_opaque_local_storage_copy_promoted": True,
            "f128_native_executes_local_no_observe_program": True,
            "f128_native_payload_words": ["binary128_hi64", "binary128_lo64"],
            "f128_truncated_arbitrary_decimal_materialization_promoted_elsewhere": True,
            "f128_native_ieee_binary128_materialization_promoted": False,
            "f128_native_arithmetic_promoted": False,
            "f128_opaque_direct_call_return_abi_promoted_elsewhere": True,
            "f128_external_sysv_abi_promoted": False,
            "f128_sret_abi_promoted": False,
            "f128_direct_expanded_gpr_call_shape_promoted_elsewhere": True,
            "f128_direct_stack_call_shape_promoted_elsewhere": True,
            "f128_overwide_call_shape_promoted": False,
            "legacy_fallback_used": False,
        },
        "cases": case_results,
    }
    canonical = stable_json(payload)
    roundtrip = stable_json(json.loads(canonical))
    if canonical != roundtrip:
        raise SystemExit("canonical JSON roundtrip changed bytes")
    payload["receipt_sha256"] = sha256_text(canonical)
    receipt_path = out_dir / "madaros_v2_s5_f128_native_opaque_storage.receipt.json"
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
        print(f"[madaros-v2-s5-f128-native-opaque-storage] receipt={receipt}")


if __name__ == "__main__":
    main()
