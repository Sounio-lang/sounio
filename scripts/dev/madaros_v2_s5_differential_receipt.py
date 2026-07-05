#!/usr/bin/env python3
"""Emit a Madaros v2 S5 native-v2 vs lean_single differential receipt.

This receipt is intentionally program-level: every promoted positive S5 source
surface that the legacy lean_single reference can execute is compiled through
the explicit native-v2 path, executed as an ELF, and compared against
SOUNIO_SOUC_ENGINE=lean_single. Cases with a known reference-output mismatch
are recorded as unavailable rather than counted as equivalence evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.differential_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_NATIVE_V2_LEAN_SINGLE_DIFFERENTIAL_PROMOTED_NOT_F128"

UNAVAILABLE_REFERENCE_CASES = {
    "f64_println_call_stdout_4_5": "lean_single prints f64 println without the trailing newline emitted by native-v2 print_char",
    "f64_let_bound_println_stdout_4_5": "lean_single prints let-bound f64 println without the trailing newline emitted by native-v2 print_char",
}

REQUIRED_CATEGORIES = {
    "scalar_i64",
    "scalar_bool",
    "normal_call_stack_args",
    "source_sret_local",
    "imported_sret_module_boundary",
    "method_sret",
    "f64_xmm0",
    "wide_int_source",
    "generic_aggregate_sret",
}


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


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("differential receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def load_module(root: Path, name: str) -> Any:
    path = root / "scripts" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import receipt helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
        env=merged_env,
    )
    return proc.returncode, proc.stdout or b"", proc.stderr or b""


def load_machine_module(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema in {path}: {payload.get('schema')!r}")
    if payload.get("source") != "native_v2_build_machine_module":
        raise SystemExit(f"MachineModule source mismatch in {path}")
    if payload.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"MachineModule export flag missing in {path}")
    if payload.get("active") is not True:
        raise SystemExit(f"MachineModule inactive in {path}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit(f"MachineModule used legacy fallback in {path}")
    return payload


def case_source_rows(root: Path) -> list[dict[str, Any]]:
    stack = load_module(root, "madaros_v2_s5_stack_call_receipt")
    source_sret = load_module(root, "madaros_v2_s5_source_sret_receipt")
    imported_sret = load_module(root, "madaros_v2_s5_imported_sret_receipt")
    method_sret = load_module(root, "madaros_v2_s5_method_sret_receipt")
    f64 = load_module(root, "madaros_v2_s5_f64_xmm0_receipt")
    wide = load_module(root, "madaros_v2_s5_wide_int_receipt")
    generic = load_module(root, "madaros_v2_s5_generic_aggregate_sret_receipt")

    rows: list[dict[str, Any]] = []
    for case_id, source_path, expected_exit, category in [
        (
            "scalar_i64_literal_return_42",
            root / "tests/madaros/v2_s5/scalar_i64_literal_return_42.sio",
            42,
            "scalar_i64",
        ),
        (
            "scalar_i64_direct_call_return_42",
            root / "tests/madaros/v2_s5/scalar_i64_direct_call_return_42.sio",
            42,
            "scalar_i64",
        ),
        (
            "scalar_bool_direct_call_return_1",
            root / "tests/madaros/v2_s5/scalar_bool_direct_call_return_1.sio",
            1,
            "scalar_bool",
        ),
    ]:
        rows.append(
            {
                "case_id": case_id,
                "category": category,
                "source": source_path.read_text(encoding="utf-8"),
                "expected_exit": expected_exit,
                "support_files": {},
            }
        )

    for case in stack.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "normal_call_stack_args",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in source_sret.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "source_sret_local",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in imported_sret.CASES:
        lib = imported_sret.LIB_TEMPLATE.format(params=case["params"], body=case["body"])
        main = imported_sret.MAIN_TEMPLATE.format(module_name=case["module_name"], call=case["call"])
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "imported_sret_module_boundary",
                "source": main,
                "expected_exit": int(case["expected_exit"]),
                "support_files": {f"{case['module_name']}.sio": lib},
            }
        )
    for case in method_sret.CASES:
        source = method_sret.PROGRAM_TEMPLATE.format(
            params=case["params"],
            body=case["body"],
            seed=case["seed"],
            call_args=case["call_args"],
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "method_sret",
                "source": source,
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
            }
        )
    for case in f64.CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "f64_xmm0",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
                "expected_stdout": case["expected_stdout"].decode("utf-8"),
            }
        )
    for case in wide.SOURCE_NATIVE_CASES:
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "wide_int_source",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": {},
                "wide_type": case["wide_type"],
                "wide_ops": list(case["wide_ops"]),
            }
        )
    for case in generic.CASES:
        support_files = {}
        if case["path_kind"] == "imported":
            support_files["imported_generic_wide9_lib.sio"] = generic.IMPORTED_WIDE9_LIB
        rows.append(
            {
                "case_id": case["case_id"],
                "category": "generic_aggregate_sret",
                "source": case["source"],
                "expected_exit": int(case["expected_exit"]),
                "support_files": support_files,
                "path_kind": case["path_kind"],
                "field_count": int(case["field_count"]),
                "declared_layout_bytes": int(case["declared_layout_bytes"]),
            }
        )

    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            raise SystemExit(f"duplicate differential case id: {case_id}")
        seen.add(case_id)
    return rows


def emit_case(
    root: Path,
    compiler: Path,
    reference_souc: Path,
    out_dir: Path,
    case: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_path = case_dir / f"{case_id}.sio"
    source_text = str(case["source"])
    source_path.write_text(source_text, encoding="utf-8")
    for name, text in dict(case.get("support_files", {})).items():
        (case_dir / name).write_text(str(text), encoding="utf-8")

    elf_path = case_dir / f"{case_id}.native_v2"
    mm_path = case_dir / f"{case_id}.machine_module.json"
    compile_log = case_dir / f"{case_id}.native_v2.compile.log"
    native_stdout_path = case_dir / f"{case_id}.native.stdout"
    native_stderr_path = case_dir / f"{case_id}.native.stderr"
    lean_stdout_path = case_dir / f"{case_id}.lean_single.stdout"
    lean_stderr_path = case_dir / f"{case_id}.lean_single.stderr"

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
    compile_log.write_bytes(compile_stdout + compile_stderr)
    if compile_rc != 0:
        raise SystemExit(f"{case_id} native-v2 compile failed rc={compile_rc}; log={compile_log}")
    if not elf_path.is_file():
        raise SystemExit(f"{case_id} native-v2 compile did not produce ELF")
    elf_path.chmod(0o755)
    machine_module = load_machine_module(mm_path)

    native_rc, native_stdout, native_stderr = run_command([str(elf_path)], root, timeout_s)
    native_stdout_path.write_bytes(native_stdout)
    native_stderr_path.write_bytes(native_stderr)

    lean_rc, lean_stdout, lean_stderr = run_command(
        [str(reference_souc), "run", str(source_path)],
        root,
        timeout_s,
        {"SOUNIO_SOUC_ENGINE": "lean_single"},
    )
    lean_stdout_path.write_bytes(lean_stdout)
    lean_stderr_path.write_bytes(lean_stderr)

    expected_exit = int(case["expected_exit"])
    available = case_id not in UNAVAILABLE_REFERENCE_CASES
    status = "matched" if available else "reference_unavailable"
    mismatch_reason = ""
    if native_rc != expected_exit:
        raise SystemExit(f"{case_id} native-v2 exit mismatch: expected {expected_exit}, got {native_rc}")
    if lean_rc != expected_exit:
        raise SystemExit(f"{case_id} lean_single exit mismatch: expected {expected_exit}, got {lean_rc}")
    if available and native_stdout != lean_stdout:
        raise SystemExit(
            f"{case_id} stdout mismatch: native={native_stdout!r} lean_single={lean_stdout!r}"
        )
    if not available:
        mismatch_reason = UNAVAILABLE_REFERENCE_CASES[case_id]
        if native_stdout == lean_stdout:
            raise SystemExit(f"{case_id} was listed unavailable but stdout now matches; update receipt contract")

    row = {
        "case_id": case_id,
        "category": case["category"],
        "status": status,
        "reference_available": available,
        "reference_unavailable_reason": mismatch_reason,
        "expected_exit": expected_exit,
        "native_v2_exit": native_rc,
        "lean_single_exit": lean_rc,
        "stdout_equal": native_stdout == lean_stdout,
        "stderr_equal": native_stderr == lean_stderr,
        "source_sha256": sha256_text(source_text),
        "support_file_count": len(dict(case.get("support_files", {}))),
        "support_file_sha256": {
            name: sha256_text(str(text)) for name, text in sorted(dict(case.get("support_files", {})).items())
        },
        "machine_module_schema": machine_module["schema"],
        "machine_module_json_sha256": sha256_text(stable_json(machine_module)),
        "machine_module_path": f"{case_id}/{mm_path.name}",
        "machine_module_supported": machine_module.get("supported"),
        "machine_module_unsupported_detail": machine_module.get("unsupported_detail"),
        "machine_module_legacy_fallback": machine_module["legacy_fallback"],
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "native_stdout_sha256": sha256_bytes(native_stdout),
        "native_stderr_sha256": sha256_bytes(native_stderr),
        "lean_single_stdout_sha256": sha256_bytes(lean_stdout),
        "lean_single_stderr_sha256": sha256_bytes(lean_stderr),
        "native_stdout": native_stdout.decode("utf-8", errors="replace"),
        "lean_single_stdout": lean_stdout.decode("utf-8", errors="replace"),
    }
    for key in ["expected_stdout", "wide_type", "wide_ops", "path_kind", "field_count", "declared_layout_bytes"]:
        if key in case:
            row[key] = case[key]
    row["differential_case_sha256"] = sha256_text(stable_json(row))
    return row


def emit(args: argparse.Namespace) -> int:
    root = repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    reference_souc = Path(args.reference_souc).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = case_source_rows(root)
    results = [emit_case(root, compiler, reference_souc, out_dir, case, args.timeout) for case in cases]
    matched = [row for row in results if row["status"] == "matched"]
    unavailable = [row for row in results if row["status"] == "reference_unavailable"]
    if len(results) != 33:
        raise SystemExit(f"differential receipt expected 33 cases, got {len(results)}")
    if len(matched) != 31:
        raise SystemExit(f"differential receipt expected 31 matched cases, got {len(matched)}")
    if {row["case_id"] for row in unavailable} != set(UNAVAILABLE_REFERENCE_CASES):
        raise SystemExit(f"unexpected unavailable differential cases: {[row['case_id'] for row in unavailable]}")
    matched_categories = {str(row["category"]) for row in matched}
    missing_categories = REQUIRED_CATEGORIES - matched_categories
    if missing_categories:
        raise SystemExit(f"differential matched cases missing categories: {sorted(missing_categories)}")

    payload = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "native_v2_vs_lean_single_promoted_surfaces",
        "target": "x86_64-linux",
        "native_compiler": str(compiler),
        "reference_souc": str(reference_souc),
        "reference_engine": "lean_single",
        "case_count": len(results),
        "matched_case_count": len(matched),
        "reference_unavailable_case_count": len(unavailable),
        "categories_compared": sorted(matched_categories),
        "required_categories": sorted(REQUIRED_CATEGORIES),
        "cases": results,
        "native_v2_vs_lean_single_differential_complete": True,
        "s5_differential_native_v2_lean_single_complete": True,
        "differential_native_v2_vs_lean_single_promoted": True,
        "all_reference_available_cases_match_exit_and_stdout": True,
        "all_native_v2_cases_compile_without_legacy_fallback": True,
        "all_native_v2_cases_return_expected_exit": True,
        "all_lean_single_cases_return_expected_exit": True,
        "known_reference_unavailable_cases_recorded": True,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "missing_full_obligations": ["f128 numeric tower width receipts"],
    }
    canonical, payload_sha = canonical_roundtrip(payload)
    payload["receipt_sha256"] = payload_sha
    canonical_with_hash, _ = canonical_roundtrip(payload)
    payload["canonical_roundtrip_sha256"] = sha256_text(canonical_with_hash)
    receipt_path = out_dir / "madaros_v2_s5_differential.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    print(
        "madaros-v2-s5-differential: "
        f"matched={len(matched)}/{len(results)} unavailable={len(unavailable)} "
        f"sha={payload['receipt_sha256'][:12]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_parser = sub.add_parser("emit")
    emit_parser.add_argument("--compiler", required=True)
    emit_parser.add_argument("--reference-souc", required=True)
    emit_parser.add_argument("--out-dir", required=True)
    emit_parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.cmd == "emit":
        return emit(args)
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
