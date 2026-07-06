#!/usr/bin/env python3
"""Emit a Madaros v2 S5 f128 parameter slot-layout receipt.

This receipt protects a narrow but important ABI invariant: each source f128
parameter is expanded into two non-overlapping 64-bit MachineIR slots. It also
records that the later S5 finite value-contract add/sub helper can consume that
layout without reintroducing overlap. It does not promote generic IEEE f128
helpers, external SysV f128 ABI, SRET, or full f128 execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_param_slot_layout_receipt/0.1"
STAGE = "S5_7_F128_NON_OVERLAPPING_PARAMETER_SLOTS"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
SLOT_METADATA_SCHEMA = "madaros.v2.s5.machine_module_slot_metadata/0.1"
F128_SLOT_KIND = 3
F128_WIDTH_WORDS = 2


CASES: list[dict[str, Any]] = [
    {
        "case_id": "local_two_f128_params_non_overlapping",
        "source": """fn pair(x: f128, y: f128) -> i64 { 5 }
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 0.5 as f128
  pair(a, b)
}
""",
        "callee": "pair",
        "expected_fn_count": 2,
        "expected_source_param_count": 4,
        "expected_source_f128_param_count": 2,
        "expected_f128_rows": [[0, 3, 2], [2, 3, 2]],
        "expected_supported": True,
        "expected_detail": "",
    },
    {
        "case_id": "local_f128_i64_f128_params_non_overlapping",
        "source": """fn mixed(x: f128, y: i64, z: f128) -> i64 { y }
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 0.5 as f128
  mixed(a, 7, b)
}
""",
        "callee": "mixed",
        "expected_fn_count": 2,
        "expected_source_param_count": 5,
        "expected_source_f128_param_count": 2,
        "expected_f128_rows": [[0, 3, 2], [3, 3, 2]],
        "expected_supported": True,
        "expected_detail": "",
    },
    {
        "case_id": "imported_two_f128_params_non_overlapping",
        "support_files": {"f128_layout_lib.sio": "pub fn imported_pair(x: f128, y: f128) -> i64 { 12 }\n"},
        "source": """import "f128_layout_lib.sio"
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 0.5 as f128
  imported_pair(a, b)
}
""",
        "callee": "imported_pair",
        "expected_fn_count": 3,
        "expected_source_param_count": 4,
        "expected_source_f128_param_count": 2,
        "expected_f128_rows": [[0, 3, 2], [2, 3, 2]],
        "expected_supported": True,
        "expected_detail": "",
    },
    {
        "case_id": "f128_callee_add_args_slot_layout_feeds_runtime_helper",
        "source": """fn add_f128(x: f128, y: f128) -> f128 { x + y }
fn main() -> i64 {
  let a: f128 = 1.0 as f128
  let b: f128 = 2.0 as f128
  let c: f128 = add_f128(a, b)
  let d: f128 = c
  0
}
""",
        "callee": "add_f128",
        "expected_fn_count": 2,
        "expected_source_param_count": 4,
        "expected_source_f128_param_count": 2,
        "expected_f128_rows": [[0, 3, 2], [2, 3, 2], [4, 3, 2]],
        "expected_supported": True,
        "expected_detail": "",
    },
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("f128 parameter slot-layout receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def run_command(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str]:
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
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def load_machine_module(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    module = json.loads(path.read_text(encoding="utf-8"))
    if module.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"{case['case_id']} bad MachineModule schema: {module.get('schema')!r}")
    if module.get("compiler_machine_module_exported") is not True:
        raise SystemExit(f"{case['case_id']} did not export MachineModule JSON")
    if module.get("target") != "x86_64-linux":
        raise SystemExit(f"{case['case_id']} unexpected target: {module.get('target')!r}")
    if int(module.get("fn_count", -1)) != int(case["expected_fn_count"]):
        raise SystemExit(f"{case['case_id']} fn_count mismatch: {module.get('fn_count')!r}")
    if bool(module.get("supported", False)) is not bool(case["expected_supported"]):
        raise SystemExit(f"{case['case_id']} supported mismatch: {module.get('supported')!r}")
    expected_detail = str(case["expected_detail"])
    actual_detail = str(module.get("unsupported_detail") or "")
    if actual_detail != expected_detail:
        raise SystemExit(f"{case['case_id']} unsupported_detail mismatch: {actual_detail!r}")
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case['case_id']} used legacy fallback")
    return module


def functions_by_name(module: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(fn.get("name", "")): fn for fn in module.get("functions", [])}


def f128_rows_for_function(module: dict[str, Any], fn_index: int) -> list[list[int]]:
    slot_metadata = module.get("slot_metadata")
    if not isinstance(slot_metadata, dict):
        raise SystemExit("MachineModule slot_metadata missing")
    if slot_metadata.get("schema") != SLOT_METADATA_SCHEMA:
        raise SystemExit(f"bad slot_metadata schema: {slot_metadata.get('schema')!r}")
    if slot_metadata.get("f128_execution_promoted") is not False:
        raise SystemExit("slot metadata must not claim full f128 execution")
    for fn_meta in slot_metadata.get("functions", []):
        if int(fn_meta.get("fn_index", -1)) == fn_index:
            rows: list[list[int]] = []
            for row in fn_meta.get("slots", []):
                if not isinstance(row, list) or len(row) != 3:
                    raise SystemExit(f"bad slot row for fn_index={fn_index}: {row!r}")
                normalized = [int(row[0]), int(row[1]), int(row[2])]
                if normalized[1] == F128_SLOT_KIND:
                    if normalized[2] != F128_WIDTH_WORDS:
                        raise SystemExit(f"f128 row must be width 2, got {normalized!r}")
                    rows.append(normalized)
            return rows
    return []


def assert_non_overlapping(rows: list[list[int]], case_id: str) -> None:
    occupied: set[int] = set()
    for slot, _kind, width in rows:
        for word in range(width):
            cell = slot + word
            if cell in occupied:
                raise SystemExit(f"{case_id} overlapping f128 slot word {cell} in rows={rows!r}")
            occupied.add(cell)


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_dir = out_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for rel_name, text in dict(case.get("support_files", {})).items():
        support_path = case_dir / rel_name
        support_path.parent.mkdir(parents=True, exist_ok=True)
        support_path.write_text(str(text), encoding="utf-8")

    source = str(case["source"])
    source_path = case_dir / "case.sio"
    elf_path = case_dir / "case.native_v2"
    machine_path = case_dir / "machine.json"
    log_path = case_dir / "compile.log"
    source_path.write_text(source, encoding="utf-8")

    rc, log = run_command(
        [
            str(compiler),
            "--native-v2-compile",
            str(source_path),
            "-o",
            str(elf_path),
            "--machine-module-json",
            str(machine_path),
        ],
        root,
        timeout_s,
    )
    log_path.write_text(log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case_id} native-v2 compile command failed rc={rc}; log={log_path}")
    if not machine_path.exists():
        raise SystemExit(f"{case_id} did not emit machine module JSON")

    module = load_machine_module(machine_path, case)
    fns = functions_by_name(module)
    callee = str(case["callee"])
    if callee not in fns:
        raise SystemExit(f"{case_id} missing callee {callee}")
    callee_fn = fns[callee]
    callee_index = int(callee_fn.get("index", -1))
    source_param_count = int(callee_fn.get("source_param_count", -1))
    source_f128_param_count = int(callee_fn.get("source_f128_param_count", -1))
    if source_param_count != int(case["expected_source_param_count"]):
        raise SystemExit(f"{case_id} source_param_count mismatch: {source_param_count}")
    if source_f128_param_count != int(case["expected_source_f128_param_count"]):
        raise SystemExit(f"{case_id} source_f128_param_count mismatch: {source_f128_param_count}")

    rows = f128_rows_for_function(module, callee_index)
    expected_rows = [[int(v) for v in row] for row in case["expected_f128_rows"]]
    if rows != expected_rows:
        raise SystemExit(f"{case_id} f128 rows mismatch: expected {expected_rows!r}, got {rows!r}")
    assert_non_overlapping(rows, case_id)

    return {
        "case_id": case_id,
        "callee": callee,
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, case_dir)),
        "machine_module_json_sha256": sha256_text(stable_json(module)),
        "machine_supported": bool(module.get("supported", False)),
        "machine_unsupported_detail": str(module.get("unsupported_detail") or ""),
        "source_param_count": source_param_count,
        "source_f128_param_count": source_f128_param_count,
        "expected_f128_rows": expected_rows,
        "observed_f128_rows": rows,
        "non_overlapping_f128_param_slots": True,
        "full_f128_execution_promoted": False,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [emit_case(root, compiler, out_dir, case, int(args.timeout)) for case in CASES]
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "target": "x86_64-linux",
        "case_id": "f128_parameter_slots_expand_to_non_overlapping_binary128_words",
        "case_count": len(cases),
        "cases": cases,
        "compiler_machine_module_exported": True,
        "f128_param_slot_layout_promoted": True,
        "f128_param_slots_non_overlapping": True,
        "f128_binary128_slot_kind": F128_SLOT_KIND,
        "f128_binary128_width_words": F128_WIDTH_WORDS,
        "f128_callee_add_sub_value_contract_helper_layout_promoted": True,
        "f128_full_execution_promoted": False,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_full_complete": False,
        "legacy_fallback_for_f128_param_layout": False,
        "roundtrip_contract": [
            "two_f128_parameters_lower_to_four_source_param_words",
            "f128_parameter_slots_are_low_word_rows_width_2",
            "consecutive_f128_parameters_do_not_overlap",
            "mixed_i64_f128_parameter_layout_preserves_non_overlap",
            "imported_f128_parameter_layout_matches_local_layout",
            "callee_f128_add_sub_value_contract_helper_preserves_non_overlapping_param_slots",
        ],
        "missing_full_obligations": [
            "f128 software-helper lowering with IEEE rounding and NaN/Inf contract",
            "generic f128 callee arithmetic over runtime values outside the finite add/sub value-contract helper",
            "external SysV f128 ABI/SRET and differential receipts",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path = out_dir / "madaros_v2_s5_f128_param_slot_layout.receipt.json"
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        "madaros-v2-s5-f128-param-slot-layout: "
        f"cases={receipt['case_count']} sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
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
    raise SystemExit(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
