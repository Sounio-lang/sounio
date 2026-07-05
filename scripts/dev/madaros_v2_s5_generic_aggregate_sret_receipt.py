#!/usr/bin/env python3
"""Emit a Madaros v2 S5 generic aggregate SRET receipt.

This receipt closes the shape/layout part of aggregate-return SRET coverage:
non-Big names, non-f0/f1/f2 field names, 2/4/9-field layouts, and the same
wide layout through local, imported, and method paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "madaros.v2.s5.generic_aggregate_sret_receipt/0.1"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"
STAGE_CONTRACT_LEVEL = "S5_GENERIC_AGGREGATE_SRET_LAYOUT_PROMOTED"

MIR_OP_ALLOC = 116
MIR_OP_FIELD_LOAD = 117
MIR_OP_FIELD_STORE = 118


LOCAL_PAIR_SOURCE = """struct Pair {
    left: i64,
    right: i64,
}
fn make(a: i64, b: i64) -> Pair { Pair { left: a, right: b } }
fn main() -> i64 {
    let p = make(9, 14)
    p.left + p.right
}
"""

LOCAL_QUAD_SOURCE = """struct Quad {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}
fn make(a: i64, b: i64, c: i64, d: i64) -> Quad { Quad { a: a, b: b, c: c, d: d } }
fn main() -> i64 {
    let q = make(3, 5, 7, 11)
    q.a + q.b + q.c + q.d
}
"""

LOCAL_WIDE9_SOURCE = """struct Wide9 {
    a0: i64, a1: i64, a2: i64,
    a3: i64, a4: i64, a5: i64,
    a6: i64, a7: i64, a8: i64,
}
fn make() -> Wide9 {
    Wide9 { a0: 1, a1: 2, a2: 3, a3: 4, a4: 5, a5: 6, a6: 7, a7: 8, a8: 9 }
}
fn main() -> i64 {
    let w = make()
    w.a0 + w.a1 + w.a2 + w.a3 + w.a4 + w.a5 + w.a6 + w.a7 + w.a8
}
"""

IMPORTED_WIDE9_LIB = """pub struct Wide9 {
    a0: i64, a1: i64, a2: i64,
    a3: i64, a4: i64, a5: i64,
    a6: i64, a7: i64, a8: i64,
}
pub fn make() -> Wide9 {
    Wide9 { a0: 1, a1: 2, a2: 3, a3: 4, a4: 5, a5: 6, a6: 7, a7: 8, a8: 9 }
}
"""

IMPORTED_WIDE9_MAIN = """use imported_generic_wide9_lib::{Wide9, make}
fn main() -> i64 {
    let w = make()
    w.a0 + w.a1 + w.a2 + w.a3 + w.a4 + w.a5 + w.a6 + w.a7 + w.a8
}
"""

METHOD_WIDE9_SOURCE = """struct Wide9 {
    a0: i64, a1: i64, a2: i64,
    a3: i64, a4: i64, a5: i64,
    a6: i64, a7: i64, a8: i64,
}
struct Maker { seed: i64 }
impl Maker {
    fn make(self) -> Wide9 {
        Wide9 { a0: self.seed, a1: 2, a2: 3, a3: 4, a4: 5, a5: 6, a6: 7, a7: 8, a8: 9 }
    }
}
fn main() -> i64 {
    let m = Maker { seed: 1 }
    let w = m.make()
    w.a0 + w.a1 + w.a2 + w.a3 + w.a4 + w.a5 + w.a6 + w.a7 + w.a8
}
"""

CASES: list[dict[str, Any]] = [
    {
        "case_id": "source_sret_generic_pair2_return_23",
        "source": LOCAL_PAIR_SOURCE,
        "expected_exit": 23,
        "aggregate_type": "Pair",
        "field_count": 2,
        "declared_layout_bytes": 16,
        "expected_alloc_bytes": 64,
        "expected_functions": ["make", "main"],
        "path_kind": "local",
    },
    {
        "case_id": "source_sret_generic_quad4_return_26",
        "source": LOCAL_QUAD_SOURCE,
        "expected_exit": 26,
        "aggregate_type": "Quad",
        "field_count": 4,
        "declared_layout_bytes": 32,
        "expected_alloc_bytes": 64,
        "expected_functions": ["make", "main"],
        "path_kind": "local",
    },
    {
        "case_id": "source_sret_generic_wide9_return_45",
        "source": LOCAL_WIDE9_SOURCE,
        "expected_exit": 45,
        "aggregate_type": "Wide9",
        "field_count": 9,
        "declared_layout_bytes": 72,
        "expected_alloc_bytes": 72,
        "expected_functions": ["make", "main"],
        "path_kind": "local",
    },
    {
        "case_id": "imported_sret_generic_wide9_return_45",
        "source": IMPORTED_WIDE9_MAIN,
        "lib_source": IMPORTED_WIDE9_LIB,
        "lib_name": "imported_generic_wide9_lib",
        "expected_exit": 45,
        "aggregate_type": "Wide9",
        "field_count": 9,
        "declared_layout_bytes": 72,
        "expected_alloc_bytes": 72,
        "expected_functions": ["make", "main"],
        "path_kind": "imported",
    },
    {
        "case_id": "method_sret_generic_wide9_return_45",
        "source": METHOD_WIDE9_SOURCE,
        "expected_exit": 45,
        "aggregate_type": "Wide9",
        "field_count": 9,
        "declared_layout_bytes": 72,
        "expected_alloc_bytes": 72,
        "expected_functions": ["Maker_make", "main"],
        "path_kind": "method",
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
        raise SystemExit("generic aggregate SRET receipt canonical JSON roundtrip changed bytes")
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
    payload["machine_module_json_sha256"] = sha256_text(stable_json(payload))
    return payload


def functions_by_name(module: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for fn in module.get("functions", []):
        name = str(fn.get("name", ""))
        if name:
            result.setdefault(name, []).append(fn)
    return result


def instrs(fn: dict[str, Any], opcode: int) -> list[list[Any]]:
    return [instr for instr in fn.get("instrs", []) if int(instr[0]) == opcode]


def alloc_bytes(fn: dict[str, Any]) -> list[int]:
    return [int(instr[4]) for instr in instrs(fn, MIR_OP_ALLOC)]


def field_indices(fn: dict[str, Any], opcode: int) -> list[int]:
    return [int(instr[8]) for instr in instrs(fn, opcode)]


def validate_machine_shape(module: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    by_name = functions_by_name(module)
    for name in case["expected_functions"]:
        if name not in by_name:
            raise SystemExit(f"{case['case_id']} missing MachineModule function {name}; got {sorted(by_name)}")
    main_fn = by_name["main"][0]
    callee_name = "Maker_make" if case["path_kind"] == "method" else "make"
    callee_fn = by_name[callee_name][-1]
    main_allocs = alloc_bytes(main_fn)
    callee_allocs = alloc_bytes(callee_fn)
    expected_bytes = int(case["expected_alloc_bytes"])
    field_count = int(case["field_count"])
    expected_fields = list(range(field_count))
    if expected_bytes not in main_allocs:
        raise SystemExit(f"{case['case_id']} main missing layout-derived SRET alloc {expected_bytes}; allocs={main_allocs}")
    if expected_bytes not in callee_allocs:
        raise SystemExit(f"{case['case_id']} callee missing layout-derived literal alloc {expected_bytes}; allocs={callee_allocs}")
    if 64 in main_allocs and expected_bytes != 64:
        raise SystemExit(f"{case['case_id']} main still uses fixed 64-byte SRET alloc; allocs={main_allocs}")
    callee_field_stores = field_indices(callee_fn, MIR_OP_FIELD_STORE)
    for idx in expected_fields:
        if idx not in callee_field_stores:
            raise SystemExit(f"{case['case_id']} callee did not store field index {idx}; stores={callee_field_stores}")
    main_field_loads = field_indices(main_fn, MIR_OP_FIELD_LOAD)
    if case["field_count"] <= 4:
        for idx in expected_fields:
            if idx not in main_field_loads:
                raise SystemExit(f"{case['case_id']} main did not load field index {idx}; loads={main_field_loads}")
    else:
        if main_field_loads != expected_fields:
            raise SystemExit(f"{case['case_id']} main must load all Wide9 fields in order; loads={main_field_loads}")
    return {
        "function_names": sorted(by_name),
        "main_alloc_bytes": main_allocs,
        "callee_alloc_bytes": callee_allocs,
        "expected_alloc_bytes": expected_bytes,
        "main_field_load_indices": main_field_loads,
        "callee_field_store_indices": callee_field_stores,
        "main_instr_count": int(main_fn.get("instr_count", -1)),
        "callee_instr_count": int(callee_fn.get("instr_count", -1)),
        "callee_source_is_sret": int(callee_fn.get("source_is_sret", 0)),
    }


def emit_case(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_id = str(case["case_id"])
    source_text = str(case["source"])
    source_path = out_dir / f"{case_id}.sio"
    if case.get("lib_source"):
        lib_path = out_dir / f"{case['lib_name']}.sio"
        lib_path.write_text(str(case["lib_source"]), encoding="utf-8")
    else:
        lib_path = None
    elf_path = out_dir / f"{case_id}.native_v2"
    mm_path = out_dir / f"{case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case_id}.compile.log"
    stdout_path = out_dir / f"{case_id}.stdout"
    stderr_path = out_dir / f"{case_id}.stderr"
    source_path.write_text(source_text, encoding="utf-8")
    rc, compile_stdout, compile_stderr = run_command(
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
    if rc != 0:
        raise SystemExit(f"generic aggregate SRET compile failed for {case_id} rc={rc}; log={compile_log_path}")
    elf_path.chmod(elf_path.stat().st_mode | 0o111)
    actual_exit, stdout, stderr = run_binary(elf_path, timeout_s)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    if actual_exit != int(case["expected_exit"]):
        raise SystemExit(f"{case_id} expected exit {case['expected_exit']}, got {actual_exit}")
    module = load_machine_module(mm_path)
    machine_shape = validate_machine_shape(module, case)
    row = {
        "case_id": case_id,
        "path_kind": case["path_kind"],
        "source": source_path.name,
        "lib_source": lib_path.name if lib_path else "",
        "aggregate_type": case["aggregate_type"],
        "field_count": int(case["field_count"]),
        "declared_layout_bytes": int(case["declared_layout_bytes"]),
        "expected_alloc_bytes": int(case["expected_alloc_bytes"]),
        "expected_exit": int(case["expected_exit"]),
        "actual_exit": actual_exit,
        "source_sha256": sha256_text(source_text),
        "lib_source_sha256": sha256_text(str(case["lib_source"])) if case.get("lib_source") else "",
        "elf_sha256": sha256_bytes(elf_path.read_bytes()),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "machine_module_path": mm_path.name,
        "machine_module_json_sha256": module["machine_module_json_sha256"],
        "machine_shape": machine_shape,
    }
    return row


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "madaros_v2_s5_generic_aggregate_sret.receipt.json"
    case_results = [emit_case(root, compiler, out_dir, case, args.timeout) for case in CASES]
    receipt: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "target": "x86_64-linux",
        "case_id": "generic_aggregate_sret_layout",
        "case_count": len(case_results),
        "cases": case_results,
        "s5_generic_aggregate_sret_layout_complete": True,
        "generic_aggregate_return_promoted": True,
        "generic_aggregate_local_layout_promoted": True,
        "generic_aggregate_imported_layout_promoted": True,
        "generic_aggregate_method_layout_promoted": True,
        "layout_derived_sret_alloc_promoted": True,
        "wide9_sret_alloc_72_bytes_promoted": True,
        "compiler_machine_module_exported": True,
        "real_program_mir_emitted": True,
        "real_abi_layout_emitted": True,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "pair2_local_sret_preserves_two_declared_fields_with_minimum_machine_slot",
            "quad4_local_sret_preserves_four_declared_fields_with_minimum_machine_slot",
            "wide9_local_sret_uses_72_byte_layout_alloc_not_fixed_64",
            "wide9_imported_sret_uses_72_byte_layout_alloc_not_fixed_64",
            "wide9_method_sret_uses_72_byte_layout_alloc_not_fixed_64",
            "non_Big_non_f0_field_names_are_preserved_by_field_indices",
            "native_elves_return_expected_discriminators",
        ],
        "missing_full_obligations": [
            "f128 numeric tower width receipts",
            "diagnostics and fallback semantics for unsupported layouts and numeric widths",
            "differential native-v2 vs interpreter/lean_single validation where available",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        f"madaros-v2-s5-generic-aggregate-sret: cases={receipt['case_count']} "
        f"sha={receipt['receipt_sha256'][:12]} receipt={receipt_path}"
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
    emit_p.set_defaults(func=emit)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
