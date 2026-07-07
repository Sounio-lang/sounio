#!/usr/bin/env python3
"""Emit the S5.21 f128 aggregate-field/SRET payload receipt.

This promotes the native-v2 layout and field access contract for f128 values
inside compiler-owned aggregates: a f128 field occupies two machine words,
following fields do not overlap it, FieldSet stores both words, FieldGet loads
both words, and the payload remains source-observable after an internal SRET
return. It does not promote external SysV f128 ABI, arbitrary decimal
materialization, or full IEEE arithmetic.
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


SCHEMA = "madaros.v2.s5.f128_sret_field_payload_receipt/0.1"
STAGE = "S5_21_F128_AGGREGATE_FIELD_PAYLOAD_AND_SRET_LAYOUT"
CASE_ID = "f128_aggregate_field_payload_width2_sret_source_observable"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"

MIR_OP_FIELD_LOAD = 117
MIR_OP_FIELD_STORE = 118
MIR_OP_F128_VC_BINOP = 131
MIR_OP_F128_ORDERED_CMP = 132


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    expected_exit: int
    requires_sret: bool
    requires_f128_arith: bool = False


CASES: list[Case] = [
    Case(
        "local_struct_f128_field_roundtrip_tail_nonoverlap",
        """struct BoxF128 { tag: i64, x: f128, tail: i64 }
fn main() -> i64 {
  let b = BoxF128 { tag: 40, x: 1.0 as f128, tail: 41 }
  if b.tail == 41 {
    if b.x == 1.0 as f128 { return 0 }
  }
  11
}
""",
        0,
        False,
    ),
    Case(
        "sret_f128_field_eq_one_true",
        """struct BoxF128 { tag: i64, x: f128, tail: i64 }
fn make(v: f128, n: i64) -> BoxF128 { BoxF128 { tag: n, x: v, tail: n + 1 } }
fn main() -> i64 {
  let b = make(1.0 as f128, 40)
  if b.tail == 41 {
    if b.x == 1.0 as f128 { return 0 }
  }
  12
}
""",
        0,
        True,
    ),
    Case(
        "sret_f128_field_add_decimal_value_contract",
        """struct BoxF128 { tag: i64, x: f128, tail: i64 }
fn make(v: f128, n: i64) -> BoxF128 { BoxF128 { tag: n, x: v, tail: n + 1 } }
fn main() -> i64 {
  let b = make(0.1 as f128, 40)
  let y: f128 = b.x + 0.2 as f128
  if b.tail == 41 {
    if y == 0.3 as f128 { return 0 }
  }
  13
}
""",
        0,
        True,
        True,
    ),
    Case(
        "sret_f128_field_tail_after_f128_nonoverlap",
        """struct BoxF128 { tag: i64, x: f128, tail: i64 }
fn make(v: f128, n: i64) -> BoxF128 { BoxF128 { tag: n, x: v, tail: n + 7 } }
fn main() -> i64 {
  let b = make(2.0 as f128, 50)
  if b.tag == 50 {
    if b.tail == 57 {
      if b.x == 2.0 as f128 { return 0 }
    }
  }
  14
}
""",
        0,
        True,
    ),
    Case(
        "sret_two_f128_fields_and_tail_roundtrip",
        """struct PairF128 { a: f128, b: f128, tail: i64 }
fn make(x: f128, y: f128, n: i64) -> PairF128 { PairF128 { a: x, b: y, tail: n } }
fn main() -> i64 {
  let p = make(1.0 as f128, 2.0 as f128, 43)
  if p.a == 1.0 as f128 {
    if p.b == 2.0 as f128 {
      if p.tail == 43 { return 0 }
    }
  }
  15
}
""",
        0,
        True,
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


def normalize_log(text: str, out_dir: Path) -> str:
    return text.replace(str(out_dir), "<OUT_DIR>")


def run(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int, str]:
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
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode, proc.stdout or ""


def load_machine(path: Path) -> dict[str, Any]:
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
    return payload


def flatten_instrs(module: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for fn in module.get("functions", []):
        for instr in fn.get("instrs", []):
            rows.append(instr)
    return rows


def function_names(module: dict[str, Any]) -> list[str]:
    return [str(fn.get("name", "")) for fn in module.get("functions", []) if fn.get("name")]


def module_shape(module: dict[str, Any], case: Case) -> dict[str, Any]:
    instrs = flatten_instrs(module)
    opcodes = [int(row[0]) for row in instrs if row]
    field_load_count = opcodes.count(MIR_OP_FIELD_LOAD)
    field_store_count = opcodes.count(MIR_OP_FIELD_STORE)
    f128_cmp_count = opcodes.count(MIR_OP_F128_ORDERED_CMP)
    f128_vc_binop_count = opcodes.count(MIR_OP_F128_VC_BINOP)
    if field_load_count <= 0 or field_store_count <= 0:
        raise SystemExit(f"{case.case_id}: expected MachineIR field load/store opcodes")
    if f128_cmp_count <= 0:
        raise SystemExit(f"{case.case_id}: expected source-observable f128 ordered compare")
    if case.requires_f128_arith and f128_vc_binop_count <= 0:
        raise SystemExit(f"{case.case_id}: expected f128 value-contract binop after field load")
    sret_functions = [
        str(fn.get("name", ""))
        for fn in module.get("functions", [])
        if bool(fn.get("source_is_sret"))
    ]
    if case.requires_sret and "make" not in sret_functions:
        raise SystemExit(f"{case.case_id}: expected make to lower as source SRET")
    return {
        "function_names": function_names(module),
        "sret_functions": sret_functions,
        "field_load_count": field_load_count,
        "field_store_count": field_store_count,
        "f128_ordered_cmp_count": f128_cmp_count,
        "f128_value_contract_binop_count": f128_vc_binop_count,
        "machine_function_count": len(module.get("functions", [])),
    }


def compile_and_run(root: Path, compiler: Path, out_dir: Path, case: Case, timeout_s: int) -> dict[str, Any]:
    src = out_dir / f"{case.case_id}.sio"
    elf = out_dir / f"{case.case_id}.native_v2"
    mm = out_dir / f"{case.case_id}.machine_module.json"
    compile_log_path = out_dir / f"{case.case_id}.native_v2.log"
    run_log_path = out_dir / f"{case.case_id}.run.log"
    src.write_text(case.source, encoding="utf-8")

    rc, compile_log = run(
        [str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)],
        root,
        timeout_s,
    )
    compile_log_path.write_text(compile_log, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{case.case_id}: native-v2 compile failed rc={rc}\n{compile_log}")
    if not elf.exists():
        raise SystemExit(f"{case.case_id}: native-v2 compile did not emit ELF")
    if not mm.exists():
        raise SystemExit(f"{case.case_id}: native-v2 compile did not emit MachineModule JSON")

    module = load_machine(mm)
    shape = module_shape(module, case)

    os.chmod(elf, 0o755)
    run_rc, run_log = run([str(elf)], root, timeout_s)
    run_log_path.write_text(run_log, encoding="utf-8")
    if run_rc != case.expected_exit:
        raise SystemExit(f"{case.case_id}: expected rc={case.expected_exit}, got rc={run_rc}\n{run_log}")

    return {
        "case_id": case.case_id,
        "expected_exit": case.expected_exit,
        "actual_exit": run_rc,
        "requires_sret": case.requires_sret,
        "requires_f128_arith": case.requires_f128_arith,
        "source_sha256": sha256_text(case.source),
        "compile_log_sha256": sha256_text(normalize_log(compile_log, out_dir)),
        "run_log_sha256": sha256_text(normalize_log(run_log, out_dir)),
        "machine_module_json_sha256": sha256_bytes(mm.read_bytes()),
        "elf_sha256": sha256_bytes(elf.read_bytes()),
        "machine_shape": shape,
    }


def emit_receipt(compiler: Path, out_dir: Path, timeout_s: int) -> dict[str, Any]:
    root = repo_root_from_script()
    compiler_path = compiler if compiler.is_absolute() else root / compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [compile_and_run(root, compiler_path, out_dir, case, timeout_s) for case in CASES]

    if len(cases) != 5:
        raise SystemExit("f128 SRET field payload receipt must contain 5 cases")
    if not any(row["requires_f128_arith"] for row in cases):
        raise SystemExit("f128 SRET field payload receipt must include arithmetic after field load")
    if sum(1 for row in cases if row["requires_sret"]) < 4:
        raise SystemExit("f128 SRET field payload receipt must include SRET cases")
    if not all(row["actual_exit"] == row["expected_exit"] for row in cases):
        raise SystemExit("f128 SRET field payload receipt has a failing case")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "stage_contract_level": STAGE,
        "case_id": CASE_ID,
        "status": "pass",
        "compiler": str(compiler),
        "case_count": len(cases),
        "claims": {
            "f128_aggregate_field_layout_width2_promoted": True,
            "f128_aggregate_field_payload_load_store_width2_promoted": True,
            "f128_sret_field_payload_source_observable_promoted": True,
            "f128_sret_field_tail_nonoverlap_promoted": True,
            "f128_sret_two_f128_fields_source_observable_promoted": True,
            "f128_field_payload_value_contract_arithmetic_promoted": True,
            "f128_external_sysv_abi_promoted": False,
            "f128_arbitrary_decimal_binary128_materialization_promoted": False,
            "f128_full_ieee_arithmetic_promoted": False,
            "f128_promoted": False,
        },
        "cases": cases,
    }
    receipt_sha = sha256_text(stable_json(payload))
    payload["receipt_sha256"] = receipt_sha
    receipt_path = out_dir / "madaros_v2_s5_f128_sret_field_payload.receipt.json"
    receipt_path.write_text(pretty_json(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--compiler", required=True, type=Path)
    emit.add_argument("--out-dir", required=True, type=Path)
    emit.add_argument("--timeout-s", type=int, default=60)
    args = parser.parse_args()
    if args.cmd == "emit":
        payload = emit_receipt(args.compiler, args.out_dir, args.timeout_s)
        print(pretty_json(payload), end="")
        return 0
    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
