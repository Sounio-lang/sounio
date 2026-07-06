#!/usr/bin/env python3
"""Emit a Madaros v2 S5.6 f128 arithmetic value-contract receipt.

This promotes a finite, exact binary128 arithmetic value-contract matrix
end-to-end. It deliberately does not promote generic IEEE helpers, NaN/Inf,
arbitrary decimal arithmetic, external SysV f128 ABI, SRET, or multi-arg shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_arithmetic_value_contract_receipt/0.1"
STAGE = "S5_6_F128_ARITHMETIC_VALUE_CONTRACT_NATIVE_MATERIALIZATION"
MACHINE_SCHEMA = "madaros.v2.s5.machine_module/0.1"


POSITIVE = [
    {
        "case_id": "f128_add_one_two_to_three",
        "source": """fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let y: f128 = 2.0 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "40008000000000000000000000000000",
        "expected_metadata": [1, 0, 30, 2, 1, 0],
    },
    {
        "case_id": "f128_mul_one_two_to_two",
        "source": """fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let y: f128 = 2.0 as f128
  let z: f128 = x * y
  let w: f128 = z
  0
}
""",
        "expected_hex": "40000000000000000000000000000000",
        "expected_metadata": [1, 0, 20, 2, 1, 0],
    },
    {
        "case_id": "f128_add_half_half_to_one",
        "source": """fn main() -> i64 {
  let x: f128 = 0.5 as f128
  let y: f128 = 0.5 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3fff0000000000000000000000000000",
        "expected_metadata": [1, 0, 10, 2, 1, 0],
    },
    {
        "case_id": "f128_div_one_two_to_half",
        "source": """fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let y: f128 = 2.0 as f128
  let z: f128 = x / y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3ffe0000000000000000000000000000",
        "expected_metadata": [1, 0, 5, 2, 1, 0],
    },
    {
        "case_id": "f128_chain_add_sub_to_one",
        "source": """fn main() -> i64 {
  let x: f128 = 1.0 as f128
  let y: f128 = 2.0 as f128
  let z: f128 = x + y
  let w: f128 = z - y
  let q: f128 = w
  0
}
""",
        "expected_hex": "3fff0000000000000000000000000000",
        "expected_metadata": [1, 0, 10, 2, 1, 0],
    },
    {
        "case_id": "f128_add_half_one_to_one_and_half",
        "source": """fn main() -> i64 {
  let x: f128 = 0.5 as f128
  let y: f128 = 1.0 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3fff8000000000000000000000000000",
        "expected_metadata": [1, 0, 15, 2, 1, 0],
    },
    {
        "case_id": "f128_mul_half_half_to_quarter",
        "source": """fn main() -> i64 {
  let x: f128 = 0.5 as f128
  let y: f128 = 0.5 as f128
  let z: f128 = x * y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3ffd0000000000000000000000000000",
        "expected_metadata": [1, 0, 25, 3, 2, 0],
    },
    {
        "case_id": "f128_mul_one_and_half_half_to_three_quarters",
        "source": """fn main() -> i64 {
  let x: f128 = 1.5 as f128
  let y: f128 = 0.5 as f128
  let z: f128 = x * y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3ffe8000000000000000000000000000",
        "expected_metadata": [1, 0, 75, 3, 2, 0],
    },
    {
        "case_id": "f128_add_quarter_one_to_one_and_quarter",
        "source": """fn main() -> i64 {
  let x: f128 = 0.25 as f128
  let y: f128 = 1.0 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3fff4000000000000000000000000000",
        "expected_metadata": [1, 0, 125, 3, 2, 0],
    },
    {
        "case_id": "f128_sub_half_one_to_negative_half",
        "source": """fn main() -> i64 {
  let x: f128 = 0.5 as f128
  let y: f128 = 1.0 as f128
  let z: f128 = x - y
  let w: f128 = z
  0
}
""",
        "expected_hex": "bffe0000000000000000000000000000",
        "expected_metadata": [-1, 0, 5, 2, 1, 0],
    },
    {
        "case_id": "f128_add_negative_half_one_to_half",
        "source": """fn main() -> i64 {
  let x: f128 = -0.5 as f128
  let y: f128 = 1.0 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "3ffe0000000000000000000000000000",
        "expected_metadata": [1, 0, 5, 2, 1, 0],
    },
    {
        "case_id": "f128_add_negative_half_negative_half_to_negative_one",
        "source": """fn main() -> i64 {
  let x: f128 = -0.5 as f128
  let y: f128 = -0.5 as f128
  let z: f128 = x + y
  let w: f128 = z
  0
}
""",
        "expected_hex": "bfff0000000000000000000000000000",
        "expected_metadata": [-1, 0, 10, 2, 1, 0],
    },
    {
        "case_id": "f128_mul_negative_half_half_to_negative_quarter",
        "source": """fn main() -> i64 {
  let x: f128 = -0.5 as f128
  let y: f128 = 0.5 as f128
  let z: f128 = x * y
  let w: f128 = z
  0
}
""",
        "expected_hex": "bffd0000000000000000000000000000",
        "expected_metadata": [-1, 0, 25, 3, 2, 0],
    },
    {
        "case_id": "f128_div_negative_one_two_to_negative_half",
        "source": """fn main() -> i64 {
  let x: f128 = -1.0 as f128
  let y: f128 = 2.0 as f128
  let z: f128 = x / y
  let w: f128 = z
  0
}
""",
        "expected_hex": "bffe0000000000000000000000000000",
        "expected_metadata": [-1, 0, 5, 2, 1, 0],
    },
]

NEGATIVE = [
    {
        "case_id": "f128_add_rounded_tenths_still_blocked",
        "source": """fn main() -> i64 {
  let x: f128 = 0.1 as f128
  let y: f128 = 0.2 as f128
  let z: f128 = x + y
  0
}
""",
        "expected_detail": "f128_arithmetic_pending",
    },
]


def root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pretty_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


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


def signed_i64(value: int) -> int:
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def u64_words_from_hex(hex_text: str) -> tuple[int, int]:
    bits = int(hex_text, 16)
    return (bits >> 64) & ((1 << 64) - 1), bits & ((1 << 64) - 1)


def mov_rax_imm_pattern(value: int) -> bytes:
    signed = signed_i64(value)
    if -(1 << 31) <= signed <= (1 << 31) - 1:
        return b"\x48\xc7\xc0" + (value & ((1 << 32) - 1)).to_bytes(4, "little", signed=False)
    return b"\x48\xb8" + (value & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)


def load_machine(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MACHINE_SCHEMA:
        raise SystemExit(f"bad MachineModule schema: {payload.get('schema')!r}")
    if payload.get("legacy_fallback") is not False:
        raise SystemExit("MachineModule must not use legacy fallback")
    return payload


def metadata_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    meta = module.get("f128_literal_metadata", {})
    for fn in meta.get("functions", []):
        for row in fn.get("rows", []):
            if isinstance(row, list) and len(row) >= 7:
                rows.append([int(x) for x in row])
    return rows


def emit_positive(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_dir = out_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    src = case_dir / "case.sio"
    elf = case_dir / "case.native_v2"
    mm = case_dir / "machine.json"
    src.write_text(str(case["source"]), encoding="utf-8")
    rc, log = run([str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)], root, timeout_s)
    (case_dir / "compile.log").write_text(log, encoding="utf-8")
    if rc != 0 or "native_v2_compile: emitted" not in log:
        raise SystemExit(f"{case['case_id']}: expected native-v2 ELF emission")
    if "SIGSEGV" in log or "Segmentation fault" in log or "legacy fallback" in log:
        raise SystemExit(f"{case['case_id']}: crash/fallback in compile log")
    os.chmod(elf, 0o755)
    run_rc, run_log = run([str(elf)], root, timeout_s)
    (case_dir / "run.log").write_text(run_log, encoding="utf-8")
    if run_rc != 0:
        raise SystemExit(f"{case['case_id']}: emitted ELF rc={run_rc}, expected 0")
    module = load_machine(mm)
    if module.get("supported") is not True or module.get("unsupported_detail") not in ("", None):
        raise SystemExit(f"{case['case_id']}: MachineModule must be supported")
    rows = metadata_rows(module)
    expected_metadata = list(case["expected_metadata"])
    if expected_metadata not in [row[1:7] for row in rows]:
        raise SystemExit(f"{case['case_id']}: missing result metadata {expected_metadata}")
    elf_bytes = elf.read_bytes()
    hi, lo = u64_words_from_hex(str(case["expected_hex"]))
    hi_pattern = mov_rax_imm_pattern(hi)
    lo_pattern = mov_rax_imm_pattern(lo)
    hi_found = hi_pattern in elf_bytes
    lo_found = lo_pattern in elf_bytes
    if not hi_found:
        raise SystemExit(f"{case['case_id']}: missing expected high-word immediate")
    if lo != 0 and not lo_found:
        raise SystemExit(f"{case['case_id']}: missing expected low-word immediate")
    return {
        "case_id": case["case_id"],
        "kind": "positive",
        "compile_rc": rc,
        "run_rc": run_rc,
        "machine_module_supported": True,
        "source_sha256": sha256_text(str(case["source"])),
        "elf_sha256": sha256_bytes(elf_bytes),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "expected_binary128_hex": case["expected_hex"],
        "expected_hi_u64": hi,
        "expected_hi_i64": signed_i64(hi),
        "expected_lo_u64": lo,
        "expected_lo_i64": signed_i64(lo),
        "expected_result_metadata": expected_metadata,
        "machine_module_metadata_rows": rows,
        "hi_mov_imm_pattern_hex": hi_pattern.hex(),
        "lo_mov_imm_pattern_hex": lo_pattern.hex(),
        "hi_mov_imm_pattern_found": hi_found,
        "lo_mov_imm_pattern_found": lo_found,
    }


def emit_negative(root: Path, compiler: Path, out_dir: Path, case: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    case_dir = out_dir / str(case["case_id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    src = case_dir / "case.sio"
    elf = case_dir / "case.native_v2"
    mm = case_dir / "machine.json"
    src.write_text(str(case["source"]), encoding="utf-8")
    rc, log = run([str(compiler), "--native-v2-compile", str(src), "-o", str(elf), "--machine-module-json", str(mm)], root, timeout_s)
    (case_dir / "compile.log").write_text(log, encoding="utf-8")
    if "native_v2_compile: emitted" in log or elf.exists():
        raise SystemExit(f"{case['case_id']}: unexpectedly emitted executable")
    module = load_machine(mm)
    detail = str(case["expected_detail"])
    if module.get("supported") is not False or module.get("unsupported_detail") != detail:
        raise SystemExit(f"{case['case_id']}: expected MachineModule detail {detail!r}")
    return {
        "case_id": case["case_id"],
        "kind": "negative",
        "compile_rc": rc,
        "expected_detail": detail,
        "machine_module_supported": False,
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "source_sha256": sha256_text(str(case["source"])),
        "machine_module_sha256": sha256_text(stable_json(module)),
    }


def emit(args: argparse.Namespace) -> None:
    root = root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [emit_positive(root, compiler, out_dir, c, args.timeout_s) for c in POSITIVE]
    cases.extend(emit_negative(root, compiler, out_dir, c, args.timeout_s) for c in NEGATIVE)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE,
        "case_id": "s5_6_f128_arithmetic_value_contract",
        "case_count": len(cases),
        "positive_case_count": len(POSITIVE),
        "negative_case_count": len(NEGATIVE),
        "f128_arithmetic_value_contract_promoted": True,
        "f128_native_arithmetic_promoted": True,
        "f128_native_ieee_binary128_materialization_promoted": False,
        "f128_native_general_decimal_binary128_materialization_promoted": False,
        "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
        "f128_software_helpers_promoted": False,
        "f128_nan_inf_contract_promoted": False,
        "f128_external_sysv_abi_promoted": False,
        "f128_sret_abi_promoted": False,
        "f128_native_call_abi_promoted": False,
        "f128_native_return_abi_promoted": False,
        "f128_promoted": False,
        "contract_scope": [
            "finite exact binary128 value-contract arithmetic only",
            "finite exact signed decimal-tenths plus quarter matrix materializes as binary128 words",
            "single-chain arithmetic preserves compiler value-kind metadata",
            "unsupported f128 arithmetic remains fail-closed",
        ],
        "cases": cases,
    }
    canonical = stable_json(payload)
    payload["receipt_sha256"] = sha256_text(canonical)
    receipt = out_dir / "madaros_v2_s5_f128_arithmetic_value_contract.receipt.json"
    receipt.write_text(pretty_json(payload), encoding="utf-8")
    print(f"[madaros-v2-s5-f128-arithmetic-value-contract] receipt={receipt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    emit_p = sub.add_parser("emit")
    emit_p.add_argument("--compiler", default=str(root_from_script() / "bin/madaros"))
    emit_p.add_argument("--out-dir", required=True)
    emit_p.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args()
    if args.cmd == "emit":
        emit(args)


if __name__ == "__main__":
    main()
