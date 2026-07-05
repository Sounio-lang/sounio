#!/usr/bin/env python3
"""Emit a Madaros v2 S5 f128 binary128 value-contract receipt.

This closes the value-representation prerequisite between parser-side decimal
metadata and future f128 IR/MIR/ABI lowering. It computes IEEE-754 binary128
finite decimal encodings with exact rational arithmetic and round-to-nearest,
ties-to-even. It deliberately does not promote native-v2 f128 execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA = "madaros.v2.s5.f128_binary128_value_receipt/0.1"
STAGE_CONTRACT_LEVEL = "S5_F128_BINARY128_VALUE_CONTRACT_PROMOTED_NOT_EXECUTION"

BINARY128_PRECISION = 113
BINARY128_FRACTION_BITS = 112
BINARY128_EXP_BIAS = 16383
BINARY128_EMIN = -16382
BINARY128_EXP_MAX = 0x7FFF


@dataclass(frozen=True)
class Case:
    case_id: str
    literal: str
    expected_hex: str | None = None
    expected_class: str = "normal"
    exact_decimal: bool = False


CASES: list[Case] = [
    Case("positive_zero", "0.0", "00000000000000000000000000000000", "zero", True),
    Case("negative_zero", "-0.0", "80000000000000000000000000000000", "zero", True),
    Case("one", "1.0", "3fff0000000000000000000000000000", "normal", True),
    Case("half", "0.5", "3ffe0000000000000000000000000000", "normal", True),
    Case("two", "2.0", "40000000000000000000000000000000", "normal", True),
    Case("smallest_normal", "3.36210314311209350626267781732175260259807934484647e-4932", None, "normal", False),
    Case("one_tenth_rounded", "0.1", None, "normal", False),
    Case("high_precision_probe", "1.2345678901234567890123456789012345", None, "normal", False),
]

PROBE_SOURCE = (
    "fn main() -> i64 {\n"
    "    let x: f128 = 1.2345678901234567890123456789012345 as f128\n"
    "    let y: f128 = 0.1 as f128\n"
    "    0\n"
    "}\n"
)


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


def canonical_roundtrip(payload: dict[str, Any]) -> tuple[str, str]:
    first = stable_json(payload)
    second = stable_json(json.loads(first))
    if first != second:
        raise SystemExit("f128 binary128 value receipt canonical JSON roundtrip changed bytes")
    return first, sha256_text(first)


def pow2(exp: int) -> Fraction:
    if exp >= 0:
        return Fraction(1 << exp, 1)
    return Fraction(1, 1 << (-exp))


def parse_decimal_fraction(literal: str) -> tuple[int, Fraction, dict[str, int]]:
    text = literal.strip()
    sign = 0
    if text.startswith("-"):
        sign = 1
        text = text[1:]
    elif text.startswith("+"):
        text = text[1:]

    exp10 = 0
    if "e" in text or "E" in text:
        parts = text.replace("E", "e").split("e", 1)
        text = parts[0]
        exp10 = int(parts[1])

    if "." in text:
        int_part, frac_part = text.split(".", 1)
    else:
        int_part, frac_part = text, ""
    digits_text = (int_part + frac_part).lstrip("0")
    digit_count = len(int_part + frac_part)
    scale10 = len(frac_part) - exp10
    if digits_text == "":
        value = Fraction(0, 1)
    else:
        numerator = int(digits_text)
        if scale10 >= 0:
            value = Fraction(numerator, 10**scale10)
        else:
            value = Fraction(numerator * (10 ** (-scale10)), 1)
    metadata = {
        "decimal_sign": -1 if sign else 1,
        "decimal_digit_count": digit_count,
        "decimal_scale10": scale10,
        "decimal_truncated_digits": max(0, digit_count - 36),
    }
    return sign, value, metadata


def floor_log2_fraction(x: Fraction) -> int:
    if x <= 0:
        raise ValueError("floor_log2_fraction requires x > 0")
    n = x.numerator
    d = x.denominator
    e = n.bit_length() - d.bit_length()
    if x < pow2(e):
        e -= 1
    return e


def round_fraction_to_int_nearest_even(x: Fraction) -> tuple[int, bool, bool]:
    q, r = divmod(x.numerator, x.denominator)
    twice_r = r * 2
    if twice_r < x.denominator:
        return q, False, r == 0
    if twice_r > x.denominator:
        return q + 1, False, False
    return (q + (q & 1), True, False)


def binary128_from_decimal(literal: str) -> dict[str, Any]:
    sign, value, decimal_metadata = parse_decimal_fraction(literal)
    if value == 0:
        bits = sign << 127
        return {
            "literal": literal,
            "class": "zero",
            "sign": sign,
            "exponent_field": 0,
            "fraction_hi": 0,
            "fraction_lo": 0,
            "hex": f"{bits:032x}",
            "rounded": False,
            "tie_to_even": False,
            "exact": True,
            **decimal_metadata,
        }

    e = floor_log2_fraction(value)
    if e >= BINARY128_EMIN:
        scaled = value / pow2(e - BINARY128_FRACTION_BITS)
        significand, tie_to_even, exact = round_fraction_to_int_nearest_even(scaled)
        if significand == (1 << BINARY128_PRECISION):
            significand >>= 1
            e += 1
        exponent_field = e + BINARY128_EXP_BIAS
        if exponent_field >= BINARY128_EXP_MAX:
            raise SystemExit(f"binary128 overflow for finite receipt case: {literal}")
        fraction = significand - (1 << BINARY128_FRACTION_BITS)
        cls = "normal"
    else:
        scaled = value / pow2(BINARY128_EMIN - BINARY128_FRACTION_BITS)
        fraction, tie_to_even, exact = round_fraction_to_int_nearest_even(scaled)
        if fraction == (1 << BINARY128_FRACTION_BITS):
            exponent_field = 1
            fraction = 0
            cls = "normal"
        else:
            exponent_field = 0
            cls = "subnormal"

    if fraction < 0 or fraction >= (1 << BINARY128_FRACTION_BITS):
        raise SystemExit(f"bad binary128 fraction for {literal}: {fraction}")
    bits = (sign << 127) | (exponent_field << BINARY128_FRACTION_BITS) | fraction
    return {
        "literal": literal,
        "class": cls,
        "sign": sign,
        "exponent_field": exponent_field,
        "fraction_hi": (fraction >> 64) & ((1 << 48) - 1),
        "fraction_lo": fraction & ((1 << 64) - 1),
        "hex": f"{bits:032x}",
        "rounded": not exact,
        "tie_to_even": tie_to_even,
        "exact": exact,
        **decimal_metadata,
    }


def emit(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = out_dir / "f128_binary128_value_probe.sio"
    check_log_path = out_dir / "f128_binary128_value_probe.check.log"
    source_path.write_text(PROBE_SOURCE, encoding="utf-8")
    check_rc, check_stdout, check_stderr = run_command([str(compiler), "check", str(source_path)], root, args.timeout)
    check_log = check_stdout + check_stderr
    check_log_path.write_text(check_log, encoding="utf-8")
    if check_rc != 0 or "check: OK" not in check_log:
        raise SystemExit(f"f128 binary128 value probe must pass frontend check; log={check_log_path}")

    case_rows: list[dict[str, Any]] = []
    for case in CASES:
        row = binary128_from_decimal(case.literal)
        row["case_id"] = case.case_id
        if case.expected_hex is not None and row["hex"] != case.expected_hex:
            raise SystemExit(f"{case.case_id} expected hex {case.expected_hex}, got {row['hex']}")
        if row["class"] != case.expected_class:
            raise SystemExit(f"{case.case_id} expected class {case.expected_class}, got {row['class']}")
        if case.exact_decimal and row["exact"] is not True:
            raise SystemExit(f"{case.case_id} expected exact binary128 representation")
        case_rows.append(row)

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "f128_decimal_to_binary128_value_contract",
        "case_count": len(case_rows),
        "rounding_mode": "roundTiesToEven",
        "target_format": {
            "name": "IEEE-754 binary128",
            "precision_bits": BINARY128_PRECISION,
            "fraction_bits": BINARY128_FRACTION_BITS,
            "exponent_bits": 15,
            "exponent_bias": BINARY128_EXP_BIAS,
            "emin": BINARY128_EMIN,
        },
        "cases": case_rows,
        "probe_source": PROBE_SOURCE,
        "probe_source_sha256": sha256_text(PROBE_SOURCE),
        "probe_check_rc": check_rc,
        "probe_check_log_sha256": sha256_text(check_log),
        "f128_binary128_value_contract_complete": True,
        "f128_binary128_round_ties_to_even_recorded": True,
        "f128_binary128_sign_exponent_fraction_recorded": True,
        "f128_binary128_anchor_cases_verified": True,
        "f128_binary128_decimal_metadata_bridge_recorded": True,
        "f128_promoted": False,
        "s5_ready": False,
        "s5_implemented": False,
        "s5_full_complete": False,
        "roundtrip_contract": [
            "finite_decimal_literal_to_binary128_uses_exact_rational_arithmetic",
            "binary128_rounding_mode_is_roundTiesToEven",
            "zero_sign_is_preserved",
            "normal_anchor_encodings_for_0_5_1_0_2_0_are_verified",
            "high_precision_decimal_probe_frontend_checks_as_f128",
            "receipt_records_sign_exponent_and_112_fraction_bits_as_two_limbs",
            "receipt_does_not_promote_native_v2_f128_execution",
        ],
        "missing_full_obligations": [
            "f128 IR opcodes and constructors",
            "f128 MachineIR lowering that emits slot kind 3 with two 64-bit limbs",
            "f128 SysV ABI classification and call-return signature metadata",
            "f128 software helper lowering with IEEE rounding and NaN/Inf contract",
            "f128 native-v2 execution and differential receipts",
        ],
    }
    receipt["receipt_sha256"] = sha256_text(stable_json(receipt))
    _, canonical_sha = canonical_roundtrip(receipt)
    receipt["canonical_roundtrip_sha256"] = canonical_sha
    receipt_path = out_dir / "madaros_v2_s5_f128_binary128_value.receipt.json"
    receipt_path.write_text(pretty_json(receipt), encoding="utf-8")
    print(
        "madaros-v2-s5-f128-binary128-value: "
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
    emit_p.set_defaults(func=emit)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
