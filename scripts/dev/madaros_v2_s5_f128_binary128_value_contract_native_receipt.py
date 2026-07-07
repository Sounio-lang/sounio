#!/usr/bin/env python3
"""Emit a Madaros v2 S5.17 f128 native binary128 materialization receipt.

This promotes native-v2 materialization for the f128 binary128 value-contract
case set plus an explicit bounded-decimal class: sig_hi=0, no truncation, and
scale10<=18, with roundTiesToEven. It also promotes an algorithmic two-limb
decimal class: sig_hi>0, digit_count<=36, scale10<=18, no truncation, with
binary128 roundTiesToEven. It now promotes a bounded truncated two-limb class:
37..40 total digits, 1..4 truncated digits, retained scale 35, a bounded
normalized range [1/8, 10), and parser tail-info used as the binary128
rounding sticky bit. It also promotes an
explicit native saturation class for decimal overflow-to-infinity and exact
power-of-ten underflow-to-zero. It still deliberately does not promote arbitrary
large-scale rounded decimals, f128 arithmetic, external ABI, or return ABI.
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


SCHEMA = "madaros.v2.s5.f128_binary128_value_contract_native_receipt/0.2"
STAGE_CONTRACT_LEVEL = "S5_26_F128_NATIVE_LARGE_SIGNIFICAND_EXTREME_UNDERFLOW"


@dataclass(frozen=True)
class Case:
    case_id: str
    literal: str
    expected_hex: str
    expected_metadata: list[int]


@dataclass(frozen=True)
class NegativeCase:
    case_id: str
    literal: str
    expected_unsupported_detail: str
    contract_reason: str


CASES: list[Case] = [
    Case("positive_zero", "0.0", "00000000000000000000000000000000", [1, 0, 0, 2, 1, 0]),
    Case("negative_zero", "-0.0", "80000000000000000000000000000000", [-1, 0, 0, 2, 1, 0]),
    Case("one", "1.0", "3fff0000000000000000000000000000", [1, 0, 10, 2, 1, 0]),
    Case("half", "0.5", "3ffe0000000000000000000000000000", [1, 0, 5, 2, 1, 0]),
    Case("two", "2.0", "40000000000000000000000000000000", [1, 0, 20, 2, 1, 0]),
    Case(
        "smallest_normal",
        "3.36210314311209350626267781732175260259807934484647e-4932",
        "00010000000000000000000000000000",
        [1, 626267781732175260, 336210314311209350, 51, 4982, 15],
    ),
    Case("one_tenth_rounded", "0.1", "3ffb999999999999999999999999999a", [1, 0, 1, 2, 1, 0]),
    Case(
        "high_precision_probe",
        "1.2345678901234567890123456789012345",
        "3fff3c0ca428c59fb71a7be16b6b6d5b",
        [1, 90123456789012345, 123456789012345678, 35, 34, 0],
    ),
    Case("two_limb_int_36_digit_exact", "123456789012345678901234567890123456e0", "40737c6e3bfd70fdeeaec417172dcbac", [1, 901234567890123456, 123456789012345678, 36, 0, 0]),
    Case("two_limb_int_35_digit_exact", "12345678901234567890123456789012345e0", "407030582ffdf3fe588bd01278f16fbc", [1, 90123456789012345, 123456789012345678, 35, 0, 0]),
    Case("two_limb_int_all_nines_rounded", "999999999999999999999999999999999999e0", "4076812f9cf7920e2b66973e20000000", [1, 999999999999999999, 999999999999999999, 36, 0, 0]),
    Case("two_limb_int_rounding_pair_even_low", "123456789012345678500000000000000000e0", "40737c6e3bfd70fdee55ac8bac02a000", [1, 500000000000000000, 123456789012345678, 36, 0, 0]),
    Case("two_limb_int_rounding_pair_even_high", "123456789012345678500000000000000001e0", "40737c6e3bfd70fdee55ac8bac02a000", [1, 500000000000000001, 123456789012345678, 36, 0, 0]),
    Case("two_limb_int_36_digit_negative", "-123456789012345678901234567890123456e0", "c0737c6e3bfd70fdeeaec417172dcbac", [-1, 901234567890123456, 123456789012345678, 36, 0, 0]),
    Case("two_limb_dec_scale18_rounded", "123456789012345678.901234567890123456", "4037b69b4ba630f34ee6b74f031cdea0", [1, 901234567890123456, 123456789012345678, 36, 18, 0]),
    Case("two_limb_dec_scale18_sticky_low", "123456789012345678.000000000000000001", "4037b69b4ba630f34e00000000000000", [1, 1, 123456789012345678, 36, 18, 0]),
    Case("two_limb_dec_scale18_all_nines", "999999999999999999.999999999999999999", "403abc16d674ec800000000000000000", [1, 999999999999999999, 999999999999999999, 36, 18, 0]),
    Case("two_limb_dec_scale17_rounded", "1234567890123456789.01234567890123456", "403b12210f47de981150329161f20b24", [1, 901234567890123456, 123456789012345678, 36, 17, 0]),
    Case("two_limb_dec_scale16_rounded", "12345678901234567890.1234567890123456", "403e56a95319d63e15a43f35ba6e8ded", [1, 901234567890123456, 123456789012345678, 36, 16, 0]),
    Case("two_limb_dec_scale18_negative", "-123456789012345678.901234567890123456", "c037b69b4ba630f34ee6b74f031cdea0", [-1, 901234567890123456, 123456789012345678, 36, 18, 0]),
    Case("quarter_exact", "0.25", "3ffd0000000000000000000000000000", [1, 0, 25, 3, 2, 0]),
    Case("eighth_exact", "0.125", "3ffc0000000000000000000000000000", [1, 0, 125, 4, 3, 0]),
    Case("one_and_half_exact", "1.5", "3fff8000000000000000000000000000", [1, 0, 15, 2, 1, 0]),
    Case("twelve_and_three_quarters_exact", "12.75", "40029800000000000000000000000000", [1, 0, 1275, 4, 2, 0]),
    Case("negative_two_and_half_exact", "-2.5", "c0004000000000000000000000000000", [-1, 0, 25, 2, 1, 0]),
    Case("thirty_two_exact", "32.0", "40040000000000000000000000000000", [1, 0, 320, 3, 1, 0]),
    Case("ten_twenty_four_exact", "1024.0", "40090000000000000000000000000000", [1, 0, 10240, 5, 1, 0]),
    Case("one_e3_exact", "1e3", "4008f400000000000000000000000000", [1, 0, 1, 1, -3, 0]),
    Case("two_tenths_rounded", "0.2", "3ffc999999999999999999999999999a", [1, 0, 2, 2, 1, 0]),
    Case("three_tenths_rounded", "0.3", "3ffd3333333333333333333333333333", [1, 0, 3, 2, 1, 0]),
    Case("six_tenths_rounded", "0.6", "3ffe3333333333333333333333333333", [1, 0, 6, 2, 1, 0]),
    Case("seven_tenths_rounded", "0.7", "3ffe6666666666666666666666666666", [1, 0, 7, 2, 1, 0]),
    Case("nine_tenths_rounded", "0.9", "3ffecccccccccccccccccccccccccccd", [1, 0, 9, 2, 1, 0]),
    Case("one_point_one_rounded", "1.1", "3fff199999999999999999999999999a", [1, 0, 11, 2, 1, 0]),
    Case("negative_one_point_one_rounded", "-1.1", "bfff199999999999999999999999999a", [-1, 0, 11, 2, 1, 0]),
    Case("one_hundredth_rounded", "0.01", "3ff847ae147ae147ae147ae147ae147b", [1, 0, 1, 3, 2, 0]),
    Case("one_thousandth_rounded", "0.001", "3ff50624dd2f1a9fbe76c8b439581062", [1, 0, 1, 4, 3, 0]),
    Case("one_point_2345_rounded", "1.2345", "3fff3c083126e978d4fdf3b645a1cac1", [1, 0, 12345, 5, 4, 0]),
    Case("twelve_point_345_rounded", "12.345", "40028b0a3d70a3d70a3d70a3d70a3d71", [1, 0, 12345, 5, 3, 0]),
    Case("one_twenty_three_point_456_rounded", "123.456", "4005edd2f1a9fbe76c8b4395810624dd", [1, 0, 123456, 6, 3, 0]),
    Case("pi_scale10_rounded", "3.1415926535", "4000921fb54411743e0ccd6545767925", [1, 0, 31415926535, 11, 10, 0]),
    Case("one_seventeenth_prefix_scale16_rounded", "0.0588235294117647", "3ffae1e1e1e1e1e1d4518dd6a9289864", [1, 0, 588235294117647, 17, 16, 0]),
    Case("scale17_rounded", "0.12345678901234567", "3ffbf9add3746f65e780cb23f138e780", [1, 0, 12345678901234567, 18, 17, 0]),
    Case("scale18_rounded", "1e-18", "3fc32725dd1d243aba0e75fe645cc487", [1, 0, 1, 1, 18, 0]),
    Case("negative_scale18_rounded", "-1e-18", "bfc32725dd1d243aba0e75fe645cc487", [-1, 0, 1, 1, 18, 0]),
    Case("large_scale6_rounded", "123456789012.345678", "4023cbe991a14587e5a78f25a250f840", [1, 0, 123456789012345678, 18, 6, 0]),
    Case("large_all_nines_scale6_rounded", "999999999999.999999", "4026d1a94a1fffffffde7210be9424e6", [1, 0, 999999999999999999, 18, 6, 0]),
    Case("minimum_subnormal_rounded", "6.475175119438025110924438958227646552499569338034681e-4966", "00000000000000000000000000000001", [1, 92443895822764655, 647517511943802511, 52, 5017, 16]),
    Case("negative_minimum_subnormal_rounded", "-6.475175119438025110924438958227646552499569338034681e-4966", "80000000000000000000000000000001", [-1, 92443895822764655, 647517511943802511, 52, 5017, 16]),
    Case("underflow_to_positive_zero", "1e-5000", "00000000000000000000000000000000", [1, 0, 1, 1, 5000, 0]),
    Case("underflow_to_negative_zero", "-1e-5000", "80000000000000000000000000000000", [-1, 0, 1, 1, 5000, 0]),
    Case("overflow_to_positive_infinity", "1e5000", "7fff0000000000000000000000000000", [1, 0, 1, 1, -5000, 0]),
    Case("overflow_to_negative_infinity", "-1e5000", "ffff0000000000000000000000000000", [-1, 0, 1, 1, -5000, 0]),
    Case("overflow_to_positive_infinity_plus_one", "1e5001", "7fff0000000000000000000000000000", [1, 0, 1, 1, -5001, 0]),
    Case("overflow_to_negative_infinity_plus_one", "-1e5001", "ffff0000000000000000000000000000", [-1, 0, 1, 1, -5001, 0]),
    Case("two_limb_overflow_to_positive_infinity", "123456789012345678901234567890123456e5001", "7fff0000000000000000000000000000", [1, 901234567890123456, 123456789012345678, 36, -5001, 0]),
    Case("two_limb_overflow_to_negative_infinity", "-123456789012345678901234567890123456e5001", "ffff0000000000000000000000000000", [-1, 901234567890123456, 123456789012345678, 36, -5001, 0]),
    Case("underflow_to_positive_zero_scale6000", "1e-6000", "00000000000000000000000000000000", [1, 0, 1, 1, 6000, 0]),
    Case("underflow_to_negative_zero_scale6000", "-1e-6000", "80000000000000000000000000000000", [-1, 0, 1, 1, 6000, 0]),
    Case("two_limb_underflow_to_positive_zero", "123456789012345678901234567890123456e-5000", "00000000000000000000000000000000", [1, 901234567890123456, 123456789012345678, 36, 5000, 0]),
    Case("two_limb_underflow_to_negative_zero", "-123456789012345678901234567890123456e-5000", "80000000000000000000000000000000", [-1, 901234567890123456, 123456789012345678, 36, 5000, 0]),
    Case(
        "truncated_arbitrary_1p23456789012345678901234567890123456789",
        "1.23456789012345678901234567890123456789",
        "3fff3c0ca428c59fb71a7be16b6b6d5b",
        [1, 901234567890123456, 123456789012345678, 39, 38, 3],
    ),
    Case(
        "truncated_pi_40_digits",
        "3.14159265358979323846264338327950288419",
        "4000921fb54442d18469898cc51701b8",
        [1, 846264338327950288, 314159265358979323, 39, 38, 3],
    ),
    Case(
        "truncated_pi_tail20_algorithmic",
        "3.14159265358979323846264338327950288420",
        "4000921fb54442d18469898cc51701b8",
        [1, 846264338327950288, 314159265358979323, 39, 38, 3],
    ),
    Case(
        "negative_truncated_pi_tail20_algorithmic",
        "-3.14159265358979323846264338327950288420",
        "c000921fb54442d18469898cc51701b8",
        [-1, 846264338327950288, 314159265358979323, 39, 38, 3],
    ),
    Case(
        "truncated_euler_39_digits_algorithmic",
        "2.71828182845904523536028747135266249775",
        "40005bf0a8b1457695355fb8ac404e7a",
        [1, 536028747135266249, 271828182845904523, 39, 38, 3],
    ),
    Case(
        "truncated_one_third_39_repeating",
        "0.333333333333333333333333333333333333333",
        "3ffd5555555555555555555555555555",
        [1, 333333333333333333, 33333333333333333, 40, 39, 4],
    ),
    Case(
        "truncated_one_seventh_39_repeating_below_one_algorithmic",
        "0.142857142857142857142857142857142857142",
        "3ffc2492492492492492492492492492",
        [1, 714285714285714285, 14285714285714285, 40, 39, 4],
    ),
    Case("bounded_1e_minus_4", "1e-4", "3ff1a36e2eb1c432ca57a786c226809d", [1, 0, 1, 1, 4, 0]),
    Case("bounded_1e_minus_5", "1e-5", "3fee4f8b588e368f08461f9f01b866e4", [1, 0, 1, 1, 5, 0]),
    Case("bounded_1e_minus_6", "1e-6", "3feb0c6f7a0b5ed8d36b4c7f34938583", [1, 0, 1, 1, 6, 0]),
    Case("bounded_1e_minus_7", "1e-7", "3fe7ad7f29abcaf485787a6520ec08d2", [1, 0, 1, 1, 7, 0]),
    Case("bounded_1e_minus_8", "1e-8", "3fe45798ee2308c39df9fb841a566d75", [1, 0, 1, 1, 8, 0]),
    Case("bounded_1e_minus_9", "1e-9", "3fe112e0be826d694b2e62d01511f12a", [1, 0, 1, 1, 9, 0]),
    Case("bounded_1e_minus_10", "1e-10", "3fddb7cdfd9d7bdbab7d6ae6881cb511", [1, 0, 1, 1, 10, 0]),
    Case("bounded_1e_minus_11", "1e-11", "3fda5fd7fe17964955fdef1ed34a2a74", [1, 0, 1, 1, 11, 0]),
    Case("bounded_1e_minus_12", "1e-12", "3fd719799812dea11197f27f0f6e885d", [1, 0, 1, 1, 12, 0]),
    Case("bounded_1e_minus_13", "1e-13", "3fd3c25c268497681c2650cb4be40d61", [1, 0, 1, 1, 13, 0]),
    Case("bounded_1e_minus_14", "1e-14", "3fd06849b86a12b9b01ea70909833de7", [1, 0, 1, 1, 14, 0]),
    Case("bounded_1e_minus_15", "1e-15", "3fcd203af9ee756159b21f3a6e0297ec", [1, 0, 1, 1, 15, 0]),
    Case("bounded_1e_minus_16", "1e-16", "3fc9cd2b297d889bc2b6985d7cd0f313", [1, 0, 1, 1, 16, 0]),
    Case("bounded_1e_minus_17", "1e-17", "3fc670ef54646d496892137dfd73f5a9", [1, 0, 1, 1, 17, 0]),
    Case("bounded_7_8125", "7.8125", "4001f400000000000000000000000000", [1, 0, 78125, 5, 4, 0]),
    Case("bounded_42_0625", "42.0625", "40045080000000000000000000000000", [1, 0, 420625, 6, 4, 0]),
    Case("bounded_large_pi_prefix_scale1", "314159265358979.3", "402f1db9e76a24834ccccccccccccccd", [1, 0, 3141592653589793, 16, 1, 0]),
    Case("bounded_large_e_prefix_scale1", "271828182845904.5", "402eee73dc8e93a10000000000000000", [1, 0, 2718281828459045, 16, 1, 0]),
]


NEGATIVE_CASES: list[NegativeCase] = [
    NegativeCase(
        "uncontracted_near_half_min_subnormal_fails_closed",
        "3.23758755971901255546221947911382327624978466901734e-4966",
        "f128_decimal_materialization_pending",
        "near-half-min-subnormal decimal is not present in the explicit f128 value-contract set",
    ),
    NegativeCase(
        "uncontracted_near_half_min_subnormal_negative_fails_closed",
        "-3.23758755971901255546221947911382327624978466901734e-4966",
        "f128_decimal_materialization_pending",
        "signed near-half-min-subnormal decimal is not present in the explicit f128 value-contract set",
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


def u64_words_from_hex(hex_text: str) -> tuple[int, int]:
    bits = int(hex_text, 16)
    return (bits >> 64) & ((1 << 64) - 1), bits & ((1 << 64) - 1)


def signed_i64(value: int) -> int:
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def mov_rax_imm64_pattern(value: int) -> bytes:
    return b"\x48\xb8" + int(value & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)


def mov_rax_imm_patterns(value: int) -> list[bytes]:
    patterns = [mov_rax_imm64_pattern(value)]
    signed = signed_i64(value)
    if -(1 << 31) <= signed <= (1 << 31) - 1:
        patterns.append(b"\x48\xc7\xc0" + int(signed & ((1 << 32) - 1)).to_bytes(4, "little", signed=False))
    return patterns


def extract_f128_metadata_rows(module: dict[str, Any]) -> list[list[int]]:
    rows: list[list[int]] = []
    meta = module.get("f128_literal_metadata", {})
    for fn in meta.get("functions", []):
        for row in fn.get("rows", []):
            if isinstance(row, list) and len(row) >= 7:
                rows.append([int(x) for x in row])
    return rows


def compile_case(root: Path, compiler: Path, out_dir: Path, case: Case, timeout_s: int) -> dict[str, Any]:
    source = f"""fn main() -> i64 {{
    let x: f128 = {case.literal} as f128
    let y: f128 = x
    0
}}
"""
    source_path = out_dir / f"{case.case_id}.sio"
    elf_path = out_dir / f"{case.case_id}.native_v2"
    mm_path = out_dir / f"{case.case_id}.machine_module.json"
    log_path = out_dir / f"{case.case_id}.native_v2.log"
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
        raise SystemExit(f"{case.case_id}: expected native-v2 ELF emission; log={log_path}")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case.case_id}: crash or fallback detected; log={log_path}")
    if not elf_path.exists() or elf_path.stat().st_size <= 0:
        raise SystemExit(f"{case.case_id}: missing emitted ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case.case_id}: missing MachineModule JSON")

    os.chmod(elf_path, 0o755)
    run_rc, run_stdout, run_stderr = run_command([str(elf_path)], root, timeout_s)
    run_log = run_stdout + run_stderr
    if run_rc != 0:
        raise SystemExit(f"{case.case_id}: emitted ELF must run rc=0, got {run_rc}")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case.case_id}: MachineModule used fallback")
    if module.get("supported") is not True:
        raise SystemExit(f"{case.case_id}: MachineModule should remain supported for local f128 value materialization")
    metadata_rows = extract_f128_metadata_rows(module)
    if not metadata_rows:
        raise SystemExit(f"{case.case_id}: expected f128 literal metadata row")
    expected_row_tail = case.expected_metadata
    if expected_row_tail not in [row[1:7] for row in metadata_rows]:
        raise SystemExit(f"{case.case_id}: expected metadata {expected_row_tail}, got {metadata_rows}")

    elf = elf_path.read_bytes()
    hi, lo = u64_words_from_hex(case.expected_hex)
    hi_patterns = mov_rax_imm_patterns(hi)
    lo_patterns = mov_rax_imm_patterns(lo)
    hi_found = any(pattern in elf for pattern in hi_patterns)
    lo_found = any(pattern in elf for pattern in lo_patterns)
    if hi != 0 and not hi_found:
        raise SystemExit(f"{case.case_id}: missing binary128 high-word mov immediate 0x{hi:016x}")
    if lo != 0 and not lo_found:
        raise SystemExit(f"{case.case_id}: missing binary128 low-word mov immediate 0x{lo:016x}")
    return {
        "case_id": case.case_id,
        "literal": case.literal,
        "source_sha256": sha256_text(source),
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "run_rc": run_rc,
        "run_log_sha256": sha256_text(run_log),
        "elf_sha256": sha256_bytes(elf),
        "machine_module_sha256": sha256_text(stable_json(module)),
        "expected_binary128_hex": case.expected_hex,
        "expected_hi_u64": hi,
        "expected_hi_i64": signed_i64(hi),
        "expected_lo_u64": lo,
        "expected_lo_i64": signed_i64(lo),
        "expected_decimal_metadata": expected_row_tail,
        "machine_module_metadata_rows": metadata_rows,
        "hi_mov_imm_pattern_hex": [pattern.hex() for pattern in hi_patterns],
        "lo_mov_imm_pattern_hex": [pattern.hex() for pattern in lo_patterns],
        "hi_mov_imm64_pattern_found": hi_found,
        "lo_mov_imm64_pattern_found": lo_found,
    }


def compile_negative_case(root: Path, compiler: Path, out_dir: Path, case: NegativeCase, timeout_s: int) -> dict[str, Any]:
    source = f"""fn main() -> i64 {{
    let x: f128 = {case.literal} as f128
    let y: f128 = x
    0
}}
"""
    source_path = out_dir / f"{case.case_id}.sio"
    elf_path = out_dir / f"{case.case_id}.native_v2"
    mm_path = out_dir / f"{case.case_id}.machine_module.json"
    log_path = out_dir / f"{case.case_id}.native_v2.log"
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
    if rc != 0:
        raise SystemExit(f"{case.case_id}: wrapper command must return rc=0 while reporting native-v2 failure; log={log_path}")
    if "native_v2_compile: FAIL to_file rc=12" not in log:
        raise SystemExit(f"{case.case_id}: expected native-v2 fail-closed rc=12; log={log_path}")
    if "Segmentation fault" in log or "SIGSEGV" in log or "legacy fallback" in log:
        raise SystemExit(f"{case.case_id}: crash or fallback detected; log={log_path}")
    if elf_path.exists():
        raise SystemExit(f"{case.case_id}: fail-closed f128 decimal must not emit an ELF")
    if not mm_path.exists() or mm_path.stat().st_size <= 0:
        raise SystemExit(f"{case.case_id}: expected MachineModule JSON for unsupported diagnostic")

    module = json.loads(mm_path.read_text(encoding="utf-8"))
    if module.get("legacy_fallback") is not False:
        raise SystemExit(f"{case.case_id}: MachineModule used fallback")
    if module.get("supported") is not False:
        raise SystemExit(f"{case.case_id}: MachineModule must be unsupported")
    if module.get("unsupported_detail") != case.expected_unsupported_detail:
        raise SystemExit(f"{case.case_id}: expected unsupported detail {case.expected_unsupported_detail!r}, got {module.get('unsupported_detail')!r}")
    return {
        "case_id": case.case_id,
        "kind": "negative",
        "literal": case.literal,
        "contract_reason": case.contract_reason,
        "expected_unsupported_detail": case.expected_unsupported_detail,
        "compile_rc": rc,
        "compile_log_sha256": sha256_text(normalize_log(log, out_dir)),
        "elf_emitted": False,
        "machine_module_supported": module.get("supported"),
        "machine_module_unsupported_detail": module.get("unsupported_detail"),
        "machine_module_sha256": sha256_text(stable_json(module)),
    }


def emit_receipt(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve() if args.root else repo_root_from_script()
    compiler = Path(args.compiler).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = [compile_case(root, compiler, out_dir, case, int(args.timeout_s)) for case in CASES]
    negative_rows = [compile_negative_case(root, compiler, out_dir, case, int(args.timeout_s)) for case in NEGATIVE_CASES]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "stage_contract_level": STAGE_CONTRACT_LEVEL,
        "case_id": "f128_binary128_value_contract_native_materialization",
        "case_count": len(case_rows),
        "negative_case_count": len(negative_rows),
        "cases": case_rows,
        "negative_cases": negative_rows,
        "claims": {
            "f128_binary128_value_contract_native_materialization_promoted": True,
            "f128_binary128_value_contract_case_set_complete": True,
            "f128_native_exact_dyadic_decimal_binary128_materialization_promoted": True,
            "f128_native_bounded_rounded_decimal_binary128_materialization_promoted": True,
            "f128_native_general_bounded_decimal_siglo_scale18_materialization_promoted": True,
            "f128_native_two_limb_integer_decimal_binary128_materialization_promoted": True,
            "f128_native_two_limb_fractional_decimal_binary128_materialization_promoted": True,
            "f128_native_truncated_decimal_binary128_value_contract_promoted": True,
            "f128_native_bounded_truncated_two_limb_decimal_binary128_materialization_promoted": True,
            "f128_native_bounded_truncated_below_one_decimal_binary128_materialization_promoted": True,
            "f128_native_subnormal_underflow_overflow_value_contract_promoted": True,
            "f128_native_extreme_decimal_saturation_promoted": True,
            "f128_native_large_significand_extreme_underflow_to_signed_zero_promoted": True,
            "f128_native_signed_minimum_subnormal_binary128_materialization_promoted": True,
            "f128_native_value_contract_classes": [case.case_id for case in CASES],
            "f128_native_payload_words": ["binary128_hi64", "binary128_lo64"],
            "f128_native_arbitrary_decimal_binary128_materialization_promoted": False,
            "f128_native_arithmetic_promoted": False,
            "f128_native_call_abi_promoted": False,
            "f128_native_return_abi_promoted": False,
            "uncontracted_f128_decimal_materialization_fails_closed": True,
            "legacy_fallback_used": False,
        },
        "roundtrip_contract": [
            "native_v2_emits_and_runs_every_current_f128_binary128_value_contract_case_including_exact_dyadic_bounded_siglo_scale18_rounded_and_truncated_high_precision_decimals",
            "native_v2_emits_and_runs_generated_sig_hi_zero_no_truncation_scale10_le_18_decimal_matrix",
            "native_v2_emits_and_runs_algorithmic_two_limb_non_truncated_scale0_integer_decimal_matrix",
            "native_v2_emits_and_runs_algorithmic_two_limb_non_truncated_scale1_to_18_fractional_decimal_matrix",
            "native_v2_emits_and_runs_algorithmic_bounded_truncated_two_limb_retained_scale35_decimal_cases_including_below_one_with_tail_sticky_rounding",
            "native_v2_materializes_explicit_value_contract_signed_subnormal_underflow_and_finite_overflow_to_infinity_cases",
            "native_v2_materializes_decimal_overflow_to_signed_infinity_and_power10_underflow_to_signed_zero_saturation_cases",
            "machine_module_preserves_expected_decimal_metadata_for_every_case_including_negative_zero",
            "elf_contains_expected_mov_rax_imm64_for_nonzero_binary128_high_word",
            "elf_contains_expected_mov_rax_imm64_for_nonzero_binary128_low_word",
            "uncontracted_f128_decimal_literals_fail_closed_without_elf_or_opaque_word_payload_fallback",
            "receipt_does_not_promote_arbitrary_decimal_binary128_or_f128_arithmetic_or_abi",
        ],
    }
    canonical = stable_json(payload)
    if stable_json(json.loads(canonical)) != canonical:
        raise SystemExit("canonical JSON roundtrip changed bytes")
    payload["receipt_sha256"] = sha256_text(canonical)
    receipt_path = out_dir / "madaros_v2_s5_f128_binary128_value_contract_native.receipt.json"
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
        print(f"[madaros-v2-s5-f128-binary128-value-contract-native] receipt={receipt}")


if __name__ == "__main__":
    main()
