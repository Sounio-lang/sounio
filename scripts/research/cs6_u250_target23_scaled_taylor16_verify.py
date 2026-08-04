#!/usr/bin/env python3
"""Independently verify the step-scaled Taylor-16 integer transcript."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs6_u250_target23_picard_step_verify as pv


ONE = 1 << 96
H = (ONE >> 8, ONE >> 8)
ORDER = 16
Interval = tuple[int, int]


def fail(message: str) -> None:
    raise ValueError(f"scaled Taylor-16 verify error: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kv(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.count("=") != 1:
            fail(f"malformed summary line: {path.name}")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not value or key in fields:
            fail(f"invalid summary field: {key}")
        fields[key] = value
    return fields


def floor_value(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_value(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def divide(value: Interval, divisor: int) -> Interval:
    return floor_value(Fraction(value[0], divisor)), ceil_value(Fraction(value[1], divisor))


def sum_intervals(values: list[Interval]) -> Interval:
    lower = sum(value[0] for value in values)
    upper = sum(value[1] for value in values)
    return lower, upper


def scaled_divide(value: Interval, divisor: int) -> Interval:
    return divide(pv.times(value, H), divisor)


def recurrence(state: tuple[Interval, ...], zs: Interval, order: int) -> list[list[Interval]]:
    result = [[state[axis] if degree == 0 else (0, 0) for degree in range(order + 1)] for axis in range(4)]
    for degree in range(order):
        xy = sum_intervals([pv.times(result[0][j], result[1][degree - j]) for j in range(degree + 1)])
        yy = sum_intervals([pv.times(result[1][j], result[1][degree - j]) for j in range(degree + 1)])
        yw = sum_intervals([pv.times(result[1][j], result[2][degree - j]) for j in range(degree + 1)])
        result[0][degree + 1] = scaled_divide(pv.minus((2 * yy[0], 2 * yy[1]), xy), degree + 1)
        result[1][degree + 1] = scaled_divide(pv.minus(xy, pv.half(pv.plus(yw, pv.times(zs, result[1][degree])))), degree + 1)
        result[2][degree + 1] = scaled_divide(pv.minus(xy, pv.plus(result[2][degree], zs if degree == 0 else (0, 0))), degree + 1)
        constant = pv.plus(pv.half(zs), (ONE, ONE)) if degree == 0 else (0, 0)
        result[3][degree + 1] = scaled_divide(pv.minus(pv.minus(pv.minus(result[0][degree], result[1][degree]), pv.half(result[2][degree])), constant), degree + 1)
    return result


def evaluate(intervals: tuple[Interval, ...]) -> tuple[int, ...]:
    initial, box, zs = intervals[:4], intervals[4:8], intervals[8]
    if any(lower > upper for lower, upper in intervals):
        return (0,) * 152 + (-1,)
    if pv.evaluate(initial, box, zs)[-1] != 1:
        return (0,) * 152 + (-4,)
    center = recurrence(initial, zs, ORDER - 1)
    wide = recurrence(box, zs, ORDER)
    polynomial = [sum_intervals(center[axis]) for axis in range(4)]
    # This is a Lagrange remainder enclosure, not a center-series tail sum.
    lagrange_remainder = [wide[axis][ORDER] for axis in range(4)]
    next_state = [pv.plus(polynomial[axis], lagrange_remainder[axis]) for axis in range(4)]
    words = [endpoint for degree in range(ORDER) for axis in range(4) for endpoint in center[axis][degree]]
    words += [endpoint for interval in lagrange_remainder + polynomial + next_state for endpoint in interval]
    return tuple(words) + (1,)


def decode(path: Path) -> list[int]:
    raw = path.read_bytes()
    if len(raw) % 16:
        fail(f"unaligned binary: {path.name}")
    return [int.from_bytes(raw[index:index + 16], "little", signed=True) for index in range(0, len(raw), 16)]


def verify(receipt: Path) -> None:
    root = Path.cwd()
    summary = kv(receipt / "summary.txt")
    predecessor = root / "scripts/research/receipts/cs6_u250_target23_picard_step_v1/inputs.bin"
    required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-vectors.v1",
        "CONTRACT_SHA256": digest(root / "scripts/research/cs6_u250_target23_scaled_taylor16_contract_v1.txt"),
        "GENERATOR_SHA256": digest(root / "scripts/research/cs6_u250_target23_scaled_taylor16_generate.py"),
        "PREDECESSOR_INPUTS_SHA256": digest(predecessor),
        "CASES": "3", "ACCEPTED_CASES": "1", "REFUSED_CASES": "2",
        "INPUT_WORDS": "54", "OUTPUT_WORDS": "459",
        "CASES_SHA256": digest(receipt / "cases.tsv"),
        "INPUTS_SHA256": digest(receipt / "inputs.bin"),
        "EXPECTED_SHA256": digest(receipt / "expected.bin"),
        "EXACT_RATIONAL_SCALED_TAYLOR_RECONSTRUCTION": "true",
        "HLS_CSIM": "false", "HLS_SYNTHESIS": "false", "PHYSICAL_FPGA_EXECUTION": "false",
        "FULL_ORBIT_CERTIFICATE": "false", "LEAF_WIDE_CERTIFICATE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "NOVELTY_OR_PRIORITY_CLAIMED": "false", "OPEN_PROBLEM_SOLVED": "false",
    }
    for key, value in required.items():
        if summary.get(key) != value:
            fail(f"summary mismatch: {key}")
    inputs, expected = decode(receipt / "inputs.bin"), decode(receipt / "expected.bin")
    if len(inputs) != 54 or len(expected) != 459 or inputs[:18] != decode(predecessor)[:18]:
        fail("input or output binding mismatch")
    rows = list(csv.DictReader((receipt / "cases.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    statuses = []
    for index, row in enumerate(rows):
        words = inputs[18 * index:18 * (index + 1)]
        intervals = tuple((words[offset], words[offset + 1]) for offset in range(0, 18, 2))
        result = evaluate(intervals)
        if tuple(expected[153 * index:153 * (index + 1)]) != result:
            fail(f"transcript mismatch at case {index + 1}")
        if row["CASE_INDEX"] != str(index + 1) or row["STATUS"] != str(result[-1]):
            fail("case metadata mismatch")
        if result[-1] == 1:
            remainder, next_state = result[128:136], result[144:152]
            maximum = max(abs(word) for word in remainder)
            width = max(next_state[2 * axis + 1] - next_state[2 * axis] for axis in range(4))
            if row["MAX_REMAINDER_ABS_RAW"] != str(maximum) or row["MAX_NEXT_WIDTH_RAW"] != str(width):
                fail("remainder metrics mismatch")
            if maximum != 50104134 or width != 100184611:
                fail("frozen Taylor-16 certificate drift")
        statuses.append(result[-1])
    if statuses != [1, -4, -1]:
        fail("status population drift")

    csim = kv(receipt / "csim-summary.txt")
    hardware = root / "hardware/fpga/u250_target23_scaled_taylor16"
    csim_required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-csim.v1",
        "BUILDER": "VM100_VITIS_U250_BUILDER",
        "VITIS_VERSION": "2025.1",
        "PART": "xcu250-figd2104-2L-e",
        "KERNEL_SHA256": digest(hardware / "kernel.cpp"),
        "TESTBENCH_SHA256": digest(hardware / "testbench.cpp"),
        "TCL_SHA256": digest(hardware / "run_hls_csim.tcl"),
        "INPUTS_SHA256": digest(receipt / "inputs.bin"),
        "EXPECTED_SHA256": digest(receipt / "expected.bin"),
        "CSIM_LOG_SHA256": digest(receipt / "csim.log"),
        "CSIM_CASES": "3", "CSIM_WORDS": "459", "CSIM_MISMATCHES": "0",
        "TARGET23_SCALED_TAYLOR16_CSIM_PASS": "true",
        "PHYSICAL_FPGA_EXECUTION": "false",
    }
    for key, value in csim_required.items():
        if csim.get(key) != value:
            fail(f"CSim receipt mismatch: {key}")
    log = (receipt / "csim.log").read_text(encoding="utf-8")
    if "CSIM_MISMATCHES=0" not in log or "TARGET23_SCALED_TAYLOR16_CSIM_PASS=true" not in log:
        fail("CSim log verdict missing")

    csynth = kv(receipt / "csynth-summary.txt")
    csynth_required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-csynth.v1",
        "BUILDER": "VM100_VITIS_U250_BUILDER", "VITIS_VERSION": "2025.1",
        "PART": "xcu250-figd2104-2L-e",
        "KERNEL_SHA256": digest(hardware / "kernel.cpp"),
        "TCL_SHA256": digest(hardware / "run_hls_csynth.tcl"),
        "CSYNTH_LOG_SHA256": digest(receipt / "csynth.log"),
        "CSYNTH_REPORT_SHA256": digest(receipt / "csynth.rpt"),
        "TARGET_CLOCK_NS": "4.00", "ESTIMATED_CLOCK_NS": "2.920",
        "ESTIMATED_FMAX_MHZ": "342.47", "LATENCY_MIN_CYCLES": "838",
        "LATENCY_MAX_CYCLES": "42271", "LATENCY_MAX_US_AT_TARGET_CLOCK": "169.084",
        "INTERVAL_MIN_CYCLES": "839", "INTERVAL_MAX_CYCLES": "42272",
        "BRAM_18K": "19", "DSP": "1350", "FF": "152148", "LUT": "88020", "URAM": "0",
        "ALL_LOOP_CONSTRAINTS_SATISFIED": "false", "RTL_GENERATED": "true",
        "HLS_SYNTHESIS_PASS": "true", "PHYSICAL_FPGA_EXECUTION": "false",
    }
    for key, value in csynth_required.items():
        if csynth.get(key) != value:
            fail(f"CSynth receipt mismatch: {key}")
    synth_log = (receipt / "csynth.log").read_text(encoding="utf-8")
    if "Finished Command csynth_design" not in synth_log or "Estimated Fmax: 342.47 MHz" not in synth_log:
        fail("CSynth log verdict missing")

    negative = kv(receipt / "xclbin-500mhz-negative-summary.txt")
    negative_required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-xclbin-negative.v1",
        "KERNEL_FREQUENCY_MHZ": "500", "POST_PLACEMENT_WNS_NS": "-2.486",
        "CONGESTION_DETECTED": "true", "IMPLEMENTATION_STATUS": "ERROR",
        "XCLBIN_GENERATED": "false",
        "BUILD_LOG_SHA256": digest(receipt / "xclbin-500mhz-negative.log"),
        "VIVADO_LOG_SHA256": digest(receipt / "xclbin-500mhz-vivado-negative.log"),
        "NEGATIVE_EVIDENCE_RETAINED": "true",
    }
    for key, value in negative_required.items():
        if negative.get(key) != value:
            fail(f"500 MHz negative receipt mismatch: {key}")
    negative_log = (receipt / "xclbin-500mhz-vivado-negative.log").read_text(encoding="utf-8")
    if "WNS=-2.486" not in negative_log or "Congestion is preventing" not in negative_log:
        fail("500 MHz negative signatures missing")

    xclbin = kv(receipt / "xclbin-200mhz-summary.txt")
    xclbin_required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-xclbin.v1",
        "BUILDER": "VM100_VITIS_U250_BUILDER", "VITIS_VERSION": "2025.1",
        "PLATFORM": "xilinx_u250_gen3x16_xdma_4_1_202210_1",
        "KERNEL_SHA256": digest(hardware / "kernel.cpp"),
        "BUILD_SCRIPT_SHA256": digest(hardware / "build_xclbin.sh"),
        "CONNECTIVITY_SHA256": digest(hardware / "connectivity.cfg"),
        "BUILD_LOG_SHA256": digest(receipt / "xclbin-200mhz-build.log"),
        "TIMING_REPORT_GZIP_SHA256": digest(receipt / "xclbin-200mhz-timing-routed.rpt.gz"),
        "TIMING_REPORT_UNCOMPRESSED_SHA256": "a29cd28aadc91a0e117e7baa64c99e0c0568b6b7792ac1245b8233cc31137325",
        "XCLBIN_INFO_SHA256": digest(receipt / "target23_scaled_taylor16.xclbin.info"),
        "LINK_SUMMARY_SHA256": digest(receipt / "target23_scaled_taylor16.xclbin.link_summary"),
        "REQUESTED_KERNEL_FREQUENCY_MHZ": "200",
        "ACHIEVED_KERNEL_FREQUENCY_MHZ": "102.9", "AUTO_FREQUENCY_SCALING": "true",
        "ROUTED_WNS_NS": "0.020", "ROUTED_TNS_NS": "0.000",
        "ROUTED_TIMING_CONSTRAINTS_MET": "true", "XCLBIN_SIZE_BYTES": "56974000",
        "XCLBIN_SHA256": "15e73b5ffaa892d35c43025bc29879275d1da82d6ff58ea8119546173eb24c75",
        "XCLBIN_GENERATED": "true",
    }
    for key, value in xclbin_required.items():
        if xclbin.get(key) != value:
            fail(f"xclbin receipt mismatch: {key}")
    build_log = (receipt / "xclbin-200mhz-build.log").read_text(encoding="utf-8")
    with gzip.open(receipt / "xclbin-200mhz-timing-routed.rpt.gz", "rt", encoding="utf-8") as stream:
        timing = stream.read()
    timing_raw_digest = hashlib.sha256(timing.encode("utf-8")).hexdigest()
    if timing_raw_digest != xclbin_required["TIMING_REPORT_UNCOMPRESSED_SHA256"]:
        fail("uncompressed routed timing digest mismatch")
    xclbin_info = (receipt / "target23_scaled_taylor16.xclbin.info").read_text(encoding="utf-8")
    if "frequency has been automatically changed to 102.9 MHz" not in build_log:
        fail("automatic frequency scaling signature missing")
    if "Created build/target23_scaled_taylor16.xclbin" not in build_log:
        fail("xclbin creation signature missing")
    if "0.020        0.000" not in timing or "All user specified timing constraints are met." not in timing:
        fail("routed timing signatures missing")
    if "Requested Freq: 200 MHz" not in xclbin_info or "Achieved Freq:  102.9 MHz" not in xclbin_info:
        fail("xclbin frequency signatures missing")

    physical = kv(receipt / "physical-u250-summary.txt")
    physical_required = {
        "SCHEMA": "sounio.cs6.u250-target23-scaled-taylor16-physical.v1",
        "HOST": "dl380-proxmox", "DEVICE_BDF": "0000:d8:00.1",
        "SHELL": "xilinx_u250_gen3x16_xdma_shell_4_1", "XRT_VERSION": "2.23.0",
        "DEVICE_READY": "true",
        "DEVICE_EXAMINE_SHA256": digest(receipt / "physical-device-examine.txt"),
        "HOST_SOURCE_SHA256": digest(hardware / "host.cpp"),
        "INPUTS_SHA256": digest(receipt / "inputs.bin"),
        "EXPECTED_SHA256": digest(receipt / "expected.bin"),
        "XCLBIN_SHA256": xclbin_required["XCLBIN_SHA256"],
        "RUN_LOG_SHA256": digest(receipt / "physical-u250-run.log"),
        "FPGA_DEVICE_INDEX": "0", "FPGA_CASES": "3", "FPGA_WORDS": "459",
        "FPGA_MISMATCHES": "0", "TARGET23_SCALED_TAYLOR16_PHYSICAL_PASS": "true",
    }
    for key, value in physical_required.items():
        if physical.get(key) != value:
            fail(f"physical U250 receipt mismatch: {key}")
    device_log = (receipt / "physical-device-examine.txt").read_text(encoding="utf-8")
    run_log = (receipt / "physical-u250-run.log").read_text(encoding="utf-8")
    if "xilinx_u250_gen3x16_xdma_shell_4_1" not in device_log or "Yes" not in device_log:
        fail("physical U250 readiness signature missing")
    if "FPGA_MISMATCHES=0" not in run_log or "TARGET23_SCALED_TAYLOR16_PHYSICAL_PASS=true" not in run_log:
        fail("physical U250 verdict missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        verify(args.receipt)
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("VERIFY_SCHEMA=sounio.cs6.u250-target23-scaled-taylor16-verification.v1")
    print("VERIFIED_CASES=3")
    print("MAX_REMAINDER_ABS_RAW=50104134")
    print("MAX_NEXT_WIDTH_RAW=100184611")
    print("BOUNDED_ONE_STEP_TAYLOR_ENDPOINT_CERTIFICATE=true")
    print("HLS_CSIM_VERIFIED=true")
    print("HLS_SYNTHESIS_VERIFIED=true")
    print("XCLBIN_500MHZ_NEGATIVE_VERIFIED=true")
    print("XCLBIN_200MHZ_REQUEST_102_9MHZ_ACHIEVED_VERIFIED=true")
    print("PHYSICAL_U250_WORDS_VERIFIED=459")
    print("PHYSICAL_U250_EXECUTION_VERIFIED=true")
    print("TARGET23_SCALED_TAYLOR16_VERIFY_PASS=true")


if __name__ == "__main__":
    main()
