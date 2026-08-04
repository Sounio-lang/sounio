#!/usr/bin/env python3
"""Reject adversarial mutations of the scaled Taylor-16 receipt."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from cs6_u250_target23_scaled_taylor16_verify import verify


def rejected(source: Path, relative: str, offset: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="cs6-taylor16-mutation-") as directory:
        receipt = Path(directory) / "receipt"
        shutil.copytree(source, receipt)
        path = receipt / relative
        data = bytearray(path.read_bytes())
        data[offset] ^= 1
        path.write_bytes(data)
        try:
            verify(receipt)
        except (KeyError, OSError, ValueError):
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    mutations = [
        ("input_bit", "inputs.bin", 0),
        ("coefficient_bit", "expected.bin", 0),
        ("remainder_bit", "expected.bin", 128 * 16),
        ("next_state_bit", "expected.bin", 144 * 16),
        ("csim_log_bit", "csim.log", 0),
        ("csim_summary_bit", "csim-summary.txt", 0),
        ("csynth_log_bit", "csynth.log", 0),
        ("csynth_report_bit", "csynth.rpt", 0),
        ("csynth_summary_bit", "csynth-summary.txt", 0),
        ("xclbin_negative_log_bit", "xclbin-500mhz-vivado-negative.log", 0),
        ("xclbin_negative_summary_bit", "xclbin-500mhz-negative-summary.txt", 0),
        ("xclbin_build_log_bit", "xclbin-200mhz-build.log", 0),
        ("xclbin_timing_report_bit", "xclbin-200mhz-timing-routed.rpt.gz", 0),
        ("xclbin_info_bit", "target23_scaled_taylor16.xclbin.info", 0),
        ("xclbin_link_summary_bit", "target23_scaled_taylor16.xclbin.link_summary", 0),
        ("xclbin_summary_bit", "xclbin-200mhz-summary.txt", 0),
        ("physical_device_bit", "physical-device-examine.txt", 0),
        ("physical_run_bit", "physical-u250-run.log", 0),
        ("physical_summary_bit", "physical-u250-summary.txt", 0),
    ]
    for name, path, offset in mutations:
        passed = rejected(args.receipt, path, offset)
        print(f"MUTATION={name}\tREJECTED={str(passed).lower()}")
        if not passed:
            raise SystemExit(f"mutation survived: {name}")
    print(f"MUTATIONS_TOTAL={len(mutations)}")
    print(f"MUTATIONS_REJECTED={len(mutations)}")
    print("TARGET23_SCALED_TAYLOR16_MUTATIONS_PASS=true")


if __name__ == "__main__":
    main()
