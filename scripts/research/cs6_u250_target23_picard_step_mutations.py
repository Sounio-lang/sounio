#!/usr/bin/env python3
"""Require rejection of adversarial target-23 Picard receipt mutations."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from cs6_u250_target23_picard_step_verify import verify


def flip(path: Path, offset: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if old not in text:
        raise ValueError(f"mutation anchor mismatch: {path.name}:{old}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def rejected(source: Path, mutation) -> bool:
    with tempfile.TemporaryDirectory(prefix="cs6-picard-mutation-") as directory:
        receipt = Path(directory) / "receipt"
        shutil.copytree(source, receipt)
        mutation(receipt)
        try:
            verify(receipt, True)
        except (KeyError, OSError, ValueError):
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    mutations = {
        "expected_transcript_bit": lambda receipt: flip(receipt / "expected.bin", 0),
        "candidate_box_bit": lambda receipt: flip(receipt / "inputs.bin", 8 * 16),
        "case_status": lambda receipt: replace(receipt / "cases.tsv", "\t1\t12543560845867825682829769920", "\t-4\t12543560845867825682829769920"),
        "csim_mismatch": lambda receipt: replace(receipt / "csim-summary.txt", "CSIM_MISMATCHES=0", "CSIM_MISMATCHES=1"),
        "kernel_binding": lambda receipt: replace(receipt / "csim-summary.txt", "KERNEL_SHA256=92af7a3586c1969d9cd39a23b603a1e18795ce0b4b7924bbb951a352a6a5bf75", "KERNEL_SHA256=" + "0" * 64),
        "synthesis_binding": lambda receipt: replace(receipt / "csynth-summary.txt", "CSYNTH_REPORT_SHA256=d1086c32f811b1e889285dfe9c6d40cdcffcbe88fcdfce8ba9e7a95a7343428c", "CSYNTH_REPORT_SHA256=" + "0" * 64),
        "contract_binding": lambda receipt: replace(receipt / "summary.txt", "CONTRACT_SHA256=02bb782b44370fde73fad55288881b9c3bc480947c147d641d2a2db9404299de", "CONTRACT_SHA256=" + "0" * 64),
        "claim_escalation": lambda receipt: replace(receipt / "summary.txt", "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true"),
    }
    total = 0
    for name, mutation in mutations.items():
        passed = rejected(args.receipt, mutation)
        print(f"MUTATION={name}\tREJECTED={str(passed).lower()}")
        if not passed:
            raise SystemExit(f"mutation survived: {name}")
        total += 1
    print("SCHEMA=sounio.cs6.u250-target23-picard-step-mutations.v1")
    print(f"MUTATIONS_TOTAL={total}")
    print(f"MUTATIONS_REJECTED={total}")
    print("TARGET23_PICARD_MUTATIONS_PASS=true")


if __name__ == "__main__":
    main()
