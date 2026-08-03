#!/usr/bin/env python3
"""Require rejection of critical arithmetic-receipt tampering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

from cs6_u250_validated_dyadic_verify import verify


def rewrite_manifest(receipt: Path) -> None:
    manifest = receipt / "artifact-files.sha256"
    if not manifest.exists():
        return
    lines = []
    for path in sorted(candidate for candidate in receipt.rglob("*") if candidate.is_file() and candidate != manifest):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(receipt)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")


def flip_binary(path: Path, offset: int) -> None:
    raw = bytearray(path.read_bytes())
    raw[offset] ^= 1
    path.write_bytes(raw)


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise SystemExit(f"mutation anchor mismatch: {path.name}:{old}")
    path.write_text(text.replace(old, new), encoding="ascii")


def mutate_case_status(receipt: Path) -> None:
    path = receipt / "cases.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    rows[0]["STATUS"] = "-3"
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mutate_divisor_binding(receipt: Path) -> None:
    path = receipt / "cases.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    rows[0]["DIVISOR"] = "5"
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    raw = bytearray((receipt / "inputs.bin").read_bytes())
    raw[4 * 16:5 * 16] = int(5).to_bytes(16, "little", signed=True)
    (receipt / "inputs.bin").write_bytes(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    mutations = {
        "expected_endpoint_bit": lambda receipt: flip_binary(receipt / "expected.bin", 0),
        "input_endpoint_bit": lambda receipt: flip_binary(receipt / "inputs.bin", 0),
        "case_status": mutate_case_status,
        "divisor_binding": mutate_divisor_binding,
        "csim_mismatch": lambda receipt: replace(receipt / "csim-summary.txt", "CSIM_MISMATCHES=0", "CSIM_MISMATCHES=1"),
        "contract_binding": lambda receipt: replace(receipt / "summary.txt", "CONTRACT_SHA256=067b04a4519d051b5c01480c33f99d676cf95c7c596ff171d5c9a73a02d3b76d", "CONTRACT_SHA256=" + "0" * 64),
        "claim_escalation": lambda receipt: replace(receipt / "summary.txt", "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true"),
    }
    rejected = 0
    for identity, mutation in mutations.items():
        with tempfile.TemporaryDirectory(prefix=f"cs6-u250-dyadic-{identity}-") as temporary:
            candidate = Path(temporary) / "receipt"
            shutil.copytree(args.receipt, candidate)
            mutation(candidate)
            rewrite_manifest(candidate)
            try:
                verify(candidate)
            except (KeyError, OSError, ValueError):
                rejected += 1
                print(f"MUTATION={identity}\tREJECTED=true")
            else:
                print(f"MUTATION={identity}\tREJECTED=false")
    print("SCHEMA=sounio.cs6.u250-validated-dyadic-mutations.v1")
    print(f"MUTATIONS_TOTAL={len(mutations)}")
    print(f"MUTATIONS_REJECTED={rejected}")
    passed = rejected == len(mutations)
    print(f"VALIDATED_DYADIC_MUTATIONS_PASS={str(passed).lower()}")
    if not passed:
        raise SystemExit("validated dyadic mutation escaped")


if __name__ == "__main__":
    main()
