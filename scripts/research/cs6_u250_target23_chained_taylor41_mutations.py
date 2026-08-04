#!/usr/bin/env python3
"""Reject direct and coherently rehashed mutations of the Taylor-41 chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

from cs6_u250_target23_chained_taylor41_verify import verify


def replace_summary(receipt: Path, key: str, value: str) -> None:
    path = receipt / "summary.txt"
    lines = path.read_text(encoding="ascii").splitlines()
    path.write_text("\n".join(value if line.startswith(f"{key}=") else line for line in lines) + "\n", encoding="ascii")


def set_digest(receipt: Path, key: str, relative: str) -> None:
    digest = hashlib.sha256((receipt / relative).read_bytes()).hexdigest()
    replace_summary(receipt, key, f"{key}={digest}")


def bit_flip(receipt: Path, relative: str, offset: int = 0) -> None:
    path = receipt / relative
    data = bytearray(path.read_bytes())
    data[offset] ^= 1
    path.write_bytes(data)


def coherent_chain(receipt: Path) -> None:
    path = receipt / "chain.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    rows[799]["CENTER_W_RAW"] = str(int(rows[799]["CENTER_W_RAW"]) + 1)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    set_digest(receipt, "CHAIN_SHA256", "chain.tsv")


def coherent_event(receipt: Path) -> None:
    path = receipt / "events.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    rows[0]["NORMAL_LOWER_RAW"] = str(int(rows[0]["NORMAL_LOWER_RAW"]) + 1)
    write_events(path, rows)
    set_digest(receipt, "EVENTS_SHA256", "events.tsv")


def write_events(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coherent_event_negative_normal(receipt: Path) -> None:
    path = receipt / "events.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    lower = int(rows[0]["NORMAL_LOWER_RAW"])
    upper = int(rows[0]["NORMAL_UPPER_RAW"])
    rows[0]["NORMAL_LOWER_RAW"] = str(-upper)
    rows[0]["NORMAL_UPPER_RAW"] = str(-lower)
    write_events(path, rows)
    set_digest(receipt, "EVENTS_SHA256", "events.tsv")


def coherent_event_wide_bracket(receipt: Path) -> None:
    path = receipt / "events.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    extra = 1 << (192 - 50)
    rows[0]["LOCAL_HIGH_RAW"] = str(int(rows[0]["LOCAL_HIGH_RAW"]) + extra)
    rows[0]["GLOBAL_HIGH_RAW"] = str(int(rows[0]["GLOBAL_HIGH_RAW"]) + extra)
    write_events(path, rows)
    set_digest(receipt, "EVENTS_SHA256", "events.tsv")


def coherent_event_swap(receipt: Path) -> None:
    path = receipt / "events.tsv"
    rows = list(csv.DictReader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    rows[0], rows[1] = rows[1], rows[0]
    write_events(path, rows)
    set_digest(receipt, "EVENTS_SHA256", "events.tsv")


def coherent_expected(receipt: Path) -> None:
    bit_flip(receipt, "expected.bin", 28 * 9000)
    set_digest(receipt, "EXPECTED_SHA256", "expected.bin")


def rejected(source: Path, mutation: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="cs6-chain41-mutation-") as directory:
        receipt = Path(directory) / "receipt"
        shutil.copytree(source, receipt)
        if mutation == "summary":
            bit_flip(receipt, "summary.txt")
        elif mutation == "inputs":
            bit_flip(receipt, "inputs.bin")
        elif mutation == "expected":
            bit_flip(receipt, "expected.bin")
        elif mutation == "chain":
            bit_flip(receipt, "chain.tsv")
        elif mutation == "events":
            bit_flip(receipt, "events.tsv")
        elif mutation == "hardware_inputs":
            bit_flip(receipt, "hardware_inputs.bin")
        elif mutation == "coherent_chain":
            coherent_chain(receipt)
        elif mutation == "coherent_event":
            coherent_event(receipt)
        elif mutation == "coherent_event_negative_normal":
            coherent_event_negative_normal(receipt)
        elif mutation == "coherent_event_wide_bracket":
            coherent_event_wide_bracket(receipt)
        elif mutation == "coherent_event_swap":
            coherent_event_swap(receipt)
        elif mutation == "coherent_expected":
            coherent_expected(receipt)
        else:
            raise ValueError(mutation)
        try:
            verify(receipt)
        except (KeyError, OSError, ValueError):
            return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    mutations = (
        "summary", "inputs", "expected", "chain", "events", "hardware_inputs",
        "coherent_chain", "coherent_event", "coherent_event_negative_normal",
        "coherent_event_wide_bracket", "coherent_event_swap", "coherent_expected",
    )
    for mutation in mutations:
        result = rejected(args.receipt, mutation)
        print(f"MUTATION={mutation}\tREJECTED={str(result).lower()}")
        if not result:
            raise SystemExit(f"mutation survived: {mutation}")
    print(f"MUTATIONS_TOTAL={len(mutations)}")
    print(f"MUTATIONS_REJECTED={len(mutations)}")
    print("TARGET23_CHAINED_TAYLOR41_MUTATIONS_PASS=true")


if __name__ == "__main__":
    main()
