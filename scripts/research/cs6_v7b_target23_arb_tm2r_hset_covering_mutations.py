#!/usr/bin/env python3
"""Negative mutations for the local h-set covering candidate verifier."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def rejected(verifier: Path, receipts: Path) -> bool:
    completed = subprocess.run(
        [sys.executable, str(verifier), "--receipts", str(receipts)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode != 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    verifier = script_dir / "cs6_v7b_target23_arb_tm2r_hset_covering_verify.py"
    source = script_dir / "receipts" / "cs6_v7b_target23_arb_tm2r_hset_covering_v1"
    aggregate = (source / "aggregate.txt").read_text(encoding="ascii")
    scenarios = (
        (
            "false-cover-promotion",
            "LOCAL_HSET_COVERING_RELATION_CERTIFICATE=false",
            "LOCAL_HSET_COVERING_RELATION_CERTIFICATE=true",
        ),
        (
            "false-exit-promotion",
            "EXIT_FACE_INEQUALITIES_CERTIFICATE=false",
            "EXIT_FACE_INEQUALITIES_CERTIFICATE=true",
        ),
        (
            "overlap-erasure",
            next(line for line in aggregate.splitlines() if line.startswith("EXIT_FACE_OVERLAP_Q=")),
            "EXIT_FACE_OVERLAP_Q=0",
        ),
        (
            "determinant-sign",
            next(line for line in aggregate.splitlines() if line.startswith("NORMALIZED_RETURN_DETERMINANT_UPPER_Q=")),
            "NORMALIZED_RETURN_DETERMINANT_UPPER_Q=0",
        ),
        (
            "entry-margin",
            next(line for line in aggregate.splitlines() if line.startswith("ENTRY_MARGIN_Q=")),
            "ENTRY_MARGIN_Q=-1",
        ),
        (
            "carrier-hash",
            next(line for line in aggregate.splitlines() if line.startswith("CARRIER_RECEIPT_SHA256_XLEL=")),
            "CARRIER_RECEIPT_SHA256_XLEL=" + "0" * 64,
        ),
        ("chaos-promotion", "CHAOS_PROVED=false", "CHAOS_PROVED=true"),
        ("point-fallback", "POINT_FALLBACK_USED=false", "POINT_FALLBACK_USED=true"),
    )
    rejected_count = 0
    with tempfile.TemporaryDirectory(prefix="cs6-hset-mutations-") as temp:
        for name, old, new in scenarios:
            case = Path(temp) / name
            shutil.copytree(source, case)
            replace_once(case / "aggregate.txt", old, new)
            if not rejected(verifier, case):
                print(f"MUTATION_ACCEPTED={name}", file=sys.stderr)
                return 1
            print(f"MUTATION_REJECTED={name}")
            rejected_count += 1
    print(f"MUTATIONS_REJECTED={rejected_count}")
    print("MUTATION_AUDIT=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
