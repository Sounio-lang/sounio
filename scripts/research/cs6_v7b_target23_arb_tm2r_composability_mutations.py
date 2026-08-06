#!/usr/bin/env python3
"""Negative mutations for the full-support composability verifier."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    value = path.read_text(encoding="ascii")
    if value.count(old) != 1:
        raise RuntimeError(f"expected one occurrence of {old!r} in {path.name}")
    path.write_text(value.replace(old, new, 1), encoding="ascii")


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
    verifier = script_dir / "cs6_v7b_target23_arb_tm2r_composability_verify.py"
    source = script_dir / "receipts/cs6_v7b_target23_arb_tm2r_composability_v1"
    aggregate = (source / "aggregate.txt").read_text(encoding="ascii")

    def field(name: str) -> str:
        return next(line for line in aggregate.splitlines() if line.startswith(name + "="))

    scenarios = (
        ("exit-margin-zero", "aggregate.txt", field("EXIT_MARGIN_Q"), "EXIT_MARGIN_Q=0"),
        ("degree-erasure", "aggregate.txt", "COVERING_DEGREE=1", "COVERING_DEGREE=0"),
        ("determinant-zero", "aggregate.txt", field("NORMALIZED_RETURN_DETERMINANT_LOWER_Q"), "NORMALIZED_RETURN_DETERMINANT_LOWER_Q=0"),
        ("recurrent-promotion", "aggregate.txt", "RECURRENT_COVERING_GRAPH_CERTIFICATE=false", "RECURRENT_COVERING_GRAPH_CERTIFICATE=true"),
        ("point-fallback", "support_XLEL.json", '"point_fallback_used":false', '"point_fallback_used":true'),
        ("box-flattening", "support_XLEH.json", '"box_flattening_used":false', '"box_flattening_used":true'),
    )
    rejected_count = 0
    with tempfile.TemporaryDirectory(prefix="cs6-composability-mutations-") as temp:
        for name, filename, old, new in scenarios:
            case = Path(temp) / name
            shutil.copytree(source, case)
            replace_once(case / filename, old, new)
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
