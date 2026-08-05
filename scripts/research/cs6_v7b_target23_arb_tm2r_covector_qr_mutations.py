#!/usr/bin/env python3
"""Negative mutations for the QR/C2 anchored covering verifier."""

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
    verifier = script_dir / "cs6_v7b_target23_arb_tm2r_covector_qr_verify.py"
    source = script_dir / "receipts/cs6_v7b_target23_arb_tm2r_covector_qr_v1"
    aggregate = (source / "aggregate.txt").read_text(encoding="ascii")

    def field(name: str) -> str:
        return next(line for line in aggregate.splitlines() if line.startswith(name + "="))

    scenarios = (
        ("qr-false-promotion", "aggregate.txt", "DYNAMIC_TRIPLETON_DIRECTIONAL_IMPROVEMENT_CERTIFICATE=false", "DYNAMIC_TRIPLETON_DIRECTIONAL_IMPROVEMENT_CERTIFICATE=true"),
        ("c2-derivative-zero", "aggregate.txt", field("C2_MEAN_VALUE_DU_DXI_LOWER_Q"), "C2_MEAN_VALUE_DU_DXI_LOWER_Q=0"),
        ("anchored-gap-zero", "aggregate.txt", field("ANCHORED_EXIT_FACE_GAP_Q"), "ANCHORED_EXIT_FACE_GAP_Q=0"),
        ("target-radius-negative", "aggregate.txt", field("TARGET_U_RADIUS_Q"), "TARGET_U_RADIUS_Q=-1"),
        ("exit-margin-zero", "aggregate.txt", "EXIT_MARGIN_Q=1", "EXIT_MARGIN_Q=0"),
        ("degree-erasure", "aggregate.txt", "COVERING_DEGREE=1", "COVERING_DEGREE=0"),
        ("recurrent-graph-promotion", "aggregate.txt", "RECURRENT_COVERING_GRAPH_CERTIFICATE=false", "RECURRENT_COVERING_GRAPH_CERTIFICATE=true"),
        ("chaos-promotion", "aggregate.txt", "CHAOS_PROVED=false", "CHAOS_PROVED=true"),
        ("open-problem-promotion", "aggregate.txt", "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true"),
        ("point-fallback", "aggregate.txt", "POINT_FALLBACK_USED=false", "POINT_FALLBACK_USED=true"),
        ("tripleton-kind", "face_LEFT_XLEH_ROOT.json", '"carrier_kind":"DYNAMIC_TRIPLETON"', '"carrier_kind":"DYNAMIC_DOUBLETON"'),
        ("doubleton-evidence", "doubleton_LEFT_XLEH_ROOT.stderr.txt", "end_step=617", "end_step=618"),
    )
    rejected_count = 0
    with tempfile.TemporaryDirectory(prefix="cs6-covector-qr-mutations-") as temp:
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
