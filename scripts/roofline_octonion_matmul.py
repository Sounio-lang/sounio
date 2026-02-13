#!/usr/bin/env python3
"""
Generate roofline point data (arithmetic intensity, achieved GFLOP/s) for the
octonion matmul Criterion benchmarks.

This intentionally has *no* third-party Python dependencies so it can run in
minimal environments.

Input:
  target/criterion/octonion_matmul/matmul_{N}x{N}/new/estimates.json

Output CSV columns:
  n,time_ns,gflops,intensity_conservative,intensity_ideal,flops,bytes_conservative,bytes_ideal

Arithmetic model assumptions (for AI estimates):
  - Octonion multiply: 120 FLOPs (as documented in benches/compiler/octonion_benchmark.rs)
  - Octonion add: 8 FLOPs
  - Naive matmul implementation does, per inner-loop iteration:
      sum = sum + (a * b)
    i.e. 1 mul + 1 add = 128 FLOPs per k
  - Total FLOPs ~= 128 * N^3

Memory traffic bounds (bytes moved):
  - Conservative (naive / no reuse): for each multiply, load 2 octonions (64 B),
    plus write the output matrix once (32 B per output).
      bytes = 64*N^3 + 32*N^2
  - Ideal (perfect reuse): read A once, read B once, write C once.
      bytes = 96*N^2

These are estimates to position points on a roofline plot; the manuscript should
state the assumptions (or replace them with profiler-derived traffic).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MATMUL_DIR_RE = re.compile(r"^matmul_(\d+)x(\d+)$")


@dataclass(frozen=True)
class MatmulPoint:
    n: int
    time_ns: float
    flops: int
    bytes_conservative: int
    bytes_ideal: int

    @property
    def gflops(self) -> float:
        # FLOPs / ns == GFLOP/s
        return float(self.flops) / float(self.time_ns)

    @property
    def intensity_conservative(self) -> float:
        return float(self.flops) / float(self.bytes_conservative)

    @property
    def intensity_ideal(self) -> float:
        return float(self.flops) / float(self.bytes_ideal)


def iter_estimates(criterion_dir: Path) -> Iterable[MatmulPoint]:
    for child in sorted(criterion_dir.iterdir()):
        if not child.is_dir():
            continue
        m = MATMUL_DIR_RE.match(child.name)
        if not m:
            continue
        n1 = int(m.group(1))
        n2 = int(m.group(2))
        if n1 != n2:
            # We currently only handle square matmul ids.
            continue

        estimates_path = child / "new" / "estimates.json"
        if not estimates_path.exists():
            continue

        with estimates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        time_ns = float(data["mean"]["point_estimate"])

        # FLOPs = (120 mul + 8 add) * N^3
        flops = 128 * (n1**3)

        bytes_conservative = (64 * (n1**3)) + (32 * (n1**2))
        bytes_ideal = 96 * (n1**2)

        yield MatmulPoint(
            n=n1,
            time_ns=time_ns,
            flops=flops,
            bytes_conservative=bytes_conservative,
            bytes_ideal=bytes_ideal,
        )


def write_csv(points: list[MatmulPoint], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "time_ns",
                "gflops",
                "intensity_conservative",
                "intensity_ideal",
                "flops",
                "bytes_conservative",
                "bytes_ideal",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.n,
                    f"{p.time_ns:.6f}",
                    f"{p.gflops:.6f}",
                    f"{p.intensity_conservative:.6f}",
                    f"{p.intensity_ideal:.6f}",
                    p.flops,
                    p.bytes_conservative,
                    p.bytes_ideal,
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--criterion-dir",
        required=True,
        type=Path,
        help="Path to Criterion group dir, e.g. target/criterion/octonion_matmul",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        type=Path,
        help="Output CSV path, e.g. docs/compiler/figures/octonion_matmul_points.csv",
    )
    args = ap.parse_args()

    criterion_dir: Path = args.criterion_dir
    out_csv: Path = args.out_csv

    if not criterion_dir.exists():
        raise SystemExit(f"criterion dir not found: {criterion_dir}")

    points = sorted(iter_estimates(criterion_dir), key=lambda p: p.n)
    if not points:
        raise SystemExit(
            f"no matmul estimates found under {criterion_dir} (expected matmul_*x*/new/estimates.json)"
        )

    write_csv(points, out_csv)

    print(f"Wrote {len(points)} points -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
