#!/usr/bin/env python3
"""Generate reproducible Bateman-vs-Euler error tables for the IEEE paper."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple


DOSE_MG = 500.0
KA_PER_H = 1.0
KE_PER_H = 0.1
CHECKPOINTS_H = (2.0, 6.0, 12.0, 24.0)


def bateman_reference(t_h: float) -> Tuple[float, float]:
    gut = DOSE_MG * math.exp(-KA_PER_H * t_h)
    central = (
        DOSE_MG
        * KA_PER_H
        / (KA_PER_H - KE_PER_H)
        * (math.exp(-KE_PER_H * t_h) - math.exp(-KA_PER_H * t_h))
    )
    return gut, central


def euler_solution(dt_h: float, t_final_h: float = 24.0) -> Dict[float, Tuple[float, float]]:
    n_steps = int(round(t_final_h / dt_h))
    t = 0.0
    gut = DOSE_MG
    central = 0.0
    checkpoints: Dict[float, Tuple[float, float]] = {}

    for _ in range(n_steps):
        dgut = -KA_PER_H * gut
        dcentral = KA_PER_H * gut - KE_PER_H * central
        gut += dgut * dt_h
        central += dcentral * dt_h
        t += dt_h

        for cp in CHECKPOINTS_H:
            if cp not in checkpoints and abs(t - cp) <= (dt_h / 2.0 + 1e-12):
                checkpoints[cp] = (gut, central)

    return checkpoints


def rows_for_dt(dt_h: float) -> Iterable[dict]:
    approx = euler_solution(dt_h)
    for cp in CHECKPOINTS_H:
        gut_sim, central_sim = approx[cp]
        gut_ref, central_ref = bateman_reference(cp)
        central_rel_err_pct = (
            abs(central_sim - central_ref) / central_ref * 100.0 if central_ref != 0 else 0.0
        )
        yield {
            "dt_h": dt_h,
            "time_h": cp,
            "gut_sim": gut_sim,
            "central_sim": central_sim,
            "gut_ref": gut_ref,
            "central_ref": central_ref,
            "central_rel_err_pct": central_rel_err_pct,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to stdout.",
    )
    args = parser.parse_args()

    all_rows = list(rows_for_dt(0.1)) + list(rows_for_dt(0.01))
    fieldnames = [
        "dt_h",
        "time_h",
        "gut_sim",
        "central_sim",
        "gut_ref",
        "central_ref",
        "central_rel_err_pct",
    ]

    if args.output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
