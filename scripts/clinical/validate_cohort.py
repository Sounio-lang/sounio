#!/usr/bin/env python3
"""Validate TDM cohort CSV against schema + clinical-range checks.

This script runs identical checks against synthetic and real-world
cohort CSVs. It is a pre-flight gate that must pass before the
Sounio Knightian prediction pipeline ingests any data.

Usage:
    python scripts/clinical/validate_cohort.py <path_to_csv>
    python scripts/clinical/validate_cohort.py --strict <path_to_csv>

Exit codes:
    0 — all checks passed
    1 — schema violation (missing columns or wrong types)
    2 — range violation (clinically implausible values)
    3 — distribution violation (cohort distribution far from
        published popPK literature; only with --strict)

Schema (matches both synthetic v2 and MIMIC-IV ETL output):
    patient_id              str
    age                     int     (years, 18-110)
    sex                     str     ('M' or 'F')
    weight_kg               float   (30-250)
    height_cm               int     (140-220)
    scr_mg_dl               float   (0.3-10.0)
    crcl_ml_min             int     (5-400; ARC up to ~350 in ICU)
    dose_mg                 int     (250-4000)
    interval_h              int     (8, 12, 24)
    measured_cmin_mg_l      float   (0.5-100)
    sofa                    int     (0-24)
    nephrotoxic_coexposure  int     (0 or 1)
    outcome_cure            int     (0 or 1)
    outcome_aki_kdigo       int     (0, 1, 2, or 3 — KDIGO stage)

Distribution checks (--strict):
    Cmin distribution should overlap published ICU TDM literature:
      sub-therapeutic (< 10 mg/L)   : 20-40%
      therapeutic (10-20 mg/L)      : 30-55%
      supra-therapeutic (> 20 mg/L) : 15-40%

    AKI rate should be in the 8-35% range (Filippone 2017,
    Lodise 2014; upper end captures ICU heavy-cohort populations
    with high baseline AKI risk). Mortality (1 - cure) should be
    5-30% (van Hal 2013, Chuang 2014).

Math-review record:
    None required at the validation script level — this enforces
    the dataset SCHEMA and clinical RANGE bounds, not any
    epistemic claim. The downstream Knightian pipeline already
    has its own math-review checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


SCHEMA_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "weight_kg",
    "height_cm",
    "scr_mg_dl",
    "crcl_ml_min",
    "dose_mg",
    "interval_h",
    "measured_cmin_mg_l",
    "sofa",
    "nephrotoxic_coexposure",
    "outcome_cure",
    "outcome_aki_kdigo",
]


@dataclass
class RangeBound:
    name: str
    lo: float
    hi: float
    integer: bool = False


RANGE_BOUNDS = [
    RangeBound("age", 18, 110, integer=True),
    RangeBound("weight_kg", 30.0, 250.0),
    RangeBound("height_cm", 140, 220, integer=True),
    RangeBound("scr_mg_dl", 0.3, 10.0),
    RangeBound("crcl_ml_min", 5, 400, integer=True),
    RangeBound("dose_mg", 250, 4000, integer=True),
    RangeBound("interval_h", 8, 24, integer=True),
    RangeBound("measured_cmin_mg_l", 0.5, 100.0),
    RangeBound("sofa", 0, 24, integer=True),
    RangeBound("nephrotoxic_coexposure", 0, 1, integer=True),
    RangeBound("outcome_cure", 0, 1, integer=True),
    RangeBound("outcome_aki_kdigo", 0, 3, integer=True),
]


def fail(code: int, msg: str) -> None:
    sys.stderr.write(f"VALIDATE FAIL ({code}): {msg}\n")
    sys.exit(code)


def check_schema(header: list[str]) -> None:
    if header != SCHEMA_COLUMNS:
        diff_missing = set(SCHEMA_COLUMNS) - set(header)
        diff_extra = set(header) - set(SCHEMA_COLUMNS)
        msg = f"schema mismatch — missing {sorted(diff_missing)}, " \
              f"extra {sorted(diff_extra)}"
        fail(1, msg)


def parse_value(name: str, raw: str, integer: bool) -> float:
    try:
        if integer:
            return float(int(raw))
        return float(raw)
    except ValueError:
        fail(1, f"column {name}: cannot parse {raw!r} as "
                f"{'int' if integer else 'float'}")
        return 0.0  # unreachable, satisfies type-checker


def check_ranges(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Returns the rows after parsing and range validation. Fails
    on first range violation."""
    materialised = []
    for row_idx, row in enumerate(rows):
        if row.get("sex") not in ("M", "F"):
            fail(2, f"row {row_idx}: sex must be 'M' or 'F', got "
                    f"{row.get('sex')!r}")
        for bound in RANGE_BOUNDS:
            value = parse_value(bound.name, row[bound.name],
                                bound.integer)
            if value < bound.lo or value > bound.hi:
                fail(2, f"row {row_idx}: {bound.name} = {value} "
                        f"outside [{bound.lo}, {bound.hi}]")
        materialised.append(row)
    return materialised


def check_distributions(rows: list[dict[str, str]],
                        strict: bool) -> None:
    n = len(rows)
    if n == 0:
        fail(2, "empty cohort")

    cmin_values = [float(r["measured_cmin_mg_l"]) for r in rows]
    sub = sum(1 for v in cmin_values if v < 10.0) / n
    therapeutic = sum(1 for v in cmin_values
                      if 10.0 <= v <= 20.0) / n
    supra = sum(1 for v in cmin_values if v > 20.0) / n

    aki_rate = sum(int(r["outcome_aki_kdigo"]) > 0
                   for r in rows) / n
    cure_rate = sum(int(r["outcome_cure"]) for r in rows) / n

    print("Distribution summary:")
    print(f"  N                    : {n}")
    print(f"  sub-therapeutic %    : {sub * 100:.1f}%")
    print(f"  therapeutic %        : {therapeutic * 100:.1f}%")
    print(f"  supra-therapeutic %  : {supra * 100:.1f}%")
    print(f"  AKI rate (KDIGO ≥1)  : {aki_rate * 100:.1f}%")
    print(f"  Cure rate            : {cure_rate * 100:.1f}%")

    if not strict:
        return

    issues = []
    if not (0.20 <= sub <= 0.40):
        issues.append(
            f"sub-therapeutic {sub*100:.1f}% outside 20-40%")
    if not (0.30 <= therapeutic <= 0.55):
        issues.append(
            f"therapeutic {therapeutic*100:.1f}% outside 30-55%")
    if not (0.15 <= supra <= 0.40):
        issues.append(
            f"supra-therapeutic {supra*100:.1f}% outside 15-40%")
    if not (0.08 <= aki_rate <= 0.35):
        issues.append(
            f"AKI rate {aki_rate*100:.1f}% outside 8-35% "
            f"(Filippone 2017 + ICU heavy-cohort upper end)")
    if not (0.70 <= cure_rate <= 0.95):
        issues.append(
            f"cure rate {cure_rate*100:.1f}% outside 70-95% "
            f"(van Hal 2013, Chuang 2014)")

    if issues:
        msg = "; ".join(issues)
        fail(3, f"distribution check failed: {msg}")


def check_uniqueness(rows: list[dict[str, str]]) -> None:
    ids = [r["patient_id"] for r in rows]
    counts = Counter(ids)
    duplicates = [pid for pid, n in counts.items() if n > 1]
    if duplicates:
        fail(1, f"duplicate patient_ids: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to cohort CSV")
    parser.add_argument("--strict", action="store_true",
                        help="Enforce distribution checks "
                             "(literature-aligned ranges)")
    args = parser.parse_args()

    with open(args.csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            fail(1, "empty CSV — no header")
        check_schema(list(reader.fieldnames))
        rows = check_ranges(reader)

    check_uniqueness(rows)
    check_distributions(rows, args.strict)

    print(f"\nVALIDATE OK: {args.csv_path} "
          f"({'strict' if args.strict else 'permissive'} mode)")


if __name__ == "__main__":
    main()
