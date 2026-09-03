#!/usr/bin/env python3
"""Readiness gate for rich ADHD-200 O-SSM manifests.

This gate checks data support and feature sanity. It is not a model evaluation
and makes no clinical claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.adhd200_manifest_readiness_gate.v1"
CLAIM_BOUNDARY = (
    "Dataset readiness only; no diagnostic, biomarker, mechanism, treatment, "
    "or O-SSM superiority claim."
)


MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-999", "-9999"}


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def parse_float(value: str) -> float | None:
    if is_missing(value):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if not math.isfinite(result):
        return None
    return result


def read_manifest(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        if raw.startswith("#"):
            body = raw[1:].strip()
            if "=" in body:
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
            continue
        data_lines.append(raw)
    if not data_lines:
        raise SystemExit(f"manifest has no tabular rows: {path}")
    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = list(reader)
    return meta, rows


def summarize_numeric(rows: list[dict[str, str]], column: str) -> dict[str, Any]:
    values = [parse_float(row.get(column, "")) for row in rows]
    valid = [value for value in values if value is not None]
    distinct = len({round(value, 12) for value in valid})
    missing = len(values) - len(valid)
    if valid:
        mean = sum(valid) / len(valid)
        variance = sum((value - mean) * (value - mean) for value in valid) / len(valid)
        min_value = min(valid)
        max_value = max(valid)
    else:
        mean = variance = min_value = max_value = None
    return {
        "column": column,
        "row_count": len(rows),
        "valid_count": len(valid),
        "missing_count": missing,
        "missing_frac": missing / len(rows) if rows else 1.0,
        "distinct_count": distinct,
        "mean": mean,
        "variance": variance,
        "min": min_value,
        "max": max_value,
    }


def summarize_numeric_by_site(rows: list[dict[str, str]], column: str) -> list[dict[str, Any]]:
    by_site: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_site.setdefault(row.get("site", ""), []).append(row)
    return [summarize_numeric(site_rows, column) | {"site": site} for site, site_rows in sorted(by_site.items())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--primary-phenotypes",
        default="inattention,hyperactivity_impulsivity,adhd_total",
        help="Comma-separated phenotype columns required for confirmatory readiness.",
    )
    parser.add_argument("--min-sites", type=int, default=2)
    parser.add_argument("--min-class-count", type=int, default=1)
    parser.add_argument("--max-primary-missing-frac", type=float, default=0.35)
    parser.add_argument("--min-primary-distinct", type=int, default=3)
    parser.add_argument("--min-feature-variance", type=float, default=1.0e-10)
    parser.add_argument("--min-feature-nonzero-frac", type=float, default=0.01)
    args = parser.parse_args()

    meta, rows = read_manifest(args.manifest)
    fieldnames = list(rows[0].keys()) if rows else []
    required_base = ["subject_id", "label", "site"]
    primary = [value.strip() for value in args.primary_phenotypes.split(",") if value.strip()]
    feature_cols = [name for name in fieldnames if name.startswith("f") and name[1:].isdigit()]

    failures: list[str] = []
    for column in required_base:
        if column not in fieldnames:
            failures.append(f"missing_required_column:{column}")
    if len(feature_cols) != 64:
        failures.append(f"feature_column_count:{len(feature_cols)}")

    label_counts = Counter(row.get("label", "") for row in rows)
    site_counts = Counter(row.get("site", "") for row in rows)
    if len(site_counts) < args.min_sites:
        failures.append("site_count_below_minimum")
    if label_counts.get("ADHD", 0) < args.min_class_count:
        failures.append("adhd_count_below_minimum")
    if label_counts.get("TD", 0) < args.min_class_count:
        failures.append("td_count_below_minimum")

    phenotype_summary = []
    for column in primary:
        if column not in fieldnames:
            failures.append(f"missing_primary_phenotype:{column}")
            phenotype_summary.append({"column": column, "present": False})
            continue
        summary = summarize_numeric(rows, column)
        summary["present"] = True
        summary["by_site"] = summarize_numeric_by_site(rows, column)
        phenotype_summary.append(summary)
        if summary["missing_frac"] > args.max_primary_missing_frac:
            failures.append(f"primary_missingness_above_threshold:{column}")
        if summary["distinct_count"] < args.min_primary_distinct:
            failures.append(f"primary_distinct_below_minimum:{column}")
        if summary["variance"] is None or summary["variance"] <= 0.0:
            failures.append(f"primary_nonvarying:{column}")
        for site_summary in summary["by_site"]:
            if site_summary["valid_count"] == 0:
                failures.append(f"primary_absent_in_site:{column}:{site_summary['site']}")
            elif site_summary["distinct_count"] < 2:
                failures.append(f"primary_nonvarying_in_site:{column}:{site_summary['site']}")

    feature_values: list[float] = []
    nonfinite_features = 0
    for row in rows:
        for column in feature_cols:
            value = parse_float(row.get(column, ""))
            if value is None:
                nonfinite_features += 1
            else:
                feature_values.append(value)
    if feature_values:
        feature_mean = sum(feature_values) / len(feature_values)
        feature_variance = sum((value - feature_mean) * (value - feature_mean) for value in feature_values) / len(feature_values)
        feature_nonzero_frac = sum(1 for value in feature_values if abs(value) > 1.0e-12) / len(feature_values)
    else:
        feature_mean = 0.0
        feature_variance = 0.0
        feature_nonzero_frac = 0.0
    if nonfinite_features:
        failures.append("nonfinite_features")
    if feature_variance < args.min_feature_variance:
        failures.append("feature_variance_below_minimum")
    if feature_nonzero_frac < args.min_feature_nonzero_frac:
        failures.append("feature_nonzero_fraction_below_minimum")

    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(args.manifest.resolve()),
        "status": "fail" if failures else "pass",
        "failures": failures,
        "metadata": meta,
        "row_count": len(rows),
        "field_count": len(fieldnames),
        "site_count": len(site_counts),
        "site_counts": dict(sorted(site_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "primary_phenotypes": primary,
        "phenotype_summary": phenotype_summary,
        "feature_column_count": len(feature_cols),
        "feature_value_count": len(feature_values) + nonfinite_features,
        "finite_feature_value_count": len(feature_values),
        "nonfinite_feature_value_count": nonfinite_features,
        "feature_mean": feature_mean,
        "feature_variance": feature_variance,
        "feature_nonzero_frac": feature_nonzero_frac,
        "thresholds": {
            "min_sites": args.min_sites,
            "min_class_count": args.min_class_count,
            "max_primary_missing_frac": args.max_primary_missing_frac,
            "min_primary_distinct": args.min_primary_distinct,
            "min_feature_variance": args.min_feature_variance,
            "min_feature_nonzero_frac": args.min_feature_nonzero_frac,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("ADHD-200 manifest readiness gate failed: " + ",".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
