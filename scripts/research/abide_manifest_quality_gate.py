#!/usr/bin/env python3
"""Fail fast when a Brain O-SSM ABIDE manifest has no usable feature signal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from abide_campaign_lib import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-nonzero-frac", type=float, default=0.01)
    parser.add_argument("--min-variance", type=float, default=1.0e-10)
    parser.add_argument("--min-class-count", type=int, default=1)
    args = parser.parse_args()

    meta, records = load_manifest(args.manifest)
    values: list[float] = []
    label_counts = {0: 0, 1: 0}
    site_counts: dict[str, int] = {}
    for record in records:
        label_counts[record.label] = label_counts.get(record.label, 0) + 1
        site_counts[record.site] = site_counts.get(record.site, 0) + 1
        for step in record.sequence:
            values.extend(step)

    total = len(values)
    finite_values = [value for value in values if math.isfinite(value)]
    finite_count = len(finite_values)
    nonzero_count = sum(1 for value in finite_values if abs(value) > 1.0e-12)
    nonzero_frac = nonzero_count / finite_count if finite_count else 0.0
    mean = sum(finite_values) / finite_count if finite_count else 0.0
    variance = (
        sum((value - mean) * (value - mean) for value in finite_values) / finite_count
        if finite_count
        else 0.0
    )

    result = {
        "schema": "brain_ossm.abide_manifest_quality_gate.v1",
        "manifest": str(Path(args.manifest).resolve()),
        "subject_count": len(records),
        "site_count": len(site_counts),
        "seq_len": meta.seq_len,
        "input_dim": meta.input_dim,
        "feature_value_count": total,
        "finite_feature_value_count": finite_count,
        "nonzero_feature_value_count": nonzero_count,
        "nonzero_feature_frac": nonzero_frac,
        "feature_mean": mean,
        "feature_variance": variance,
        "label_counts": label_counts,
        "thresholds": {
            "min_nonzero_frac": args.min_nonzero_frac,
            "min_variance": args.min_variance,
            "min_class_count": args.min_class_count,
        },
    }

    failures: list[str] = []
    if not records:
        failures.append("no_subjects")
    if finite_count != total:
        failures.append("non_finite_features")
    if nonzero_frac < args.min_nonzero_frac:
        failures.append("low_nonzero_fraction")
    if variance < args.min_variance:
        failures.append("low_feature_variance")
    if label_counts.get(0, 0) < args.min_class_count or label_counts.get(1, 0) < args.min_class_count:
        failures.append("class_count_below_minimum")

    result["status"] = "fail" if failures else "pass"
    result["failures"] = failures
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit("ABIDE manifest quality gate failed: " + ",".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
