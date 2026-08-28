#!/usr/bin/env python3
"""Audit Algebra-C associator targets before any model smoke.

The gate is deliberately about target quality, not model performance.  It
rejects categorical or one-sided targets before they can masquerade as a
continuous associator-fidelity endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.algebra_c.target_audit.v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", required=True, type=Path, help="associator_targets.tsv")
    ap.add_argument("--manifest", type=Path, help="optional manifest used for hashing/provenance")
    ap.add_argument("--min-distinct-ratio", type=float, default=0.80)
    ap.add_argument("--max-tie-fraction", type=float, default=0.20)
    ap.add_argument("--require-both-signs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require-site-both-signs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    if not rows:
        return {}, [], ["target table is empty"]
    errors: list[str] = []
    values = [float(row["target_scalar"]) for row in rows]
    rounded = [round(value, 10) for value in values]
    distinct_counts = Counter(rounded)
    tied_rows = sum(count for count in distinct_counts.values() if count > 1)
    tie_sensitivity: dict[str, dict[str, float]] = {}
    for digits in (8, 10, 12):
        counts = Counter(round(value, digits) for value in values)
        tied = sum(count for count in counts.values() if count > 1)
        tie_sensitivity[str(digits)] = {
            "distinct_target_values": len(counts),
            "distinct_ratio": len(counts) / len(rows),
            "tie_fraction": tied / len(rows),
        }
    signs = Counter(int(row["target_sign"]) for row in rows)
    sites: dict[str, list[dict[str, str]]] = defaultdict(list)
    components = Counter(int(row["target_component"]) for row in rows)
    for row in rows:
        sites[row["site"]].append(row)
    site_rows: list[dict[str, Any]] = []
    for site, chunk in sorted(sites.items()):
        site_values = [float(row["target_scalar"]) for row in chunk]
        site_signs = Counter(int(row["target_sign"]) for row in chunk)
        site_distinct = len({round(value, 10) for value in site_values})
        site_rows.append(
            {
                "site": site,
                "rows": len(chunk),
                "distinct_target_values": site_distinct,
                "sign_neg": site_signs[-1],
                "sign_pos": site_signs[1],
                "target_min": min(site_values),
                "target_max": max(site_values),
            }
        )
    summary = {
        "row_count": len(rows),
        "site_count": len(sites),
        "target_min": min(values),
        "target_max": max(values),
        "target_mean": statistics.fmean(values),
        "target_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "distinct_target_values": len(distinct_counts),
        "distinct_ratio": len(distinct_counts) / len(rows),
        "tie_fraction": tied_rows / len(rows),
        "sign_counts": {str(key): signs[key] for key in sorted(signs)},
        "target_components": {str(key): components[key] for key in sorted(components)},
        "tie_sensitivity_decimal_places": tie_sensitivity,
    }
    if summary["target_min"] == summary["target_max"]:
        errors.append("target is constant")
    return summary, site_rows, errors


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.targets)
    summary, site_rows, errors = summarize(rows)
    sign_counts = {int(key): value for key, value in summary.get("sign_counts", {}).items()}
    if summary.get("distinct_ratio", 0.0) < args.min_distinct_ratio:
        errors.append(
            f"distinct ratio {summary.get('distinct_ratio', 0.0):.6f} < {args.min_distinct_ratio:.6f}"
        )
    if summary.get("tie_fraction", 1.0) > args.max_tie_fraction:
        errors.append(
            f"tie fraction {summary.get('tie_fraction', 1.0):.6f} > {args.max_tie_fraction:.6f}"
        )
    if args.require_both_signs and (sign_counts.get(-1, 0) == 0 or sign_counts.get(1, 0) == 0):
        errors.append("target must contain both negative and positive signs")
    if args.require_site_both_signs:
        for row in site_rows:
            if int(row["sign_neg"]) == 0 or int(row["sign_pos"]) == 0:
                errors.append(f"site {row['site']} lacks both target signs")
    decision = "ALGEBRA_C_TARGET_AUDIT_PASS" if not errors else "ALGEBRA_C_TARGET_AUDIT_FAIL"
    payload = {
        "schema": SCHEMA,
        "decision": decision,
        "targets": str(args.targets),
        "targets_sha256": sha256_file(args.targets),
        "manifest": str(args.manifest) if args.manifest else None,
        "manifest_sha256": sha256_file(args.manifest) if args.manifest else None,
        "thresholds": {
            "min_distinct_ratio": args.min_distinct_ratio,
            "max_tie_fraction": args.max_tie_fraction,
            "require_both_signs": args.require_both_signs,
            "require_site_both_signs": args.require_site_both_signs,
        },
        "summary": summary,
        "errors": errors,
    }
    (out / "target_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(out / "target_audit_by_site.tsv", site_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
