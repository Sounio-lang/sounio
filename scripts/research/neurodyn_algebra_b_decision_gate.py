#!/usr/bin/env python3
"""Apply the preregistered Algebra-B attribution decision table.

The top-level gate has exactly four scientific routes:

1. O-SSM crosses threshold, A8 does not, projection collapses.
2. O-SSM crosses threshold, A8 also crosses threshold.
3. O-SSM is subthreshold and reformulation attempts remain.
4. Reformulation attempts are exhausted while subthreshold.

Null expansion is allowed only after route 1 is reached.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.algebra_b_decision_gate.v1"
OSS_THRESHOLD = 55.0
A8_THRESHOLD = 55.0
PROJECTION_COLLAPSE_THRESHOLD = 55.0
MAX_REFORMULATIONS_DEFAULT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path, help="data_audit/associator_data_audit.json")
    parser.add_argument("--true-run", required=True, type=Path, help="O/H true Slurm run directory")
    parser.add_argument("--null-run", action="append", default=[], type=Path)
    parser.add_argument("--a8-true-run", type=Path)
    parser.add_argument("--a8-null-run", action="append", default=[], type=Path)
    parser.add_argument("--projection-true-run", type=Path)
    parser.add_argument("--projection-null-run", action="append", default=[], type=Path)
    parser.add_argument("--a8-model", default="A8-SSM")
    parser.add_argument("--projection-model", default="O-SSM")
    parser.add_argument("--attempt-count", type=int, default=0)
    parser.add_argument("--max-reformulations", type=int, default=MAX_REFORMULATIONS_DEFAULT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def model_metric(run: Path, model: str) -> dict[str, float]:
    rows = read_tsv(run / "results" / "overall_metrics.tsv")
    for row in rows:
        if row["model"] == model:
            return {
                "balanced_accuracy_pct": float(row["balanced_accuracy_pct_mean"]),
                "auroc_pct": float(row["auroc_pct_mean"]),
                "brier": float(row["brier_mean"]),
                "ece_pct": float(row["ece_pct_mean"]),
            }
    raise SystemExit(f"model={model} not found in {run / 'results' / 'overall_metrics.tsv'}")


def hidden_probe_metric(run: Path, model: str) -> float | None:
    path = run / "hidden_state_separability" / "hidden_state_separability_summary.tsv"
    if not path.exists():
        return None
    rows = read_tsv(path)
    for row in rows:
        if row.get("model") == model:
            key = "nearest_centroid_balanced_accuracy_mean"
            if key in row:
                return float(row[key])
    return None


def envelope(true_value: float, null_values: list[float]) -> dict[str, Any]:
    return {
        "true": true_value,
        "null_count": len(null_values),
        "null_min": min(null_values) if null_values else None,
        "null_max": max(null_values) if null_values else None,
        "null_mean": statistics.fmean(null_values) if null_values else None,
        "null_ge_true": sum(1 for value in null_values if value >= true_value),
        "plus_one_p_ge_true": (1 + sum(1 for value in null_values if value >= true_value)) / (len(null_values) + 1)
        if null_values
        else None,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    a8_ba = payload["attribution"]["a8_balanced_accuracy_pct"]
    projection_ba = payload["attribution"]["projection_balanced_accuracy_pct"]
    a8_ba_text = "NA" if a8_ba is None else f"{a8_ba:.6f}"
    projection_ba_text = "NA" if projection_ba is None else f"{projection_ba:.6f}"
    lines = [
        "# Algebra-B Decision Gate",
        "",
        "Claim boundary: synthetic non-clinical algebra-necessity gate only.",
        "",
        f"Decision: `{payload['decision']}`",
        f"Route: `{payload['route_id']}`",
        "",
        payload["interpretation"],
        "",
        "## Four-Route Contract",
        "",
        "- `1`: O-SSM >= 55%, A8 < 55%, associative projection < 55%; algebraic necessity candidate. Only this route permits 99 nulls.",
        "- `2`: O-SSM >= 55%, A8 >= 55%; dimensionality/capacity effect, not octonionic necessity.",
        "- `3`: O-SSM < 55% and reformulation attempts remain; one new preregistered reformulation is allowed.",
        "- `4`: O-SSM < 55% and reformulation attempts are exhausted; terminal negative for fixed-dim6 design.",
        "",
        "## Core Metrics",
        "",
        "| metric | true | null max | null >= true | plus-one p |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, env in payload["envelopes"].items():
        null_max = "NA" if env["null_max"] is None else f"{env['null_max']:.6f}"
        pval = "NA" if env["plus_one_p_ge_true"] is None else f"{env['plus_one_p_ge_true']:.6f}"
        lines.append(
            f"| {metric} | {env['true']:.6f} | {null_max} | {env['null_ge_true']}/{env['null_count']} | {pval} |"
        )
    lines.extend(
        [
            "",
            "## Attribution Controls",
            "",
            f"- O-SSM BA: `{payload['attribution']['o_balanced_accuracy_pct']:.6f}`",
            f"- H-SSM BA: `{payload['attribution']['h_balanced_accuracy_pct']:.6f}`",
            f"- A8 BA: `{a8_ba_text}`",
            f"- associative projection BA: `{projection_ba_text}`",
            f"- attempt count/max reformulations: `{payload['attempt_count']}` / `{payload['max_reformulations']}`",
            "",
            "## Missing Controls",
            "",
            *[f"- {item}" for item in payload["missing_controls"]],
            "",
            "## Shortcut Audit",
            "",
            f"- raw-flat leave-site BA: `{payload['shortcut_audit']['raw_flat_leave_site_ba']:.6f}`",
            f"- fail shortcut threshold: `55.000000`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    raw_flat_ba = float(audit["leave_site_raw_flat_nearest_centroid"]["balanced_accuracy"])

    true_o = model_metric(args.true_run, "O-SSM")
    true_h = model_metric(args.true_run, "H-SSM")
    true_a8 = model_metric(args.a8_true_run, args.a8_model) if args.a8_true_run is not None else None
    true_projection = (
        model_metric(args.projection_true_run, args.projection_model)
        if args.projection_true_run is not None
        else None
    )
    null_o = [model_metric(path, "O-SSM") for path in args.null_run]
    hidden_true = hidden_probe_metric(args.true_run, "O-SSM")
    hidden_null = [value for path in args.null_run if (value := hidden_probe_metric(path, "O-SSM")) is not None]

    envelopes = {
        "o_balanced_accuracy_pct": envelope(
            true_o["balanced_accuracy_pct"], [row["balanced_accuracy_pct"] for row in null_o]
        ),
        "o_auroc_pct": envelope(true_o["auroc_pct"], [row["auroc_pct"] for row in null_o]),
    }
    if hidden_true is not None:
        envelopes["o_hidden_centroid_ba_pct"] = envelope(hidden_true, hidden_null)

    missing_controls: list[str] = []
    if len(args.null_run) < int(prereg.get("null_count", 99)):
        missing_controls.append(f"99 null runs required; found {len(args.null_run)}")
    if args.a8_true_run is None:
        missing_controls.append("A8-SSM direct-sum associative 8-D true run missing")
    if len(args.a8_null_run) < int(prereg.get("null_count", 99)):
        missing_controls.append(f"A8-SSM null runs incomplete; found {len(args.a8_null_run)}")
    if args.projection_true_run is None:
        missing_controls.append("associative-projection O-SSM true run missing")
    if len(args.projection_null_run) < int(prereg.get("null_count", 99)):
        missing_controls.append(f"associative-projection null runs incomplete; found {len(args.projection_null_run)}")
    if hidden_true is None:
        missing_controls.append("O-SSM hidden-state probe missing")

    fail_shortcut = raw_flat_ba >= OSS_THRESHOLD
    oss_subthreshold = true_o["balanced_accuracy_pct"] < OSS_THRESHOLD
    attempts_exhausted = args.attempt_count >= args.max_reformulations
    o_clears_nulls = (
        len(args.null_run) >= int(prereg.get("null_count", 99))
        and envelopes["o_balanced_accuracy_pct"]["null_ge_true"] == 0
        and envelopes["o_auroc_pct"]["null_ge_true"] == 0
        and envelopes["o_balanced_accuracy_pct"]["plus_one_p_ge_true"] is not None
        and envelopes["o_balanced_accuracy_pct"]["plus_one_p_ge_true"] <= 0.01
    )
    h_gap = true_o["balanced_accuracy_pct"] - true_h["balanced_accuracy_pct"]

    if fail_shortcut:
        route_id = "3" if not attempts_exhausted else "4"
        decision = "ALGEBRA_B_FAIL_SHORTCUT"
        interpretation = (
            "The raw-flat audit crosses the shortcut threshold. Treat this as subthreshold/invalid for the "
            "four-route contract; only a preregistered reformulation may proceed if attempts remain."
        )
    elif oss_subthreshold and attempts_exhausted:
        route_id = "4"
        decision = "ALGEBRA_B_ROUTE4_TERMINAL_FIXEDDIM6_NEGATIVE"
        interpretation = (
            "O-SSM is below 55% and the preregistered reformulation budget is exhausted. "
            "Terminate this fixed-dim6 algebraic-necessity design."
        )
    elif oss_subthreshold:
        route_id = "3"
        decision = "ALGEBRA_B_ROUTE3_SUBTHRESHOLD_REFORMULATION_ALLOWED"
        interpretation = (
            "O-SSM is below 55%. Do not expand nulls. One preregistered reformulation is allowed only "
            "if it increments the attempt counter and fixes seed/threshold before running."
        )
    elif true_a8 is None:
        route_id = "NEEDS_ATTRIBUTION_CONTROLS"
        decision = "ALGEBRA_B_NEEDS_A8_AND_PROJECTION_BEFORE_ROUTE_ASSIGNMENT"
        interpretation = (
            "O-SSM crosses 55%, but route 1 versus route 2 cannot be assigned until A8 and associative "
            "projection controls are present. Do not run 99 nulls yet."
        )
    elif true_projection is None:
        route_id = "NEEDS_PROJECTION_CONTROL"
        decision = "ALGEBRA_B_NEEDS_ASSOCIATIVE_PROJECTION_BEFORE_ROUTE_ASSIGNMENT"
        interpretation = (
            "O-SSM crosses 55% and A8 is below 55%, but route 1 cannot be assigned until the "
            "associative-projection O-SSM true control is present. Do not run 99 nulls yet."
        )
    elif true_a8["balanced_accuracy_pct"] >= A8_THRESHOLD:
        route_id = "2"
        decision = "ALGEBRA_B_ROUTE2_DIMENSIONALITY_NOT_ALGEBRA"
        interpretation = (
            "O-SSM crosses 55%, but the paired associative 8-D A8 control also crosses 55%. "
            "Attribute the effect to dimensionality/capacity, not octonionic necessity."
        )
    elif true_projection["balanced_accuracy_pct"] >= PROJECTION_COLLAPSE_THRESHOLD:
        route_id = "2"
        decision = "ALGEBRA_B_ROUTE2_ASSOCIATIVE_PROJECTION_DOES_NOT_COLLAPSE"
        interpretation = (
            "O-SSM crosses 55% and A8 is below threshold, but the associative projection does not collapse. "
            "The octonionic product is not necessary under this assay."
        )
    elif h_gap < 3.0:
        route_id = "2"
        decision = "ALGEBRA_B_ROUTE2_HSSM_MATCHES_CAPACITY_NOT_ALGEBRA"
        interpretation = "O-SSM crosses 55%, but H-SSM is within the preregistered 3.0 pp margin."
    elif len(args.null_run) == 0:
        route_id = "1"
        decision = "ALGEBRA_B_ROUTE1_ATTRIBUTION_READY_FOR_99_NULLS"
        interpretation = (
            "O-SSM crosses 55%, A8 is below 55%, associative projection collapses, and H-SSM is below the margin. "
            "This is the only route that permits the 99-null expansion."
        )
    elif not o_clears_nulls:
        route_id = "1"
        decision = "ALGEBRA_B_ROUTE1_ATTRIBUTION_POSITIVE_BUT_NULLS_FAIL"
        interpretation = "Attribution controls pass, but the full null envelope does not. Do not promote."
    else:
        route_id = "1"
        decision = "ALGEBRA_B_ROUTE1_NULL_VALIDATED_METHOD_READY"
        interpretation = (
            "O-SSM crosses 55%, A8 is below 55%, associative projection collapses, H-SSM is below the margin, "
            "and 99 nulls collapse. A method piece may be drafted under the synthetic-only claim boundary."
        )

    rows = [
        {"condition": "true", "model": "O-SSM", "run": str(args.true_run), **true_o},
        {"condition": "true", "model": "H-SSM", "run": str(args.true_run), **true_h},
    ]
    if true_a8 is not None:
        rows.append({"condition": "true", "model": args.a8_model, "run": str(args.a8_true_run), **true_a8})
    if true_projection is not None:
        rows.append(
            {
                "condition": "true",
                "model": f"projection:{args.projection_model}",
                "run": str(args.projection_true_run),
                **true_projection,
            }
        )
    rows.extend({"condition": "null", "model": "O-SSM", "run": str(path), **metric} for path, metric in zip(args.null_run, null_o))

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "prereg": str(args.prereg),
        "audit": str(args.audit),
        "true_run": str(args.true_run),
        "null_runs": [str(path) for path in args.null_run],
        "shortcut_audit": {"raw_flat_leave_site_ba": raw_flat_ba},
        "o_minus_h_ba_pp": h_gap,
        "attempt_count": args.attempt_count,
        "max_reformulations": args.max_reformulations,
        "attribution": {
            "o_balanced_accuracy_pct": true_o["balanced_accuracy_pct"],
            "h_balanced_accuracy_pct": true_h["balanced_accuracy_pct"],
            "a8_balanced_accuracy_pct": None if true_a8 is None else true_a8["balanced_accuracy_pct"],
            "projection_balanced_accuracy_pct": None
            if true_projection is None
            else true_projection["balanced_accuracy_pct"],
        },
        "envelopes": envelopes,
        "missing_controls": missing_controls,
        "route_id": route_id,
        "decision": decision,
        "interpretation": interpretation,
    }
    write_tsv(out / "algebra_b_decision_runs.tsv", rows)
    (out / "algebra_b_decision_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "algebra_b_decision_gate.md").write_text(markdown(payload), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
