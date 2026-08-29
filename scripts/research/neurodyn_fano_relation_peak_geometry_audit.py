#!/usr/bin/env python3
"""Audit relation-peak geometry across Fano transfer diagnostics.

This is deliberately a descriptive assay gate, not a significance test. It
compares pre-registered true/null groups on the shape of the best directed
relation: transfer, margin to the runner-up, hidden-state cosine/score, and
strong/anti edge counts. If null runs look as peaky as true runs, the relation
winner is treated as non-selective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SCHEMA = "neurodyn.fano_relation_peak_geometry_audit.v1"
CLAIM_BOUNDARY = (
    "Synthetic Fano relation geometry diagnostic only. No clinical, biomarker, "
    "biological mechanism, solved-transfer, global classifier, fixed-relation "
    "replication, or broad O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="GROUP:LABEL:PATH to fano_line_transfer_relation_summary.tsv",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="O-SSM")
    ap.add_argument("--target-relation", default="2->0")
    ap.add_argument("--true-group", default="true")
    ap.add_argument("--null-group", default="null")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_input(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"expected GROUP:LABEL:PATH input, got: {spec}")
    group, label, raw_path = parts
    if not group or not label or not raw_path:
        raise SystemExit(f"group, label, and path must be non-empty: {spec}")
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"input path does not exist: {path}")
    return group, label, path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise SystemExit(f"missing column {key!r}") from exc


def inum(row: dict[str, str], key: str) -> int:
    return int(round(fnum(row, key)))


def relation_summary(group: str, label: str, path: Path, model: str, target: str) -> dict[str, Any]:
    rows = [row for row in read_rows(path) if row.get("model") == model]
    if not rows:
        raise SystemExit(f"no rows for model={model} in {path}")
    rows.sort(key=lambda row: fnum(row, "transfer_accuracy_mean"), reverse=True)
    top = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    target_rows = [row for row in rows if row.get("shared_pos_pair") == target]
    if len(target_rows) != 1:
        raise SystemExit(f"{label}: expected exactly one target row {target}, got {len(target_rows)}")
    target_row = target_rows[0]
    top_transfer = fnum(top, "transfer_accuracy_mean")
    second_transfer = fnum(second, "transfer_accuracy_mean")
    target_transfer = fnum(target_row, "transfer_accuracy_mean")
    target_rank = next(
        idx for idx, row in enumerate(rows, start=1) if row.get("shared_pos_pair") == target
    )
    return {
        "group": group,
        "label": label,
        "model": model,
        "path": str(path),
        "target_relation": target,
        "top_relation": top["shared_pos_pair"],
        "top_transfer": top_transfer,
        "second_relation": second["shared_pos_pair"],
        "second_transfer": second_transfer,
        "top_margin": top_transfer - second_transfer,
        "top_cosine": fnum(top, "cosine_mean"),
        "top_abs_cosine": abs(fnum(top, "cosine_mean")),
        "top_score": fnum(top, "score_mean"),
        "top_abs_score": abs(fnum(top, "score_mean")),
        "top_strong_edge_count": inum(top, "strong_edge_count"),
        "top_anti_transfer_edge_count": inum(top, "anti_transfer_edge_count"),
        "top_edge_count": inum(top, "edge_count"),
        "target_transfer": target_transfer,
        "target_rank": target_rank,
        "target_margin_to_top": target_transfer - top_transfer,
        "target_cosine": fnum(target_row, "cosine_mean"),
        "target_abs_cosine": abs(fnum(target_row, "cosine_mean")),
        "target_score": fnum(target_row, "score_mean"),
        "target_abs_score": abs(fnum(target_row, "score_mean")),
        "target_strong_edge_count": inum(target_row, "strong_edge_count"),
        "target_anti_transfer_edge_count": inum(target_row, "anti_transfer_edge_count"),
        "target_edge_count": inum(target_row, "edge_count"),
        "target_gate_pass": target_transfer > 50.0 and target_rank == 1,
        "relation_count": len(rows),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"run_count": 0}
    fields = [
        "top_transfer",
        "top_margin",
        "top_abs_cosine",
        "top_abs_score",
        "top_strong_edge_count",
        "top_anti_transfer_edge_count",
        "target_transfer",
        "target_abs_cosine",
        "target_abs_score",
        "target_strong_edge_count",
        "target_anti_transfer_edge_count",
    ]
    out: dict[str, Any] = {"run_count": len(rows)}
    for field in fields:
        vals = [float(row[field]) for row in rows]
        out[f"mean_{field}"] = mean(vals)
        out[f"std_{field}"] = pstdev(vals) if len(vals) > 1 else 0.0
        out[f"max_{field}"] = max(vals)
        out[f"min_{field}"] = min(vals)
    out["target_gate_pass_count"] = sum(1 for row in rows if row["target_gate_pass"])
    out["top_relation_counts"] = {
        rel: sum(1 for row in rows if row["top_relation"] == rel)
        for rel in sorted({row["top_relation"] for row in rows})
    }
    return out


def group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    return {group: summarize_group(group_rows) for group, group_rows in sorted(by_group.items())}


def selectivity_decision(payload: dict[str, Any], true_group: str, null_group: str) -> dict[str, Any]:
    groups = payload["group_summary"]
    true = groups.get(true_group, {})
    null = groups.get(null_group, {})
    if not true:
        return {"decision": "PEAK_GEOMETRY_TRUE_GROUP_MISSING", "reason": f"missing group {true_group!r}"}
    if not null:
        return {"decision": "PEAK_GEOMETRY_NULL_GROUP_MISSING", "reason": f"missing group {null_group!r}"}

    true_mean_margin = float(true["mean_top_margin"])
    null_max_margin = float(null["max_top_margin"])
    true_mean_abs_cos = float(true["mean_top_abs_cosine"])
    null_max_abs_cos = float(null["max_top_abs_cosine"])
    true_mean_transfer = float(true["mean_top_transfer"])
    null_max_transfer = float(null["max_top_transfer"])
    target_gate_pass = int(true.get("target_gate_pass_count", 0))
    true_runs = int(true.get("run_count", 0))

    if target_gate_pass < true_runs:
        return {
            "decision": "PEAK_GEOMETRY_NOT_PROMOTED_TARGET_GATE_FAILS",
            "reason": "not all true runs preserve the pre-registered target relation",
            "true_target_gate_pass_count": target_gate_pass,
            "true_run_count": true_runs,
        }
    if true_mean_margin <= null_max_margin or true_mean_transfer <= null_max_transfer:
        return {
            "decision": "PEAK_GEOMETRY_NOT_SELECTIVE_VS_NULL",
            "reason": "null peak transfer or margin reaches/exceeds true mean peak geometry",
            "true_mean_top_transfer": true_mean_transfer,
            "null_max_top_transfer": null_max_transfer,
            "true_mean_top_margin": true_mean_margin,
            "null_max_top_margin": null_max_margin,
            "true_mean_top_abs_cosine": true_mean_abs_cos,
            "null_max_top_abs_cosine": null_max_abs_cos,
        }
    return {
        "decision": "PEAK_GEOMETRY_CANDIDATE_REQUIRES_LARGER_NULL",
        "reason": "true peaks exceed current null envelope on transfer and margin, but this is descriptive only",
        "true_mean_top_transfer": true_mean_transfer,
        "null_max_top_transfer": null_max_transfer,
        "true_mean_top_margin": true_mean_margin,
        "null_max_top_margin": null_max_margin,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Fano Relation Peak Geometry Audit",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        f"Model: `{payload['model']}`",
        f"Target relation: `{payload['target_relation']}`",
        f"Decision: `{payload['decision']['decision']}`",
        "",
        "## Runs",
        "",
        "| group | label | top | top transfer | margin | abs cosine | abs score | strong/anti | target transfer | target rank | target gate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {group} | {label} | `{top_relation}` | {top_transfer:.6f} | {top_margin:.6f} | {top_abs_cosine:.6f} | {top_abs_score:.6f} | {top_strong_edge_count}/{top_anti_transfer_edge_count} | {target_transfer:.6f} | {target_rank} | {gate} |".format(
                group=row["group"],
                label=row["label"],
                top_relation=row["top_relation"],
                top_transfer=row["top_transfer"],
                top_margin=row["top_margin"],
                top_abs_cosine=row["top_abs_cosine"],
                top_abs_score=row["top_abs_score"],
                top_strong_edge_count=row["top_strong_edge_count"],
                top_anti_transfer_edge_count=row["top_anti_transfer_edge_count"],
                target_transfer=row["target_transfer"],
                target_rank=row["target_rank"],
                gate="PASS" if row["target_gate_pass"] else "FAIL",
            )
        )
    lines.extend(["", "## Group Summary", ""])
    lines.append(
        "| group | runs | mean top transfer | max top transfer | mean margin | max margin | mean abs cosine | target gate pass | top relation counts |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for group, item in payload["group_summary"].items():
        counts = ", ".join(f"{rel}:{count}" for rel, count in item.get("top_relation_counts", {}).items())
        lines.append(
            "| {group} | {run_count} | {mean_top_transfer:.6f} | {max_top_transfer:.6f} | {mean_top_margin:.6f} | {max_top_margin:.6f} | {mean_top_abs_cosine:.6f} | {target_gate_pass_count}/{run_count} | {counts} |".format(
                group=group,
                run_count=item["run_count"],
                mean_top_transfer=item["mean_top_transfer"],
                max_top_transfer=item["max_top_transfer"],
                mean_top_margin=item["mean_top_margin"],
                max_top_margin=item["max_top_margin"],
                mean_top_abs_cosine=item["mean_top_abs_cosine"],
                target_gate_pass_count=item["target_gate_pass_count"],
                counts=counts,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["decision"]["reason"],
            "",
            "A peak-geometry signal is not considered useful if pair-label nulls produce comparable or stronger peak transfer/margin, or if true runs fail the pre-registered target gate. This audit is descriptive and should not be used as clinical, mechanistic, or superiority evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = [
        relation_summary(group, label, path, args.model, args.target_relation)
        for group, label, path in map(parse_input, args.input)
    ]
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "model": args.model,
        "target_relation": args.target_relation,
        "true_group": args.true_group,
        "null_group": args.null_group,
        "rows": rows,
        "group_summary": group_summaries(rows),
    }
    payload["decision"] = selectivity_decision(payload, args.true_group, args.null_group)

    (out / "fano_relation_peak_geometry_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(out / "fano_relation_peak_geometry_rows.tsv", rows)
    (out / "fano_relation_peak_geometry_audit.md").write_text(markdown(payload), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
