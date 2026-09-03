#!/usr/bin/env python3
"""Summarize top-relation stability across Fano relation diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.fano_relation_stability_summary.v1"
CLAIM_BOUNDARY = (
    "Synthetic Fano relation diagnostic only. No clinical, biomarker, "
    "biological mechanism, solved-transfer, fixed-relation replication, "
    "global classifier, or broad O-SSM superiority claim."
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
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_input(spec: str) -> tuple[str, str, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"expected GROUP:LABEL:PATH input, got: {spec}")
    group, label, path = parts
    if not group or not label:
        raise SystemExit(f"group and label must be non-empty: {spec}")
    return group, label, Path(path)


def relation_row(group: str, label: str, path: Path, model: str) -> dict[str, Any]:
    rows = [row for row in read_tsv(path) if row.get("model") == model]
    if not rows:
        raise SystemExit(f"no rows for model={model} in {path}")
    rows.sort(key=lambda row: float(row["transfer_accuracy_mean"]), reverse=True)
    top = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    top_transfer = float(top["transfer_accuracy_mean"])
    second_transfer = float(second["transfer_accuracy_mean"])
    top_margin = top_transfer - second_transfer
    return {
        "group": group,
        "label": label,
        "model": model,
        "path": str(path),
        "top_relation": top["shared_pos_pair"],
        "top_transfer": top_transfer,
        "second_relation": second["shared_pos_pair"],
        "second_transfer": second_transfer,
        "top_margin": top_margin,
        "top_above_chance": top_transfer > 50.0,
        "top_gate_pass": top_transfer > 50.0 and top_margin > 0.0,
        "relation_count": len(rows),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    groups: dict[str, Any] = {}
    for group, group_rows in sorted(by_group.items()):
        relation_counts = Counter(str(row["top_relation"]) for row in group_rows)
        gate_pass_rows = [row for row in group_rows if row["top_gate_pass"]]
        groups[group] = {
            "run_count": len(group_rows),
            "top_relation_counts": dict(sorted(relation_counts.items())),
            "top_gate_pass_count": len(gate_pass_rows),
            "top_gate_pass_fraction": len(gate_pass_rows) / len(group_rows) if group_rows else 0.0,
            "mean_top_transfer": sum(float(row["top_transfer"]) for row in group_rows) / len(group_rows),
            "mean_top_margin": sum(float(row["top_margin"]) for row in group_rows) / len(group_rows),
        }
    return groups


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Fano Relation Stability Summary",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "| group | label | top relation | top transfer | second relation | second transfer | margin | gate |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {group} | {label} | `{top_relation}` | {top_transfer:.6f} | `{second_relation}` | {second_transfer:.6f} | {top_margin:.6f} | {gate} |".format(
                group=row["group"],
                label=row["label"],
                top_relation=row["top_relation"],
                top_transfer=row["top_transfer"],
                second_relation=row["second_relation"],
                second_transfer=row["second_transfer"],
                top_margin=row["top_margin"],
                gate="PASS" if row["top_gate_pass"] else "FAIL",
            )
        )
    lines.extend(["", "## Group Summary", ""])
    lines.extend(["| group | runs | top gate pass | mean top transfer | mean top margin | top relation counts |"])
    lines.extend(["| --- | ---: | ---: | ---: | ---: | --- |"])
    for group, item in payload["group_summary"].items():
        counts = ", ".join(f"{rel}:{count}" for rel, count in item["top_relation_counts"].items())
        lines.append(
            "| {group} | {run_count} | {top_gate_pass_count}/{run_count} | {mean_top_transfer:.6f} | {mean_top_margin:.6f} | {counts} |".format(
                group=group,
                run_count=item["run_count"],
                top_gate_pass_count=item["top_gate_pass_count"],
                mean_top_transfer=item["mean_top_transfer"],
                mean_top_margin=item["mean_top_margin"],
                counts=counts,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This report is relation-agnostic: it asks which directed relation is top in each run. It does not show fixed-relation replication unless the same top relation recurs across independent true manifests under a predeclared rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = [relation_row(group, label, path, args.model) for group, label, path in map(parse_input, args.input)]
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "model": args.model,
        "rows": rows,
        "group_summary": summarize(rows),
    }
    (out / "fano_relation_stability_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(out / "fano_relation_stability_rows.tsv", rows)
    (out / "fano_relation_stability_summary.md").write_text(markdown(payload), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
