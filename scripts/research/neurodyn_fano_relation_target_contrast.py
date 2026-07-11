#!/usr/bin/env python3
"""Summarize a target Fano relation against non-target relation drift.

The Fano diagnostics emit one row per model/relation. This helper turns those
rows into a compact target-vs-distractor table so a follow-up auxiliary can be
designed around the relation that actually failed, rather than around generic
clean/noisy invariance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "Synthetic Fano relation diagnostic only; no clinical, biomarker, "
    "biological mechanism, solved-transfer, or broad O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL:PATH",
        help="Condition label plus relation summary TSV path.",
    )
    ap.add_argument("--target-relation", default="2->0")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_input(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise SystemExit(f"--input must be LABEL:PATH, got {spec!r}")
    label, raw_path = spec.split(":", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"empty label in --input {spec!r}")
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"missing input path for {label}: {path}")
    return label, path


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise SystemExit(f"missing column {key!r}") from exc


def summarize_condition(label: str, path: Path, target: str) -> list[dict[str, Any]]:
    rows = read_rows(path)
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        model = row.get("model", "")
        relation = row.get("shared_pos_pair", "")
        if not model or not relation:
            raise SystemExit(f"malformed row in {path}: {row}")
        by_model.setdefault(model, []).append(row)

    out: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        target_rows = [row for row in model_rows if row["shared_pos_pair"] == target]
        if len(target_rows) != 1:
            raise SystemExit(f"{label} {model}: expected one target {target}, got {len(target_rows)}")
        target_row = target_rows[0]
        target_transfer = fnum(target_row, "transfer_accuracy_mean")
        sorted_rows = sorted(model_rows, key=lambda row: fnum(row, "transfer_accuracy_mean"), reverse=True)
        target_rank = 1
        for idx, row in enumerate(sorted_rows, start=1):
            if row["shared_pos_pair"] == target:
                target_rank = idx
                break
        non_target = [row for row in sorted_rows if row["shared_pos_pair"] != target]
        best = non_target[0]
        worst = sorted_rows[-1]
        best_transfer = fnum(best, "transfer_accuracy_mean")
        out.append(
            {
                "condition": label,
                "input_path": str(path),
                "model": model,
                "target_relation": target,
                "target_transfer": target_transfer,
                "target_rank": target_rank,
                "target_strong_edges": int(float(target_row.get("strong_edge_count", "0"))),
                "target_anti_edges": int(float(target_row.get("anti_transfer_edge_count", "0"))),
                "best_non_target_relation": best["shared_pos_pair"],
                "best_non_target_transfer": best_transfer,
                "target_minus_best_non_target": target_transfer - best_transfer,
                "worst_relation": worst["shared_pos_pair"],
                "worst_transfer": fnum(worst, "transfer_accuracy_mean"),
                "relation_count": len(model_rows),
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "condition",
        "model",
        "target_relation",
        "target_transfer",
        "target_rank",
        "target_strong_edges",
        "target_anti_edges",
        "best_non_target_relation",
        "best_non_target_transfer",
        "target_minus_best_non_target",
        "worst_relation",
        "worst_transfer",
        "relation_count",
        "input_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})


def write_markdown(path: Path, rows: list[dict[str, Any]], target: str) -> None:
    lines = [
        "# NeuroDyn Fano Relation Target Contrast",
        "",
        f"Target relation: `{target}`",
        "",
        CLAIM_BOUNDARY,
        "",
        "| condition | model | target transfer | target rank | best non-target | best non-target transfer | margin |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {model} | {target_transfer:.6f} | {target_rank} | `{best_non_target_relation}` | {best_non_target_transfer:.6f} | {target_minus_best_non_target:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Gate",
            "",
            "A target relation is not considered preserved unless its transfer is above chance, "
            "its target rank is `1`, and its margin over the best non-target relation is positive. "
            "Rows failing this gate indicate relation drift and should not trigger null expansion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for spec in args.input:
        label, path = parse_input(spec)
        inputs.append({"label": label, "path": str(path)})
        rows.extend(summarize_condition(label, path, args.target_relation))

    summary = {
        "schema": "neurodyn.fano_relation_target_contrast.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "target_relation": args.target_relation,
        "inputs": inputs,
        "rows": rows,
        "decision": "RELATION_TARGET_CONTRAST_READY",
    }
    (out / "fano_relation_target_contrast.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(out / "fano_relation_target_contrast.tsv", rows)
    write_markdown(out / "fano_relation_target_contrast.md", rows, args.target_relation)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
