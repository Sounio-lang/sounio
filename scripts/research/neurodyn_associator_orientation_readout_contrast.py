#!/usr/bin/env python3
"""Contrast trained readout scores on original associator pair orientation.

For pair-label null runs, the evaluated labels are permuted but subject IDs keep
their original ``__positive`` / ``__negative`` suffixes. This probe asks a fixed
target question: does the trained model assign a higher score to the original
associator-positive member of each pair than to its swapped-sign mate?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.associator_orientation_readout_contrast.v1"
CLAIM_BOUNDARY = (
    "Synthetic associator orientation readout diagnostic only. This is not a "
    "clinical, biomarker, mechanistic, null-validated, or broad O-SSM claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Original true-orientation manifest.")
    ap.add_argument("--run", action="append", required=True, type=Path, help="Run directory with brain_ossm_abide.raw.txt.")
    ap.add_argument("--condition", action="append", required=True, help="Condition label for matching --run.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"manifest has no rows: {path}")
    return list(csv.DictReader(lines, delimiter="\t"))


def pair_id(subject_id: str) -> str:
    if "__" not in subject_id:
        raise SystemExit(f"subject id lacks pair suffix: {subject_id}")
    return subject_id.split("__", 1)[0]


def orientation(subject_id: str) -> str:
    if subject_id.endswith("__positive"):
        return "positive"
    if subject_id.endswith("__negative"):
        return "negative"
    raise SystemExit(f"subject id lacks associator orientation suffix: {subject_id}")


def parse_pred_rows(raw_output: Path, manifest_rows: list[dict[str, str]], condition: str, run: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = raw_output.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith("PRED\t"):
            idx += 1
            continue
        parts = [part for part in line.split("\t") if part]
        line_no = idx + 1
        next_idx = idx + 1
        while len(parts) < 9 and next_idx < len(lines) and lines[next_idx].startswith("\t"):
            parts.extend(part for part in lines[next_idx].split("\t") if part)
            next_idx += 1
        if len(parts) != 9:
            raise SystemExit(f"malformed PRED at {raw_output}:{line_no}: expected 9 fields got {len(parts)}")
        _, model, seed, subj_raw, site, label, prob_micros, pred, assoc_micros = parts
        subj = int(subj_raw)
        if subj < 0 or subj >= len(manifest_rows):
            raise SystemExit(f"subject index out of range at {raw_output}:{idx}: {subj}")
        subject_id = manifest_rows[subj]["subject_id"]
        rows.append(
            {
                "condition": condition,
                "run": str(run),
                "model": model,
                "seed": int(seed),
                "subject_index": subj,
                "subject_id": subject_id,
                "pair_id": pair_id(subject_id),
                "orientation": orientation(subject_id),
                "site": site,
                "eval_label": int(label),
                "prob": int(prob_micros) / 1_000_000.0,
                "pred": int(pred),
                "assoc": int(assoc_micros) / 1_000_000.0,
            }
        )
        idx = next_idx
    if not rows:
        raise SystemExit(f"no PRED rows found in {raw_output}")
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def pair_contrasts(pred_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in pred_rows:
        key = (row["condition"], row["model"], row["seed"], row["pair_id"])
        by_key[key][row["orientation"]] = row

    out: list[dict[str, Any]] = []
    for (condition, model, seed, pid), parts in sorted(by_key.items()):
        if "positive" not in parts or "negative" not in parts:
            continue
        pos = parts["positive"]
        neg = parts["negative"]
        margin = float(pos["prob"]) - float(neg["prob"])
        if margin > 0.0:
            acc = 1.0
        elif margin < 0.0:
            acc = 0.0
        else:
            acc = 0.5
        out.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "pair_id": pid,
                "site": pos["site"],
                "positive_prob": pos["prob"],
                "negative_prob": neg["prob"],
                "orientation_margin": margin,
                "orientation_correct": acc,
                "positive_eval_label": pos["eval_label"],
                "negative_eval_label": neg["eval_label"],
            }
        )
    if not out:
        raise SystemExit("no complete positive/negative prediction pairs found")
    return out


def summarize_pairs(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[(row["condition"], row["model"], int(row["seed"]))].append(row)

    seed_rows: list[dict[str, Any]] = []
    for (condition, model, seed), chunk in sorted(by_seed.items()):
        seed_rows.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "pair_count": len(chunk),
                "orientation_accuracy_pct": 100.0 * mean([float(row["orientation_correct"]) for row in chunk]),
                "orientation_margin_mean": mean([float(row["orientation_margin"]) for row in chunk]),
                "orientation_margin_std": pstdev([float(row["orientation_margin"]) for row in chunk]),
            }
        )

    by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_model[(row["condition"], row["model"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (condition, model), chunk in sorted(by_model.items()):
        accs = [float(row["orientation_accuracy_pct"]) for row in chunk]
        margins = [float(row["orientation_margin_mean"]) for row in chunk]
        summary_rows.append(
            {
                "condition": condition,
                "model": model,
                "seed_count": len(chunk),
                "pair_count_per_seed": int(chunk[0]["pair_count"]) if chunk else 0,
                "orientation_accuracy_pct_mean": mean(accs),
                "orientation_accuracy_pct_std": pstdev(accs),
                "orientation_margin_mean": mean(margins),
                "orientation_margin_std": pstdev(margins),
            }
        )
    return seed_rows, summary_rows


def null_envelope(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_rows = [row for row in summary_rows if row["condition"] == "true" and row["model"] == "O-SSM"]
    null_rows = [row for row in summary_rows if row["condition"].startswith("null") and row["model"] == "O-SSM"]
    if not true_rows or not null_rows:
        return {}
    true = true_rows[0]
    null_acc = [float(row["orientation_accuracy_pct_mean"]) for row in null_rows]
    null_margin = [float(row["orientation_margin_mean"]) for row in null_rows]
    true_acc = float(true["orientation_accuracy_pct_mean"])
    true_margin = float(true["orientation_margin_mean"])
    acc_ge = sum(1 for value in null_acc if value >= true_acc)
    margin_ge = sum(1 for value in null_margin if value >= true_margin)
    return {
        "true_o_orientation_accuracy_pct": true_acc,
        "true_o_orientation_margin_mean": true_margin,
        "null_count": len(null_rows),
        "null_o_orientation_accuracy_pct_mean": mean(null_acc),
        "null_o_orientation_accuracy_pct_max": max(null_acc),
        "null_o_orientation_margin_mean": mean(null_margin),
        "null_o_orientation_margin_max": max(null_margin),
        "null_accuracy_ge_true_count": acc_ge,
        "null_margin_ge_true_count": margin_ge,
        "empirical_accuracy_p_ge_true": (1.0 + acc_ge) / (1.0 + len(null_rows)),
        "empirical_margin_p_ge_true": (1.0 + margin_ge) / (1.0 + len(null_rows)),
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
        "# NeuroDyn Associator Orientation Readout Contrast",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Summary",
        "",
        "| condition | model | orientation accuracy | margin mean |",
        "|---|---|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {condition} | {model} | {acc:.6f} | {margin:.9f} |".format(
                condition=row["condition"],
                model=row["model"],
                acc=row["orientation_accuracy_pct_mean"],
                margin=row["orientation_margin_mean"],
            )
        )
    if payload["null_envelope"]:
        env = payload["null_envelope"]
        lines.extend(
            [
                "",
                "## Null Envelope",
                "",
                f"- true O orientation accuracy: `{env['true_o_orientation_accuracy_pct']:.6f}`",
                f"- null O orientation accuracy mean: `{env['null_o_orientation_accuracy_pct_mean']:.6f}`",
                f"- null O orientation accuracy max: `{env['null_o_orientation_accuracy_pct_max']:.6f}`",
                f"- accuracy plus-one p_ge_true: `{env['empirical_accuracy_p_ge_true']:.6f}`",
                f"- true O orientation margin: `{env['true_o_orientation_margin_mean']:.9f}`",
                f"- null O orientation margin mean: `{env['null_o_orientation_margin_mean']:.9f}`",
                f"- null O orientation margin max: `{env['null_o_orientation_margin_max']:.9f}`",
                f"- margin plus-one p_ge_true: `{env['empirical_margin_p_ge_true']:.6f}`",
            ]
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts must match")
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(args.manifest)
    pred_rows: list[dict[str, Any]] = []
    for run, condition in zip(args.run, args.condition, strict=True):
        pred_rows.extend(parse_pred_rows(run / "brain_ossm_abide.raw.txt", manifest_rows, condition, run))
    pair_rows = pair_contrasts(pred_rows)
    seed_rows, summary_rows = summarize_pairs(pair_rows)
    envelope = null_envelope(summary_rows)
    interpretation = (
        "This fixed-orientation contrast asks whether trained readout probabilities rank "
        "the original associator-positive member of each pair above the swapped-sign member. "
        "For pair-label null runs, evaluation labels are ignored here; the target remains the "
        "original manifest orientation. This is an associator-specific diagnostic, not a "
        "standalone superiority claim."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "runs": [{"condition": c, "run": str(r)} for r, c in zip(args.run, args.condition, strict=True)],
        "pair_rows": len(pair_rows),
        "seed_rows": len(seed_rows),
        "summary": summary_rows,
        "null_envelope": envelope,
        "interpretation": interpretation,
    }
    (args.output_dir / "associator_orientation_readout_contrast.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(args.output_dir / "associator_orientation_readout_pair_detail.tsv", pair_rows)
    write_tsv(args.output_dir / "associator_orientation_readout_seed_summary.tsv", seed_rows)
    write_tsv(args.output_dir / "associator_orientation_readout_summary.tsv", summary_rows)
    if envelope:
        write_tsv(args.output_dir / "associator_orientation_readout_null_envelope.tsv", [envelope])
    (args.output_dir / "associator_orientation_readout_contrast.md").write_text(markdown(payload), encoding="utf-8")
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in args.output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
