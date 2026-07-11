#!/usr/bin/env python3
"""Fixed-orientation hidden-state contrast for octonionic associator runs."""

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

from neurodyn_hidden_state_separability import parse_state_rows
from neurodyn_frozen_hidden_linear_readout import (
    auroc,
    balanced_accuracy,
    predict,
    train_logistic,
    zscore,
)


SCHEMA = "neurodyn.associator_orientation_hidden_contrast.v1"
CLAIM_BOUNDARY = (
    "Synthetic associator orientation hidden-state diagnostic only. This is not a "
    "clinical, biomarker, mechanistic, null-validated, or broad O-SSM claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Original true-orientation manifest.")
    ap.add_argument("--run", action="append", required=True, type=Path, help="Run directory with brain_ossm_abide.raw.txt.")
    ap.add_argument("--condition", action="append", required=True, help="Condition label for matching --run.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--fixed-l2", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
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


def orientation_label(subject_id: str) -> int:
    if subject_id.endswith("__positive"):
        return 1
    if subject_id.endswith("__negative"):
        return 0
    raise SystemExit(f"subject id lacks associator orientation suffix: {subject_id}")


def pair_id(subject_id: str) -> str:
    if "__" not in subject_id:
        raise SystemExit(f"subject id lacks pair suffix: {subject_id}")
    return subject_id.split("__", 1)[0]


def attach_orientation(rows: list[dict[str, Any]], manifest_rows: list[dict[str, str]], condition: str, run: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["subject_index"])
        if idx < 0 or idx >= len(manifest_rows):
            raise SystemExit(f"subject index out of range: {idx}")
        subject_id = manifest_rows[idx]["subject_id"]
        item = dict(row)
        item["condition"] = condition
        item["run"] = str(run)
        item["subject_id"] = subject_id
        item["pair_id"] = pair_id(subject_id)
        item["orientation_label"] = orientation_label(subject_id)
        out.append(item)
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(vec[d] for vec in vectors) / len(vectors) for d in range(dim)]


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def centroid_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)
    for (condition, model, seed), chunk in sorted(groups.items()):
        for holdout_site in sorted({row["site"] for row in chunk}):
            train = [row for row in chunk if row["site"] != holdout_site]
            test = [row for row in chunk if row["site"] == holdout_site]
            pos = centroid([row["h"] for row in train if row["orientation_label"] == 1])
            neg = centroid([row["h"] for row in train if row["orientation_label"] == 0])
            if not pos or not neg:
                continue
            labels: list[int] = []
            preds: list[int] = []
            scores: list[float] = []
            for row in test:
                dpos = dist2(row["h"], pos)
                dneg = dist2(row["h"], neg)
                score = dneg - dpos
                labels.append(int(row["orientation_label"]))
                preds.append(1 if score >= 0.0 else 0)
                scores.append(score)
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "holdout_site": holdout_site,
                    "probe": "centroid",
                    "rows": len(test),
                    "balanced_accuracy": balanced_accuracy(labels, preds),
                    "auroc": auroc(labels, scores),
                    "score_mean": mean(scores),
                    "score_std": pstdev(scores),
                }
            )
    return out


def logistic_probe(rows: list[dict[str, Any]], *, fixed_l2: float, epochs: int, lr: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)
    for (condition, model, seed), chunk in sorted(groups.items()):
        for holdout_site in sorted({row["site"] for row in chunk}):
            train = [row for row in chunk if row["site"] != holdout_site]
            test = [row for row in chunk if row["site"] == holdout_site]
            if not train or not test:
                continue
            x_train_raw = [row["h"] for row in train]
            x_test_raw = [row["h"] for row in test]
            x_train, _, _ = zscore(x_train_raw, x_train_raw)
            x_test, _, _ = zscore(x_train_raw, x_test_raw)
            y_train = [int(row["orientation_label"]) for row in train]
            y_test = [int(row["orientation_label"]) for row in test]
            w, b = train_logistic(x_train, y_train, l2=fixed_l2, epochs=epochs, lr=lr)
            preds, scores = predict(w, b, x_test)
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "holdout_site": holdout_site,
                    "probe": "frozen_logistic",
                    "rows": len(test),
                    "balanced_accuracy": balanced_accuracy(y_test, preds),
                    "auroc": auroc(y_test, scores),
                    "score_mean": mean(scores),
                    "score_std": pstdev(scores),
                    "weight_norm": math.sqrt(sum(v * v for v in w)),
                }
            )
    return out


def summarize(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(row["condition"], row["model"], row["probe"])].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model, probe), chunk in sorted(grouped.items()):
        bas = [float(row["balanced_accuracy"]) for row in chunk]
        aucs = [float(row["auroc"]) for row in chunk]
        out.append(
            {
                "condition": condition,
                "model": model,
                "probe": probe,
                "fold_count": len(chunk),
                "balanced_accuracy_mean": mean(bas),
                "balanced_accuracy_std": pstdev(bas),
                "auroc_mean": mean(aucs),
                "auroc_std": pstdev(aucs),
                "score_mean": mean([float(row["score_mean"]) for row in chunk]),
            }
        )
    return out


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["probe"]) for row in summary})
    for model, probe in keys:
        true = [row for row in summary if row["condition"] == "true" and row["model"] == model and row["probe"] == probe]
        nulls = [row for row in summary if row["condition"].startswith("null") and row["model"] == model and row["probe"] == probe]
        if not true or not nulls:
            continue
        t = true[0]
        null_ba = [float(row["balanced_accuracy_mean"]) for row in nulls]
        null_auc = [float(row["auroc_mean"]) for row in nulls]
        true_ba = float(t["balanced_accuracy_mean"])
        true_auc = float(t["auroc_mean"])
        ba_ge = sum(1 for value in null_ba if value >= true_ba)
        auc_ge = sum(1 for value in null_auc if value >= true_auc)
        out.append(
            {
                "model": model,
                "probe": probe,
                "true_balanced_accuracy_mean": true_ba,
                "null_balanced_accuracy_mean": mean(null_ba),
                "null_balanced_accuracy_max": max(null_ba),
                "null_ba_ge_true_count": ba_ge,
                "empirical_ba_p_ge_true": (1.0 + ba_ge) / (1.0 + len(nulls)),
                "true_auroc_mean": true_auc,
                "null_auroc_mean": mean(null_auc),
                "null_auroc_max": max(null_auc),
                "null_auroc_ge_true_count": auc_ge,
                "empirical_auroc_p_ge_true": (1.0 + auc_ge) / (1.0 + len(nulls)),
            }
        )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# NeuroDyn Associator Orientation Hidden Contrast",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "## Summary",
        "",
        "| condition | model | probe | BA | AUROC |",
        "|---|---|---|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {condition} | {model} | {probe} | {ba:.6f} | {auc:.6f} |".format(
                condition=row["condition"],
                model=row["model"],
                probe=row["probe"],
                ba=row["balanced_accuracy_mean"],
                auc=row["auroc_mean"],
            )
        )
    lines.extend(["", "## Null Envelope", "", "| model | probe | true BA | null max BA | p BA | true AUROC | null max AUROC | p AUROC |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["null_envelope"]:
        lines.append(
            "| {model} | {probe} | {tba:.6f} | {nba:.6f} | {pba:.6f} | {tauc:.6f} | {nauc:.6f} | {pauc:.6f} |".format(
                model=row["model"],
                probe=row["probe"],
                tba=row["true_balanced_accuracy_mean"],
                nba=row["null_balanced_accuracy_max"],
                pba=row["empirical_ba_p_ge_true"],
                tauc=row["true_auroc_mean"],
                nauc=row["null_auroc_max"],
                pauc=row["empirical_auroc_p_ge_true"],
            )
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
    rows: list[dict[str, Any]] = []
    for run, condition in zip(args.run, args.condition, strict=True):
        rows.extend(attach_orientation(parse_state_rows(run / "brain_ossm_abide.raw.txt"), manifest_rows, condition, run))
    details = centroid_probe(rows) + logistic_probe(rows, fixed_l2=args.fixed_l2, epochs=args.epochs, lr=args.lr)
    summary = summarize(details)
    envelope = null_envelope(summary)
    interpretation = (
        "This diagnostic uses the original manifest orientation as the fixed target, "
        "even for pair-label null runs. Centroid probes test geometry directly; frozen "
        "logistic probes ask whether a simple fold-local readout can recover orientation "
        "from hidden states when the integrated readout is not fixed-target selective."
    )
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "runs": [{"condition": c, "run": str(r)} for r, c in zip(args.run, args.condition, strict=True)],
        "fixed_l2": args.fixed_l2,
        "epochs": args.epochs,
        "lr": args.lr,
        "detail_rows": len(details),
        "summary": summary,
        "null_envelope": envelope,
        "interpretation": interpretation,
    }
    (args.output_dir / "associator_orientation_hidden_contrast.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_tsv(args.output_dir / "associator_orientation_hidden_detail.tsv", details)
    write_tsv(args.output_dir / "associator_orientation_hidden_summary.tsv", summary)
    write_tsv(args.output_dir / "associator_orientation_hidden_null_envelope.tsv", envelope)
    (args.output_dir / "associator_orientation_hidden_contrast.md").write_text(markdown(payload), encoding="utf-8")
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in args.output_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
