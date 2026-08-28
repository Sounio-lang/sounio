#!/usr/bin/env python3
"""Train a small linear readout on frozen Brain O-SSM STATE_TRACE hidden states.

This probe answers whether hidden states contain linearly recoverable signal
when the integrated model readout fails. It trains only on traced hidden states
within each seed and leave-site split; it does not retrain the recurrent model.
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

from neurodyn_hidden_state_separability import parse_state_rows


SCHEMA = "neurodyn.frozen_hidden_linear_readout.v1"
CLAIM_BOUNDARY = (
    "Frozen hidden-state linear readout diagnostic only. This is not a clinical, "
    "mechanistic, biomarker, solved-associator, or O-SSM superiority claim."
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def balanced_accuracy(labels: list[int], preds: list[int]) -> float:
    pos = [i for i, label in enumerate(labels) if label == 1]
    neg = [i for i, label in enumerate(labels) if label == 0]
    if not pos or not neg:
        return 0.0
    tpr = sum(1 for i in pos if preds[i] == 1) / len(pos)
    tnr = sum(1 for i in neg if preds[i] == 0) / len(neg)
    return 100.0 * (tpr + tnr) / 2.0


def auroc(labels: list[int], scores: list[float]) -> float:
    pos = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    neg = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not pos or not neg:
        return 50.0
    wins = 0.0
    total = 0
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
            total += 1
    return 100.0 * wins / total


def zscore(train_x: list[list[float]], values: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    dim = len(train_x[0])
    means = [sum(row[d] for row in train_x) / len(train_x) for d in range(dim)]
    stds: list[float] = []
    for d, m in enumerate(means):
        var = sum((row[d] - m) ** 2 for row in train_x) / len(train_x)
        stds.append(math.sqrt(var) if var > 1.0e-12 else 1.0)
    return [[(row[d] - means[d]) / stds[d] for d in range(dim)] for row in values], means, stds


def train_logistic(
    x: list[list[float]],
    y: list[int],
    *,
    l2: float,
    epochs: int,
    lr: float,
) -> tuple[list[float], float]:
    dim = len(x[0])
    w = [0.0] * dim
    pos_frac = max(1.0e-6, min(1.0 - 1.0e-6, sum(y) / len(y)))
    b = math.log(pos_frac / (1.0 - pos_frac))
    n = len(x)
    for _ in range(epochs):
        gw = [0.0] * dim
        gb = 0.0
        for row, label in zip(x, y, strict=True):
            p = sigmoid(b + sum(wi * xi for wi, xi in zip(w, row, strict=True)))
            err = p - label
            gb += err
            for d, xi in enumerate(row):
                gw[d] += err * xi
        for d in range(dim):
            grad = gw[d] / n + l2 * w[d]
            w[d] -= lr * grad
        b -= lr * (gb / n)
    return w, b


def predict(w: list[float], b: float, x: list[list[float]]) -> tuple[list[int], list[float]]:
    scores = [b + sum(wi * xi for wi, xi in zip(w, row, strict=True)) for row in x]
    return [1 if score >= 0.0 else 0 for score in scores], scores


def fit_select_lambda(train_rows: list[dict[str, Any]], l2_values: list[float], epochs: int, lr: float) -> tuple[float, list[float], float]:
    sites = sorted({row["site"] for row in train_rows})
    best_l2 = l2_values[0]
    best_ba = -1.0
    for l2 in l2_values:
        labels_all: list[int] = []
        preds_all: list[int] = []
        for site in sites:
            inner_train = [row for row in train_rows if row["site"] != site]
            inner_val = [row for row in train_rows if row["site"] == site]
            if not inner_train or not inner_val:
                continue
            x_train_raw = [row["h"] for row in inner_train]
            x_val_raw = [row["h"] for row in inner_val]
            x_train, _, _ = zscore(x_train_raw, x_train_raw)
            x_val, _, _ = zscore(x_train_raw, x_val_raw)
            y_train = [int(row["label"]) for row in inner_train]
            w, b = train_logistic(x_train, y_train, l2=l2, epochs=epochs, lr=lr)
            preds, _ = predict(w, b, x_val)
            labels_all.extend(int(row["label"]) for row in inner_val)
            preds_all.extend(preds)
        ba = balanced_accuracy(labels_all, preds_all)
        if ba > best_ba:
            best_ba = ba
            best_l2 = l2
    x_train_raw = [row["h"] for row in train_rows]
    x_train, _, _ = zscore(x_train_raw, x_train_raw)
    y_train = [int(row["label"]) for row in train_rows]
    w, b = train_logistic(x_train, y_train, l2=best_l2, epochs=epochs, lr=lr)
    return best_l2, w, b


def run_probe(rows: list[dict[str, Any]], l2_values: list[float], epochs: int, lr: float, fixed_l2: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], int(row["seed"]))].append(row)

    for (model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        for holdout_site in sites:
            train_rows = [row for row in chunk if row["site"] != holdout_site]
            test_rows = [row for row in chunk if row["site"] == holdout_site]
            if not train_rows or not test_rows:
                continue
            if fixed_l2 is None:
                selected_l2, w, b = fit_select_lambda(train_rows, l2_values, epochs, lr)
            else:
                selected_l2 = fixed_l2
            x_train_raw = [row["h"] for row in train_rows]
            x_test_raw = [row["h"] for row in test_rows]
            x_train, _, _ = zscore(x_train_raw, x_train_raw)
            # Refit on fold train after selection, then apply train statistics to holdout.
            y_train = [int(row["label"]) for row in train_rows]
            w, b = train_logistic(x_train, y_train, l2=selected_l2, epochs=epochs, lr=lr)
            x_test, _, _ = zscore(x_train_raw, x_test_raw)
            labels = [int(row["label"]) for row in test_rows]
            preds, scores = predict(w, b, x_test)
            detail.append(
                {
                    "model": model,
                    "seed": seed,
                    "holdout_site": holdout_site,
                    "rows": len(test_rows),
                    "selected_l2": selected_l2,
                    "balanced_accuracy": balanced_accuracy(labels, preds),
                    "auroc": auroc(labels, scores),
                    "score_mean": mean(scores),
                    "score_std": pstdev(scores),
                    "positive_pred_frac": mean([1.0 if pred == 1 else 0.0 for pred in preds]),
                    "weight_norm": math.sqrt(sum(value * value for value in w)),
                }
            )

    summary: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        by_model[str(row["model"])].append(row)
    for model, model_rows in sorted(by_model.items()):
        bas = [float(row["balanced_accuracy"]) for row in model_rows]
        aucs = [float(row["auroc"]) for row in model_rows]
        summary.append(
            {
                "model": model,
                "fold_count": len(model_rows),
                "balanced_accuracy_mean": mean(bas),
                "balanced_accuracy_std": pstdev(bas),
                "auroc_mean": mean(aucs),
                "auroc_std": pstdev(aucs),
                "weight_norm_mean": mean([float(row["weight_norm"]) for row in model_rows]),
            }
        )
    return detail, summary


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-output", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--l2", default="0.0,0.001,0.01,0.1,1.0")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--models", default="", help="Optional comma-separated model filter, e.g. O-SSM")
    ap.add_argument("--fixed-l2", type=float, default=None, help="Skip inner model selection and use this L2 value")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    l2_values = [float(value) for value in args.l2.split(",") if value.strip()]

    rows = parse_state_rows(args.raw_output)
    if args.models.strip():
        keep = {value.strip() for value in args.models.split(",") if value.strip()}
        rows = [row for row in rows if row["model"] in keep]
        if not rows:
            raise SystemExit(f"no STATE_TRACE rows left after --models={args.models!r}")
    detail, summary = run_probe(rows, l2_values, args.epochs, args.lr, args.fixed_l2)
    if not detail:
        raise SystemExit("no probe rows generated")

    write_tsv(args.output_dir / "frozen_hidden_linear_readout_detail.tsv", detail)
    write_tsv(args.output_dir / "frozen_hidden_linear_readout_summary.tsv", summary)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_output": str(args.raw_output),
        "l2_values": l2_values,
        "fixed_l2": args.fixed_l2,
        "epochs": args.epochs,
        "lr": args.lr,
        "state_trace_rows": len(rows),
        "summary": summary,
    }
    (args.output_dir / "frozen_hidden_linear_readout.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Frozen Hidden Linear Readout",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Raw output: `{args.raw_output}`",
        f"- STATE_TRACE rows: `{len(rows)}`",
        f"- L2 grid: `{','.join(str(v) for v in l2_values)}`",
        f"- Epochs: `{args.epochs}`",
        f"- LR: `{args.lr}`",
        "",
        "| Model | folds | BA mean | BA std | AUROC mean | AUROC std | weight norm mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        md.append(
            "| {model} | {fold_count} | {ba:.6f} | {ba_std:.6f} | {auc:.6f} | {auc_std:.6f} | {wn:.6f} |".format(
                model=row["model"],
                fold_count=row["fold_count"],
                ba=float(row["balanced_accuracy_mean"]),
                ba_std=float(row["balanced_accuracy_std"]),
                auc=float(row["auroc_mean"]),
                auc_std=float(row["auroc_std"]),
                wn=float(row["weight_norm_mean"]),
            )
        )
    (args.output_dir / "frozen_hidden_linear_readout.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sha_rows = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        sha_rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
