#!/usr/bin/env python3
"""Fold-local Algebra-C continuous associator fidelity probe.

This reads trained hidden-state traces and asks whether a linear readout fitted
on training sites predicts the held-out per-sequence associator scalar.
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
from neurodyn_orientation_topk_prototype import read_manifest


SCHEMA = "neurodyn.algebra_c.fidelity_probe.v1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path, help="associator_targets.tsv")
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--top-frac", type=float, default=0.20)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
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


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma = mean(a)
    mb = mean(b)
    va = sum((x - ma) * (x - ma) for x in a)
    vb = sum((y - mb) * (y - mb) for y in b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def r2_score(y: list[float], pred: list[float]) -> float:
    if not y:
        return 0.0
    mu = mean(y)
    ss_tot = sum((value - mu) * (value - mu) for value in y)
    if ss_tot <= 0.0:
        return 0.0
    ss_res = sum((value - estimate) * (value - estimate) for value, estimate in zip(y, pred, strict=True))
    return 1.0 - ss_res / ss_tot


def sign_auc(y: list[float], scores: list[float]) -> float | None:
    pos = [score for value, score in zip(y, scores, strict=True) if value > 0.0]
    neg = [score for value, score in zip(y, scores, strict=True) if value < 0.0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else None


def calibration_slope(y: list[float], pred: list[float]) -> float:
    mp = mean(pred)
    my = mean(y)
    denom = sum((p - mp) * (p - mp) for p in pred)
    if denom <= 0.0:
        return 0.0
    return sum((p - mp) * (value - my) for value, p in zip(y, pred, strict=True)) / denom


def topk_enrichment(y: list[float], pred: list[float], frac: float) -> float:
    if not y:
        return 0.0
    k = max(1, int(round(len(y) * frac)))
    global_mean = mean([abs(value) for value in y])
    if global_mean <= 0.0:
        return 0.0
    picked = sorted(range(len(pred)), key=lambda idx: abs(pred[idx]), reverse=True)[:k]
    return mean([abs(y[idx]) for idx in picked]) / global_mean


def standardize(train_x: list[list[float]], rows_x: list[list[float]]) -> list[list[float]]:
    dim = len(train_x[0]) if train_x else 0
    mu = [mean([row[d] for row in train_x]) for d in range(dim)]
    sd: list[float] = []
    for d in range(dim):
        var = mean([(row[d] - mu[d]) * (row[d] - mu[d]) for row in train_x])
        sd.append(math.sqrt(var) if var > 1.0e-18 else 1.0)
    return [[(row[d] - mu[d]) / sd[d] for d in range(dim)] for row in rows_x]


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(a)
    aug = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1.0e-12:
            continue
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) <= 1.0e-18:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_ridge(train_x: list[list[float]], train_y: list[float], ridge: float) -> list[float]:
    if not train_x:
        return []
    p = len(train_x[0]) + 1
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for x, y in zip(train_x, train_y, strict=True):
        xb = [1.0] + x
        for i in range(p):
            xty[i] += xb[i] * y
            for j in range(p):
                xtx[i][j] += xb[i] * xb[j]
    for i in range(1, p):
        xtx[i][i] += ridge
    return solve_linear(xtx, xty)


def predict(weights: list[float], x: list[float]) -> float:
    return weights[0] + sum(w * value for w, value in zip(weights[1:], x, strict=True))


def subject_meta(reference_manifest: Path, targets: Path) -> dict[int, dict[str, Any]]:
    manifest_rows = read_manifest(reference_manifest)
    target_rows = {row["subject_id"]: row for row in read_tsv(targets)}
    out: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(manifest_rows):
        target = target_rows.get(row["subject_id"])
        if target is None:
            raise SystemExit(f"missing associator target for {row['subject_id']}")
        out[idx] = {
            "subject_id": row["subject_id"],
            "site": row["site"],
            "target_scalar": float(target["target_scalar"]),
            "target_sign": int(target["target_sign"]),
            "target_component": int(target["target_component"]),
        }
    return out


def collect_rows(run: Path, condition: str, meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    state_rows = parse_state_rows(run / "brain_ossm_abide.raw.txt")
    out: list[dict[str, Any]] = []
    for row in state_rows:
        m = meta.get(int(row["subject_index"]))
        if m is None:
            continue
        out.append(
            {
                "condition": condition,
                "model": row["model"],
                "seed": int(row["seed"]),
                "subject_id": m["subject_id"],
                "site": m["site"],
                "x": list(row["h"]),
                "y": float(m["target_scalar"]),
                "target_sign": int(m["target_sign"]),
                "target_component": int(m["target_component"]),
            }
        )
    return out


def evaluate(rows: list[dict[str, Any]], ridge: float, top_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)
    for (condition, model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        y_all: list[float] = []
        pred_all: list[float] = []
        for site in sites:
            train = [row for row in chunk if row["site"] != site]
            test = [row for row in chunk if row["site"] == site]
            if not train or not test:
                continue
            train_x_raw = [row["x"] for row in train]
            train_x = standardize(train_x_raw, train_x_raw)
            weights = fit_ridge(train_x, [row["y"] for row in train], ridge)
            test_x = standardize(train_x_raw, [row["x"] for row in test])
            for row, x in zip(test, test_x, strict=True):
                pred = predict(weights, x)
                y_all.append(row["y"])
                pred_all.append(pred)
                detail.append(
                    {
                        "condition": condition,
                        "model": model,
                        "seed": seed,
                        "holdout_site": site,
                        "subject_id": row["subject_id"],
                        "target_scalar": row["y"],
                        "prediction": pred,
                        "target_sign": row["target_sign"],
                        "target_component": row["target_component"],
                    }
                )
        auc = sign_auc(y_all, pred_all)
        seed_rows.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "subject_count": len(y_all),
                "site_count": len(sites),
                "spearman": spearman(y_all, pred_all),
                "r2": r2_score(y_all, pred_all),
                "sign_auc": "" if auc is None else auc,
                "calibration_slope": calibration_slope(y_all, pred_all),
                "topk_enrichment": topk_enrichment(y_all, pred_all, top_frac),
            }
        )
    return detail, seed_rows


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["model"])].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model), chunk in sorted(groups.items()):
        numeric_keys = ["spearman", "r2", "calibration_slope", "topk_enrichment"]
        row: dict[str, Any] = {
            "condition": condition,
            "model": model,
            "seed_count": len(chunk),
            "subject_count_mean": mean([float(item["subject_count"]) for item in chunk]),
        }
        for key in numeric_keys:
            vals = [float(item[key]) for item in chunk]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_std"] = pstdev(vals)
        auc_vals = [float(item["sign_auc"]) for item in chunk if item["sign_auc"] != ""]
        row["sign_auc_mean"] = "" if not auc_vals else mean(auc_vals)
        row["sign_auc_std"] = "" if not auc_vals else pstdev(auc_vals)
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts must match")
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    meta = subject_meta(args.reference_manifest, args.targets)
    rows: list[dict[str, Any]] = []
    for run, condition in zip(args.run, args.condition, strict=True):
        rows.extend(collect_rows(run, condition, meta))
    detail, seed_rows = evaluate(rows, args.ridge, args.top_frac)
    summary_rows = summarize(seed_rows)
    write_tsv(out / "fidelity_predictions.tsv", detail)
    write_tsv(out / "fidelity_per_seed.tsv", seed_rows)
    write_tsv(out / "fidelity_summary.tsv", summary_rows)
    payload = {
        "schema": SCHEMA,
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "targets": str(args.targets),
        "targets_sha256": sha256_file(args.targets),
        "ridge": args.ridge,
        "top_frac": args.top_frac,
        "runs": [
            {"condition": condition, "path": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")}
            for run, condition in zip(args.run, args.condition, strict=True)
        ],
        "summary": summary_rows,
    }
    (out / "fidelity_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
