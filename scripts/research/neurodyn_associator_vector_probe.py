#!/usr/bin/env python3
"""Fit a fold-local associator-vector readout from hidden pair deltas.

This is a posthoc diagnostic, not a training change.  It asks whether the
trained hidden states contain a linearly readable signed associator vector:
target[assoc_dim] = sign(positive_assoc_value), all other components zero.
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
from neurodyn_orientation_topk_prototype import orientation, pair_id, read_manifest


SCHEMA = "neurodyn.associator_vector_probe.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--associator-triples", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_triple_meta(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        out[f"assoc_pair_{int(row['pair_id']):04d}"] = row
    return out


def subject_meta(reference_manifest: Path, triples: Path) -> dict[int, dict[str, Any]]:
    rows = read_manifest(reference_manifest)
    triple_meta = read_triple_meta(triples)
    out: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        pid = pair_id(row["subject_id"])
        tm = triple_meta.get(pid)
        if tm is None:
            raise SystemExit(f"missing associator triple metadata for {pid}")
        assoc_value = float(tm["positive_assoc_value"])
        assoc_dim = int(tm["assoc_dim"])
        out[idx] = {
            "subject_id": row["subject_id"],
            "pair_id": pid,
            "orientation": orientation(row["subject_id"]),
            "site": row["site"],
            "assoc_dim": assoc_dim,
            "assoc_sign": 1 if assoc_value > 0.0 else -1,
            "target": [1.0 if d == assoc_dim and assoc_value > 0.0 else -1.0 if d == assoc_dim else 0.0 for d in range(8)],
        }
    return out


def collect_pair_rows(run: Path, condition: str, meta: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    state_rows = parse_state_rows(run / "brain_ossm_abide.raw.txt")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows:
        m = meta.get(int(row["subject_index"]))
        if m is None:
            continue
        grouped[(row["model"], int(row["seed"]))][f"{m['pair_id']}::{m['orientation']}"] = {**row, **m}

    out: list[dict[str, Any]] = []
    for (model, seed), chunk in sorted(grouped.items()):
        for pid in sorted({v["pair_id"] for v in chunk.values()}):
            pos = chunk.get(f"{pid}::positive")
            neg = chunk.get(f"{pid}::negative")
            if pos is None or neg is None:
                continue
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": pos["site"],
                    "assoc_dim": int(pos["assoc_dim"]),
                    "assoc_sign": int(pos["assoc_sign"]),
                    "x": [a - b for a, b in zip(pos["h"], neg["h"], strict=True)],
                    "y": list(pos["target"]),
                }
            )
    return out


def standardize(train_x: list[list[float]], rows_x: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    dim = len(train_x[0]) if train_x else 0
    mu = [mean([row[d] for row in train_x]) for d in range(dim)]
    sd: list[float] = []
    for d in range(dim):
        var = mean([(row[d] - mu[d]) * (row[d] - mu[d]) for row in train_x])
        s = math.sqrt(var)
        sd.append(s if s > 1.0e-9 else 1.0)
    out = [[(row[d] - mu[d]) / sd[d] for d in range(dim)] for row in rows_x]
    return out, mu, sd


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


def fit_ridge(train_x: list[list[float]], train_y: list[list[float]], ridge: float) -> list[list[float]]:
    if not train_x:
        return []
    p = len(train_x[0]) + 1
    ydim = len(train_y[0])
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [[0.0 for _ in range(ydim)] for _ in range(p)]
    for x, y in zip(train_x, train_y, strict=True):
        xb = [1.0] + x
        for i in range(p):
            for j in range(p):
                xtx[i][j] += xb[i] * xb[j]
            for k in range(ydim):
                xty[i][k] += xb[i] * y[k]
    for i in range(1, p):
        xtx[i][i] += ridge
    weights: list[list[float]] = []
    for k in range(ydim):
        weights.append(solve_linear(xtx, [xty[i][k] for i in range(p)]))
    return weights


def predict(weights: list[list[float]], x: list[float]) -> list[float]:
    xb = [1.0] + x
    return [sum(w[i] * xb[i] for i in range(len(xb))) for w in weights]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(v * v for v in a))
    nb = math.sqrt(sum(v * v for v in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


def evaluate(rows: list[dict[str, Any]], ridge: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    for (condition, model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        correct_dim = 0
        correct_sign = 0
        correct_both = 0
        total = 0
        cosines: list[float] = []
        margins: list[float] = []
        for site in sites:
            train = [row for row in chunk if row["site"] != site]
            test = [row for row in chunk if row["site"] == site]
            if not train or not test:
                continue
            train_x_raw = [row["x"] for row in train]
            train_x, _, _ = standardize(train_x_raw, train_x_raw)
            weights = fit_ridge(train_x, [row["y"] for row in train], ridge)
            test_x, _, _ = standardize(train_x_raw, [row["x"] for row in test])
            for row, x in zip(test, test_x, strict=True):
                pred = predict(weights, x)
                pred_dim = max(range(len(pred)), key=lambda d: abs(pred[d]))
                pred_sign = 1 if pred[pred_dim] >= 0.0 else -1
                true_dim = int(row["assoc_dim"])
                true_sign = int(row["assoc_sign"])
                dim_ok = int(pred_dim == true_dim)
                sign_ok = int((1 if pred[true_dim] >= 0.0 else -1) == true_sign)
                both_ok = int(dim_ok == 1 and pred_sign == true_sign)
                sorted_abs = sorted((abs(v) for v in pred), reverse=True)
                margin = sorted_abs[0] - (sorted_abs[1] if len(sorted_abs) > 1 else 0.0)
                c = cosine(pred, row["y"])
                correct_dim += dim_ok
                correct_sign += sign_ok
                correct_both += both_ok
                total += 1
                cosines.append(c)
                margins.append(margin)
                detail.append(
                    {
                        "condition": condition,
                        "model": model,
                        "seed": seed,
                        "holdout_site": site,
                        "pair_id": row["pair_id"],
                        "assoc_dim": true_dim,
                        "assoc_sign": true_sign,
                        "pred_dim": pred_dim,
                        "pred_sign": pred_sign,
                        "dim_correct": dim_ok,
                        "sign_correct": sign_ok,
                        "both_correct": both_ok,
                        "cosine": c,
                        "margin": margin,
                        "pred_vector": ",".join(f"{v:.8f}" for v in pred),
                    }
                )
        seed_rows.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "pair_count": total,
                "dim_accuracy": 100.0 * correct_dim / total if total else 0.0,
                "sign_accuracy": 100.0 * correct_sign / total if total else 0.0,
                "both_accuracy": 100.0 * correct_both / total if total else 0.0,
                "cosine_mean": mean(cosines),
                "cosine_std": pstdev(cosines),
                "margin_mean": mean(margins),
            }
        )
    return seed_rows, detail


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["model"])].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model), rows in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "model": model,
                "seed_count": len(rows),
                "pair_count_mean": mean([float(r["pair_count"]) for r in rows]),
                "dim_accuracy_mean": mean([float(r["dim_accuracy"]) for r in rows]),
                "dim_accuracy_std": pstdev([float(r["dim_accuracy"]) for r in rows]),
                "sign_accuracy_mean": mean([float(r["sign_accuracy"]) for r in rows]),
                "sign_accuracy_std": pstdev([float(r["sign_accuracy"]) for r in rows]),
                "both_accuracy_mean": mean([float(r["both_accuracy"]) for r in rows]),
                "both_accuracy_std": pstdev([float(r["both_accuracy"]) for r in rows]),
                "cosine_mean": mean([float(r["cosine_mean"]) for r in rows]),
                "cosine_std": pstdev([float(r["cosine_mean"]) for r in rows]),
                "margin_mean": mean([float(r["margin_mean"]) for r in rows]),
            }
        )
    return out


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary}):
        true = [row for row in summary if row["condition"] == "true" and row["model"] == model]
        nulls = [row for row in summary if row["condition"].startswith("null") and row["model"] == model]
        if not true or not nulls:
            continue
        t = true[0]
        for metric in ("dim_accuracy_mean", "sign_accuracy_mean", "both_accuracy_mean", "cosine_mean", "margin_mean"):
            tv = float(t[metric])
            vals = [float(row[metric]) for row in nulls]
            ge = sum(1 for v in vals if v >= tv)
            out.append(
                {
                    "model": model,
                    "metric": metric,
                    "true": tv,
                    "null_count": len(vals),
                    "null_min": min(vals),
                    "null_max": max(vals),
                    "null_mean": mean(vals),
                    "null_ge_true": ge,
                    "plus_one_p_ge_true": (ge + 1.0) / (len(vals) + 1.0),
                }
            )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts differ")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "associator_vector_probe_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")

    meta = subject_meta(args.reference_manifest, args.associator_triples)
    all_rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        all_rows.extend(collect_pair_rows(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})

    seed_rows, detail = evaluate(all_rows, args.ridge)
    summary = summarize(seed_rows)
    envelope = null_envelope(summary)

    write_tsv(out / "associator_vector_probe_seed_detail.tsv", seed_rows)
    write_tsv(out / "associator_vector_probe_pair_detail.tsv", detail)
    write_tsv(out / "associator_vector_probe_summary.tsv", summary)
    write_tsv(out / "associator_vector_probe_null_envelope.tsv", envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic posthoc associator-vector hidden readout diagnostic only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "ridge": args.ridge,
        "runs": runs,
        "summary": summary,
        "null_envelope": envelope,
    }
    (out / "associator_vector_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Associator Vector Probe", "", payload["claim_boundary"], ""]
    for row in summary:
        md.append(
            f"- `{row['condition']}` `{row['model']}` dim `{float(row['dim_accuracy_mean']):.6f}`, "
            f"sign `{float(row['sign_accuracy_mean']):.6f}`, both `{float(row['both_accuracy_mean']):.6f}`, "
            f"cosine `{float(row['cosine_mean']):.6f}`"
        )
    for row in envelope:
        md.append(
            f"- envelope `{row['model']}` `{row['metric']}` true `{float(row['true']):.6f}`, "
            f"null max `{float(row['null_max']):.6f}`, null_ge `{row['null_ge_true']}/{row['null_count']}`"
        )
    md.append("")
    (out / "associator_vector_probe.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
