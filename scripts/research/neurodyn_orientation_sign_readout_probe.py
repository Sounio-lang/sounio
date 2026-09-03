#!/usr/bin/env python3
"""Probe hidden pair deltas with a sign-aligned logistic readout."""

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

from neurodyn_frozen_hidden_linear_readout import auroc, balanced_accuracy, predict, train_logistic, zscore
from neurodyn_hidden_state_separability import parse_state_rows
from neurodyn_orientation_topk_prototype import orientation, pair_id, read_manifest


SCHEMA = "neurodyn.orientation_sign_readout_probe.v1"


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


def read_triple_meta(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {f"assoc_pair_{int(row['pair_id']):04d}": row for row in rows}


def subject_meta(reference_manifest: Path, triples: Path) -> dict[int, dict[str, str]]:
    rows = read_manifest(reference_manifest)
    triple_meta = read_triple_meta(triples)
    out: dict[int, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        pid = pair_id(row["subject_id"])
        tm = triple_meta.get(pid)
        if tm is None:
            raise SystemExit(f"missing triple metadata for {pid}")
        out[idx] = {
            "subject_id": row["subject_id"],
            "pair_id": pid,
            "orientation": orientation(row["subject_id"]),
            "site": row["site"],
            "assoc_dim": tm["assoc_dim"],
            "assoc_sign_label": "1" if float(tm["positive_assoc_value"]) > 0.0 else "0",
            "triple": tm["triple"],
        }
    return out


def collect_pair_deltas(run: Path, condition: str, meta: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in parse_state_rows(run / "brain_ossm_abide.raw.txt"):
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
                    "assoc_dim": pos["assoc_dim"],
                    "triple": pos["triple"],
                    "label": int(pos["assoc_sign_label"]),
                    "x": [a - b for a, b in zip(pos["h"], neg["h"], strict=True)],
                }
            )
    return out


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], *, l2: float, epochs: int, lr: float) -> dict[str, Any] | None:
    y_train = [int(row["label"]) for row in train]
    y_test = [int(row["label"]) for row in test]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return None
    x_train_raw = [row["x"] for row in train]
    x_test_raw = [row["x"] for row in test]
    x_train, _, _ = zscore(x_train_raw, x_train_raw)
    x_test, _, _ = zscore(x_train_raw, x_test_raw)
    w, b = train_logistic(x_train, y_train, l2=l2, epochs=epochs, lr=lr)
    preds, scores = predict(w, b, x_test)
    return {
        "rows": len(test),
        "balanced_accuracy": balanced_accuracy(y_test, preds),
        "accuracy": 100.0 * sum(1 for a, p in zip(y_test, preds, strict=True) if a == p) / len(y_test),
        "auroc": auroc(y_test, scores),
        "score_mean": mean(scores),
        "score_std": pstdev(scores),
        "positive_pred_frac": mean([1.0 if pred == 1 else 0.0 for pred in preds]),
        "weight_norm": math.sqrt(sum(value * value for value in w)),
    }


def pair_number(pair_id_value: str) -> int:
    return int(pair_id_value.rsplit("_", 1)[1])


def evaluate(
    rows: list[dict[str, Any]],
    *,
    l2_values: list[float],
    epochs: int,
    lr: float,
    global_folds: int,
    within_dim_folds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    for (condition, model, seed), chunk in sorted(groups.items()):
        for l2 in l2_values:
            for mode in ("global_pair_group_holdout", "within_dim_group_holdout", "leave_dim_global"):
                folds: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
                if mode == "global_pair_group_holdout":
                    for fold in range(global_folds):
                        test = [row for row in chunk if pair_number(row["pair_id"]) % global_folds == fold]
                        train = [row for row in chunk if pair_number(row["pair_id"]) % global_folds != fold]
                        folds.append((f"group_{fold}", train, test))
                elif mode == "within_dim_group_holdout":
                    for dim in sorted({row["assoc_dim"] for row in chunk}):
                        dim_rows = [row for row in chunk if row["assoc_dim"] == dim]
                        for fold in range(within_dim_folds):
                            test = [row for row in dim_rows if (pair_number(row["pair_id"]) // 7) % within_dim_folds == fold]
                            train = [row for row in dim_rows if (pair_number(row["pair_id"]) // 7) % within_dim_folds != fold]
                            if train and test:
                                folds.append((f"dim_{dim}_group_{fold}", train, test))
                elif mode == "leave_dim_global":
                    for dim in sorted({row["assoc_dim"] for row in chunk}):
                        folds.append((f"dim_{dim}", [row for row in chunk if row["assoc_dim"] != dim], [row for row in chunk if row["assoc_dim"] == dim]))
                for fold_id, train, test in folds:
                    result = fit_predict(train, test, l2=l2, epochs=epochs, lr=lr)
                    if result is None:
                        continue
                    detail.append(
                        {
                            "condition": condition,
                            "model": model,
                            "seed": seed,
                            "l2": l2,
                            "mode": mode,
                            "fold_id": fold_id,
                            **result,
                        }
                    )

    summary: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        by_key[(row["condition"], row["model"], float(row["l2"]), row["mode"])].append(row)
    for (condition, model, l2, mode), rows_for_key in sorted(by_key.items()):
        summary.append(
            {
                "condition": condition,
                "model": model,
                "l2": l2,
                "mode": mode,
                "fold_count": len(rows_for_key),
                "balanced_accuracy_mean": mean([float(row["balanced_accuracy"]) for row in rows_for_key]),
                "balanced_accuracy_std": pstdev([float(row["balanced_accuracy"]) for row in rows_for_key]),
                "accuracy_mean": mean([float(row["accuracy"]) for row in rows_for_key]),
                "accuracy_std": pstdev([float(row["accuracy"]) for row in rows_for_key]),
                "auroc_mean": mean([float(row["auroc"]) for row in rows_for_key]),
                "auroc_std": pstdev([float(row["auroc"]) for row in rows_for_key]),
                "weight_norm_mean": mean([float(row["weight_norm"]) for row in rows_for_key]),
            }
        )
    return detail, summary


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["model"], float(row["l2"]), row["mode"]) for row in summary})
    for model, l2, mode in keys:
        true = [row for row in summary if row["condition"] == "true" and row["model"] == model and float(row["l2"]) == l2 and row["mode"] == mode]
        nulls = [row for row in summary if row["condition"].startswith("null") and row["model"] == model and float(row["l2"]) == l2 and row["mode"] == mode]
        if not true or not nulls:
            continue
        for metric in ("balanced_accuracy_mean", "accuracy_mean", "auroc_mean"):
            tv = float(true[0][metric])
            vals = [float(row[metric]) for row in nulls]
            ge = sum(1 for value in vals if value >= tv)
            out.append(
                {
                    "model": model,
                    "l2": l2,
                    "mode": mode,
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--associator-triples", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--l2", default="0.01,0.1,1.0,10.0")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--global-folds", type=int, default=7)
    ap.add_argument("--within-dim-folds", type=int, default=4)
    ap.add_argument("--models", default="", help="Optional comma-separated model filter, e.g. O-SSM")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts differ")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "orientation_sign_readout_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")

    meta = subject_meta(args.reference_manifest, args.associator_triples)
    rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        rows.extend(collect_pair_deltas(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})
    if args.models.strip():
        keep = {value.strip() for value in args.models.split(",") if value.strip()}
        rows = [row for row in rows if row["model"] in keep]
        if not rows:
            raise SystemExit(f"no rows left after --models={args.models!r}")
    l2_values = [float(value) for value in args.l2.split(",") if value.strip()]
    detail, summary = evaluate(
        rows,
        l2_values=l2_values,
        epochs=args.epochs,
        lr=args.lr,
        global_folds=args.global_folds,
        within_dim_folds=args.within_dim_folds,
    )
    envelope = null_envelope(summary)

    write_tsv(out / "orientation_sign_readout_detail.tsv", detail)
    write_tsv(out / "orientation_sign_readout_summary.tsv", summary)
    write_tsv(out / "orientation_sign_readout_null_envelope.tsv", envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic sign-aligned hidden-delta readout probe only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "l2_values": l2_values,
        "epochs": args.epochs,
        "lr": args.lr,
        "global_folds": args.global_folds,
        "within_dim_folds": args.within_dim_folds,
        "models": args.models,
        "runs": runs,
        "summary": summary,
        "null_envelope": envelope,
    }
    (out / "orientation_sign_readout.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Orientation Sign Readout Probe", "", payload["claim_boundary"], ""]
    for row in envelope:
        md.append(
            f"- `{row['model']}` `{row['mode']}` l2=`{row['l2']}` `{row['metric']}`: "
            f"true `{float(row['true']):.6f}`, null max `{float(row['null_max']):.6f}`, "
            f"null_ge `{row['null_ge_true']}/{row['null_count']}`, p+1 `{float(row['plus_one_p_ge_true']):.6f}`"
        )
    md.append("")
    (out / "orientation_sign_readout.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
