#!/usr/bin/env python3
"""Train an explicit fold-local associator-vector head on hidden pair deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from neurodyn_associator_vector_probe import read_triple_meta
from neurodyn_hidden_state_separability import parse_state_rows
from neurodyn_orientation_topk_prototype import orientation, pair_id, read_manifest


SCHEMA = "neurodyn.associator_vector_head_probe.v1"


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
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.001)
    ap.add_argument("--feature-mode", choices=("linear", "poly2"), default="poly2")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


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
        assoc_sign = 1 if assoc_value > 0.0 else -1
        out[idx] = {
            "subject_id": row["subject_id"],
            "pair_id": pid,
            "orientation": orientation(row["subject_id"]),
            "site": row["site"],
            "assoc_dim": assoc_dim,
            "assoc_sign": assoc_sign,
            "signed_class": assoc_dim if assoc_sign > 0 else 8 + assoc_dim,
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
                    "signed_class": int(pos["signed_class"]),
                    "x": [a - b for a, b in zip(pos["h"], neg["h"], strict=True)],
                }
            )
    return out


def featurize(x: list[float], mode: str) -> list[float]:
    if mode == "linear":
        return list(x)
    return list(x) + [abs(v) for v in x] + [v * v for v in x]


def standardize(train_x: list[list[float]], rows_x: list[list[float]]) -> list[list[float]]:
    dim = len(train_x[0]) if train_x else 0
    mu = [mean([row[d] for row in train_x]) for d in range(dim)]
    sd: list[float] = []
    for d in range(dim):
        var = mean([(row[d] - mu[d]) * (row[d] - mu[d]) for row in train_x])
        s = math.sqrt(var)
        sd.append(s if s > 1.0e-9 else 1.0)
    return [[(row[d] - mu[d]) / sd[d] for d in range(dim)] for row in rows_x]


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(max(-60.0, min(60.0, v - m))) for v in logits]
    s = sum(exps)
    return [v / s for v in exps]


def train_softmax(
    xs: list[list[float]],
    ys: list[int],
    class_count: int,
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> list[list[float]]:
    if not xs:
        return []
    p = len(xs[0]) + 1
    rng = random.Random(seed)
    weights = [[rng.uniform(-0.01, 0.01) for _ in range(p)] for _ in range(class_count)]
    n = float(len(xs))
    for _ in range(epochs):
        grad = [[0.0 for _ in range(p)] for _ in range(class_count)]
        for x, y in zip(xs, ys, strict=True):
            xb = [1.0] + x
            probs = softmax([sum(weights[k][i] * xb[i] for i in range(p)) for k in range(class_count)])
            for k in range(class_count):
                err = probs[k] - (1.0 if k == y else 0.0)
                for i in range(p):
                    grad[k][i] += err * xb[i]
        for k in range(class_count):
            for i in range(p):
                reg = 0.0 if i == 0 else l2 * weights[k][i]
                weights[k][i] -= lr * ((grad[k][i] / n) + reg)
    return weights


def predict(weights: list[list[float]], x: list[float]) -> tuple[int, float]:
    xb = [1.0] + x
    logits = [sum(w[i] * xb[i] for i in range(len(xb))) for w in weights]
    probs = softmax(logits)
    pred = max(range(len(probs)), key=lambda k: probs[k])
    sorted_probs = sorted(probs, reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)
    return pred, margin


def signed_to_dim_sign(cls: int) -> tuple[int, int]:
    if cls >= 8:
        return cls - 8, -1
    return cls, 1


def evaluate(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    for (condition, model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        dim_ok = 0
        signed_ok = 0
        sign_ok = 0
        total = 0
        dim_margins: list[float] = []
        signed_margins: list[float] = []
        for site in sites:
            train = [row for row in chunk if row["site"] != site]
            test = [row for row in chunk if row["site"] == site]
            train_raw = [featurize(row["x"], args.feature_mode) for row in train]
            test_raw = [featurize(row["x"], args.feature_mode) for row in test]
            train_x = standardize(train_raw, train_raw)
            test_x = standardize(train_raw, test_raw)
            dim_w = train_softmax(
                train_x,
                [int(row["assoc_dim"]) for row in train],
                8,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
                seed=int(seed) + 11,
            )
            signed_w = train_softmax(
                train_x,
                [int(row["signed_class"]) for row in train],
                16,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
                seed=int(seed) + 29,
            )
            for row, x in zip(test, test_x, strict=True):
                pred_dim, dim_margin = predict(dim_w, x)
                pred_signed, signed_margin = predict(signed_w, x)
                signed_dim, signed_sign = signed_to_dim_sign(pred_signed)
                true_dim = int(row["assoc_dim"])
                true_sign = int(row["assoc_sign"])
                dim_hit = int(pred_dim == true_dim)
                signed_hit = int(pred_signed == int(row["signed_class"]))
                sign_hit = int(signed_sign == true_sign)
                dim_ok += dim_hit
                signed_ok += signed_hit
                sign_ok += sign_hit
                total += 1
                dim_margins.append(dim_margin)
                signed_margins.append(signed_margin)
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
                        "pred_signed_class": pred_signed,
                        "pred_signed_dim": signed_dim,
                        "pred_signed_sign": signed_sign,
                        "dim_correct": dim_hit,
                        "signed_correct": signed_hit,
                        "sign_correct": sign_hit,
                        "dim_margin": dim_margin,
                        "signed_margin": signed_margin,
                    }
                )
        seed_rows.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "pair_count": total,
                "dim_accuracy": 100.0 * dim_ok / total if total else 0.0,
                "signed_accuracy": 100.0 * signed_ok / total if total else 0.0,
                "sign_accuracy": 100.0 * sign_ok / total if total else 0.0,
                "dim_margin_mean": mean(dim_margins),
                "signed_margin_mean": mean(signed_margins),
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
                "signed_accuracy_mean": mean([float(r["signed_accuracy"]) for r in rows]),
                "signed_accuracy_std": pstdev([float(r["signed_accuracy"]) for r in rows]),
                "sign_accuracy_mean": mean([float(r["sign_accuracy"]) for r in rows]),
                "sign_accuracy_std": pstdev([float(r["sign_accuracy"]) for r in rows]),
                "dim_margin_mean": mean([float(r["dim_margin_mean"]) for r in rows]),
                "signed_margin_mean": mean([float(r["signed_margin_mean"]) for r in rows]),
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
    if (out / "associator_vector_head_probe_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    meta = subject_meta(args.reference_manifest, args.associator_triples)
    all_rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        all_rows.extend(collect_pair_rows(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})
    seed_rows, detail = evaluate(all_rows, args)
    summary = summarize(seed_rows)
    write_tsv(out / "associator_vector_head_probe_seed_detail.tsv", seed_rows)
    write_tsv(out / "associator_vector_head_probe_pair_detail.tsv", detail)
    write_tsv(out / "associator_vector_head_probe_summary.tsv", summary)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic posthoc explicit associator-vector head diagnostic only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "feature_mode": args.feature_mode,
        "runs": runs,
        "summary": summary,
    }
    (out / "associator_vector_head_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Associator Vector Head Probe", "", payload["claim_boundary"], ""]
    for row in summary:
        md.append(
            f"- `{row['condition']}` `{row['model']}` dim `{float(row['dim_accuracy_mean']):.6f}`, "
            f"signed `{float(row['signed_accuracy_mean']):.6f}`, sign `{float(row['sign_accuracy_mean']):.6f}`"
        )
    md.append("")
    (out / "associator_vector_head_probe.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
