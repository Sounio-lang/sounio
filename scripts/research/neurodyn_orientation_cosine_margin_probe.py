#!/usr/bin/env python3
"""Evaluate cosine-margin readouts over associator-sign-aligned hidden deltas."""

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

from neurodyn_frozen_hidden_linear_readout import auroc, balanced_accuracy
from neurodyn_hidden_state_separability import parse_state_rows
from neurodyn_orientation_topk_prototype import orientation, pair_id, parse_k_values, read_manifest


SCHEMA = "neurodyn.orientation_cosine_margin_probe.v1"


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


def unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(value * value for value in vec))
    if n <= 1.0e-12:
        return [0.0 for _ in vec]
    return [value / n for value in vec]


def masked_unit(vec: list[float], dims: list[int]) -> list[float]:
    out = [0.0 for _ in vec]
    for dim in dims:
        out[dim] = vec[dim]
    return unit(out)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(row[d] for row in vectors) / len(vectors) for d in range(dim)]


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
            "assoc_sign": "1" if float(tm["positive_assoc_value"]) > 0.0 else "-1",
            "triple": tm["triple"],
        }
    return out


def collect_deltas(run: Path, condition: str, meta: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
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
            assoc_sign = float(pos["assoc_sign"])
            raw_delta = [a - b for a, b in zip(pos["h"], neg["h"], strict=True)]
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": pos["site"],
                    "assoc_dim": pos["assoc_dim"],
                    "assoc_sign": pos["assoc_sign"],
                    "triple": pos["triple"],
                    "x": [assoc_sign * value for value in raw_delta],
                }
            )
    return out


def pair_number(pair_id_value: str) -> int:
    return int(pair_id_value.rsplit("_", 1)[1])


def train_threshold(scores: list[float]) -> float:
    if not scores:
        return 0.0
    sorted_scores = sorted(scores)
    candidates = [sorted_scores[0] - 1.0e-9, sorted_scores[-1] + 1.0e-9]
    candidates.extend((a + b) / 2.0 for a, b in zip(sorted_scores, sorted_scores[1:]))
    best = 0.0
    best_acc = -1.0
    for cand in candidates:
        acc = sum(1 for score in scores if score >= cand) / len(scores)
        if acc > best_acc:
            best_acc = acc
            best = cand
    return best


def fold_result(train: list[dict[str, Any]], test: list[dict[str, Any]], *, k: int, threshold_mode: str) -> dict[str, Any] | None:
    if not train or not test:
        return None
    proto_full = unit(mean_vec([unit(row["x"]) for row in train]))
    if not proto_full:
        return None
    dims = sorted(range(len(proto_full)), key=lambda idx: abs(proto_full[idx]), reverse=True)[: min(k, len(proto_full))]
    proto = masked_unit(proto_full, dims)
    train_scores = [dot(masked_unit(row["x"], dims), proto) for row in train]
    test_scores = [dot(masked_unit(row["x"], dims), proto) for row in test]
    threshold = train_threshold(train_scores) if threshold_mode == "train_max_acc" else 0.0
    labels = [1 for _ in test_scores]
    preds = [1 if score >= threshold else 0 for score in test_scores]
    # A one-class aligned target has no balanced accuracy. Report aligned-hit rate
    # as accuracy and use mean margin for continuous null envelopes.
    hit = 100.0 * sum(preds) / len(preds) if preds else 0.0
    return {
        "rows": len(test_scores),
        "aligned_hit_rate": hit,
        "score_mean": mean(test_scores),
        "score_std": pstdev(test_scores),
        "threshold": threshold,
        "train_score_mean": mean(train_scores),
        "train_score_std": pstdev(train_scores),
        "dims": ",".join(str(dim) for dim in dims),
    }


def evaluate(rows: list[dict[str, Any]], *, k_values: list[int], threshold_modes: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)
    for (condition, model, seed), chunk in sorted(groups.items()):
        for threshold_mode in threshold_modes:
            for k in k_values:
                for mode in ("global_pair_group_holdout", "leave_dim_global"):
                    folds: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
                    if mode == "global_pair_group_holdout":
                        for fold in range(7):
                            folds.append(
                                (
                                    f"group_{fold}",
                                    [row for row in chunk if pair_number(row["pair_id"]) % 7 != fold],
                                    [row for row in chunk if pair_number(row["pair_id"]) % 7 == fold],
                                )
                            )
                    else:
                        for dim in sorted({row["assoc_dim"] for row in chunk}):
                            folds.append(
                                (
                                    f"dim_{dim}",
                                    [row for row in chunk if row["assoc_dim"] != dim],
                                    [row for row in chunk if row["assoc_dim"] == dim],
                                )
                            )
                    for fold_id, train, test in folds:
                        result = fold_result(train, test, k=k, threshold_mode=threshold_mode)
                        if result is None:
                            continue
                        detail.append(
                            {
                                "condition": condition,
                                "model": model,
                                "seed": seed,
                                "threshold_mode": threshold_mode,
                                "mode": mode,
                                "k": k,
                                "fold_id": fold_id,
                                **result,
                            }
                        )
    summary: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        by_key[(row["condition"], row["model"], row["threshold_mode"], row["mode"], int(row["k"]))].append(row)
    for (condition, model, threshold_mode, mode, k), rows_for_key in sorted(by_key.items()):
        summary.append(
            {
                "condition": condition,
                "model": model,
                "threshold_mode": threshold_mode,
                "mode": mode,
                "k": k,
                "fold_count": len(rows_for_key),
                "aligned_hit_rate_mean": mean([float(row["aligned_hit_rate"]) for row in rows_for_key]),
                "aligned_hit_rate_std": pstdev([float(row["aligned_hit_rate"]) for row in rows_for_key]),
                "score_mean": mean([float(row["score_mean"]) for row in rows_for_key]),
                "score_std": pstdev([float(row["score_mean"]) for row in rows_for_key]),
                "threshold_mean": mean([float(row["threshold"]) for row in rows_for_key]),
            }
        )
    return detail, summary


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["model"], row["threshold_mode"], row["mode"], int(row["k"])) for row in summary})
    for model, threshold_mode, mode, k in keys:
        true = [
            row for row in summary
            if row["condition"] == "true"
            and row["model"] == model
            and row["threshold_mode"] == threshold_mode
            and row["mode"] == mode
            and int(row["k"]) == k
        ]
        nulls = [
            row for row in summary
            if row["condition"].startswith("null")
            and row["model"] == model
            and row["threshold_mode"] == threshold_mode
            and row["mode"] == mode
            and int(row["k"]) == k
        ]
        if not true or not nulls:
            continue
        for metric in ("aligned_hit_rate_mean", "score_mean"):
            tv = float(true[0][metric])
            vals = [float(row[metric]) for row in nulls]
            ge = sum(1 for value in vals if value >= tv)
            out.append(
                {
                    "model": model,
                    "threshold_mode": threshold_mode,
                    "mode": mode,
                    "k": k,
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
    ap.add_argument("--models", default="O-SSM")
    ap.add_argument("--k-values", default="1,2,4,8,16")
    ap.add_argument("--threshold-modes", default="zero,train_max_acc")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts differ")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "orientation_cosine_margin_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    meta = subject_meta(args.reference_manifest, args.associator_triples)
    rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        rows.extend(collect_deltas(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})
    if args.models.strip():
        keep = {value.strip() for value in args.models.split(",") if value.strip()}
        rows = [row for row in rows if row["model"] in keep]
    k_values = parse_k_values(args.k_values)
    threshold_modes = [value.strip() for value in args.threshold_modes.split(",") if value.strip()]
    detail, summary = evaluate(rows, k_values=k_values, threshold_modes=threshold_modes)
    envelope = null_envelope(summary)
    write_tsv(out / "orientation_cosine_margin_detail.tsv", detail)
    write_tsv(out / "orientation_cosine_margin_summary.tsv", summary)
    write_tsv(out / "orientation_cosine_margin_null_envelope.tsv", envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic sign-aligned cosine-margin probe only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "k_values": k_values,
        "threshold_modes": threshold_modes,
        "models": args.models,
        "runs": runs,
        "summary": summary,
        "null_envelope": envelope,
    }
    (out / "orientation_cosine_margin.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Orientation Cosine Margin Probe", "", payload["claim_boundary"], ""]
    for row in envelope:
        md.append(
            f"- `{row['model']}` `{row['threshold_mode']}` `{row['mode']}` k=`{row['k']}` `{row['metric']}`: "
            f"true `{float(row['true']):.6f}`, null max `{float(row['null_max']):.6f}`, "
            f"null_ge `{row['null_ge_true']}/{row['null_count']}`, p+1 `{float(row['plus_one_p_ge_true']):.6f}`"
        )
    md.append("")
    (out / "orientation_cosine_margin.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
