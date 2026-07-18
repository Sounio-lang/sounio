#!/usr/bin/env python3
"""Audit whether the octonionic associator manifest itself carries the target.

This is intentionally model-free, but not a substitute for model evidence. It
answers a narrower question: before training O-SSM, do the raw and normalized
input trajectories expose the intended associator sign under leave-site splits?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.octonionic_associator_data_audit.v1"


def oct_mul(a: list[float], b: list[float]) -> list[float]:
    a0, a1, a2, a3, a4, a5, a6, a7 = a
    b0, b1, b2, b3, b4, b5, b6, b7 = b
    return [
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6,
        a0 * b2 + a2 * b0 - a1 * b3 + a3 * b1 + a4 * b6 - a6 * b4 + a5 * b7 - a7 * b5,
        a0 * b3 + a3 * b0 + a1 * b2 - a2 * b1 + a4 * b7 - a7 * b4 - a5 * b6 + a6 * b5,
        a0 * b4 + a4 * b0 - a1 * b5 + a5 * b1 - a2 * b6 + a6 * b2 - a3 * b7 + a7 * b3,
        a0 * b5 + a5 * b0 + a1 * b4 - a4 * b1 - a2 * b7 + a7 * b2 + a3 * b6 - a6 * b3,
        a0 * b6 + a6 * b0 + a1 * b7 - a7 * b1 + a2 * b4 - a4 * b2 - a3 * b5 + a5 * b3,
        a0 * b7 + a7 * b0 - a1 * b6 + a6 * b1 + a2 * b5 - a5 * b2 + a3 * b4 - a4 * b3,
    ]


def associator(a: list[float], b: list[float], c: list[float]) -> list[float]:
    left = oct_mul(oct_mul(a, b), c)
    right = oct_mul(a, oct_mul(b, c))
    return [left[i] - right[i] for i in range(8)]


def trajectory_assoc_signal(flat: list[float], normalize_len: bool = True) -> list[float]:
    seq_len = len(flat) // 8
    signal = [0.0] * 8
    for step in range(seq_len - 2):
        av = associator(
            flat[step * 8 : (step + 1) * 8],
            flat[(step + 1) * 8 : (step + 2) * 8],
            flat[(step + 2) * 8 : (step + 3) * 8],
        )
        for i, value in enumerate(av):
            signal[i] += value
    if normalize_len and seq_len:
        signal = [value / seq_len for value in signal]
    return signal


def block_mean(flat: list[float], block_idx: int, block_count: int = 8) -> list[float]:
    seq_len = len(flat) // 8
    start = (block_idx * seq_len) // block_count
    end = ((block_idx + 1) * seq_len) // block_count
    if end <= start:
        end = start + 1
    out = [0.0] * 8
    for step in range(start, end):
        for i, value in enumerate(flat[step * 8 : (step + 1) * 8]):
            out[i] += value
    return [value / (end - start) for value in out]


def core_block_assoc_signal(flat: list[float]) -> list[float]:
    """Associator over the stretched manifest core blocks 1, 2, and 3."""
    return associator(block_mean(flat, 1), block_mean(flat, 2), block_mean(flat, 3))


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t")
        rows: list[dict[str, Any]] = []
        for row in reader:
            features = [float(row[f"f{i}"]) for i in range(256)]
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "pair_id": row["subject_id"].split("__", 1)[0],
                    "label": int(row["label"]),
                    "sign": 1 if int(row["label"]) == 1 else -1,
                    "site": row["site"],
                    "features": features,
                }
            )
    return rows


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    out = [0.0] * len(vectors[0])
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += value
    return [value / len(vectors) for value in out]


def dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return 50.0 * (sens + spec)


def zscore_by_train(train: list[list[float]], values: list[list[float]]) -> list[list[float]]:
    means = mean_vector(train)
    stds: list[float] = []
    for i, mean in enumerate(means):
        var = sum((row[i] - mean) ** 2 for row in train) / max(1, len(train))
        stds.append(math.sqrt(var) if var > 1.0e-12 else 1.0)
    return [[(row[i] - means[i]) / stds[i] for i in range(len(row))] for row in values]


def leave_site_nearest_centroid(rows: list[dict[str, Any]], feature_key: str) -> dict[str, Any]:
    sites = sorted({row["site"] for row in rows})
    detail = []
    all_y: list[int] = []
    all_pred: list[int] = []
    for site in sites:
        train = [row for row in rows if row["site"] != site]
        test = [row for row in rows if row["site"] == site]
        pos = mean_vector([row[feature_key] for row in train if row["label"] == 1])
        neg = mean_vector([row[feature_key] for row in train if row["label"] == 0])
        y_true = [row["label"] for row in test]
        y_pred = [1 if dist2(row[feature_key], pos) <= dist2(row[feature_key], neg) else 0 for row in test]
        all_y.extend(y_true)
        all_pred.extend(y_pred)
        detail.append({"site": site, "n": len(test), "balanced_accuracy": balanced_accuracy(y_true, y_pred)})
    return {"balanced_accuracy": balanced_accuracy(all_y, all_pred), "detail": detail}


def leave_site_normalized_assoc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sites = sorted({row["site"] for row in rows})
    audit_rows: list[dict[str, Any]] = []
    detail = []
    all_y: list[int] = []
    all_pred: list[int] = []
    for site in sites:
        train = [row for row in rows if row["site"] != site]
        test = [row for row in rows if row["site"] == site]
        train_norm = zscore_by_train([row["features"] for row in train], [row["features"] for row in train])
        test_norm = zscore_by_train([row["features"] for row in train], [row["features"] for row in test])
        train_features = [trajectory_assoc_signal(vec) for vec in train_norm]
        test_features = [trajectory_assoc_signal(vec) for vec in test_norm]
        train_aug = [dict(row, norm_assoc=vec) for row, vec in zip(train, train_features)]
        test_aug = [dict(row, norm_assoc=vec) for row, vec in zip(test, test_features)]
        pos = mean_vector([row["norm_assoc"] for row in train_aug if row["label"] == 1])
        neg = mean_vector([row["norm_assoc"] for row in train_aug if row["label"] == 0])
        y_true = [row["label"] for row in test_aug]
        y_pred = [1 if dist2(row["norm_assoc"], pos) <= dist2(row["norm_assoc"], neg) else 0 for row in test_aug]
        all_y.extend(y_true)
        all_pred.extend(y_pred)
        detail.append({"site": site, "n": len(test), "balanced_accuracy": balanced_accuracy(y_true, y_pred)})
        audit_rows.extend(test_aug)
    signed_norms = []
    for row in audit_rows:
        vec = row["norm_assoc"]
        dominant = max(range(8), key=lambda i: abs(vec[i]))
        signed_norms.append(row["sign"] * vec[dominant])
    return {
        "balanced_accuracy": balanced_accuracy(all_y, all_pred),
        "detail": detail,
        "signed_dominant_mean": sum(signed_norms) / len(signed_norms) if signed_norms else 0.0,
        "signed_dominant_min": min(signed_norms) if signed_norms else 0.0,
    }


def leave_site_normalized_core_assoc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sites = sorted({row["site"] for row in rows})
    detail = []
    all_y: list[int] = []
    all_pred: list[int] = []
    signed_norms = []
    for site in sites:
        train = [row for row in rows if row["site"] != site]
        test = [row for row in rows if row["site"] == site]
        train_norm = zscore_by_train([row["features"] for row in train], [row["features"] for row in train])
        test_norm = zscore_by_train([row["features"] for row in train], [row["features"] for row in test])
        train_features = [core_block_assoc_signal(vec) for vec in train_norm]
        test_features = [core_block_assoc_signal(vec) for vec in test_norm]
        train_aug = [dict(row, norm_core_assoc=vec) for row, vec in zip(train, train_features)]
        test_aug = [dict(row, norm_core_assoc=vec) for row, vec in zip(test, test_features)]
        pos = mean_vector([row["norm_core_assoc"] for row in train_aug if row["label"] == 1])
        neg = mean_vector([row["norm_core_assoc"] for row in train_aug if row["label"] == 0])
        y_true = [row["label"] for row in test_aug]
        y_pred = [1 if dist2(row["norm_core_assoc"], pos) <= dist2(row["norm_core_assoc"], neg) else 0 for row in test_aug]
        all_y.extend(y_true)
        all_pred.extend(y_pred)
        detail.append({"site": site, "n": len(test), "balanced_accuracy": balanced_accuracy(y_true, y_pred)})
        for row in test_aug:
            vec = row["norm_core_assoc"]
            dominant = max(range(8), key=lambda i: abs(vec[i]))
            signed_norms.append(row["sign"] * vec[dominant])
    return {
        "balanced_accuracy": balanced_accuracy(all_y, all_pred),
        "detail": detail,
        "signed_dominant_mean": sum(signed_norms) / len(signed_norms) if signed_norms else 0.0,
        "signed_dominant_min": min(signed_norms) if signed_norms else 0.0,
    }


def pair_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    residuals = []
    core_residuals = []
    raw_distances = []
    assoc_distances = []
    core_assoc_distances = []
    bad_pairs = []
    for pair_id, pair_rows in by_pair.items():
        if len(pair_rows) != 2 or {row["label"] for row in pair_rows} != {0, 1}:
            bad_pairs.append(pair_id)
            continue
        pos = next(row for row in pair_rows if row["label"] == 1)
        neg = next(row for row in pair_rows if row["label"] == 0)
        residuals.append(max(abs(pos["raw_assoc"][i] + neg["raw_assoc"][i]) for i in range(8)))
        if "raw_core_assoc" in pos:
            core_residuals.append(max(abs(pos["raw_core_assoc"][i] + neg["raw_core_assoc"][i]) for i in range(8)))
            core_assoc_distances.append(math.sqrt(dist2(pos["raw_core_assoc"], neg["raw_core_assoc"])))
        raw_distances.append(math.sqrt(dist2(pos["features"], neg["features"])))
        assoc_distances.append(math.sqrt(dist2(pos["raw_assoc"], neg["raw_assoc"])))
    return {
        "pair_count": len(by_pair),
        "bad_pair_count": len(bad_pairs),
        "assoc_sign_flip_residual_max": max(residuals) if residuals else None,
        "core_assoc_sign_flip_residual_max": max(core_residuals) if core_residuals else None,
        "raw_pair_distance_mean": sum(raw_distances) / len(raw_distances) if raw_distances else None,
        "assoc_pair_distance_mean": sum(assoc_distances) / len(assoc_distances) if assoc_distances else None,
        "core_assoc_pair_distance_mean": sum(core_assoc_distances) / len(core_assoc_distances) if core_assoc_distances else None,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    for row in rows:
        row["raw_assoc"] = trajectory_assoc_signal(row["features"])
        row["raw_core_assoc"] = core_block_assoc_signal(row["features"])
        row["raw_assoc_norm"] = math.sqrt(sum(value * value for value in row["raw_assoc"]))
        row["raw_core_assoc_norm"] = math.sqrt(sum(value * value for value in row["raw_core_assoc"]))
        row["raw_assoc_dominant_dim"] = max(range(8), key=lambda i: abs(row["raw_assoc"][i]))
        row["raw_core_assoc_dominant_dim"] = max(range(8), key=lambda i: abs(row["raw_core_assoc"][i]))
        row["raw_assoc_signed_dominant"] = row["sign"] * row["raw_assoc"][row["raw_assoc_dominant_dim"]]
        row["raw_core_assoc_signed_dominant"] = row["sign"] * row["raw_core_assoc"][row["raw_core_assoc_dominant_dim"]]

    raw_flat = leave_site_nearest_centroid(rows, "features")
    raw_assoc = leave_site_nearest_centroid(rows, "raw_assoc")
    raw_core_assoc = leave_site_nearest_centroid(rows, "raw_core_assoc")
    norm_assoc = leave_site_normalized_assoc(rows)
    norm_core_assoc = leave_site_normalized_core_assoc(rows)
    pairs = pair_audit(rows)

    signed = [row["raw_assoc_signed_dominant"] for row in rows]
    core_signed = [row["raw_core_assoc_signed_dominant"] for row in rows]
    site_counts = defaultdict(lambda: {"n": 0, "label_0": 0, "label_1": 0})
    for row in rows:
        site_counts[row["site"]]["n"] += 1
        site_counts[row["site"]][f"label_{row['label']}"] += 1

    summary = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic manifest/input audit only. This is not model, clinical, biomarker, or mechanistic evidence.",
        "manifest": str(args.manifest),
        "row_count": len(rows),
        "site_count": len(site_counts),
        "pair_audit": pairs,
        "site_counts": dict(sorted(site_counts.items())),
        "raw_assoc_signed_dominant_mean": sum(signed) / len(signed) if signed else 0.0,
        "raw_assoc_signed_dominant_min": min(signed) if signed else 0.0,
        "raw_core_assoc_signed_dominant_mean": sum(core_signed) / len(core_signed) if core_signed else 0.0,
        "raw_core_assoc_signed_dominant_min": min(core_signed) if core_signed else 0.0,
        "leave_site_raw_flat_nearest_centroid": raw_flat,
        "leave_site_raw_assoc_nearest_centroid": raw_assoc,
        "leave_site_raw_core_assoc_nearest_centroid": raw_core_assoc,
        "leave_site_train_zscored_assoc_nearest_centroid": norm_assoc,
        "leave_site_train_zscored_core_assoc_nearest_centroid": norm_core_assoc,
    }

    detail = [
        {
            "subject_id": row["subject_id"],
            "pair_id": row["pair_id"],
            "site": row["site"],
            "label": row["label"],
            "raw_assoc_norm": row["raw_assoc_norm"],
            "raw_core_assoc_norm": row["raw_core_assoc_norm"],
            "raw_assoc_dominant_dim": row["raw_assoc_dominant_dim"],
            "raw_core_assoc_dominant_dim": row["raw_core_assoc_dominant_dim"],
            "raw_assoc_signed_dominant": row["raw_assoc_signed_dominant"],
            "raw_core_assoc_signed_dominant": row["raw_core_assoc_signed_dominant"],
            **{f"raw_assoc_{i}": row["raw_assoc"][i] for i in range(8)},
            **{f"raw_core_assoc_{i}": row["raw_core_assoc"][i] for i in range(8)},
        }
        for row in rows
    ]
    write_tsv(out / "associator_data_audit_detail.tsv", detail)
    write_tsv(
        out / "associator_data_audit_summary.tsv",
        [
            {
                "row_count": summary["row_count"],
                "site_count": summary["site_count"],
                "pair_count": pairs["pair_count"],
                "bad_pair_count": pairs["bad_pair_count"],
                "assoc_sign_flip_residual_max": pairs["assoc_sign_flip_residual_max"],
                "core_assoc_sign_flip_residual_max": pairs["core_assoc_sign_flip_residual_max"],
                "raw_flat_leave_site_ba": raw_flat["balanced_accuracy"],
                "raw_assoc_leave_site_ba": raw_assoc["balanced_accuracy"],
                "raw_core_assoc_leave_site_ba": raw_core_assoc["balanced_accuracy"],
                "train_zscored_assoc_leave_site_ba": norm_assoc["balanced_accuracy"],
                "train_zscored_core_assoc_leave_site_ba": norm_core_assoc["balanced_accuracy"],
                "raw_assoc_signed_dominant_mean": summary["raw_assoc_signed_dominant_mean"],
                "raw_core_assoc_signed_dominant_mean": summary["raw_core_assoc_signed_dominant_mean"],
                "train_zscored_assoc_signed_dominant_mean": norm_assoc["signed_dominant_mean"],
                "train_zscored_core_assoc_signed_dominant_mean": norm_core_assoc["signed_dominant_mean"],
            }
        ],
    )
    md = [
        "# Octonionic Associator Data Audit",
        "",
        summary["claim_boundary"],
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Sites: `{summary['site_count']}`",
        f"- Pairs: `{pairs['pair_count']}`",
        f"- Bad pairs: `{pairs['bad_pair_count']}`",
        f"- Raw associator sign-flip residual max: `{pairs['assoc_sign_flip_residual_max']}`",
        f"- Raw core-block associator sign-flip residual max: `{pairs['core_assoc_sign_flip_residual_max']}`",
        f"- Raw flat leave-site nearest-centroid BA: `{raw_flat['balanced_accuracy']:.6f}`",
        f"- Raw associator leave-site nearest-centroid BA: `{raw_assoc['balanced_accuracy']:.6f}`",
        f"- Raw core-block associator leave-site nearest-centroid BA: `{raw_core_assoc['balanced_accuracy']:.6f}`",
        f"- Train-zscored associator leave-site nearest-centroid BA: `{norm_assoc['balanced_accuracy']:.6f}`",
        f"- Train-zscored core-block associator leave-site nearest-centroid BA: `{norm_core_assoc['balanced_accuracy']:.6f}`",
        f"- Raw signed dominant associator mean: `{summary['raw_assoc_signed_dominant_mean']:.6f}`",
        f"- Raw signed dominant core-block associator mean: `{summary['raw_core_assoc_signed_dominant_mean']:.6f}`",
        f"- Train-zscored signed dominant associator mean: `{norm_assoc['signed_dominant_mean']:.6f}`",
        f"- Train-zscored signed dominant core-block associator mean: `{norm_core_assoc['signed_dominant_mean']:.6f}`",
        "",
        "## Interpretation",
        "",
        "If raw associator separability is high but train-zscored associator separability collapses, the manifest is not necessarily wrong, but the current preprocessing/model input is erasing or distorting the constructed nonassociative target. If both raw and normalized associator separability are high while O-SSM remains near chance, the problem is more likely in the training/readout path.",
    ]
    (out / "associator_data_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "associator_data_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sha_rows = []
    for path in sorted(out.iterdir()):
        if path.name == "SHA256SUMS":
            continue
        sha_rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sha_rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
