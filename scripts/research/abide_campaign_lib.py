#!/usr/bin/env python3
"""Shared helpers for the Brain O-SSM ABIDE campaign.

This module is intentionally stdlib-first so it can be used from:
- local smoke scripts
- Slurm jobs without a heavy Python environment
- post-processing steps that should not depend on pandas/numpy

The external deep baselines layer imports this module and adds a torch
dependency only for the actual model training.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_SEEDS: list[int] = [
    55555,
    11111,
    99887,
    22222,
    44444,
    66666,
    77777,
    88888,
    33333,
    12321,
    24680,
    13579,
    54321,
    67890,
    11223,
    44567,
    77889,
    99001,
    31415,
    27182,
]

_EXPLICIT_FEATURE_RE = re.compile(r"^t(?P<time>\d+)_f(?P<feature>\d+)$")
_FLAT_FEATURE_RE = re.compile(r"^f(?P<index>\d+)$")


@dataclass
class ManifestMeta:
    path: str
    schema: str
    seq_len: int
    input_dim: int
    feature_layout: str
    subject_count: int
    site_count: int
    label_space: str
    extra: dict[str, str]


@dataclass
class SubjectRecord:
    subject_id: str
    label: int
    site: str
    sequence: list[list[float]]


def seed_schedule(count: int) -> list[int]:
    if count <= len(BASE_SEEDS):
        return BASE_SEEDS[:count]
    seeds = list(BASE_SEEDS)
    x = BASE_SEEDS[-1]
    while len(seeds) < count:
        x = (1103515245 * x + 12345) % 2_147_483_647
        seeds.append(x)
    return seeds


def _label_from_raw(raw: str) -> int:
    value = raw.strip().lower()
    if value in {"1", "asd", "aut"}:
        return 1
    if value in {"0", "control", "td", "hc"}:
        return 0
    raise ValueError(f"unsupported label value: {raw!r}")


def _read_manifest_lines(path: Path) -> tuple[dict[str, str], list[str]]:
    metadata: dict[str, str] = {}
    content_lines: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                payload = stripped[1:].strip()
                if "=" in payload:
                    key, value = payload.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            content_lines.append(line)
    if not content_lines:
        raise ValueError(f"manifest has no header/data rows: {path}")
    return metadata, content_lines


def load_manifest(path: str | Path) -> tuple[ManifestMeta, list[SubjectRecord]]:
    manifest_path = Path(path)
    metadata, lines = _read_manifest_lines(manifest_path)

    reader = csv.DictReader(lines, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"manifest has no tabular header: {manifest_path}")

    fieldnames = list(reader.fieldnames)
    for required in ("subject_id", "label", "site"):
        if required not in fieldnames:
            raise ValueError(f"manifest missing required column {required!r}: {manifest_path}")

    explicit_columns: list[tuple[int, int, str]] = []
    flat_columns: list[tuple[int, str]] = []
    for name in fieldnames:
        match = _EXPLICIT_FEATURE_RE.match(name)
        if match:
            explicit_columns.append((int(match.group("time")), int(match.group("feature")), name))
            continue
        match = _FLAT_FEATURE_RE.match(name)
        if match:
            flat_columns.append((int(match.group("index")), name))

    if explicit_columns:
        max_time = max(item[0] for item in explicit_columns)
        max_feature = max(item[1] for item in explicit_columns)
        seq_len = int(metadata.get("seq_len", max_time + 1))
        input_dim = int(metadata.get("input_dim", max_feature + 1))
        feature_layout = metadata.get("feature_layout", "explicit_seq")
        ordered_columns = sorted(explicit_columns)
        if len(ordered_columns) != seq_len * input_dim:
            raise ValueError(
                f"explicit feature grid mismatch: seq_len={seq_len} input_dim={input_dim} "
                f"but found {len(ordered_columns)} columns"
            )
        column_order = [name for _, _, name in ordered_columns]
    elif flat_columns:
        ordered_flat = [name for _, name in sorted(flat_columns)]
        feature_count = len(ordered_flat)
        if feature_count <= 0:
            raise ValueError(f"manifest has no feature columns: {manifest_path}")
        if "seq_len" in metadata:
            seq_len = int(metadata["seq_len"])
        elif feature_count % 8 == 0:
            seq_len = 8
        else:
            seq_len = 1
        if feature_count % seq_len != 0:
            raise ValueError(
                f"flat feature count {feature_count} is not divisible by seq_len={seq_len}"
            )
        input_dim = int(metadata.get("input_dim", feature_count // seq_len))
        if seq_len * input_dim != feature_count:
            raise ValueError(
                f"manifest seq_len/input_dim mismatch: seq_len={seq_len} input_dim={input_dim} "
                f"feature_count={feature_count}"
            )
        feature_layout = metadata.get("feature_layout", "flat")
        column_order = ordered_flat
    else:
        raise ValueError(f"manifest has no recognized feature columns: {manifest_path}")

    records: list[SubjectRecord] = []
    for row in reader:
        sequence = [[0.0 for _ in range(input_dim)] for _ in range(seq_len)]
        if feature_layout == "explicit_seq":
            for time_idx, feature_idx, column_name in explicit_columns:
                sequence[time_idx][feature_idx] = float(row[column_name])
        else:
            flat_values = [float(row[column_name]) for column_name in column_order]
            for index, value in enumerate(flat_values):
                time_idx = index // input_dim
                feature_idx = index % input_dim
                sequence[time_idx][feature_idx] = value

        records.append(
            SubjectRecord(
                subject_id=row["subject_id"],
                label=_label_from_raw(row["label"]),
                site=row["site"].strip(),
                sequence=sequence,
            )
        )

    sites = sorted({record.site for record in records})
    meta = ManifestMeta(
        path=str(manifest_path),
        schema=metadata.get("schema", "brain_ossm.abide.v1"),
        seq_len=seq_len,
        input_dim=input_dim,
        feature_layout=feature_layout,
        subject_count=len(records),
        site_count=len(sites),
        label_space=metadata.get("label_space", "asd_vs_control"),
        extra=dict(metadata),
    )
    return meta, records


def grouped_leave_one_site_out(records: list[SubjectRecord]) -> list[str]:
    return sorted({record.site for record in records})


def select_sites(records: list[SubjectRecord], max_sites: int | None) -> list[SubjectRecord]:
    if not max_sites or max_sites <= 0:
        return list(records)
    keep_sites = set(grouped_leave_one_site_out(records)[:max_sites])
    return [record for record in records if record.site in keep_sites]


def limit_subjects(records: list[SubjectRecord], limit_subjects: int | None) -> list[SubjectRecord]:
    if not limit_subjects or limit_subjects <= 0 or len(records) <= limit_subjects:
        return list(records)
    grouped: dict[str, list[SubjectRecord]] = {}
    for record in records:
        grouped.setdefault(record.site, []).append(record)
    kept: list[SubjectRecord] = []
    per_site = max(1, limit_subjects // max(1, len(grouped)))
    for site in sorted(grouped):
        site_records = grouped[site]
        kept.extend(site_records[:per_site])
    return kept[:limit_subjects]


def write_json(path: str | Path, payload: object) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_tsv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _binary_confusion(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for label, prob in zip(labels, probs, strict=True):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0.0:
        return default
    return num / den


def binary_metrics(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict[str, float]:
    confusion = _binary_confusion(labels, probs, threshold=threshold)
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    n = len(labels)

    accuracy = 100.0 * _safe_div(tp + tn, n, default=0.0)
    tpr = _safe_div(tp, tp + fn, default=0.0)
    tnr = _safe_div(tn, tn + fp, default=0.0)
    balanced_accuracy = 50.0 * (tpr + tnr)

    precision_pos = _safe_div(tp, tp + fp, default=0.0)
    recall_pos = tpr
    f1_pos = _safe_div(2.0 * precision_pos * recall_pos, precision_pos + recall_pos, default=0.0)

    precision_neg = _safe_div(tn, tn + fn, default=0.0)
    recall_neg = tnr
    f1_neg = _safe_div(2.0 * precision_neg * recall_neg, precision_neg + recall_neg, default=0.0)
    macro_f1 = 50.0 * (f1_pos + f1_neg)

    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        auroc = 50.0
    else:
        ranked = sorted(zip(probs, labels), key=lambda item: item[0])
        rank_sum = 0.0
        for rank, (_, label) in enumerate(ranked, start=1):
            if label == 1:
                rank_sum += rank
        auroc = 100.0 * ((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))

    brier = sum((prob - float(label)) ** 2 for label, prob in zip(labels, probs, strict=True)) / max(1, n)

    ece = 0.0
    bin_count = 10
    for bin_idx in range(bin_count):
        lower = bin_idx / bin_count
        upper = (bin_idx + 1) / bin_count
        bucket: list[tuple[int, float]] = []
        for label, prob in zip(labels, probs, strict=True):
            if bin_idx == bin_count - 1:
                hit = lower <= prob <= upper
            else:
                hit = lower <= prob < upper
            if hit:
                bucket.append((label, prob))
        if not bucket:
            continue
        mean_conf = sum(prob for _, prob in bucket) / len(bucket)
        mean_acc = sum(1.0 if ((prob >= threshold) == (label == 1)) else 0.0 for label, prob in bucket) / len(bucket)
        ece += abs(mean_acc - mean_conf) * (len(bucket) / n) * 100.0

    return {
        "accuracy_pct": accuracy,
        "balanced_accuracy_pct": balanced_accuracy,
        "macro_f1_pct": macro_f1,
        "auroc_pct": auroc,
        "brier": brier,
        "ece_pct": ece,
        "subjects": float(n),
    }


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    values_list = list(values)
    if not values_list:
        return 0.0, 0.0
    if len(values_list) == 1:
        return values_list[0], 0.0
    return statistics.mean(values_list), statistics.pstdev(values_list)


def summarize_prediction_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    per_seed: list[dict[str, object]] = []
    per_site_seed: list[dict[str, object]] = []

    model_seed_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    model_seed_site_groups: dict[tuple[str, int, str], list[dict[str, object]]] = {}

    for row in rows:
        model = str(row["model"])
        seed = int(row["seed"])
        site = str(row["site"])
        model_seed_groups.setdefault((model, seed), []).append(row)
        model_seed_site_groups.setdefault((model, seed, site), []).append(row)

    for (model, seed), group in sorted(model_seed_groups.items()):
        labels = [int(item["label"]) for item in group]
        probs = [float(item["prob"]) for item in group]
        metrics = binary_metrics(labels, probs)
        assoc_values = [float(item["assoc"]) for item in group if item.get("assoc") is not None]
        assoc_mean = sum(assoc_values) / len(assoc_values) if assoc_values else None
        per_seed.append(
            {
                "model": model,
                "seed": seed,
                "site_count": len({str(item["site"]) for item in group}),
                "subjects": len(group),
                "accuracy_pct": round(metrics["accuracy_pct"], 6),
                "balanced_accuracy_pct": round(metrics["balanced_accuracy_pct"], 6),
                "macro_f1_pct": round(metrics["macro_f1_pct"], 6),
                "auroc_pct": round(metrics["auroc_pct"], 6),
                "brier": round(metrics["brier"], 8),
                "ece_pct": round(metrics["ece_pct"], 6),
                "assoc_mean": None if assoc_mean is None else round(assoc_mean, 6),
            }
        )

    for (model, seed, site), group in sorted(model_seed_site_groups.items()):
        labels = [int(item["label"]) for item in group]
        probs = [float(item["prob"]) for item in group]
        metrics = binary_metrics(labels, probs)
        assoc_values = [float(item["assoc"]) for item in group if item.get("assoc") is not None]
        assoc_mean = sum(assoc_values) / len(assoc_values) if assoc_values else None
        per_site_seed.append(
            {
                "model": model,
                "seed": seed,
                "site": site,
                "subjects": len(group),
                "accuracy_pct": round(metrics["accuracy_pct"], 6),
                "balanced_accuracy_pct": round(metrics["balanced_accuracy_pct"], 6),
                "macro_f1_pct": round(metrics["macro_f1_pct"], 6),
                "auroc_pct": round(metrics["auroc_pct"], 6),
                "brier": round(metrics["brier"], 8),
                "ece_pct": round(metrics["ece_pct"], 6),
                "assoc_mean": None if assoc_mean is None else round(assoc_mean, 6),
            }
        )

    overall: list[dict[str, object]] = []
    per_site: list[dict[str, object]] = []

    overall_groups: dict[str, list[dict[str, object]]] = {}
    per_site_groups: dict[tuple[str, str], list[dict[str, object]]] = {}

    for row in per_seed:
        overall_groups.setdefault(str(row["model"]), []).append(row)
    for row in per_site_seed:
        per_site_groups.setdefault((str(row["model"]), str(row["site"])), []).append(row)

    metric_fields = [
        "accuracy_pct",
        "balanced_accuracy_pct",
        "macro_f1_pct",
        "auroc_pct",
        "brier",
        "ece_pct",
    ]

    for model, group in sorted(overall_groups.items()):
        row: dict[str, object] = {
            "model": model,
            "seed_count": len(group),
            "site_count": group[0]["site_count"] if group else 0,
            "subjects": group[0]["subjects"] if group else 0,
        }
        for field in metric_fields:
            mean_value, std_value = _mean_std(float(item[field]) for item in group)
            row[f"{field}_mean"] = round(mean_value, 6)
            row[f"{field}_std"] = round(std_value, 6)
        assoc_values = [float(item["assoc_mean"]) for item in group if item.get("assoc_mean") is not None]
        assoc_mean, assoc_std = _mean_std(assoc_values)
        row["assoc_mean_mean"] = round(assoc_mean, 6) if assoc_values else None
        row["assoc_mean_std"] = round(assoc_std, 6) if assoc_values else None
        overall.append(row)

    for (model, site), group in sorted(per_site_groups.items()):
        row = {
            "model": model,
            "site": site,
            "seed_count": len(group),
        }
        subject_mean, _ = _mean_std(float(item["subjects"]) for item in group)
        row["subjects_mean"] = round(subject_mean, 3)
        for field in metric_fields:
            mean_value, std_value = _mean_std(float(item[field]) for item in group)
            row[f"{field}_mean"] = round(mean_value, 6)
            row[f"{field}_std"] = round(std_value, 6)
        assoc_values = [float(item["assoc_mean"]) for item in group if item.get("assoc_mean") is not None]
        assoc_mean, assoc_std = _mean_std(assoc_values)
        row["assoc_mean_mean"] = round(assoc_mean, 6) if assoc_values else None
        row["assoc_mean_std"] = round(assoc_std, 6) if assoc_values else None
        per_site.append(row)

    return {
        "overall": overall,
        "per_seed": per_seed,
        "per_site": per_site,
        "per_site_seed": per_site_seed,
    }
