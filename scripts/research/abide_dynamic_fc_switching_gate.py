#!/usr/bin/env python3
"""Evaluate temporal switch-event baselines on ABIDE dynamic-FC targets.

This consumes ``dynamic_fc_window_table.tsv`` from
``abide_dynamic_fc_switching_target.py`` and predicts whether the current
dynamic-FC state differs from the previous window's state. Features are built
only from history available before the target window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "neurodyn.abide_dynamic_fc_switching_gate.v1"
CLAIM_BOUNDARY = (
    "Switch-event temporal prediction gate only; no diagnostic, biomarker, "
    "mechanism, ASD-detection, clinical-decision, or O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--models",
        default="base_rate,persistence,logistic,gru_reservoir,hssm_reservoir,ossm_reservoir,trained_hssm,trained_ossm",
        help=(
            "Comma-separated models: base_rate,persistence,logistic,"
            "gru_reservoir,hssm_reservoir,ossm_reservoir,trained_hssm,trained_ossm."
        ),
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--reservoir-hidden", type=int, default=24)
    parser.add_argument("--reservoir-seed", type=int, default=20260708)
    parser.add_argument("--trained-candidates", type=int, default=5)
    parser.add_argument("--null-permutations", type=int, default=0)
    parser.add_argument("--min-test-events", type=int, default=20)
    parser.add_argument("--min-test-positive-events", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def oct_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normed octonion multiplication with Fano line (2,5,7) positive."""
    r = np.zeros(8, dtype=np.float64)
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3] - a[4] * b[4] - a[5] * b[5] - a[6] * b[6] - a[7] * b[7]
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2] + a[4] * b[5] - a[5] * b[4] - a[6] * b[7] + a[7] * b[6]
    r[2] = a[0] * b[2] + a[2] * b[0] - a[1] * b[3] + a[3] * b[1] + a[4] * b[6] - a[6] * b[4] + a[5] * b[7] - a[7] * b[5]
    r[3] = a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1] + a[4] * b[7] - a[7] * b[4] - a[5] * b[6] + a[6] * b[5]
    r[4] = a[0] * b[4] + a[4] * b[0] - a[1] * b[5] + a[5] * b[1] - a[2] * b[6] + a[6] * b[2] - a[3] * b[7] + a[7] * b[3]
    r[5] = a[0] * b[5] + a[5] * b[0] + a[1] * b[4] - a[4] * b[1] - a[2] * b[7] + a[7] * b[2] + a[3] * b[6] - a[6] * b[3]
    r[6] = a[0] * b[6] + a[6] * b[0] + a[1] * b[7] - a[7] * b[1] + a[2] * b[4] - a[4] * b[2] - a[3] * b[5] + a[5] * b[3]
    r[7] = a[0] * b[7] + a[7] * b[0] - a[1] * b[6] + a[6] * b[1] + a[2] * b[5] - a[5] * b[2] + a[3] * b[4] - a[4] * b[3]
    return r


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(4, dtype=np.float64)
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return r


def octonion_self_check() -> dict[str, float | int]:
    e2 = np.zeros(8, dtype=np.float64)
    e5 = np.zeros(8, dtype=np.float64)
    e2[2] = 1.0
    e5[5] = 1.0
    prod = oct_mul(e2, e5)
    rng = np.random.default_rng(424242)
    max_composition_err = 0.0
    max_alternative_err = 0.0
    for _ in range(32):
        a = rng.normal(0.0, 1.0, size=8)
        b = rng.normal(0.0, 1.0, size=8)
        ab = oct_mul(a, b)
        max_composition_err = max(
            max_composition_err,
            abs(float(ab @ ab) - float(a @ a) * float(b @ b)),
        )
        alt = oct_mul(oct_mul(a, a), b) - oct_mul(a, oct_mul(a, b))
        max_alternative_err = max(max_alternative_err, float(np.linalg.norm(alt)))
    return {
        "e2_times_e5_dominant_dim": int(np.argmax(np.abs(prod))),
        "e2_times_e5_dim7_value": float(prod[7]),
        "composition_err_max": max_composition_err,
        "alternative_err_max": max_alternative_err,
    }


def sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def event_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row.get("fold_id") or not row.get("subject_id") or not row.get("k"):
            continue
        grouped[(row["fold_id"], int(row["k"]), row["subject_id"])].append(row)

    out: list[dict[str, Any]] = []
    for (fold_id, k, subject_id), chunk in sorted(grouped.items()):
        ordered = sorted(chunk, key=lambda row: int(row["window_index"]))
        previous_state: int | None = None
        previous_switch = 0
        run_length = 0
        for row in ordered:
            state = int(row["state"])
            window_index = int(row["window_index"])
            window_count = max(1, int(row["window_count"]))
            if previous_state is not None and str(row.get("switch_event", "")) in {"0", "1"}:
                features = [0.0] * (k + 4)
                if 0 <= previous_state < k:
                    features[previous_state] = 1.0
                features[k] = float(previous_switch)
                features[k + 1] = min(run_length, 20) / 20.0
                features[k + 2] = window_index / max(1, window_count - 1)
                features[k + 3] = math.log1p(window_count)
                out.append(
                    {
                        "fold_id": fold_id,
                        "holdout_site": row.get("holdout_site", ""),
                        "k": k,
                        "subject_id": subject_id,
                        "site": row.get("site", ""),
                        "label": row.get("label", ""),
                        "split": row.get("split", ""),
                        "window_index": window_index,
                        "window_count": window_count,
                        "previous_state": previous_state,
                        "previous_switch": previous_switch,
                        "run_length_before_event": run_length,
                        "target": int(row["switch_event"]),
                        "features": features,
                    }
                )
            if previous_state is None or state != previous_state:
                run_length = 1
            else:
                run_length += 1
            previous_switch = 0 if previous_state is None else int(state != previous_state)
            previous_state = state
    return out


def split_fold(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    return train, test


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma = np.where(sigma > 1.0e-8, sigma, 1.0)
    return (x - mu) / sigma, mu, sigma


def train_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x, mu, sigma = standardize(train_x, train_x)
    y = train_y.astype(np.float64)
    w = np.zeros(x.shape[1], dtype=np.float64)
    pos = float(np.clip(y.mean(), 1.0e-6, 1.0 - 1.0e-6))
    b = math.log(pos / (1.0 - pos))
    for _ in range(max(1, epochs)):
        p = sigmoid_array(x @ w + b)
        err = p - y
        grad_w = (x.T @ err) / x.shape[0] + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, mu, sigma


def predict_logistic(w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray, x: np.ndarray) -> np.ndarray:
    zx = (x - mu) / sigma
    return sigmoid_array(zx @ w + b)


def reservoir_features(rows: list[dict[str, Any]], *, hidden_dim: int, seed: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, hidden_dim), dtype=np.float64)
    dim = len(rows[0]["features"])
    rng = np.random.default_rng(seed)
    wxz = rng.normal(0.0, 0.35 / math.sqrt(dim), size=(dim, hidden_dim))
    whz = rng.normal(0.0, 0.35 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
    wxr = rng.normal(0.0, 0.35 / math.sqrt(dim), size=(dim, hidden_dim))
    whr = rng.normal(0.0, 0.35 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
    wxh = rng.normal(0.0, 0.35 / math.sqrt(dim), size=(dim, hidden_dim))
    whh = rng.normal(0.0, 0.35 / math.sqrt(hidden_dim), size=(hidden_dim, hidden_dim))
    bz = rng.normal(0.0, 0.02, size=hidden_dim)
    br = rng.normal(0.0, 0.02, size=hidden_dim)
    bh = rng.normal(0.0, 0.02, size=hidden_dim)

    ordered_indices = sorted(
        range(len(rows)),
        key=lambda idx: (rows[idx]["fold_id"], rows[idx]["subject_id"], int(rows[idx]["window_index"])),
    )
    h_by_subject: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(hidden_dim, dtype=np.float64))
    out = np.zeros((len(rows), hidden_dim), dtype=np.float64)
    for idx in ordered_indices:
        row = rows[idx]
        key = (row["fold_id"], row["subject_id"])
        h = h_by_subject[key]
        x = np.asarray(row["features"], dtype=np.float64)
        z = sigmoid_array(x @ wxz + h @ whz + bz)
        r = sigmoid_array(x @ wxr + h @ whr + br)
        h_tilde = np.tanh(x @ wxh + (r * h) @ whh + bh)
        h_new = (1.0 - z) * h + z * h_tilde
        h_by_subject[key] = h_new
        out[idx] = h_new
    return out


def to_oct_inputs(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    train_x = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
    all_x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    zx, _, _ = standardize(train_x, all_x)
    zx = np.clip(zx, -5.0, 5.0)
    if zx.shape[1] < 8:
        padded = np.zeros((zx.shape[0], 8), dtype=np.float64)
        padded[:, : zx.shape[1]] = zx
        zx = padded
    elif zx.shape[1] > 8:
        zx = zx[:, :8]
    return {id(row): zx[idx] for idx, row in enumerate(rows)}


def algebraic_reservoir_features(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    algebra: str,
    seed: int,
    a_override: np.ndarray | None = None,
) -> np.ndarray:
    if not rows:
        return np.zeros((0, 9), dtype=np.float64)
    if a_override is None:
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, size=8)
    else:
        a = np.asarray(a_override, dtype=np.float64).copy()
    a = normalize_algebra_parameter(a, algebra)

    x_by_id = to_oct_inputs(rows, train_rows)
    ordered_indices = sorted(
        range(len(rows)),
        key=lambda idx: (rows[idx]["fold_id"], rows[idx]["subject_id"], int(rows[idx]["window_index"])),
    )
    h0 = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    h_by_subject: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: h0.copy())
    assoc_by_subject: dict[tuple[str, str], float] = defaultdict(float)
    count_by_subject: dict[tuple[str, str], int] = defaultdict(int)
    out = np.zeros((len(rows), 9), dtype=np.float64)
    for idx in ordered_indices:
        row = rows[idx]
        key = (row["fold_id"], row["subject_id"])
        h = h_by_subject[key]
        x = x_by_id[id(row)]
        if algebra == "ossm":
            ah = oct_mul(a, h)
            left = oct_mul(ah, x)
            right = oct_mul(a, oct_mul(h, x))
            diff = left - right
            assoc_step = min(math.sqrt(max(float(diff @ diff), 0.0)), 1000.0)
            h_new = np.clip(left + 0.2 * right, -5.0, 5.0)
            h_new = np.where(np.isfinite(h_new), h_new, 0.0)
        elif algebra == "hssm":
            left0 = quat_mul(quat_mul(a[:4], h[:4]), x[:4])
            left1 = quat_mul(quat_mul(a[4:], h[4:]), x[4:])
            h_new = np.zeros(8, dtype=np.float64)
            h_new[:4] = np.clip(1.2 * left0, -5.0, 5.0)
            h_new[4:] = np.clip(1.2 * left1, -5.0, 5.0)
            assoc_step = 0.0
        else:
            raise ValueError(f"unknown algebra: {algebra}")
        h_by_subject[key] = h_new
        assoc_by_subject[key] += assoc_step
        count_by_subject[key] += 1
        out[idx, :8] = h_new
        out[idx, 8] = assoc_by_subject[key] / max(1, count_by_subject[key]) if algebra == "ossm" else 0.0
    return out


def normalize_algebra_parameter(a: np.ndarray, algebra: str) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    norm = float(np.linalg.norm(out))
    if norm <= 1.0e-12:
        out = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        out = out / norm
    if algebra == "hssm":
        n0 = float(np.linalg.norm(out[:4]))
        n1 = float(np.linalg.norm(out[4:]))
        out[:4] = out[:4] / n0 if n0 > 1.0e-12 else np.asarray([1.0, 0.0, 0.0, 0.0])
        out[4:] = out[4:] / n1 if n1 > 1.0e-12 else np.asarray([1.0, 0.0, 0.0, 0.0])
    return out


def trained_algebraic_features(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    train_y: np.ndarray,
    *,
    algebra: str,
    seed: int,
    candidates: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_mask = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_states: np.ndarray | None = None
    best_a: np.ndarray | None = None
    tried = max(1, candidates)
    for idx in range(tried):
        if idx == 0:
            raw_a = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            raw_a = rng.normal(0.0, 1.0, size=8)
        a = normalize_algebra_parameter(raw_a, algebra)
        states = algebraic_reservoir_features(rows, train_rows, algebra=algebra, seed=seed + idx, a_override=a)
        w, b, mu, sigma = train_logistic(states[train_mask], train_y, epochs=args.epochs, lr=args.lr, l2=args.l2)
        train_probs = predict_logistic(w, b, mu, sigma, states[train_mask]).tolist()
        train_labels = [int(value) for value in train_y.tolist()]
        loss = log_loss(train_labels, train_probs)
        if loss < best_loss:
            best_loss = loss
            best_states = states
            best_a = a
    if best_states is None or best_a is None:
        raise RuntimeError("trained algebraic surface failed to select a candidate")
    return best_states, {
        "trained_candidate_count": tried,
        "selected_train_log_loss": best_loss,
        "selected_a_norm": float(np.linalg.norm(best_a)),
    }


def brier(labels: list[int], probs: list[float]) -> float:
    return sum((p - y) * (p - y) for y, p in zip(labels, probs, strict=True)) / len(labels)


def log_loss(labels: list[int], probs: list[float]) -> float:
    total = 0.0
    for y, p in zip(labels, probs, strict=True):
        q = min(1.0 - 1.0e-12, max(1.0e-12, p))
        total += -(y * math.log(q) + (1 - y) * math.log(1.0 - q))
    return total / len(labels)


def auroc(labels: list[int], probs: list[float]) -> float | None:
    pos = [p for y, p in zip(labels, probs, strict=True) if y == 1]
    neg = [p for y, p in zip(labels, probs, strict=True) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
            total += 1
    return wins / total


def average_precision(labels: list[int], probs: list[float]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    ordered = sorted(zip(probs, labels, strict=True), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positive_count
        precision = tp / max(1, tp + fp)
        if label == 1:
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return ap


def metric_row(model: str, fold_id: str, k: int, rows: list[dict[str, Any]], probs: list[float]) -> dict[str, Any]:
    labels = [int(row["target"]) for row in rows]
    positives = sum(labels)
    auc = auroc(labels, probs)
    ap = average_precision(labels, probs)
    return {
        "model": model,
        "fold_id": fold_id,
        "k": k,
        "test_events": len(labels),
        "test_positive_events": positives,
        "test_prevalence": positives / len(labels) if labels else 0.0,
        "brier": brier(labels, probs),
        "log_loss": log_loss(labels, probs),
        "auroc": "" if auc is None else auc,
        "average_precision": "" if ap is None else ap,
        "mean_predicted_prob": sum(probs) / len(probs) if probs else 0.0,
        "null_permutations": 0,
        "null_average_precision_mean": "",
        "null_average_precision_p_ge": "",
        "null_brier_mean": "",
        "null_log_loss_mean": "",
        "trained_candidate_count": "",
        "selected_train_log_loss": "",
    }


def add_readout_nulls(
    row: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_rows: list[dict[str, Any]],
    observed_probs: list[float],
    args: argparse.Namespace,
    *,
    seed: int,
) -> None:
    if args.null_permutations <= 0:
        return
    labels = [int(test_row["target"]) for test_row in test_rows]
    observed_ap = average_precision(labels, observed_probs)
    rng = np.random.default_rng(seed)
    aps: list[float] = []
    briers: list[float] = []
    losses: list[float] = []
    for _ in range(args.null_permutations):
        perm_y = np.asarray(train_y, dtype=np.float64).copy()
        rng.shuffle(perm_y)
        w, b, mu, sigma = train_logistic(train_x, perm_y, epochs=args.epochs, lr=args.lr, l2=args.l2)
        probs = predict_logistic(w, b, mu, sigma, test_x).tolist()
        ap = average_precision(labels, probs)
        if ap is not None:
            aps.append(ap)
        briers.append(brier(labels, probs))
        losses.append(log_loss(labels, probs))
    row["null_permutations"] = args.null_permutations
    row["null_average_precision_mean"] = sum(aps) / len(aps) if aps else ""
    if observed_ap is not None and aps:
        row["null_average_precision_p_ge"] = (1 + sum(1 for value in aps if value >= observed_ap)) / (len(aps) + 1)
    row["null_brier_mean"] = sum(briers) / len(briers) if briers else ""
    row["null_log_loss_mean"] = sum(losses) / len(losses) if losses else ""


def mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) != ""]
    if not values:
        return None
    return sum(values) / len(values)


def run_gate(rows: list[dict[str, Any]], models: set[str], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    predictions: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    failures: list[str] = []
    by_fold_k: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_fold_k[(row["fold_id"], int(row["k"]))].append(row)

    for (fold_id, k), fold_rows in sorted(by_fold_k.items()):
        train_rows, test_rows = split_fold(fold_rows)
        if len(test_rows) < args.min_test_events:
            failures.append(f"{fold_id}:test_events_below_minimum")
        if sum(int(row["target"]) for row in test_rows) < args.min_test_positive_events:
            failures.append(f"{fold_id}:test_positive_events_below_minimum")
        if not train_rows or not test_rows:
            continue
        train_x = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
        train_y = np.asarray([row["target"] for row in train_rows], dtype=np.float64)
        test_x = np.asarray([row["features"] for row in test_rows], dtype=np.float64)
        train_rate = float(np.clip(train_y.mean(), 1.0e-6, 1.0 - 1.0e-6))

        fold_probs: dict[str, list[float]] = {}
        null_features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "base_rate" in models:
            fold_probs["base_rate"] = [train_rate] * len(test_rows)
        if "persistence" in models:
            fold_probs["persistence"] = [
                0.95 if int(row["previous_switch"]) == 1 else 0.05 for row in test_rows
            ]
        if "logistic" in models:
            w, b, mu, sigma = train_logistic(train_x, train_y, epochs=args.epochs, lr=args.lr, l2=args.l2)
            fold_probs["logistic"] = predict_logistic(w, b, mu, sigma, test_x).tolist()
            null_features["logistic"] = (train_x, test_x)
        if "gru_reservoir" in models:
            reservoir = reservoir_features(fold_rows, hidden_dim=args.reservoir_hidden, seed=args.reservoir_seed + k)
            train_mask = np.asarray([row["split"] == "train" for row in fold_rows], dtype=bool)
            test_mask = np.asarray([row["split"] == "test" for row in fold_rows], dtype=bool)
            y_all = np.asarray([row["target"] for row in fold_rows], dtype=np.float64)
            w, b, mu, sigma = train_logistic(
                reservoir[train_mask],
                y_all[train_mask],
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
            )
            fold_probs["gru_reservoir"] = predict_logistic(w, b, mu, sigma, reservoir[test_mask]).tolist()
            null_features["gru_reservoir"] = (reservoir[train_mask], reservoir[test_mask])
        training_meta: dict[str, dict[str, Any]] = {}
        for model, algebra in [("hssm_reservoir", "hssm"), ("ossm_reservoir", "ossm")]:
            if model not in models:
                continue
            states = algebraic_reservoir_features(
                fold_rows,
                train_rows,
                algebra=algebra,
                seed=args.reservoir_seed + k + (104729 if algebra == "ossm" else 524287),
            )
            train_mask = np.asarray([row["split"] == "train" for row in fold_rows], dtype=bool)
            test_mask = np.asarray([row["split"] == "test" for row in fold_rows], dtype=bool)
            y_all = np.asarray([row["target"] for row in fold_rows], dtype=np.float64)
            w, b, mu, sigma = train_logistic(
                states[train_mask],
                y_all[train_mask],
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
            )
            fold_probs[model] = predict_logistic(w, b, mu, sigma, states[test_mask]).tolist()
            null_features[model] = (states[train_mask], states[test_mask])
        for model, algebra in [("trained_hssm", "hssm"), ("trained_ossm", "ossm")]:
            if model not in models:
                continue
            train_mask = np.asarray([row["split"] == "train" for row in fold_rows], dtype=bool)
            test_mask = np.asarray([row["split"] == "test" for row in fold_rows], dtype=bool)
            y_all = np.asarray([row["target"] for row in fold_rows], dtype=np.float64)
            states, meta = trained_algebraic_features(
                fold_rows,
                train_rows,
                y_all[train_mask],
                algebra=algebra,
                seed=args.reservoir_seed + k + (130363 if algebra == "ossm" else 99991),
                candidates=args.trained_candidates,
                args=args,
            )
            w, b, mu, sigma = train_logistic(
                states[train_mask],
                y_all[train_mask],
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
            )
            fold_probs[model] = predict_logistic(w, b, mu, sigma, states[test_mask]).tolist()
            null_features[model] = (states[train_mask], states[test_mask])
            training_meta[model] = meta

        for model, probs in sorted(fold_probs.items()):
            detail_row = metric_row(model, fold_id, k, test_rows, probs)
            if model in training_meta:
                detail_row.update(training_meta[model])
            if model in null_features:
                null_train_x, null_test_x = null_features[model]
                add_readout_nulls(
                    detail_row,
                    null_train_x,
                    train_y,
                    null_test_x,
                    test_rows,
                    probs,
                    args,
                    seed=args.reservoir_seed + k + sum(ord(ch) for ch in model),
                )
            detail.append(detail_row)
            for row, prob in zip(test_rows, probs, strict=True):
                predictions.append(
                    {
                        "model": model,
                        "fold_id": fold_id,
                        "k": k,
                        "subject_id": row["subject_id"],
                        "site": row["site"],
                        "label": row["label"],
                        "window_index": row["window_index"],
                        "target": row["target"],
                        "predicted_prob": f"{prob:.10f}",
                        "previous_state": row["previous_state"],
                        "previous_switch": row["previous_switch"],
                        "run_length_before_event": row["run_length_before_event"],
                    }
                )
    summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in detail}):
        model_rows = [row for row in detail if row["model"] == model]
        summary.append(
            {
                "model": model,
                "fold_count": len(model_rows),
                "test_events": sum(int(row["test_events"]) for row in model_rows),
                "test_positive_events": sum(int(row["test_positive_events"]) for row in model_rows),
                "test_prevalence_mean": mean_numeric(model_rows, "test_prevalence"),
                "brier_mean": mean_numeric(model_rows, "brier"),
                "log_loss_mean": mean_numeric(model_rows, "log_loss"),
                "auroc_mean": mean_numeric(model_rows, "auroc"),
                "average_precision_mean": mean_numeric(model_rows, "average_precision"),
                "mean_predicted_prob": mean_numeric(model_rows, "mean_predicted_prob"),
                "null_permutations_mean": mean_numeric(model_rows, "null_permutations"),
                "null_average_precision_mean": mean_numeric(model_rows, "null_average_precision_mean"),
                "null_average_precision_p_ge_mean": mean_numeric(model_rows, "null_average_precision_p_ge"),
                "null_brier_mean": mean_numeric(model_rows, "null_brier_mean"),
                "null_log_loss_mean": mean_numeric(model_rows, "null_log_loss_mean"),
                "trained_candidate_count_mean": mean_numeric(model_rows, "trained_candidate_count"),
                "selected_train_log_loss_mean": mean_numeric(model_rows, "selected_train_log_loss"),
            }
        )
    return predictions, summary, failures


def decide(summary: list[dict[str, Any]], failures: list[str], models: set[str]) -> tuple[str, list[str]]:
    reasons = list(failures)
    if not summary:
        return "BLOCKED_NO_EVALUABLE_FOLDS", reasons or ["no summary rows"]
    if failures:
        return "UNDERCONTROLLED_LOW_EVENT_SUPPORT", reasons
    if "gru_reservoir" not in models:
        reasons.append("generic_recurrent_control_missing")
        return "UNDERCONTROLLED_NO_GENERIC_RECURRENT_CONTROL", reasons
    if "base_rate" not in models:
        reasons.append("base_rate_control_missing")
        return "UNDERCONTROLLED_BASELINE_MISSING", reasons
    if "ossm_reservoir" not in models:
        reasons.append("no_o_ssm_surface_in_this_gate")
        return "BASELINE_GATE_READY_NO_O_SSM_CLAIM", reasons
    if "hssm_reservoir" not in models:
        reasons.append("associative_hssm_control_missing")
        return "UNDERCONTROLLED_O_SSM_WITHOUT_ASSOCIATIVE_CONTROL", reasons
    if "trained_ossm" in models and "trained_hssm" in models:
        reasons.append("trained_python_o_ssm_surface_not_full_sounio_model")
        reasons.append("no_promotion_without_full_sounio_campaign_and_retrained_nulls")
        return "TRAINED_O_SSM_GATE_EXECUTED_NO_PROMOTION", reasons
    reasons.append("reservoir_o_ssm_surface_only_not_full_trained_sio_model")
    reasons.append("no_promotion_without_grouped_site_gate_and_retrained_nulls")
    return "O_SSM_RESERVOIR_GATE_EXECUTED_NO_PROMOTION", reasons


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {value.strip() for value in args.models.split(",") if value.strip()}
    allowed = {
        "base_rate",
        "persistence",
        "logistic",
        "gru_reservoir",
        "hssm_reservoir",
        "ossm_reservoir",
        "trained_hssm",
        "trained_ossm",
    }
    unknown = sorted(models - allowed)
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")

    source_rows = read_tsv(args.window_table)
    rows = event_rows(source_rows)
    predictions, summary, failures = run_gate(rows, models, args)
    verdict, reasons = decide(summary, failures, models)
    oct_check = octonion_self_check()
    if (
        oct_check["e2_times_e5_dominant_dim"] != 7
        or abs(float(oct_check["e2_times_e5_dim7_value"]) - 1.0) > 1.0e-12
        or float(oct_check["composition_err_max"]) > 1.0e-10
        or float(oct_check["alternative_err_max"]) > 1.0e-10
    ):
        raise SystemExit(f"octonion self-check failed: {oct_check}")

    pred_fields = [
        "model",
        "fold_id",
        "k",
        "subject_id",
        "site",
        "label",
        "window_index",
        "target",
        "predicted_prob",
        "previous_state",
        "previous_switch",
        "run_length_before_event",
    ]
    summary_fields = [
        "model",
        "fold_count",
        "test_events",
        "test_positive_events",
        "test_prevalence_mean",
        "brier_mean",
        "log_loss_mean",
        "auroc_mean",
        "average_precision_mean",
        "mean_predicted_prob",
        "null_permutations_mean",
        "null_average_precision_mean",
        "null_average_precision_p_ge_mean",
        "null_brier_mean",
        "null_log_loss_mean",
        "trained_candidate_count_mean",
        "selected_train_log_loss_mean",
    ]
    if predictions:
        write_tsv(args.output_dir / "dynamic_fc_switching_gate_predictions.tsv", predictions, pred_fields)
    else:
        write_tsv(args.output_dir / "dynamic_fc_switching_gate_predictions.tsv", [], pred_fields)
    write_tsv(args.output_dir / "dynamic_fc_switching_gate_summary.tsv", summary, summary_fields)
    counts = Counter((row["split"], row["target"]) for row in rows)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "window_table": str(args.window_table),
        "verdict": verdict,
        "reasons": reasons,
        "models": sorted(models),
        "octonion_self_check": oct_check,
        "parameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "reservoir_hidden": args.reservoir_hidden,
            "reservoir_seed": args.reservoir_seed,
            "trained_candidates": args.trained_candidates,
            "null_permutations": args.null_permutations,
            "min_test_events": args.min_test_events,
            "min_test_positive_events": args.min_test_positive_events,
        },
        "event_rows": len(rows),
        "event_counts": {f"{split}:{target}": count for (split, target), count in sorted(counts.items())},
        "summary": summary,
    }
    (args.output_dir / "dynamic_fc_switching_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "# ABIDE Dynamic-FC Switching Gate",
        "",
        CLAIM_BOUNDARY,
        "",
        f"- Verdict: `{verdict}`",
        f"- Event rows: {len(rows)}",
        f"- Models: {', '.join(sorted(models))}",
        "",
        "## Summary",
        "",
        "| model | test events | positives | Brier | log loss | AUROC | AUPRC | null AP p>= |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        md_lines.append(
            "| {model} | {test_events} | {test_positive_events} | {brier_mean:.6f} | "
            "{log_loss_mean:.6f} | {auroc_mean} | {average_precision_mean} | {null_p} |".format(
                **{
                    **row,
                    "brier_mean": float(row["brier_mean"] or 0.0),
                    "log_loss_mean": float(row["log_loss_mean"] or 0.0),
                    "auroc_mean": ""
                    if row["auroc_mean"] is None
                    else f"{float(row['auroc_mean']):.6f}",
                    "average_precision_mean": ""
                    if row["average_precision_mean"] is None
                    else f"{float(row['average_precision_mean']):.6f}",
                    "null_p": ""
                    if row["null_average_precision_p_ge_mean"] is None
                    else f"{float(row['null_average_precision_p_ge_mean']):.6f}",
                }
            )
        )
    if reasons:
        md_lines.extend(["", "## Reasons", ""])
        md_lines.extend(f"- {reason}" for reason in reasons)
    (args.output_dir / "dynamic_fc_switching_gate.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print(f"verdict={verdict}")
    print(f"outputs={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
