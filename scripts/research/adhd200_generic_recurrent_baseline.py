#!/usr/bin/env python3
"""Generic recurrent dimensional baselines for ADHD-200 manifests.

This is a lightweight non-hypercomplex control surface for pilot runs. It uses
deterministic GRU-style recurrent reservoirs plus site-held-out ridge readouts
over dimensional phenotypes, and can optionally run small trained Elman-RNN
regressors. It is not a full S4/Transformer suite; it is the small reproducible
generic-control layer that should be present before interpreting O-SSM hidden
states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from neurodyn_adhd_dimensional_state_probe import (
    CLAIM_BOUNDARY as STATE_PROBE_CLAIM_BOUNDARY,
    covariate_vector,
    parse_float,
    pearson,
    pstdev,
    r2_score,
    ridge_fit_predict,
    spearman,
    subject_universe_sha256,
    vector_from_columns,
)


SCHEMA = "neurodyn.adhd200_generic_recurrent_baseline.v1"
CLAIM_BOUNDARY = (
    "Generic recurrent baseline only. This is not a diagnostic, "
    "biomarker, treatment-response, biological-mechanism, or O-SSM superiority claim."
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def read_manifest(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        if raw.startswith("#"):
            body = raw[1:].strip()
            if "=" in body:
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
            continue
        data_lines.append(raw)
    if not data_lines:
        raise SystemExit(f"manifest has no rows: {path}")
    return meta, list(csv.DictReader(data_lines, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def feature_matrix(rows: list[dict[str, str]], seq_len: int, input_dim: int) -> np.ndarray:
    feature_cols = [f"f{i}" for i in range(seq_len * input_dim)]
    matrix: list[list[float]] = []
    for idx, row in enumerate(rows):
        values = vector_from_columns(row, feature_cols)
        if values is None:
            raise SystemExit(f"manifest row {idx} has nonnumeric recurrent feature")
        matrix.append(values)
    x = np.array(matrix, dtype=float).reshape(len(rows), seq_len, input_dim)
    mu = x.mean(axis=(0, 1), keepdims=True)
    sigma = x.std(axis=(0, 1), keepdims=True)
    sigma = np.where(sigma > 1.0e-12, sigma, 1.0)
    return (x - mu) / sigma


def model_hidden_dim(model: str) -> int:
    if model in {"gru_reservoir", "trained_rnn"}:
        return 16
    if model in {"gru_reservoir_wide", "trained_rnn_wide"}:
        return 32
    raise SystemExit(f"unknown generic recurrent baseline model: {model}")


def is_reservoir_model(model: str) -> bool:
    return model in {"gru_reservoir", "gru_reservoir_wide"}


def is_trained_model(model: str) -> bool:
    return model in {"trained_rnn", "trained_rnn_wide"}


def init_weights(seed: int, input_dim: int, hidden_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    in_scale = 1.0 / math.sqrt(max(1, input_dim))
    hid_scale = 0.75 / math.sqrt(max(1, hidden_dim))
    return {
        "wz": rng.normal(0.0, in_scale, size=(input_dim, hidden_dim)),
        "wr": rng.normal(0.0, in_scale, size=(input_dim, hidden_dim)),
        "wn": rng.normal(0.0, in_scale, size=(input_dim, hidden_dim)),
        "uz": rng.normal(0.0, hid_scale, size=(hidden_dim, hidden_dim)),
        "ur": rng.normal(0.0, hid_scale, size=(hidden_dim, hidden_dim)),
        "un": rng.normal(0.0, hid_scale, size=(hidden_dim, hidden_dim)),
        "bz": rng.normal(0.0, 0.05, size=(hidden_dim,)),
        "br": rng.normal(0.0, 0.05, size=(hidden_dim,)),
        "bn": rng.normal(0.0, 0.05, size=(hidden_dim,)),
    }


def run_gru_reservoir(x: np.ndarray, seed: int, model: str) -> np.ndarray:
    hidden_dim = model_hidden_dim(model)
    weights = init_weights(seed, x.shape[2], hidden_dim)
    h = np.zeros((x.shape[0], hidden_dim), dtype=float)
    for step in range(x.shape[1]):
        xt = x[:, step, :]
        z = sigmoid(xt @ weights["wz"] + h @ weights["uz"] + weights["bz"])
        r = sigmoid(xt @ weights["wr"] + h @ weights["ur"] + weights["br"])
        n = np.tanh(xt @ weights["wn"] + (r * h) @ weights["un"] + weights["bn"])
        h = (1.0 - z) * n + z * h
    return h


def init_rnn_params(seed: int, input_dim: int, hidden_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "wxh": rng.normal(0.0, 1.0 / math.sqrt(max(1, input_dim)), size=(input_dim, hidden_dim)),
        "whh": rng.normal(0.0, 0.45 / math.sqrt(max(1, hidden_dim)), size=(hidden_dim, hidden_dim)),
        "b": np.zeros(hidden_dim, dtype=float),
        "wo": rng.normal(0.0, 0.15 / math.sqrt(max(1, hidden_dim)), size=(hidden_dim,)),
        "bo": np.zeros(1, dtype=float),
    }


def rnn_forward(x: np.ndarray, params: dict[str, np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    h = np.zeros((x.shape[0], params["b"].shape[0]), dtype=float)
    states: list[np.ndarray] = []
    for step in range(x.shape[1]):
        h = np.tanh(x[:, step, :] @ params["wxh"] + h @ params["whh"] + params["b"])
        states.append(h)
    pred = states[-1] @ params["wo"] + float(params["bo"][0])
    return pred, states


def train_rnn_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    model: str,
    epochs: int,
    lr: float,
    l2: float,
) -> dict[str, np.ndarray]:
    params = init_rnn_params(seed, x_train.shape[2], model_hidden_dim(model))
    n = max(1, x_train.shape[0])
    for _ in range(epochs):
        pred, states = rnn_forward(x_train, params)
        dy = (2.0 / n) * (pred - y_train)
        grads = {
            "wxh": np.zeros_like(params["wxh"]),
            "whh": np.zeros_like(params["whh"]),
            "b": np.zeros_like(params["b"]),
            "wo": states[-1].T @ dy + l2 * params["wo"],
            "bo": np.array([dy.sum()]),
        }
        dh = dy[:, None] * params["wo"][None, :]
        for step in range(x_train.shape[1] - 1, -1, -1):
            h = states[step]
            h_prev = states[step - 1] if step > 0 else np.zeros_like(h)
            da = dh * (1.0 - h * h)
            grads["wxh"] += x_train[:, step, :].T @ da + l2 * params["wxh"]
            grads["whh"] += h_prev.T @ da + l2 * params["whh"]
            grads["b"] += da.sum(axis=0)
            dh = da @ params["whh"].T
        total_norm = math.sqrt(sum(float((grad * grad).sum()) for grad in grads.values()))
        scale = 1.0 if total_norm <= 5.0 else 5.0 / total_norm
        for key in params:
            params[key] = params[key] - lr * scale * grads[key]
    return params


def trained_rnn_predictions(
    manifest_rows: list[dict[str, str]],
    x: np.ndarray,
    models: list[str],
    seeds: list[int],
    phenotypes: list[str],
    epochs: int,
    lr: float,
    l2: float,
    null_permutations: int,
    min_test_variance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    sites = sorted({row.get("site", "") for row in manifest_rows})
    labels = [row.get("label", "") for row in manifest_rows]
    subject_ids = [row.get("subject_id", str(idx)) for idx, row in enumerate(manifest_rows)]
    for model in models:
        for phenotype in phenotypes:
            pooled_rows: list[dict[str, Any]] = []
            for seed in seeds:
                y_all: list[float] = []
                pred_all: list[float] = []
                seed_detail: list[dict[str, Any]] = []
                null_spearman: list[float] = []
                for site in sites:
                    train_idx = [
                        idx
                        for idx, row in enumerate(manifest_rows)
                        if row.get("site", "") != site and parse_float(row.get(phenotype, "")) is not None
                    ]
                    test_idx = [
                        idx
                        for idx, row in enumerate(manifest_rows)
                        if row.get("site", "") == site and parse_float(row.get(phenotype, "")) is not None
                    ]
                    if len(train_idx) < 3 or not test_idx:
                        continue
                    y_train_raw = np.array([float(parse_float(manifest_rows[idx].get(phenotype, ""))) for idx in train_idx], dtype=float)
                    y_test = [float(parse_float(manifest_rows[idx].get(phenotype, ""))) for idx in test_idx]
                    if len(y_test) > 1 and statistics.pvariance(y_test) < min_test_variance:
                        continue
                    y_mu = float(y_train_raw.mean())
                    y_sigma = float(y_train_raw.std())
                    if y_sigma <= 1.0e-12:
                        y_sigma = 1.0
                    y_train = (y_train_raw - y_mu) / y_sigma
                    params = train_rnn_regressor(x[train_idx], y_train, seed, model, epochs, lr, l2)
                    pred_z, hidden_states = rnn_forward(x[test_idx], params)
                    preds = (pred_z * y_sigma + y_mu).tolist()
                    y_all.extend(y_test)
                    pred_all.extend(float(value) for value in preds)
                    for local_idx, subject_idx in enumerate(test_idx):
                        seed_detail.append(
                            {
                                "model": model,
                                "seed": seed,
                                "phenotype": phenotype,
                                "holdout_site": site,
                                "subject_index": subject_idx,
                                "subject_id": subject_ids[subject_idx],
                                "site": site,
                                "label": labels[subject_idx],
                                "target": y_test[local_idx],
                                "prediction": float(preds[local_idx]),
                                "h_norm": float(np.linalg.norm(hidden_states[-1][local_idx])),
                            }
                        )
                    if null_permutations > 0:
                        rng = np.random.default_rng(seed * 4001 + len(phenotype) * 331 + len(site))
                        for _ in range(null_permutations):
                            shuffled = y_train.copy()
                            rng.shuffle(shuffled)
                            null_params = train_rnn_regressor(
                                x[train_idx],
                                shuffled,
                                seed + 17017,
                                model,
                                max(1, epochs // 2),
                                lr,
                                l2,
                            )
                            null_pred_z, _ = rnn_forward(x[test_idx], null_params)
                            null_preds = (null_pred_z * y_sigma + y_mu).tolist()
                            null_spearman.append(spearman(y_test, [float(value) for value in null_preds]))
                detail.extend(seed_detail)
                if y_all:
                    observed = spearman(y_all, pred_all)
                    if null_spearman:
                        ge = sum(1 for value in null_spearman if value >= observed)
                        null_p_ge = (ge + 1.0) / (len(null_spearman) + 1.0)
                        null_mean = mean(null_spearman)
                        null_std = pstdev(null_spearman)
                    else:
                        null_p_ge = 1.0
                        null_mean = 0.0
                        null_std = 0.0
                    pooled_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "phenotype": phenotype,
                            "surface": "trained_recurrent_prediction",
                            "sites_evaluated": len({row["holdout_site"] for row in seed_detail}),
                            "pooled_n": len(y_all),
                            "pearson": pearson(y_all, pred_all),
                            "spearman": observed,
                            "r2": r2_score(y_all, pred_all),
                            "null_permutations": len(null_spearman),
                            "null_spearman_mean": null_mean,
                            "null_spearman_std": null_std,
                            "null_spearman_p_ge": null_p_ge,
                            "null_mode": "phenotype_permutation_retrained_generic_recurrent_regressor",
                        }
                    )
            for row in pooled_rows:
                detail.append(row | {"holdout_site": "__pooled__"})
            summary.append(
                {
                    "model": model,
                    "phenotype": phenotype,
                    "surface": "trained_recurrent_prediction",
                    "seed_count": len(pooled_rows),
                    "sites_evaluated_mean": mean([float(row["sites_evaluated"]) for row in pooled_rows]),
                    "pooled_n_mean": mean([float(row["pooled_n"]) for row in pooled_rows]),
                    "pearson_mean": mean([float(row["pearson"]) for row in pooled_rows]),
                    "pearson_std": pstdev([float(row["pearson"]) for row in pooled_rows]),
                    "spearman_mean": mean([float(row["spearman"]) for row in pooled_rows]),
                    "spearman_std": pstdev([float(row["spearman"]) for row in pooled_rows]),
                    "r2_mean": mean([float(row["r2"]) for row in pooled_rows]),
                    "null_permutations_mean": mean([float(row["null_permutations"]) for row in pooled_rows]),
                    "null_spearman_mean": mean([float(row["null_spearman_mean"]) for row in pooled_rows]),
                    "null_spearman_p_ge_mean": mean([float(row["null_spearman_p_ge"]) for row in pooled_rows]),
                    "canonical_metric": "pooled_per_seed_spearman_mean",
                }
            )
    return summary, detail


def build_rows(
    manifest_rows: list[dict[str, str]],
    x: np.ndarray,
    models: list[str],
    seeds: list[int],
    phenotypes: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    feature_cols = [f"f{i}" for i in range(x.shape[1] * x.shape[2])]
    for model in models:
        for seed in seeds:
            hidden = run_gru_reservoir(x, seed, model)
            for idx, row in enumerate(manifest_rows):
                out.append(
                    {
                        "model": model,
                        "seed": seed,
                        "subject_index": idx,
                        "subject_id": row.get("subject_id", str(idx)),
                        "site": row.get("site", ""),
                        "label": row.get("label", ""),
                        "h": hidden[idx].tolist(),
                        "raw_features": vector_from_columns(row, feature_cols),
                        "covariates": covariate_vector(row),
                        "phenotypes": {name: parse_float(row.get(name, "")) for name in phenotypes},
                    }
                )
    return out


def run_probe(
    rows: list[dict[str, Any]],
    phenotypes: list[str],
    surfaces: list[str],
    l2: float,
    null_permutations: int,
    min_test_variance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        seeds = sorted({int(row["seed"]) for row in model_rows})
        for phenotype in phenotypes:
            for surface in surfaces:
                fold_metrics: list[dict[str, Any]] = []
                for seed in seeds:
                    seed_rows = [row for row in model_rows if int(row["seed"]) == seed]
                    sites = sorted({row["site"] for row in seed_rows})
                    y_all: list[float] = []
                    pred_all: list[float] = []
                    for site in sites:
                        train = [row for row in seed_rows if row["site"] != site and row["phenotypes"].get(phenotype) is not None]
                        test = [row for row in seed_rows if row["site"] == site and row["phenotypes"].get(phenotype) is not None]
                        if surface == "hidden":
                            x_train_list = [row["h"] for row in train]
                            x_test_list = [row["h"] for row in test]
                        elif surface == "static_input_summary":
                            train = [row for row in train if row["raw_features"] is not None]
                            test = [row for row in test if row["raw_features"] is not None]
                            x_train_list = [row["raw_features"] for row in train]
                            x_test_list = [row["raw_features"] for row in test]
                        elif surface == "covariates":
                            train = [row for row in train if row["covariates"] is not None]
                            test = [row for row in test if row["covariates"] is not None]
                            x_train_list = [row["covariates"] for row in train]
                            x_test_list = [row["covariates"] for row in test]
                        else:
                            raise SystemExit(f"unknown surface: {surface}")
                        if len(train) < 3 or not test:
                            continue
                        y_train = np.array([float(row["phenotypes"][phenotype]) for row in train], dtype=float)
                        y_test = [float(row["phenotypes"][phenotype]) for row in test]
                        if len(y_test) > 1 and statistics.pvariance(y_test) < min_test_variance:
                            continue
                        preds = ridge_fit_predict(
                            np.array(x_train_list, dtype=float),
                            y_train,
                            np.array(x_test_list, dtype=float),
                            l2=l2,
                        ).tolist()
                        y_all.extend(y_test)
                        pred_all.extend(float(value) for value in preds)
                        fold_metrics.append(
                            {
                                "model": model,
                                "seed": seed,
                                "phenotype": phenotype,
                                "surface": surface,
                                "holdout_site": site,
                                "train_n": len(train),
                                "holdout_n": len(test),
                                "pearson": pearson(y_test, preds),
                                "spearman": spearman(y_test, preds),
                                "r2": r2_score(y_test, preds),
                            }
                        )
                    if y_all:
                        observed_spearman = spearman(y_all, pred_all)
                        null_spearman: list[float] = []
                        if null_permutations > 0:
                            rng = np.random.default_rng(seed * 1009 + len(phenotype) * 9176 + len(surface) * 131)
                            for _ in range(null_permutations):
                                null_y_all: list[float] = []
                                null_pred_all: list[float] = []
                                for site in sites:
                                    train = [
                                        row
                                        for row in seed_rows
                                        if row["site"] != site and row["phenotypes"].get(phenotype) is not None
                                    ]
                                    test = [
                                        row
                                        for row in seed_rows
                                        if row["site"] == site and row["phenotypes"].get(phenotype) is not None
                                    ]
                                    if surface == "hidden":
                                        x_train_list = [row["h"] for row in train]
                                        x_test_list = [row["h"] for row in test]
                                    elif surface == "static_input_summary":
                                        train = [row for row in train if row["raw_features"] is not None]
                                        test = [row for row in test if row["raw_features"] is not None]
                                        x_train_list = [row["raw_features"] for row in train]
                                        x_test_list = [row["raw_features"] for row in test]
                                    elif surface == "covariates":
                                        train = [row for row in train if row["covariates"] is not None]
                                        test = [row for row in test if row["covariates"] is not None]
                                        x_train_list = [row["covariates"] for row in train]
                                        x_test_list = [row["covariates"] for row in test]
                                    else:
                                        raise SystemExit(f"unknown surface: {surface}")
                                    if len(train) < 3 or not test:
                                        continue
                                    y_test = [float(row["phenotypes"][phenotype]) for row in test]
                                    if len(y_test) > 1 and statistics.pvariance(y_test) < min_test_variance:
                                        continue
                                    y_train_values = np.array([float(row["phenotypes"][phenotype]) for row in train], dtype=float)
                                    rng.shuffle(y_train_values)
                                    preds = ridge_fit_predict(
                                        np.array(x_train_list, dtype=float),
                                        y_train_values,
                                        np.array(x_test_list, dtype=float),
                                        l2=l2,
                                    ).tolist()
                                    null_y_all.extend(y_test)
                                    null_pred_all.extend(float(value) for value in preds)
                                if null_y_all:
                                    null_spearman.append(spearman(null_y_all, null_pred_all))
                        if null_spearman:
                            ge = sum(1 for value in null_spearman if value >= observed_spearman)
                            null_p_ge = (ge + 1.0) / (len(null_spearman) + 1.0)
                            null_mean = mean(null_spearman)
                            null_std = pstdev(null_spearman)
                        else:
                            null_p_ge = 1.0
                            null_mean = 0.0
                            null_std = 0.0
                        fold_metrics.append(
                            {
                                "model": model,
                                "seed": seed,
                                "phenotype": phenotype,
                                "surface": surface,
                                "holdout_site": "__pooled__",
                                "train_n": 0,
                                "holdout_n": len(y_all),
                                "pearson": pearson(y_all, pred_all),
                                "spearman": observed_spearman,
                                "r2": r2_score(y_all, pred_all),
                                "null_permutations": len(null_spearman),
                                "null_spearman_mean": null_mean,
                                "null_spearman_std": null_std,
                                "null_spearman_p_ge": null_p_ge,
                                "null_mode": "phenotype_permutation_train_only_frozen_generic_recurrent_state",
                            }
                        )
                detail.extend(fold_metrics)
                pooled = [row for row in fold_metrics if row["holdout_site"] == "__pooled__"]
                site_rows = [row for row in fold_metrics if row["holdout_site"] != "__pooled__"]
                summary.append(
                    {
                        "model": model,
                        "phenotype": phenotype,
                        "surface": surface,
                        "seed_count": len(pooled),
                        "sites_evaluated_mean": mean(
                            [len({row["holdout_site"] for row in site_rows if row["seed"] == pooled_row["seed"]}) for pooled_row in pooled]
                        ),
                        "pooled_n_mean": mean([float(row["holdout_n"]) for row in pooled]),
                        "pearson_mean": mean([float(row["pearson"]) for row in pooled]),
                        "pearson_std": pstdev([float(row["pearson"]) for row in pooled]),
                        "spearman_mean": mean([float(row["spearman"]) for row in pooled]),
                        "spearman_std": pstdev([float(row["spearman"]) for row in pooled]),
                        "r2_mean": mean([float(row["r2"]) for row in pooled]),
                        "null_permutations_mean": mean([float(row.get("null_permutations", 0)) for row in pooled]),
                        "null_spearman_mean": mean([float(row.get("null_spearman_mean", 0.0)) for row in pooled]),
                        "null_spearman_p_ge_mean": mean([float(row.get("null_spearman_p_ge", 1.0)) for row in pooled]),
                        "canonical_metric": "pooled_per_seed_spearman_mean",
                    }
                )
    return summary, detail


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_dynamic_features(path: Path, rows: list[dict[str, Any]], phenotypes: list[str]) -> None:
    out: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "model": row["model"],
            "seed": row["seed"],
            "subject_index": row["subject_index"],
            "subject_id": row["subject_id"],
            "site": row["site"],
            "label": row["label"],
        }
        for phenotype in phenotypes:
            value = row["phenotypes"].get(phenotype)
            record[phenotype] = "" if value is None else value
        for idx, value in enumerate(row["h"]):
            record[f"h{idx}"] = value
        out.append(record)
    write_tsv(path, out)


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ADHD-200 Generic Recurrent Baseline",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "This pilot baseline uses deterministic GRU-style recurrent reservoirs plus ridge readouts and optional small trained Elman-RNN regressors.",
        "It is a generic non-hypercomplex control surface, not a full S4/Transformer baseline suite.",
        "",
        "| model | phenotype | surface | seeds | sites eval mean | pooled n mean | Spearman mean | null p>= mean | Pearson mean | R2 mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {model} | {phenotype} | {surface} | {seed_count} | {sites:.3f} | {pooled_n:.3f} | {spearman:.6f} | {null_p:.6f} | {pearson:.6f} | {r2:.6f} |".format(
                model=row["model"],
                phenotype=row["phenotype"],
                surface=row["surface"],
                seed_count=row["seed_count"],
                sites=row["sites_evaluated_mean"],
                pooled_n=row["pooled_n_mean"],
                spearman=row["spearman_mean"],
                null_p=row["null_spearman_p_ge_mean"],
                pearson=row["pearson_mean"],
                r2=row["r2_mean"],
            )
        )
    lines.extend(["", f"State-probe claim boundary inherited for comparison: {STATE_PROBE_CLAIM_BOUNDARY}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", default="gru_reservoir,gru_reservoir_wide")
    parser.add_argument("--seeds", default="55555,11111,99887,22222,44444,66666,77777,88888,33333,12321,24680,13579,54321,67890,11223,44567,77889,99001,31415,27182")
    parser.add_argument("--primary-phenotypes", default="inattention,hyperactivity_impulsivity,adhd_total")
    parser.add_argument("--surfaces", default="hidden,covariates,static_input_summary")
    parser.add_argument("--trained-epochs", type=int, default=80)
    parser.add_argument("--trained-lr", type=float, default=0.01)
    parser.add_argument("--trained-l2", type=float, default=1.0e-4)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--null-permutations", type=int, default=20)
    parser.add_argument("--min-test-variance", type=float, default=1.0e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta, manifest_rows = read_manifest(args.manifest)
    seq_len = int(meta.get("seq_len", "8"))
    input_dim = int(meta.get("input_dim", "8"))
    phenotypes = [value.strip() for value in args.primary_phenotypes.split(",") if value.strip()]
    surfaces = [value.strip() for value in args.surfaces.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    unknown_models = [model for model in models if not (is_reservoir_model(model) or is_trained_model(model))]
    if unknown_models:
        raise SystemExit(f"unknown model(s): {','.join(unknown_models)}")

    x = feature_matrix(manifest_rows, seq_len, input_dim)
    reservoir_models = [model for model in models if is_reservoir_model(model)]
    trained_models = [model for model in models if is_trained_model(model)]
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    trained_summary: list[dict[str, Any]] = []
    trained_detail: list[dict[str, Any]] = []
    if reservoir_models:
        rows = build_rows(manifest_rows, x, reservoir_models, seeds, phenotypes)
        summary, detail = run_probe(rows, phenotypes, surfaces, args.l2, args.null_permutations, args.min_test_variance)
        export_dynamic_features(output_dir / "adhd_generic_recurrent_dynamic_features.tsv", rows, phenotypes)
    else:
        (output_dir / "adhd_generic_recurrent_dynamic_features.tsv").write_text("", encoding="utf-8")
    if trained_models:
        trained_summary, trained_detail = trained_rnn_predictions(
            manifest_rows,
            x,
            trained_models,
            seeds,
            phenotypes,
            args.trained_epochs,
            args.trained_lr,
            args.trained_l2,
            args.null_permutations,
            args.min_test_variance,
        )
    write_tsv(output_dir / "adhd_generic_recurrent_baseline_summary.tsv", summary + trained_summary)
    write_tsv(output_dir / "adhd_generic_recurrent_baseline_detail.tsv", detail)
    write_tsv(output_dir / "adhd_trained_generic_recurrent_predictions.tsv", trained_detail)

    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "manifest_metadata": meta,
        "subject_universe_sha256": meta.get("subject_universe_sha256", subject_universe_sha256(manifest_rows)),
        "row_count": len(manifest_rows),
        "seq_len": seq_len,
        "input_dim": input_dim,
        "models": models,
        "reservoir_models": reservoir_models,
        "trained_models": trained_models,
        "seeds": seeds,
        "primary_phenotypes": phenotypes,
        "surfaces": surfaces,
        "ridge_l2": args.l2,
        "trained_epochs": args.trained_epochs,
        "trained_lr": args.trained_lr,
        "trained_l2": args.trained_l2,
        "null_permutations_requested": args.null_permutations,
        "null_modes": [
            "phenotype_permutation_train_only_frozen_generic_recurrent_state",
            "phenotype_permutation_retrained_generic_recurrent_regressor",
        ],
        "baseline_boundary": "Reservoir plus small trained RNN baselines; full S4/Transformer suite remains required before promotion-scale claims.",
        "summary": summary + trained_summary,
    }
    (output_dir / "adhd_generic_recurrent_baseline.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "adhd_generic_recurrent_baseline.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
