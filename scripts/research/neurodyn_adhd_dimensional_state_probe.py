#!/usr/bin/env python3
"""Site-aware dimensional probes for ADHD-200 O-SSM STATE_TRACE features.

This script joins traced hidden states from brain_ossm_abide output to the rich
ADHD-200 manifest. It exports dynamic state features and evaluates simple
leave-one-site-out ridge probes for dimensional ADHD subscales. It is a
diagnostic instrument, not a clinical model.
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

from neurodyn_hidden_state_separability import parse_state_rows


SCHEMA = "neurodyn.adhd_dimensional_state_probe.v1"
CLAIM_BOUNDARY = (
    "Dimensional hidden-state diagnostic only. This is not a diagnostic, "
    "biomarker, treatment-response, biological-mechanism, or O-SSM superiority claim."
)

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-999", "-9999"}


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_TOKENS


def parse_float(value: str) -> float | None:
    if is_missing(value):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def canonical_label_code(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"1", "adhd", "asd", "aut"}:
        return 1
    if text in {"0", "td", "hc", "control", "typically developing"}:
        return 0
    return None


def canonical_label_text(value: Any) -> str:
    code = canonical_label_code(value)
    if code == 1:
        return "ADHD"
    if code == 0:
        return "TD"
    return str(value).strip()


def site_hash(value: str) -> int:
    h = 5381
    for byte in value.encode("utf-8"):
        h = h * 33 + byte
    return abs(h)


def trace_site_matches(trace_site: Any, manifest_site: str) -> bool:
    trace_text = str(trace_site)
    if trace_text == manifest_site:
        return True
    try:
        return int(trace_text) == site_hash(manifest_site)
    except ValueError:
        return False


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
    reader = csv.DictReader(data_lines, delimiter="\t")
    return meta, list(reader)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject_universe_sha256(rows: list[dict[str, str]]) -> str:
    """Hash ordered subject_id/label/site rows; keep in sync with the preparer."""
    h = hashlib.sha256()
    for row in rows:
        h.update(row.get("subject_id", "").encode("utf-8"))
        h.update(b"\t")
        h.update(canonical_label_text(row.get("label", "")).encode("utf-8"))
        h.update(b"\t")
        h.update(row.get("site", "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return 0.0
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) * (v - mx) for v in x)
    vy = sum((v - my) * (v - my) for v in y)
    if vx <= 1.0e-20 or vy <= 1.0e-20:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return cov / math.sqrt(vx * vy)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx + 1
        while j < len(order) and values[order[j]] == values[order[idx]]:
            j += 1
        rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            ranks[order[k]] = rank
        idx = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    return pearson(rankdata(x), rankdata(y))


def r2_score(y: list[float], pred: list[float]) -> float:
    if len(y) < 2:
        return 0.0
    y_mean = mean(y)
    ss_tot = sum((value - y_mean) * (value - y_mean) for value in y)
    if ss_tot <= 1.0e-20:
        return 0.0
    ss_res = sum((a - b) * (a - b) for a, b in zip(y, pred, strict=True))
    return 1.0 - ss_res / ss_tot


def mae(y: list[float], pred: list[float]) -> float:
    return mean([abs(a - b) for a, b in zip(y, pred, strict=True)])


def zscore_train_apply(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma > 1.0e-12, sigma, 1.0)
    return (x_train - mu) / sigma, (x_test - mu) / sigma


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float) -> np.ndarray:
    x_train_z, x_test_z = zscore_train_apply(x_train, x_test)
    design_train = np.column_stack([np.ones(x_train_z.shape[0]), x_train_z])
    design_test = np.column_stack([np.ones(x_test_z.shape[0]), x_test_z])
    penalty = np.eye(design_train.shape[1]) * l2
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(design_train.T @ design_train + penalty) @ design_train.T @ y_train
    return design_test @ weights


def metric_value(name: str, y: list[float], pred: list[float]) -> float:
    if name == "spearman":
        return spearman(y, pred)
    if name == "pearson":
        return pearson(y, pred)
    if name == "r2":
        return r2_score(y, pred)
    if name == "mae":
        return mae(y, pred)
    raise SystemExit(f"unknown metric: {name}")


def vector_from_columns(row: dict[str, str], cols: list[str]) -> list[float] | None:
    values: list[float] = []
    for col in cols:
        value = parse_float(row.get(col, ""))
        if value is None:
            return None
        values.append(value)
    return values


def covariate_vector(row: dict[str, str]) -> list[float] | None:
    values: list[float] = []
    for col in ["age", "iq", "mean_fd"]:
        value = parse_float(row.get(col, ""))
        values.append(0.0 if value is None else value)
    sex = row.get("sex", "").strip().lower()
    if sex in {"1", "m", "male"}:
        values.append(1.0)
    elif sex in {"2", "f", "female"}:
        values.append(0.0)
    else:
        values.append(0.5)
    med = row.get("medication_status", "").strip().lower()
    if med in {"1", "yes", "y", "true", "medicated", "on"}:
        values.append(1.0)
    elif med in {"0", "no", "n", "false", "unmedicated", "off"}:
        values.append(0.0)
    else:
        values.append(0.5)
    return values


def joined_rows(
    raw_output: Path,
    manifest_rows: list[dict[str, str]],
    ossm_rows: list[dict[str, str]] | None,
    phenotypes: list[str],
) -> list[dict[str, Any]]:
    state_rows = parse_state_rows(raw_output)
    trace_subjects = {(int(row["seed"]), int(row["subject_index"])) for row in state_rows}
    if ossm_rows is not None and len(ossm_rows) != len(manifest_rows):
        raise SystemExit(f"rich manifest/view row count mismatch: {len(manifest_rows)} vs {len(ossm_rows)}")
    if ossm_rows is not None:
        for idx, (rich, view) in enumerate(zip(manifest_rows, ossm_rows, strict=True)):
            for column in ["subject_id", "site"]:
                if rich.get(column, "") != view.get(column, ""):
                    raise SystemExit(f"rich manifest/view mismatch at row {idx} column {column}")
            if canonical_label_code(rich.get("label", "")) != canonical_label_code(view.get("label", "")):
                raise SystemExit(f"rich manifest/view mismatch at row {idx} column label")
    out: list[dict[str, Any]] = []
    for row in state_rows:
        idx = int(row["subject_index"])
        if idx < 0 or idx >= len(manifest_rows):
            raise SystemExit(f"STATE_TRACE subject_index out of bounds: {idx}")
        manifest = manifest_rows[idx]
        view = ossm_rows[idx] if ossm_rows is not None else manifest
        if not manifest.get("subject_id", ""):
            raise SystemExit(f"manifest row {idx} has no subject_id")
        if not manifest.get("site", ""):
            raise SystemExit(f"manifest row {idx} has no site")
        expected_label = canonical_label_code(manifest.get("label", ""))
        if expected_label is None:
            raise SystemExit(f"manifest row {idx} has unsupported label: {manifest.get('label', '')}")
        if not trace_site_matches(row["site"], str(view.get("site", ""))):
            raise SystemExit(f"STATE_TRACE site mismatch at row {idx}: trace={row['site']} manifest={view.get('site', '')}")
        if int(row["label"]) != expected_label:
            raise SystemExit(f"STATE_TRACE label mismatch at row {idx}: trace={row['label']} manifest={manifest.get('label', '')}")
        phenotype_values = {name: parse_float(manifest.get(name, "")) for name in phenotypes}
        feature_cols = [f"f{i}" for i in range(64)]
        raw_features = vector_from_columns(manifest, feature_cols)
        covariates = covariate_vector(manifest)
        out.append(
            {
                "model": row["model"],
                "seed": int(row["seed"]),
                "subject_index": idx,
                "subject_id": manifest.get("subject_id", str(idx)),
                "site": manifest.get("site", row["site"]),
                "label": manifest.get("label", str(row["label"])),
                "h": row["h"],
                "raw_features": raw_features,
                "covariates": covariates,
                "phenotypes": phenotype_values,
            }
        )
    if not out:
        raise SystemExit("no STATE_TRACE rows joined to manifest")
    expected_trace_count = len({int(row["seed"]) for row in state_rows}) * len(manifest_rows)
    if len(trace_subjects) != expected_trace_count:
        raise SystemExit(f"STATE_TRACE subject coverage mismatch: got {len(trace_subjects)}, expected {expected_trace_count}")
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
    models = sorted({row["model"] for row in rows})
    if len(models) > 1:
        reference_model = models[0]
        reference_set = {(int(row["seed"]), str(row["subject_id"])) for row in rows if row["model"] == reference_model}
        for model in models[1:]:
            model_set = {(int(row["seed"]), str(row["subject_id"])) for row in rows if row["model"] == model}
            if model_set != reference_set:
                missing = sorted(reference_set - model_set)[:5]
                extra = sorted(model_set - reference_set)[:5]
                raise SystemExit(
                    f"cross-model subject/seed mismatch for {model} vs {reference_model}; "
                    f"missing_examples={missing}; extra_examples={extra}"
                )
    for model in models:
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
                        if len(train) < 3 or not test:
                            continue
                        if surface == "hidden":
                            x_train_list = [row["h"] for row in train]
                            x_test_list = [row["h"] for row in test]
                        elif surface in {"static_input_summary", "raw_features"}:
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
                        x_train = np.array(x_train_list, dtype=float)
                        x_test = np.array(x_test_list, dtype=float)
                        preds = ridge_fit_predict(x_train, y_train, x_test, l2=l2).tolist()
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
                                "mae": mae(y_test, preds),
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
                                    train = [row for row in seed_rows if row["site"] != site and row["phenotypes"].get(phenotype) is not None]
                                    test = [row for row in seed_rows if row["site"] == site and row["phenotypes"].get(phenotype) is not None]
                                    if surface == "hidden":
                                        x_train_list = [row["h"] for row in train]
                                        x_test_list = [row["h"] for row in test]
                                    elif surface in {"static_input_summary", "raw_features"}:
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
                                    x_train = np.array(x_train_list, dtype=float)
                                    x_test = np.array(x_test_list, dtype=float)
                                    preds = ridge_fit_predict(x_train, y_train_values, x_test, l2=l2).tolist()
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
                                "mae": mae(y_all, pred_all),
                                "null_permutations": len(null_spearman),
                                "null_spearman_mean": null_mean,
                                "null_spearman_std": null_std,
                                "null_spearman_p_ge": null_p_ge,
                                "null_mode": "phenotype_permutation_train_only_frozen_recurrent_state",
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
                        "sites_evaluated_mean": mean([len({row["holdout_site"] for row in site_rows if row["seed"] == pooled_row["seed"]}) for pooled_row in pooled]),
                        "pooled_n_mean": mean([float(row["holdout_n"]) for row in pooled]),
                        "pearson_mean": mean([float(row["pearson"]) for row in pooled]),
                        "pearson_std": pstdev([float(row["pearson"]) for row in pooled]),
                        "spearman_mean": mean([float(row["spearman"]) for row in pooled]),
                        "spearman_std": pstdev([float(row["spearman"]) for row in pooled]),
                        "r2_mean": mean([float(row["r2"]) for row in pooled]),
                        "mae_mean": mean([float(row["mae"]) for row in pooled]),
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
        "# ADHD Dimensional O-SSM State Probe",
        "",
        f"Claim boundary: {CLAIM_BOUNDARY}",
        "",
        "| model | phenotype | surface | seeds | sites eval mean | pooled n mean | Spearman mean | null p>= mean | Pearson mean | R2 mean | MAE mean |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {model} | {phenotype} | {surface} | {seed_count} | {sites:.3f} | {pooled_n:.3f} | {spearman:.6f} | {null_p:.6f} | {pearson:.6f} | {r2:.6f} | {mae:.6f} |".format(
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
                mae=row["mae_mean"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this is a site-held-out dimensional diagnostic over traced hidden states.",
            "The canonical metric is the pooled-per-seed Spearman summary, not a naive average of per-site rows.",
            "A positive value is only a hypothesis-generating representation result unless matched baselines, nulls, and leakage audits also pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ossm-manifest", type=Path, default=None, help="Compatibility view consumed by the O-SSM runner.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--primary-phenotypes", default="inattention,hyperactivity_impulsivity,adhd_total")
    parser.add_argument("--surfaces", default="hidden,covariates,static_input_summary")
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--null-permutations", type=int, default=20)
    parser.add_argument("--min-test-variance", type=float, default=1.0e-12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    phenotypes = [value.strip() for value in args.primary_phenotypes.split(",") if value.strip()]
    surfaces = [value.strip() for value in args.surfaces.split(",") if value.strip()]
    manifest_meta, manifest_rows = read_manifest(args.manifest)
    ossm_meta: dict[str, str] = {}
    ossm_rows: list[dict[str, str]] | None = None
    if args.ossm_manifest is not None:
        ossm_meta, ossm_rows = read_manifest(args.ossm_manifest)
        rich_universe = manifest_meta.get("subject_universe_sha256", subject_universe_sha256(manifest_rows))
        ossm_universe = ossm_meta.get("subject_universe_sha256", subject_universe_sha256(ossm_rows))
        if rich_universe != ossm_universe:
            raise SystemExit(f"subject universe mismatch: rich={rich_universe} ossm={ossm_universe}")
    rows = joined_rows(args.raw_output, manifest_rows, ossm_rows, phenotypes)
    summary, detail = run_probe(rows, phenotypes, surfaces, args.l2, args.null_permutations, args.min_test_variance)
    export_dynamic_features(output_dir / "adhd_dimensional_dynamic_features.tsv", rows, phenotypes)
    write_tsv(output_dir / "adhd_dimensional_state_probe_summary.tsv", summary)
    write_tsv(output_dir / "adhd_dimensional_state_probe_detail.tsv", detail)

    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "raw_output": str(args.raw_output.resolve()),
        "raw_output_sha256": sha256(args.raw_output),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "manifest_metadata": manifest_meta,
        "ossm_manifest": str(args.ossm_manifest.resolve()) if args.ossm_manifest is not None else "",
        "ossm_manifest_sha256": sha256(args.ossm_manifest) if args.ossm_manifest is not None else "",
        "ossm_manifest_metadata": ossm_meta,
        "subject_universe_sha256": manifest_meta.get("subject_universe_sha256", subject_universe_sha256(manifest_rows)),
        "joined_state_rows": len(rows),
        "primary_phenotypes": phenotypes,
        "surfaces": surfaces,
        "ridge_l2": args.l2,
        "null_permutations_requested": args.null_permutations,
        "null_mode": "phenotype_permutation_train_only_frozen_recurrent_state",
        "state_trace_provenance": "STATE_TRACE final hidden state from brain_ossm_abide; recurrent model is not retrained by this probe.",
        "summary": summary,
    }
    (output_dir / "adhd_dimensional_state_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "adhd_dimensional_state_probe.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
