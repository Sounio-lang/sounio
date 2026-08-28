#!/usr/bin/env python3
"""Benchmark the DrugBank FAERS DDI manifest with linear and O-SSM-style models."""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from ddi_campaign_lib import (
    CYP_ORDER,
    DDISample,
    load_ddi_manifest,
    load_ddi_samples,
    partial_correlation,
    pearson_r,
    spearman_r,
    stdev_or_zero,
    write_ddi_manifest,
    write_json,
    write_tsv,
)


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
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def seed_schedule(count: int) -> list[int]:
    if count <= len(BASE_SEEDS):
        return BASE_SEEDS[:count]
    seeds = list(BASE_SEEDS)
    x = BASE_SEEDS[-1]
    while len(seeds) < count:
        x = (1103515245 * x + 12345) % 2_147_483_647
        seeds.append(x)
    return seeds


def scalar_covariates(sample: DDISample) -> np.ndarray:
    values = [
        math.log1p(sample.n_drug_triples),
        math.log1p(sample.n_positive_triples),
        math.log1p(sample.cases),
        math.log1p(sample.temporal),
        float("nan") if sample.mean_age is None else sample.mean_age,
        float("nan") if sample.frac_female is None else sample.frac_female,
        float("nan") if sample.frac_hcp is None else sample.frac_hcp,
        float("nan") if sample.frac_us is None else sample.frac_us,
        float(sample.concomitant_total),
        *[float(value) for value in sample.concomitant_by_cyp],
    ]
    return np.array(values, dtype=np.float64)


def symmetric_features(sample: DDISample) -> np.ndarray:
    bag = np.zeros(len(CYP_ORDER), dtype=np.float64)
    for basis_index in (sample.basis_a, sample.basis_b, sample.basis_c):
        bag[basis_index - 1] = 1.0
    return np.concatenate([bag, scalar_covariates(sample)])


def ordered_features(sample: DDISample) -> np.ndarray:
    return np.concatenate(
        [np.array(sample.sequence, dtype=np.float64).reshape(-1), scalar_covariates(sample)]
    )


def random_unit_vector(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=length)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        vector[0] = 1.0
        norm = 1.0
    return vector / norm


def oct_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(8, dtype=np.float64)
    r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2] = a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3] = a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4] = a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5] = a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6] = a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7] = a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = np.zeros(4, dtype=np.float64)
    r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
    r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
    r[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
    r[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
    return r


def ossm_representation(
    sample: DDISample,
    seed: int,
    mode: str,
    include_assoc: bool,
) -> np.ndarray:
    sequence = np.array(sample.sequence, dtype=np.float64)
    covars = scalar_covariates(sample)
    if mode == "oct":
        a = random_unit_vector(8, seed)
        h = np.zeros(8, dtype=np.float64)
        h[0] = 1.0
        assoc_acc = 0.0
        for step in sequence:
            ah = oct_mul(a, h)
            left = oct_mul(ah, step)
            hx = oct_mul(h, step)
            right = oct_mul(a, hx)
            diff = left - right
            assoc_acc += min(float(np.linalg.norm(diff)), 1000.0)
            h = np.clip(left + 0.2 * right, -5.0, 5.0)
            h = np.where(np.isfinite(h), h, 0.0)
        if include_assoc:
            return np.concatenate([h, covars, np.array([assoc_acc / len(sequence)], dtype=np.float64)])
        return np.concatenate([h, covars])

    a = random_unit_vector(8, seed)
    q1 = a[:4] / max(np.linalg.norm(a[:4]), 1e-12)
    q2 = a[4:] / max(np.linalg.norm(a[4:]), 1e-12)
    h = np.zeros(8, dtype=np.float64)
    h[0] = 1.0
    for step in sequence:
        left1 = quat_mul(quat_mul(q1, h[:4]), step[:4])
        left2 = quat_mul(quat_mul(q2, h[4:]), step[4:])
        h = np.clip(
            np.concatenate([left1 + 0.2 * step[:4], left2 + 0.2 * step[4:]]),
            -5.0,
            5.0,
        )
        h = np.where(np.isfinite(h), h, 0.0)
    return np.concatenate([h, covars, np.array([0.0], dtype=np.float64)])


def build_matrix(samples: list[DDISample], model_name: str, seed: int | None = None) -> np.ndarray:
    if model_name == "symmetric_linear":
        return np.vstack([symmetric_features(sample) for sample in samples])
    if model_name == "ordered_linear":
        return np.vstack([ordered_features(sample) for sample in samples])
    if model_name == "h_ssm":
        assert seed is not None
        return np.vstack([ossm_representation(sample, seed, "quat", include_assoc=True) for sample in samples])
    if model_name == "o_ssm":
        assert seed is not None
        return np.vstack([ossm_representation(sample, seed, "oct", include_assoc=True) for sample in samples])
    if model_name == "o_ssm_no_assoc":
        assert seed is not None
        return np.vstack([ossm_representation(sample, seed, "oct", include_assoc=False) for sample in samples])
    raise ValueError(f"unsupported model: {model_name}")


def impute_and_scale(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    def fill(matrix: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(matrix), medians, matrix)

    train_filled = fill(train_x)
    val_filled = fill(val_x)
    test_filled = fill(test_x)

    means = train_filled.mean(axis=0)
    stds = train_filled.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return (
        (train_filled - means) / stds,
        (val_filled - means) / stds,
        (test_filled - means) / stds,
    )


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    design = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    lhs = design.T @ design + ridge_lambda * reg
    rhs = design.T @ train_y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def predict_ridge(weights: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    design = np.concatenate([matrix, np.ones((matrix.shape[0], 1), dtype=np.float64)], axis=1)
    return design @ weights


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    if weight_sum < 1e-12:
        return mae(y_true, y_pred)
    return float(np.sum(np.abs(y_pred - y_true) * weights) / weight_sum)


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def split_plan(samples: list[DDISample]) -> list[tuple[int, int, int]]:
    fold_ids = sorted({sample.fold_id for sample in samples})
    if len(fold_ids) < 3:
        raise RuntimeError("benchmark requires at least 3 folds for train/val/test")
    plan = []
    for fold_id in fold_ids:
        val_fold = fold_ids[(fold_ids.index(fold_id) + 1) % len(fold_ids)]
        plan.append((fold_id, val_fold, fold_id))
    return plan


def summarize_metrics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["model"])].append(record)

    rows: list[dict[str, object]] = []
    for model_name, model_rows in sorted(grouped.items()):
        maes = [float(row["mae"]) for row in model_rows]
        rmses = [float(row["rmse"]) for row in model_rows]
        weighted_maes = [float(row["weighted_mae_temporal"]) for row in model_rows]
        biases = [float(row["bias"]) for row in model_rows]
        pearsons = [float(row["pearson_r"]) for row in model_rows]
        spearmans = [float(row["spearman_r"]) for row in model_rows]
        rows.append(
            {
                "model": model_name,
                "run_count": len(model_rows),
                "seed_count": len({row["seed"] for row in model_rows}),
                "mae_mean": statistics.fmean(maes),
                "mae_std": stdev_or_zero(maes),
                "rmse_mean": statistics.fmean(rmses),
                "rmse_std": stdev_or_zero(rmses),
                "weighted_mae_temporal_mean": statistics.fmean(weighted_maes),
                "weighted_mae_temporal_std": stdev_or_zero(weighted_maes),
                "bias_mean": statistics.fmean(biases),
                "bias_std": stdev_or_zero(biases),
                "pearson_r_mean": statistics.fmean(pearsons),
                "pearson_r_std": stdev_or_zero(pearsons),
                "spearman_r_mean": statistics.fmean(spearmans),
                "spearman_r_std": stdev_or_zero(spearmans),
            }
        )
    return rows


def covariate_robustness(samples: list[DDISample], predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model_sample: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in predictions:
        by_model_sample[(str(row["model"]), str(row["triple_id"]))].append(float(row["prediction"]))

    sample_by_id = {sample.triple_id: sample for sample in samples}
    models = sorted({model for model, _ in by_model_sample})
    rows: list[dict[str, object]] = []
    for model_name in models:
        triples = sorted(triple_id for current_model, triple_id in by_model_sample if current_model == model_name)
        averaged_predictions = [statistics.fmean(by_model_sample[(model_name, triple_id)]) for triple_id in triples]
        targets = [sample_by_id[triple_id].asymmetry for triple_id in triples]
        raw_r, raw_p = pearson_r(averaged_predictions, targets)

        exposure_covars = []
        demo_covars = []
        full_pred = []
        full_target = []
        for triple_id, pred in zip(triples, averaged_predictions, strict=True):
            sample = sample_by_id[triple_id]
            exposure_covars.append([
                math.log1p(sample.n_drug_triples),
                math.log1p(sample.n_positive_triples),
                math.log1p(sample.cases),
                math.log1p(sample.temporal),
                float(sample.concomitant_total),
            ])
            if sample.mean_age is not None and sample.frac_female is not None:
                demo_covars.append([
                    sample.mean_age,
                    sample.frac_female,
                    float(sample.concomitant_total),
                ])
                full_pred.append(pred)
                full_target.append(sample.asymmetry)

        exposure_controls = list(map(list, zip(*exposure_covars, strict=True)))
        exposure_r, exposure_p = partial_correlation(averaged_predictions, targets, exposure_controls)

        if len(full_pred) >= 6:
            demo_controls = list(map(list, zip(*demo_covars, strict=True)))
            demo_r, demo_p = partial_correlation(full_pred, full_target, demo_controls)
            demo_n = len(full_pred)
        else:
            demo_r = demo_p = None
            demo_n = 0

        rows.append(
            {
                "model": model_name,
                "n_samples": len(triples),
                "raw_pearson_r": raw_r,
                "raw_pearson_p": raw_p,
                "partial_exposure_r": exposure_r,
                "partial_exposure_p": exposure_p,
                "partial_demo_r": demo_r,
                "partial_demo_p": demo_p,
                "partial_demo_n": demo_n,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DrugBank FAERS DDI manifest with O-SSM baselines")
    parser.add_argument(
        "--output-dir",
        default="artifacts/research/ddi_drugbank_benchmark",
        help="Output directory for manifest, raw predictions, and summaries",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to an existing manifest. If omitted, the script will materialize the canonical manifest first.",
    )
    parser.add_argument(
        "--seed-count",
        type=int,
        default=20,
        help="Number of random seeds for the seeded O/H-SSM representations",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
    else:
        manifest_path = output_dir / "ddi_drugbank_manifest.tsv"
        if not manifest_path.exists():
            samples = load_ddi_samples(repo_root)
            write_ddi_manifest(samples, manifest_path)

    samples = load_ddi_manifest(manifest_path)
    fold_ids = sorted({sample.fold_id for sample in samples})
    seeds = seed_schedule(args.seed_count)
    deterministic_models = ["symmetric_linear", "ordered_linear"]
    seeded_models = ["h_ssm", "o_ssm", "o_ssm_no_assoc"]

    per_run_metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for test_fold in fold_ids:
        val_fold = fold_ids[(fold_ids.index(test_fold) + 1) % len(fold_ids)]
        train_samples = [sample for sample in samples if sample.fold_id not in {test_fold, val_fold}]
        val_samples = [sample for sample in samples if sample.fold_id == val_fold]
        test_samples = [sample for sample in samples if sample.fold_id == test_fold]

        all_models = [(model_name, None) for model_name in deterministic_models]
        all_models.extend((model_name, seed) for model_name in seeded_models for seed in seeds)

        for model_name, seed in all_models:
            train_x = build_matrix(train_samples, model_name, seed)
            val_x = build_matrix(val_samples, model_name, seed)
            test_x = build_matrix(test_samples, model_name, seed)
            train_x, val_x, test_x = impute_and_scale(train_x, val_x, test_x)

            train_y = np.array([sample.asymmetry for sample in train_samples], dtype=np.float64)
            val_y = np.array([sample.asymmetry for sample in val_samples], dtype=np.float64)
            combined_train_x = np.concatenate([train_x, val_x], axis=0)
            combined_train_y = np.concatenate([train_y, val_y], axis=0)

            best_lambda = LAMBDA_GRID[0]
            best_val_mae = float("inf")
            for ridge_lambda in LAMBDA_GRID:
                weights = fit_ridge(train_x, train_y, ridge_lambda)
                current_val_pred = predict_ridge(weights, val_x)
                current_val_mae = mae(val_y, current_val_pred)
                if current_val_mae < best_val_mae:
                    best_val_mae = current_val_mae
                    best_lambda = ridge_lambda

            final_weights = fit_ridge(combined_train_x, combined_train_y, best_lambda)
            test_pred = predict_ridge(final_weights, test_x)
            test_y = np.array([sample.asymmetry for sample in test_samples], dtype=np.float64)
            temporal_weights = np.array([sample.temporal for sample in test_samples], dtype=np.float64)
            pearson_val, _ = pearson_r(test_pred.tolist(), test_y.tolist())
            spearman_val, _ = spearman_r(test_pred.tolist(), test_y.tolist())

            per_run_metrics.append(
                {
                    "model": model_name,
                    "seed": "deterministic" if seed is None else seed,
                    "fold_id": test_fold,
                    "val_fold_id": val_fold,
                    "n_train": len(train_samples),
                    "n_val": len(val_samples),
                    "n_test": len(test_samples),
                    "ridge_lambda": best_lambda,
                    "mae": mae(test_y, test_pred),
                    "rmse": rmse(test_y, test_pred),
                    "weighted_mae_temporal": weighted_mae(test_y, test_pred, temporal_weights),
                    "bias": mean_bias(test_y, test_pred),
                    "pearson_r": pearson_val,
                    "spearman_r": spearman_val,
                }
            )

            for sample, pred in zip(test_samples, test_pred.tolist(), strict=True):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "seed": "deterministic" if seed is None else seed,
                        "fold_id": test_fold,
                        "val_fold_id": val_fold,
                        "triple_id": sample.triple_id,
                        "anchor_group": sample.anchor_group,
                        "fano": int(sample.fano),
                        "target": sample.asymmetry,
                        "prediction": pred,
                        "abs_error": abs(pred - sample.asymmetry),
                        "assoc_norm": sample.assoc_norm,
                        "cases": sample.cases,
                        "temporal": sample.temporal,
                    }
                )

    overall_rows = summarize_metrics(per_run_metrics)
    covariate_rows = covariate_robustness(samples, prediction_rows)

    write_tsv(
        output_dir / "fold_metrics.tsv",
        per_run_metrics,
        fieldnames=[
            "model",
            "seed",
            "fold_id",
            "val_fold_id",
            "n_train",
            "n_val",
            "n_test",
            "ridge_lambda",
            "mae",
            "rmse",
            "weighted_mae_temporal",
            "bias",
            "pearson_r",
            "spearman_r",
        ],
    )
    write_tsv(
        output_dir / "predictions.tsv",
        prediction_rows,
        fieldnames=[
            "model",
            "seed",
            "fold_id",
            "val_fold_id",
            "triple_id",
            "anchor_group",
            "fano",
            "target",
            "prediction",
            "abs_error",
            "assoc_norm",
            "cases",
            "temporal",
        ],
    )
    write_tsv(
        output_dir / "overall_metrics.tsv",
        overall_rows,
        fieldnames=[
            "model",
            "run_count",
            "seed_count",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "weighted_mae_temporal_mean",
            "weighted_mae_temporal_std",
            "bias_mean",
            "bias_std",
            "pearson_r_mean",
            "pearson_r_std",
            "spearman_r_mean",
            "spearman_r_std",
        ],
    )
    write_tsv(
        output_dir / "covariate_robustness.tsv",
        covariate_rows,
        fieldnames=[
            "model",
            "n_samples",
            "raw_pearson_r",
            "raw_pearson_p",
            "partial_exposure_r",
            "partial_exposure_p",
            "partial_demo_r",
            "partial_demo_p",
            "partial_demo_n",
        ],
    )

    best_row = min(overall_rows, key=lambda row: float(row["weighted_mae_temporal_mean"]))
    summary = {
        "dataset": "faers_drugbank.csv",
        "manifest": str(manifest_path.relative_to(repo_root)),
        "samples": len(samples),
        "folds": len(fold_ids),
        "seed_count_seeded_models": len(seeds),
        "models": overall_rows,
        "covariate_robustness": covariate_rows,
        "best_model_by_weighted_mae_temporal": {
            "model": best_row["model"],
            "weighted_mae_temporal_mean": best_row["weighted_mae_temporal_mean"],
        },
    }
    write_json(output_dir / "summary.json", summary)

    print("DDI DrugBank benchmark complete")
    print(f"Manifest: {manifest_path.relative_to(repo_root)}")
    for row in overall_rows:
        print(
            f"{row['model']}: weighted_mae_temporal={row['weighted_mae_temporal_mean']:.4f} "
            f"pearson={row['pearson_r_mean']:.4f}"
        )


if __name__ == "__main__":
    main()

