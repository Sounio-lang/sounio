#!/usr/bin/env python3
"""Run generic non-hypercomplex Algebra-C fidelity baselines.

The first required generic capacity control is a small real-valued GRU.  The
`gru_wide` option doubles the hidden width as an under-control warning check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from abide_campaign_lib import (
    SubjectRecord,
    grouped_leave_one_site_out,
    limit_subjects,
    load_manifest,
    seed_schedule,
)


SCHEMA = "neurodyn.algebra_c.external_baselines.v1"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("torch is required for Algebra-C external baselines; use --dry-run for config checks") from exc
    return torch, nn


def read_targets(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["subject_id"]: float(row["target_scalar"]) for row in csv.DictReader(handle, delimiter="\t")}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[indexed[k][0]] = rank
        i = j
    return out


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ma = mean(a)
    mb = mean(b)
    va = sum((x - ma) * (x - ma) for x in a)
    vb = sum((y - mb) * (y - mb) for y in b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb)


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def r2_score(y: list[float], pred: list[float]) -> float:
    if not y:
        return 0.0
    mu = mean(y)
    ss_tot = sum((value - mu) * (value - mu) for value in y)
    if ss_tot <= 0.0:
        return 0.0
    ss_res = sum((value - estimate) * (value - estimate) for value, estimate in zip(y, pred, strict=True))
    return 1.0 - ss_res / ss_tot


def sign_auc(y: list[float], scores: list[float]) -> float | None:
    pos = [score for value, score in zip(y, scores, strict=True) if value > 0.0]
    neg = [score for value, score in zip(y, scores, strict=True) if value < 0.0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total if total else None


def calibration_slope(y: list[float], pred: list[float]) -> float:
    mp = mean(pred)
    my = mean(y)
    denom = sum((p - mp) * (p - mp) for p in pred)
    if denom <= 0.0:
        return 0.0
    return sum((p - mp) * (value - my) for value, p in zip(y, pred, strict=True)) / denom


def _site_seed(site: str, seed: int) -> int:
    h = 5381
    for ch in site:
        h = (h * 33 + ord(ch)) % 2_147_483_647
    return (seed * 1315423911 + h) % 2_147_483_647


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("requested --device cuda but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _records_to_tensor(torch, records: list[SubjectRecord], targets: dict[str, float], indices: list[int]):
    x = torch.tensor([records[idx].sequence for idx in indices], dtype=torch.float32)
    y = torch.tensor([targets[records[idx].subject_id] for idx in indices], dtype=torch.float32)
    return x, y


def _standardize(train_x, test_x):
    mu = train_x.mean(dim=0, keepdim=True)
    sd = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mu) / sd, (test_x - mu) / sd


def _iter_batches(torch, x, y, batch_size: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=generator)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def _build_model(torch, nn, model_name: str, seq_len: int, input_dim: int, hidden_dim: int):
    actual_hidden_dim = hidden_dim * 2 if model_name == "gru_wide" else hidden_dim

    class GRURegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(input_dim, actual_hidden_dim, batch_first=True)
            self.head = nn.Linear(actual_hidden_dim, 1)

        def forward(self, batch):
            _, hidden = self.rnn(batch)
            return self.head(hidden[-1]).squeeze(-1)

    class TransformerRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            d_model = actual_hidden_dim
            self.proj = nn.Linear(input_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=max(64, hidden_dim * 2),
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d_model, 1)

        def forward(self, batch):
            encoded = self.encoder(self.proj(batch) + self.pos)
            return self.head(encoded.mean(dim=1)).squeeze(-1)

    model_map = {
        "gru": GRURegressor,
        "gru_wide": GRURegressor,
        "transformer": TransformerRegressor,
    }
    if model_name not in model_map:
        raise ValueError(f"unsupported model: {model_name}")
    return model_map[model_name]()


def parameter_count(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _train_model(torch, nn, model, x_train, y_train, epochs: int, batch_size: int, lr: float, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    loss_fn = nn.MSELoss()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in _iter_batches(torch, x_train, y_train, batch_size, epoch + 17):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def run_one_model(torch, nn, records, targets, model_name: str, seed: int, epochs: int, batch_size: int, lr: float, hidden_dim: int, device):
    rows: list[dict[str, Any]] = []
    param_count = None
    for holdout_site in grouped_leave_one_site_out(records):
        train_indices = [idx for idx, record in enumerate(records) if record.site != holdout_site]
        test_indices = [idx for idx, record in enumerate(records) if record.site == holdout_site]
        x_train, y_train = _records_to_tensor(torch, records, targets, train_indices)
        x_test, y_test = _records_to_tensor(torch, records, targets, test_indices)
        x_train, x_test = _standardize(x_train, x_test)
        _set_seed(torch, _site_seed(holdout_site, seed))
        model = _build_model(torch, nn, model_name, x_train.shape[1], x_train.shape[2], hidden_dim).to(device)
        param_count = parameter_count(model)
        _train_model(torch, nn, model, x_train, y_train, epochs, batch_size, lr, device)
        model.eval()
        with torch.no_grad():
            pred = model(x_test.to(device)).detach().cpu().tolist()
        for idx, target, estimate in zip(test_indices, y_test.tolist(), pred, strict=True):
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "site": holdout_site,
                    "subject_id": records[idx].subject_id,
                    "target_scalar": float(target),
                    "prediction": float(estimate),
                    "parameter_count": param_count,
                }
            )
    return rows


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), int(row["seed"])), []).append(row)
    for (model, seed), chunk in sorted(groups.items()):
        y = [float(row["target_scalar"]) for row in chunk]
        pred = [float(row["prediction"]) for row in chunk]
        auc = sign_auc(y, pred)
        per_seed.append(
            {
                "model": model,
                "seed": seed,
                "site_count": len({row["site"] for row in chunk}),
                "subject_count": len(chunk),
                "parameter_count": int(chunk[0]["parameter_count"]),
                "spearman": spearman(y, pred),
                "r2": r2_score(y, pred),
                "sign_auc": "" if auc is None else auc,
                "calibration_slope": calibration_slope(y, pred),
            }
        )
    overall: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in per_seed:
        by_model.setdefault(str(row["model"]), []).append(row)
    for model, chunk in sorted(by_model.items()):
        overall_row: dict[str, Any] = {
            "model": model,
            "seed_count": len(chunk),
            "parameter_count": int(chunk[0]["parameter_count"]),
            "subject_count_mean": mean([float(row["subject_count"]) for row in chunk]),
        }
        for key in ("spearman", "r2", "calibration_slope"):
            vals = [float(row[key]) for row in chunk]
            overall_row[f"{key}_mean"] = mean(vals)
            overall_row[f"{key}_std"] = pstdev(vals)
        auc_vals = [float(row["sign_auc"]) for row in chunk if row["sign_auc"] != ""]
        overall_row["sign_auc_mean"] = "" if not auc_vals else mean(auc_vals)
        overall_row["sign_auc_std"] = "" if not auc_vals else pstdev(auc_vals)
        overall.append(overall_row)
    return overall, per_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", default="gru")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-subjects", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta, records = load_manifest(args.manifest)
    records = limit_subjects(records, args.limit_subjects or None)
    targets = read_targets(args.targets)
    missing = [record.subject_id for record in records if record.subject_id not in targets]
    if missing:
        raise SystemExit(f"targets missing {len(missing)} manifest subjects; first={missing[0]}")
    model_list = [item.strip() for item in args.models.split(",") if item.strip()]
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": SCHEMA,
        "manifest": asdict(meta),
        "targets": str(args.targets),
        "run_config": {
            "models": model_list,
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "device": args.device,
            "limit_subjects": args.limit_subjects,
        },
        "subjects_after_filter": len(records),
        "sites_after_filter": len(grouped_leave_one_site_out(records)),
    }
    (out / "baseline_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    torch, nn = _require_torch()
    device = _pick_device(torch, args.device)
    all_rows: list[dict[str, Any]] = []
    seed_list = seed_schedule(args.seeds)
    total = len(model_list) * len(seed_list)
    run_idx = 0
    for model_name in model_list:
        for seed in seed_list:
            run_idx += 1
            started = time.perf_counter()
            print(f"[algebra-c-external] start model={model_name} seed={seed} run={run_idx}/{total}", flush=True)
            rows = run_one_model(torch, nn, records, targets, model_name, seed, args.epochs, args.batch_size, args.lr, args.hidden_dim, device)
            all_rows.extend(rows)
            print(f"[algebra-c-external] done model={model_name} seed={seed} elapsed_s={time.perf_counter() - started:.2f}", flush=True)
    overall, per_seed = summarize_prediction_rows(all_rows)
    write_tsv(out / "prediction_rows.tsv", all_rows)
    write_tsv(out / "per_seed_metrics.tsv", per_seed)
    write_tsv(out / "overall_metrics.tsv", overall)
    payload = {**config, "overall": overall}
    (out / "overall_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
