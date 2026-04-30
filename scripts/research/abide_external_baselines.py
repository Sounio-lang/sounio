#!/usr/bin/env python3
"""Run external deep-sequence baselines for the Brain O-SSM ABIDE campaign."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from abide_campaign_lib import (
    ManifestMeta,
    SubjectRecord,
    grouped_leave_one_site_out,
    limit_subjects,
    load_manifest,
    select_sites,
    seed_schedule,
    summarize_prediction_rows,
    write_json,
    write_tsv,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via CLI
        raise SystemExit(
            "torch is required for abide_external_baselines.py. "
            "Install a torch wheel in the job environment or use --dry-run."
        ) from exc
    return torch, nn


def _site_seed(site: str, seed: int) -> int:
    h = 5381
    for ch in site:
        h = (h * 33 + ord(ch)) % 2_147_483_647
    return (seed * 1315423911 + h) % 2_147_483_647


def _split_train_fraction(
    records: list[SubjectRecord],
    train_indices: list[int],
    fraction: float,
    seed: int,
) -> list[int]:
    if fraction >= 0.999:
        return train_indices
    rng = random.Random(seed)
    pos = [idx for idx in train_indices if records[idx].label == 1]
    neg = [idx for idx in train_indices if records[idx].label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_keep = max(1, round(len(pos) * fraction)) if pos else 0
    neg_keep = max(1, round(len(neg) * fraction)) if neg else 0
    subset = pos[:pos_keep] + neg[:neg_keep]
    rng.shuffle(subset)
    return subset


def _records_to_tensor(torch, records: list[SubjectRecord], indices: list[int]):
    xs = [records[idx].sequence for idx in indices]
    ys = [records[idx].label for idx in indices]
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


def _standardize(train_x, test_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def _apply_corruption(torch, x, seed: int, drop_channel_frac: float, noise_std: float):
    if drop_channel_frac <= 0.0 and noise_std <= 0.0:
        return x
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    out = x.clone()
    if drop_channel_frac > 0.0:
        channel_count = out.shape[-1]
        drop_count = min(channel_count, max(1, int(round(channel_count * drop_channel_frac))))
        perm = torch.randperm(channel_count, generator=generator)
        mask = torch.ones(channel_count, dtype=out.dtype)
        mask[perm[:drop_count]] = 0.0
        out = out * mask.view(1, 1, channel_count)
    if noise_std > 0.0:
        noise = torch.randn(out.shape, generator=generator, dtype=out.dtype)
        out = out + noise * noise_std
    return out


def _pick_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("requested --device cuda but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_batches(torch, x, y, batch_size: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(x.shape[0], generator=generator)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]


def _train_model(torch, nn, model_name: str, x_train, y_train, epochs: int, batch_size: int, lr: float, device):
    seq_len = x_train.shape[1]
    input_dim = x_train.shape[2]

    class LSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_dim, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, batch):
            _, (hidden, _) = self.rnn(batch)
            return self.head(hidden[-1]).squeeze(-1)

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(input_dim, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, batch):
            _, hidden = self.rnn(batch)
            return self.head(hidden[-1]).squeeze(-1)

    class TransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            d_model = 32
            self.proj = nn.Linear(input_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=64,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(d_model, 1)

        def forward(self, batch):
            hidden = self.proj(batch) + self.pos
            encoded = self.encoder(hidden)
            pooled = encoded.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    class TCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            channels = 32
            self.net = nn.Sequential(
                nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(channels, 1)

        def forward(self, batch):
            hidden = self.net(batch.transpose(1, 2)).squeeze(-1)
            return self.head(hidden).squeeze(-1)

    model_map = {
        "lstm": LSTMClassifier,
        "gru": GRUClassifier,
        "transformer": TransformerClassifier,
        "tcn": TCNClassifier,
    }
    if model_name not in model_map:
        raise ValueError(f"unsupported model: {model_name}")

    model = model_map[model_name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in _iter_batches(torch, x_train, y_train, batch_size=batch_size, seed=epoch + 17):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model


def _run_one_model(
    torch,
    nn,
    records: list[SubjectRecord],
    meta: ManifestMeta,
    model_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device,
    train_fraction: float,
    drop_channel_frac: float,
    noise_std: float,
) -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    for holdout_site in grouped_leave_one_site_out(records):
        train_indices = [idx for idx, record in enumerate(records) if record.site != holdout_site]
        test_indices = [idx for idx, record in enumerate(records) if record.site == holdout_site]
        train_indices = _split_train_fraction(records, train_indices, train_fraction, _site_seed(holdout_site, seed))
        x_train, y_train = _records_to_tensor(torch, records, train_indices)
        x_test, y_test = _records_to_tensor(torch, records, test_indices)
        x_train, x_test = _standardize(x_train, x_test)
        corruption_seed = _site_seed(holdout_site, seed + 1000)
        x_train = _apply_corruption(torch, x_train, corruption_seed, drop_channel_frac, noise_std)
        x_test = _apply_corruption(torch, x_test, corruption_seed, drop_channel_frac, noise_std)

        _set_seed(torch, seed)
        model = _train_model(
            torch,
            nn,
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            logits = model(x_test.to(device))
            probs = torch.sigmoid(logits).detach().cpu().tolist()
        labels = y_test.tolist()
        for label, prob in zip(labels, probs, strict=True):
            prediction_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "site": holdout_site,
                    "label": int(label),
                    "prob": float(prob),
                    "pred": 1 if prob >= 0.5 else 0,
                    "assoc": None,
                }
            )
    return prediction_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ABIDE external deep-baseline suite")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="lstm,gru,transformer,tcn")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--drop-channel-frac", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--limit-subjects", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta, records = load_manifest(args.manifest)
    records = select_sites(records, args.max_sites or None)
    records = limit_subjects(records, args.limit_subjects or None)
    sites = grouped_leave_one_site_out(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "schema": "brain_ossm.abide.external.v1",
        "manifest": asdict(meta),
        "run_config": {
            "models": [item.strip() for item in args.models.split(",") if item.strip()],
            "seeds": args.seeds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "train_fraction": args.train_fraction,
            "drop_channel_frac": args.drop_channel_frac,
            "noise_std": args.noise_std,
            "max_sites": args.max_sites,
            "limit_subjects": args.limit_subjects,
        },
        "subjects_after_filter": len(records),
        "sites_after_filter": len(sites),
    }
    write_json(output_dir / "manifest_meta.json", config_payload)

    if args.dry_run:
        print(f"manifest={args.manifest}")
        print(f"subjects={len(records)} sites={len(sites)} seq_len={meta.seq_len} input_dim={meta.input_dim}")
        print(f"models={config_payload['run_config']['models']}")
        return 0

    torch, nn = _require_torch()
    device = _pick_device(torch, args.device)

    prediction_rows: list[dict[str, object]] = []
    seed_list = seed_schedule(args.seeds)
    model_list = config_payload["run_config"]["models"]
    total_runs = len(model_list) * len(seed_list)
    run_idx = 0
    for model_name in model_list:
        for seed in seed_list:
            run_idx += 1
            started = time.perf_counter()
            print(
                f"[external] start model={model_name} seed={seed} "
                f"run={run_idx}/{total_runs} sites={len(sites)} "
                f"subjects={len(records)} tf={args.train_fraction:.3f} "
                f"drop={args.drop_channel_frac:.3f} noise={args.noise_std:.3f}",
                flush=True,
            )
            run_rows = _run_one_model(
                torch=torch,
                nn=nn,
                records=records,
                meta=meta,
                model_name=model_name,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                train_fraction=args.train_fraction,
                drop_channel_frac=args.drop_channel_frac,
                noise_std=args.noise_std,
            )
            prediction_rows.extend(run_rows)
            seed_summary = summarize_prediction_rows(run_rows)["per_seed"]
            if seed_summary:
                metrics = seed_summary[0]
                print(
                    f"[external] done  model={model_name} seed={seed} "
                    f"bal={metrics['balanced_accuracy_pct']:.6f} "
                    f"acc={metrics['accuracy_pct']:.6f} "
                    f"auroc={metrics['auroc_pct']:.6f} "
                    f"elapsed_s={time.perf_counter() - started:.2f}",
                    flush=True,
                )

    summaries = summarize_prediction_rows(prediction_rows)
    payload = {
        "schema": "brain_ossm.abide.external.v1",
        "manifest": asdict(meta),
        "run_config": config_payload["run_config"],
        "overall": summaries["overall"],
    }
    write_json(output_dir / "overall_metrics.json", payload)
    write_tsv(
        output_dir / "overall_metrics.tsv",
        summaries["overall"],
        [
            "model",
            "seed_count",
            "site_count",
            "subjects",
            "accuracy_pct_mean",
            "accuracy_pct_std",
            "balanced_accuracy_pct_mean",
            "balanced_accuracy_pct_std",
            "macro_f1_pct_mean",
            "macro_f1_pct_std",
            "auroc_pct_mean",
            "auroc_pct_std",
            "brier_mean",
            "brier_std",
            "ece_pct_mean",
            "ece_pct_std",
            "assoc_mean_mean",
            "assoc_mean_std",
        ],
    )
    write_tsv(
        output_dir / "per_seed_metrics.tsv",
        summaries["per_seed"],
        [
            "model",
            "seed",
            "site_count",
            "subjects",
            "accuracy_pct",
            "balanced_accuracy_pct",
            "macro_f1_pct",
            "auroc_pct",
            "brier",
            "ece_pct",
            "assoc_mean",
        ],
    )
    write_tsv(
        output_dir / "per_site_metrics.tsv",
        summaries["per_site"],
        [
            "model",
            "site",
            "seed_count",
            "subjects_mean",
            "accuracy_pct_mean",
            "accuracy_pct_std",
            "balanced_accuracy_pct_mean",
            "balanced_accuracy_pct_std",
            "macro_f1_pct_mean",
            "macro_f1_pct_std",
            "auroc_pct_mean",
            "auroc_pct_std",
            "brier_mean",
            "brier_std",
            "ece_pct_mean",
            "ece_pct_std",
            "assoc_mean_mean",
            "assoc_mean_std",
        ],
    )
    write_tsv(
        output_dir / "prediction_rows.tsv",
        prediction_rows,
        ["model", "seed", "site", "label", "prob", "pred", "assoc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
