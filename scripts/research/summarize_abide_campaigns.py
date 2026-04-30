#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROFILE_NAMES = {
    "0": "o_default",
    "1": "o_g2_transfer",
    "2": "o_assoc_stable",
    "3": "o_alg_v1",
    "4": "o_alg_v1_mandel",
}

PROJ_NAMES = {
    "0": "identity",
    "1": "learned_linear",
    "2": "residual_linear",
    "3": "mandelbrot_d2_residual",
    "4": "mandelbrot_d2_hybrid",
}


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def load_run_config(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open() as fh:
      for raw in fh:
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v
    return out


def summarize_run(run_dir: Path) -> dict[str, str]:
    leaderboard = load_tsv(run_dir / "results" / "campaign" / "leaderboard.tsv")
    config = load_run_config(run_dir / "abide_run_config.tsv")
    sounio = {row["model"]: row for row in leaderboard if row["source"] == "sounio"}
    externals = [row for row in leaderboard if row["source"] == "external"]
    best_external = max(externals, key=lambda row: float(row["balanced_accuracy_pct_mean"]))
    o_bal = float(sounio["O-SSM"]["balanced_accuracy_pct_mean"])
    h_bal = float(sounio["H-SSM"]["balanced_accuracy_pct_mean"])
    return {
        "run_id": run_dir.name,
        "train_fraction": config.get("train_fraction", ""),
        "oct_profile": PROFILE_NAMES.get(config.get("oct_profile_id", ""), config.get("oct_profile_id", "")),
        "oct_input_proj": PROJ_NAMES.get(config.get("oct_input_proj_mode", ""), config.get("oct_input_proj_mode", "")),
        "h_input_proj": PROJ_NAMES.get(config.get("h_input_proj_mode", ""), config.get("h_input_proj_mode", "")),
        "subjects": sounio["O-SSM"]["subjects"],
        "seed_count": sounio["O-SSM"]["seed_count"],
        "o_bal": f"{o_bal:.6f}",
        "h_bal": f"{h_bal:.6f}",
        "o_minus_h": f"{(o_bal - h_bal):.6f}",
        "best_external": best_external["model"],
        "best_external_bal": best_external["balanced_accuracy_pct_mean"],
        "o_minus_best_external": f"{(o_bal - float(best_external['balanced_accuracy_pct_mean'])):.6f}",
    }


def load_champion(summary_path: Path) -> dict[str, str]:
    rows = load_tsv(summary_path)
    best = max(rows, key=lambda row: float(row["sounio_o_bal"]))
    o_bal = float(best["sounio_o_bal"])
    h_bal = float(best["sounio_h_bal"])
    best_external_bal = float(best["best_external_bal"])
    return {
        "run_id": best["run_id"],
        "train_fraction": "0.25",
        "oct_profile": "g2preset_assocreg",
        "oct_input_proj": "identity",
        "h_input_proj": "identity",
        "subjects": "1023",
        "seed_count": "20",
        "o_bal": f"{o_bal:.6f}",
        "h_bal": f"{h_bal:.6f}",
        "o_minus_h": f"{(o_bal - h_bal):.6f}",
        "best_external": best["best_external"],
        "best_external_bal": f"{best_external_bal:.6f}",
        "o_minus_best_external": f"{(o_bal - best_external_bal):.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", default=[], help="Fetched run directory")
    parser.add_argument("--champion-summary", help="Path to g2_transfer_summary.tsv")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    if args.champion_summary:
        rows.append(load_champion(Path(args.champion_summary)))
    for run_dir in args.run_dir:
        rows.append(summarize_run(Path(run_dir)))

    fieldnames = [
        "run_id",
        "train_fraction",
        "oct_profile",
        "oct_input_proj",
        "h_input_proj",
        "subjects",
        "seed_count",
        "o_bal",
        "h_bal",
        "o_minus_h",
        "best_external",
        "best_external_bal",
        "o_minus_best_external",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
