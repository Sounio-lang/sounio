#!/usr/bin/env python3
"""Audit leave-site stability of original positive-negative hidden deltas."""

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

from neurodyn_hidden_state_separability import parse_state_rows


SCHEMA = "neurodyn.orientation_delta_stability.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    if not lines:
        raise SystemExit(f"manifest has no rows: {path}")
    return list(csv.DictReader(lines, delimiter="\t"))


def pair_id(subject_id: str) -> str:
    if "__" not in subject_id:
        raise SystemExit(f"subject id lacks orientation suffix: {subject_id}")
    return subject_id.split("__", 1)[0]


def orientation(subject_id: str) -> str:
    if subject_id.endswith("__positive"):
        return "positive"
    if subject_id.endswith("__negative"):
        return "negative"
    raise SystemExit(f"subject id lacks fixed-target suffix: {subject_id}")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: list[float], b: list[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path, help="True fixed-orientation manifest.")
    ap.add_argument("--run", action="append", required=True, type=Path, help="Run directory with brain_ossm_abide.raw.txt.")
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def build_subject_meta(reference_manifest: Path) -> dict[int, dict[str, str]]:
    rows = read_manifest(reference_manifest)
    return {
        idx: {
            "subject_id": row["subject_id"],
            "pair_id": pair_id(row["subject_id"]),
            "orientation": orientation(row["subject_id"]),
            "site": row["site"],
        }
        for idx, row in enumerate(rows)
    }


def collect_pair_deltas(run: Path, condition: str, subject_meta: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    raw = run / "brain_ossm_abide.raw.txt"
    state_rows = parse_state_rows(raw)
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows:
        meta = subject_meta.get(int(row["subject_index"]))
        if meta is None:
            continue
        key = (row["model"], int(row["seed"]))
        grouped[key][f"{meta['pair_id']}::{meta['orientation']}"] = {**row, **meta}

    out: list[dict[str, Any]] = []
    for (model, seed), chunk in sorted(grouped.items()):
        pair_ids = sorted({value["pair_id"] for value in chunk.values()})
        for pid in pair_ids:
            pos = chunk.get(f"{pid}::positive")
            neg = chunk.get(f"{pid}::negative")
            if pos is None or neg is None:
                continue
            delta = [a - b for a, b in zip(pos["h"], neg["h"], strict=True)]
            out.append(
                {
                    "condition": condition,
                    "run": str(run),
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": pos["site"],
                    "delta": delta,
                    "delta_norm": norm(delta),
                }
            )
    return out


def summarize_condition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    for (condition, model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        correct = 0
        total = 0
        holdout_cosines: list[float] = []
        for site in sites:
            train = [row["delta"] for row in chunk if row["site"] != site]
            test = [row for row in chunk if row["site"] == site]
            direction = mean_vec(train)
            if not direction:
                continue
            for row in test:
                c = cosine(row["delta"], direction)
                holdout_cosines.append(c)
                if c > 0.0:
                    correct += 1
                total += 1
        mean_delta = mean_vec([row["delta"] for row in chunk])
        pairwise_cosines: list[float] = []
        for i, left in enumerate(chunk):
            for right in chunk[i + 1 :]:
                pairwise_cosines.append(cosine(left["delta"], right["delta"]))
        delta_norms = [float(row["delta_norm"]) for row in chunk]
        seed_summary = {
            "condition": condition,
            "model": model,
            "seed": seed,
            "pair_count": len(chunk),
            "delta_norm_mean": mean(delta_norms),
            "mean_delta_norm": norm(mean_delta),
            "mean_delta_norm_over_delta_norm": norm(mean_delta) / mean(delta_norms) if mean(delta_norms) > 0 else 0.0,
            "pairwise_cosine_mean": mean(pairwise_cosines),
            "pairwise_cosine_std": pstdev(pairwise_cosines),
            "leave_site_delta_accuracy": 100.0 * correct / total if total else 0.0,
            "leave_site_delta_cosine_mean": mean(holdout_cosines),
            "leave_site_delta_cosine_std": pstdev(holdout_cosines),
        }
        summary.append(seed_summary)
        for row in chunk:
            detail.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "pair_id": row["pair_id"],
                    "site": row["site"],
                    "delta_norm": row["delta_norm"],
                }
            )
    return summary, detail


def summarize_model(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["model"])].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model), rows in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "model": model,
                "seed_count": len(rows),
                "leave_site_delta_accuracy_mean": mean([float(r["leave_site_delta_accuracy"]) for r in rows]),
                "leave_site_delta_accuracy_std": pstdev([float(r["leave_site_delta_accuracy"]) for r in rows]),
                "leave_site_delta_cosine_mean": mean([float(r["leave_site_delta_cosine_mean"]) for r in rows]),
                "leave_site_delta_cosine_std": pstdev([float(r["leave_site_delta_cosine_mean"]) for r in rows]),
                "pairwise_cosine_mean": mean([float(r["pairwise_cosine_mean"]) for r in rows]),
                "mean_delta_norm_over_delta_norm_mean": mean([float(r["mean_delta_norm_over_delta_norm"]) for r in rows]),
                "delta_norm_mean": mean([float(r["delta_norm_mean"]) for r in rows]),
            }
        )
    return out


def null_envelope(model_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in model_summary}):
        true_rows = [row for row in model_summary if row["condition"] == "true" and row["model"] == model]
        null_rows = [row for row in model_summary if row["condition"].startswith("null") and row["model"] == model]
        if not true_rows or not null_rows:
            continue
        true = true_rows[0]
        for metric in ("leave_site_delta_accuracy_mean", "leave_site_delta_cosine_mean", "pairwise_cosine_mean", "mean_delta_norm_over_delta_norm_mean"):
            t = float(true[metric])
            vals = [float(row[metric]) for row in null_rows]
            ge = sum(1 for v in vals if v >= t)
            out.append(
                {
                    "model": model,
                    "metric": metric,
                    "true": t,
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "orientation_delta_stability_summary.tsv"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {args.output_dir}")

    subject_meta = build_subject_meta(args.reference_manifest)
    all_pair_rows: list[dict[str, Any]] = []
    run_meta = []
    for run, condition in zip(args.run, args.condition, strict=True):
        all_pair_rows.extend(collect_pair_deltas(run, condition, subject_meta))
        raw = run / "brain_ossm_abide.raw.txt"
        run_meta.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(raw)})
    seed_rows, detail_rows = summarize_condition(all_pair_rows)
    model_summary = summarize_model(seed_rows)
    envelope = null_envelope(model_summary)

    write_tsv(summary_path, model_summary)
    write_tsv(args.output_dir / "orientation_delta_stability_seed_detail.tsv", seed_rows)
    write_tsv(args.output_dir / "orientation_delta_stability_pair_detail.tsv", detail_rows)
    write_tsv(args.output_dir / "orientation_delta_stability_null_envelope.tsv", envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic original-orientation hidden-delta stability diagnostic only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "runs": run_meta,
        "model_summary": model_summary,
        "null_envelope": envelope,
    }
    (args.output_dir / "orientation_delta_stability.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Orientation Delta Stability", "", payload["claim_boundary"], ""]
    for row in model_summary:
        md.append(
            f"- `{row['condition']}` `{row['model']}` leave-site delta accuracy: "
            f"`{float(row['leave_site_delta_accuracy_mean']):.6f}`; cosine: "
            f"`{float(row['leave_site_delta_cosine_mean']):.6f}`"
        )
    md.append("")
    for row in envelope:
        md.append(
            f"- envelope `{row['model']}` `{row['metric']}`: true `{float(row['true']):.6f}`, "
            f"null max `{float(row['null_max']):.6f}`, null_ge `{row['null_ge_true']}/{row['null_count']}`, "
            f"p+1 `{float(row['plus_one_p_ge_true']):.6f}`"
        )
    md.append("")
    (args.output_dir / "orientation_delta_stability.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
