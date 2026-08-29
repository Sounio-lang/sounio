#!/usr/bin/env python3
"""Evaluate top-k prototype orientation decisions from hidden pair deltas."""

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


SCHEMA = "neurodyn.orientation_topk_prototype.v1"


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
        raise SystemExit(f"subject id lacks pair suffix: {subject_id}")
    return subject_id.split("__", 1)[0]


def orientation(subject_id: str) -> str:
    if subject_id.endswith("__positive"):
        return "positive"
    if subject_id.endswith("__negative"):
        return "negative"
    raise SystemExit(f"subject id lacks orientation suffix: {subject_id}")


def dot_masked(a: list[float], b: list[float], dims: list[int]) -> float:
    return sum(a[d] * b[d] for d in dims)


def norm_masked(a: list[float], dims: list[int]) -> float:
    return math.sqrt(sum(a[d] * a[d] for d in dims))


def cosine_masked(a: list[float], b: list[float], dims: list[int]) -> float:
    na = norm_masked(a, dims)
    nb = norm_masked(b, dims)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot_masked(a, b, dims) / (na * nb)


def mean_vec(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def parse_k_values(text: str) -> list[int]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--k-values", default="1,2,4,8,16")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def subject_meta(reference_manifest: Path) -> dict[int, dict[str, str]]:
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


def collect_deltas(run: Path, condition: str, meta: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    raw = run / "brain_ossm_abide.raw.txt"
    state_rows = parse_state_rows(raw)
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_rows:
        m = meta.get(int(row["subject_index"]))
        if m is None:
            continue
        grouped[(row["model"], int(row["seed"]))][f"{m['pair_id']}::{m['orientation']}"] = {**row, **m}

    out: list[dict[str, Any]] = []
    for (model, seed), chunk in sorted(grouped.items()):
        for pid in sorted({v["pair_id"] for v in chunk.values()}):
            pos = chunk.get(f"{pid}::positive")
            neg = chunk.get(f"{pid}::negative")
            if pos is None or neg is None:
                continue
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": pos["site"],
                    "delta": [a - b for a, b in zip(pos["h"], neg["h"], strict=True)],
                }
            )
    return out


def evaluate(rows: list[dict[str, Any]], k_values: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    for (condition, model, seed), chunk in sorted(groups.items()):
        sites = sorted({row["site"] for row in chunk})
        for k in k_values:
            correct = 0
            total = 0
            scores: list[float] = []
            for site in sites:
                train = [row["delta"] for row in chunk if row["site"] != site]
                test = [row for row in chunk if row["site"] == site]
                proto = mean_vec(train)
                if not proto:
                    continue
                dims = sorted(range(len(proto)), key=lambda d: abs(proto[d]), reverse=True)[: min(k, len(proto))]
                for row in test:
                    score = cosine_masked(row["delta"], proto, dims)
                    scores.append(score)
                    if score > 0.0:
                        correct += 1
                    total += 1
                    detail.append(
                        {
                            "condition": condition,
                            "model": model,
                            "seed": seed,
                            "k": k,
                            "holdout_site": site,
                            "pair_id": row["pair_id"],
                            "score": score,
                            "correct": 1 if score > 0.0 else 0,
                            "dims": ",".join(str(d) for d in dims),
                        }
                    )
            seed_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "k": k,
                    "pair_count": total,
                    "accuracy": 100.0 * correct / total if total else 0.0,
                    "score_mean": mean(scores),
                    "score_std": pstdev(scores),
                }
            )
    return seed_rows, detail


def model_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["model"], int(row["k"]))].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model, k), rows in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "model": model,
                "k": k,
                "seed_count": len(rows),
                "accuracy_mean": mean([float(r["accuracy"]) for r in rows]),
                "accuracy_std": pstdev([float(r["accuracy"]) for r in rows]),
                "score_mean": mean([float(r["score_mean"]) for r in rows]),
                "score_std": pstdev([float(r["score_mean"]) for r in rows]),
            }
        )
    return out


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary}):
        for k in sorted({int(row["k"]) for row in summary}):
            true = [row for row in summary if row["condition"] == "true" and row["model"] == model and int(row["k"]) == k]
            nulls = [row for row in summary if row["condition"].startswith("null") and row["model"] == model and int(row["k"]) == k]
            if not true or not nulls:
                continue
            t = true[0]
            for metric in ("accuracy_mean", "score_mean"):
                tv = float(t[metric])
                vals = [float(row[metric]) for row in nulls]
                ge = sum(1 for v in vals if v >= tv)
                out.append(
                    {
                        "model": model,
                        "k": k,
                        "metric": metric,
                        "true": tv,
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
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "orientation_topk_prototype_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")
    k_values = parse_k_values(args.k_values)
    meta = subject_meta(args.reference_manifest)
    all_rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        all_rows.extend(collect_deltas(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})
    seed_rows, detail = evaluate(all_rows, k_values)
    summary = model_summary(seed_rows)
    envelope = null_envelope(summary)

    write_tsv(out / "orientation_topk_prototype_summary.tsv", summary)
    write_tsv(out / "orientation_topk_prototype_seed_detail.tsv", seed_rows)
    write_tsv(out / "orientation_topk_prototype_pair_detail.tsv", detail)
    write_tsv(out / "orientation_topk_prototype_null_envelope.tsv", envelope)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic top-k prototype orientation diagnostic only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "k_values": k_values,
        "runs": runs,
        "model_summary": summary,
        "null_envelope": envelope,
    }
    (out / "orientation_topk_prototype.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# Orientation Top-K Prototype", "", payload["claim_boundary"], ""]
    for row in envelope:
        md.append(
            f"- `{row['model']}` k=`{row['k']}` `{row['metric']}`: true `{float(row['true']):.6f}`, "
            f"null max `{float(row['null_max']):.6f}`, null_ge `{row['null_ge_true']}/{row['null_count']}`, "
            f"p+1 `{float(row['plus_one_p_ge_true']):.6f}`"
        )
    md.append("")
    (out / "orientation_topk_prototype.md").write_text("\n".join(md), encoding="utf-8")
    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
