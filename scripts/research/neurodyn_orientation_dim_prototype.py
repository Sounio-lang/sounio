#!/usr/bin/env python3
"""Evaluate associator-dimension conditioned prototype orientation readouts."""

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
from neurodyn_orientation_topk_prototype import (
    cosine_masked,
    mean_vec,
    orientation,
    pair_id,
    parse_k_values,
    read_manifest,
)


SCHEMA = "neurodyn.orientation_dim_prototype.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def pstdev(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-manifest", required=True, type=Path)
    ap.add_argument("--associator-triples", required=True, type=Path)
    ap.add_argument("--run", action="append", required=True, type=Path)
    ap.add_argument("--condition", action="append", required=True)
    ap.add_argument("--k-values", default="1,2,4,8,16")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_triple_meta(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = f"assoc_pair_{int(row['pair_id']):04d}"
        out[key] = row
    return out


def subject_meta(reference_manifest: Path, triples: Path) -> dict[int, dict[str, str]]:
    rows = read_manifest(reference_manifest)
    triple_meta = read_triple_meta(triples)
    out: dict[int, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        pid = pair_id(row["subject_id"])
        tm = triple_meta.get(pid)
        if tm is None:
            raise SystemExit(f"missing associator triple metadata for {pid}")
        out[idx] = {
            "subject_id": row["subject_id"],
            "pair_id": pid,
            "orientation": orientation(row["subject_id"]),
            "site": row["site"],
            "assoc_dim": tm["assoc_dim"],
            "assoc_sign": "1" if float(tm["positive_assoc_value"]) > 0.0 else "-1",
            "triple": tm["triple"],
        }
    return out


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
            raw_delta = [a - b for a, b in zip(pos["h"], neg["h"], strict=True)]
            assoc_sign = float(pos["assoc_sign"])
            out.append(
                {
                    "condition": condition,
                    "model": model,
                    "seed": seed,
                    "pair_id": pid,
                    "site": pos["site"],
                    "assoc_dim": pos["assoc_dim"],
                    "assoc_sign": pos["assoc_sign"],
                    "triple": pos["triple"],
                    "raw_delta": raw_delta,
                    "aligned_delta": [assoc_sign * value for value in raw_delta],
                }
            )
    return out


def score_rows(
    *,
    chunk: list[dict[str, Any]],
    k: int,
    mode: str,
    basis: str,
    condition: str,
    model: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct = 0
    total = 0
    scores: list[float] = []
    detail: list[dict[str, Any]] = []
    dims_all = sorted({row["assoc_dim"] for row in chunk})

    for row in chunk:
        delta_key = "aligned_delta" if basis == "assoc_sign_aligned" else "raw_delta"
        if mode == "global_pair_holdout":
            train = [r[delta_key] for r in chunk if r["pair_id"] != row["pair_id"]]
            family = "global"
        elif mode == "within_dim_pair_holdout":
            train = [
                r[delta_key]
                for r in chunk
                if r["assoc_dim"] == row["assoc_dim"] and r["pair_id"] != row["pair_id"]
            ]
            family = f"dim_{row['assoc_dim']}"
        elif mode == "leave_dim_global":
            train = [r[delta_key] for r in chunk if r["assoc_dim"] != row["assoc_dim"]]
            family = "not_" + f"dim_{row['assoc_dim']}"
        else:
            raise ValueError(mode)
        proto = mean_vec(train)
        if not proto:
            continue
        top_dims = sorted(range(len(proto)), key=lambda d: abs(proto[d]), reverse=True)[: min(k, len(proto))]
        score = cosine_masked(row[delta_key], proto, top_dims)
        scores.append(score)
        if score > 0.0:
            correct += 1
        total += 1
        detail.append(
            {
                "condition": condition,
                "model": model,
                "seed": seed,
                "mode": mode,
                "basis": basis,
                "k": k,
                "pair_id": row["pair_id"],
                "assoc_dim": row["assoc_dim"],
                "assoc_sign": row["assoc_sign"],
                "prototype_family": family,
                "score": score,
                "correct": 1 if score > 0.0 else 0,
                "dims": ",".join(str(d) for d in top_dims),
            }
        )

    return (
        {
            "condition": condition,
            "model": model,
            "seed": seed,
            "mode": mode,
            "basis": basis,
            "k": k,
            "pair_count": total,
            "assoc_dim_count": len(dims_all),
            "accuracy": 100.0 * correct / total if total else 0.0,
            "score_mean": mean(scores),
            "score_std": pstdev(scores),
        },
        detail,
    )


def evaluate(rows: list[dict[str, Any]], k_values: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["model"], int(row["seed"]))].append(row)

    modes = ["global_pair_holdout", "within_dim_pair_holdout", "leave_dim_global"]
    bases = ["raw_orientation", "assoc_sign_aligned"]
    for (condition, model, seed), chunk in sorted(groups.items()):
        for basis in bases:
            for mode in modes:
                for k in k_values:
                    seed_row, seed_detail = score_rows(
                        chunk=chunk,
                        k=k,
                        mode=mode,
                        basis=basis,
                        condition=condition,
                        model=model,
                        seed=seed,
                    )
                    seed_rows.append(seed_row)
                    detail.extend(seed_detail)
    return seed_rows, detail


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        groups[(row["condition"], row["model"], row["basis"], row["mode"], int(row["k"]))].append(row)
    out: list[dict[str, Any]] = []
    for (condition, model, basis, mode, k), rows in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "model": model,
                "basis": basis,
                "mode": mode,
                "k": k,
                "seed_count": len(rows),
                "pair_count_mean": mean([float(r["pair_count"]) for r in rows]),
                "accuracy_mean": mean([float(r["accuracy"]) for r in rows]),
                "accuracy_std": pstdev([float(r["accuracy"]) for r in rows]),
                "score_mean": mean([float(r["score_mean"]) for r in rows]),
                "score_std": pstdev([float(r["score_mean"]) for r in rows]),
            }
        )
    return out


def null_envelope(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    models = sorted({row["model"] for row in summary})
    bases = sorted({row["basis"] for row in summary})
    modes = sorted({row["mode"] for row in summary})
    ks = sorted({int(row["k"]) for row in summary})
    for model in models:
        for basis in bases:
            for mode in modes:
                for k in ks:
                    true = [
                        row
                        for row in summary
                        if row["condition"] == "true"
                        and row["model"] == model
                        and row["basis"] == basis
                        and row["mode"] == mode
                        and int(row["k"]) == k
                    ]
                    nulls = [
                        row
                        for row in summary
                        if row["condition"].startswith("null")
                        and row["model"] == model
                        and row["basis"] == basis
                        and row["mode"] == mode
                        and int(row["k"]) == k
                    ]
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
                                "basis": basis,
                                "mode": mode,
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if len(args.run) != len(args.condition):
        raise SystemExit("--run and --condition counts differ")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "orientation_dim_prototype_summary.tsv").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")

    meta = subject_meta(args.reference_manifest, args.associator_triples)
    k_values = parse_k_values(args.k_values)
    all_rows: list[dict[str, Any]] = []
    runs = []
    for run, condition in zip(args.run, args.condition, strict=True):
        all_rows.extend(collect_deltas(run, condition, meta))
        runs.append({"condition": condition, "run": str(run), "raw_sha256": sha256_file(run / "brain_ossm_abide.raw.txt")})

    seed_rows, detail = evaluate(all_rows, k_values)
    summary = summarize(seed_rows)
    envelope = null_envelope(summary)

    write_tsv(out / "orientation_dim_prototype_seed_detail.tsv", seed_rows)
    write_tsv(out / "orientation_dim_prototype_pair_detail.tsv", detail)
    write_tsv(out / "orientation_dim_prototype_summary.tsv", summary)
    write_tsv(out / "orientation_dim_prototype_null_envelope.tsv", envelope)

    payload = {
        "schema": SCHEMA,
        "claim_boundary": "Synthetic associator-dimension prototype diagnostic only.",
        "reference_manifest": str(args.reference_manifest),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "k_values": k_values,
        "runs": runs,
        "summary": summary,
        "null_envelope": envelope,
    }
    (out / "orientation_dim_prototype.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = ["# Orientation Dimension Prototype", "", payload["claim_boundary"], ""]
    for row in envelope:
        md.append(
            f"- `{row['model']}` `{row['basis']}` `{row['mode']}` k=`{row['k']}` `{row['metric']}`: "
            f"true `{float(row['true']):.6f}`, null max `{float(row['null_max']):.6f}`, "
            f"null_ge `{row['null_ge_true']}/{row['null_count']}`, "
            f"p+1 `{float(row['plus_one_p_ge_true']):.6f}`"
        )
    md.append("")
    (out / "orientation_dim_prototype.md").write_text("\n".join(md), encoding="utf-8")

    sums = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256_file(path)}  {path.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
