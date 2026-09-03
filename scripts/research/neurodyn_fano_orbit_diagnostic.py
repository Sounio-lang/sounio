#!/usr/bin/env python3
"""Diagnose Fano-line orbit geometry from Brain O-SSM STATE_TRACE output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PAIR_RE = re.compile(r"synthetic_pair_(\d+)__(oriented|swapped)$")
FANO_SITE_RE = re.compile(r"fano_line_\d+_(\d+)-(\d+)-(\d+)$")


def read_manifest(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter="\t")
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        for subject_idx, row in enumerate(reader):
            sid = row[idx["subject_id"]]
            m = PAIR_RE.match(sid)
            if not m:
                raise SystemExit(f"manifest subject_id does not match synthetic pair pattern: {sid}")
            pair_idx = int(m.group(1))
            orientation = m.group(2)
            label_raw = row[idx["label"]]
            try:
                label = int(label_raw)
            except ValueError:
                if orientation == "oriented":
                    label = 1
                elif orientation == "swapped":
                    label = 0
                else:
                    raise
            rows.append(
                {
                    "subject_idx": subject_idx,
                    "subject_id": sid,
                    "label": label,
                    "label_raw": label_raw,
                    "site": row[idx["site"]],
                    "pair_id": f"synthetic_pair_{pair_idx:04d}",
                    "pair_idx": pair_idx,
                    "orientation": orientation,
                }
            )
    site_count = len({r["site"] for r in rows})
    if site_count <= 0:
        raise SystemExit("manifest has no sites")
    for r in rows:
        r["line_idx"] = r["pair_idx"] % site_count
        r["orbit_idx"] = r["pair_idx"] // site_count
    return rows, site_count


def collect_parts(lines: list[str], i: int, expected: int) -> tuple[list[str], int]:
    parts = [part for part in lines[i].split("\t") if part]
    j = i + 1
    while len(parts) < expected and j < len(lines) and lines[j].startswith("\t"):
        parts.extend(part for part in lines[j].split("\t") if part)
        j += 1
    return parts, j


def parse_raw(path: Path) -> tuple[dict[tuple[str, int, int], list[float]], dict[tuple[str, int, int], int]]:
    states: dict[tuple[str, int, int], list[float]] = {}
    preds: dict[tuple[str, int, int], int] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("STATE_TRACE\t"):
            parts, next_i = collect_parts(lines, i, 22)
            if len(parts) != 22:
                raise SystemExit(f"malformed STATE_TRACE near line {i + 1}: expected 22 fields, got {len(parts)}")
            _, model, seed_s, subject_s, _site, _label, *raw_values = parts
            seed = int(seed_s)
            subject_idx = int(subject_s)
            values = [int(value) / 1_000_000.0 for value in raw_values]
            states[(model, seed, subject_idx)] = values
            i = next_i
            continue
        if line.startswith("PRED\t"):
            parts, next_i = collect_parts(lines, i, 9)
            if len(parts) == 8:
                _, model, seed_s, _site, _label, _prob, pred_s, _assoc = parts
                subject_idx = -1
            elif len(parts) == 9:
                _, model, seed_s, subject_s, _site, _label, _prob, pred_s, _assoc = parts
                subject_idx = int(subject_s)
            else:
                raise SystemExit(f"malformed PRED near line {i + 1}: expected 8 or 9 fields, got {len(parts)}")
            seed = int(seed_s)
            pred = int(pred_s)
            if subject_idx >= 0:
                preds[(model, seed, subject_idx)] = pred
            i = next_i
            continue
        i += 1
    return states, preds


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def add_into(dst: list[float], src: list[float]) -> None:
    for i, v in enumerate(src):
        dst[i] += v


def scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def cosine(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom <= 1e-12:
        return 0.0
    return dot(a, b) / denom


def balanced_accuracy(items: list[tuple[int, int]]) -> float:
    tp = tn = fp = fn = 0
    for label, pred in items:
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return 50.0 * (tpr + tnr)


def build_pair_rows(
    manifest_rows: list[dict[str, Any]],
    states: dict[tuple[str, int, int], list[float]],
    preds: dict[tuple[str, int, int], int],
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in manifest_rows:
        by_pair[r["pair_id"]].append(r)

    out: list[dict[str, Any]] = []
    model_seeds = sorted({(m, s) for (m, s, _) in states})
    for model, seed in model_seeds:
        for pair_id, members in sorted(by_pair.items()):
            pos = next((r for r in members if r["label"] == 1), None)
            neg = next((r for r in members if r["label"] == 0), None)
            if pos is None or neg is None:
                continue
            pos_state = states.get((model, seed, pos["subject_idx"]))
            neg_state = states.get((model, seed, neg["subject_idx"]))
            if pos_state is None or neg_state is None:
                continue
            pos_pred = preds.get((model, seed, pos["subject_idx"]))
            neg_pred = preds.get((model, seed, neg["subject_idx"]))
            pred_items = []
            if pos_pred is not None:
                pred_items.append((1, pos_pred))
            if neg_pred is not None:
                pred_items.append((0, neg_pred))
            delta = sub(pos_state, neg_state)
            out.append(
                {
                    "model": model,
                    "seed": seed,
                    "pair_id": pair_id,
                    "pair_idx": pos["pair_idx"],
                    "orbit_idx": pos["orbit_idx"],
                    "line_idx": pos["line_idx"],
                    "site": pos["site"],
                    "delta": delta,
                    "delta_norm": norm(delta),
                    "pair_balanced_accuracy": balanced_accuracy(pred_items) if pred_items else None,
                }
            )
    return out


def enrich_orbit_alignment(pair_rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pair_rows:
        groups[(r["model"], r["seed"], r["orbit_idx"])].append(r)
    for group_rows in groups.values():
        if not group_rows:
            continue
        center = [0.0] * len(group_rows[0]["delta"])
        for r in group_rows:
            add_into(center, r["delta"])
        center = scale(center, 1.0 / len(group_rows))
        for r in group_rows:
            leave_center = [0.0] * len(r["delta"])
            leave_n = 0
            for other in group_rows:
                if other is r:
                    continue
                add_into(leave_center, other["delta"])
                leave_n += 1
            if leave_n:
                leave_center = scale(leave_center, 1.0 / leave_n)
            else:
                leave_center = center
            r["orbit_center_cosine"] = cosine(r["delta"], center)
            r["orbit_leave_one_cosine"] = cosine(r["delta"], leave_center)
            r["orbit_center_distance"] = norm(sub(r["delta"], center))


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        rec = {k: v for k, v in zip(keys, key)}
        for field in ["delta_norm", "orbit_center_cosine", "orbit_leave_one_cosine", "orbit_center_distance"]:
            vals = [float(r[field]) for r in group if field in r]
            rec[f"{field}_mean"] = mean(vals) if vals else 0.0
            rec[f"{field}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        ba_vals = [float(r["pair_balanced_accuracy"]) for r in group if r.get("pair_balanced_accuracy") is not None]
        rec["pair_balanced_accuracy_mean"] = mean(ba_vals) if ba_vals else 0.0
        rec["row_count"] = len(group)
        out.append(rec)
    return out


def line_transfer_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model_seed_line: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pair_rows:
        by_model_seed_line[(r["model"], r["seed"], r["line_idx"])].append(r)

    centers: dict[tuple[str, int, int], list[float]] = {}
    for key, rows in by_model_seed_line.items():
        if not rows:
            continue
        center = [0.0] * len(rows[0]["delta"])
        for r in rows:
            add_into(center, r["delta"])
        centers[key] = scale(center, 1.0 / len(rows))

    out: list[dict[str, Any]] = []
    model_seeds = sorted({(r["model"], r["seed"]) for r in pair_rows})
    line_indices = sorted({int(r["line_idx"]) for r in pair_rows})
    site_by_line = {int(r["line_idx"]): r["site"] for r in pair_rows}
    for model, seed in model_seeds:
        for source_line in line_indices:
            center = centers.get((model, seed, source_line))
            if center is None:
                continue
            center_norm = norm(center)
            for target_line in line_indices:
                target_rows = by_model_seed_line.get((model, seed, target_line), [])
                if not target_rows:
                    continue
                correct = 0
                total = 0
                score_values: list[float] = []
                cos_values: list[float] = []
                for r in target_rows:
                    score = dot(r["delta"], center)
                    score_values.append(score)
                    cos_values.append(cosine(r["delta"], center))
                    if score >= 0.0:
                        correct += 1
                    total += 1
                out.append(
                    {
                        "model": model,
                        "seed": seed,
                        "source_line_idx": source_line,
                        "source_site": site_by_line.get(source_line, ""),
                        "target_line_idx": target_line,
                        "target_site": site_by_line.get(target_line, ""),
                        "source_center_norm": center_norm,
                        "transfer_accuracy": 100.0 * correct / total if total else 0.0,
                        "score_mean": mean(score_values),
                        "cosine_mean": mean(cos_values),
                        "pair_count": total,
                    }
                )
    return out


def aggregate_line_transfer(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[
            (
                r["model"],
                int(r["source_line_idx"]),
                str(r["source_site"]),
                int(r["target_line_idx"]),
                str(r["target_site"]),
            )
        ].append(r)
    out: list[dict[str, Any]] = []
    for (model, source_line, source_site, target_line, target_site), group in sorted(groups.items()):
        acc = [float(r["transfer_accuracy"]) for r in group]
        score = [float(r["score_mean"]) for r in group]
        cos = [float(r["cosine_mean"]) for r in group]
        diag = 1 if source_line == target_line else 0
        out.append(
            {
                "condition": condition,
                "model": model,
                "source_line_idx": source_line,
                "source_site": source_site,
                "target_line_idx": target_line,
                "target_site": target_site,
                "is_diagonal": diag,
                "transfer_accuracy_mean": mean(acc),
                "transfer_accuracy_std": pstdev(acc) if len(acc) > 1 else 0.0,
                "score_mean": mean(score),
                "cosine_mean": mean(cos),
                "seed_count": len(group),
            }
        )
    return out


def summarize_line_transfer(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[str(r["model"])].append(r)
    out: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        diag = [float(r["transfer_accuracy_mean"]) for r in model_rows if int(r["is_diagonal"]) == 1]
        off = [float(r["transfer_accuracy_mean"]) for r in model_rows if int(r["is_diagonal"]) == 0]
        anti = [r for r in model_rows if int(r["is_diagonal"]) == 0 and float(r["transfer_accuracy_mean"]) < 50.0]
        strong = [r for r in model_rows if int(r["is_diagonal"]) == 0 and float(r["transfer_accuracy_mean"]) > 55.0]
        best = max((r for r in model_rows if int(r["is_diagonal"]) == 0), key=lambda r: float(r["transfer_accuracy_mean"]), default=None)
        worst = min((r for r in model_rows if int(r["is_diagonal"]) == 0), key=lambda r: float(r["transfer_accuracy_mean"]), default=None)
        out.append(
            {
                "condition": condition,
                "model": model,
                "diagonal_transfer_accuracy_mean": mean(diag),
                "offdiag_transfer_accuracy_mean": mean(off),
                "offdiag_transfer_accuracy_std": pstdev(off) if len(off) > 1 else 0.0,
                "strong_offdiag_count": len(strong),
                "anti_transfer_offdiag_count": len(anti),
                "best_offdiag": f"{best['source_site']}->{best['target_site']}" if best else "",
                "best_offdiag_accuracy": float(best["transfer_accuracy_mean"]) if best else 0.0,
                "worst_offdiag": f"{worst['source_site']}->{worst['target_site']}" if worst else "",
                "worst_offdiag_accuracy": float(worst["transfer_accuracy_mean"]) if worst else 0.0,
            }
        )
    return out


def fano_units(site: str) -> tuple[int, int, int]:
    m = FANO_SITE_RE.match(site)
    if not m:
        return (-1, -1, -1)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def annotate_fano_relation(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        source_units = fano_units(str(r["source_site"]))
        target_units = fano_units(str(r["target_site"]))
        shared = sorted(set(source_units) & set(target_units))
        if int(r["is_diagonal"]) == 1:
            r["shared_unit"] = "diag"
            r["source_shared_pos"] = -1
            r["target_shared_pos"] = -1
            r["shared_pos_pair"] = "diag"
            continue
        if len(shared) != 1:
            r["shared_unit"] = "none"
            r["source_shared_pos"] = -1
            r["target_shared_pos"] = -1
            r["shared_pos_pair"] = "none"
            continue
        unit = shared[0]
        source_pos = source_units.index(unit)
        target_pos = target_units.index(unit)
        r["shared_unit"] = unit
        r["source_shared_pos"] = source_pos
        r["target_shared_pos"] = target_pos
        r["shared_pos_pair"] = f"{source_pos}->{target_pos}"


def aggregate_relation_transfer(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if int(r["is_diagonal"]) == 1:
            continue
        groups[(str(r["model"]), str(r["shared_pos_pair"]))].append(r)
    out: list[dict[str, Any]] = []
    for (model, relation), group in sorted(groups.items()):
        acc = [float(r["transfer_accuracy_mean"]) for r in group]
        cos = [float(r["cosine_mean"]) for r in group]
        scores = [float(r["score_mean"]) for r in group]
        strong = [r for r in group if float(r["transfer_accuracy_mean"]) > 55.0]
        anti = [r for r in group if float(r["transfer_accuracy_mean"]) < 50.0]
        out.append(
            {
                "condition": condition,
                "model": model,
                "shared_pos_pair": relation,
                "transfer_accuracy_mean": mean(acc),
                "transfer_accuracy_std": pstdev(acc) if len(acc) > 1 else 0.0,
                "cosine_mean": mean(cos),
                "score_mean": mean(scores),
                "strong_edge_count": len(strong),
                "anti_transfer_edge_count": len(anti),
                "edge_count": len(group),
            }
        )
    return out


def read_edge_plan(path: Path) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"source_site", "target_site"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"edge plan missing columns {sorted(missing)}: {path}")
        for row in reader:
            flag = row.get("selected", "1").strip()
            if flag in {"", "0", "false", "False", "FALSE", "no", "No", "NO"}:
                continue
            source = row["source_site"]
            target = row["target_site"]
            if source == target:
                continue
            selected.add((source, target))
    if not selected:
        raise SystemExit(f"edge plan selected no off-diagonal edges: {path}")
    return selected


def filter_transfer_matrix(rows: list[dict[str, Any]], selected_edges: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        edge = (str(r["source_site"]), str(r["target_site"]))
        if edge in selected_edges:
            out.append(r)
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]], drop_delta: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = [k for k in rows[0].keys() if not (drop_delta and k == "delta")]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument(
        "--edge-plan",
        type=Path,
        default=None,
        help="Optional TSV with source_site/target_site/selected columns for balanced directed-edge diagnostics.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows, site_count = read_manifest(args.manifest)
    states, preds = parse_raw(args.raw_output)
    pair_rows = build_pair_rows(manifest_rows, states, preds)
    enrich_orbit_alignment(pair_rows)
    for r in pair_rows:
        r["condition"] = args.condition

    pair_out = []
    for r in pair_rows:
        pair_out.append({k: v for k, v in r.items() if k != "delta"})
    summary = aggregate(pair_rows, ["condition", "model"])
    site_summary = aggregate(pair_rows, ["condition", "model", "site", "line_idx"])
    orbit_summary = aggregate(pair_rows, ["condition", "model", "orbit_idx"])
    transfer_detail = line_transfer_rows(pair_rows)
    transfer_matrix = aggregate_line_transfer(transfer_detail, args.condition)
    annotate_fano_relation(transfer_matrix)
    transfer_summary = summarize_line_transfer(transfer_matrix, args.condition)
    relation_summary = aggregate_relation_transfer(transfer_matrix, args.condition)
    edge_plan_filtered_matrix: list[dict[str, Any]] = []
    edge_plan_relation_summary: list[dict[str, Any]] = []
    selected_edges: set[tuple[str, str]] | None = None
    if args.edge_plan is not None:
        selected_edges = read_edge_plan(args.edge_plan)
        edge_plan_filtered_matrix = filter_transfer_matrix(transfer_matrix, selected_edges)
        if not edge_plan_filtered_matrix:
            raise SystemExit(f"edge plan matched no transfer-matrix rows: {args.edge_plan}")
        edge_plan_relation_summary = aggregate_relation_transfer(edge_plan_filtered_matrix, args.condition)

    write_tsv(args.output_dir / "fano_orbit_pair_detail.tsv", pair_out)
    write_tsv(args.output_dir / "fano_orbit_summary.tsv", summary)
    write_tsv(args.output_dir / "fano_orbit_site_summary.tsv", site_summary)
    write_tsv(args.output_dir / "fano_orbit_orbit_summary.tsv", orbit_summary)
    write_tsv(args.output_dir / "fano_line_transfer_seed_detail.tsv", transfer_detail)
    write_tsv(args.output_dir / "fano_line_transfer_matrix.tsv", transfer_matrix)
    write_tsv(args.output_dir / "fano_line_transfer_summary.tsv", transfer_summary)
    write_tsv(args.output_dir / "fano_line_transfer_relation_summary.tsv", relation_summary)
    if selected_edges is not None:
        write_tsv(args.output_dir / "fano_line_transfer_matrix_edge_plan_filtered.tsv", edge_plan_filtered_matrix)
        write_tsv(args.output_dir / "fano_line_transfer_relation_summary_edge_plan_filtered.tsv", edge_plan_relation_summary)

    payload = {
        "schema": "neurodyn.fano_orbit_diagnostic.v1",
        "condition": args.condition,
        "raw_output": str(args.raw_output),
        "manifest": str(args.manifest),
        "site_count": site_count,
        "state_count": len(states),
        "prediction_count": len(preds),
        "pair_row_count": len(pair_rows),
        "summary": summary,
        "line_transfer_summary": transfer_summary,
        "line_transfer_relation_summary": relation_summary,
        "edge_plan": str(args.edge_plan) if args.edge_plan is not None else "",
        "edge_plan_selected_edges": len(selected_edges) if selected_edges is not None else 0,
        "edge_plan_line_transfer_relation_summary": edge_plan_relation_summary,
    }
    (args.output_dir / "fano_orbit_diagnostic.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
