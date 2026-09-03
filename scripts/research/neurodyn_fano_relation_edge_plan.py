#!/usr/bin/env python3
"""Build a relation-balanced directed Fano-line edge plan.

The Fano-line transfer diagnostic has uneven directed-edge counts per shared
position relation. This helper selects an equal number of off-diagonal
source->target edges for every relation so follow-up diagnostics can test
whether a relation-level effect survives edge-count balancing.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FANO_SITE_RE = re.compile(r"fano_line_(\d+)_(\d+)-(\d+)-(\d+)$")
RELATION_RE = re.compile(r"^[0-2]->[0-2]$")


def read_sites(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open(newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")), delimiter="\t")
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        if "site" not in idx:
            raise SystemExit(f"manifest missing site column: {manifest}")
        seen: dict[str, dict[str, Any]] = {}
        for row in reader:
            site = row[idx["site"]]
            if site in seen:
                continue
            m = FANO_SITE_RE.match(site)
            if not m:
                raise SystemExit(f"site is not a fano_line site: {site}")
            seen[site] = {
                "site": site,
                "line_idx": int(m.group(1)),
                "units": (int(m.group(2)), int(m.group(3)), int(m.group(4))),
            }
    rows = sorted(seen.values(), key=lambda r: int(r["line_idx"]))
    if len(rows) != 7:
        raise SystemExit(f"expected 7 Fano-line sites, got {len(rows)}")
    return rows


def relation(source_units: tuple[int, int, int], target_units: tuple[int, int, int]) -> tuple[str, int]:
    shared = sorted(set(source_units) & set(target_units))
    if len(shared) != 1:
        return ("none", -1)
    unit = shared[0]
    return (f"{source_units.index(unit)}->{target_units.index(unit)}", unit)


def build_edges(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for source in sites:
        for target in sites:
            if source["site"] == target["site"]:
                continue
            rel, shared_unit = relation(source["units"], target["units"])
            out.append(
                {
                    "source_line_idx": source["line_idx"],
                    "source_site": source["site"],
                    "target_line_idx": target["line_idx"],
                    "target_site": target["site"],
                    "shared_pos_pair": rel,
                    "shared_unit": shared_unit,
                    "selected": 0,
                }
            )
    return out


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026070607)
    ap.add_argument(
        "--edges-per-relation",
        type=int,
        default=0,
        help="0 means use the minimum available count across relations.",
    )
    ap.add_argument(
        "--target-relation",
        default="",
        help="Optional pre-registered relation such as 2->0. Emits a selected target-only plan.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.target_relation and not RELATION_RE.match(args.target_relation):
        raise SystemExit(f"--target-relation must look like 0->2, got {args.target_relation!r}")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir exists and is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = read_sites(args.manifest)
    edges = build_edges(sites)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge["shared_pos_pair"] == "none":
            raise SystemExit("Fano edge without unique shared unit")
        by_relation[str(edge["shared_pos_pair"])].append(edge)

    available_counts = {rel: len(items) for rel, items in sorted(by_relation.items())}
    if args.edges_per_relation > 0:
        edges_per_relation = args.edges_per_relation
    else:
        edges_per_relation = min(available_counts.values())
    too_small = {rel: n for rel, n in available_counts.items() if n < edges_per_relation}
    if too_small:
        raise SystemExit(f"not enough edges for requested balance: {too_small}")

    rng = random.Random(args.seed)
    selected_keys: set[tuple[str, str]] = set()
    for rel, rel_edges in sorted(by_relation.items()):
        candidates = list(rel_edges)
        rng.shuffle(candidates)
        for edge in sorted(candidates[:edges_per_relation], key=lambda r: (r["source_site"], r["target_site"])):
            selected_keys.add((str(edge["source_site"]), str(edge["target_site"])))

    for edge in edges:
        key = (str(edge["source_site"]), str(edge["target_site"]))
        edge["selected"] = 1 if key in selected_keys else 0

    selected = [edge for edge in edges if int(edge["selected"]) == 1]
    target_selected = []
    if args.target_relation:
        target_available = by_relation.get(args.target_relation, [])
        if not target_available:
            raise SystemExit(f"target relation is not available: {args.target_relation}")
        target_selected = [edge for edge in selected if str(edge["shared_pos_pair"]) == args.target_relation]
        if len(target_selected) != edges_per_relation:
            raise SystemExit(
                "target relation did not survive balanced selection: "
                f"{args.target_relation} selected={len(target_selected)} expected={edges_per_relation}"
            )
    selected_counts: dict[str, int] = defaultdict(int)
    for edge in selected:
        selected_counts[str(edge["shared_pos_pair"])] += 1
    source_counts: dict[str, int] = defaultdict(int)
    target_counts: dict[str, int] = defaultdict(int)
    for edge in selected:
        source_counts[str(edge["source_site"])] += 1
        target_counts[str(edge["target_site"])] += 1

    write_tsv(args.output_dir / "fano_relation_balanced_edge_plan.tsv", edges)
    write_tsv(args.output_dir / "fano_relation_balanced_edge_plan_selected.tsv", selected)
    if args.target_relation:
        write_tsv(args.output_dir / "fano_relation_fixed_target_edge_plan_selected.tsv", target_selected)
    summary = {
        "schema": "neurodyn.fano_relation_edge_plan.v1",
        "manifest": str(args.manifest),
        "seed": args.seed,
        "site_count": len(sites),
        "directed_offdiag_edges": len(edges),
        "available_relation_counts": available_counts,
        "edges_per_relation": edges_per_relation,
        "selected_edges": len(selected),
        "selected_relation_counts": dict(sorted(selected_counts.items())),
        "selected_source_counts": dict(sorted(source_counts.items())),
        "selected_target_counts": dict(sorted(target_counts.items())),
        "selected_source_count_mean": mean(source_counts.values()) if source_counts else 0.0,
        "selected_target_count_mean": mean(target_counts.values()) if target_counts else 0.0,
        "target_relation": args.target_relation,
        "target_edges_available": len(by_relation.get(args.target_relation, [])) if args.target_relation else 0,
        "target_edges_selected": len(target_selected),
        "target_selected_edge_plan": (
            str(args.output_dir / "fano_relation_fixed_target_edge_plan_selected.tsv")
            if args.target_relation
            else ""
        ),
        "decision": (
            "RELATION_FIXED_TARGET_EDGE_PLAN_READY"
            if args.target_relation
            else "RELATION_COUNTERBALANCED_EDGE_PLAN_READY"
        ),
        "claim_boundary": (
            "Synthetic Fano relation counterbalance plan only; no clinical, "
            "biomarker, mechanistic, solved-transfer, or superiority claim."
        ),
    }
    (args.output_dir / "fano_relation_balanced_edge_plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
