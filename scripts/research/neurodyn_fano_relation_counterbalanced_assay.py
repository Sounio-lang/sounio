#!/usr/bin/env python3
"""Build a relation-counterbalanced Fano-line NeuroDyn assay package.

This packages the temporal manifest and the directed-edge relation plan at
generation time. The model still trains on subjects whose observations live on
Fano lines; the edge plan defines the pre-registered transfer diagnostic over
directed source-line -> target-line relations.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "Synthetic non-clinical relation-counterbalanced Fano transfer assay only. "
    "No clinical, biomarker, biological mechanism, solved-transfer, or broad "
    "O-SSM superiority claim."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=280)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--noise-std", type=float, default=0.015)
    ap.add_argument("--magnitude", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=2026070710)
    ap.add_argument("--edge-seed", type=int, default=2026070711)
    ap.add_argument("--edges-per-relation", type=int, default=2)
    ap.add_argument(
        "--target-relation",
        default="",
        help="Optional pre-registered relation such as 2->0 for fixed-target diagnostics.",
    )
    ap.add_argument("--anchor", choices=("zero", "unit_real"), default="unit_real")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(message)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rel_counts = summary["selected_relation_counts"]
    source_counts = summary["selected_source_counts"]
    target_counts = summary["selected_target_counts"]
    lines = [
        "# NeuroDyn Fano Relation-Counterbalanced Assay",
        "",
        f"Generated: {summary['created_at_utc']}",
        "",
        "## Scope",
        "",
        CLAIM_BOUNDARY,
        "",
        "## Manifest",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Pairs: `{summary['pairs']}`",
        f"- Records: `{summary['records']}`",
        f"- Sites: `{summary['sites']}`",
        f"- Sequence length: `{summary['seq_len']}`",
        f"- Magnitude: `{summary['magnitude']}`",
        f"- Noise std: `{summary['noise_std']}`",
        f"- Anchor: `{summary['anchor']}`",
        "",
        "## Invariant Gate",
        "",
        f"- mean max abs diff: `{summary['invariant_max']['mean_max_abs_diff']}`",
        f"- delta max abs diff: `{summary['invariant_max']['delta_max_abs_diff']}`",
        f"- start max abs diff: `{summary['invariant_max']['start_max_abs_diff']}`",
        f"- end max abs diff: `{summary['invariant_max']['end_max_abs_diff']}`",
        f"- energy abs diff: `{summary['invariant_max']['energy_abs_diff']}`",
        f"- same unordered multiset min: `{summary['invariant_max']['same_unordered_multiset_min']}`",
        "",
        "## Relation Edge Plan",
        "",
        f"- Edge plan: `{summary['edge_plan']}`",
        f"- Directed off-diagonal edges: `{summary['directed_offdiag_edges']}`",
        f"- Selected edges: `{summary['selected_edges']}`",
        f"- Edges per relation: `{summary['edges_per_relation']}`",
        f"- Target relation: `{summary['target_relation'] or 'none'}`",
        f"- Target selected edges: `{summary['target_edges_selected']}`",
        "",
        "| relation | selected_edges |",
        "| --- | ---: |",
    ]
    for rel, count in sorted(rel_counts.items()):
        lines.append(f"| {rel} | {count} |")
    lines.extend(
        [
            "",
            "## Source Counts",
            "",
            "| source_site | selected_edges |",
            "| --- | ---: |",
        ]
    )
    for site, count in sorted(source_counts.items()):
        lines.append(f"| {site} | {count} |")
    lines.extend(
        [
            "",
            "## Target Counts",
            "",
            "| target_site | selected_edges |",
            "| --- | ---: |",
        ]
    )
    for site, count in sorted(target_counts.items()):
        lines.append(f"| {site} | {count} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{summary['decision']}`",
            "",
            "This package is ready for a single true-label run plus one matched pair-label null smoke. "
            "Do not use it as clinical or mechanistic evidence without the trained run, null envelope, "
            "checkpoint replay, and hidden-state diagnostics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out = args.output_dir.resolve()
    fail_if(out.exists() and any(out.iterdir()) and not args.overwrite, f"output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest_script = script_dir / "neurodyn_noncommutative_temporal_manifest.py"
    edge_plan_script = script_dir / "neurodyn_fano_relation_edge_plan.py"
    manifest_cmd = [
        sys.executable,
        str(manifest_script),
        "--output-dir",
        str(out),
        "--pairs",
        str(args.pairs),
        "--sites",
        "7",
        "--seq-len",
        str(args.seq_len),
        "--noise-std",
        str(args.noise_std),
        "--magnitude",
        str(args.magnitude),
        "--seed",
        str(args.seed),
        "--line-mode",
        "fano_cycle",
        "--site-mode",
        "fano_line",
        "--anchor",
        args.anchor,
    ]
    if args.overwrite:
        manifest_cmd.append("--overwrite")
    run(manifest_cmd)

    manifest = out / "noncommutative_temporal_manifest.tsv"
    edge_dir = out / "relation_edge_plan"
    edge_cmd = [
        sys.executable,
        str(edge_plan_script),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(edge_dir),
        "--seed",
        str(args.edge_seed),
        "--edges-per-relation",
        str(args.edges_per_relation),
    ]
    if args.target_relation:
        edge_cmd.extend(["--target-relation", args.target_relation])
    if args.overwrite:
        edge_cmd.append("--overwrite")
    run(edge_cmd)

    manifest_summary = read_json(out / "noncommutative_temporal_prepare_summary.json")
    edge_summary = read_json(edge_dir / "fano_relation_balanced_edge_plan_summary.json")
    invariant_max = manifest_summary["invariant_max"]
    fail_if(manifest_summary["sites"] != 7, "expected 7 Fano-line sites")
    fail_if(manifest_summary["line_mode"] != "fano_cycle", "expected fano_cycle line mode")
    fail_if(manifest_summary["site_mode"] != "fano_line", "expected fano_line site mode")
    fail_if(manifest_summary["label_counts"] != {"0": args.pairs, "1": args.pairs}, "label counts are not pair-balanced")
    for site, counts in manifest_summary["site_label_counts"].items():
        fail_if(counts["0"] != counts["1"], f"site labels not balanced for {site}: {counts}")
    for key, value in invariant_max.items():
        if key == "same_unordered_multiset_min":
            fail_if(int(value) != 1, "paired unordered multiset invariant failed")
        else:
            fail_if(float(value) > 1e-9, f"paired invariant failed for {key}: {value}")
    rel_counts = edge_summary["selected_relation_counts"]
    fail_if(len(set(rel_counts.values())) != 1, f"selected relation counts are not balanced: {rel_counts}")
    fail_if(edge_summary["selected_edges"] != args.edges_per_relation * 9, "unexpected selected edge count")
    if args.target_relation:
        fail_if(edge_summary["target_relation"] != args.target_relation, "edge summary target relation mismatch")
        fail_if(
            edge_summary["target_edges_selected"] != args.edges_per_relation,
            f"target selected edge count mismatch: {edge_summary['target_edges_selected']}",
        )

    selected_rows = read_csv_rows(edge_dir / "fano_relation_balanced_edge_plan_selected.tsv")
    target_rows = []
    if args.target_relation:
        target_rows = read_csv_rows(edge_dir / "fano_relation_fixed_target_edge_plan_selected.tsv")
    summary = {
        "schema": (
            "neurodyn.fano_relation_fixed_target_assay.v1"
            if args.target_relation
            else "neurodyn.fano_relation_counterbalanced_assay.v1"
        ),
        "created_at_utc": manifest_summary["created_at_utc"],
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(out),
        "manifest": str(manifest),
        "edge_plan": str(edge_dir / "fano_relation_balanced_edge_plan.tsv"),
        "selected_edge_plan": str(edge_dir / "fano_relation_balanced_edge_plan_selected.tsv"),
        "target_selected_edge_plan": edge_summary["target_selected_edge_plan"],
        "pairs": manifest_summary["pairs"],
        "records": manifest_summary["records"],
        "sites": manifest_summary["sites"],
        "seq_len": manifest_summary["seq_len"],
        "noise_std": manifest_summary["noise_std"],
        "magnitude": manifest_summary["magnitude"],
        "anchor": manifest_summary["anchor"],
        "seed": args.seed,
        "edge_seed": args.edge_seed,
        "label_counts": manifest_summary["label_counts"],
        "site_label_counts": manifest_summary["site_label_counts"],
        "invariant_max": invariant_max,
        "directed_offdiag_edges": edge_summary["directed_offdiag_edges"],
        "selected_edges": edge_summary["selected_edges"],
        "edges_per_relation": edge_summary["edges_per_relation"],
        "selected_relation_counts": edge_summary["selected_relation_counts"],
        "selected_source_counts": edge_summary["selected_source_counts"],
        "selected_target_counts": edge_summary["selected_target_counts"],
        "selected_edge_rows": len(selected_rows),
        "target_relation": args.target_relation,
        "target_edges_available": edge_summary["target_edges_available"],
        "target_edges_selected": edge_summary["target_edges_selected"],
        "target_edge_rows": len(target_rows),
        "decision": (
            "RELATION_FIXED_TARGET_ASSAY_READY_FOR_SINGLE_SLURM_TRUE_PLUS_NULL_SMOKE"
            if args.target_relation
            else "RELATION_COUNTERBALANCED_ASSAY_READY_FOR_SINGLE_SLURM_TRUE_PLUS_NULL_SMOKE"
        ),
        "next_gate": (
            "Run one true-label hidden-only O-SSM job plus one matched pair-label null, "
            "then replay diagnostics with --edge-plan and the pre-registered target relation "
            "before expanding to five nulls."
        ),
    }
    (out / "relation_counterbalanced_assay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out / "relation_counterbalanced_assay_summary.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
