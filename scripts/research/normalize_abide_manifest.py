#!/usr/bin/env python3
"""Normalize an ABIDE manifest into the versioned Brain O-SSM campaign contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from abide_campaign_lib import limit_subjects, load_manifest, select_sites


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize ABIDE manifest contract")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout", choices=["flat", "explicit-seq"], default="flat")
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--limit-subjects", type=int, default=0)
    args = parser.parse_args()

    meta, records = load_manifest(args.input)
    records = select_sites(records, args.max_sites or None)
    records = limit_subjects(records, args.limit_subjects or None)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_sites = sorted({record.site for record in records})

    fieldnames = ["subject_id", "label", "site"]
    if args.layout == "flat":
        for flat_index in range(meta.seq_len * meta.input_dim):
            fieldnames.append(f"f{flat_index}")
        feature_layout = "flat"
    else:
        for time_idx in range(meta.seq_len):
            for feature_idx in range(meta.input_dim):
                fieldnames.append(f"t{time_idx}_f{feature_idx}")
        feature_layout = "explicit_seq"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# schema={meta.schema or 'brain_ossm.abide.v2'}\n")
        handle.write(f"# seq_len={meta.seq_len}\n")
        handle.write(f"# input_dim={meta.input_dim}\n")
        handle.write(f"# feature_layout={feature_layout}\n")
        handle.write("# label_space=asd_vs_control\n")
        handle.write("# split_policy=leave_one_site_out\n")
        handle.write(f"# source_manifest={meta.path}\n")
        handle.write(f"# source_subject_count={meta.subject_count}\n")
        handle.write(f"# source_site_count={meta.site_count}\n")
        for passthrough_key in ("atlas", "feature_recipe", "roi_pooling", "time_pooling"):
            if passthrough_key in meta.extra:
                handle.write(f"# {passthrough_key}={meta.extra[passthrough_key]}\n")
        handle.write(f"# subject_count={len(records)}\n")
        handle.write(f"# site_count={len(filtered_sites)}\n")
        if args.max_sites:
            handle.write(f"# max_sites={args.max_sites}\n")
        if args.limit_subjects:
            handle.write(f"# limit_subjects={args.limit_subjects}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            row = {
                "subject_id": record.subject_id,
                "label": "ASD" if record.label == 1 else "CONTROL",
                "site": record.site,
            }
            if args.layout == "flat":
                flat_values = [value for step in record.sequence for value in step]
                for flat_index, value in enumerate(flat_values):
                    row[f"f{flat_index}"] = f"{value:.10f}"
            else:
                for time_idx, step in enumerate(record.sequence):
                    for feature_idx, value in enumerate(step):
                        row[f"t{time_idx}_f{feature_idx}"] = f"{value:.10f}"
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
