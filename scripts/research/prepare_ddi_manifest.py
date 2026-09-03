#!/usr/bin/env python3
"""Prepare the canonical DrugBank FAERS DDI manifest for O-SSM benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddi_campaign_lib import load_ddi_samples, write_ddi_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DDI DrugBank manifest for O-SSM benchmark")
    parser.add_argument(
        "--output-dir",
        default="artifacts/research/ddi_drugbank_benchmark",
        help="Output directory for the manifest and downstream benchmark artifacts",
    )
    parser.add_argument(
        "--dataset-name",
        default="faers_drugbank.csv",
        help="Source CSV under data/ to materialize",
    )
    parser.add_argument(
        "--fold-count",
        type=int,
        default=5,
        help="Number of stratified folds to assign in the manifest metadata",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_ddi_samples(repo_root, dataset_name=args.dataset_name, fold_count=args.fold_count)
    manifest_path = output_dir / "ddi_drugbank_manifest.tsv"
    write_ddi_manifest(samples, manifest_path, fold_count=args.fold_count)

    fano_count = sum(1 for sample in samples if sample.fano)
    nonfano_count = len(samples) - fano_count
    print(f"Samples: {len(samples)}")
    print(f"Fano: {fano_count}")
    print(f"Non-Fano: {nonfano_count}")
    print(f"Written: {manifest_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()

