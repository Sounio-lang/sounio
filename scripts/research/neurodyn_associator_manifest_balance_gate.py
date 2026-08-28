#!/usr/bin/env python3
"""Gate site/label/associator-dimension balance for associator manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "neurodyn.associator_manifest_balance_gate.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_id(subject_id: str) -> str:
    return subject_id.split("__", 1)[0]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t"))


def read_triples(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {f"assoc_pair_{int(row['pair_id']):04d}": row for row in rows}


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--associator-triples", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "associator_manifest_balance_gate.json").exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {out}")

    manifest_rows = read_manifest(args.manifest)
    triples = read_triples(args.associator_triples)
    pair_site: dict[str, str] = {}
    label_by_site: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest_rows:
        pid = pair_id(row["subject_id"])
        pair_site.setdefault(pid, row["site"])
        if pair_site[pid] != row["site"]:
            raise SystemExit(f"pair split across sites: {pid}")
        label_by_site[row["site"]][row["label"]] += 1

    site_dim_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dim_site_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sign_site_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for pid, site in sorted(pair_site.items()):
        meta = triples.get(pid)
        if meta is None:
            raise SystemExit(f"missing triple metadata for {pid}")
        dim = str(meta["assoc_dim"])
        sign = "positive" if float(meta["positive_assoc_value"]) > 0.0 else "negative"
        site_dim_counts[site][dim] += 1
        dim_site_counts[dim][site] += 1
        sign_site_counts[site][sign] += 1

    site_rows: list[dict[str, Any]] = []
    all_dims = sorted({dim for counts in site_dim_counts.values() for dim in counts}, key=int)
    for site in sorted(site_dim_counts):
        row: dict[str, Any] = {
            "site": site,
            "pair_count": sum(site_dim_counts[site].values()),
            "label_0": label_by_site[site].get("0", 0),
            "label_1": label_by_site[site].get("1", 0),
            "assoc_sign_positive": sign_site_counts[site].get("positive", 0),
            "assoc_sign_negative": sign_site_counts[site].get("negative", 0),
            "assoc_dim_nonzero_count": sum(1 for dim in all_dims if site_dim_counts[site].get(dim, 0) > 0),
        }
        for dim in all_dims:
            row[f"assoc_dim_{int(dim):02d}"] = site_dim_counts[site].get(dim, 0)
        site_rows.append(row)

    dim_rows: list[dict[str, Any]] = []
    for dim in all_dims:
        counts = dim_site_counts[dim]
        dim_rows.append(
            {
                "assoc_dim": dim,
                "site_nonzero_count": sum(1 for site in counts if counts[site] > 0),
                "pair_count": sum(counts.values()),
                "min_pairs_per_nonzero_site": min(counts.values()) if counts else 0,
                "max_pairs_per_site": max(counts.values()) if counts else 0,
            }
        )

    label_balanced = all(counts.get("0", 0) == counts.get("1", 0) for counts in label_by_site.values())
    site_dim_nonzero_min = min(row["assoc_dim_nonzero_count"] for row in site_rows) if site_rows else 0
    dim_site_nonzero_min = min(row["site_nonzero_count"] for row in dim_rows) if dim_rows else 0
    deterministic_site_dim = any(row["assoc_dim_nonzero_count"] == 1 for row in site_rows)
    decision = (
        "ASSOCIATOR_MANIFEST_SITE_DIM_DECOUPLED_READY"
        if label_balanced and site_dim_nonzero_min >= 4 and dim_site_nonzero_min >= 4 and not deterministic_site_dim
        else "ASSOCIATOR_MANIFEST_BALANCE_GATE_NOT_READY"
    )
    payload = {
        "schema": SCHEMA,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "associator_triples": str(args.associator_triples),
        "associator_triples_sha256": sha256_file(args.associator_triples),
        "site_count": len(site_rows),
        "pair_count": len(pair_site),
        "assoc_dim_count": len(all_dims),
        "label_balanced_by_site": label_balanced,
        "site_dim_nonzero_min": site_dim_nonzero_min,
        "dim_site_nonzero_min": dim_site_nonzero_min,
        "deterministic_site_dim": deterministic_site_dim,
        "decision": decision,
    }
    write_tsv(out / "site_assoc_dim_balance.tsv", site_rows)
    write_tsv(out / "assoc_dim_site_balance.tsv", dim_rows)
    (out / "associator_manifest_balance_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Associator Manifest Balance Gate",
        "",
        f"Decision: `{decision}`",
        "",
        f"- pairs: `{payload['pair_count']}`",
        f"- sites: `{payload['site_count']}`",
        f"- assoc dims: `{payload['assoc_dim_count']}`",
        f"- label balanced by site: `{label_balanced}`",
        f"- min assoc dims per site: `{site_dim_nonzero_min}`",
        f"- min sites per assoc dim: `{dim_site_nonzero_min}`",
        f"- deterministic site->dim coupling: `{deterministic_site_dim}`",
        "",
    ]
    (out / "associator_manifest_balance_gate.md").write_text("\n".join(md), encoding="utf-8")
    with (out / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "SHA256SUMS"):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
