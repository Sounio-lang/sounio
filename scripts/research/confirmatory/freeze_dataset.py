#!/usr/bin/env python3
"""
C1 — Confirmatory dataset freeze for the Rfam OctTree lane.

Reads the exploratory corpus (rfam_structures.fasta, 108,072 records) and the
Rfam clan membership table, then emits a hash-closed freeze bundle:

  freeze/
    records.tsv.gz        per-record: id, family, clan, seq_len, eligible, sha256(record)
    manifest.json         sources + global hashes + counts + eligibility rule
    split.json            clan-held-out 70/15/15 assignment (groups intact)
    SHA256SUMS

Eligibility (predeclared, outcomes blind): 32 <= seq_len <= 1024,
len(seq) == len(struct), struct alphabet subset of "().".

Split: deterministic greedy bin-packing of clan groups by descending record
count into train/val/test at 70/15/15 of eligible records. Unclanned families
become singleton pseudo-clans. No outcome is inspected at any point.
"""

import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

FASTA = Path("/workspace/sounio/datasets/rna_secondary_structure/rfam_structures.fasta")
CLANIN = Path("/workspace/.wt/kimi-cli1/datasets/rna_secondary_structure/freeze/Rfam.clanin")
MEMBERSHIP = Path("/workspace/.wt/kimi-cli1/datasets/rna_secondary_structure/freeze/clan_membership.txt.gz")
OUTDIR = Path("/workspace/.wt/kimi-cli1/scripts/research/confirmatory/freeze")

MIN_LEN, MAX_LEN = 32, 1024
VALID_CHARS = set("().")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_membership(path: Path):
    fam2clan = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            clan, fam = line.split("\t")[:2]
            fam2clan[fam] = clan
    return fam2clan


def parse_fasta(path: Path, fam2clan):
    records = []
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            header = lines[i]
            fam = "unknown"
            for part in header.split():
                if part.startswith("family="):
                    fam = part.split("=", 1)[1]
                    break
            if i + 2 < len(lines):
                seq = lines[i + 1].strip()
                ss = lines[i + 2].strip()
                rid = header[1:].split()[0]
                eligible = (
                    MIN_LEN <= len(seq) <= MAX_LEN
                    and len(seq) == len(ss)
                    and set(ss) <= VALID_CHARS
                )
                raw = (header + "\n" + seq + "\n" + ss).encode()
                records.append(
                    {
                        "id": rid,
                        "family": fam,
                        "clan": fam2clan.get(fam, f"NOCLAN_{fam}"),
                        "seq_len": len(seq),
                        "eligible": eligible,
                        "sha256": sha(raw),
                    }
                )
            i += 3
        else:
            i += 1
    return records


def make_split(records):
    groups = defaultdict(lambda: {"eligible": 0, "total": 0})
    for r in records:
        g = groups[r["clan"]]
        g["total"] += 1
        g["eligible"] += int(r["eligible"])
    total_eligible = sum(g["eligible"] for g in groups.values())
    targets = {
        "train": 0.70 * total_eligible,
        "val": 0.15 * total_eligible,
        "test": 0.15 * total_eligible,
    }
    # Deterministic order: descending eligible count, tie-break by clan id.
    order = sorted(groups.items(), key=lambda kv: (-kv[1]["eligible"], kv[0]))
    assignment = {}
    loads = {"train": 0, "val": 0, "test": 0}
    priority = {"train": 0, "val": 1, "test": 2}
    for clan, g in order:
        if g["eligible"] == 0:
            # No eligible records: park in train (never used), recorded explicitly.
            assignment[clan] = "train_ineligible_only"
            continue
        # Water-filling: lowest load/target ratio wins; ties by split priority.
        chosen = min(
            ("train", "val", "test"),
            key=lambda s: (loads[s] / targets[s], priority[s]),
        )
        assignment[clan] = chosen
        loads[chosen] += g["eligible"]
    return assignment, groups, loads, total_eligible


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fam2clan = load_membership(MEMBERSHIP)
    records = parse_fasta(FASTA, fam2clan)
    assert len(records) == 108072, f"corpus drift: {len(records)} != 108072"

    assignment, groups, loads, total_eligible = make_split(records)

    # Per-record TSV with split assignment.
    tsv_path = OUTDIR / "records.tsv.gz"
    with gzip.open(tsv_path, "wt") as f:
        f.write("id\tfamily\tclan\tseq_len\teligible\tsplit\tsha256\n")
        for r in records:
            f.write(
                f"{r['id']}\t{r['family']}\t{r['clan']}\t{r['seq_len']}\t"
                f"{int(r['eligible'])}\t{assignment[r['clan']]}\t{r['sha256']}\n"
            )

    fams = {r["family"] for r in records}
    clans = {r["clan"] for r in records}
    eligible_n = sum(r["eligible"] for r in records)
    manifest = {
        "freeze_version": "1.0.0",
        "freeze_date": "2026-08-09",
        "purpose": "C1 confirmatory freeze — Rfam OctTree lane (outcomes blind)",
        "sources": {
            "rfam_structures_fasta": {
                "path": str(FASTA),
                "sha256": sha(FASTA.read_bytes()),
                "note": "exploratory corpus, 108,072 records, headers carry family=RF*",
            },
            "rfam_clanin": {
                "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.clanin",
                "sha256": sha(CLANIN.read_bytes()),
            },
            "clan_membership": {
                "url": "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/clan_membership.txt.gz",
                "sha256": sha(MEMBERSHIP.read_bytes()),
            },
        },
        "eligibility_rule": "32 <= seq_len <= 1024; len(seq)==len(struct); struct alphabet subset of ().",
        "counts": {
            "records": len(records),
            "eligible": eligible_n,
            "families": len(fams),
            "clans_plus_pseudo": len(clans),
            "clans_from_membership": len({c for c in clans if not c.startswith('NOCLAN_')}),
        },
        "split": {
            "rule": "clan-held-out; groups intact; deterministic greedy by descending eligible count; targets 70/15/15 of eligible records",
            "eligible_records": {"train": loads["train"], "val": loads["val"], "test": loads["test"]},
            "total_eligible": total_eligible,
        },
    }
    manifest_path = OUTDIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    split_doc = {
        "freeze_version": "1.0.0",
        "rule": manifest["split"]["rule"],
        "clan_assignment": dict(sorted(assignment.items())),
    }
    split_path = OUTDIR / "split.json"
    split_path.write_text(json.dumps(split_doc, indent=2) + "\n")

    sums = []
    for p in (tsv_path, manifest_path, split_path):
        sums.append(f"{sha(p.read_bytes())}  {p.name}")
    (OUTDIR / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    print(f"records={len(records)} eligible={eligible_n} families={len(fams)} groups={len(clans)}")
    print(f"split eligible: train={loads['train']} val={loads['val']} test={loads['test']}")
    print(f"frozen at {OUTDIR}")


if __name__ == "__main__":
    sys.exit(main())
