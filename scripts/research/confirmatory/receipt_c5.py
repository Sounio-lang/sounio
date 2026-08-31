#!/usr/bin/env python3
"""C5 — final receipt: binds the whole confirmatory into one hash-closed doc.

Collects:
  - lane + git commits (C0..C5) at receipt time
  - freeze hashes (C1 manifest, C2 families, C3 golden)
  - pre-results amendments (task_freeze.md sha256)
  - tensor receipt (Sounio producer vs controls.py)
  - all result cells present (sha256 sidecars re-verified)
  - aggregate_c4.json verdict (Python) — Julia verdict is produced by
    validator.jl in the same run and embedded when
    results/validator_c5_report.json exists (written by the Julia side)

Writes receipt_c5.json + receipt_c5.sha256. Idempotent; safe to run on a
partial grid (marks cells_complete=false).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = HERE.parents[2]  # /workspace/.wt/kimi-cli1


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_sha(p: Path) -> str:
    return sha(p.read_bytes())


def git(args):
    return subprocess.run(["git", "-C", str(LANE)] + args,
                          capture_output=True, text=True, check=True).stdout.strip()


def main():
    receipt = {"kind": "c5_receipt", "lane": None, "commits": {},
               "freeze": {}, "amendments": {}, "tensors": {},
               "cells": {}, "verdicts": {}}

    receipt["lane"] = git(["rev-parse", "--abbrev-ref", "HEAD"])
    receipt["commits"]["HEAD"] = git(["rev-parse", "HEAD"])
    receipt["commits"]["log_oneline_12"] = git(["log", "--oneline", "-12"]).splitlines()
    receipt["commits"]["worktree_dirty"] = bool(git(["status", "--porcelain"]))

    fr = HERE / "freeze"
    receipt["freeze"] = {
        "manifest_sha256": file_sha(fr / "manifest.json"),
        "records_sha256": file_sha(fr / "records.tsv.gz"),
        "split_sha256": file_sha(fr / "split.json"),
        "families_sha256": file_sha(HERE / "families.json"),
        "golden_sha256_file": file_sha(HERE / "golden_corruptions.json"),
    }
    receipt["amendments"]["task_freeze_sha256"] = file_sha(HERE / "task_freeze.md")

    tr = json.loads((HERE / "tensor_receipt.json").read_text())
    receipt["tensors"] = {
        "tensor_receipt_sha256": file_sha(HERE / "tensor_receipt.json"),
        "verdict": tr["verdict"],
        "cl3_exact_match": tr["cl3_exact_match"],
        "oct_isomorphism": tr["oct_isomorphism"],
    }

    cells = {}
    resdir = HERE / "results"
    n_ok = 0
    for p in sorted(resdir.glob("seed*_L*.json")):
        side = p.with_suffix(".sha256")
        okk = side.exists() and sha(p.read_bytes()) == side.read_text().split()[0].strip()
        cells[p.stem] = {"sha256": sha(p.read_bytes()), "sidecar_ok": okk}
        n_ok += int(okk)
    receipt["cells"] = {
        "present": len(cells), "expected": 80, "sidecars_ok": n_ok,
        "complete": len(cells) == 80 and n_ok == 80,
        "detail": cells,
    }

    agg = resdir / "aggregate_c4.json"
    if agg.exists():
        a = json.loads(agg.read_text())
        receipt["verdicts"]["python"] = {
            "aggregate_sha256": file_sha(agg),
            "promotion": a.get("promotion"),
        }
    vj = resdir / "validator_c5_report.json"
    if vj.exists():
        receipt["verdicts"]["julia"] = {
            "report_sha256": file_sha(vj),
            "report": json.loads(vj.read_text()),
        }

    raw = json.dumps(receipt, indent=2) + "\n"
    (HERE / "receipt_c5.json").write_text(raw)
    (HERE / "receipt_c5.sha256").write_text(f"{sha(raw.encode())}  receipt_c5.json\n")
    print(f"cells: {receipt['cells']['present']}/80 sidecars_ok={n_ok} "
          f"complete={receipt['cells']['complete']}")
    print(f"receipt_c5.json sha256={sha(raw.encode())[:16]}…")
    if receipt["commits"]["worktree_dirty"]:
        print("WARNING: worktree dirty — commit before final receipt")


if __name__ == "__main__":
    sys.exit(main() or 0)
