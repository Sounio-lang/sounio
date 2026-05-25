#!/usr/bin/env python3
"""Moonshot A: bind ABIDE cohort baseline to accepted Sinkhorn runtime artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SUMMARY = ROOT / "artifacts" / "research" / "abide_orc" / "cohort_summary.tsv"
DEFAULT_ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
DEFAULT_ORC_DIR = ROOT / "artifacts" / "research" / "abide_orc"
DEFAULT_MANIFEST = ROOT / "artifacts" / "research" / "brain_ossm_full_cohort" / "abide_roi_manifest.tsv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_phase_status() -> Path | None:
    found = sorted(
        Path("/tmp").glob("moonshot-a-phase-status.*/moonshot_a_phase_status.v1.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return found[-1] if found else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def cohen_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    pooled = math.sqrt((std(a) ** 2 + std(b) ** 2) / 2.0)
    return (mean(a) - mean(b)) / pooled if pooled > 0 else float("nan")


def summarize_feature(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    values = [finite_float(r.get(field, "")) for r in rows]
    finite = [v for v in values if v is not None]
    return {
        "field": field,
        "n": len(finite),
        "mean": mean(finite),
        "std": std(finite),
        "min": min(finite) if finite else float("nan"),
        "max": max(finite) if finite else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--abide-dir", default=str(DEFAULT_ABIDE_DIR))
    ap.add_argument("--orc-dir", default=str(DEFAULT_ORC_DIR))
    ap.add_argument("--phenotype-manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--phase-status", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary_path = Path(args.summary)
    abide_dir = Path(args.abide_dir)
    orc_dir = Path(args.orc_dir)
    phenotype_manifest = Path(args.phenotype_manifest)
    phase_path = Path(args.phase_status) if args.phase_status else latest_phase_status()
    if phase_path is None:
        raise SystemExit("moonshot_a_abide_cohort_manifest: missing PASS_FULL phase-status artifact")
    if not summary_path.is_file():
        raise SystemExit(f"moonshot_a_abide_cohort_manifest: missing cohort summary: {summary_path}")

    phase = load_json(phase_path)
    rows = list(csv.DictReader(summary_path.open(), delimiter="\t"))
    if not rows:
        raise SystemExit("moonshot_a_abide_cohort_manifest: cohort summary has no rows")

    subject_count = len(rows)
    input_count = len(list(abide_dir.glob("*_rois_cc200.1D"))) if abide_dir.is_dir() else 0
    orc_npy_count = len(list(orc_dir.glob("*_orc.npy"))) if orc_dir.is_dir() else 0
    label_counts = Counter(r.get("label", "UNKNOWN") for r in rows)
    site_counts = Counter(r.get("site", "UNKNOWN") for r in rows)
    labelled = [r for r in rows if r.get("label") != "UNKNOWN"]
    asd = [r for r in labelled if r.get("label") == "ASD"]
    td = [r for r in labelled if r.get("label") in {"TD", "TC", "CONTROL"}]

    mean_orc_asd = [v for r in asd if (v := finite_float(r.get("mean", ""))) is not None]
    mean_orc_td = [v for r in td if (v := finite_float(r.get("mean", ""))) is not None]
    std_orc_asd = [v for r in asd if (v := finite_float(r.get("std", ""))) is not None]
    std_orc_td = [v for r in td if (v := finite_float(r.get("std", ""))) is not None]

    scalar = phase.get("gpu_runtime", {}).get("scalar_slurm") or {}
    epi = phase.get("gpu_runtime", {}).get("f32_epistemic_slurm") or {}
    scalar_pass = scalar.get("status") == "pass"
    epi_pass = epi.get("status") == "pass"
    phase_full = phase.get("status") == "PASS_FULL"
    summary_complete = subject_count > 0 and orc_npy_count >= subject_count

    status = "PASS_COHORT_BASELINE_READY" if phase_full and scalar_pass and epi_pass and summary_complete else "FAIL"
    doc = {
        "schema": "sounio.moonshot_a.abide_cohort_manifest.v1",
        "status": status,
        "inputs": {
            "phase_status": {
                "path": str(phase_path),
                "sha256": sha256(phase_path),
                "status": phase.get("status", ""),
            },
            "cohort_summary": {
                "path": str(summary_path),
                "bytes": summary_path.stat().st_size,
                "sha256": sha256(summary_path),
            },
            "phenotype_manifest": {
                "path": str(phenotype_manifest),
                "bytes": phenotype_manifest.stat().st_size if phenotype_manifest.exists() else 0,
                "sha256": sha256(phenotype_manifest) if phenotype_manifest.exists() else "",
            },
        },
        "runtime_baseline": {
            "scalar_job": scalar.get("job", {}).get("id", ""),
            "scalar_ptx_sha256": scalar.get("ptx", {}).get("sha256", ""),
            "scalar_status": scalar.get("status", ""),
            "f32_epistemic_job": epi.get("job", {}).get("id", ""),
            "f32_epistemic_ptx_sha256": epi.get("ptx", {}).get("sha256", ""),
            "f32_epistemic_status": epi.get("status", ""),
            "f32_epistemic_variance_interpretation": "finite_overflow_bound_not_calibrated_coverage",
        },
        "cohort_baseline": {
            "abide_input_files": input_count,
            "orc_npy_files": orc_npy_count,
            "summary_subjects": subject_count,
            "label_counts": dict(sorted(label_counts.items())),
            "site_count": len(site_counts),
            "labelled_subjects": len(labelled),
            "features": {
                "mean_orc": summarize_feature(rows, "mean"),
                "std_orc": summarize_feature(rows, "std"),
                "frac_pos": summarize_feature(rows, "frac_pos"),
                "frac_neg": summarize_feature(rows, "frac_neg"),
            },
            "asd_td_screen": {
                "asd_n": len(mean_orc_asd),
                "td_n": len(mean_orc_td),
                "mean_orc_asd_mean": mean(mean_orc_asd),
                "mean_orc_td_mean": mean(mean_orc_td),
                "mean_orc_cohen_d": cohen_d(mean_orc_asd, mean_orc_td),
                "std_orc_asd_mean": mean(std_orc_asd),
                "std_orc_td_mean": mean(std_orc_td),
                "std_orc_cohen_d": cohen_d(std_orc_asd, std_orc_td),
            },
        },
        "next_campaign": {
            "goal": "extend Moonshot A from slice evidence to cohort-scale f32-epistemic ORC",
            "run_surface": "Slurm/Foundry only; do not run full cohort stress in /workspace/sounio",
            "first_gate": "preflight cohort artifact hashes and runtime baseline before submitting a new cohort GPU job",
            "candidate_driver": "scripts/research/abide_cohort_orc_sweep.py",
        },
        "boundaries": [
            "uses_existing_scalar_abide_orc_cohort_summary_as_baseline",
            "does_not_rerun_full_abide_cohort",
            "does_not_claim_f32_epistemic_cohort_revalidation_yet",
            "asd_td_screen_is_descriptive_not_confirmatory",
            "f32_epistemic_variance_is_finite_overflow_bound_not_calibrated_coverage",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        "moonshot_a_abide_cohort_manifest: "
        f"{status} subjects={subject_count} labelled={len(labelled)} "
        f"scalar={scalar.get('status','missing')} f32_epistemic={epi.get('status','missing')} "
        f"out={out}"
    )
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
