#!/usr/bin/env python3
"""Moonshot A: compare f32-epistemic ABIDE cohort output with scalar baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCALAR = ROOT / "artifacts" / "research" / "abide_orc" / "cohort_summary.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def cohen_d(rows: list[dict[str, str]], field: str) -> float | None:
    asd = [f(r, field) for r in rows if r["label"] == "ASD"]
    td = [f(r, field) for r in rows if r["label"] == "TD"]
    if not asd or not td:
        return None
    pooled = math.sqrt((pstdev(asd) ** 2 + pstdev(td) ** 2) / 2.0)
    return (mean(asd) - mean(td)) / pooled if pooled > 0 else None


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar-summary", default=str(DEFAULT_SCALAR))
    ap.add_argument("--f32-summary", required=True)
    ap.add_argument("--f32-artifact", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scalar_path = Path(args.scalar_summary)
    f32_path = Path(args.f32_summary)
    artifact_path = Path(args.f32_artifact)
    out_path = Path(args.out)

    scalar_rows = read_rows(scalar_path)
    f32_rows = read_rows(f32_path)
    scalar_by_subject = {r["subject"]: r for r in scalar_rows}
    f32_by_subject = {r["subject"]: r for r in f32_rows}
    subjects = sorted(set(scalar_by_subject) & set(f32_by_subject))
    scalar_only = sorted(set(scalar_by_subject) - set(f32_by_subject))
    f32_only = sorted(set(f32_by_subject) - set(scalar_by_subject))
    artifact = json.loads(artifact_path.read_text())

    deltas = []
    scalar_means = []
    f32_means = []
    for subject in subjects:
        s = scalar_by_subject[subject]
        e = f32_by_subject[subject]
        scalar_mean = f(s, "mean")
        f32_mean = f(e, "mean")
        scalar_means.append(scalar_mean)
        f32_means.append(f32_mean)
        deltas.append({
            "subject": subject,
            "label": e["label"],
            "site": e["site"],
            "mean_delta": f32_mean - scalar_mean,
            "std_delta": f(e, "std") - f(s, "std"),
        })

    mean_deltas = [d["mean_delta"] for d in deltas]
    std_deltas = [d["std_delta"] for d in deltas]
    var_inf = sum(int(float(r.get("var_inf_count") or 0)) for r in f32_rows)
    var_nan = sum(int(float(r.get("var_nan_count") or 0)) for r in f32_rows)
    var_neg = sum(int(float(r.get("var_negative_count") or 0)) for r in f32_rows)
    var_bad = sum(1 for r in f32_rows if r.get("var_bad_first_index") not in ("", "-1"))

    status = "PASS_COHORT_ANALYSIS_READY"
    reasons: list[str] = []
    if artifact.get("status") != "pass":
        status = "FAIL"
        reasons.append("f32_artifact_not_pass")
    if artifact.get("campaign", {}).get("limit") != "0":
        status = "FAIL"
        reasons.append("f32_artifact_not_limit_zero")
    if len(subjects) != 1034 or scalar_only or f32_only:
        status = "FAIL"
        reasons.append("subject_set_mismatch")
    if var_inf or var_nan or var_neg or var_bad:
        status = "FAIL"
        reasons.append("variance_diagnostics_not_clean")

    doc = {
        "schema": "sounio.moonshot_a.abide_f32_cohort_analysis.v1",
        "status": status,
        "reasons": reasons,
        "inputs": {
            "scalar_summary": str(scalar_path),
            "scalar_summary_sha256": sha256(scalar_path),
            "f32_summary": str(f32_path),
            "f32_summary_sha256": sha256(f32_path),
            "f32_artifact": str(artifact_path),
            "f32_artifact_sha256": sha256(artifact_path),
        },
        "cohort": {
            "joined_subjects": len(subjects),
            "scalar_only_subjects": scalar_only[:16],
            "f32_only_subjects": f32_only[:16],
            "label_counts": {
                label: sum(1 for r in f32_rows if r["label"] == label)
                for label in sorted({r["label"] for r in f32_rows})
            },
        },
        "scalar_vs_f32": {
            "mean_delta_mean": mean(mean_deltas),
            "mean_delta_abs_max": max(abs(x) for x in mean_deltas),
            "std_delta_mean": mean(std_deltas),
            "std_delta_abs_max": max(abs(x) for x in std_deltas),
            "mean_correlation": corr(scalar_means, f32_means),
            "scalar_mean_orc_cohen_d": cohen_d(scalar_rows, "mean"),
            "f32_mean_orc_cohen_d": cohen_d(f32_rows, "mean"),
        },
        "variance": {
            "inf_count": var_inf,
            "nan_count": var_nan,
            "negative_count": var_neg,
            "bad_first_index_count": var_bad,
            "max_output_var": artifact.get("variance", {}).get("max_output_var"),
        },
        "boundaries": [
            "descriptive_alignment_analysis_not_confirmatory_biomarker_claim",
            "f32_variance_is_sensitivity_diagnostic_not_calibrated_coverage",
            "cohort_scope_is_accepted_1034_subject_scalar_baseline_not_raw_1035_directory",
            "does_not_claim_clinical_utility_or_diagnosis",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        "moonshot_a_abide_f32_cohort_analysis: "
        f"{status} joined={len(subjects)} "
        f"mean_delta_abs_max={doc['scalar_vs_f32']['mean_delta_abs_max']:.6g} "
        f"out={out_path}"
    )
    return 0 if status == "PASS_COHORT_ANALYSIS_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
