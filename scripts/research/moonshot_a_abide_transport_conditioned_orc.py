#!/usr/bin/env python3
"""Moonshot A: bind ABIDE scalar/f32 ORC summaries to 168/ZD transport shifts."""

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


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def cohen_d_from_values(values: list[dict[str, float | str]], field: str) -> float | None:
    asd = [float(r[field]) for r in values if r["label"] == "ASD"]
    td = [float(r[field]) for r in values if r["label"] == "TD"]
    if not asd or not td:
        return None
    pooled = math.sqrt((pstdev(asd) ** 2 + pstdev(td) ** 2) / 2.0)
    return (mean(asd) - mean(td)) / pooled if pooled > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar-summary", default=str(DEFAULT_SCALAR))
    ap.add_argument("--f32-summary", required=True)
    ap.add_argument("--f32-artifact", required=True)
    ap.add_argument("--transport-artifact", required=True)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args()

    scalar_path = Path(args.scalar_summary)
    f32_path = Path(args.f32_summary)
    f32_artifact_path = Path(args.f32_artifact)
    transport_path = Path(args.transport_artifact)
    out_json = Path(args.out_json)
    out_tsv = Path(args.out_tsv)

    scalar_rows = read_rows(scalar_path)
    f32_rows = read_rows(f32_path)
    scalar_by_subject = {r["subject"]: r for r in scalar_rows}
    f32_by_subject = {r["subject"]: r for r in f32_rows}
    subjects = sorted(set(scalar_by_subject) & set(f32_by_subject))
    scalar_only = sorted(set(scalar_by_subject) - set(f32_by_subject))
    f32_only = sorted(set(f32_by_subject) - set(scalar_by_subject))

    f32_artifact = json.loads(f32_artifact_path.read_text())
    transport = json.loads(transport_path.read_text())
    baseline_kappa = float(transport["baseline"]["kappa_mean"])
    classes = []
    for row in transport["classes"]:
        if row.get("is_padding", False):
            continue
        shift = float(row["kappa_mean"]) - baseline_kappa
        classes.append({
            "class_index": int(row["class_index"]),
            "fiber_label": int(row["fiber_label"]),
            "transport_conditioned_orc_delta": shift,
            "abs_delta": abs(shift),
            "forgotten_indices": row.get("forgotten_indices", []),
            "primitives": row.get("primitives", []),
        })
    classes.sort(key=lambda r: (-float(r["abs_delta"]), int(r["class_index"])))
    selected = classes[: args.classes]

    feature_rows: list[dict[str, float | int | str]] = []
    scalar_means: list[float] = []
    f32_means: list[float] = []
    scalar_f32_deltas: list[float] = []
    for subject in subjects:
        s = scalar_by_subject[subject]
        e = f32_by_subject[subject]
        scalar_mean = f(s, "mean")
        f32_mean = f(e, "mean")
        scalar_f32_delta = f32_mean - scalar_mean
        scalar_means.append(scalar_mean)
        f32_means.append(f32_mean)
        scalar_f32_deltas.append(scalar_f32_delta)
        for cls in selected:
            transport_delta = float(cls["transport_conditioned_orc_delta"])
            feature_rows.append({
                "subject": subject,
                "label": e["label"],
                "site": e["site"],
                "class_168_probe_id": int(cls["class_index"]),
                "fiber_label": int(cls["fiber_label"]),
                "baseline_orc": scalar_mean,
                "f32_epistemic_orc": f32_mean,
                "scalar_f32_delta": scalar_f32_delta,
                "transport_conditioned_orc_delta": transport_delta,
                "transport_conditioned_orc": f32_mean + transport_delta,
                "residual_vs_scalar_f32_delta": transport_delta - scalar_f32_delta,
                "variance_bound": f32_artifact.get("variance", {}).get("max_output_var"),
                "claim_scope": "descriptive_transport_conditioned_feature_not_clinical_or_diagnostic",
            })

    transport_abs = [float(c["abs_delta"]) for c in selected]
    mean_abs_scalar_f32_delta = mean(abs(x) for x in scalar_f32_deltas)
    max_abs_scalar_f32_delta = max(abs(x) for x in scalar_f32_deltas)
    status = "PASS_TRANSPORT_CONDITIONED_ORC_READY"
    reasons: list[str] = []
    if f32_artifact.get("status") != "pass":
        status = "FAIL"
        reasons.append("f32_artifact_not_pass")
    if f32_artifact.get("campaign", {}).get("limit") != "0":
        status = "FAIL"
        reasons.append("f32_artifact_not_limit_zero")
    if len(subjects) != 1034 or scalar_only or f32_only:
        status = "FAIL"
        reasons.append("subject_set_mismatch")
    if len(selected) != args.classes:
        status = "FAIL"
        reasons.append("selected_class_count_mismatch")
    if max(transport_abs or [0.0]) <= 1e-5:
        status = "FAIL"
        reasons.append("transport_delta_collapsed_to_zero")
    if max(transport_abs or [0.0]) <= mean_abs_scalar_f32_delta:
        status = "FAIL"
        reasons.append("transport_delta_not_above_scalar_f32_mean_abs_drift")
    if transport.get("zd_census", {}).get("runtime_slots_evaluated") != 168:
        status = "FAIL"
        reasons.append("transport_runtime_slots_not_168")
    if transport.get("modulation_mode") != "subspace":
        status = "FAIL"
        reasons.append("transport_artifact_not_subspace_mode")

    cols = [
        "subject",
        "label",
        "site",
        "class_168_probe_id",
        "fiber_label",
        "baseline_orc",
        "f32_epistemic_orc",
        "scalar_f32_delta",
        "transport_conditioned_orc_delta",
        "transport_conditioned_orc",
        "residual_vs_scalar_f32_delta",
        "variance_bound",
        "claim_scope",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=cols, delimiter="\t")
        writer.writeheader()
        writer.writerows(feature_rows)

    by_subject_best = []
    # The selected class shifts are global transport-prototype features, so the
    # per-subject best absolute residual is a compact sanity summary.
    for subject in subjects:
        rows = [r for r in feature_rows if r["subject"] == subject]
        best = max(rows, key=lambda r: abs(float(r["residual_vs_scalar_f32_delta"])))
        by_subject_best.append(best)

    doc = {
        "schema": "sounio.moonshot_a.abide_transport_conditioned_orc.v1",
        "status": status,
        "reasons": reasons,
        "inputs": {
            "scalar_summary": str(scalar_path),
            "scalar_summary_sha256": sha256(scalar_path),
            "f32_summary": str(f32_path),
            "f32_summary_sha256": sha256(f32_path),
            "f32_artifact": str(f32_artifact_path),
            "f32_artifact_sha256": sha256(f32_artifact_path),
            "transport_artifact": str(transport_path),
            "transport_artifact_sha256": sha256(transport_path),
            "feature_tsv": str(out_tsv),
            "feature_tsv_sha256": sha256(out_tsv),
        },
        "cohort": {
            "joined_subjects": len(subjects),
            "scalar_only_subjects": scalar_only[:16],
            "f32_only_subjects": f32_only[:16],
            "selected_classes": len(selected),
            "feature_rows": len(feature_rows),
            "label_counts": {
                label: sum(1 for r in f32_rows if r["label"] == label)
                for label in sorted({r["label"] for r in f32_rows})
            },
        },
        "scalar_vs_f32": {
            "mean_delta_mean": mean(scalar_f32_deltas),
            "mean_delta_abs_mean": mean_abs_scalar_f32_delta,
            "mean_delta_abs_max": max_abs_scalar_f32_delta,
            "mean_correlation": corr(scalar_means, f32_means),
        },
        "transport_conditioned": {
            "selected_class_indices": [int(c["class_index"]) for c in selected],
            "selected_class_fiber_labels": [int(c["fiber_label"]) for c in selected],
            "delta_abs_max": max(transport_abs),
            "delta_abs_mean": mean(transport_abs),
            "delta_abs_max_vs_scalar_f32_mean_abs_ratio": max(transport_abs) / mean_abs_scalar_f32_delta,
            "delta_abs_max_vs_scalar_f32_max_abs_ratio": max(transport_abs) / max_abs_scalar_f32_delta,
            "transport_conditioned_orc_cohen_d": cohen_d_from_values(by_subject_best, "transport_conditioned_orc"),
            "residual_vs_scalar_f32_delta_abs_mean": mean(
                abs(float(r["residual_vs_scalar_f32_delta"])) for r in feature_rows
            ),
        },
        "variance": {
            "variance_bound": f32_artifact.get("variance", {}).get("max_output_var"),
            "interpretation": "finite_overflow_bound_not_calibrated_coverage",
        },
        "boundaries": [
            "descriptive_transport_conditioned_feature_package_not_confirmatory_biomarker_claim",
            "uses_transport_168_runtime_prototype_not_final_zd_surgery_semantics",
            "cohort_scope_is_accepted_1034_subject_scalar_baseline_not_raw_1035_directory",
            "f32_variance_is_sensitivity_diagnostic_not_calibrated_coverage",
            "does_not_claim_clinical_utility_diagnosis_or_external_validation",
            "transport_delta_is_global_class_probe_feature_not_subject_specific_cuda_rerun",
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(
        "moonshot_a_abide_transport_conditioned_orc: "
        f"{status} joined={len(subjects)} rows={len(feature_rows)} "
        f"selected_classes={len(selected)} "
        f"transport_ratio={doc['transport_conditioned']['delta_abs_max_vs_scalar_f32_mean_abs_ratio']:.6g} "
        f"out={out_json}"
    )
    return 0 if status == "PASS_TRANSPORT_CONDITIONED_ORC_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
