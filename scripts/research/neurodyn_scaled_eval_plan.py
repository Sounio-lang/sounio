#!/usr/bin/env python3
"""Create a claim-safe next-wave evaluation plan for NeuroDyn O-SSM.

This script does not submit jobs. It turns the current P3 receipts into an
auditable scale plan with concrete gates for a larger MDD/ADHD-style evaluation:
manifest quality, O-SSM vs baselines, label/temporal controls, checkpoint replay,
trained associator extraction, and promotion boundaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

from abide_campaign_lib import load_manifest


SCHEMA = "neurodyn.scaled_eval_plan.v1"
CLAIM_BOUNDARY = (
    "Planning/operations only. No positive clinical, mechanistic, biomarker, "
    "or replicated O-SSM claim is authorized by this plan."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Current manifest used by the verified smoke.")
    parser.add_argument("--checkpoint", required=True, help="Verified trained checkpoint from the smoke.")
    parser.add_argument("--metric-matrix-dir", required=True, help="Trained associator metric matrix receipt directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the scale plan receipt.")
    parser.add_argument("--dataset-id", default="ds006731")
    parser.add_argument("--label-space", default="mdd_vs_control")
    parser.add_argument("--target-subjects", type=int, default=28)
    parser.add_argument("--target-per-class", type=int, default=14)
    parser.add_argument("--target-min-sites", type=int, default=4)
    parser.add_argument("--native-souc-engine", default="lean_single")
    parser.add_argument("--cluster-worker", default="slurm-pilot-worker-gpuorangefs-multi-mpttj")
    parser.add_argument("--durable-root", default="/mnt/cephfs/neurodyn/derivatives/ds006731")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_metric_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manifest_summary(manifest: Path) -> dict[str, Any]:
    meta, records = load_manifest(manifest)
    label_counts: dict[str, int] = {}
    site_counts: dict[str, int] = {}
    site_label_counts: dict[str, dict[str, int]] = {}
    for record in records:
        label = str(record.label)
        label_counts[label] = label_counts.get(label, 0) + 1
        site_counts[record.site] = site_counts.get(record.site, 0) + 1
        site_label_counts.setdefault(record.site, {"0": 0, "1": 0})
        site_label_counts[record.site][label] = site_label_counts[record.site].get(label, 0) + 1
    return {
        "path": str(manifest.resolve()),
        "schema": meta.schema,
        "subject_count": len(records),
        "label_counts": label_counts,
        "site_count": len(site_counts),
        "site_counts": site_counts,
        "site_label_counts": site_label_counts,
        "seq_len": meta.seq_len,
        "input_dim": meta.input_dim,
        "feature_count": meta.seq_len * meta.input_dim,
        "label_space": meta.label_space,
        "extra": meta.extra,
    }


def _array_values(checkpoint: dict[str, Any], name: str) -> list[float]:
    raw = checkpoint.get("arrays", {}).get(name, {}).get("values", [])
    return [float(value) for value in raw]


def _abs_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "nonzero_count": 0, "l1": 0.0, "linf": 0.0, "mean_abs": 0.0}
    abs_values = [abs(value) for value in values]
    return {
        "count": len(values),
        "nonzero_count": sum(1 for value in abs_values if value > 1.0e-12),
        "l1": sum(abs_values),
        "linf": max(abs_values),
        "mean_abs": sum(abs_values) / len(abs_values),
    }


def octonionic_nontriviality(checkpoint: dict[str, Any], metric_rows: list[dict[str, str]]) -> dict[str, Any]:
    arrays = checkpoint.get("arrays", {})
    a = _array_values(checkpoint, "A")
    a_fixed = _array_values(checkpoint, "A_FIXED")
    last_assoc = _array_values(checkpoint, "LAST_ASSOC_ACC")
    last_h = _array_values(checkpoint, "LAST_H")
    top_rows = []
    for row in metric_rows:
        top_rows.append(
            {
                "metric": row.get("metric", ""),
                "top1_roi": row.get("top1_roi", ""),
                "top1_p_raw": row.get("top1_p_raw", ""),
                "top1_p_holm": row.get("top1_p_holm", ""),
                "top1_cliffs_delta": row.get("top1_cliffs_delta", ""),
                "holm_survivors": row.get("holm_survivors", ""),
            }
        )
    return {
        "arrays_present": sorted(arrays),
        "a_summary": _abs_summary(a),
        "a_fixed_summary": _abs_summary(a_fixed),
        "last_assoc_acc_summary": _abs_summary(last_assoc),
        "last_h_summary": _abs_summary(last_h),
        "metric_top1": top_rows,
        "nontrivial_result_definition": [
            "trained octonionic state is nonzero and replayable in a fresh process",
            "associator/hidden trajectory signal is stable across folds/seeds, not only a single saturated readout",
            "effect survives comparative baselines and disappears or changes appropriately under label/temporal controls",
            "ROI or mechanistic interpretation is corrected for multiplicity and robust across preregistered metric family",
        ],
    }


def gate_rows(current: dict[str, Any], checkpoint: dict[str, Any], metric_rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    label_counts = current["label_counts"]
    min_class = min(int(label_counts.get("0", 0)), int(label_counts.get("1", 0)))
    holm_survivors = sum(int(float(row.get("holm_survivors") or 0)) for row in metric_rows)
    verification = checkpoint.get("verification") or {}
    octo = octonionic_nontriviality(checkpoint, metric_rows)
    a_signal = octo["a_summary"]["nonzero_count"] > 0 or octo["a_fixed_summary"]["nonzero_count"] > 0
    assoc_signal = octo["last_assoc_acc_summary"]["nonzero_count"] > 0
    hidden_signal = octo["last_h_summary"]["nonzero_count"] > 0
    return [
        {
            "gate": "manifest_scale",
            "current_status": "blocked" if current["subject_count"] < args.target_subjects else "pass",
            "current_value": current["subject_count"],
            "target": args.target_subjects,
            "why": "Avoid interpreting n=16 smoke as replicated clinical evidence.",
        },
        {
            "gate": "class_balance",
            "current_status": "blocked" if min_class < args.target_per_class else "pass",
            "current_value": min_class,
            "target": args.target_per_class,
            "why": "Nested folds need enough positives and controls after holdout.",
        },
        {
            "gate": "site_or_pseudofold_balance",
            "current_status": "pass" if current["site_count"] >= args.target_min_sites else "blocked",
            "current_value": current["site_count"],
            "target": args.target_min_sites,
            "why": "Grouped CV requires at least four balanced folds/sites.",
        },
        {
            "gate": "checkpoint_replay",
            "current_status": "pass" if verification.get("status") == "pass" else "blocked",
            "current_value": f"{verification.get('actual_balanced_accuracy_pct')} delta={verification.get('delta_pp')}",
            "target": "status=pass and delta<=0.5pp for every promoted fold",
            "why": "No associator extraction from model-free or non-replayable state.",
        },
        {
            "gate": "octonionic_state_nonzero",
            "current_status": "pass" if a_signal and hidden_signal else "blocked",
            "current_value": f"A_nonzero={octo['a_summary']['nonzero_count']} H_nonzero={octo['last_h_summary']['nonzero_count']}",
            "target": "trained A/H state nonzero for every promoted checkpoint",
            "why": "An octonionic state-space result must use trained octonionic state, not model-free scaffolding.",
        },
        {
            "gate": "associator_activity",
            "current_status": "pass" if assoc_signal else "blocked",
            "current_value": f"LAST_ASSOC_ACC_nonzero={octo['last_assoc_acc_summary']['nonzero_count']}",
            "target": "nonzero associator activity plus control-specific behavior at scale",
            "why": "Non-associative dynamics are the point of O-SSM; BA alone is not enough.",
        },
        {
            "gate": "metric_robustness",
            "current_status": "negative_control_pass",
            "current_value": f"holm_survivors={holm_survivors}",
            "target": "no biomarker claim unless corrected and robust across preregistered metric family",
            "why": "Current smoke has no Holm-corrected ROI survivor.",
        },
        {
            "gate": "baseline_superiority",
            "current_status": "missing_at_scale",
            "current_value": "H-SSM exceeded O-SSM in current smoke",
            "target": "O-SSM improvement over H-SSM and external baselines with controls",
            "why": "Utility claim must be comparative, not a solo accuracy number.",
        },
        {
            "gate": "null_controls",
            "current_status": "planned",
            "current_value": "label_shuffle + temporal_shuffle/reverse required",
            "target": "positive result must collapse under label shuffle and behave specifically under temporal controls",
            "why": "Separates real dynamics from leakage or arbitrary separability.",
        },
        {
            "gate": "octonionic_nontriviality_at_scale",
            "current_status": "planned",
            "current_value": "requires fold/seed distribution of associator and hidden trajectory diagnostics",
            "target": "nontrivial O-SSM effect is algebraically active, replayable, control-specific, and comparatively useful",
            "why": "Prevents missing a real octonionic state-space effect or overclaiming a trivial classifier artifact.",
        },
    ]


def shell_plan(args: argparse.Namespace) -> list[str]:
    root = args.durable_root.rstrip("/")
    return [
        "# 1. Expand durable ROI manifest, then build pseudo-fold and control manifests.",
        f"python3 scripts/research/neurodyn_ds006731_p3_manifest.py --survivor-root {root}/p3_candidates --rerun-root {root}/rerun_YYYYMMDDTHHMMSSZ --out-root {root}/p3_candidates --site-policy pseudo_fold4 --roi-buckets 8",
        "# 2. Run manifest quality gate before any training.",
        "python3 scripts/research/abide_manifest_quality_gate.py --manifest ${RUN_MANIFEST}",
        "# 3. Run O-SSM campaign profile only; require checkpoint persistence and replay.",
        "RUN_ID=neurodyn-ds006731-scaled-o-assoc-stable-${STAMP} SKIP_EXTERNAL_BASELINES=1 STOP_AFTER_CHECKPOINT=0 ABIDE_MANIFEST_PATH=${RUN_MANIFEST} bash slurm-jobs/brain-ossm/submit-abide-campaign-gpu.sh",
        "# 4. Run external baselines on the same manifest and folds.",
        "ABIDE_MANIFEST_PATH=${RUN_MANIFEST} bash slurm-jobs/brain-ossm/submit-abide-external-baselines-gpu.sh",
        "# 5. Gate trained associator only after checkpoint replay passes.",
        f"python3 scripts/research/neurodyn_postcampaign_trained_associator_gate.py --manifest ${{RUN_MANIFEST}} --checkpoint ${{MODEL_CKPT}} --output-dir ${{ASSOC_OUT}} --native-souc-engine {args.native_souc_engine} --execute-associator",
    ]


def markdown(plan: dict[str, Any], gates: list[dict[str, Any]]) -> str:
    lines = [
        "# NeuroDyn Scaled Evaluation Plan",
        "",
        f"- created_at_utc: `{plan['created_at_utc']}`",
        f"- dataset_id: `{plan['dataset_id']}`",
        f"- label_space: `{plan['label_space']}`",
        f"- current_manifest_subjects: `{plan['current_manifest']['subject_count']}`",
        f"- current_label_counts: `{plan['current_manifest']['label_counts']}`",
        f"- current_site_count: `{plan['current_manifest']['site_count']}`",
        f"- checkpoint_run_id: `{plan['checkpoint'].get('run_id')}`",
        f"- checkpoint_replay_status: `{plan['checkpoint'].get('verification', {}).get('status')}`",
        f"- claim_boundary: {CLAIM_BOUNDARY}",
        "",
        "## Octonionic Nontriviality",
        "",
        f"- A_nonzero_count: `{plan['octonionic_nontriviality']['a_summary']['nonzero_count']}`",
        f"- A_linf: `{plan['octonionic_nontriviality']['a_summary']['linf']}`",
        f"- A_FIXED_nonzero_count: `{plan['octonionic_nontriviality']['a_fixed_summary']['nonzero_count']}`",
        f"- LAST_H_nonzero_count: `{plan['octonionic_nontriviality']['last_h_summary']['nonzero_count']}`",
        f"- LAST_ASSOC_ACC_nonzero_count: `{plan['octonionic_nontriviality']['last_assoc_acc_summary']['nonzero_count']}`",
        "- nontrivial_result_definition:",
    ]
    lines.extend(f"  - {item}" for item in plan["octonionic_nontriviality"]["nontrivial_result_definition"])
    lines.extend(
        [
            "",
        "## Gates",
        "",
        "| gate | status | current | target |",
        "|---|---|---:|---|",
        ]
    )
    for row in gates:
        lines.append(f"| `{row['gate']}` | `{row['current_status']}` | `{row['current_value']}` | {row['target']} |")
    lines.extend(
        [
            "",
            "## Execution Sketch",
            "",
        ]
    )
    lines.extend(f"```bash\n{cmd}\n```" for cmd in plan["shell_plan"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    metric_matrix_dir = Path(args.metric_matrix_dir).resolve()

    checkpoint = read_json(checkpoint_path)
    metric_summary = read_metric_summary(metric_matrix_dir / "metric_matrix_summary.tsv")
    current = manifest_summary(manifest)
    gates = gate_rows(current, checkpoint, metric_summary, args)
    plan = {
        "schema": SCHEMA,
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "dataset_id": args.dataset_id,
        "label_space": args.label_space,
        "current_manifest": current,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": {
            "run_id": checkpoint.get("run_id"),
            "meta": checkpoint.get("meta"),
            "verification": checkpoint.get("verification"),
        },
        "metric_matrix_dir": str(metric_matrix_dir),
        "metric_summary": metric_summary,
        "octonionic_nontriviality": octonionic_nontriviality(checkpoint, metric_summary),
        "target": {
            "subjects": args.target_subjects,
            "per_class": args.target_per_class,
            "min_sites": args.target_min_sites,
            "native_souc_engine": args.native_souc_engine,
            "cluster_worker": args.cluster_worker,
            "durable_root": args.durable_root,
        },
        "promotion_rule": {
            "required": [
                "scaled manifest meets target subject/class/site gates",
                "O-SSM checkpoint replay delta <= 0.5pp for every promoted fold",
                "O-SSM compared against H-SSM and external baselines on identical manifest/folds",
                "label shuffle fails to reproduce the positive effect",
                "temporal controls behave specifically for the claimed dynamic effect",
                "trained associator uses verified checkpoints only",
                "octonionic state/associator diagnostics are nonzero, replayable, and control-specific",
            ],
            "disallowed": [
                "positive clinical claim from n=16 smoke",
                "ROI biomarker claim without Holm-corrected and robust evidence",
                "associator extraction from untrained or non-replayable state",
                "calling a plain classification delta an octonionic result without A/H/associator diagnostics",
            ],
        },
        "shell_plan": shell_plan(args),
        "claim_boundary": CLAIM_BOUNDARY,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "scaled_eval_plan.json", plan)
    write_tsv(output_dir / "scaled_eval_gates.tsv", gates)
    (output_dir / "scaled_eval_plan.md").write_text(markdown(plan, gates), encoding="utf-8")
    print(f"plan={output_dir / 'scaled_eval_plan.json'}")
    print(f"gates={output_dir / 'scaled_eval_gates.tsv'}")
    print(f"markdown={output_dir / 'scaled_eval_plan.md'}")
    blocked = [row["gate"] for row in gates if row["current_status"] in {"blocked", "missing_at_scale"}]
    print("blocked_gates=" + ",".join(blocked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
