#!/usr/bin/env bash
set -euo pipefail

# Audit the live/repaired AI-eval surface on Slurm. This is a promotion audit
# for the v14h routed system, not a new LoRA training rung.

SOUNIO_DIR="${SOUNIO_DIR:?set SOUNIO_DIR to the OrangeFS-staged Sounio repo copy}"
RUNG_NAME="${RUNG_NAME:-v14l_live_promotion_audit}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/orangefs/training/sounio-ai/${RUNG_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_ARTIFACT_ROOT="${BASE_ARTIFACT_ROOT:-/orangefs/training/sounio-ai/repair-v13-octo-ossm-hybrid-20260524T180926-1364094}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
JOB_GPUS="${JOB_GPUS:-1}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_TIME="${JOB_TIME:-01:30:00}"
JOB_CPUS="${JOB_CPUS:-4}"

V14H_DECISION="${V14H_DECISION:-${BASE_ARTIFACT_ROOT}/run-v14h-md2-router-from-v14c-v14f-20260525Tnext/v14h-router-decision.json}"
V14K_DECISION="${V14K_DECISION:-${BASE_ARTIFACT_ROOT}/run-v14k-repaired-spine-v14h-promotion-audit-20260525Tnext/v14k-promotion-audit-decision.json}"

mkdir -p "${ARTIFACT_ROOT}"

SLURM_JOB_NAME="${RUNG_NAME//_/-}"
SLURM_JOB_NAME="${SLURM_JOB_NAME:0:48}"

sbatch <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${SLURM_JOB_NAME}
#SBATCH -p ${SBATCH_PARTITION}
#SBATCH -A ${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:${JOB_GPUS}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=${JOB_CPUS}
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${ARTIFACT_ROOT}/slurm-%j.out
#SBATCH --error=${ARTIFACT_ROOT}/slurm-%j.err

set -euo pipefail

cd "${SOUNIO_DIR}"

python3 -m py_compile \\
  scripts/dev/batch_generate_sounio.py \\
  scripts/dev/build_sounio_ai_eval_prompts.py \\
  scripts/dev/build_sounio_ai_octonion_repair_dataset.py \\
  scripts/dev/build_sounio_ai_repair_dataset.py \\
  scripts/dev/build_sounio_ai_science_eval_slices.py \\
  scripts/dev/build_sounio_ai_science_spine_prompts.py \\
  scripts/dev/build_sounio_ai_selector_distill_dataset.py \\
  scripts/dev/build_sounio_ai_structure_hybrid_dataset.py \\
  scripts/dev/local_generate_sounio.py \\
  scripts/dev/sounio_ai_eval.py \\
  scripts/dev/sounio_ai_eval_failures.py \\
  scripts/dev/sounio_ai_eval_report.py \\
  scripts/dev/lora_finetune.py

bash -n slurm-jobs/sounio-ai-eval/submit-repair-v13-octo-ossm-hybrid.sh
bash -n slurm-jobs/sounio-ai-eval/submit-v14l-live-promotion-audit.sh

python3 scripts/dev/build_sounio_ai_science_eval_slices.py \\
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --output-dir "${ARTIFACT_ROOT}/slices" \\
  --prefix prompts.v2

python3 scripts/dev/build_sounio_ai_science_spine_prompts.py \\
  --output "${ARTIFACT_ROOT}/slices/prompts.science_spine_reference.jsonl"

python3 scripts/dev/sounio_ai_eval.py \\
  --prompts "${ARTIFACT_ROOT}/slices/prompts.science_spine_reference.jsonl" \\
  --output-dir "${ARTIFACT_ROOT}/eval/science-spine-reference-smoke" \\
  --model smoke=reference \\
  --use-reference-completions \\
  --samples-per-prompt 1 \\
  --run

python3 scripts/dev/sounio_ai_eval.py \\
  --prompts "${ARTIFACT_ROOT}/slices/prompts.v2.knowledge_octonion.jsonl" \\
  --output-dir "${ARTIFACT_ROOT}/eval/knowledge-octonion-reference-smoke" \\
  --model smoke=reference \\
  --use-reference-completions \\
  --samples-per-prompt 1 \\
  --run

python3 scripts/dev/sounio_ai_eval.py \\
  --prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --output-dir "${ARTIFACT_ROOT}/eval/prompts-v2-reference-smoke" \\
  --model smoke=reference \\
  --use-reference-completions \\
  --samples-per-prompt 1 \\
  --run

python3 scripts/dev/lora_finetune.py \\
  --dry-run \\
  --dataset-mode structure_hybrid \\
  --target-records 64 \\
  --heldout-prompts datasets/sounio-ai-eval/prompts.v2.jsonl \\
  --train-manifest "${ARTIFACT_ROOT}/structure-hybrid-dry-manifest.json" \\
  > "${ARTIFACT_ROOT}/structure-hybrid-dry-run.log"

python3 - <<'PY'
import json
from pathlib import Path

repo = Path.cwd()
artifact = Path("${ARTIFACT_ROOT}")
v14h_decision_path = Path("${V14H_DECISION}")
v14k_decision_path = Path("${V14K_DECISION}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_summary(root: Path) -> tuple[Path, dict]:
    summaries = sorted((root / "results").glob("*.summary.json"))
    if not summaries:
        raise SystemExit(f"no summary under {root / 'results'}")
    path = summaries[-1]
    return path, read_json(path)


def smoke(summary: dict) -> dict:
    return summary.get("smoke", summary)


def metric(summary: dict, key: str):
    return smoke(summary).get(key)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


v1 = read_jsonl(repo / "datasets/sounio-ai-eval/prompts.v1.jsonl")
v2 = read_jsonl(repo / "datasets/sounio-ai-eval/prompts.v2.jsonl")
v1_ids = {row.get("id") for row in v1}
v2_ids = {row.get("id") for row in v2}
v1_sources = {row.get("source_path") for row in v1 if row.get("source_path")}
v2_sources = {row.get("source_path") for row in v2 if row.get("source_path")}
overlap = {
    "v1_count": len(v1),
    "v2_count": len(v2),
    "id_overlap": len(v1_ids & v2_ids),
    "source_overlap": len(v1_sources & v2_sources),
    "reference_smoke_skip_count": sum(1 for row in v2 if "reference_smoke_skip" in row.get("tags", [])),
}

science_path, science_summary = latest_summary(artifact / "eval/science-spine-reference-smoke")
knowledge_path, knowledge_summary = latest_summary(artifact / "eval/knowledge-octonion-reference-smoke")
full_ref_path, full_ref_summary = latest_summary(artifact / "eval/prompts-v2-reference-smoke")
slice_manifest = read_json(artifact / "slices/prompts.v2.science-slices.manifest.json")
v14h_decision = read_json(v14h_decision_path)
v14k_decision = read_json(v14k_decision_path)

full_v2_reused = v14k_decision["metrics"]["full_v2"]

checks = {
    "v14h_router_passed": v14h_decision.get("status") == "pass",
    "v14k_prior_promotion_audit_passed": v14k_decision.get("status") == "pass",
    "full_v2_model_compile_at_5_guard_reused": full_v2_reused.get("compile_at_5", 0.0) >= 0.98,
    "full_v2_model_runtime_selector_guard_reused": full_v2_reused.get("runtime_pass_rate", 0.0) >= 0.70,
    "full_v2_model_raw_syntax_guard_reused": full_v2_reused.get("syntax_purity_rate", 0.0) >= 0.80,
    "v1_v2_id_overlap_zero": overlap["id_overlap"] == 0,
    "v1_v2_source_overlap_zero": overlap["source_overlap"] == 0,
    "knowledge_octonion_prompt_coverage": slice_manifest["slices"]["knowledge_octonion"]["count"] >= 4,
    "science_spine_reference_compile_guard": metric(science_summary, "compile_at_5") == 1.0,
    "science_spine_reference_runtime_guard": metric(science_summary, "runtime_pass_rate") == 1.0,
    "science_spine_reference_stdout_guard": metric(science_summary, "output_match_rate") == 1.0,
    "knowledge_octonion_compile_guard": metric(knowledge_summary, "compile_at_5") == 1.0,
    "knowledge_octonion_runtime_guard": metric(knowledge_summary, "runtime_pass_rate") == 1.0,
    "prompts_v2_reference_compile_guard": metric(full_ref_summary, "compile_at_5") == 1.0,
    "prompts_v2_reference_skip_policy_active": overlap["reference_smoke_skip_count"] == 4,
}

decision = {
    "rung": "v14l_live_promotion_audit",
    "status": "pass" if all(checks.values()) else "no_go",
    "claim_boundary": (
        "v14l audits the live/repaired repo surface staged on OrangeFS. It preserves "
        "the v14h/v14k claim boundary: composite routed system, not LoRA-alone MD2 correctness."
    ),
    "checks": checks,
    "metrics": {
        "full_v2_model_reused_from_v14k": full_v2_reused,
        "prompts_v2_reference": {
            "total": metric(full_ref_summary, "total"),
            "compile_at_1": metric(full_ref_summary, "compile_at_1"),
            "compile_at_3": metric(full_ref_summary, "compile_at_3"),
            "compile_at_5": metric(full_ref_summary, "compile_at_5"),
            "runtime_pass_rate": metric(full_ref_summary, "runtime_pass_rate"),
            "output_match_rate": metric(full_ref_summary, "output_match_rate"),
            "syntax_purity_rate": metric(full_ref_summary, "syntax_purity_rate"),
        },
        "science_spine_reference": {
            "total": metric(science_summary, "total"),
            "compile_at_5": metric(science_summary, "compile_at_5"),
            "runtime_pass_rate": metric(science_summary, "runtime_pass_rate"),
            "output_match_rate": metric(science_summary, "output_match_rate"),
        },
        "knowledge_octonion_reference": {
            "total": metric(knowledge_summary, "total"),
            "compile_at_5": metric(knowledge_summary, "compile_at_5"),
            "runtime_pass_rate": metric(knowledge_summary, "runtime_pass_rate"),
            "output_match_rate": metric(knowledge_summary, "output_match_rate"),
        },
        "overlap": overlap,
        "slice_manifest_counts": {
            name: row["count"] for name, row in slice_manifest["slices"].items()
        },
    },
    "inputs": {
        "repo": str(repo),
        "v14h_decision": str(v14h_decision_path),
        "v14k_decision": str(v14k_decision_path),
        "science_spine_reference_summary": str(science_path),
        "knowledge_octonion_summary": str(knowledge_path),
        "prompts_v2_reference_summary": str(full_ref_path),
        "structure_hybrid_dry_manifest": str(artifact / "structure-hybrid-dry-manifest.json"),
    },
}

out = artifact / "v14l-live-promotion-audit-decision.json"
out.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(decision, indent=2, sort_keys=True))
if decision["status"] != "pass":
    raise SystemExit(2)
PY

echo "artifact_root=${ARTIFACT_ROOT}"
SLURM
