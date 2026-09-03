#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_PHASE_STATUS_DIR:-$(mktemp -d /tmp/moonshot-a-phase-status.XXXXXX)}"
mkdir -p "$OUT_DIR"

ABIDE_DIR="$OUT_DIR/abide_epistemic_orc_slice"
MANIFEST_DIR="$OUT_DIR/transport_168_manifest"

SOUNIO_MOONSHOT_A_ABIDE_EPI_SLICE_DIR="$ABIDE_DIR" bash scripts/ci/moonshot_a_abide_epistemic_orc_slice_gate.sh >/tmp/moonshot_a_abide_epistemic_orc_slice_gate.out
SOUNIO_MOONSHOT_A_TRANSPORT_168_MANIFEST_DIR="$MANIFEST_DIR" bash scripts/ci/moonshot_a_transport_168_manifest_gate.sh >/tmp/moonshot_a_transport_168_manifest_gate.out

if [[ "${SOUNIO_MOONSHOT_A_RUN_SLURM:-0}" == "1" ]]; then
  SCALAR_DIR="$OUT_DIR/slurm_scalar"
  EPI_DIR="$OUT_DIR/slurm_f32_epistemic"
  set +e
  SOUNIO_KAXI_SINKHORN16_SLURM_DIR="$SCALAR_DIR" bash scripts/ci/kretikos_kaxi_sinkhorn16_slurm_gate.sh >/tmp/kretikos_kaxi_sinkhorn16_slurm_gate.out
  scalar_slurm_rc=$?
  SOUNIO_KAXI_SINKHORN16_EPI_SLURM_DIR="$EPI_DIR" bash scripts/ci/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.sh >/tmp/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.out
  epi_slurm_rc=$?
  set -e
  export SOUNIO_MOONSHOT_A_SLURM_GATE_RCS="scalar=$scalar_slurm_rc;f32_epistemic=$epi_slurm_rc"
  export SOUNIO_MOONSHOT_A_SCALAR_SLURM_ARTIFACT="$SCALAR_DIR/kretikos_kaxi_sinkhorn16_slurm_gate.v1.json"
  export SOUNIO_MOONSHOT_A_EPI_SLURM_ARTIFACT="$EPI_DIR/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.v1.json"
fi

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

base = pathlib.Path(sys.argv[1])

def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def artifact(kind: str, path: pathlib.Path) -> dict:
    return {"kind": kind, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}

def load_optional(path: pathlib.Path):
    if path.exists():
        data = json.loads(path.read_text())
        data["path"] = str(path)
        data["sha256"] = sha(path)
        return data
    return None

artifacts = [
    artifact("abide_epistemic_orc_slice.synthetic", base / "abide_epistemic_orc_slice" / "synthetic.json"),
    artifact("transport_168.manifest", base / "transport_168_manifest" / "moonshot_a_transport_168_manifest.v1.json"),
]
real_abide = base / "abide_epistemic_orc_slice" / "real_abide.json"
if real_abide.exists():
    artifacts.append(artifact("abide_epistemic_orc_slice.real", real_abide))

scalar_env = os.environ.get("SOUNIO_MOONSHOT_A_SCALAR_SLURM_ARTIFACT", "")
epi_env = os.environ.get("SOUNIO_MOONSHOT_A_EPI_SLURM_ARTIFACT", "")
scalar_path = pathlib.Path(scalar_env) if scalar_env else None
epi_path = pathlib.Path(epi_env) if epi_env else None
if scalar_path is None:
    found = sorted(pathlib.Path("/tmp").glob("kaxi-sinkhorn16-slurm.*/kretikos_kaxi_sinkhorn16_slurm_gate.v1.json"), key=lambda p: p.stat().st_mtime)
    scalar_path = found[-1] if found else None
if epi_path is None:
    found = sorted(pathlib.Path("/tmp").glob("kaxi-sinkhorn16-epi-slurm.*/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.v1.json"), key=lambda p: p.stat().st_mtime)
    epi_path = found[-1] if found else None

scalar = load_optional(scalar_path) if scalar_path is not None else None
epi = load_optional(epi_path) if epi_path is not None else None
if scalar:
    artifacts.append(artifact("slurm_runtime.scalar", pathlib.Path(scalar["path"])))
if epi:
    artifacts.append(artifact("slurm_runtime.f32_epistemic", pathlib.Path(epi["path"])))

gpu_pass = bool(scalar and epi and scalar.get("status") == "pass" and epi.get("status") == "pass")
if gpu_pass:
    gpu_status = "pass"
    reason = "scalar=pass;f32_epistemic=pass"
else:
    present = [x for x in (scalar, epi) if x]
    gpu_status = "artifacts_present_not_passing" if present else "not_run"
    bits = []
    for label, obj in (("scalar", scalar), ("f32_epistemic", epi)):
        if not obj:
            bits.append(f"{label}=missing")
        else:
            bits.append(f"{label}={obj.get('status','unknown')}:{obj.get('runtime',{}).get('reason','')}")
    reason = ";".join(bits)

abide = json.loads((base / "abide_epistemic_orc_slice" / "synthetic.json").read_text())
if real_abide.exists():
    real = json.loads(real_abide.read_text())
    first = real["edges"][0]
    abide_summary = {
        "subject": real["subject"],
        "edge": first["edge"],
        "kappa_mean": first["kappa_mean"],
        "diagonal_w1_sigma": first["bands"]["diagonal"]["w1_sigma"],
        "jacobian_w1_sigma": first["bands"]["jacobian"]["w1_sigma"],
        "jacobian_vs_diagonal_w1_sigma_ratio": first["jacobian_vs_diagonal"]["w1_sigma_ratio"],
    }
    real_status = "emitted"
else:
    first = abide["edges"][0]
    abide_summary = {"subject": abide["subject"], "edge": first["edge"], "kappa_mean": first["kappa_mean"]}
    real_status = "skipped_no_local_abide"

manifest = json.loads((base / "transport_168_manifest" / "moonshot_a_transport_168_manifest.v1.json").read_text())
status = "PASS_FULL" if gpu_pass else "PASS_LOCAL_GPU_PENDING"
boundaries = [
    "local_python_evidence_passed",
    "transport_168_manifest_passed",
    "does_not_claim_full_abide_cohort_revalidation",
    "does_not_claim_final_zd_surgery_semantics",
    "slurm_runtime_can_be_consumed_from_explicit_artifacts",
    "slurm_runtime_can_be_auto_discovered_from_tmp_artifacts",
]
if gpu_pass:
    boundaries.extend([
        "cuda_runtime_acceptance_limited_to_sinkhorn16_scalar_and_f32_epistemic_worker_json",
        "f32_epistemic_variance_is_finite_overflow_bound_not_calibrated_coverage",
    ])
else:
    boundaries.extend([
        "does_not_claim_cuda_runtime_acceptance",
        "phase_completion_still_requires_slurm_cuda_runtime_gates",
    ])
doc = {
    "schema": "sounio.moonshot_a.phase_status.v1",
    "status": status,
    "out_dir": str(base),
    "artifact_count": len(artifacts),
    "artifacts": artifacts,
    "local_evidence": {
        "status": "pass",
        "abide_epistemic_orc_slice": {
            "script": "scripts/ci/moonshot_a_abide_epistemic_orc_slice_gate.sh",
            "real_status": real_status,
            "real_summary": abide_summary,
        },
        "transport_168": {
            "script": "scripts/ci/moonshot_a_transport_168_manifest_gate.sh",
            "real_status": manifest["real_status"],
            "manifest_sha256": sha(base / "transport_168_manifest" / "moonshot_a_transport_168_manifest.v1.json"),
            "synthetic_shift_summary": manifest["summary"]["synthetic_shift_summary"],
        },
    },
    "gpu_runtime": {
        "status": gpu_status,
        "reason": reason,
        "slurm_mode": "auto_discovered_artifacts",
        "required_for_full_completion": [
            "scripts/ci/kretikos_kaxi_sinkhorn16_slurm_gate.sh",
            "scripts/ci/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.sh",
        ],
        "scalar_slurm": scalar,
        "f32_epistemic_slurm": epi,
        "blocker": {
            "id": "BLOCKER-MOONSHOT-A-SLURM-RESOURCES",
            "severity": "B3",
            "class": "platform-resource",
            "evidence_level": "E3",
            "owner": "cluster/Slurm availability",
            "acceptance_gate": "SOUNIO_MOONSHOT_A_RUN_SLURM=1 bash scripts/ci/moonshot_a_phase_status_gate.sh",
            "next_action": "rerun scalar and f32-epistemic Slurm runtime gates when CUDA worker resources are available",
        },
    },
    "boundaries": boundaries,
}
out = base / "moonshot_a_phase_status.v1.json"
out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(f"moonshot_a_phase_status_gate: {status} out={out} gpu_runtime_status={gpu_status} reason={reason}")
PY

PHASE_JSON="$OUT_DIR/moonshot_a_phase_status.v1.json"
COHORT_SUMMARY="artifacts/research/abide_orc/cohort_summary.tsv"
if [[ -f "$COHORT_SUMMARY" ]]; then
  COHORT_DIR="$OUT_DIR/abide_cohort_manifest"
  COHORT_JSON="$COHORT_DIR/moonshot_a_abide_cohort_manifest.v1.json"
  mkdir -p "$COHORT_DIR"
  set +e
  python3 scripts/research/moonshot_a_abide_cohort_manifest.py \
    --phase-status "$PHASE_JSON" \
    --out "$COHORT_JSON" \
    >/tmp/moonshot_a_abide_cohort_manifest.out \
    2>/tmp/moonshot_a_abide_cohort_manifest.err
  cohort_rc=$?
  set -e
  if [[ "$cohort_rc" -eq 0 ]]; then
    python3 - "$PHASE_JSON" "$COHORT_JSON" <<'PY'
import hashlib
import json
import pathlib
import sys

phase_path = pathlib.Path(sys.argv[1])
cohort_path = pathlib.Path(sys.argv[2])
phase = json.loads(phase_path.read_text())
cohort = json.loads(cohort_path.read_text())
artifact = {
    "kind": "abide_cohort.baseline_manifest",
    "path": str(cohort_path),
    "bytes": cohort_path.stat().st_size,
    "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
}
phase.setdefault("artifacts", []).append(artifact)
phase["artifact_count"] = len(phase["artifacts"])
phase.setdefault("local_evidence", {})["abide_cohort_baseline"] = {
    "script": "scripts/ci/moonshot_a_abide_cohort_manifest_gate.sh",
    "status": cohort.get("status", ""),
    "summary_subjects": cohort.get("cohort_baseline", {}).get("summary_subjects", 0),
    "labelled_subjects": cohort.get("cohort_baseline", {}).get("labelled_subjects", 0),
    "mean_orc_cohen_d": cohort.get("cohort_baseline", {}).get("asd_td_screen", {}).get("mean_orc_cohen_d", ""),
    "boundaries": cohort.get("boundaries", []),
}
if "abide_cohort_baseline_bound_to_pass_full_runtime" not in phase.setdefault("boundaries", []):
    phase["boundaries"].append("abide_cohort_baseline_bound_to_pass_full_runtime")
phase_path.write_text(json.dumps(phase, indent=2, sort_keys=True) + "\n")
print(
    "moonshot_a_phase_status_gate: cohort_baseline="
    f"{cohort.get('status')} artifact={cohort_path}"
)
PY
  else
    echo "moonshot_a_phase_status_gate: cohort_baseline=SKIPPED rc=$cohort_rc see /tmp/moonshot_a_abide_cohort_manifest.err" >&2
  fi
fi
