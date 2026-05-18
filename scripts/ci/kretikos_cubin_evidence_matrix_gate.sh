#!/usr/bin/env bash
# Matrix gate for evidence-carrying CUBIN artifacts.

set -euo pipefail

ulimit -s unlimited 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_MATRIX_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cubin_evidence_matrix}"
SUMMARY_JSON="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_MATRIX_JSON:-$OUT_ROOT/kretikos_cubin_evidence_matrix.v1.json}"
mkdir -p "$OUT_ROOT"

CASES=(
  "vec_add_f32|examples/kretikos/real_vec_add.sio|source_vec_add_f32|vec_add|vec_add_f32|vec_add_f32"
  "vec_sub_f32|examples/kretikos/real_vec_sub.sio|source_vec_sub_f32|vec_sub|vec_sub_f32|vec_sub_f32"
  "vec_mul_f32|examples/kretikos/real_vec_mul.sio|source_vec_mul_f32|vec_mul|vec_mul_f32|vec_mul_f32"
  "vec_div_f32|examples/kretikos/real_vec_div.sio|source_vec_div_f32|vec_div|vec_div_f32|vec_div_f32"
  "fma_f32|examples/kretikos/real_fma_f32.sio|source_fma_f32|fma|fma_f32|fma_f32"
  "vec_add_f64|examples/kretikos/real_vec_add_f64.sio|source_vec_add_f64|vec_add_f64|vec_add_f64|vec_add_f64"
  "vec_sub_f64|examples/kretikos/real_vec_sub_f64.sio|source_vec_sub_f64|vec_sub_f64|vec_sub_f64|vec_sub_f64"
  "vec_mul_f64|examples/kretikos/real_vec_mul_f64.sio|source_vec_mul_f64|vec_mul_f64|vec_mul_f64|vec_mul_f64"
  "vec_div_f64|examples/kretikos/real_vec_div_f64.sio|source_vec_div_f64|vec_div_f64|vec_div_f64|vec_div_f64"
  "fma_f64|examples/kretikos/real_fma_f64.sio|source_fma_f64|fma_f64|fma_f64|fma_f64"
  "epistemic_dual_output_f32|examples/kretikos/real_epistemic_dual_output.sio|source_epistemic_dual_output_f32|epistemic_dual_output_f32|epistemic_dual_f32|epistemic_dual_output_f32"
)

reports=()
for row in "${CASES[@]}"; do
  IFS='|' read -r label source kaxi_pattern ptx_pattern cubin_kind runtime_rung <<<"$row"
  case_dir="$OUT_ROOT/$label"
  case_json="$case_dir/kretikos_cubin_evidence_gate.v1.json"
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_SOURCE="$source" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_KAXI_PATTERN="$kaxi_pattern" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_PTX_PATTERN="$ptx_pattern" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_CUBIN_KIND="$cubin_kind" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_RUNTIME_RUNG="$runtime_rung" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_SM_TARGET="sm80" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_DIR="$case_dir" \
  SOUNIO_KRETIKOS_CUBIN_EVIDENCE_JSON="$case_json" \
    bash "$ROOT_DIR/scripts/ci/kretikos_cubin_evidence_gate.sh" >/dev/null

  "$ROOT_DIR/bin/kretikos" kaxi-validate-evidence "$case_json" \
    --expect "status=pass" \
    --expect "runtime.status=pass" \
    --expect "runtime.rung=$runtime_rung" \
    --expect "ptxas.status=pass" >/dev/null
  reports+=("$case_json")
done

jq -s \
  --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    schema: "sounio.kretikos.cubin-evidence-matrix.v1",
    generated_at_utc: $generated_at_utc,
    status: (if all(.[]; .status == "pass" and .runtime.status == "pass" and .ptxas.status == "pass") then "pass" else "fail" end),
    case_count: length,
    cases: map({
      feature,
      source: .source.path,
      kaxi_sha256: .kaxi.sha256,
      ptx_sha256: .ptx.sha256,
      cubin_kind: .cubin.kind,
      cubin_sha256: .cubin.sha256,
      ptxas_output_cubin_sha256: .ptxas_output_cubin.sha256,
      runtime_rung: .runtime.rung,
      runtime_reason: .runtime.reason,
      runtime_device: .runtime.device_name,
      validation_digest
    })
  }' "${reports[@]}" > "$SUMMARY_JSON"

"$ROOT_DIR/bin/kretikos" kaxi-validate-evidence "$SUMMARY_JSON" \
  --expect "status=pass" \
  --expect "case_count=11" >/dev/null

echo "kretikos_cubin_evidence_matrix_gate: PASS report=${SUMMARY_JSON#$ROOT_DIR/} cases=${#CASES[@]}"
