#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="${1:-examples/kretikos/lower_epistemic_dual_output_f32.sio}"
OUT_DIR="${SOUNIO_KRETIKOS_HPC_SOURCE_LOWERING_GATE_DIR:-$(mktemp -d /tmp/kretikos-hpc-source-lowering.XXXXXX)}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-420}"

mkdir -p "$OUT_DIR"

DOCTOR_LOG="$OUT_DIR/kretikos_hpc_doctor.out"
SOURCE_LOG="$OUT_DIR/kretikos_hpc_source_lowering.out"
GATE_JSON="$OUT_DIR/kretikos_hpc_source_lowering_gate.v1.json"

echo "kretikos_hpc_source_lowering_gate: doctor"
./bin/kretikos hpc doctor >"$DOCTOR_LOG" 2>&1
cat "$DOCTOR_LOG"

echo "kretikos_hpc_source_lowering_gate: source=$SOURCE"
WAIT_TIMEOUT_SECONDS="$WAIT_TIMEOUT_SECONDS" \
KRETIKOS_HPC_PUBLISH_RESULTS=1 \
KRETIKOS_HPC_FETCH_RESULTS=1 \
  ./bin/kretikos hpc source "$SOURCE" >"$SOURCE_LOG" 2>&1
cat "$SOURCE_LOG"

artifact_json="$(awk -F= '/^ArtifactJSON=/ { print $2; exit }' "$SOURCE_LOG")"
if [[ -z "$artifact_json" ]]; then
  echo "kretikos_hpc_source_lowering_gate: FAIL missing ArtifactJSON in $SOURCE_LOG" >&2
  exit 1
fi
if [[ ! -f "$artifact_json" ]]; then
  echo "kretikos_hpc_source_lowering_gate: FAIL artifact json not found: $artifact_json" >&2
  exit 1
fi

# Replace python heredoc with kaxi-validate-evidence + json-emit pipeline.
# All validations halt the script via set -euo pipefail on first failure.

source_name="$(basename "$SOURCE")"
case "$source_name" in
  *knowledge_dual*) expected_recognized_pattern="indexed_f32_knowledge_dual_output" ;;
  *)                expected_recognized_pattern="indexed_f32_epistemic_dual_output" ;;
esac

./bin/kretikos kaxi-validate-evidence "$artifact_json" \
  --expect "schema=sounio.kretikos.hpc-source-result.v1" \
  --expect "runtime_validation.status=pass" \
  --expect "runtime_validation.rung=epistemic_dual_output_f32" \
  --expect "runtime_validation.kernel=sounio_bare_epistemic_dual_output_f32_sm80" \
  --expect "kaxi_certificate.status=pass" \
  --expect "kaxi_certificate.reason=runtime_backed_source_lowering_certificate_pass" \
  --expect "kaxi_certificate.certificate.schema=sounio.kretikos.kaxi-lowering-certificate.v1" \
  --expect "kaxi_certificate.certificate.status=pass" \
  --expect "kaxi_certificate.certificate.lowering.profile_hint=none" \
  --expect "kaxi_certificate.certificate.lowering.fallback_path=none" \
  --expect "kaxi_certificate.certificate.lowering.recognized_pattern=$expected_recognized_pattern" \
  --expect "kaxi_certificate.certificate.lowering.kaxi_pattern=source_epistemic_dual_output_f32" \
  --expect "kaxi_certificate.certificate.runtime.runtime_validation.status=pass" \
  --expect "kaxi_certificate.certificate.runtime.runtime_validation.rung=epistemic_dual_output_f32" \
  --expect "kaxi_certificate.certificate.kaxi.pattern=source_epistemic_dual_output_f32" \
  --expect "kaxi_certificate.certificate.kaxi.epistemic_lanes.store_global_count>=2" >/dev/null

# Aggregate gate JSON via kretikos json-emit (replaces python json.dump).
__sha() { sha256sum "$1" | awk '{print $1}'; }
__af() { ./bin/kretikos kaxi-validate-evidence "$1" --print-or-empty "$2"; }
__opt_string() {
  local p="$1" v="$2"
  if [ -z "$v" ]; then printf '%s\0%s\0' "--null" "$p"
  else printf '%s\0%s\0' "--string" "${p}=${v}"
  fi
}
__opt_int() {
  local p="$1" v="$2"
  if [ -z "$v" ]; then printf '%s\0%s\0' "--null" "$p"
  else printf '%s\0%s\0' "--int" "${p}=${v}"
  fi
}
__emit_obj() { xargs -0 ./bin/kretikos json-emit "$@"; }

__rt_status=$(__af "$artifact_json" runtime_validation.status)
__rt_reason=$(__af "$artifact_json" runtime_validation.reason)
__rt_dev=$(__af "$artifact_json" runtime_validation.device_name)
__rt_drv=$(__af "$artifact_json" runtime_validation.driver_version)
__rt_ccmaj=$(__af "$artifact_json" runtime_validation.cc_major)
__rt_ccmin=$(__af "$artifact_json" runtime_validation.cc_minor)
__rt_rung=$(__af "$artifact_json" runtime_validation.rung)
__rt_kern=$(__af "$artifact_json" runtime_validation.kernel)
__runtime_obj=$({
  __opt_string status "$__rt_status"
  __opt_string reason "$__rt_reason"
  __opt_string device_name "$__rt_dev"
  __opt_string driver_version "$__rt_drv"
  __opt_int cc_major "$__rt_ccmaj"
  __opt_int cc_minor "$__rt_ccmin"
  __opt_string rung "$__rt_rung"
  __opt_string kernel "$__rt_kern"
} | __emit_obj)

__c_status=$(__af "$artifact_json" kaxi_certificate.status)
__c_reason=$(__af "$artifact_json" kaxi_certificate.reason)
__c_schema=$(__af "$artifact_json" kaxi_certificate.certificate.schema)
__c_profile=$(__af "$artifact_json" kaxi_certificate.certificate.profile.name)
__c_rec=$(__af "$artifact_json" kaxi_certificate.certificate.lowering.recognized_pattern)
__c_kp=$(__af "$artifact_json" kaxi_certificate.certificate.lowering.kaxi_pattern)
__c_ph=$(__af "$artifact_json" kaxi_certificate.certificate.lowering.profile_hint)
__c_fb=$(__af "$artifact_json" kaxi_certificate.certificate.lowering.fallback_path)
__c_sgc=$(__af "$artifact_json" kaxi_certificate.certificate.kaxi.epistemic_lanes.store_global_count)
__c_rs=$(__af "$artifact_json" kaxi_certificate.certificate.runtime.runtime_validation.status)
__c_rr=$(__af "$artifact_json" kaxi_certificate.certificate.runtime.runtime_validation.reason)
__c_ru=$(__af "$artifact_json" kaxi_certificate.certificate.runtime.runtime_validation.rung)
__c_rk=$(__af "$artifact_json" kaxi_certificate.certificate.runtime.runtime_validation.kernel)
__certificate_obj=$({
  __opt_string status "$__c_status"
  __opt_string reason "$__c_reason"
  __opt_string schema "$__c_schema"
  __opt_string profile "$__c_profile"
  __opt_string recognized_pattern "$__c_rec"
  __opt_string kaxi_pattern "$__c_kp"
  __opt_string profile_hint "$__c_ph"
  __opt_string fallback_path "$__c_fb"
  __opt_int store_global_count "$__c_sgc"
  __opt_string runtime_status "$__c_rs"
  __opt_string runtime_reason "$__c_rr"
  __opt_string runtime_rung "$__c_ru"
  __opt_string runtime_kernel "$__c_rk"
} | __emit_obj \
  --string "expected_recognized_pattern=$expected_recognized_pattern")

__job_id=$(__af "$artifact_json" job_id)
__host=$(__af "$artifact_json" host)

./bin/kretikos json-emit \
  --string "schema=sounio.kretikos.hpc-source-lowering-gate.v1" \
  --string "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --string "status=pass" \
  --string "source=$SOURCE" \
  --string "doctor_log=$(basename "$DOCTOR_LOG")" \
  --string "source_log=$(basename "$SOURCE_LOG")" \
  --string "artifact_json=$artifact_json" \
  --string "artifact_sha256=$(__sha "$artifact_json")" \
  --string-if-nonempty "job_id=$__job_id" \
  --string-if-nonempty "host=$__host" \
  --raw-json "runtime=$__runtime_obj" \
  --raw-json "certificate=$__certificate_obj" \
  --raw-json "failures=[]" \
  --array-strings "boundaries=slurm_worker_is_cuda_runtime_authority|source_has_no_profile_directive_and_must_lower_through_kaxi_lower_source|certificate_must_be_source_lowering_not_profile_level|runtime_acceptance_requires_cuda_driver_load_launch_and_copyback|does_not_claim_arbitrary_knowledge_generic_lowering|does_not_claim_arbitrary_sounio_gpu_lowering" \
  > "$GATE_JSON"

echo "kretikos_hpc_source_lowering_gate: PASS out=$OUT_DIR artifact=$artifact_json"
