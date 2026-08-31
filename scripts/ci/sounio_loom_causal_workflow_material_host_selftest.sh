#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-causal-workflow-material-host-selftest: FAIL reason=%s material_execution=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-causal-workflow-material-host-selftest: HOST_GATE_UNAVAILABLE reason=%s material_execution=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --capsule ABSOLUTE_PATH --expected-manifest-sha256 HEX --store-root ABSOLUTE_PATH [--phase-marker ABSOLUTE_PATH]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local record="$1" key="$2" line value='' count=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == "$key="* ]] || continue
    count=$((count + 1))
    value="${line#*=}"
  done < "$record"
  [[ "$count" == 1 ]] || fail "manifest field count is invalid: $key=$count"
  printf '%s\n' "$value"
}

CAPSULE=''
EXPECTED_MANIFEST_SHA256=''
STORE_ROOT=''
PHASE_MARKER=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --capsule) CAPSULE="${2:-}"; shift 2 ;;
    --expected-manifest-sha256) EXPECTED_MANIFEST_SHA256="${2:-}"; shift 2 ;;
    --store-root) STORE_ROOT="${2:-}"; shift 2 ;;
    --phase-marker) PHASE_MARKER="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$CAPSULE" == /* && "$STORE_ROOT" == /* && \
   ( -z "$PHASE_MARKER" || "$PHASE_MARKER" == /* ) && \
   "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$(id -u):$(id -g)" == 0:0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd && -d /run/systemd/system ]] ||
  unavailable 'PID 1 is not systemd'
for tool in sha256sum stat find grep sort systemctl systemd-run timeout readlink mkdir rm chmod; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ -d "$CAPSULE" && ! -L "$CAPSULE" && -z "$(find "$CAPSULE" -type l -print -quit)" ]] ||
  fail 'capsule is absent, linked, or contains a link'
[[ "$(stat -c '%u:%g:%a' "$CAPSULE")" == 0:0:555 ]] || fail 'capsule root metadata drifted'
MANIFEST="$CAPSULE/capsule.manifest.v1"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" && "$(stat -c '%u:%g:%a' "$MANIFEST")" == 0:0:444 ]] ||
  fail 'capsule manifest metadata drifted'
[[ "$(sha256_file "$MANIFEST")" == "$EXPECTED_MANIFEST_SHA256" ]] || fail 'capsule manifest hash drifted'
[[ "$(record_value "$MANIFEST" schema)" == loom-causal-workflow-material-host-capsule-v1 && \
   "$(record_value "$MANIFEST" semantic_authority)" == Sounio && \
   "$(record_value "$MANIFEST" workflow_action)" == 9037 && \
   "$(record_value "$MANIFEST" launch_action)" == 9030 && \
   "$(record_value "$MANIFEST" capsule_layout)" == unpacked-directory-v1 && \
   "$(record_value "$MANIFEST" release_path)" == release && \
   "$(record_value "$MANIFEST" authority_root_path)" == release/authority-root && \
   "$(record_value "$MANIFEST" production_activation)" == false && \
   "$(record_value "$MANIFEST" parity_open)" == false && \
   "$(record_value "$MANIFEST" claim_ready)" == false ]] || fail 'capsule posture drifted'

ENTRIES="$CAPSULE/$(record_value "$MANIFEST" payload_entries_path)"
[[ -f "$ENTRIES" && ! -L "$ENTRIES" && "$(stat -c '%u:%g:%a' "$ENTRIES")" == 0:0:444 && \
   "$(sha256_file "$ENTRIES")" == "$(record_value "$MANIFEST" payload_entries_sha256)" ]] ||
  fail 'payload entries manifest drifted'
[[ "$(wc -l < "$ENTRIES" | tr -d ' ')" == "$(record_value "$MANIFEST" payload_entry_count)" ]] ||
  fail 'payload entries count drifted'
while IFS='|' read -r kind mode digest relative; do
  [[ "$relative" =~ ^[A-Za-z0-9._/-]+$ && "$relative" != /* && "/$relative/" != *'/../'* ]] ||
    fail "unsafe payload entry: $relative"
  path="$CAPSULE/release/$relative"
  if [[ "$kind" == D ]]; then
    [[ "$digest" == - && -d "$path" && ! -L "$path" && "$(stat -c '%u:%g:%a' "$path")" == "0:0:$mode" ]] ||
      fail "payload directory drifted: $relative"
  elif [[ "$kind" == F ]]; then
    [[ "$digest" =~ ^[0-9a-f]{64}$ && -f "$path" && ! -L "$path" && \
       "$(stat -c '%u:%g:%a' "$path")" == "0:0:$mode" && \
       "$(sha256_file "$path")" == "$digest" ]] || fail "payload file drifted: $relative"
  else
    fail "payload entry kind is invalid: $kind"
  fi
done < "$ENTRIES"

require_no_nonce_secret_artifact() {
  local path status schema_version=v1
  local secret_schema="loom-causal-material-barrier-nonce-secret-${schema_version}"
  while IFS= read -r -d '' path; do
    [[ "$(basename "$path")" != barrier-nonce.secret ]] ||
      fail "raw barrier nonce secret file is present: $path"
    set +e
    grep -aFq "$secret_schema" "$path"
    status=$?
    set -e
    [[ $status -eq 1 ]] || {
      [[ $status -eq 0 ]] && fail "raw barrier nonce secret schema is present: $path"
      fail "barrier nonce secret schema scan failed: $path"
    }
  done < <(find "$@" -type f -print0)
}

require_no_nonce_secret_artifact "$CAPSULE"

require_payload_entry() {
  local field="$1" expected_kind="$2" value relative line_kind line_mode line_digest line_relative count=0
  value="$(record_value "$MANIFEST" "$field")"
  [[ "$value" == release/* ]] || fail "selected path escapes verified release: $field"
  relative="${value#release/}"
  while IFS='|' read -r line_kind line_mode line_digest line_relative; do
    [[ "$line_relative" == "$relative" ]] || continue
    count=$((count + 1))
    [[ "$line_kind" == "$expected_kind" ]] || fail "selected payload entry kind drifted: $field"
  done < "$ENTRIES"
  [[ "$count" == 1 ]] || fail "selected path is not uniquely inventoried: $field=$count"
}

for required in broker_path controller_runtime_path resident_runtime_path product_runtime_path product_fixture_runtime_path \
  material_cell_path journal_runtime_path causal_workflow_runtime_path operation_fixture_manifest_path \
  operation_fixture_bundle_path operation_catalog_manifest_path operation_result_manifest_path \
  causal_run_grant_manifest_path causal_run_grant_bundle_path causal_attest_grant_manifest_path \
  causal_attest_grant_bundle_path causal_workflow_manifest_path controller_runtime_manifest_path \
  resident_runtime_manifest_path mid_exec_runtime_path mid_exec_manifest_path \
  material_runtime_manifest_path material_host_evidence_path \
  host_selftest_path host_probe_path broker_source_path host_canary_source_path \
  material_cell_source_path; do
  value="$(record_value "$MANIFEST" "$required")"
  [[ "$value" =~ ^release/[A-Za-z0-9._/-]+$ && "/$value/" != *'/../'* ]] ||
    fail "unsafe manifest path: $required"
  require_payload_entry "$required" F
done
require_payload_entry authority_root_path D

RELEASE="$CAPSULE/$(record_value "$MANIFEST" release_path)"
AUTHORITY_ROOT="$CAPSULE/$(record_value "$MANIFEST" authority_root_path)"
BROKER="$CAPSULE/$(record_value "$MANIFEST" broker_path)"
[[ -d "$RELEASE" && -d "$AUTHORITY_ROOT/.git" && -x "$BROKER" ]] || fail 'capsule release topology drifted'
MATERIAL_CELL="$CAPSULE/$(record_value "$MANIFEST" material_cell_path)"
PACKAGED_SELFTEST="$CAPSULE/$(record_value "$MANIFEST" host_selftest_path)"
PACKAGED_PROBE="$CAPSULE/$(record_value "$MANIFEST" host_probe_path)"
BROKER_SOURCE="$CAPSULE/$(record_value "$MANIFEST" broker_source_path)"
HOST_CANARY_SOURCE="$CAPSULE/$(record_value "$MANIFEST" host_canary_source_path)"
MATERIAL_CELL_SOURCE="$CAPSULE/$(record_value "$MANIFEST" material_cell_source_path)"
MATERIAL_RUNTIME_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" material_runtime_manifest_path)"
MATERIAL_HOST_EVIDENCE="$CAPSULE/$(record_value "$MANIFEST" material_host_evidence_path)"
CAUSAL_WORKFLOW_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" causal_workflow_manifest_path)"
MID_EXEC_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" mid_exec_manifest_path)"
[[ "$(record_value "$MANIFEST" semantic_freeze_commit)" == 469d7fecaf8609275250223e7b55993c1e6f641e && \
   "$(record_value "$MANIFEST" broker_sha256)" == "$(sha256_file "$BROKER")" && \
   "$(record_value "$MANIFEST" broker_source_sha256)" == "$(sha256_file "$BROKER_SOURCE")" && \
   "$(record_value "$MANIFEST" host_canary_source_sha256)" == "$(sha256_file "$HOST_CANARY_SOURCE")" && \
   "$(record_value "$MANIFEST" material_cell_sha256)" == "$(sha256_file "$MATERIAL_CELL")" && \
   "$(record_value "$MANIFEST" material_cell_source_sha256)" == "$(sha256_file "$MATERIAL_CELL_SOURCE")" && \
   "$(record_value "$MANIFEST" material_runtime_manifest_sha256)" == "$(sha256_file "$MATERIAL_RUNTIME_MANIFEST")" && \
   "$(record_value "$MANIFEST" material_host_evidence_sha256)" == "$(sha256_file "$MATERIAL_HOST_EVIDENCE")" && \
   "$(record_value "$MANIFEST" host_selftest_sha256)" == "$(sha256_file "$PACKAGED_SELFTEST")" && \
   "$(record_value "$MANIFEST" host_probe_sha256)" == "$(sha256_file "$PACKAGED_PROBE")" ]] ||
  fail 'packaged host executable or gate hash drifted before execution'
[[ "$(record_value "$MATERIAL_RUNTIME_MANIFEST" stage)" == MATERIAL_PARITY_MID_EXEC_PROBE_READY && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" semantic_authority)" == Sounio && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" semantic_freeze_commit)" == 469d7fecaf8609275250223e7b55993c1e6f641e && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" workflow_manifest_sha256)" == "$(sha256_file "$CAUSAL_WORKFLOW_MANIFEST")" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" workflow_semantics_sha256)" == "$(record_value "$CAUSAL_WORKFLOW_MANIFEST" semantics_sha256)" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" mid_exec_manifest_sha256)" == "$(sha256_file "$MID_EXEC_MANIFEST")" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" mid_exec_semantics_sha256)" == "$(record_value "$MID_EXEC_MANIFEST" semantics_sha256)" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" known_p0_count)" == 0 && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" known_p0_fails_closed)" == true && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" controller_recovery)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" pod_loss_measured)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" material_cell_survival_measured)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" barrier_nonce_source)" == getrandom && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" barrier_nonce_storage)" == HostGuardian-memory-only && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" barrier_nonce_secret_persisted)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" transport_trust)" == trusted-privileged-root-observer && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" same_uid_peer_isolation)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" hostile_transport_isolation)" == false && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" barrier_hold_timeout_milliseconds)" == 600000 && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" guardian_request_timeout_milliseconds)" == 570000 && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" broker_timeout_milliseconds)" == 650000 && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" outer_unit_timeout_milliseconds)" == 720000 && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" host_canary_source_sha256)" == "$(sha256_file "$HOST_CANARY_SOURCE")" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" material_cell_source_sha256)" == "$(sha256_file "$MATERIAL_CELL_SOURCE")" && \
   "$(record_value "$MATERIAL_RUNTIME_MANIFEST" host_evidence_sha256)" == "$(sha256_file "$MATERIAL_HOST_EVIDENCE")" ]] ||
  fail 'material runtime posture overclaims recovery or drifted from packaged sources'
PRODUCT_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" product_runtime_path)"
[[ "$(record_value "$MANIFEST" product_runtime_language)" == OCaml &&
   "$(record_value "$MANIFEST" product_runtime_role)" == EFFECT_PARITY &&
   "$(record_value "$MANIFEST" product_fixture_runtime_language)" == Sounio &&
   "$(record_value "$MANIFEST" product_fixture_runtime_role)" == SEMANTIC_FIXTURE_PRODUCER &&
   "$(record_value "$MANIFEST" product_runtime_sha256)" == "$(sha256_file "$PRODUCT_RUNTIME")" ]] ||
  fail 'product ExecCell operational runtime provenance drifted'
CONTROLLER_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" controller_runtime_path)"
RESIDENT_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" resident_runtime_path)"
CONTROLLER_RUNTIME_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" controller_runtime_manifest_path)"
RESIDENT_RUNTIME_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" resident_runtime_manifest_path)"
CAUSAL_WORKFLOW_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" causal_workflow_runtime_path)"
MID_EXEC_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" mid_exec_runtime_path)"
[[ "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" semantic_authority)" == Sounio &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" action)" == 9030 &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" controller_commit)" == "$(record_value "$MANIFEST" controller_dependency_commit)" &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" runtime_sha256)" == "$(sha256_file "$CONTROLLER_RUNTIME")" &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" resident_runtime_manifest_sha256)" == "$(sha256_file "$RESIDENT_RUNTIME_MANIFEST")" &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" producing_language)" == Sounio &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" sounio_resident_v4_commit)" == "$(record_value "$MANIFEST" resident_dependency_commit)" &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" runtime_sha256)" == "$(sha256_file "$RESIDENT_RUNTIME")" ]] ||
  fail 'action-9030 frozen runtime provenance drifted'
[[ "$(record_value "$CAUSAL_WORKFLOW_MANIFEST" producing_language)" == Sounio &&
   "$(record_value "$CAUSAL_WORKFLOW_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_value "$CAUSAL_WORKFLOW_MANIFEST" action)" == 9037 &&
   "$(record_value "$CAUSAL_WORKFLOW_MANIFEST" executable_sha256)" == "$(sha256_file "$CAUSAL_WORKFLOW_RUNTIME")" &&
   "$(stat -c '%u:%g:%a' "$CAUSAL_WORKFLOW_RUNTIME")" == 0:0:555 ]] ||
  fail 'action-9037 Sounio runtime provenance drifted'
[[ "$(record_value "$MID_EXEC_MANIFEST" schema)" == loom-causal-workflow-mid-exec-freeze-v1 && \
   "$(record_value "$MID_EXEC_MANIFEST" stage)" == SEMANTICS_FROZEN && \
   "$(record_value "$MID_EXEC_MANIFEST" producing_language)" == Sounio && \
   "$(record_value "$MID_EXEC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY && \
   "$(record_value "$MID_EXEC_MANIFEST" action)" == 9037 && \
   "$(record_value "$MID_EXEC_MANIFEST" subordinate_contract)" == mid-exec-v1 && \
   "$(record_value "$MID_EXEC_MANIFEST" exact_counts)" == compile+ticket+launch+result+attestation:1 && \
   "$(record_value "$MID_EXEC_MANIFEST" causal_sabotage)" == PASS && \
   "$(record_value "$MID_EXEC_MANIFEST" material_execution)" == false && \
   "$(record_value "$MID_EXEC_MANIFEST" pod_loss_measured)" == false && \
   "$(record_value "$MID_EXEC_MANIFEST" executable_sha256)" == "$(sha256_file "$MID_EXEC_RUNTIME")" && \
   "$(record_value "$MANIFEST" mid_exec_runtime_sha256)" == "$(sha256_file "$MID_EXEC_RUNTIME")" && \
   "$(record_value "$MANIFEST" mid_exec_manifest_sha256)" == "$(sha256_file "$MID_EXEC_MANIFEST")" && \
   "$(record_value "$MANIFEST" mid_exec_semantics_sha256)" == "$(record_value "$MID_EXEC_MANIFEST" semantics_sha256)" && \
   "$(stat -c '%u:%g:%a' "$MID_EXEC_RUNTIME")" == 0:0:555 ]] ||
  fail 'Sounio mid-exec freeze closure drifted before execution'

release_frame="$(record_value "$MID_EXEC_MANIFEST" wire_schema) $(record_value "$MID_EXEC_MANIFEST" release_stage_word) $(record_value "$MID_EXEC_MANIFEST" release_word0) $(record_value "$MID_EXEC_MANIFEST" release_word1)"
replacement_frame="$(record_value "$MID_EXEC_MANIFEST" wire_schema) $(record_value "$MID_EXEC_MANIFEST" claim_replacement_invocation_stage) $(record_value "$MID_EXEC_MANIFEST" claim_replacement_invocation_word0) $(record_value "$MID_EXEC_MANIFEST" claim_replacement_invocation_word1)"
barrier_nonce_frame="$(record_value "$MID_EXEC_MANIFEST" wire_schema) $(record_value "$MID_EXEC_MANIFEST" claim_barrier_nonce_stage) $(record_value "$MID_EXEC_MANIFEST" claim_barrier_nonce_word0) $(record_value "$MID_EXEC_MANIFEST" claim_barrier_nonce_word1)"
[[ "$(printf '%s\n' "$release_frame" | sha256sum | cut -d ' ' -f 1)" == "$(record_value "$MANIFEST" mid_exec_release_frame_sha256)" && \
   "$(printf '%s\n' "$replacement_frame" | sha256sum | cut -d ' ' -f 1)" == "$(record_value "$MANIFEST" mid_exec_replacement_invocation_frame_sha256)" && \
   "$(printf '%s' "$(record_value "$MID_EXEC_MANIFEST" claim_replacement_invocation_decision)" | sha256sum | cut -d ' ' -f 1)" == "$(record_value "$MANIFEST" mid_exec_replacement_invocation_decision_sha256)" && \
   "$(printf '%s\n' "$barrier_nonce_frame" | sha256sum | cut -d ' ' -f 1)" == "$(record_value "$MANIFEST" mid_exec_barrier_nonce_frame_sha256)" && \
   "$(printf '%s' "$(record_value "$MID_EXEC_MANIFEST" claim_barrier_nonce_decision)" | sha256sum | cut -d ' ' -f 1)" == "$(record_value "$MANIFEST" mid_exec_barrier_nonce_decision_sha256)" ]] ||
  fail 'Sounio mid-exec release or sabotage hash drifted before execution'
[[ "$STORE_ROOT" != "$CAPSULE"/* && ! -e "$STORE_ROOT" ]] || fail 'store root is unsafe or pre-existing'
mkdir -m 0700 "$STORE_ROOT"
if [[ -n "$PHASE_MARKER" ]]; then
  [[ "$PHASE_MARKER" != "$CAPSULE"/* && ! -e "$PHASE_MARKER" ]] ||
    fail 'phase marker is unsafe or pre-existing'
  printf 'MATERIAL_HOST_UNIT_STARTED\n' > "$PHASE_MARKER"
  chmod 0400 "$PHASE_MARKER"
fi
cleanup() {
  chmod -R u+rwX "$STORE_ROOT" 2>/dev/null || true
  rm -rf "$STORE_ROOT"
}
trap cleanup EXIT

SYSTEMD_RUN="$(readlink -f "$(command -v systemd-run)")"
SYSTEMCTL="$(readlink -f "$(command -v systemctl)")"
[[ "$SYSTEMD_RUN" == /* && "$SYSTEMCTL" == /* ]] || unavailable 'canonical systemd tools are unavailable'
set +e
output="$(timeout --signal=TERM --kill-after=10s 650s "$BROKER" --selftest-causal-workflow-material-host \
  --controller-root "$AUTHORITY_ROOT" \
  --controller-runtime "$CONTROLLER_RUNTIME" \
  --resident-runtime "$RESIDENT_RUNTIME" \
  --product-root "$AUTHORITY_ROOT" \
  --product-runtime "$PRODUCT_RUNTIME" \
  --operation-fixture-manifest "$CAPSULE/$(record_value "$MANIFEST" operation_fixture_manifest_path)" \
  --operation-fixture-bundle "$CAPSULE/$(record_value "$MANIFEST" operation_fixture_bundle_path)" \
  --operation-catalog-manifest "$CAPSULE/$(record_value "$MANIFEST" operation_catalog_manifest_path)" \
  --operation-result-manifest "$CAPSULE/$(record_value "$MANIFEST" operation_result_manifest_path)" \
  --causal-run-grant-manifest "$CAPSULE/$(record_value "$MANIFEST" causal_run_grant_manifest_path)" \
  --causal-run-grant-bundle "$CAPSULE/$(record_value "$MANIFEST" causal_run_grant_bundle_path)" \
  --causal-attest-grant-manifest "$CAPSULE/$(record_value "$MANIFEST" causal_attest_grant_manifest_path)" \
  --causal-attest-grant-bundle "$CAPSULE/$(record_value "$MANIFEST" causal_attest_grant_bundle_path)" \
  --causal-workflow-manifest "$CAPSULE/$(record_value "$MANIFEST" causal_workflow_manifest_path)" \
  --causal-workflow-journal-runtime "$CAPSULE/$(record_value "$MANIFEST" journal_runtime_path)" \
  --causal-material-cell "$CAPSULE/$(record_value "$MANIFEST" material_cell_path)" \
  --causal-material-store "$STORE_ROOT" --causal-pod-loss-control enabled \
  --systemd-run "$SYSTEMD_RUN" --systemctl "$SYSTEMCTL" 2>&1)"
status=$?
set -e
[[ $status -eq 0 && "$output" == 'LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST PASS '* ]] ||
  fail "broker host selftest failed status=$status output=$output"
[[ ! -e "$STORE_ROOT/barrier-nonce.secret" ]] ||
  fail 'raw barrier nonce secret file persisted'
require_no_nonce_secret_artifact "$CAPSULE" "$STORE_ROOT"
for transport_record in mid-exec-ready.record mid-exec-sabotage.record \
  mid-exec-sabotage-refusal.record mid-exec-release-request.record \
  result.record attestation.record; do
  path="$STORE_ROOT/$transport_record"
  [[ -f "$path" ]] || continue
  set +e
  grep -aEq '^barrier_nonce=' "$path"
  transport_nonce_status=$?
  set -e
  [[ $transport_nonce_status -eq 1 ]] ||
    fail "transport-facing record exposed raw nonce field: $transport_record"
done
set +e
printf '%s\n' "$output" | grep -aEq '^barrier_nonce=[0-9a-f]{64}$'
raw_output_status=$?
set -e
[[ $raw_output_status -eq 1 ]] || fail 'host result contained a raw barrier nonce record'
for field in semantic_authority=Sounio workflow_action=9037 launch_action=9030 material_language=C++20 \
  material_role=MATERIAL_PARITY hostguardian=true dynamic_user=true compile_actual=true \
  run_exact_actual=true attest_actual=true distinct_fresh_cells=true inherited_descriptors=true \
  arbitrary_path=false action9037_executed=true dynamic_9030_bindings=true typed_handle_persistence=true \
  run_ticket_is_bearer=false run_ticket_is_execution_authority=false compile_count=1 ticket_count=1 \
  launch_count=1 result_count=1 attestation_count=1 pod_loss_synchronized=true \
  controller_recovery=false pod_loss_measured=false material_cell_survival_measured=false \
  pod_loss_boundary=MATERIAL_RUNNING_IN_EXEC mid_exec_release_authority=Sounio \
  mid_exec_release_sabotage=DENY592 barrier_nonce_source=getrandom \
  barrier_nonce_storage=HostGuardian-memory-only barrier_nonce_secret_persisted=false \
  transport_trust=trusted-privileged-root-observer same_uid_peer_isolation=false \
  hostile_transport_isolation=false \
  python_executed=false rust_executed=false \
  production_activation=false parity_open=false claim_ready=false; do
  [[ " $output " == *" $field "* ]] || fail "honest host receipt omitted $field"
done
printf 'sounio-loom-causal-workflow-material-host-selftest: HOST_LOCAL_EXECUTION_PASS manifest_sha256=%s receipt_sha256=%s semantic_authority=Sounio workflow_action=9037 launch_action=9030 material_execution=true controller_recovery=false pod_loss_measured=false material_cell_survival_measured=false production_activation=false parity_open=false claim_ready=false\n%s\n' \
  "$EXPECTED_MANIFEST_SHA256" "$(printf '%s\n' "$output" | sha256sum | cut -d ' ' -f 1)" "$output"
