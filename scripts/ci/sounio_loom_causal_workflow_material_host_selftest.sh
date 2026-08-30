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
for tool in sha256sum stat find sort systemctl systemd-run timeout readlink mkdir rm chmod; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ -d "$CAPSULE" && ! -L "$CAPSULE" && -z "$(find "$CAPSULE" -type l -print -quit)" ]] ||
  fail 'capsule is absent, linked, or contains a link'
[[ "$(stat -c '%u:%g:%a' "$CAPSULE")" == 0:0:700 ]] || fail 'capsule root metadata drifted'
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

for required in broker_path controller_runtime_path resident_runtime_path product_runtime_path \
  material_cell_path journal_runtime_path operation_fixture_manifest_path \
  operation_fixture_bundle_path operation_catalog_manifest_path operation_result_manifest_path \
  causal_run_grant_manifest_path causal_run_grant_bundle_path causal_attest_grant_manifest_path \
  causal_attest_grant_bundle_path causal_workflow_manifest_path controller_runtime_manifest_path \
  resident_runtime_manifest_path; do
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
CONTROLLER_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" controller_runtime_path)"
RESIDENT_RUNTIME="$CAPSULE/$(record_value "$MANIFEST" resident_runtime_path)"
CONTROLLER_RUNTIME_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" controller_runtime_manifest_path)"
RESIDENT_RUNTIME_MANIFEST="$CAPSULE/$(record_value "$MANIFEST" resident_runtime_manifest_path)"
[[ "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" semantic_authority)" == Sounio &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" action)" == 9030 &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" controller_commit)" == "$(record_value "$MANIFEST" controller_dependency_commit)" &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" runtime_sha256)" == "$(sha256_file "$CONTROLLER_RUNTIME")" &&
   "$(record_value "$CONTROLLER_RUNTIME_MANIFEST" resident_runtime_manifest_sha256)" == "$(sha256_file "$RESIDENT_RUNTIME_MANIFEST")" &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" producing_language)" == Sounio &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" sounio_resident_v4_commit)" == "$(record_value "$MANIFEST" resident_dependency_commit)" &&
   "$(record_value "$RESIDENT_RUNTIME_MANIFEST" runtime_sha256)" == "$(sha256_file "$RESIDENT_RUNTIME)" ]] ||
  fail 'action-9030 frozen runtime provenance drifted'
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
output="$(timeout --signal=TERM --kill-after=10s 420s "$BROKER" --selftest-causal-workflow-material-host \
  --controller-root "$AUTHORITY_ROOT" \
  --controller-runtime "$CONTROLLER_RUNTIME" \
  --resident-runtime "$RESIDENT_RUNTIME" \
  --product-root "$AUTHORITY_ROOT" \
  --product-runtime "$CAPSULE/$(record_value "$MANIFEST" product_runtime_path)" \
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
for field in semantic_authority=Sounio workflow_action=9037 launch_action=9030 material_language=C++20 \
  material_role=MATERIAL_PARITY hostguardian=true dynamic_user=true compile_actual=true \
  run_exact_actual=true attest_actual=true distinct_fresh_cells=true inherited_descriptors=true \
  arbitrary_path=false action9037_executed=true dynamic_9030_bindings=true typed_handle_persistence=true \
  run_ticket_is_bearer=false run_ticket_is_execution_authority=false compile_count=1 ticket_count=1 \
  launch_count=1 pod_loss_synchronized=true pod_loss_measured=false python_executed=false rust_executed=false \
  production_activation=false parity_open=false claim_ready=false; do
  [[ " $output " == *" $field "* ]] || fail "honest host receipt omitted $field"
done
printf 'sounio-loom-causal-workflow-material-host-selftest: HOST_MEASUREMENT_PASS manifest_sha256=%s receipt_sha256=%s semantic_authority=Sounio workflow_action=9037 launch_action=9030 material_execution=true production_activation=false parity_open=false claim_ready=false\n%s\n' \
  "$EXPECTED_MANIFEST_SHA256" "$(printf '%s\n' "$output" | sha256sum | cut -d ' ' -f 1)" "$output"
