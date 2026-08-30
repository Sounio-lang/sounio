#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'promote-loom-host-exec-quorum-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --archive ABSOLUTE_PATH --expected-sha256 HEX --mode verify|host-gate\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate manifest field: $key"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "manifest omitted field: $key"
  printf '%s\n' "$found"
}

stable_identity() {
  local path="$1"
  if [[ -L "$path" ]]; then
    printf 'LINK:%s\n' "$(readlink "$path")"
  elif [[ -e "$path" ]]; then
    printf 'NODE:%s\n' "$(stat -c '%d:%i:%f:%Y' "$path")"
  else
    printf 'ABSENT\n'
  fi
}

compare_release() {
  local source="$1" destination="$2" source_path relative target
  [[ "$(find "$source" -mindepth 1 -printf . | wc -c)" == \
     "$(find "$destination" -mindepth 1 -printf . | wc -c)" ]] || return 1
  while IFS= read -r -d '' source_path; do
    relative="${source_path#"$source"/}"
    target="$destination/$relative"
    if [[ -d "$source_path" && ! -L "$source_path" ]]; then
      [[ -d "$target" && ! -L "$target" ]] || return 1
    elif [[ -f "$source_path" && ! -L "$source_path" ]]; then
      [[ -f "$target" && ! -L "$target" ]] || return 1
      [[ "$(sha256_file "$source_path")" == "$(sha256_file "$target")" ]] || return 1
    else
      return 1
    fi
    [[ "$(stat -c '%a' "$source_path")" == "$(stat -c '%a' "$target")" ]] || return 1
    [[ "$(stat -c '%u:%g' "$target")" == 0:0 ]] || return 1
  done < <(find "$source" -mindepth 1 -print0 | sort -z)
}

ARCHIVE=''
EXPECTED_SHA256=''
MODE=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      [[ $# -ge 2 ]] || usage
      ARCHIVE="$2"
      shift 2
      ;;
    --expected-sha256)
      [[ $# -ge 2 ]] || usage
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || usage
      MODE="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done
[[ "$ARCHIVE" == /* && -f "$ARCHIVE" && ! -L "$ARCHIVE" && "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$MODE" == verify || "$MODE" == host-gate ]] || usage
for tool in sha256sum stat mktemp tar find sort wc readlink install cp mv rm sync timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required promotion tool is absent: $tool"
done
[[ "$(sha256_file "$ARCHIVE")" == "$EXPECTED_SHA256" ]] || fail 'capsule archive hash drifted'

while IFS= read -r member; do
  [[ "$member" =~ ^[A-Za-z0-9._/-]+$ && \
     ( "$member" == capsule-v1 || "$member" == capsule-v1/* ) ]] || fail "unsafe capsule member: $member"
  [[ "/$member/" != *'/../'* && "/$member/" != *'/./'* ]] || fail "capsule member traverses a directory: $member"
done < <(tar -tf "$ARCHIVE")
while IFS= read -r verbose; do
  [[ "${verbose:0:1}" == d || "${verbose:0:1}" == - ]] || fail 'capsule contains a non-file archive entry'
done < <(tar -tvf "$ARCHIVE")

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-exec-quorum-promote.XXXXXX")"
created_release=false
HOST_RELEASE=''
cleanup() {
  if [[ "$created_release" == true && -n "$HOST_RELEASE" && -d "$HOST_RELEASE" && ! -L "$HOST_RELEASE" ]]; then
    rm -rf "$HOST_RELEASE"
    sync -f "$(dirname "$HOST_RELEASE")" 2>/dev/null || true
  fi
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
tar --no-same-owner --same-permissions -xf "$ARCHIVE" -C "$WORK"

CAPSULE="$WORK/capsule-v1"
RELEASE="$CAPSULE/release"
META="$CAPSULE/meta"
CAPSULE_MANIFEST="$META/capsule.manifest.v1"
RELEASE_MANIFEST="$RELEASE/release.manifest.v1"
HOST_GATE="$META/sounio_loom_host_exec_quorum_host_gate.sh"
PROMOTER="$META/promote_loom_host_exec_quorum_capsule.sh"
for required in "$CAPSULE_MANIFEST" "$RELEASE_MANIFEST" "$HOST_GATE" "$PROMOTER"; do
  [[ -f "$required" && ! -L "$required" ]] || fail "required capsule member is absent or linked: $required"
done
[[ -z "$(find "$CAPSULE" -type l -print -quit)" ]] || fail 'capsule contains a symlink'
[[ "$(record_value "$CAPSULE_MANIFEST" schema)" == loom-host-exec-quorum-experiment-capsule-v1 ]] || fail 'capsule schema drifted'
[[ "$(record_value "$CAPSULE_MANIFEST" source_tree_state)" == CLEAN ]] || fail 'dirty-source capsule cannot reach host'
[[ "$(record_value "$CAPSULE_MANIFEST" production_activation)" == false ]] || fail 'capsule requested production activation'
[[ "$(record_value "$CAPSULE_MANIFEST" product_lane_cell_canary)" == false && \
   "$(record_value "$CAPSULE_MANIFEST" distinct_uid_product_broker_canary)" == false && \
   "$(record_value "$CAPSULE_MANIFEST" product_exec_cell_canary)" == false ]] ||
  fail 'capsule preclaimed a product lane-cell canary'
[[ "$(record_value "$CAPSULE_MANIFEST" material_grant)" == false ]] || fail 'capsule preclaimed a material grant'
[[ "$(sha256_file "$RELEASE_MANIFEST")" == "$(record_value "$CAPSULE_MANIFEST" release_manifest_sha256)" ]] || fail 'release manifest hash drifted'
[[ "$(sha256_file "$HOST_GATE")" == "$(record_value "$CAPSULE_MANIFEST" host_gate_sha256)" ]] || fail 'host gate hash drifted'
[[ "$(sha256_file "$PROMOTER")" == "$(record_value "$CAPSULE_MANIFEST" promoter_sha256)" ]] || fail 'capsule promoter hash drifted'
[[ "$(sha256_file "$0")" == "$(record_value "$CAPSULE_MANIFEST" promoter_sha256)" ]] || fail 'executing promoter differs from capsule promoter'
[[ "$(record_value "$RELEASE_MANIFEST" schema)" == loom-host-exec-quorum-experiment-release-v1 ]] || fail 'release schema drifted'
[[ "$(record_value "$RELEASE_MANIFEST" semantic_authority)" == Sounio ]] || fail 'release semantic authority drifted'
[[ "$(record_value "$RELEASE_MANIFEST" controller_language)" == OCaml ]] || fail 'release controller language drifted'
for closed in material_grant material_execution launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  [[ "$(record_value "$RELEASE_MANIFEST" "$closed")" == false ]] || fail "release preclaimed $closed"
done

RELEASE_ID="$(record_value "$RELEASE_MANIFEST" release_id)"
[[ "$RELEASE_ID" =~ ^9030-hostq-[0-9a-f]{32}$ && "$RELEASE_ID" == "$(record_value "$CAPSULE_MANIFEST" release_id)" ]] || fail 'release identity drifted'
RELEASE_MANIFEST_SHA256="$(sha256_file "$RELEASE_MANIFEST")"
if [[ "$MODE" == verify ]]; then
  printf 'LOOM_HOST_EXEC_QUORUM_CAPSULE_VERIFY PASS archive_sha256=%s release_id=%s release_manifest_sha256=%s semantic_authority=Sounio controller_language=OCaml material_role=MATERIAL_PARITY product_lane_cell_canary=false distinct_uid_product_broker_canary=false product_exec_cell_canary=false production_activation=false material_grant=false material_execution=false launch_open=false parity_open=false claim_ready=false\n' \
    "$EXPECTED_SHA256" "$RELEASE_ID" "$RELEASE_MANIFEST_SHA256"
  exit 0
fi

[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || fail 'host gate installation requires root'
[[ "$(tr -d '\n' < /proc/1/comm)" == systemd ]] || fail 'host gate installation requires PID 1 systemd'
RELEASE_PARENT=/usr/lib/sounio/loom/experiments/releases
install -d -m 0755 -o root -g root "$RELEASE_PARENT"
HOST_RELEASE="$RELEASE_PARENT/$RELEASE_ID"
stable_current_before="$(stable_identity /usr/lib/sounio/loom/current)"
stable_broker_before="$(stable_identity /usr/libexec/sounio/loom-kernel-principal-broker)"

if [[ -e "$HOST_RELEASE" || -L "$HOST_RELEASE" ]]; then
  [[ -d "$HOST_RELEASE" && ! -L "$HOST_RELEASE" ]] || fail 'existing experiment release path is unsafe'
  compare_release "$RELEASE" "$HOST_RELEASE" || fail 'existing immutable experiment release drifted'
else
  stage="$(mktemp -d "$RELEASE_PARENT/.loom-hostq-release.XXXXXX")"
  cp -a "$RELEASE/." "$stage/"
  chown -R root:root "$stage"
  sync -f "$stage" 2>/dev/null || sync
  mv -T "$stage" "$HOST_RELEASE"
  sync -f "$RELEASE_PARENT" 2>/dev/null || sync
  created_release=true
fi

set +e
host_output="$(timeout --signal=TERM --kill-after=5s 360s "$HOST_GATE" \
  --release "$HOST_RELEASE" --expected-manifest-sha256 "$RELEASE_MANIFEST_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host causal gate failed or timed out status=$host_status output=$host_output"
[[ "$host_output" == 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS '* ]] || fail 'host causal gate receipt diverged'
[[ "$host_output" == *'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* && \
   "$host_output" == *' process_witness_core=true '* && \
   "$host_output" == *' complete_effects=false '* ]] ||
  fail 'host ProcessWitness measurement is absent'
[[ "$host_output" == *'LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS '* && \
   "$host_output" == *' lane_cell_canary_attached=true '* && \
   "$host_output" == *' distinct_uid_product_broker_canary=true '* && \
   "$host_output" == *' exec_cell_attached=false '* && \
   "$host_output" == *' production_activation=false '* ]] ||
  fail 'host product DynamicUser ExecIngress measurement is absent'
[[ "$host_output" == *'loom-product-exec-cell-host: PASS '* && \
   "$host_output" == *' simultaneous_distinct_dynamic_users=true '* && \
   "$host_output" == *' command_mismatch=DENY492 '* && \
   "$host_output" == *' sabotage_exec_cell_created=false '* && \
   "$host_output" == *' sabotage_payload_executed=false '* && \
   "$host_output" == *' result_action=9033 '* && \
   "$host_output" == *' intent_action=9034 '* && \
   "$host_output" == *' result_returned=true result_presenter=read-only '* && \
   "$host_output" == *' exact_fixture_hook_switched=true local_exec_capability_used=false '* && \
   "$host_output" == *' provider_hook_switched=true provider_lifecycle_attached=true provider_fixture_language=OCaml '* && \
   "$host_output" == *' operation_catalog_action=9035 operation_result_action=9036 '* && \
   "$host_output" == *' operation_simultaneous_distinct_dynamic_users=true '* && \
   "$host_output" == *' operation_protocol=READY+ARM+CLOSE '* && \
   "$host_output" == *' operation_close_receipt_bound=true '* && \
   "$host_output" == *' operation_runtime_storage=systemd-private-symlink '* && \
   "$host_output" == *' operation_runtime_custody_pre_close=true '* && \
   "$host_output" == *' operation_runtime_released_after_close=true '* && \
   "$host_output" == *' operation_material_files=artifact+stdout+stderr '* && \
   "$host_output" == *' operation_material_files_measured=3 '* && \
   "$host_output" == *' operation_principal_self_measured=true '* && \
   "$host_output" == *' operation_descriptor_self_measured=true '* && \
   "$host_output" == *' operation_pidfd_extinct=true '* && \
   "$host_output" == *' operation_cgroup_unpopulated=true '* && \
   "$host_output" == *' operation_unit_inactive=true '* && \
   "$host_output" == *' operation_runtime_directory_extinct=true '* && \
   "$host_output" == *' operation_record_returned=true '* && \
   "$host_output" == *' operation_result_presenter=read-only '* && \
   "$host_output" == *' operation_command_mismatch=DENY492 '* && \
   "$host_output" == *' operation_sabotage_cell_created=false '* && \
   "$host_output" == *' operation_result_binding_sabotage=closed '* && \
   "$host_output" == *' operation_result_digest_sabotage=closed '* && \
   "$host_output" == *' operation_result_manifest_sabotage=closed '* && \
   "$host_output" == *' python_executed=false rust_executed=false '* && \
   "$host_output" == *' raw_event_separate=true event_projection=Sounio-9034 event_override=false intent_command_mismatch=DENY555 '* && \
   "$host_output" == *' exec_cell_attached=true '* && \
   "$host_output" == *' material_execution=true '* && \
   "$host_output" == *' test_only=true '* ]] ||
  fail 'host product ExecCell material measurement is absent'
[[ "$(stable_identity /usr/lib/sounio/loom/current)" == "$stable_current_before" ]] || fail 'production current target moved during experiment'
[[ "$(stable_identity /usr/libexec/sounio/loom-kernel-principal-broker)" == "$stable_broker_before" ]] || fail 'production broker target moved during experiment'

created_release=false
cleanup
trap - EXIT
printf '%s\n' "$host_output"
printf 'LOOM_HOST_EXEC_QUORUM_EXPERIMENT_INSTALL PASS archive_sha256=%s release_id=%s release_manifest_sha256=%s experimental_release=%s production_current_unchanged=true production_broker_unchanged=true rollback=identity-operation semantic_authority=Sounio controller_language=OCaml material_role=MATERIAL_PARITY process_witness_core=true affirmative_extinction=true complete_effects=false product_lane_cell_canary=true distinct_uid_product_broker_canary=true fleet_lane_cell_attached=false product_exec_cell_canary=true exact_fixture_result_attached=true result_returned=true result_presenter=read-only exact_fixture_hook_switched=true provider_hook_switched=true provider_lifecycle_attached=true provider_fixture_language=OCaml operation_catalog_action=9035 operation_result_action=9036 operation_exec_cell_attached=true operation_record_returned=true operation_result_presenter=read-only operation_protocol=READY+ARM+CLOSE operation_close_receipt_bound=true operation_runtime_storage=systemd-private-symlink operation_runtime_custody_pre_close=true operation_runtime_released_after_close=true operation_material_files=artifact+stdout+stderr operation_material_files_measured=3 operation_principal_self_measured=true operation_descriptor_self_measured=true operation_pidfd_extinct=true operation_cgroup_unpopulated=true operation_unit_inactive=true operation_runtime_directory_extinct=true operation_command_mismatch=DENY492 operation_sabotage_cell_created=false operation_result_binding_sabotage=closed operation_result_digest_sabotage=closed operation_result_manifest_sabotage=closed python_executed=false rust_executed=false local_exec_capability_used=false event_projection=Sounio-9034+9035 event_override=false intent_command_mismatch=DENY555 exec_cell_attached=true material_grant=true material_execution=true test_only=true production_activation=false launch_open=false parity_open=false claim_ready=false\n' \
  "$EXPECTED_SHA256" "$RELEASE_ID" "$RELEASE_MANIFEST_SHA256" "$HOST_RELEASE"
