#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
TEST_MODE="${SOUNIO_SPARK_PAIR_TEST_MODE:-}"
if [[ "$TEST_MODE" == fixture-v1 ]]; then
  ROOT_DIR="${SOUNIO_SOURCE_ROOT:-}"
  POLICY="${SOUNIO_SPARK_PAIR_POLICY:-$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1}"
  FREEZE="${SOUNIO_SPARK_PAIR_FREEZE:-$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1}"
  AUTHORITY="${SOUNIO_SPARK_PAIR_AUTHORITY:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-arbiter}"
  BUILD="${SOUNIO_SPARK_PAIR_BUILD:-$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh}"
  BACKEND="${SOUNIO_SPARK_PAIR_BACKEND:-$ROOT_DIR/scripts/dev/spark_pair_arbiter_k8s_backend.sh}"
else
  ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
  POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
  FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
  AUTHORITY="$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-arbiter"
  BUILD="$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh"
  BACKEND="$ROOT_DIR/scripts/dev/spark_pair_arbiter_k8s_backend.sh"
fi
HOLDER="${SOUNIO_SPARK_PAIR_HOLDER:-pireus-$(hostname)-$$}"
RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-}"
COMMAND_TIMEOUT="${SOUNIO_SPARK_PAIR_COMMAND_TIMEOUT:-190}"
RECOVERY_ACTIVE=0
PAIR_ACQUIRED=0
CURRENT_EPOCH=0
LAST_RECEIPT=''

fail() {
  printf 'spark-pair-arbiter: REFUSE: %s\n' "$*" >&2
  exit 42
}

verify_runtime_root() {
  if [[ "$TEST_MODE" == fixture-v1 ]]; then
    [[ -n "$ROOT_DIR" ]] || fail 'fixture-v1 requires an explicit source root'
    [[ "$(basename "$SCRIPT_PATH")" == spark-pair-arbiter-fixture ]] || \
      fail 'fixture-v1 is forbidden in the canonical controller'
  else
    [[ -z "$TEST_MODE" ]] || fail 'unknown test mode'
    [[ -z "${SOUNIO_SOURCE_ROOT:-}${SOUNIO_SPARK_PAIR_POLICY:-}${SOUNIO_SPARK_PAIR_FREEZE:-}${SOUNIO_SPARK_PAIR_AUTHORITY:-}${SOUNIO_SPARK_PAIR_BUILD:-}${SOUNIO_SPARK_PAIR_BACKEND:-}${SOUNIO_SPARK_PAIR_COMMAND_TIMEOUT:-}" ]] || \
      fail 'runtime path overrides are forbidden in the canonical controller'
  fi
}

configure_command_timeout() {
  [[ "$COMMAND_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fail 'controller command timeout must be positive'
  if [[ "$TEST_MODE" != fixture-v1 ]]; then
    [[ "$COMMAND_TIMEOUT" == "$(policy_value "$POLICY" controller_command_timeout_seconds)" ]] || \
      fail 'controller command timeout drifted from material policy'
  fi
}

policy_value() {
  local file="$1" key="$2" value count
  [[ -r "$file" ]] || fail "policy surface missing: $file"
  count="$(sed -n "s/^${key}=//p" "$file" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "policy key is missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$file")"
  [[ -n "$value" ]] || fail "policy key is empty: $key"
  printf '%s\n' "$value"
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

verify_frozen_file() {
  local source_key="$1" hash_key="$2" expected actual path
  path="$ROOT_DIR/$(policy_value "$FREEZE" "$source_key")"
  expected="$(policy_value "$FREEZE" "$hash_key")"
  [[ -r "$path" ]] || fail "frozen file missing: $path"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || fail "frozen file drifted: $source_key"
}

verify_frozen_authority() {
  local expected actual
  [[ "$(policy_value "$FREEZE" status)" == SEMANTICS_FROZEN ]] || \
    fail 'Sounio semantics are not frozen'
  [[ "$(policy_value "$FREEZE" semantic_authority)" == Sounio ]] || \
    fail 'semantic authority is not Sounio'

  verify_frozen_file authority_source authority_sha256
  verify_frozen_file adapter_source adapter_sha256
  verify_frozen_file expectations_source expectations_sha256
  verify_frozen_file compiler_source compiler_sha256
  verify_frozen_file material_policy_source material_policy_sha256
  verify_frozen_file material_controller_source material_controller_sha256
  verify_frozen_file material_backend_source material_backend_sha256
  verify_frozen_file admission_manifest_source admission_manifest_sha256
  verify_frozen_file host_fence_manifest_source host_fence_manifest_sha256
  verify_frozen_file device_barrier_source device_barrier_source_sha256
  verify_frozen_file device_barrier_arm64_gate_source device_barrier_arm64_gate_sha256
  verify_frozen_file installer_source installer_sha256
  verify_frozen_file selftest_source selftest_sha256
  verify_frozen_file mock_backend_source mock_backend_sha256
  verify_frozen_file live_gate_source live_gate_sha256
  verify_frozen_file ci_workflow_source ci_workflow_sha256
  verify_frozen_file parity_open_source parity_open_sha256
  verify_frozen_file dgx_material_slurm_source dgx_material_slurm_sha256
  verify_frozen_file dgx_material_cuda_source dgx_material_cuda_sha256
  verify_frozen_file dgx_material_header_source dgx_material_header_sha256

  if [[ ! -x "$AUTHORITY" ]]; then
    SOUNIO_SPARK_PAIR_OUTPUT="$AUTHORITY" "$BUILD" >/dev/null
  fi
  expected="$(policy_value "$FREEZE" native_executable_sha256)"
  actual="$(sha256_file "$AUTHORITY")"
  [[ "$actual" == "$expected" ]] || fail 'authority executable is not the frozen Sounio artifact'
  [[ -x "$BACKEND" ]] || fail "material backend missing: $BACKEND"
  [[ "$(sha256_file "$POLICY")" == "$(policy_value "$FREEZE" material_policy_sha256)" ]] || \
    fail 'selected material policy is not frozen'
  [[ "$(sha256_file "$BACKEND")" == "$(policy_value "$FREEZE" material_backend_sha256)" ]] || \
    fail 'selected material backend is not frozen'

}

init_receipt_dir() {
  if [[ -z "$RECEIPT_DIR" ]]; then
    RECEIPT_DIR="$(policy_value "$POLICY" state_dir)/receipts"
  fi
  mkdir -p "$RECEIPT_DIR"
}

backend() {
  local output status command_text
  printf -v command_text '%q ' "$BACKEND" --policy "$POLICY" --freeze "$FREEZE" "$@"
  if output="$(timeout "$COMMAND_TIMEOUT" "$BACKEND" --policy "$POLICY" --freeze "$FREEZE" "$@" 2>&1)"; then
    status=0
  else
    status=$?
  fi
  write_material_result "$command_text" "$status" "$output"
  if [[ $status -eq 0 ]]; then
    printf '%s\n' "$output"
  else
    printf '%s\n' "$output" >&2
  fi
  return "$status"
}

write_material_result() {
  local command_text="$1" status="$2" output="$3" receipt result decision_hash=none
  receipt="$RECEIPT_DIR/material-$(date -u +%Y%m%dT%H%M%S%N)-$$.receipt"
  [[ -z "$LAST_RECEIPT" || ! -r "$LAST_RECEIPT" ]] || decision_hash="$(sha256_file "$LAST_RECEIPT")"
  if [[ "$status" == 0 ]]; then result=PASS; else result=FAIL; fi
  {
    printf 'schema=sounio-spark-pair-material-result-v1\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'sounio_source_sha256=%s\n' "$(policy_value "$FREEZE" authority_sha256)"
    printf 'semantics_freeze_sha256=%s\n' "$(sha256_file "$FREEZE")"
    printf 'decision_producer_language=Sounio\n'
    printf 'decision_language_role=SEMANTIC_AUTHORITY\n'
    printf 'result_producer_language=Bash\n'
    printf 'result_language_role=MATERIAL_BRIDGE\n'
    printf 'toolchain=%s\n' "$(policy_value "$FREEZE" compiler_identity)"
    printf 'hardware=%s:%s,%s:%s\n' \
      "$(policy_value "$POLICY" node_0_k8s)" "$(policy_value "$POLICY" node_0_uid)" \
      "$(policy_value "$POLICY" node_1_k8s)" "$(policy_value "$POLICY" node_1_uid)"
    printf 'command=%s\n' "$command_text"
    printf 'decision_receipt_sha256=%s\n' "$decision_hash"
    printf 'result=%s\n' "$result"
    printf 'exit_status=%s\n' "$status"
    printf 'result_output_sha256=%s\n' "$(printf '%s' "$output" | sha256sum | cut -d ' ' -f 1)"
  } > "$receipt"
  sha256_file "$receipt" > "$receipt.sha256"
}

frame_field() {
  local frame="$1" wanted="$2" token key value found=''
  for token in $frame; do
    key="${token%%=*}"
    value="${token#*=}"
    if [[ "$key" == "$wanted" ]]; then
      [[ -z "$found" ]] || fail "backend duplicated frame field: $wanted"
      found="$value"
    fi
  done
  [[ -n "$found" ]] || fail "backend omitted frame field: $wanted"
  printf '%s\n' "$found"
}

observe() {
  local frame status
  set +e
  frame="$(backend facts --holder "$HOLDER" 2>&1)"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    write_bridge_denial OBSERVATION_FAILED "status=$status output=$(tr '\n' ' ' <<<"$frame")"
    fail 'backend observation failed or timed out'
  fi
  [[ "$frame" != *$'\n'* ]] || fail 'backend emitted a multiline fact frame'
  printf '%s\n' "$frame"
}

write_bridge_denial() {
  local stage="$1" reason="$2" receipt
  receipt="$RECEIPT_DIR/bridge-deny-$(date -u +%Y%m%dT%H%M%S%N)-$$.receipt"
  {
    printf 'schema=%s\n' "$(policy_value "$POLICY" receipt_schema)"
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'sounio_source_sha256=%s\n' "$(policy_value "$FREEZE" authority_sha256)"
    printf 'semantics_freeze_sha256=%s\n' "$(sha256_file "$FREEZE")"
    printf 'receipt_emitter_language=Bash\n'
    printf 'receipt_emitter_role=MATERIAL_BRIDGE\n'
    printf 'decision_producer_language=NONE\n'
    printf 'decision_language_role=NO_SEMANTIC_DECISION\n'
    printf 'stage=%s\n' "$stage"
    printf 'result=DENY\n'
    printf 'reason=%s\n' "$reason"
  } > "$receipt"
  sha256_file "$receipt" > "$receipt.sha256"
}

state_number() {
  case "$1" in
    UNINITIALIZED) printf '0\n' ;;
    SLURM_OWNED) printf '1\n' ;;
    DRAINING_SLURM) printf '2\n' ;;
    SLURM_QUIESCENT) printf '3\n' ;;
    DETACHING_SLURMD) printf '4\n' ;;
    K8S_RESERVING) printf '5\n' ;;
    K8S_OWNED) printf '6\n' ;;
    K8S_RELEASING) printf '7\n' ;;
    VERIFYING_GPU_CLEAN) printf '8\n' ;;
    SLURM_RESTORING) printf '9\n' ;;
    RECOVERY_REQUIRED) printf '10\n' ;;
    *) fail "unknown material state: $1" ;;
  esac
}

action_name() {
  case "$1" in
    1) printf 'BOOTSTRAP_SLURM\n' ;; 2) printf 'BEGIN_ACQUIRE\n' ;;
    3) printf 'CONFIRM_DRAIN\n' ;; 4) printf 'BEGIN_DETACH\n' ;;
    5) printf 'BEGIN_RESERVE\n' ;; 6) printf 'CONFIRM_RESERVATIONS\n' ;;
    7) printf 'HEARTBEAT\n' ;; 8) printf 'BEGIN_RELEASE\n' ;;
    9) printf 'CONFIRM_GPU_CLEAN\n' ;; 10) printf 'BEGIN_RESTORE\n' ;;
    11) printf 'ENTER_RECOVERY\n' ;; 12) printf 'CONFIRM_RESTORE\n' ;;
    13) printf 'RECOVER_TO_SLURM\n' ;; 14) printf 'STATUS\n' ;;
    15) printf 'RECOVERY_CLEAR_K8S\n' ;; 16) printf 'RECOVERY_PROBE_CLEAN\n' ;;
    17) printf 'RECOVERY_DELETE_RESERVATIONS\n' ;; 18) printf 'RECOVERY_RESTORE_SLURMD\n' ;;
    19) printf 'RECOVERY_RESUME_SLURM\n' ;;
    20) printf 'RECOVERY_DETACH_SLURMD\n' ;; 21) printf 'RECOVERY_CREATE_RESERVATIONS\n' ;;
    22) printf 'RECOVERY_DRAIN_SLURM\n' ;;
    23) printf 'BOOTSTRAP_DRAIN_SLURM\n' ;;
    24) printf 'BOOTSTRAP_INSTALL_FENCE\n' ;;
    25) printf 'BOOTSTRAP_INSTALL_SLURMD\n' ;;
    26) printf 'BOOTSTRAP_RESUME_SLURM\n' ;;
    27) printf 'BOOTSTRAP_TAKEOVER\n' ;;
    28) printf 'BOOTSTRAP_INITIALIZE\n' ;;
    29) printf 'INSTALL_HOST_FENCE\n' ;;
    30) printf 'FENCE_HOST_PAIR\n' ;;
    31) printf 'GRANT_HOST_SLURM\n' ;;
    32) printf 'GRANT_HOST_K8S\n' ;;
    *) fail "unknown action code: $1" ;;
  esac
}

write_receipt() {
  local action="$1" epoch="$2" frame="$3" result="$4" expected_to="$5" receipt source_hash freeze_hash toolchain hardware
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  toolchain="$(policy_value "$FREEZE" compiler_identity)"
  hardware="controller=$(uname -m):$(hostname),nodes=$(policy_value "$POLICY" node_0_k8s):$(policy_value "$POLICY" node_0_uid),$(policy_value "$POLICY" node_1_k8s):$(policy_value "$POLICY" node_1_uid),nodeset=$(policy_value "$POLICY" nodeset_uid)"
  receipt="$RECEIPT_DIR/epoch-${epoch}-$(date -u +%Y%m%dT%H%M%S%N)-action-${action}.receipt"
  {
    printf 'schema=%s\n' "$(policy_value "$POLICY" receipt_schema)"
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'sounio_source_sha256=%s\n' "$source_hash"
    printf 'semantics_freeze_sha256=%s\n' "$freeze_hash"
    printf 'receipt_emitter_language=Bash\n'
    printf 'receipt_emitter_role=MATERIAL_BRIDGE\n'
    printf 'decision_producer_language=Sounio\n'
    printf 'decision_language_role=SEMANTIC_AUTHORITY\n'
    printf 'decision_executable_sha256=%s\n' "$(sha256_file "$AUTHORITY")"
    printf 'material_policy_sha256=%s\n' "$(sha256_file "$POLICY")"
    printf 'material_backend_sha256=%s\n' "$(sha256_file "$BACKEND")"
    printf 'admission_manifest_sha256=%s\n' "$(policy_value "$FREEZE" admission_manifest_sha256)"
    printf 'host_fence_manifest_sha256=%s\n' "$(policy_value "$FREEZE" host_fence_manifest_sha256)"
    printf 'toolchain=%s\n' "$toolchain"
    printf 'hardware=%s\n' "$hardware"
    printf 'command=%s %s\n' "$AUTHORITY" "$action"
    printf 'action_code=%s\n' "$action"
    printf 'epoch=%s\n' "$epoch"
    printf 'from_state=%s\n' "$(frame_field "$frame" state)"
    printf 'expected_to_state=%s\n' "$expected_to"
    printf 'frame=%s\n' "$frame"
    printf 'result=%s\n' "$result"
  } > "$receipt"
  sha256_file "$receipt" > "$receipt.sha256"
  LAST_RECEIPT="$receipt"
}

admit_frame() {
  local action="$1" expected_state="$2" expected_to="$3" frame="$4" state epoch observed authority_mask slurm_mask k8s_mask host_mask result status
  state="$(frame_field "$frame" state)"
  epoch="$(frame_field "$frame" epoch)"
  observed="$(frame_field "$frame" observed_epoch)"
  authority_mask="$(frame_field "$frame" authority_mask)"
  slurm_mask="$(frame_field "$frame" slurm_mask)"
  k8s_mask="$(frame_field "$frame" k8s_mask)"
  host_mask="$(frame_field "$frame" host_mask)"
  [[ "$state" == "$expected_state" ]] || fail "expected $expected_state, observed $state"
  set +e
  result="$(timeout "$COMMAND_TIMEOUT" "$AUTHORITY" 9025 "$action" "$(state_number "$state")" \
    "$epoch" "$observed" "$authority_mask" "$slurm_mask" "$k8s_mask" "$host_mask" 2>&1)"
  status=$?
  set -e
  write_receipt "$action" "$epoch" "$frame" "$result" "$expected_to"
  [[ $status -eq 0 ]] || fail "Sounio refused action $action: $result"
  [[ "$result" == SOUNIO_SPARK_PAIR_ALLOW* ]] || fail "malformed Sounio result: $result"
  [[ "$(frame_field "$result" action)" == "$(action_name "$action")" ]] || fail 'Sounio action binding mismatch'
  [[ "$(frame_field "$result" from)" == "$expected_state" ]] || fail 'Sounio source-state binding mismatch'
  [[ "$(frame_field "$result" to)" == "$expected_to" ]] || fail 'Sounio destination-state binding mismatch'
  [[ "$(frame_field "$result" code)" == 0 ]] || fail 'Sounio result code is not zero'
  CURRENT_EPOCH="$epoch"
}

admit() {
  local action="$1" expected_state="$2" expected_to="$3" frame
  frame="$(observe)"
  admit_frame "$action" "$expected_state" "$expected_to" "$frame"
}

transition() {
  local action="$1" from="$2" to="$3"
  admit "$action" "$from" "$to"
  backend lease-transition --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --from "$from" --to "$to" \
    --receipt "$LAST_RECEIPT" >/dev/null
}

enter_recovery() {
  local frame state
  [[ $CURRENT_EPOCH -gt 0 ]] || return 1
  frame="$(observe)" || return 1
  state="$(frame_field "$frame" state)" || return 1
  [[ "$state" == RECOVERY_REQUIRED ]] && return 0
  transition 11 "$state" RECOVERY_REQUIRED
}

authorize_stay() {
  local action="$1" state="$2"
  admit "$action" "$state" "$state"
}

rollback_to_slurm() {
  local frame state
  [[ $RECOVERY_ACTIVE -eq 0 ]] || return 1
  RECOVERY_ACTIVE=1
  enter_recovery || return 1
  authorize_stay 29 RECOVERY_REQUIRED || return 1
  backend install-host-fence --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 15 RECOVERY_REQUIRED || return 1
  backend stop-workloads --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  backend delete-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 22 RECOVERY_REQUIRED || return 1
  backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 30 RECOVERY_REQUIRED || return 1
  backend fence-host-pair --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 20 RECOVERY_REQUIRED || return 1
  backend detach-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 31 RECOVERY_REQUIRED || return 1
  backend grant-host-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 18 RECOVERY_REQUIRED || return 1
  backend restore-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 19 RECOVERY_REQUIRED || return 1
  backend resume-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  frame="$(observe)" || return 1
  state="$(frame_field "$frame" state)" || return 1
  [[ "$state" == RECOVERY_REQUIRED ]] || return 1
  admit 13 RECOVERY_REQUIRED SLURM_OWNED || return 1
  backend lease-transition --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
    --from RECOVERY_REQUIRED --to SLURM_OWNED --receipt "$LAST_RECEIPT" >/dev/null || return 1
  PAIR_ACQUIRED=0
  RECOVERY_ACTIVE=0
}

acquire_pair() {
  local lease reserve_receipt reservation_pid grant_heartbeat_pid reserve_status=0
  lease="$(backend lease-acquire --holder "$HOLDER")" || fail 'Lease acquisition refused'
  CURRENT_EPOCH="$(frame_field "$lease" epoch)"
  [[ "$CURRENT_EPOCH" =~ ^[1-9][0-9]*$ ]] || fail 'backend returned an invalid epoch'

  PAIR_ACQUIRED=1
  transition 2 SLURM_OWNED DRAINING_SLURM
  backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 3 DRAINING_SLURM SLURM_QUIESCENT
  authorize_stay 30 SLURM_QUIESCENT
  backend fence-host-pair --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 4 SLURM_QUIESCENT DETACHING_SLURMD
  backend detach-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 5 DETACHING_SLURMD K8S_RESERVING
  reserve_receipt="$LAST_RECEIPT"
  authorize_stay 32 K8S_RESERVING
  backend grant-host-k8s --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  backend create-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
    --receipt "$reserve_receipt" >/dev/null &
  reservation_pid=$!
  reservation_grant_heartbeat_loop &
  grant_heartbeat_pid=$!
  while kill -0 "$reservation_pid" >/dev/null 2>&1; do
    sleep 1
    if ! kill -0 "$grant_heartbeat_pid" >/dev/null 2>&1; then
      kill "$reservation_pid" >/dev/null 2>&1 || true
      wait "$reservation_pid" >/dev/null 2>&1 || true
      return 42
    fi
  done
  wait "$reservation_pid" || reserve_status=$?
  kill "$grant_heartbeat_pid" >/dev/null 2>&1 || true
  wait "$grant_heartbeat_pid" >/dev/null 2>&1 || true
  [[ $reserve_status -eq 0 ]] || return "$reserve_status"
  transition 6 K8S_RESERVING K8S_OWNED
}

reservation_grant_heartbeat_loop() {
  local interval
  interval="$(policy_value "$POLICY" heartbeat_seconds)"
  while true; do
    sleep "$interval"
    authorize_stay 32 K8S_RESERVING
    backend grant-host-k8s --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
      --receipt "$LAST_RECEIPT" >/dev/null
  done
}

release_pair() {
  transition 8 K8S_OWNED K8S_RELEASING
  backend stop-workloads --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 30 K8S_RELEASING
  backend fence-host-pair --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 9 K8S_RELEASING VERIFYING_GPU_CLEAN
  transition 10 VERIFYING_GPU_CLEAN SLURM_RESTORING
  backend delete-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 31 SLURM_RESTORING
  backend grant-host-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  backend restore-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  backend resume-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 12 SLURM_RESTORING SLURM_OWNED
  PAIR_ACQUIRED=0
}

heartbeat_loop() {
  local interval
  interval="$(policy_value "$POLICY" heartbeat_seconds)"
  while true; do
    sleep "$interval"
    authorize_stay 32 K8S_OWNED
    backend grant-host-k8s --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
    admit 7 K8S_OWNED K8S_OWNED
    backend lease-renew --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  done
}

hold_pair() {
  local seconds="$1" heartbeat_pid hold_status=0 deadline
  acquire_pair
  heartbeat_loop &
  heartbeat_pid=$!
  deadline=$((SECONDS + seconds))
  while (( SECONDS < deadline )); do
    sleep 1
    if ! kill -0 "$heartbeat_pid" >/dev/null 2>&1; then
      hold_status=42
      break
    fi
  done
  kill "$heartbeat_pid" >/dev/null 2>&1 || true
  wait "$heartbeat_pid" >/dev/null 2>&1 || true
  [[ $hold_status -eq 0 ]] || return "$hold_status"
  release_pair
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  if [[ $status -ne 0 && $PAIR_ACQUIRED -eq 1 ]]; then
    printf 'spark-pair-arbiter: transition failed; attempting fenced rollback\n' >&2
    if ! rollback_to_slurm; then
      printf 'spark-pair-arbiter: RECOVERY_REQUIRED; Slurm remains fenced\n' >&2
    fi
  fi
  exit "$status"
}

usage() {
  printf 'Usage: %s verify | bootstrap-init | bootstrap | bootstrap-recover | bootstrap-migrate-recover | status | hold SECONDS | recover\n' "$0" >&2
  exit 64
}

bootstrap_sequence() {
  authorize_stay 24 UNINITIALIZED
  backend install-fence --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 29 UNINITIALIZED
  backend install-host-fence --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 23 UNINITIALIZED
  backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 30 UNINITIALIZED
  backend fence-host-pair --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 31 UNINITIALIZED
  backend grant-host-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 25 UNINITIALIZED
  backend install-gpu-bound-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  authorize_stay 26 UNINITIALIZED
  backend resume-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 1 UNINITIALIZED SLURM_OWNED
}

main() {
  local command="${1:-}" seconds frame state migration_needed
  verify_runtime_root
  configure_command_timeout
  verify_frozen_authority
  if [[ "$command" != verify ]]; then init_receipt_dir; fi
  trap on_exit EXIT INT TERM
  case "$command" in
    verify)
      printf 'SPARK_PAIR_VERIFY_PASS source=%s freeze=%s backend=%s admission=%s host_fence=%s\n' \
        "$(policy_value "$FREEZE" authority_sha256)" "$(sha256_file "$FREEZE")" \
        "$(policy_value "$FREEZE" material_backend_sha256)" \
        "$(policy_value "$FREEZE" admission_manifest_sha256)" \
        "$(policy_value "$FREEZE" host_fence_manifest_sha256)"
      ;;
    bootstrap-init)
      frame="$(backend prebootstrap-facts --holder "$HOLDER")"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      admit_frame 28 UNINITIALIZED UNINITIALIZED "$frame"
      frame="$(backend bootstrap-lease --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
        --receipt "$LAST_RECEIPT")"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      bootstrap_sequence
      printf 'SPARK_PAIR_BOOTSTRAP_PASS epoch=%s\n' "$CURRENT_EPOCH"
      ;;
    bootstrap)
      frame="$(observe)"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      bootstrap_sequence
      printf 'SPARK_PAIR_BOOTSTRAP_PASS epoch=%s\n' "$CURRENT_EPOCH"
      ;;
    bootstrap-recover)
      frame="$(observe)"
      state="$(frame_field "$frame" state)"
      [[ "$state" == UNINITIALIZED ]] || fail "bootstrap recovery requires UNINITIALIZED, observed $state"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      admit 27 UNINITIALIZED UNINITIALIZED
      frame="$(backend lease-bootstrap-recovery-acquire --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
        --receipt "$LAST_RECEIPT")"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      bootstrap_sequence
      printf 'SPARK_PAIR_BOOTSTRAP_RECOVERY_PASS epoch=%s\n' "$CURRENT_EPOCH"
      ;;
    bootstrap-migrate-recover)
      migration_needed=0
      if frame="$(backend bootstrap-migration-facts --holder "$HOLDER" 2>/dev/null)"; then
        migration_needed=1
      else
        frame="$(observe)"
      fi
      state="$(frame_field "$frame" state)"
      [[ "$state" == UNINITIALIZED ]] || fail "bootstrap migration requires UNINITIALIZED, observed $state"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      admit_frame 27 UNINITIALIZED UNINITIALIZED "$frame"
      if [[ "$migration_needed" == 1 ]]; then
        backend bootstrap-migrate-freeze --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
          --receipt "$LAST_RECEIPT" >/dev/null
      fi
      frame="$(backend lease-bootstrap-recovery-acquire --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
        --receipt "$LAST_RECEIPT")"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      bootstrap_sequence
      printf 'SPARK_PAIR_BOOTSTRAP_MIGRATION_RECOVERY_PASS epoch=%s\n' "$CURRENT_EPOCH"
      ;;
    status)
      frame="$(observe)"
      state="$(frame_field "$frame" state)"
      admit 14 "$state" "$state"
      printf 'SPARK_PAIR_STATUS %s\n' "$frame"
      ;;
    hold)
      seconds="${2:-}"
      [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || usage
      hold_pair "$seconds"
      printf 'SPARK_PAIR_HOLD_PASS epoch=%s seconds=%s\n' "$CURRENT_EPOCH" "$seconds"
      ;;
    recover)
      frame="$(observe)"
      state="$(frame_field "$frame" state)"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      admit 11 "$state" RECOVERY_REQUIRED
      frame="$(backend lease-recovery-acquire --holder "$HOLDER" --epoch "$CURRENT_EPOCH" \
        --from "$state" --receipt "$LAST_RECEIPT")"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      rollback_to_slurm || fail 'manual recovery could not prove Slurm ownership'
      printf 'SPARK_PAIR_RECOVERY_PASS epoch=%s\n' "$CURRENT_EPOCH"
      ;;
    *) usage ;;
  esac
  trap - EXIT INT TERM
}

main "$@"
