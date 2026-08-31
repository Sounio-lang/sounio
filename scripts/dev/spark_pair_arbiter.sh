#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
POLICY="${SOUNIO_SPARK_PAIR_POLICY:-$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1}"
FREEZE="${SOUNIO_SPARK_PAIR_FREEZE:-$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1}"
AUTHORITY="${SOUNIO_SPARK_PAIR_AUTHORITY:-$ROOT_DIR/tools/cluster/_build/default/sounio-spark-pair-arbiter}"
BUILD="${SOUNIO_SPARK_PAIR_BUILD:-$ROOT_DIR/scripts/dev/build_sounio_spark_pair_arbiter.sh}"
BACKEND="${SOUNIO_SPARK_PAIR_BACKEND:-$ROOT_DIR/scripts/dev/spark_pair_arbiter_k8s_backend.sh}"
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
  verify_frozen_file material_policy_source material_policy_sha256
  verify_frozen_file material_controller_source material_controller_sha256
  verify_frozen_file material_backend_source material_backend_sha256
  verify_frozen_file admission_manifest_source admission_manifest_sha256
  verify_frozen_file installer_source installer_sha256
  verify_frozen_file selftest_source selftest_sha256
  verify_frozen_file live_gate_source live_gate_sha256

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
  timeout "$COMMAND_TIMEOUT" "$BACKEND" --policy "$POLICY" --freeze "$FREEZE" "$@"
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
  local frame
  frame="$(backend facts --holder "$HOLDER")" || fail 'backend observation failed or timed out'
  [[ "$frame" != *$'\n'* ]] || fail 'backend emitted a multiline fact frame'
  printf '%s\n' "$frame"
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
    *) fail "unknown action code: $1" ;;
  esac
}

write_receipt() {
  local action="$1" epoch="$2" frame="$3" result="$4" expected_to="$5" receipt source_hash freeze_hash toolchain hardware
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  toolchain="$(policy_value "$FREEZE" compiler_identity)"
  hardware="controller=$(uname -m):$(hostname),nodes=$(policy_value "$POLICY" node_0_k8s):$(policy_value "$POLICY" node_0_uid),$(policy_value "$POLICY" node_1_k8s):$(policy_value "$POLICY" node_1_uid),nodeset=$(policy_value "$POLICY" nodeset_uid)"
  receipt="$RECEIPT_DIR/epoch-${epoch}-$(date -u +%Y%m%dT%H%M%S)-action-${action}.receipt"
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

admit() {
  local action="$1" expected_state="$2" expected_to="$3" frame state epoch observed authority_mask slurm_mask k8s_mask result status
  frame="$(observe)"
  state="$(frame_field "$frame" state)"
  epoch="$(frame_field "$frame" epoch)"
  observed="$(frame_field "$frame" observed_epoch)"
  authority_mask="$(frame_field "$frame" authority_mask)"
  slurm_mask="$(frame_field "$frame" slurm_mask)"
  k8s_mask="$(frame_field "$frame" k8s_mask)"
  [[ "$state" == "$expected_state" ]] || fail "expected $expected_state, observed $state"
  set +e
  result="$(timeout "$COMMAND_TIMEOUT" "$AUTHORITY" 9024 "$action" "$(state_number "$state")" \
    "$epoch" "$observed" "$authority_mask" "$slurm_mask" "$k8s_mask" 2>&1)"
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
  authorize_stay 15 RECOVERY_REQUIRED || return 1
  backend stop-workloads --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  backend delete-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 22 RECOVERY_REQUIRED || return 1
  backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 20 RECOVERY_REQUIRED || return 1
  backend detach-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 21 RECOVERY_REQUIRED || return 1
  backend create-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 16 RECOVERY_REQUIRED || return 1
  backend probe-clean --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
  authorize_stay 17 RECOVERY_REQUIRED || return 1
  backend delete-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null || return 1
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
  local lease
  lease="$(backend lease-acquire --holder "$HOLDER")" || fail 'Lease acquisition refused'
  CURRENT_EPOCH="$(frame_field "$lease" epoch)"
  [[ "$CURRENT_EPOCH" =~ ^[1-9][0-9]*$ ]] || fail 'backend returned an invalid epoch'

  transition 2 SLURM_OWNED DRAINING_SLURM
  PAIR_ACQUIRED=1
  backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 3 DRAINING_SLURM SLURM_QUIESCENT
  transition 4 SLURM_QUIESCENT DETACHING_SLURMD
  backend detach-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 5 DETACHING_SLURMD K8S_RESERVING
  backend create-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 6 K8S_RESERVING K8S_OWNED
}

release_pair() {
  transition 8 K8S_OWNED K8S_RELEASING
  backend stop-workloads --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  backend probe-clean --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
  transition 9 K8S_RELEASING VERIFYING_GPU_CLEAN
  transition 10 VERIFYING_GPU_CLEAN SLURM_RESTORING
  backend delete-reservations --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
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
  printf 'Usage: %s verify | bootstrap | status | hold SECONDS | recover\n' "$0" >&2
  exit 64
}

main() {
  local command="${1:-}" seconds frame state
  verify_frozen_authority
  if [[ "$command" != verify ]]; then init_receipt_dir; fi
  trap on_exit EXIT INT TERM
  case "$command" in
    verify)
      printf 'SPARK_PAIR_VERIFY_PASS source=%s freeze=%s backend=%s admission=%s\n' \
        "$(policy_value "$FREEZE" authority_sha256)" "$(sha256_file "$FREEZE")" \
        "$(policy_value "$FREEZE" material_backend_sha256)" \
        "$(policy_value "$FREEZE" admission_manifest_sha256)"
      ;;
    bootstrap)
      frame="$(observe)"
      CURRENT_EPOCH="$(frame_field "$frame" epoch)"
      authorize_stay 23 UNINITIALIZED
      backend drain-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
      authorize_stay 24 UNINITIALIZED
      backend install-fence --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
      authorize_stay 25 UNINITIALIZED
      backend install-gpu-bound-slurmd --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
      authorize_stay 26 UNINITIALIZED
      backend resume-slurm --holder "$HOLDER" --epoch "$CURRENT_EPOCH" --receipt "$LAST_RECEIPT" >/dev/null
      transition 1 UNINITIALIZED SLURM_OWNED
      printf 'SPARK_PAIR_BOOTSTRAP_PASS epoch=%s\n' "$CURRENT_EPOCH"
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
