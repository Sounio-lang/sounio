#!/usr/bin/env bash

set -euo pipefail
umask 077

MOCK_DIR="${SOUNIO_SPARK_PAIR_MOCK_DIR:?SOUNIO_SPARK_PAIR_MOCK_DIR is required}"
mkdir -p "$MOCK_DIR"
exec 9>"$MOCK_DIR/lock"
flock -x 9

fail() {
  printf 'mock-spark-pair-backend: FAIL: %s\n' "$*" >&2
  exit 42
}

read_value() {
  local key="$1" fallback="$2"
  if [[ -f "$MOCK_DIR/$key" ]]; then
    sed -n '1p' "$MOCK_DIR/$key"
  else
    printf '%s\n' "$fallback"
  fi
}

write_value() {
  printf '%s\n' "$2" > "$MOCK_DIR/$1"
}

arg_value() {
  local wanted="$1"
  shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$wanted" ]]; then
      [[ $# -ge 2 ]] || fail "missing value for $wanted"
      printf '%s\n' "$2"
      return 0
    fi
    shift
  done
  fail "missing argument $wanted"
}

maybe_fail() {
  local command="$1" marker
  if [[ "${SOUNIO_SPARK_PAIR_MOCK_SLEEP_COMMAND:-}" == "$command" ]]; then
    sleep "${SOUNIO_SPARK_PAIR_MOCK_SLEEP_SECONDS:-5}"
  fi
  if [[ "${SOUNIO_SPARK_PAIR_MOCK_FAIL:-}" == "$command" ]]; then
    marker="$MOCK_DIR/failure-injected-$command"
    if [[ ! -f "$marker" ]]; then
      : > "$marker"
      fail "injected $command failure"
    fi
  fi
}

add_bit() {
  local mask="$1" power="$2" truth="$3"
  if [[ "$truth" == 1 ]]; then printf '%s\n' "$((mask + power))"; else printf '%s\n' "$mask"; fi
}

receipt_value() {
  local receipt="$1" key="$2"
  [[ -r "$receipt" && -r "$receipt.sha256" ]] || fail 'receipt missing'
  sed -n "s/^${key}=//p" "$receipt"
}

verify_receipt() {
  local receipt="$1" epoch="$2" actions="$3" action digest
  digest="$(sed -n '1p' "$receipt.sha256")"
  [[ "$digest" == "$(sha256sum "$receipt" | cut -d ' ' -f 1)" ]] || fail 'receipt digest mismatch'
  [[ "$(receipt_value "$receipt" epoch)" == "$epoch" ]] || fail 'receipt epoch mismatch'
  [[ "$(receipt_value "$receipt" decision_producer_language)" == Sounio ]] || fail 'receipt is not a Sounio decision'
  action="$(receipt_value "$receipt" action_code)"
  case " $actions " in *" $action "*) ;; *) fail "receipt action $action denied" ;; esac
}

guard() {
  local actions="$1" holder epoch receipt state
  shift
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  state="$(read_value state '')"
  [[ "$(read_value holder '')" == "$holder" ]] || fail 'holder mismatch'
  [[ "$(read_value epoch 0)" == "$epoch" ]] || fail 'epoch mismatch'
  verify_receipt "$receipt" "$epoch" "$actions"
  [[ "$(receipt_value "$receipt" expected_to_state)" == "$state" ]] || fail 'receipt destination mismatch'
}

facts() {
  local holder state epoch lease_holder authority=1 slurm=0 k8s=0 live reservations
  holder="$(arg_value --holder "$@")"
  state="$(read_value state SLURM_OWNED)"
  epoch="$(read_value epoch 1)"
  lease_holder="$(read_value holder slurm-owned)"
  if [[ "$(read_value recovered_live 0)" == 1 ]]; then
    live=1
  else
    live="${SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE:-1}"
  fi
  reservations="$(read_value reservations 0)"

  [[ "$lease_holder" == "$holder" ]] && authority="$(add_bit "$authority" 8 1)"
  authority="$(add_bit "$authority" 16 "$live")"
  authority="$(add_bit "$authority" 32 1)"
  authority="$(add_bit "$authority" 64 "${SOUNIO_SPARK_PAIR_MOCK_NODESET_MATCH:-$(read_value nodeset_match 1)}")"
  authority="$(add_bit "$authority" 128 "${SOUNIO_SPARK_PAIR_MOCK_PLUGIN_EXCLUSIVE:-1}")"
  authority="$(add_bit "$authority" 256 "${SOUNIO_SPARK_PAIR_MOCK_ADMISSION_READY:-$(read_value admission_ready 1)}")"
  authority="$(add_bit "$authority" 512 "${SOUNIO_SPARK_PAIR_MOCK_TAINT_EXACT:-$(read_value taint_exact 1)}")"

  slurm="$(add_bit "$slurm" 1 "${SOUNIO_SPARK_PAIR_MOCK_GPU_REQUEST_EXACT:-1}")"
  slurm="$(add_bit "$slurm" 2 "${SOUNIO_SPARK_PAIR_MOCK_SLURM_READY:-1}")"
  slurm="$(add_bit "$slurm" 4 "$(read_value drained 0)")"
  slurm="$(add_bit "$slurm" 8 "${SOUNIO_SPARK_PAIR_MOCK_JOBS_ZERO:-1}")"
  slurm="$(add_bit "$slurm" 16 "${SOUNIO_SPARK_PAIR_MOCK_ALLOCATIONS_ZERO:-1}")"
  slurm="$(add_bit "$slurm" 32 "$(read_value slurmd_absent 0)")"
  slurm="$(add_bit "$slurm" 64 "$(read_value slurmd_bound 1)")"
  slurm="$(add_bit "$slurm" 128 "$(read_value resume_verified 1)")"

  k8s="$(add_bit "$k8s" 1 "${SOUNIO_SPARK_PAIR_MOCK_CAPACITY_ONE:-1}")"
  [[ "$reservations" == 2 ]] && k8s="$(add_bit "$k8s" 2 1)"
  [[ "$reservations" == 0 ]] && k8s="$(add_bit "$k8s" 256 1)"
  [[ "$reservations" == 2 ]] && k8s="$(add_bit "$k8s" 4 1)"
  [[ "$reservations" == 2 ]] && k8s="$(add_bit "$k8s" 8 1)"
  k8s="$(add_bit "$k8s" 16 "$(read_value nvml_clean 0)")"
  k8s="$(add_bit "$k8s" 32 "$(read_value workloads_zero 1)")"
  k8s="$(add_bit "$k8s" 64 "$(read_value stale_zero 1)")"
  k8s="$(add_bit "$k8s" 128 "$live")"
  k8s="$(add_bit "$k8s" 512 "${SOUNIO_SPARK_PAIR_MOCK_UNEXPECTED_ZERO:-1}")"

  authority="${SOUNIO_SPARK_PAIR_MOCK_AUTHORITY_MASK:-$authority}"
  slurm="${SOUNIO_SPARK_PAIR_MOCK_SLURM_MASK:-$slurm}"
  k8s="${SOUNIO_SPARK_PAIR_MOCK_K8S_MASK:-$k8s}"
  printf 'state=%s epoch=%s observed_epoch=%s authority_mask=%s slurm_mask=%s k8s_mask=%s\n' \
    "$state" "$epoch" "${SOUNIO_SPARK_PAIR_MOCK_OBSERVED_EPOCH:-$epoch}" "$authority" "$slurm" "$k8s"
}

lease_acquire() {
  local holder state epoch
  [[ "${SOUNIO_SPARK_PAIR_MOCK_FREEZE_BOUND:-1}" == 1 ]] || fail 'mock Lease freeze binding mismatch'
  holder="$(arg_value --holder "$@")"
  state="$(read_value state SLURM_OWNED)"
  [[ "$state" == SLURM_OWNED ]] || fail "state $state is not acquirable"
  epoch=$(( $(read_value epoch 1) + 1 ))
  write_value epoch "$epoch"
  write_value holder "$holder"
  write_value lease_live 1
  printf 'epoch=%s state=SLURM_OWNED\n' "$epoch"
}

lease_transition() {
  local holder epoch from to receipt action
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  from="$(arg_value --from "$@")"
  to="$(arg_value --to "$@")"
  receipt="$(arg_value --receipt "$@")"
  [[ "$(read_value holder '')" == "$holder" ]] || fail 'holder mismatch'
  [[ "$(read_value epoch 0)" == "$epoch" ]] || fail 'epoch mismatch'
  [[ "$(read_value state '')" == "$from" ]] || fail 'state mismatch'
  case "$from:$to" in
    UNINITIALIZED:SLURM_OWNED) action=1 ;;
    SLURM_OWNED:DRAINING_SLURM) action=2 ;;
    DRAINING_SLURM:SLURM_QUIESCENT) action=3 ;;
    SLURM_QUIESCENT:DETACHING_SLURMD) action=4 ;;
    DETACHING_SLURMD:K8S_RESERVING) action=5 ;;
    K8S_RESERVING:K8S_OWNED) action=6 ;;
    K8S_OWNED:K8S_RELEASING) action=8 ;;
    K8S_RELEASING:VERIFYING_GPU_CLEAN) action=9 ;;
    VERIFYING_GPU_CLEAN:SLURM_RESTORING) action=10 ;;
    SLURM_RESTORING:SLURM_OWNED) action=12 ;;
    RECOVERY_REQUIRED:SLURM_OWNED) action=13 ;;
    *:RECOVERY_REQUIRED) action=11 ;;
    *) fail 'unsupported transition' ;;
  esac
  verify_receipt "$receipt" "$epoch" "$action"
  [[ "$(receipt_value "$receipt" from_state)" == "$from" ]] || fail 'receipt source mismatch'
  [[ "$(receipt_value "$receipt" expected_to_state)" == "$to" ]] || fail 'receipt destination mismatch'
  write_value state "$to"
  if [[ "$to" == SLURM_OWNED ]]; then write_value holder slurm-owned; fi
}

lease_recovery_acquire() {
  local holder epoch from receipt next
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  from="$(arg_value --from "$@")"
  receipt="$(arg_value --receipt "$@")"
  [[ "$(read_value epoch 0)" == "$epoch" ]] || fail 'recovery epoch mismatch'
  [[ "$(read_value state '')" == "$from" ]] || fail 'recovery state mismatch'
  verify_receipt "$receipt" "$epoch" 11
  [[ "$(receipt_value "$receipt" from_state)" == "$from" ]] || fail 'recovery receipt source mismatch'
  [[ "$(receipt_value "$receipt" expected_to_state)" == RECOVERY_REQUIRED ]] || fail 'recovery receipt destination mismatch'
  next=$((epoch + 1))
  write_value epoch "$next"
  write_value holder "$holder"
  write_value state RECOVERY_REQUIRED
  write_value recovered_live 1
  printf 'epoch=%s state=RECOVERY_REQUIRED\n' "$next"
}

lease_bootstrap_recovery_acquire() {
  local holder epoch receipt live stored_holder next
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  [[ "${SOUNIO_SPARK_PAIR_MOCK_FREEZE_BOUND:-1}" == 1 ]] || fail 'mock Lease freeze binding mismatch'
  [[ "${SOUNIO_SPARK_PAIR_MOCK_JOURNAL_BOUND:-1}" == 1 ]] || fail 'mock journal freeze binding mismatch'
  [[ "$(read_value epoch 0)" == "$epoch" ]] || fail 'bootstrap recovery epoch mismatch'
  [[ "$(read_value state '')" == UNINITIALIZED ]] || fail 'bootstrap recovery state mismatch'
  verify_receipt "$receipt" "$epoch" 27
  [[ "$(receipt_value "$receipt" from_state)" == UNINITIALIZED ]] || fail 'bootstrap recovery receipt source mismatch'
  [[ "$(receipt_value "$receipt" expected_to_state)" == UNINITIALIZED ]] || fail 'bootstrap recovery receipt destination mismatch'
  live="${SOUNIO_SPARK_PAIR_MOCK_LEASE_LIVE:-1}"
  stored_holder="$(read_value holder '')"
  if [[ "$live" == 1 && "$stored_holder" != "$holder" ]]; then
    fail "cannot recover live bootstrap held by $stored_holder"
  fi
  next=$((epoch + 1))
  write_value epoch "$next"
  write_value holder "$holder"
  write_value state UNINITIALIZED
  write_value recovered_live 1
  write_value nodeset_match 1
  write_value journal 1
  printf 'epoch=%s state=UNINITIALIZED\n' "$next"
}

main() {
  [[ "${1:-}" == --policy && $# -ge 5 ]] || fail 'bad backend invocation'
  shift 2
  [[ "${1:-}" == --freeze ]] || fail 'bad backend invocation'
  shift 2
  local command="${1:-}"
  shift || true
  maybe_fail "$command"
  case "$command" in
    prebootstrap-facts)
      [[ ! -f "$MOCK_DIR/state" ]] || fail 'mock Lease already exists'
      printf 'state=UNINITIALIZED epoch=1 observed_epoch=1 authority_mask=97 slurm_mask=26 k8s_mask=768\n'
      ;;
    bootstrap-lease)
      [[ ! -f "$MOCK_DIR/state" ]] || fail 'mock Lease already exists'
      holder="$(arg_value --holder "$@")"
      epoch="$(arg_value --epoch "$@")"
      receipt="$(arg_value --receipt "$@")"
      [[ "$epoch" == 1 ]] || fail 'initial epoch mismatch'
      verify_receipt "$receipt" "$epoch" 28
      write_value state UNINITIALIZED
      write_value epoch 1
      write_value holder "$holder"
      write_value drained 0
      write_value slurmd_absent 0
      write_value slurmd_bound 1
      write_value reservations 0
      write_value nvml_clean 0
      write_value workloads_zero 1
      write_value stale_zero 1
      write_value resume_verified 1
      write_value nodeset_match 1
      write_value admission_ready 0
      write_value taint_exact 0
      if [[ "${SOUNIO_SPARK_PAIR_MOCK_FAIL_AFTER_LEASE:-0}" == 1 ]]; then
        write_value journal 0
        fail 'injected action28 failure after Lease creation'
      fi
      write_value journal 1
      printf 'epoch=1 state=UNINITIALIZED\n'
      ;;
    fixture-slurm-owned)
      write_value state SLURM_OWNED
      write_value epoch 1
      write_value holder slurm-owned
      write_value drained 0
      write_value slurmd_absent 0
      write_value slurmd_bound 1
      write_value reservations 0
      write_value nvml_clean 0
      write_value workloads_zero 1
      write_value stale_zero 1
      write_value resume_verified 1
      write_value admission_ready 1
      write_value taint_exact 1
      write_value journal 1
      ;;
    fixture-uninitialized)
      write_value state UNINITIALIZED
      write_value epoch 1
      write_value holder bootstrap-old
      write_value drained 0
      write_value slurmd_absent 0
      write_value slurmd_bound 1
      write_value reservations 0
      write_value nvml_clean 0
      write_value workloads_zero 1
      write_value stale_zero 1
      write_value resume_verified 1
      write_value nodeset_match 1
      write_value admission_ready 0
      write_value taint_exact 0
      write_value journal 1
      ;;
    facts) facts "$@" ;;
    lease-acquire) lease_acquire "$@" ;;
    lease-recovery-acquire) lease_recovery_acquire "$@" ;;
    lease-bootstrap-recovery-acquire) lease_bootstrap_recovery_acquire "$@" ;;
    lease-transition) lease_transition "$@" ;;
    lease-renew)
      [[ "$(read_value holder '')" == "$(arg_value --holder "$@")" ]] || fail 'heartbeat holder mismatch'
      [[ "$(read_value state '')" == K8S_OWNED ]] || fail 'heartbeat outside K8S_OWNED'
      verify_receipt "$(arg_value --receipt "$@")" "$(arg_value --epoch "$@")" 7
      ;;
    enter-recovery) write_value state RECOVERY_REQUIRED ;;
    drain-slurm)
      guard '2 22 23' "$@"
      printf 'drain-slurm\n' >> "$MOCK_DIR/effects"
      write_value drained 1
      write_value resume_verified 0
      ;;
    install-fence)
      guard 24 "$@"
      printf 'install-fence\n' >> "$MOCK_DIR/effects"
      write_value admission_ready 1
      write_value taint_exact 1
      ;;
    install-gpu-bound-slurmd)
      guard 25 "$@"
      write_value nodeset_generation 2
      write_value nodeset_match 1
      write_value slurmd_absent 0
      write_value slurmd_bound 1
      ;;
    detach-slurmd) guard '4 20' "$@"; write_value slurmd_absent 1; write_value slurmd_bound 0 ;;
    create-reservations)
      guard '5 21' "$@"
      if [[ "${SOUNIO_SPARK_PAIR_MOCK_PARTIAL_RESERVATION:-0}" == 1 && \
            ! -f "$MOCK_DIR/partial-reservation-injected" ]]; then
        : > "$MOCK_DIR/partial-reservation-injected"
        write_value reservations 1
      else
        write_value reservations 2
        write_value nvml_clean 1
      fi
      ;;
    stop-workloads)
      guard '8 15' "$@"
      [[ "${SOUNIO_SPARK_PAIR_MOCK_STICKY_WORKLOAD:-0}" != 1 ]] || fail 'injected terminating workload'
      write_value workloads_zero 1
      write_value stale_zero 1
      ;;
    probe-clean) guard '8 16' "$@"; write_value nvml_clean 1 ;;
    delete-reservations) guard '10 15 17' "$@"; write_value reservations 0 ;;
    restore-slurmd) guard '10 18' "$@"; write_value slurmd_absent 0; write_value slurmd_bound 1 ;;
    resume-slurm) guard '10 19 26' "$@"; write_value drained 0; write_value resume_verified 1 ;;
    *) fail "unknown command $command" ;;
  esac
}

main "$@"
