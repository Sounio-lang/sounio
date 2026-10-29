#!/usr/bin/env bash

set -euo pipefail
umask 077

[[ $# -eq 2 ]] || {
  printf 'usage: %s HOST_FENCE_SCRIPT WORK_DIR\n' "$0" >&2
  exit 64
}

HOST_FENCE_SCRIPT="$1"
WORK_DIR="$2"
HOST_ROOT="$WORK_DIR/host"
export NODE_NAME=spark-3c59
export PIREUS_HOST_ROOT="$HOST_ROOT"
export PIREUS_HOST_FENCE_LIBRARY_MODE=1

mkdir -p "$HOST_ROOT/proc/sys/kernel/random" "$HOST_ROOT/var/lib/pireus-spark-pair"
printf '11111111-1111-4111-8111-111111111111\n' > \
  "$HOST_ROOT/proc/sys/kernel/random/boot_id"
printf '100.00 0.00\n' > "$HOST_ROOT/proc/uptime"

# shellcheck source=/dev/null
source "$HOST_FENCE_SCRIPT"

status_once() { return 0; }
monotonic_seconds() { printf '100\n'; }
device_barrier_attach() { return 0; }
device_barrier_detach() { return 0; }
barrier_state=attached
device_barrier_attached() { [[ "$barrier_state" == attached ]]; }
device_barrier_detached() { [[ "$barrier_state" == detached ]]; }

printf 'mode=COMMITTING_SLURM\n' > "$GRANT_FILE"
barrier_state=detached
device_barrier_relation_valid || {
  printf 'host-fence-unit: COMMITTING_SLURM rejected its detached barrier\n' >&2
  exit 1
}
printf 'mode=COMMITTING_K8S\n' > "$GRANT_FILE"
device_barrier_relation_valid || {
  printf 'host-fence-unit: COMMITTING_K8S rejected its detached barrier\n' >&2
  exit 1
}
barrier_state=attached
if device_barrier_relation_valid; then
  printf 'host-fence-unit: transient grant accepted an attached barrier\n' >&2
  exit 1
fi
rm -f "$GRANT_FILE"

# Select the barrier probe from the grant mode. Do not call the probe that
# is required to fail: status-attached on a detached SLURM host writes
# "expected device barrier is not attached" to stderr.
(
  attached_calls=0
  detached_calls=0
  compute_calls=0
  attached_fail=0
  detached_fail=0
  device_barrier_attached() {
    attached_calls=$((attached_calls + 1))
    if [[ "$attached_fail" == 1 ]]; then
      printf 'pireus-device-barrier: expected device barrier is not attached\n' >&2
      return 42
    fi
    return 0
  }
  device_barrier_detached() {
    detached_calls=$((detached_calls + 1))
    if [[ "$detached_fail" == 1 ]]; then
      printf 'pireus-device-barrier: device barrier is unexpectedly attached\n' >&2
      return 42
    fi
    return 0
  }
  external_compute_processes_absent() {
    compute_calls=$((compute_calls + 1))
    return "${compute_rc:-0}"
  }
  external_mps_processes_absent() { return 0; }
  external_gpu_fds_absent() { return 0; }
  consumer_err="$(mktemp)"
  consumer_out="$(mktemp)"
  rc=0
  err=''

  reset_counts() {
    attached_calls=0
    detached_calls=0
    compute_calls=0
    attached_fail=0
    detached_fail=0
    compute_rc=0
  }

  run_consumers() {
    rc=0
    gpu_consumer_set_exact >"$consumer_out" 2>"$consumer_err" || rc=$?
    err="$(cat "$consumer_err")"
  }

  expect_detached() {
    local mode="$1"
    reset_counts
    if [[ -n "$mode" ]]; then
      printf 'mode=%s\n' "$mode" > "$GRANT_FILE"
    else
      rm -f "$GRANT_FILE"
    fi
    run_consumers
    [[ "$rc" == 0 && "$attached_calls" == 0 && "$detached_calls" == 1 && \
       "$compute_calls" == 1 && -z "$err" ]] || {
      printf 'host-fence-unit: mode %s called the attached probe or hid stderr (rc=%s attached=%s detached=%s compute=%s err=%s)\n' \
        "${mode:-<empty>}" "$rc" "$attached_calls" "$detached_calls" "$compute_calls" "$err" >&2
      exit 1
    }
  }

  expect_detached SLURM
  expect_detached COMMITTING_SLURM
  expect_detached COMMITTING_K8S
  expect_detached K8S
  expect_detached ''

  reset_counts
  detached_fail=1
  printf 'mode=SLURM\n' > "$GRANT_FILE"
  run_consumers
  [[ "$rc" == 42 && "$attached_calls" == 0 && "$detached_calls" == 1 && \
     "$err" == 'pireus-device-barrier: device barrier is unexpectedly attached' ]] || {
    printf 'host-fence-unit: detached failure was swallowed (rc=%s attached=%s err=%s)\n' \
      "$rc" "$attached_calls" "$err" >&2
    exit 1
  }

  reset_counts
  compute_rc=1
  printf 'mode=SLURM\n' > "$GRANT_FILE"
  run_consumers
  [[ "$rc" == 1 && "$attached_calls" == 0 && "$detached_calls" == 1 && \
     "$compute_calls" == 1 && -z "$err" ]] || {
    printf 'host-fence-unit: SLURM consumer failure did not propagate (rc=%s attached=%s compute=%s err=%s)\n' \
      "$rc" "$attached_calls" "$compute_calls" "$err" >&2
    exit 1
  }

  reset_counts
  printf 'mode=FENCED\n' > "$GRANT_FILE"
  run_consumers
  [[ "$rc" == 0 && "$attached_calls" == 1 && "$detached_calls" == 0 && \
     "$compute_calls" == 0 && -z "$err" ]] || {
    printf 'host-fence-unit: FENCED did not take the attached path (rc=%s attached=%s detached=%s compute=%s err=%s)\n' \
      "$rc" "$attached_calls" "$detached_calls" "$compute_calls" "$err" >&2
    exit 1
  }

  reset_counts
  attached_fail=1
  printf 'mode=FENCED\n' > "$GRANT_FILE"
  run_consumers
  [[ "$rc" == 42 && "$attached_calls" == 1 && "$detached_calls" == 0 && \
     "$err" == 'pireus-device-barrier: expected device barrier is not attached' ]] || {
    printf 'host-fence-unit: FENCED attached failure was swallowed (rc=%s detached=%s err=%s)\n' \
      "$rc" "$detached_calls" "$err" >&2
    exit 1
  }

  reset_counts
  printf 'mode=NOT_A_MODE\n' > "$GRANT_FILE"
  run_consumers
  [[ "$rc" == 1 && "$attached_calls" == 0 && "$detached_calls" == 0 && -z "$err" ]] || {
    printf 'host-fence-unit: unknown grant mode probed a barrier (rc=%s attached=%s detached=%s err=%s)\n' \
      "$rc" "$attached_calls" "$detached_calls" "$err" >&2
    exit 1
  }
  rm -f "$consumer_err" "$consumer_out"
)
rm -f "$GRANT_FILE"

systemd_root="$WORK_DIR/systemd-root"
install -D -m 0755 "$HOST_FENCE_SCRIPT" \
  "$systemd_root/usr/local/lib/pireus/spark-pair-host-fence"
install -d "$systemd_root/etc/systemd/system" "$systemd_root/usr/lib/systemd"
watchdog_unit_text > "$systemd_root/etc/systemd/system/pireus-spark-host-fence.service"
watchdog_boot_unit_text > "$systemd_root/etc/systemd/system/pireus-spark-host-fence-boot.service"
cp -a /usr/lib/systemd/system "$systemd_root/usr/lib/systemd/"
systemd-analyze --root="$systemd_root" verify \
  pireus-spark-host-fence-boot.service pireus-spark-host-fence.service

source_sha="$(printf 'a%.0s' {1..64})"
freeze_sha="$(printf 'b%.0s' {1..64})"
transaction="$(printf 'c%.0s' {1..64})"
lease_uid=22222222-2222-4222-8222-222222222222
base_rv=101
intent_rv=102
decision_sha="$(printf 'd%.0s' {1..64})"
prepare1="$(printf 'e%.0s' {1..64})"

printf 'source_sha256=%s\nfreeze_sha256=%s\n' "$source_sha" "$freeze_sha" > \
  "$BINDING_FILE"
prepare_grant SLURM 7 holder "$source_sha" "$freeze_sha" "$transaction" \
  "$lease_uid" "$base_rv" "$decision_sha" >/dev/null
prepare0="$(sha256sum "$PREPARE_FILE" | cut -d ' ' -f 1)"
pair_digest="$(printf 'transaction_id=%s\nlease_uid=%s\nbase_lease_resource_version=%s\nnode0_prepare=%s\nnode1_prepare=%s\n' \
  "$transaction" "$lease_uid" "$base_rv" "$prepare0" "$prepare1" | \
  sha256sum | cut -d ' ' -f 1)"

commit_grant SLURM 7 holder "$source_sha" "$freeze_sha" "$transaction" \
  "$lease_uid" "$base_rv" "$decision_sha" "$prepare0" "$prepare1" \
  "$pair_digest" "$intent_rv"
grant_valid
[[ "$(grant_field lease_resource_version)" == "$intent_rv" ]] || \
  { printf 'host-fence-unit: grant was not bound to intent RV\n' >&2; exit 1; }

rm -f "$GRANT_FILE"
prepare_grant SLURM 7 holder "$source_sha" "$freeze_sha" "$transaction" \
  "$lease_uid" "$base_rv" "$decision_sha" >/dev/null
if commit_grant SLURM 7 holder "$source_sha" "$freeze_sha" "$transaction" \
  "$lease_uid" "$base_rv" "$decision_sha" "$prepare0" "$prepare1" \
  "$(printf '0%.0s' {1..64})" "$intent_rv"; then
  printf 'host-fence-unit: arbitrary pair digest was accepted\n' >&2
  exit 1
fi
[[ ! -e "$GRANT_FILE" ]] || {
  printf 'host-fence-unit: denied digest created an authorizing grant\n' >&2
  exit 1
}

if commit_grant SLURM 7 holder "$source_sha" "$freeze_sha" "$transaction" \
  "$lease_uid" "$base_rv" "$decision_sha" "$prepare1" "$prepare0" \
  "$pair_digest" "$intent_rv"; then
  printf 'host-fence-unit: swapped prepare receipts were accepted\n' >&2
  exit 1
fi

sandbox_json="$WORK_DIR/sandboxes.json"
managed_uid=6e7520b1-f6e3-45bd-8673-1d64d102fedf
managed_slice="$HOST_ROOT/sys/fs/cgroup/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod${managed_uid//-/_}.slice"
mkdir -p "$managed_slice"
printf '%s\n' '{"items":[{"metadata":{"namespace":"slurm-pilot","name":"slurm-pilot-worker-spark-f57ls","uid":"6e7520b1-f6e3-45bd-8673-1d64d102fedf"},"labels":{"app.kubernetes.io/name":"slurmd","app.kubernetes.io/instance":"slurm-pilot-worker-spark","io.kubernetes.pod.uid":"6e7520b1-f6e3-45bd-8673-1d64d102fedf"},"state":"SANDBOX_READY"}]}' > "$sandbox_json"
crictl_host() { cat "$sandbox_json"; }
host_ns() {
  [[ "$1" == /usr/bin/jq ]] || return 1
  shift
  /usr/bin/jq "$@"
}
[[ "$(managed_pair_cgroup_dirs)" == "$managed_slice" ]] || {
  printf 'host-fence-unit: canonical READY sandbox did not map to one cgroup\n' >&2
  exit 1
}
rm -rf "$managed_slice"
if managed_pair_cgroup_dirs >/dev/null 2>&1; then
  printf 'host-fence-unit: READY sandbox without a cgroup was accepted\n' >&2
  exit 1
fi
printf '%s\n' '{"items":[{"metadata":{"namespace":"slurm-pilot","name":"slurm-pilot-worker-spark-f57ls","uid":"6e7520b1-f6e3-45bd-8673-1d64d102fedf"},"labels":{"app.kubernetes.io/name":"slurmd","app.kubernetes.io/instance":"slurm-pilot-worker-spark","io.kubernetes.pod.uid":"6e7520b1-f6e3-45bd-8673-1d64d102fedf"},"state":"SANDBOX_NOTREADY"}]}' > "$sandbox_json"
mkdir -p "$managed_slice"
: > "$managed_slice/cgroup.kill"
kill_managed_pair_cgroups
[[ "$(cat "$managed_slice/cgroup.kill")" == 1 ]] || {
  printf 'host-fence-unit: NOTREADY sandbox cgroup was not atomically killed\n' >&2
  exit 1
}
rm -rf "$managed_slice"
[[ -z "$(managed_pair_cgroup_dirs)" ]] || {
  printf 'host-fence-unit: NOTREADY sandbox without a cgroup emitted a kill target\n' >&2
  exit 1
}
printf '%s\n' '{"items":[{"metadata":{"namespace":"slurm-pilot","name":"slurm-pilot-worker-spark-f57ls","uid":"malformed"},"labels":{"app.kubernetes.io/name":"slurmd","app.kubernetes.io/instance":"slurm-pilot-worker-spark","io.kubernetes.pod.uid":"malformed"},"state":"SANDBOX_READY"}]}' > "$sandbox_json"
if managed_pair_sandboxes >/dev/null 2>&1; then
  printf 'host-fence-unit: malformed READY sandbox identity was accepted\n' >&2
  exit 1
fi
printf '%s\n' '{"items":[]}' > "$sandbox_json"
[[ -z "$(managed_pair_sandboxes)" ]] || {
  printf 'host-fence-unit: empty managed sandbox set was not accepted\n' >&2
  exit 1
}

# A PID may exit after enumeration. Accept that race, but keep a live unknown
# process fail-closed.
mkdir -p "$HOST_ROOT/proc/7001" "$HOST_ROOT/proc/7002"
pid_is_admitted_gpu_consumer() {
  local pid="$1"
  [[ "$pid" != 7001 ]] || rm -rf "$HOST_ROOT/proc/$pid"
  return 1
}
host_ns() {
  if [[ "$1" == /usr/bin/nvidia-smi ]]; then
    printf '7001\n'
    return 0
  fi
  return 1
}
external_compute_processes_absent || {
  printf 'host-fence-unit: vanished compute PID was rejected\n' >&2
  exit 1
}
host_ns() {
  if [[ "$1" == /usr/bin/nvidia-smi ]]; then
    printf '7002\n'
    return 0
  fi
  return 1
}
if external_compute_processes_absent; then
  printf 'host-fence-unit: live unknown compute PID was accepted\n' >&2
  exit 1
fi
rm -rf "$HOST_ROOT/proc/7002"

protected_snapshot() {
  printf 'boot_id=%s\n' "$(cat "$HOST_ROOT/proc/sys/kernel/random/boot_id")"
}
capture_protected_baseline
printf '33333333-3333-4333-8333-333333333333\n' > \
  "$HOST_ROOT/proc/sys/kernel/random/boot_id"
capture_protected_baseline
grep -Fq 'boot_id=33333333-3333-4333-8333-333333333333' "$PROTECTED_BASELINE"

ping_file="$WORK_DIR/watchdog-ping"
grant_valid() { return 0; }
stop_gpu_docker_containers() { return 0; }
disable_known_gpu_services() { return 0; }
legacy_gpu_inventory_exact() { return 0; }
known_gpu_services_quiesced() { return 0; }
managed_gpu_restarts_blocked() { return 0; }
active_docker_gpu_claims_zero() { return 0; }
gpu_consumer_set_exact() { return 0; }
managed_gpu_cgroups_empty() { return 0; }
managed_pair_cgroups_empty() { return 0; }
live_memory_floor_met() { return 0; }
protected_resources_unchanged() { return 1; }
watchdog_ping() { : > "$ping_file"; }
if active_enforcement_cycle; then
  printf 'host-fence-unit: failed final predicate was accepted\n' >&2
  exit 1
fi
[[ ! -e "$ping_file" ]] || {
  printf 'host-fence-unit: failed cycle advanced the watchdog heartbeat\n' >&2
  exit 1
}

printf 'HOST_FENCE_UNIT_PASS pair_digest=DENY swapped_receipts=DENY intent_rv=PASS committing_barrier=PASS cgroup_mapping=DENY pid_exit_race=PASS live_unknown_pid=DENY notready_kill=PASS systemd_graph=PASS reboot_baseline=PASS failed_cycle_heartbeat=DENY\n'
