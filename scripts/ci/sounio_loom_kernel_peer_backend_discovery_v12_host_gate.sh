#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT=''
BINARY_SHA256=''
PROFILE_SHA256=''
TEST_UID=61234
PROFILE_NAME=sounio-loom-kernel-peer-backend-discovery-v12

fail() {
  printf 'sounio-loom-kernel-peer-backend-discovery-v12-host-gate: FAIL: %s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --root PATH --binary-sha256 HEX --profile-sha256 HEX\n' "$0" >&2
  exit 64
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "record omitted field: $key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) [[ $# -ge 2 ]] || usage; ROOT="$2"; shift 2 ;;
    --binary-sha256) [[ $# -ge 2 ]] || usage; BINARY_SHA256="$2"; shift 2 ;;
    --profile-sha256) [[ $# -ge 2 ]] || usage; PROFILE_SHA256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$ROOT" == /var/tmp/loom-kernel-peer-backend-discovery-v12-* ]] || fail 'root path is unsafe'
[[ "$BINARY_SHA256" =~ ^[0-9a-f]{64}$ && "$PROFILE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'input hash is malformed'
[[ "$(id -u)" == 0 ]] || fail 'host gate requires the root guardian'

BINARY="$ROOT/loom-peer-backend-discovery"
PROFILE="$ROOT/loom-kernel-peer-backend-discovery-v12.profile"
RUN="$ROOT/run"
READY="$RUN/target.ready"
CONTROL="$RUN/target.release"
REPORT="$RUN/target.report"
TARGET_STDOUT="$RUN/target.stdout"
TARGET_STDERR="$RUN/target.stderr"
CGROUP="/sys/fs/cgroup/loom-kernel-peer-backend-discovery-v12-${ROOT##*-}"
profile_loaded=false
target_pid=''

cleanup() {
  if [[ "$profile_loaded" == true ]]; then
    apparmor_parser -R "$PROFILE" >/dev/null 2>&1 || true
    profile_loaded=false
  fi
  if [[ -n "$target_pid" ]] && kill -0 "$target_pid" 2>/dev/null; then
    kill -KILL "$target_pid" >/dev/null 2>&1 || true
    wait "$target_pid" >/dev/null 2>&1 || true
  fi
  rmdir "$CGROUP/target" "$CGROUP/attacker" "$CGROUP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for tool in apparmor_parser aa-exec setpriv sha256sum grep; do
  command -v "$tool" >/dev/null 2>&1 || fail "required host tool is absent: $tool"
done
[[ -x "$BINARY" && -f "$BINARY" && ! -L "$BINARY" ]] || fail 'binary is absent or linked'
[[ -f "$PROFILE" && ! -L "$PROFILE" ]] || fail 'profile is absent or linked'
[[ "$(sha256sum "$BINARY" | cut -d ' ' -f 1)" == "$BINARY_SHA256" ]] || fail 'binary hash drifted'
[[ "$(sha256sum "$PROFILE" | cut -d ' ' -f 1)" == "$PROFILE_SHA256" ]] || fail 'profile hash drifted'

kernel="$(uname -r)"
active_lsm="$(cat /sys/kernel/security/lsm)"
[[ ",$active_lsm," == *,apparmor,* ]] || fail 'AppArmor is not active'
if [[ ",$active_lsm," == *,bpf,* ]]; then
  bpf_lsm_active=true
else
  bpf_lsm_active=false
fi
config="/boot/config-$kernel"
[[ -r "$config" ]] || fail 'kernel configuration is unavailable'
grep -Fxq 'CONFIG_BPF_LSM=y' "$config" || fail 'kernel was not built with BPF LSM support'
[[ -r /sys/kernel/btf/vmlinux ]] || fail 'kernel BTF is unavailable'
btf_sha256="$(sha256sum /sys/kernel/btf/vmlinux | cut -d ' ' -f 1)"
grep -Eq ' [tT] security_task_prlimit$' /proc/kallsyms || fail 'generic task_prlimit LSM hook is absent'
if grep -Eq ' [tT] apparmor_task_prlimit$' /proc/kallsyms; then
  apparmor_task_prlimit_hook=true
else
  apparmor_task_prlimit_hook=false
fi

if getent passwd "$TEST_UID" >/dev/null 2>&1 || getent group "$TEST_UID" >/dev/null 2>&1; then
  fail "test UID/GID is allocated: $TEST_UID"
fi
for status in /proc/[0-9]*/status; do
  [[ -r "$status" ]] || continue
  uid_line="$(grep -m1 '^Uid:' "$status" || true)"
  set -- $uid_line
  if [[ "${2:-}" == "$TEST_UID" || "${3:-}" == "$TEST_UID" ||
        "${4:-}" == "$TEST_UID" || "${5:-}" == "$TEST_UID" ]]; then
    fail "test UID is already live: $TEST_UID"
  fi
done

install -d -m 0700 -o "$TEST_UID" -g "$TEST_UID" "$RUN"
mkdir "$CGROUP"
mkdir "$CGROUP/target" "$CGROUP/attacker"
apparmor_parser -r "$PROFILE"
profile_loaded=true
grep -Fq "$PROFILE_NAME" /sys/kernel/security/apparmor/profiles || fail 'profile did not become active'

(
  printf '%s\n' "$BASHPID" > "$CGROUP/target/cgroup.procs"
  exec setpriv --reuid="$TEST_UID" --regid="$TEST_UID" --clear-groups -- \
    aa-exec -p "$PROFILE_NAME" -- "$BINARY" --target "$READY" "$CONTROL" "$REPORT"
) >"$TARGET_STDOUT" 2>"$TARGET_STDERR" &
target_pid=$!

for _ in $(seq 1 200); do
  [[ -f "$READY" ]] && break
  if ! kill -0 "$target_pid" 2>/dev/null; then
    fail "target exited before readiness: $(cat "$TARGET_STDERR" 2>/dev/null || true)"
  fi
  sleep 0.025
done
[[ -f "$READY" ]] || fail 'target readiness timed out'
ready="$(tr -d '\n' < "$READY")"
[[ "$(field "$ready" pid)" == "$target_pid" ]] || fail 'target PID drifted across exec'
expected_vector="$TEST_UID/$TEST_UID/$TEST_UID/$TEST_UID"
[[ "$(field "$ready" uid_vector)" == "$expected_vector" &&
   "$(field "$ready" gid_vector)" == "$expected_vector" ]] ||
  fail 'target does not occupy one complete kernel UID/GID vector'
[[ "$(field "$ready" rlimit_soft)" == 1024 && "$(field "$ready" rlimit_hard)" == 2048 ]] ||
  fail 'target initial rlimit drifted'

target_label="$(tr -d '\n' < "/proc/$target_pid/attr/current")"
[[ "$target_label" == "$PROFILE_NAME (enforce)" ]] || fail "target label is not enforced: $target_label"
target_cgroup="$(tr -d '\n' < "/proc/$target_pid/cgroup")"
[[ "$target_cgroup" == *"/${CGROUP##*/}/target" ]] || fail 'target cgroup identity drifted'
stat_record="$(cat "/proc/$target_pid/stat")"
read -r -a stat_fields <<< "${stat_record##*) }"
target_start_tick="${stat_fields[19]:-}"
[[ "$target_start_tick" =~ ^[0-9]+$ && "$target_start_tick" != 0 ]] || fail 'target start tick is invalid'

signal_output="$(
  (
    printf '%s\n' "$BASHPID" > "$CGROUP/attacker/cgroup.procs"
    exec setpriv --reuid="$TEST_UID" --regid="$TEST_UID" --clear-groups -- \
      "$BINARY" --attack-signal "$target_pid"
  )
)"
prlimit_output="$(
  (
    printf '%s\n' "$BASHPID" > "$CGROUP/attacker/cgroup.procs"
    exec setpriv --reuid="$TEST_UID" --regid="$TEST_UID" --clear-groups -- \
      "$BINARY" --attack-prlimit "$target_pid"
  )
)"

touch "$CONTROL"
chown "$TEST_UID:$TEST_UID" "$CONTROL"
wait "$target_pid" || fail "target failed after observation: $(cat "$TARGET_STDERR" 2>/dev/null || true)"
target_pid=''
[[ -f "$REPORT" ]] || fail 'target state report is absent'
report="$(tr -d '\n' < "$REPORT")"
[[ "$(field "$signal_output" uid_vector)" == "$expected_vector" &&
   "$(field "$prlimit_output" uid_vector)" == "$expected_vector" ]] ||
  fail 'attacker kernel UID vector differs from target'
[[ "$(field "$signal_output" gid_vector)" == "$expected_vector" &&
   "$(field "$prlimit_output" gid_vector)" == "$expected_vector" ]] ||
  fail 'attacker kernel GID vector differs from target'

signal_rc="$(field "$signal_output" syscall_rc)"
signal_errno="$(field "$signal_output" syscall_errno)"
signal_seen="$(field "$report" signal_term_seen)"
if [[ "$signal_rc" == -1 && "$signal_errno" != 0 && "$signal_seen" == 0 ]]; then
  signal_observation=REFUSED_BEFORE_EFFECT
elif [[ "$signal_rc" == 0 && "$signal_seen" == 1 ]]; then
  signal_observation=EFFECT_COMPLETED
else
  fail 'signal observation is ambiguous'
fi

prlimit_rc="$(field "$prlimit_output" syscall_rc)"
prlimit_errno="$(field "$prlimit_output" syscall_errno)"
prior_soft="$(field "$prlimit_output" prior_soft)"
observed_soft="$(field "$prlimit_output" observed_soft)"
target_soft="$(field "$report" rlimit_soft)"
if [[ "$prlimit_rc" == 0 && "$prlimit_errno" == 0 && "$prior_soft" == 1024 &&
      "$observed_soft" == 768 && "$target_soft" == 768 ]]; then
  prlimit_observation=EFFECT_COMPLETED
elif [[ "$prlimit_rc" == -1 && "$prlimit_errno" != 0 && "$target_soft" == 1024 ]]; then
  prlimit_observation=REFUSED_BEFORE_EFFECT
else
  fail 'prlimit64 observation is ambiguous'
fi

apparmor_parser -R "$PROFILE"
profile_loaded=false
if grep -Fq "$PROFILE_NAME" /sys/kernel/security/apparmor/profiles; then
  fail 'AppArmor discovery profile did not become extinct'
fi
rmdir "$CGROUP/target" "$CGROUP/attacker" "$CGROUP"
[[ ! -e "$CGROUP" ]] || fail 'discovery cgroups did not become extinct'

printf '%s\n' "$ready" "$signal_output" "$prlimit_output" "$report"
printf 'sounio-loom-kernel-peer-backend-discovery-v12-host-gate: HOST_MEASUREMENT_PASS host=%s kernel=%s hardware=x86_64 semantic_authority=Sounio action=9025 backend=AppArmor profile=%s active_lsm=%s bpf_lsm_config=true bpf_lsm_active=%s btf_sha256=%s security_task_prlimit_hook=true apparmor_task_prlimit_hook=%s same_kuid=true all_four_uid_slots_equal=true attacker_syscalls_open=true target_label_enforced=true target_start_tick=%s target_cgroup_distinct=true attacker_cgroup_distinct=true signal_observation=%s signal_errno=%s prlimit_observation=%s prlimit_errno=%s prlimit_prior_soft=%s prlimit_target_soft=%s probed_operations=2 frozen_operations=10 policy_extinct=true cgroups_extinct=true backend_discovery=true backend_candidate_complete=false stop_rule=no-admissible-receiver-mediator material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(hostname)" "$kernel" "$PROFILE_NAME" "$active_lsm" "$bpf_lsm_active" \
  "$btf_sha256" "$apparmor_task_prlimit_hook" "$target_start_tick" \
  "$signal_observation" "$signal_errno" "$prlimit_observation" "$prlimit_errno" \
  "$prior_soft" "$target_soft"
