#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-host-principal-cell-host-gate: FAIL reason=%s kernel_distinct_principal_candidate=false material_grant=false grant_extinction=false launch_open=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-host-principal-cell-host-gate: HOST_GATE_UNAVAILABLE reason=%s kernel_distinct_principal_candidate=false material_grant=false grant_extinction=false launch_open=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --binary ABSOLUTE_PATH --expected-sha256 HEX\n' "$0" >&2
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
  fail "measurement omitted field: $key"
}

BINARY=''
EXPECTED_SHA256=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      [[ $# -ge 2 ]] || usage
      BINARY="$2"
      shift 2
      ;;
    --expected-sha256)
      [[ $# -ge 2 ]] || usage
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$BINARY" == /* && "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] || unavailable 'PID 1 is not systemd'
[[ -d /run/systemd/system ]] || unavailable 'systemd runtime directory is absent'
for tool in systemctl systemd-run sha256sum stat hostname uname timeout; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ -f "$BINARY" && ! -L "$BINARY" && -x "$BINARY" ]] || fail 'native PrincipalCell binary is absent, linked, or non-executable'
[[ "$(stat -c '%u:%g:%a:%h' "$BINARY")" == '0:0:555:1' ]] || fail 'native PrincipalCell binary metadata is unsafe'
ACTUAL_SHA256="$(sha256sum "$BINARY" | cut -d ' ' -f 1)"
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || fail 'native PrincipalCell binary hash drifted on host'

nonce="$$-$(date +%s%N)"
UNIT_A="sounio-loom-principal-cell-a-$nonce.service"
UNIT_B="sounio-loom-principal-cell-b-$nonce.service"
PID_A=''
PID_B=''

cleanup() {
  local unit
  for unit in "$UNIT_A" "$UNIT_B"; do
    systemctl stop "$unit" >/dev/null 2>&1 || true
    systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

launch_cell() {
  local unit="$1"
  systemd-run --quiet --unit="$unit" --service-type=exec --collect \
    --property=DynamicUser=yes \
    --property=UMask=0077 \
    --property=NoNewPrivileges=yes \
    --property=PrivateTmp=yes \
    --property=PrivateDevices=yes \
    --property=PrivateNetwork=yes \
    --property=ProtectSystem=strict \
    --property=ProtectHome=yes \
    --property=ProtectKernelTunables=yes \
    --property=ProtectKernelModules=yes \
    --property=ProtectKernelLogs=yes \
    --property=ProtectControlGroups=yes \
    --property=ProtectClock=yes \
    --property=ProtectProc=invisible \
    --property=ProcSubset=pid \
    --property=RestrictNamespaces=yes \
    --property=RestrictSUIDSGID=yes \
    --property=LockPersonality=yes \
    --property=MemoryDenyWriteExecute=yes \
    --property=RestrictRealtime=yes \
    --property=SystemCallArchitectures=native \
    --property=RestrictAddressFamilies=AF_UNIX \
    --property="ReadOnlyPaths=$BINARY" \
    "$BINARY" --cell-hold --seconds 90 >/dev/null
}

launch_cell "$UNIT_A"
launch_cell "$UNIT_B"

wait_for_cell() {
  local unit="$1" pid='' attempt
  for attempt in $(seq 1 100); do
    pid="$(systemctl show "$unit" --property MainPID --value 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 && -r "/proc/$pid/status" ]] &&
       systemctl is-active --quiet "$unit"; then
      printf '%s' "$pid"
      return 0
    fi
    sleep 0.05
  done
  fail "transient PrincipalCell did not become live: $unit"
}

PID_A="$(wait_for_cell "$UNIT_A")"
PID_B="$(wait_for_cell "$UNIT_B")"
[[ "$PID_A" != "$PID_B" ]] || fail 'transient PrincipalCells share one PID'

for unit in "$UNIT_A" "$UNIT_B"; do
  [[ "$(systemctl show "$unit" --property DynamicUser --value)" == yes ]] || fail "$unit lost DynamicUser"
  [[ "$(systemctl show "$unit" --property NoNewPrivileges --value)" == yes ]] || fail "$unit lost NoNewPrivileges"
  [[ "$(systemctl show "$unit" --property ProtectSystem --value)" == strict ]] || fail "$unit lost ProtectSystem=strict"
  [[ "$(systemctl show "$unit" --property ProtectProc --value)" == invisible ]] || fail "$unit lost ProtectProc=invisible"
  [[ "$(systemctl show "$unit" --property ProcSubset --value)" == pid ]] || fail "$unit lost ProcSubset=pid"
  [[ "$(systemctl show "$unit" --property PrivateNetwork --value)" == yes ]] || fail "$unit lost PrivateNetwork"
  [[ "$(systemctl show "$unit" --property MemoryDenyWriteExecute --value)" == yes ]] || fail "$unit lost MemoryDenyWriteExecute"
done

set +e
measurement="$(timeout --signal=TERM --kill-after=2s 20s \
  "$BINARY" --measure --pid-a "$PID_A" --pid-b "$PID_B" \
  --unit-a "$UNIT_A" --unit-b "$UNIT_B" 2>&1)"
measurement_status=$?
set -e
[[ $measurement_status -eq 0 ]] || fail "native hostile measurement failed or timed out status=$measurement_status output=$measurement"
[[ "$measurement" == 'LOOM_HOST_PRINCIPAL_CELL_MEASUREMENT PASS '* ]] || fail 'native hostile measurement did not pass'

for expectation in \
  'semantic_authority=Sounio' 'action=9030' 'language_role=MATERIAL_PARITY' \
  'uid_distinct=true' 'gid_distinct=true' 'cgroup_distinct=true' \
  'pidfd_live=true' 'start_tick_stable=true' 'signal_cross_uid=EPERM' \
  'ptrace_cross_uid=EPERM' 'process_vm_readv_cross_uid=EPERM' \
  'copied_pidfd_signal=EPERM' 'copied_pidfd_getfd=EPERM' \
  'reciprocal_attacks=refused' 'kernel_distinct_principal_candidate=true' \
  'same_uid_peer_isolation=false' 'material_grant=false' \
  'grant_extinction=false' 'exec_attached=false' 'launch_open=false'; do
  [[ " $measurement " == *" $expectation "* ]] || fail "hostile measurement omitted $expectation"
done

UID_A="$(field "$measurement" uid_a)"
GID_A="$(field "$measurement" gid_a)"
UID_B="$(field "$measurement" uid_b)"
GID_B="$(field "$measurement" gid_b)"
read -r _ SYSTEMD_VERSION _ < <(systemctl --version | sed -n '1p')
[[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || fail 'systemd version is not canonical'

cleanup
trap - EXIT
for pid in "$PID_A" "$PID_B"; do
  for attempt in $(seq 1 100); do
    [[ ! -e "/proc/$pid" ]] && break
    sleep 0.05
  done
  [[ ! -e "/proc/$pid" ]] || fail "PrincipalCell process survived service cleanup: $pid"
done
for unit in "$UNIT_A" "$UNIT_B"; do
  systemctl is-active --quiet "$unit" && fail "PrincipalCell unit survived cleanup: $unit"
done

printf 'sounio-loom-host-principal-cell-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio action=9030 material_producer=C++20 material_role=MATERIAL_PARITY transitory=true host=%s kernel=%s architecture=%s systemd_version=%s binary_sha256=%s pid_a=%s uid_a=%s gid_a=%s pid_b=%s uid_b=%s gid_b=%s simultaneous_uid_distinct=true simultaneous_gid_distinct=true cgroup_distinct=true pidfd_live=true start_tick_stable=true signal_cross_uid=EPERM proc_mem_cross_uid=%s ptrace_cross_uid=EPERM process_vm_readv_cross_uid=EPERM proc_fd_cross_uid=%s copied_pidfd_signal=EPERM copied_pidfd_getfd=EPERM reciprocal_attacks=refused dynamic_user=true no_new_privileges=true protect_system=strict protect_proc=invisible private_network=true capabilities=zero process_cleanup=observed kernel_distinct_principal_candidate=true same_uid_peer_isolation=false material_grant=false grant_extinction=false exec_attached=false commit_attached=false ci_attached=false launch_open=false measurement_sha256=%s\n' \
  "$(hostname)" "$(uname -r)" "$(uname -m)" "$SYSTEMD_VERSION" \
  "$ACTUAL_SHA256" "$PID_A" "$UID_A" "$GID_A" "$PID_B" "$UID_B" "$GID_B" \
  "$(field "$measurement" proc_mem_cross_uid)" "$(field "$measurement" proc_fd_cross_uid)" \
  "$(printf '%s\n' "$measurement" | sha256sum | cut -d ' ' -f 1)"
