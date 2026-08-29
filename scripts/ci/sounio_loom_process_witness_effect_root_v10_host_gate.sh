#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-process-witness-effect-root-v10-host-gate: FAIL reason=%s root_treatment=false bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-process-witness-effect-root-v10-host-gate: HOST_GATE_UNAVAILABLE reason=%s root_treatment=false bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --root ABSOLUTE_PATH --cell-sha256 HEX --tree-sha256 HEX --mutator ABSOLUTE_PATH --mutator-sha256 HEX\n' "$0" >&2
  exit 64
}

ROOT=''
CELL_SHA256=''
TREE_SHA256=''
MUTATOR=''
MUTATOR_SHA256=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || usage
      ROOT="$2"
      shift 2
      ;;
    --cell-sha256)
      [[ $# -ge 2 ]] || usage
      CELL_SHA256="$2"
      shift 2
      ;;
    --tree-sha256)
      [[ $# -ge 2 ]] || usage
      TREE_SHA256="$2"
      shift 2
      ;;
    --mutator)
      [[ $# -ge 2 ]] || usage
      MUTATOR="$2"
      shift 2
      ;;
    --mutator-sha256)
      [[ $# -ge 2 ]] || usage
      MUTATOR_SHA256="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$ROOT" == /* && "$ROOT" != / && "$ROOT" =~ ^[A-Za-z0-9._/-]+$ &&
   "$MUTATOR" == /* && "$MUTATOR" != / && "$MUTATOR" =~ ^[A-Za-z0-9._/-]+$ &&
   "$CELL_SHA256" =~ ^[0-9a-f]{64}$ && "$TREE_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$MUTATOR_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  usage
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] ||
  unavailable 'PID 1 is not systemd'
for tool in systemctl systemd-run sha256sum stat readelf find readlink mkfifo \
            mount umount journalctl; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ "$(stat -c '%F:%u:%g:%a:%h' "$MUTATOR" 2>/dev/null)" == \
   'regular file:0:0:500:1' ]] || unavailable 'namespace mutator metadata drifted'
[[ "$(sha256sum "$MUTATOR" | cut -d ' ' -f 1)" == "$MUTATOR_SHA256" ]] ||
  fail 'namespace mutator hash drifted'
if readelf -l "$MUTATOR" | grep -q 'INTERP'; then
  fail 'namespace mutator retained a dynamic interpreter'
fi

expected_paths="$ROOT/dev
$ROOT/dev/null
$ROOT/loom
$ROOT/loom/effect-cell
$ROOT/loom/effect-policy-v10.freeze.v1
$ROOT/loom/payload
$ROOT/loom/payload.freeze.v1
$ROOT/proc
$ROOT/run
$ROOT/run/systemd
$ROOT/run/systemd/incoming
$ROOT/sys
$ROOT/tmp
$ROOT/var
$ROOT/var/tmp"
actual_paths="$(find "$ROOT" -mindepth 1 -printf '%p\n' | sort)"
[[ "$actual_paths" == "$expected_paths" ]] || fail 'host root path set drifted'
for directory in "$ROOT" "$ROOT/loom" "$ROOT/dev" "$ROOT/proc" "$ROOT/tmp" \
                 "$ROOT/run" "$ROOT/run/systemd" "$ROOT/run/systemd/incoming" \
                 "$ROOT/sys" "$ROOT/var" "$ROOT/var/tmp"; do
  [[ "$(stat -c '%F:%u:%g:%a' "$directory")" == 'directory:0:0:555' ]] ||
    fail "host root directory metadata drifted: $directory"
done
for binary in "$ROOT/loom/effect-cell" "$ROOT/loom/payload"; do
  [[ "$(stat -c '%F:%u:%g:%a:%h' "$binary")" == 'regular file:0:0:555:1' ]] ||
    fail "host root binary metadata drifted: $binary"
  if readelf -l "$binary" | grep -q 'INTERP'; then
    fail "host root binary retained a dynamic interpreter: $binary"
  fi
done
for manifest in "$ROOT/loom/payload.freeze.v1" \
                "$ROOT/loom/effect-policy-v10.freeze.v1"; do
  [[ "$(stat -c '%F:%u:%g:%a:%h' "$manifest")" == 'regular file:0:0:444:1' ]] ||
    fail "host root manifest metadata drifted: $manifest"
done
[[ "$(stat -c '%F:%t:%T' "$ROOT/dev/null")" == 'character special file:1:3' ]] ||
  fail 'host root /dev/null is not character device 1:3'
[[ "$(sha256sum "$ROOT/loom/effect-cell" | cut -d ' ' -f 1)" == "$CELL_SHA256" ]] ||
  fail 'host root cell hash drifted'
[[ "$(sha256sum "$ROOT/loom/payload" | cut -d ' ' -f 1)" == \
  7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d ]] ||
  fail 'host root Sounio payload hash drifted'
[[ "$(sha256sum "$ROOT/loom/payload.freeze.v1" | cut -d ' ' -f 1)" == \
  624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da ]] ||
  fail 'host root payload manifest hash drifted'
[[ "$(sha256sum "$ROOT/loom/effect-policy-v10.freeze.v1" | cut -d ' ' -f 1)" == \
  9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda ]] ||
  fail 'host root policy manifest hash drifted'

nonce="$$-$(date +%s%N)"
UNIT="sounio-loom-effect-root-v10-$nonce.service"
SABOTAGE_UNIT="sounio-loom-effect-root-v10-missing-incoming-$nonce.service"
SYS_SABOTAGE_UNIT="sounio-loom-effect-root-v10-missing-sys-$nonce.service"
VAR_TMP_SABOTAGE_UNIT="sounio-loom-effect-root-v10-missing-var-tmp-$nonce.service"
LIVE_PROC_UNIT="sounio-loom-effect-root-v10-live-procfs-$nonce.service"
WRONG_PROC_UNIT="sounio-loom-effect-root-v10-wrong-proc-source-$nonce.service"
WRITABLE_PROC_UNIT="sounio-loom-effect-root-v10-writable-proc-$nonce.service"
NONEMPTY_PROC_UNIT="sounio-loom-effect-root-v10-nonempty-proc-$nonce.service"
CONTROL="/run/sounio-loom-effect-root-v10-$nonce.control"
OUTPUT="/run/sounio-loom-effect-root-v10-$nonce.output"
ERROR="/run/sounio-loom-effect-root-v10-$nonce.error"
SABOTAGE_OUTPUT="/run/sounio-loom-effect-root-v10-missing-incoming-$nonce.output"
SABOTAGE_ERROR="/run/sounio-loom-effect-root-v10-missing-incoming-$nonce.error"
SYS_SABOTAGE_OUTPUT="/run/sounio-loom-effect-root-v10-missing-sys-$nonce.output"
SYS_SABOTAGE_ERROR="/run/sounio-loom-effect-root-v10-missing-sys-$nonce.error"
VAR_TMP_SABOTAGE_OUTPUT="/run/sounio-loom-effect-root-v10-missing-var-tmp-$nonce.output"
VAR_TMP_SABOTAGE_ERROR="/run/sounio-loom-effect-root-v10-missing-var-tmp-$nonce.error"
LIVE_PROC_CONTROL="/run/sounio-loom-effect-root-v10-live-procfs-$nonce.control"
LIVE_PROC_OUTPUT="/run/sounio-loom-effect-root-v10-live-procfs-$nonce.output"
LIVE_PROC_ERROR="/run/sounio-loom-effect-root-v10-live-procfs-$nonce.error"
WRONG_PROC_CONTROL="/run/sounio-loom-effect-root-v10-wrong-proc-source-$nonce.control"
WRONG_PROC_OUTPUT="/run/sounio-loom-effect-root-v10-wrong-proc-source-$nonce.output"
WRONG_PROC_ERROR="/run/sounio-loom-effect-root-v10-wrong-proc-source-$nonce.error"
WRITABLE_PROC_CONTROL="/run/sounio-loom-effect-root-v10-writable-proc-$nonce.control"
WRITABLE_PROC_OUTPUT="/run/sounio-loom-effect-root-v10-writable-proc-$nonce.output"
WRITABLE_PROC_ERROR="/run/sounio-loom-effect-root-v10-writable-proc-$nonce.error"
NONEMPTY_PROC_OUTPUT="/run/sounio-loom-effect-root-v10-nonempty-proc-$nonce.output"
NONEMPTY_PROC_ERROR="/run/sounio-loom-effect-root-v10-nonempty-proc-$nonce.error"
CLIENT_PID=''
CELL_PID=''
TYPED_CLIENT_PID=''
TYPED_CELL_PID=''
TYPED_FD_OPEN=false
ROOT_MOUNTED=false
TYPED_UNITS=("$LIVE_PROC_UNIT" "$WRONG_PROC_UNIT" "$WRITABLE_PROC_UNIT" "$NONEMPTY_PROC_UNIT")
TYPED_PATHS=(
  "$LIVE_PROC_CONTROL" "$LIVE_PROC_OUTPUT" "$LIVE_PROC_ERROR"
  "$WRONG_PROC_CONTROL" "$WRONG_PROC_OUTPUT" "$WRONG_PROC_ERROR"
  "$WRITABLE_PROC_CONTROL" "$WRITABLE_PROC_OUTPUT" "$WRITABLE_PROC_ERROR"
  "$NONEMPTY_PROC_OUTPUT" "$NONEMPTY_PROC_ERROR"
)

cleanup() {
  { exec 9>&-; } 2>/dev/null || true
  { exec 8>&-; } 2>/dev/null || true
  TYPED_FD_OPEN=false
  if [[ -n "$UNIT" ]]; then
    systemctl stop "$UNIT" >/dev/null 2>&1 || true
    systemctl reset-failed "$UNIT" >/dev/null 2>&1 || true
  fi
  if [[ -n "$SABOTAGE_UNIT" ]]; then
    systemctl stop "$SABOTAGE_UNIT" >/dev/null 2>&1 || true
    systemctl reset-failed "$SABOTAGE_UNIT" >/dev/null 2>&1 || true
  fi
  if [[ -n "$SYS_SABOTAGE_UNIT" ]]; then
    systemctl stop "$SYS_SABOTAGE_UNIT" >/dev/null 2>&1 || true
    systemctl reset-failed "$SYS_SABOTAGE_UNIT" >/dev/null 2>&1 || true
  fi
  if [[ -n "$VAR_TMP_SABOTAGE_UNIT" ]]; then
    systemctl stop "$VAR_TMP_SABOTAGE_UNIT" >/dev/null 2>&1 || true
    systemctl reset-failed "$VAR_TMP_SABOTAGE_UNIT" >/dev/null 2>&1 || true
  fi
  for typed_unit in "${TYPED_UNITS[@]}"; do
    systemctl stop "$typed_unit" >/dev/null 2>&1 || true
    systemctl reset-failed "$typed_unit" >/dev/null 2>&1 || true
  done
  if [[ "$ROOT_MOUNTED" == true ]]; then
    umount "$ROOT" >/dev/null 2>&1 || true
    ROOT_MOUNTED=false
  fi
  rm -f "$CONTROL" "$OUTPUT" "$ERROR" "$SABOTAGE_OUTPUT" "$SABOTAGE_ERROR" \
    "$SYS_SABOTAGE_OUTPUT" "$SYS_SABOTAGE_ERROR"
  rm -f "$VAR_TMP_SABOTAGE_OUTPUT" "$VAR_TMP_SABOTAGE_ERROR"
  rm -f "${TYPED_PATHS[@]}"
}
trap cleanup EXIT

mount_immutable_root() {
  mount --bind "$ROOT" "$ROOT"
  ROOT_MOUNTED=true
  mount -o remount,bind,ro,nosuid "$ROOT"
}

unmount_immutable_root() {
  umount "$ROOT"
  ROOT_MOUNTED=false
}

start_typed_hold() {
  local unit="$1" control="$2" output="$3" error="$4" bind_paths="$5"
  rm -f "$control" "$output" "$error"
  mkfifo -m 0600 "$control"
  exec 8<>"$control"
  TYPED_FD_OPEN=true
  systemd-run --quiet --unit="$unit" --service-type=exec --pipe --wait \
    --property="RootDirectory=$ROOT" \
    --property=MountAPIVFS=no \
    --property=DynamicUser=yes \
    --property=UMask=0077 \
    --property=NoNewPrivileges=yes \
    --property=PrivateTmp=yes \
    --property="BindReadOnlyPaths=$bind_paths" \
    --property=PrivateDevices=no \
    --property=PrivateNetwork=yes \
    --property=ProtectSystem=strict \
    --property=ProtectHome=read-only \
    --property=RestrictNamespaces=yes \
    --property=RestrictSUIDSGID=yes \
    --property=LockPersonality=yes \
    --property=MemoryDenyWriteExecute=yes \
    --property=RestrictRealtime=yes \
    --property=SystemCallArchitectures=native \
    --property=KillMode=mixed \
    --property=TimeoutStopSec=2s \
    -- /loom/effect-cell --root-hold \
       --policy-manifest /loom/effect-policy-v10.freeze.v1 \
    < "$control" > "$output" 2> "$error" &
  TYPED_CLIENT_PID=$!

  for attempt in $(seq 1 200); do
    if grep -q '^LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_ROOT_READY PASS ' "$output" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$TYPED_CLIENT_PID" 2>/dev/null; then
      wait "$TYPED_CLIENT_PID" || true
      fail "typed /proc control exited before READY: unit=$unit stderr=$(tr '\n' ' ' < "$error" 2>/dev/null)"
    fi
    sleep 0.05
  done
  grep -q '^LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_ROOT_READY PASS ' "$output" ||
    fail "typed /proc control did not emit READY: $unit"
  TYPED_CELL_PID="$(systemctl show "$unit" --property MainPID --value)"
  [[ "$TYPED_CELL_PID" =~ ^[0-9]+$ && "$TYPED_CELL_PID" -gt 1 &&
     -r "/proc/$TYPED_CELL_PID/mountinfo" ]] ||
    fail "typed /proc control MainPID is unavailable: $unit"
}

release_typed_hold() {
  local unit="$1" error="$2"
  [[ "$TYPED_FD_OPEN" == true ]] || fail "typed /proc control descriptor is closed: $unit"
  printf 'X' >&8
  exec 8>&-
  TYPED_FD_OPEN=false
  wait "$TYPED_CLIENT_PID" ||
    fail "typed /proc control failed during release: unit=$unit stderr=$(tr '\n' ' ' < "$error" 2>/dev/null)"
  for attempt in $(seq 1 100); do
    [[ ! -e "/proc/$TYPED_CELL_PID" ]] && break
    sleep 0.05
  done
  [[ ! -e "/proc/$TYPED_CELL_PID" ]] ||
    fail "typed /proc control survived release: $unit"
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
}

mount_immutable_root
mkfifo -m 0600 "$CONTROL"
exec 9<>"$CONTROL"
systemd-run --quiet --unit="$UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp $ROOT/tmp:/var/tmp /sys:/sys" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v10.freeze.v1 \
  < "$CONTROL" > "$OUTPUT" 2> "$ERROR" &
CLIENT_PID=$!

for attempt in $(seq 1 200); do
  if grep -q '^LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_ROOT_READY PASS ' "$OUTPUT" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    wait "$CLIENT_PID" || true
    unit_state="$(systemctl show "$UNIT" --property Result --property ExecMainStatus \
      --property ExecMainCode 2>/dev/null | tr '\n' ' ')"
    unit_journal="$(journalctl --no-pager -n 12 -u "$UNIT" 2>/dev/null | tr '\n' ' ')"
    fail "root-hold exited before READY: stderr=$(tr '\n' ' ' < "$ERROR" 2>/dev/null) state=$unit_state journal=$unit_journal"
  fi
  sleep 0.05
done
ready="$(sed -n '1p' "$OUTPUT")"
[[ "$ready" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_ROOT_READY\ PASS* ]] ||
  fail 'root-hold did not emit READY'
for expectation in \
  'semantic_authority=Sounio' 'action=9025' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' 'root_read_only=true' \
  'root_exact=true' 'dynamic_linker_visible=false' 'host_root_visible=false' \
  'tmp_read_only=true' 'fd_inventory=0+1+2' \
  "cell_sha256=$CELL_SHA256" \
  'payload_sha256=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d' \
  'policy_manifest_sha256=9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda' \
  'typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB' \
  'effective_mount_truth=DynamicUser+disconnected+strict+read-only' \
  'identity_typed_mounts=CAPSULE_EMPTY_BIND' \
  'property_authority=CONFIGURATION_ONLY' \
  'filesystem_authority=ROOT_HOST_MOUNTINFO' \
  'temporary_sources=SAME_IMMUTABLE_ROOT_TMP' \
  'temporary_read_only=true' 'temporary_empty=true' \
  'typed_structural_mounts=/proc:CAPSULE_EMPTY_BIND' \
  'forbidden_mounts=/home+/root+/run+/var+/etc' \
  'systemd_mount=/run/systemd/incoming' 'systemd_sys_mount=/sys' \
  'principal_readable=false' 'principal_enumeration=forbidden' \
  'empty_observer=ROOT_HOST' 'mount_observer=ROOT_HOST' \
  'extinction_observer=ROOT_HOST' \
  'systemd_sys_ready_filesystem=sysfs' 'systemd_sys_ready_read_only=true' \
  'var_tmp_read_only=true' 'var_tmp_source=IMMUTABLE_ROOT_TMP' \
  'material_coverage=false' 'complete_effects=false' \
  'material_execution=false' 'launch_open=false' 'claim_ready=false'; do
  [[ " $ready " == *" $expectation "* ]] || fail "root READY omitted $expectation"
done

CELL_PID="$(systemctl show "$UNIT" --property MainPID --value)"
[[ "$CELL_PID" =~ ^[0-9]+$ && "$CELL_PID" -gt 1 && -r "/proc/$CELL_PID/status" ]] ||
  fail 'root-hold MainPID is unavailable'
[[ "$(systemctl show "$UNIT" --property DynamicUser --value)" == yes ]] ||
  fail 'root-hold lost DynamicUser'
[[ "$(systemctl show "$UNIT" --property RootDirectory --value)" == "$ROOT" ]] ||
  fail 'root-hold RootDirectory drifted'
[[ "$(systemctl show "$UNIT" --property MountAPIVFS --value)" == no ]] ||
  fail 'root-hold MountAPIVFS drifted'
[[ "$(systemctl show "$UNIT" --property PrivateTmp --value)" == yes ]] ||
  fail 'root-hold lost its disconnected private temporary namespace'
[[ "$(systemctl show "$UNIT" --property PrivateDevices --value)" == no ]] ||
  fail 'root-hold replaced the exact /dev/null view'
[[ "$(systemctl show "$UNIT" --property PrivateNetwork --value)" == yes ]] ||
  fail 'root-hold lost its private network namespace'
[[ "$(systemctl show "$UNIT" --property ProtectSystem --value)" == strict ]] ||
  fail 'root-hold lost strict system protection'
[[ "$(systemctl show "$UNIT" --property ProtectHome --value)" == read-only ]] ||
  fail 'root-hold lost read-only home protection'
[[ "$(systemctl show "$UNIT" --property LockPersonality --value)" == yes ]] ||
  fail 'root-hold lost LockPersonality'
[[ "$(systemctl show "$UNIT" --property MemoryDenyWriteExecute --value)" == yes ]] ||
  fail 'root-hold lost MemoryDenyWriteExecute'

status="$(< "/proc/$CELL_PID/status")"
uid_line="$(printf '%s\n' "$status" | grep '^Uid:')"
gid_line="$(printf '%s\n' "$status" | grep '^Gid:')"
read -r _ uid_real uid_effective uid_saved uid_fs <<< "$uid_line"
read -r _ gid_real gid_effective gid_saved gid_fs <<< "$gid_line"
[[ "$uid_real" != 0 && "$uid_real" == "$uid_effective" &&
   "$uid_real" == "$uid_saved" && "$uid_real" == "$uid_fs" ]] ||
  fail 'root-hold UID vector is not one non-root DynamicUser'
[[ "$gid_real" != 0 && "$gid_real" == "$gid_effective" &&
   "$gid_real" == "$gid_saved" && "$gid_real" == "$gid_fs" ]] ||
  fail 'root-hold GID vector is not one non-root DynamicUser'
[[ "$(printf '%s\n' "$status" | grep '^NoNewPrivs:' | tr -d '[:space:]A-Za-z:')" == 1 ]] ||
  fail 'root-hold lost no_new_privs'
[[ "$(printf '%s\n' "$status" | grep '^Seccomp:' | tr -d '[:space:]A-Za-z:')" == 2 ]] ||
  fail 'root-hold did not install seccomp filter mode'
[[ "$(printf '%s\n' "$status" | grep '^CapEff:' | cut -f 2)" == 0000000000000000 ]] ||
  fail 'root-hold retained effective capabilities'
[[ "$(printf '%s\n' "$status" | grep '^CapAmb:' | cut -f 2)" == 0000000000000000 ]] ||
  fail 'root-hold retained ambient capabilities'
process_root="$(readlink "/proc/$CELL_PID/root")"
process_root_identity="$(stat -Lc '%d:%i' "/proc/$CELL_PID/root")"
expected_root_identity="$(stat -Lc '%d:%i' "$ROOT")"
[[ "$process_root_identity" == "$expected_root_identity" ]] ||
  fail "root-hold process root identity drifted: link=$process_root observed=$process_root_identity expected=$expected_root_identity"
process_executable_identity="$(stat -Lc '%d:%i' "/proc/$CELL_PID/exe")"
expected_executable_identity="$(stat -Lc '%d:%i' "$ROOT/loom/effect-cell")"
[[ "$process_executable_identity" == "$expected_executable_identity" ]] ||
  fail "root-hold executable identity drifted: observed=$process_executable_identity expected=$expected_executable_identity"
[[ "$(readlink "/proc/$CELL_PID/ns/mnt")" != "$(readlink /proc/1/ns/mnt)" ]] ||
  fail 'root-hold shares PID 1 mount namespace'
[[ "$(readlink "/proc/$CELL_PID/ns/net")" != "$(readlink /proc/1/ns/net)" ]] ||
  fail 'root-hold shares PID 1 network namespace'
grep -Fq "$UNIT" "/proc/$CELL_PID/cgroup" || fail 'root-hold cgroup identity drifted'
[[ "$(stat -c '%F:%u:%g' "/proc/$CELL_PID/root/run/systemd/incoming")" == \
   'directory:0:0' ]] || fail 'root observer saw invalid incoming metadata'
incoming_entry="$(find "/proc/$CELL_PID/root/run/systemd/incoming" \
  -mindepth 1 -print -quit)"
[[ -z "$incoming_entry" ]] ||
  fail "root observer found a nonempty incoming rendezvous: $incoming_entry"

fd_paths=(/proc/"$CELL_PID"/fd/*)
[[ ${#fd_paths[@]} -eq 3 ]] || fail "root-hold fd inventory is not exact: ${#fd_paths[@]}"
for descriptor in 0 1 2; do
  [[ -e "/proc/$CELL_PID/fd/$descriptor" ]] || fail "root-hold omitted fd $descriptor"
done

root_mount_ro=false
root_mount_device=''
tmp_mount_ro=false
tmp_mount_root=''
incoming_mount=false
incoming_mount_id=''
incoming_mount_root=''
sys_mount=false
sys_mount_root=''
sys_mount_filesystem=''
sys_mount_source=''
sys_mount_ro=false
var_tmp_mount=false
var_tmp_mount_root=''
var_tmp_mount_ro=false
proc_mount_count=0
proc_mount_root=''
proc_mount_device=''
proc_mount_filesystem=''
proc_mount_source=''
proc_mount_vfs_ro=false
forbidden_mounts_observed=()
forbidden_mount_records=()
while IFS= read -r mount_line; do
  read -r -a fields <<< "$mount_line"
  [[ ${#fields[@]} -ge 6 ]] || fail 'root-hold mountinfo is malformed'
  mountpoint="${fields[4]}"
  options=",${fields[5]},"
  if [[ "$mountpoint" == / && "$options" == *,ro,* ]]; then
    root_mount_ro=true
    root_mount_device="${fields[2]}"
  fi
  if [[ "$mountpoint" == /tmp && "$options" == *,ro,* ]]; then
    tmp_mount_ro=true
    tmp_mount_root="${fields[3]}"
  fi
  if [[ "$mountpoint" == /run/systemd/incoming ]]; then
    incoming_mount=true
    incoming_mount_id="${fields[0]}"
    incoming_mount_root="${fields[3]}"
  fi
  if [[ "$mountpoint" == /sys ]]; then
    separator=0
    for index in "${!fields[@]}"; do
      if [[ "${fields[$index]}" == - ]]; then separator="$index"; break; fi
    done
    [[ "$separator" -gt 5 && ${#fields[@]} -gt $((separator + 3)) ]] ||
      fail 'root-hold /sys mountinfo lacks a filesystem boundary'
    sys_mount=true
    sys_mount_root="${fields[3]}"
    sys_mount_filesystem="${fields[$((separator + 1))]}"
    sys_mount_source="${fields[$((separator + 2))]}"
    super_options=",${fields[$((separator + 3))]},"
    if [[ "$options" == *,ro,* || "$super_options" == *,ro,* ]]; then
      sys_mount_ro=true
    fi
  fi
  if [[ "$mountpoint" == /var/tmp ]]; then
    separator=0
    for index in "${!fields[@]}"; do
      if [[ "${fields[$index]}" == - ]]; then separator="$index"; break; fi
    done
    [[ "$separator" -gt 5 && ${#fields[@]} -gt $((separator + 3)) ]] ||
      fail 'root-hold /var/tmp mountinfo lacks a filesystem boundary'
    var_tmp_mount=true
    var_tmp_mount_root="${fields[3]}"
    super_options=",${fields[$((separator + 3))]},"
    if [[ "$options" == *,ro,* || "$super_options" == *,ro,* ]]; then
      var_tmp_mount_ro=true
    fi
  fi
  if [[ "$mountpoint" == /proc ]]; then
    separator=0
    for index in "${!fields[@]}"; do
      if [[ "${fields[$index]}" == - ]]; then separator="$index"; break; fi
    done
    [[ "$separator" -gt 5 && ${#fields[@]} -gt $((separator + 3)) ]] ||
      fail 'root-hold /proc mountinfo lacks a filesystem boundary'
    proc_mount_count=$((proc_mount_count + 1))
    proc_mount_root="${fields[3]}"
    proc_mount_device="${fields[2]}"
    proc_mount_filesystem="${fields[$((separator + 1))]}"
    proc_mount_source="${fields[$((separator + 2))]}"
    if [[ "$options" == *,ro,* ]]; then proc_mount_vfs_ro=true; fi
  fi
  case "$mountpoint" in
    /home|/root|/run|/var|/etc)
      forbidden_mounts_observed+=("$mountpoint")
      forbidden_mount_records+=("$mount_line")
      ;;
  esac
done < "/proc/$CELL_PID/mountinfo"
[[ "$root_mount_ro" == true ]] || fail 'root-hold mountinfo lacks a read-only root'
[[ "$tmp_mount_ro" == true ]] || fail 'root-hold mountinfo lacks a read-only /tmp'
[[ "$var_tmp_mount" == true && "$var_tmp_mount_ro" == true ]] ||
  fail 'root-hold mountinfo lacks a read-only /var/tmp'
[[ "$tmp_mount_root" == "$ROOT/tmp" && "$var_tmp_mount_root" == "$ROOT/tmp" ]] ||
  fail "temporary mount sources drifted: tmp=$tmp_mount_root var_tmp=$var_tmp_mount_root"
[[ "$incoming_mount" == true ]] ||
  fail 'root-hold mountinfo lacks the systemd incoming propagation mount'
[[ "$incoming_mount_id" =~ ^[0-9]+$ ]] ||
  fail 'root-hold incoming mount identity is malformed'
[[ "$incoming_mount_root" == "/systemd/propagate/$UNIT" ]] ||
  fail "systemd incoming mount-relative source drifted: $incoming_mount_root"
incoming_source_identity="$(stat -Lc '%d:%i' "/run/systemd/propagate/$UNIT")"
incoming_target_identity="$(stat -Lc '%d:%i' "/proc/$CELL_PID/root/run/systemd/incoming")"
[[ "$incoming_source_identity" == "$incoming_target_identity" ]] ||
  fail "systemd incoming source identity drifted: source=$incoming_source_identity target=$incoming_target_identity"
[[ "$sys_mount" == true ]] || fail 'root-hold mountinfo lacks the /sys mount'
[[ "$sys_mount_root" == / && "$sys_mount_filesystem" == sysfs &&
   "$sys_mount_source" == sysfs && "$sys_mount_ro" == true ]] ||
  fail "systemd /sys mount drifted: root=$sys_mount_root filesystem=$sys_mount_filesystem source=$sys_mount_source read_only=$sys_mount_ro"
[[ "$proc_mount_count" == 1 ]] ||
  fail "typed /proc mount count drifted: observed=$proc_mount_count"
[[ "$proc_mount_root" == "$ROOT/proc" ]] ||
  fail "typed /proc mount root drifted: observed=$proc_mount_root expected=$ROOT/proc"
[[ "$proc_mount_device" == "$root_mount_device" ]] ||
  fail "typed /proc backing device drifted: proc=$proc_mount_device root=$root_mount_device"
case "$proc_mount_filesystem" in
  proc|procfs|sysfs|tmpfs)
    fail "typed /proc exposed a live virtual filesystem: $proc_mount_filesystem"
    ;;
esac
[[ "$proc_mount_vfs_ro" == true ]] || fail 'typed /proc bind is writable'
proc_source_identity="$(stat -Lc '%d:%i' "$ROOT/proc")"
proc_target_identity="$(stat -Lc '%d:%i' "/proc/$CELL_PID/root/proc")"
[[ "$proc_source_identity" == "$proc_target_identity" ]] ||
  fail "typed /proc source identity drifted: source=$proc_source_identity target=$proc_target_identity"
[[ "$(stat -c '%F:%u:%g:%a' "/proc/$CELL_PID/root/proc")" == \
   'directory:0:0:555' ]] || fail 'typed /proc metadata drifted'
proc_entry="$(find "/proc/$CELL_PID/root/proc" -mindepth 1 -print -quit)"
[[ -z "$proc_entry" ]] || fail "typed /proc bind is nonempty: $proc_entry"
if [[ ${#forbidden_mounts_observed[@]} -ne 0 ]]; then
  forbidden_mounts_joined="$(IFS=+; printf '%s' "${forbidden_mounts_observed[*]}")"
  forbidden_records_joined="$(IFS=' | '; printf '%s' "${forbidden_mount_records[*]}")"
  fail "root-hold exposed forbidden mounts: paths=$forbidden_mounts_joined mountinfo=$forbidden_records_joined"
fi

printf 'X' >&9
exec 9>&-
wait "$CLIENT_PID" || fail "root-hold client failed: $(tr '\n' ' ' < "$ERROR" 2>/dev/null)"
for attempt in $(seq 1 100); do
  [[ ! -e "/proc/$CELL_PID" ]] && break
  sleep 0.05
done
[[ ! -e "/proc/$CELL_PID" ]] || fail 'root-hold process survived release'
if systemctl is-active --quiet "$UNIT"; then
  fail 'root-hold unit survived process extinction'
fi
[[ ! -e "/proc/$CELL_PID/ns/mnt" ]] ||
  fail 'root-hold mount namespace survived process extinction'

read -r _ SYSTEMD_VERSION _ < <(systemctl --version | sed -n '1p')
[[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || fail 'systemd version is not canonical'
ready_sha256="$(printf '%s\n' "$ready" | sha256sum | cut -d ' ' -f 1)"

systemctl reset-failed "$UNIT" >/dev/null 2>&1 || true
umount "$ROOT"
ROOT_MOUNTED=false
rmdir "$ROOT/run/systemd/incoming" || fail 'bootstrap sabotage could not remove incoming'
mount --bind "$ROOT" "$ROOT"
ROOT_MOUNTED=true
mount -o remount,bind,ro,nosuid "$ROOT"

sabotage_client_status=0
if systemd-run --quiet --unit="$SABOTAGE_UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp $ROOT/tmp:/var/tmp /sys:/sys" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v10.freeze.v1 \
  < /dev/null > "$SABOTAGE_OUTPUT" 2> "$SABOTAGE_ERROR"; then
  fail 'bootstrap sabotage executed despite missing systemd incoming mountpoint'
else
  sabotage_client_status=$?
fi

sabotage_result="$(systemctl show "$SABOTAGE_UNIT" --property Result --value)"
sabotage_exec_status="$(systemctl show "$SABOTAGE_UNIT" --property ExecMainStatus --value)"
sabotage_exec_code="$(systemctl show "$SABOTAGE_UNIT" --property ExecMainCode --value)"
sabotage_journal="$(journalctl --no-pager -n 20 -u "$SABOTAGE_UNIT" 2>/dev/null)"
[[ "$sabotage_client_status" -ne 0 ]] || fail 'bootstrap sabotage client status was zero'
[[ "$sabotage_result" == exit-code ]] ||
  fail "bootstrap sabotage result drifted: $sabotage_result"
[[ "$sabotage_exec_status" == 226 ]] ||
  fail "bootstrap sabotage did not refuse with 226/NAMESPACE: $sabotage_exec_status"
[[ "$sabotage_exec_code" == 1 ]] ||
  fail "bootstrap sabotage exit code kind drifted: $sabotage_exec_code"
[[ ! -s "$SABOTAGE_OUTPUT" ]] || fail 'bootstrap sabotage reached effect-cell output'
[[ "$sabotage_journal" == *'/run/systemd/incoming'* ]] ||
  fail 'bootstrap sabotage journal omitted exact missing mountpoint'

systemctl reset-failed "$SABOTAGE_UNIT" >/dev/null 2>&1 || true
umount "$ROOT"
ROOT_MOUNTED=false
install -d -m 0555 "$ROOT/run/systemd/incoming"
rmdir "$ROOT/sys" || fail 'bootstrap sabotage could not remove sys'
mount --bind "$ROOT" "$ROOT"
ROOT_MOUNTED=true
mount -o remount,bind,ro,nosuid "$ROOT"

sys_sabotage_client_status=0
if systemd-run --quiet --unit="$SYS_SABOTAGE_UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp $ROOT/tmp:/var/tmp /sys:/sys" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v10.freeze.v1 \
  < /dev/null > "$SYS_SABOTAGE_OUTPUT" 2> "$SYS_SABOTAGE_ERROR"; then
  fail 'bootstrap sabotage executed despite missing /sys mountpoint'
else
  sys_sabotage_client_status=$?
fi

sys_sabotage_result="$(systemctl show "$SYS_SABOTAGE_UNIT" --property Result --value)"
sys_sabotage_exec_status="$(systemctl show "$SYS_SABOTAGE_UNIT" --property ExecMainStatus --value)"
sys_sabotage_exec_code="$(systemctl show "$SYS_SABOTAGE_UNIT" --property ExecMainCode --value)"
sys_sabotage_journal="$(journalctl --no-pager -n 20 -u "$SYS_SABOTAGE_UNIT" 2>/dev/null)"
[[ "$sys_sabotage_client_status" -ne 0 ]] || fail 'missing-sys client status was zero'
[[ "$sys_sabotage_result" == exit-code ]] ||
  fail "missing-sys result drifted: $sys_sabotage_result"
[[ "$sys_sabotage_exec_status" == 226 ]] ||
  fail "missing-sys did not refuse with 226/NAMESPACE: $sys_sabotage_exec_status"
[[ "$sys_sabotage_exec_code" == 1 ]] ||
  fail "missing-sys exit code kind drifted: $sys_sabotage_exec_code"
[[ ! -s "$SYS_SABOTAGE_OUTPUT" ]] || fail 'missing-sys reached effect-cell output'
[[ "$sys_sabotage_journal" == *'/sys'* ]] ||
  fail 'missing-sys journal omitted exact missing mountpoint'

systemctl reset-failed "$SYS_SABOTAGE_UNIT" >/dev/null 2>&1 || true
umount "$ROOT"
ROOT_MOUNTED=false
install -d -m 0555 "$ROOT/sys"
rmdir "$ROOT/var/tmp" || fail 'bootstrap sabotage could not remove var tmp'
mount --bind "$ROOT" "$ROOT"
ROOT_MOUNTED=true
mount -o remount,bind,ro,nosuid "$ROOT"

var_tmp_sabotage_client_status=0
if systemd-run --quiet --unit="$VAR_TMP_SABOTAGE_UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp $ROOT/tmp:/var/tmp /sys:/sys" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v10.freeze.v1 \
  < /dev/null > "$VAR_TMP_SABOTAGE_OUTPUT" 2> "$VAR_TMP_SABOTAGE_ERROR"; then
  fail 'bootstrap sabotage executed despite missing /var/tmp mountpoint'
else
  var_tmp_sabotage_client_status=$?
fi

var_tmp_sabotage_result="$(systemctl show "$VAR_TMP_SABOTAGE_UNIT" --property Result --value)"
var_tmp_sabotage_exec_status="$(systemctl show "$VAR_TMP_SABOTAGE_UNIT" --property ExecMainStatus --value)"
var_tmp_sabotage_exec_code="$(systemctl show "$VAR_TMP_SABOTAGE_UNIT" --property ExecMainCode --value)"
var_tmp_sabotage_journal="$(journalctl --no-pager -n 20 -u "$VAR_TMP_SABOTAGE_UNIT" 2>/dev/null)"
[[ "$var_tmp_sabotage_client_status" -ne 0 ]] || fail 'missing-var-tmp client status was zero'
[[ "$var_tmp_sabotage_result" == exit-code ]] ||
  fail "missing-var-tmp result drifted: $var_tmp_sabotage_result"
[[ "$var_tmp_sabotage_exec_status" == 226 ]] ||
  fail "missing-var-tmp did not refuse with 226/NAMESPACE: $var_tmp_sabotage_exec_status"
[[ "$var_tmp_sabotage_exec_code" == 1 ]] ||
  fail "missing-var-tmp exit code kind drifted: $var_tmp_sabotage_exec_code"
[[ ! -s "$VAR_TMP_SABOTAGE_OUTPUT" ]] || fail 'missing-var-tmp reached effect-cell output'
[[ "$var_tmp_sabotage_journal" == *'/var/tmp'* ]] ||
  fail 'missing-var-tmp journal omitted exact missing mountpoint'

systemctl reset-failed "$VAR_TMP_SABOTAGE_UNIT" >/dev/null 2>&1 || true
umount "$ROOT"
ROOT_MOUNTED=false
install -d -m 0555 "$ROOT/var/tmp"

BASE_BIND_PATHS="$ROOT/tmp:/tmp $ROOT/tmp:/var/tmp /sys:/sys"

# DENY453: the path is structurally present but becomes a live procfs.
mount_immutable_root
start_typed_hold "$LIVE_PROC_UNIT" "$LIVE_PROC_CONTROL" "$LIVE_PROC_OUTPUT" \
  "$LIVE_PROC_ERROR" "$BASE_BIND_PATHS"
live_proc_mutation="$($MUTATOR --pid "$TYPED_CELL_PID" --operation live-procfs)"
[[ "$live_proc_mutation" == LOOM_MOUNT_NAMESPACE_MUTATOR_PASS\ * &&
   "$live_proc_mutation" == *' operation=live-procfs semantic_decision=false' ]] ||
  fail "live-procfs mutator receipt drifted: $live_proc_mutation"
live_procfs_count=0
while IFS= read -r mount_line; do
  read -r -a fields <<< "$mount_line"
  [[ "${fields[4]:-}" == /proc ]] || continue
  separator=0
  for index in "${!fields[@]}"; do
    if [[ "${fields[$index]}" == - ]]; then separator="$index"; break; fi
  done
  [[ "$separator" -gt 5 && ${#fields[@]} -gt $((separator + 1)) ]] ||
    fail 'live-procfs control mountinfo lacks a filesystem boundary'
  if [[ "${fields[$((separator + 1))]}" == proc ]]; then
    live_procfs_count=$((live_procfs_count + 1))
  fi
done < "/proc/$TYPED_CELL_PID/mountinfo"
[[ "$live_procfs_count" -ge 1 ]] ||
  fail 'live-procfs control did not expose procfs at /proc'
release_typed_hold "$LIVE_PROC_UNIT" "$LIVE_PROC_ERROR"
unmount_immutable_root

# DENY454: all other facts remain inert, but /proc names the wrong source object.
mount_immutable_root
start_typed_hold "$WRONG_PROC_UNIT" "$WRONG_PROC_CONTROL" "$WRONG_PROC_OUTPUT" \
  "$WRONG_PROC_ERROR" "$BASE_BIND_PATHS $ROOT/tmp:/proc"
wrong_proc_expected_identity="$(stat -Lc '%d:%i' "$ROOT/proc")"
wrong_proc_injected_identity="$(stat -Lc '%d:%i' "$ROOT/tmp")"
wrong_proc_target_identity="$(stat -Lc '%d:%i' "/proc/$TYPED_CELL_PID/root/proc")"
[[ "$wrong_proc_target_identity" == "$wrong_proc_injected_identity" ]] ||
  fail "wrong-source control missed injected identity: target=$wrong_proc_target_identity injected=$wrong_proc_injected_identity"
[[ "$wrong_proc_target_identity" != "$wrong_proc_expected_identity" ]] ||
  fail 'wrong-source control retained the canonical /proc source identity'
wrong_proc_entry="$(find "/proc/$TYPED_CELL_PID/root/proc" -mindepth 1 -print -quit)"
[[ -z "$wrong_proc_entry" ]] || fail "wrong-source control is not empty: $wrong_proc_entry"
release_typed_hold "$WRONG_PROC_UNIT" "$WRONG_PROC_ERROR"
unmount_immutable_root

# DENY455: preserve source identity and contents, then flip only the VFS write bit.
mount_immutable_root
start_typed_hold "$WRITABLE_PROC_UNIT" "$WRITABLE_PROC_CONTROL" \
  "$WRITABLE_PROC_OUTPUT" "$WRITABLE_PROC_ERROR" "$BASE_BIND_PATHS"
writable_proc_source_identity="$(stat -Lc '%d:%i' "$ROOT/proc")"
writable_proc_target_identity="$(stat -Lc '%d:%i' "/proc/$TYPED_CELL_PID/root/proc")"
[[ "$writable_proc_source_identity" == "$writable_proc_target_identity" ]] ||
  fail 'writable-proc control drifted before intervention'
writable_proc_mutation="$($MUTATOR --pid "$TYPED_CELL_PID" --operation writable-proc-bind)"
[[ "$writable_proc_mutation" == LOOM_MOUNT_NAMESPACE_MUTATOR_PASS\ * &&
   "$writable_proc_mutation" == *' operation=writable-proc-bind semantic_decision=false' ]] ||
  fail "writable-proc mutator receipt drifted: $writable_proc_mutation"
writable_proc_mount=false
while IFS= read -r mount_line; do
  read -r -a fields <<< "$mount_line"
  if [[ "${fields[4]:-}" == /proc && ",${fields[5]:-}," == *,rw,* ]]; then
    writable_proc_mount=true
  fi
done < "/proc/$TYPED_CELL_PID/mountinfo"
[[ "$writable_proc_mount" == true ]] ||
  fail 'writable-proc control did not flip the /proc VFS mount to rw'
[[ "$(stat -Lc '%d:%i' "/proc/$TYPED_CELL_PID/root/proc")" == \
   "$writable_proc_source_identity" ]] ||
  fail 'writable-proc control changed source identity'
writable_proc_entry="$(find "/proc/$TYPED_CELL_PID/root/proc" -mindepth 1 -print -quit)"
[[ -z "$writable_proc_entry" ]] ||
  fail "writable-proc control changed contents: $writable_proc_entry"
release_typed_hold "$WRITABLE_PROC_UNIT" "$WRITABLE_PROC_ERROR"
unmount_immutable_root

# DENY456: the effect-cell itself must refuse when the canonical bind is nonempty.
chmod 0755 "$ROOT/proc"
printf 'typed-proc-nonempty-control\n' > "$ROOT/proc/.loom-nonempty-control"
chmod 0444 "$ROOT/proc/.loom-nonempty-control"
chmod 0555 "$ROOT/proc"
mount_immutable_root
nonempty_proc_client_status=0
if systemd-run --quiet --unit="$NONEMPTY_PROC_UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=yes \
  --property="BindReadOnlyPaths=$BASE_BIND_PATHS" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=strict \
  --property=ProtectHome=read-only \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v10.freeze.v1 \
  < /dev/null > "$NONEMPTY_PROC_OUTPUT" 2> "$NONEMPTY_PROC_ERROR"; then
  fail 'nonempty-proc control reached the hold state'
else
  nonempty_proc_client_status=$?
fi
nonempty_proc_result="$(systemctl show "$NONEMPTY_PROC_UNIT" --property Result --value)"
nonempty_proc_exec_status="$(systemctl show "$NONEMPTY_PROC_UNIT" --property ExecMainStatus --value)"
nonempty_proc_exec_code="$(systemctl show "$NONEMPTY_PROC_UNIT" --property ExecMainCode --value)"
[[ "$nonempty_proc_client_status" -ne 0 ]] || fail 'nonempty-proc client status was zero'
[[ "$nonempty_proc_result" == exit-code && "$nonempty_proc_exec_status" == 70 &&
   "$nonempty_proc_exec_code" == 1 ]] ||
  fail "nonempty-proc refusal drifted: result=$nonempty_proc_result status=$nonempty_proc_exec_status code=$nonempty_proc_exec_code"
[[ ! -s "$NONEMPTY_PROC_OUTPUT" ]] || fail 'nonempty-proc control emitted READY'
nonempty_proc_error="$(tr '\n' ' ' < "$NONEMPTY_PROC_ERROR" 2>/dev/null)"
[[ "$nonempty_proc_error" == *'LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_CLOSED reason=immutable-root directory is not empty: /proc'* ]] ||
  fail "nonempty-proc refusal omitted the V10 rule: $nonempty_proc_error"
systemctl reset-failed "$NONEMPTY_PROC_UNIT" >/dev/null 2>&1 || true
unmount_immutable_root
chmod 0755 "$ROOT/proc"
rm -f "$ROOT/proc/.loom-nonempty-control"
chmod 0555 "$ROOT/proc"

cleanup
trap - EXIT

printf 'sounio-loom-process-witness-effect-root-v10-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio producer=C++20+Sounio role=MATERIAL_PARITY action=9025 host=%s kernel=%s architecture=%s systemd_version=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE tree_sha256=%s cell_sha256=%s payload_sha256=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d policy_manifest_sha256=9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda namespace_mutator_sha256=%s namespace_mutator_role=MATERIAL_PARITY namespace_mutator_semantic_decision=false typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only identity_typed_mounts=CAPSULE_EMPTY_BIND property_private_tmp_observed=yes property_authority=CONFIGURATION_ONLY filesystem_authority=ROOT_HOST_MOUNTINFO temporary_sources=SAME_IMMUTABLE_ROOT_TMP temporary_read_only=true temporary_empty=true typed_structural_mounts=/proc:CAPSULE_EMPTY_BIND forbidden_mounts=/home+/root+/run+/var+/etc root_owned=true root_read_only=true root_exact=true root_object_identity=dev+inode executable_object_identity=dev+inode dynamic_user=true uid=%s gid=%s mount_namespace=private network_namespace=private private_tmp=disconnected protect_system=strict protect_home=read-only private_devices=false proc_treatment=CAPSULE_EMPTY_BIND proc_mount_count=1 proc_mount_source_identity=dev+inode proc_mount_filesystem=CAPSULE_ROOT_FILESYSTEM proc_mount_root_owned=true proc_mount_contents=empty proc_mount_vfs_read_only=true proc_mount_principal_writable=false procfs_visible=false tmp_read_only=true var_tmp_read_only=true var_tmp_source=IMMUTABLE_ROOT_TMP systemd_mount_path=/run/systemd/incoming systemd_mount_source=/run/systemd/propagate/EXACT_UNIT incoming_source_identity=dev+inode principal_readable=false principal_enumeration=forbidden root_observed_empty=true empty_observer=ROOT_HOST mount_observer=ROOT_HOST extinction_observer=ROOT_HOST incoming_mount_id=%s incoming_mount_extinction=observed systemd_sys_mount_path=/sys systemd_sys_ready_filesystem=sysfs systemd_sys_ready_source=sysfs systemd_sys_ready_read_only=true fd_inventory=0+1+2 capabilities=zero no_new_privileges=true seccomp=true process_extinction=observed ready_sha256=%s root_treatment=true bootstrap_sabotage=true bootstrap_missing_incoming_status=226/NAMESPACE bootstrap_missing_sys_status=226/NAMESPACE bootstrap_missing_var_tmp_status=226/NAMESPACE typed_proc_sabotages=4 bootstrap_live_procfs=DENY453 bootstrap_wrong_proc_source=DENY454 bootstrap_writable_proc_bind=DENY455 bootstrap_nonempty_proc_bind=DENY456 material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(hostname)" "$(uname -r)" "$(uname -m)" "$SYSTEMD_VERSION" \
  "$TREE_SHA256" "$CELL_SHA256" "$MUTATOR_SHA256" "$uid_real" "$gid_real" \
  "$incoming_mount_id" "$ready_sha256"
