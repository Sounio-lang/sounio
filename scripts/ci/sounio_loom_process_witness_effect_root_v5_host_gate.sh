#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-process-witness-effect-root-v5-host-gate: FAIL reason=%s root_treatment=false bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-process-witness-effect-root-v5-host-gate: HOST_GATE_UNAVAILABLE reason=%s root_treatment=false bootstrap_sabotage=false material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --root ABSOLUTE_PATH --cell-sha256 HEX --tree-sha256 HEX\n' "$0" >&2
  exit 64
}

ROOT=''
CELL_SHA256=''
TREE_SHA256=''
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
    *) usage ;;
  esac
done

[[ "$ROOT" == /* && "$ROOT" != / && "$ROOT" =~ ^[A-Za-z0-9._/-]+$ &&
   "$CELL_SHA256" =~ ^[0-9a-f]{64}$ && "$TREE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  usage
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] ||
  unavailable 'PID 1 is not systemd'
for tool in systemctl systemd-run sha256sum stat readelf find readlink mkfifo \
            mount umount journalctl; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done

expected_paths="$ROOT/dev
$ROOT/dev/null
$ROOT/loom
$ROOT/loom/effect-cell
$ROOT/loom/effect-policy-v5.freeze.v1
$ROOT/loom/payload
$ROOT/loom/payload.freeze.v1
$ROOT/proc
$ROOT/run
$ROOT/run/systemd
$ROOT/run/systemd/incoming
$ROOT/sys
$ROOT/tmp"
actual_paths="$(find "$ROOT" -mindepth 1 -printf '%p\n' | sort)"
[[ "$actual_paths" == "$expected_paths" ]] || fail 'host root path set drifted'
for directory in "$ROOT" "$ROOT/loom" "$ROOT/dev" "$ROOT/proc" "$ROOT/tmp" \
                 "$ROOT/run" "$ROOT/run/systemd" "$ROOT/run/systemd/incoming" \
                 "$ROOT/sys"; do
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
                "$ROOT/loom/effect-policy-v5.freeze.v1"; do
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
[[ "$(sha256sum "$ROOT/loom/effect-policy-v5.freeze.v1" | cut -d ' ' -f 1)" == \
  f17fc7d776db557d2655e00036f4014b4a7a38d8ed16e74786471415c49908f7 ]] ||
  fail 'host root policy manifest hash drifted'

nonce="$$-$(date +%s%N)"
UNIT="sounio-loom-effect-root-v5-$nonce.service"
SABOTAGE_UNIT="sounio-loom-effect-root-v5-missing-incoming-$nonce.service"
SYS_SABOTAGE_UNIT="sounio-loom-effect-root-v5-missing-sys-$nonce.service"
CONTROL="/run/sounio-loom-effect-root-v5-$nonce.control"
OUTPUT="/run/sounio-loom-effect-root-v5-$nonce.output"
ERROR="/run/sounio-loom-effect-root-v5-$nonce.error"
SABOTAGE_OUTPUT="/run/sounio-loom-effect-root-v5-missing-incoming-$nonce.output"
SABOTAGE_ERROR="/run/sounio-loom-effect-root-v5-missing-incoming-$nonce.error"
SYS_SABOTAGE_OUTPUT="/run/sounio-loom-effect-root-v5-missing-sys-$nonce.output"
SYS_SABOTAGE_ERROR="/run/sounio-loom-effect-root-v5-missing-sys-$nonce.error"
CLIENT_PID=''
CELL_PID=''
ROOT_MOUNTED=false

cleanup() {
  { exec 9>&-; } 2>/dev/null || true
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
  if [[ "$ROOT_MOUNTED" == true ]]; then
    umount "$ROOT" >/dev/null 2>&1 || true
    ROOT_MOUNTED=false
  fi
  rm -f "$CONTROL" "$OUTPUT" "$ERROR" "$SABOTAGE_OUTPUT" "$SABOTAGE_ERROR" \
    "$SYS_SABOTAGE_OUTPUT" "$SYS_SABOTAGE_ERROR"
}
trap cleanup EXIT

mount --bind "$ROOT" "$ROOT"
ROOT_MOUNTED=true
mount -o remount,bind,ro,nosuid "$ROOT"
mkfifo -m 0600 "$CONTROL"
exec 9<>"$CONTROL"
systemd-run --quiet --unit="$UNIT" --service-type=exec --pipe --wait \
  --property="RootDirectory=$ROOT" \
  --property=MountAPIVFS=no \
  --property=DynamicUser=yes \
  --property=UMask=0077 \
  --property=NoNewPrivileges=yes \
  --property=PrivateTmp=no \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=no \
  --property=ProtectHome=no \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v5.freeze.v1 \
  < "$CONTROL" > "$OUTPUT" 2> "$ERROR" &
CLIENT_PID=$!

for attempt in $(seq 1 200); do
  if grep -q '^LOOM_PROCESS_WITNESS_EFFECT_POLICY_V5_ROOT_READY PASS ' "$OUTPUT" 2>/dev/null; then
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
[[ "$ready" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_V5_ROOT_READY\ PASS* ]] ||
  fail 'root-hold did not emit READY'
for expectation in \
  'semantic_authority=Sounio' 'action=9025' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' 'root_read_only=true' \
  'root_exact=true' 'dynamic_linker_visible=false' 'host_root_visible=false' \
  'proc_treatment=absent' 'tmp_read_only=true' 'fd_inventory=0+1+2' \
  "cell_sha256=$CELL_SHA256" \
  'payload_sha256=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d' \
  'policy_manifest_sha256=f17fc7d776db557d2655e00036f4014b4a7a38d8ed16e74786471415c49908f7' \
  'systemd_mount=/run/systemd/incoming' 'systemd_sys_mount=/sys' \
  'systemd_sys_ready_filesystem=sysfs' 'systemd_sys_ready_read_only=true' \
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
[[ "$(systemctl show "$UNIT" --property PrivateTmp --value)" == no ]] ||
  fail 'root-hold gained a writable private /tmp'
[[ "$(systemctl show "$UNIT" --property PrivateDevices --value)" == no ]] ||
  fail 'root-hold replaced the exact /dev/null view'
[[ "$(systemctl show "$UNIT" --property PrivateNetwork --value)" == yes ]] ||
  fail 'root-hold lost its private network namespace'
[[ "$(systemctl show "$UNIT" --property ProtectSystem --value)" == no ]] ||
  fail 'root-hold did not leave the immutable root mount authoritative'
[[ "$(systemctl show "$UNIT" --property ProtectHome --value)" == no ]] ||
  fail 'root-hold added a redundant home mount policy'
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
[[ "$(readlink "/proc/$CELL_PID/root")" == "$ROOT" ]] ||
  fail 'root-hold process root escaped the materialized root'
[[ "$(readlink "/proc/$CELL_PID/exe")" == "$ROOT/loom/effect-cell" ]] ||
  fail 'root-hold executable drifted'
[[ "$(readlink "/proc/$CELL_PID/ns/mnt")" != "$(readlink /proc/1/ns/mnt)" ]] ||
  fail 'root-hold shares PID 1 mount namespace'
[[ "$(readlink "/proc/$CELL_PID/ns/net")" != "$(readlink /proc/1/ns/net)" ]] ||
  fail 'root-hold shares PID 1 network namespace'
grep -Fq "$UNIT" "/proc/$CELL_PID/cgroup" || fail 'root-hold cgroup identity drifted'

fd_paths=(/proc/"$CELL_PID"/fd/*)
[[ ${#fd_paths[@]} -eq 3 ]] || fail "root-hold fd inventory is not exact: ${#fd_paths[@]}"
for descriptor in 0 1 2; do
  [[ -e "/proc/$CELL_PID/fd/$descriptor" ]] || fail "root-hold omitted fd $descriptor"
done

root_mount_ro=false
tmp_mount_ro=false
incoming_mount=false
incoming_mount_root=''
sys_mount=false
sys_mount_root=''
sys_mount_filesystem=''
sys_mount_source=''
sys_mount_ro=false
forbidden_mount=''
while IFS= read -r mount_line; do
  read -r -a fields <<< "$mount_line"
  [[ ${#fields[@]} -ge 6 ]] || fail 'root-hold mountinfo is malformed'
  mountpoint="${fields[4]}"
  options=",${fields[5]},"
  if [[ "$mountpoint" == / && "$options" == *,ro,* ]]; then
    root_mount_ro=true
  fi
  if [[ "$mountpoint" == /tmp && "$options" == *,ro,* ]]; then
    tmp_mount_ro=true
  fi
  if [[ "$mountpoint" == /run/systemd/incoming ]]; then
    incoming_mount=true
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
  case "$mountpoint" in
    /proc|/home|/root|/run|/var|/etc) forbidden_mount="$mountpoint" ;;
  esac
done < "/proc/$CELL_PID/mountinfo"
[[ "$root_mount_ro" == true ]] || fail 'root-hold mountinfo lacks a read-only root'
[[ "$tmp_mount_ro" == true ]] || fail 'root-hold mountinfo lacks a read-only /tmp'
[[ "$incoming_mount" == true ]] ||
  fail 'root-hold mountinfo lacks the systemd incoming propagation mount'
[[ "$incoming_mount_root" == "/run/systemd/propagate/$UNIT" ]] ||
  fail "systemd incoming mount source drifted: $incoming_mount_root"
[[ "$sys_mount" == true ]] || fail 'root-hold mountinfo lacks the /sys mount'
[[ "$sys_mount_root" == / && "$sys_mount_filesystem" == sysfs &&
   "$sys_mount_source" == sysfs && "$sys_mount_ro" == true ]] ||
  fail "systemd /sys mount drifted: root=$sys_mount_root filesystem=$sys_mount_filesystem source=$sys_mount_source read_only=$sys_mount_ro"
[[ -z "$forbidden_mount" ]] || fail "root-hold exposed forbidden mount: $forbidden_mount"

printf 'X' >&9
exec 9>&-
wait "$CLIENT_PID" || fail "root-hold client failed: $(tr '\n' ' ' < "$ERROR" 2>/dev/null)"
for attempt in $(seq 1 100); do
  [[ ! -e "/proc/$CELL_PID" ]] && break
  sleep 0.05
done
[[ ! -e "/proc/$CELL_PID" ]] || fail 'root-hold process survived release'

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
  --property=PrivateTmp=no \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=no \
  --property=ProtectHome=no \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v5.freeze.v1 \
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
  --property=PrivateTmp=no \
  --property="BindReadOnlyPaths=$ROOT/tmp:/tmp" \
  --property=PrivateDevices=no \
  --property=PrivateNetwork=yes \
  --property=ProtectSystem=no \
  --property=ProtectHome=no \
  --property=RestrictNamespaces=yes \
  --property=RestrictSUIDSGID=yes \
  --property=LockPersonality=yes \
  --property=MemoryDenyWriteExecute=yes \
  --property=RestrictRealtime=yes \
  --property=SystemCallArchitectures=native \
  --property=KillMode=mixed \
  --property=TimeoutStopSec=2s \
  -- /loom/effect-cell --root-hold \
     --policy-manifest /loom/effect-policy-v5.freeze.v1 \
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
cleanup
trap - EXIT

printf 'sounio-loom-process-witness-effect-root-v5-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio producer=C++20+Sounio role=MATERIAL_PARITY action=9025 host=%s kernel=%s architecture=%s systemd_version=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE tree_sha256=%s cell_sha256=%s payload_sha256=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d policy_manifest_sha256=f17fc7d776db557d2655e00036f4014b4a7a38d8ed16e74786471415c49908f7 root_owned=true root_read_only=true root_exact=true dynamic_user=true uid=%s gid=%s mount_namespace=private network_namespace=private private_tmp=false private_devices=false proc_treatment=absent tmp_read_only=true systemd_mount_path=/run/systemd/incoming systemd_mount_source=/run/systemd/propagate/EXACT_UNIT systemd_sys_mount_path=/sys systemd_sys_ready_filesystem=sysfs systemd_sys_ready_source=sysfs systemd_sys_ready_read_only=true fd_inventory=0+1+2 capabilities=zero no_new_privileges=true seccomp=true process_extinction=observed ready_sha256=%s root_treatment=true bootstrap_sabotage=true bootstrap_missing_incoming_status=226/NAMESPACE bootstrap_missing_sys_status=226/NAMESPACE material_sabotages=0 material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(hostname)" "$(uname -r)" "$(uname -m)" "$SYSTEMD_VERSION" \
  "$TREE_SHA256" "$CELL_SHA256" "$uid_real" "$gid_real" "$ready_sha256"
