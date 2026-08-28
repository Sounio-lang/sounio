#!/usr/bin/env bash

set -euo pipefail
umask 077

SOCKET_UNIT=sounio-loom-principal-broker.socket
SERVICE_UNIT=sounio-loom-principal-broker.service
SOCKET_PATH=/run/sounio/loom-principal-broker.sock
BROKER_LINK=/usr/libexec/sounio/loom-kernel-principal-broker
CONFIG=/etc/sounio/loom-principal-broker.conf

unavailable() {
  printf 'sounio-loom-kernel-principal-broker-host-gate: HOST_GATE_UNAVAILABLE reason=%s material_broker=false\n' "$*" >&2
  exit 77
}

fail() {
  printf 'sounio-loom-kernel-principal-broker-host-gate: FAIL reason=%s material_broker=false\n' "$*" >&2
  exit 1
}

receipt_value() {
  local receipt="$1"
  local key="$2"
  local line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate receipt field: $key"
      found="$value"
    fi
  done < "$receipt"
  [[ -n "$found" ]] || fail "receipt omitted field: $key"
  printf '%s\n' "$found"
}

config_value() {
  local key="$1"
  local line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate configuration field: $key"
      found="$value"
    fi
  done < "$CONFIG"
  [[ -n "$found" ]] || fail "configuration omitted field: $key"
  printf '%s\n' "$found"
}

root_file_mode() {
  local path="$1"
  local expected_mode="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "required regular file is absent: $path"
  [[ "$(stat -c '%u:%g:%a:%h' "$path")" == "0:0:$expected_mode:1" ]] ||
    fail "file ownership, mode, or link count drifted: $path"
}

[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] || unavailable 'PID 1 is not systemd'
[[ -d /run/systemd/system ]] || unavailable 'systemd runtime directory is absent'
command -v systemctl >/dev/null 2>&1 || unavailable 'systemctl is absent'

[[ -L "$BROKER_LINK" ]] || fail 'stable broker path is not a symlink'
[[ "$(stat -c '%u:%g' "$BROKER_LINK")" == 0:0 ]] || fail 'stable broker symlink is not root-owned'
BROKER_TARGET="$(readlink "$BROKER_LINK")"
[[ "$BROKER_TARGET" =~ ^/usr/lib/sounio/loom/releases/9027-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}/loom-kernel-principal-broker$ ]] ||
  fail 'stable broker target is outside an action 9027 release'
RELEASE_DIR="${BROKER_TARGET%/loom-kernel-principal-broker}"
RECEIPT="$RELEASE_DIR/install.receipt.v1"
MANIFEST="$RELEASE_DIR/kernel_principal_lease_authority.freeze.v1"
AUTHORITY="$RELEASE_DIR/sounio-loom-kernel-principal-lease-authority-runtime"

root_file_mode "$BROKER_TARGET" 555
root_file_mode "$MANIFEST" 444
root_file_mode "$AUTHORITY" 555
root_file_mode "$RECEIPT" 444
root_file_mode "$RELEASE_DIR/install_loom_kernel_principal_broker.sh" 444
root_file_mode "$CONFIG" 600
root_file_mode "/etc/systemd/system/$SOCKET_UNIT" 644
root_file_mode "/etc/systemd/system/$SERVICE_UNIT" 644
root_file_mode "/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md" 444
root_file_mode "/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md" 444
[[ "$(stat -c '%u:%g:%a' "$RELEASE_DIR")" == 0:0:555 ]] || fail 'release directory metadata drifted'

[[ "$(receipt_value "$RECEIPT" semantic_producer)" == Sounio ]] || fail 'receipt semantic producer drifted'
[[ "$(receipt_value "$RECEIPT" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'receipt semantic role drifted'
[[ "$(receipt_value "$RECEIPT" material_producer)" == C++20 ]] || fail 'receipt material producer drifted'
[[ "$(receipt_value "$RECEIPT" material_role)" == MATERIAL_PARITY ]] || fail 'receipt material role drifted'
[[ "$(receipt_value "$RECEIPT" material_transitory)" == true ]] || fail 'receipt lost transitory marker'
[[ "$(receipt_value "$RECEIPT" launch_open)" == false ]] || fail 'receipt opened launch'
[[ "$(receipt_value "$RECEIPT" recycle_open)" == false ]] || fail 'receipt opened recycle'
[[ "$(receipt_value "$RECEIPT" material_broker)" == false ]] || fail 'receipt promoted material broker'

MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
AUTHORITY_SHA256="$(sha256sum "$AUTHORITY" | cut -d ' ' -f 1)"
BROKER_SHA256="$(sha256sum "$BROKER_TARGET" | cut -d ' ' -f 1)"
[[ "$MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" manifest_sha256)" ]] || fail 'installed manifest hash drifted'
[[ "$AUTHORITY_SHA256" == "$(receipt_value "$RECEIPT" authority_sha256)" ]] || fail 'installed authority hash drifted'
[[ "$BROKER_SHA256" == "$(receipt_value "$RECEIPT" broker_sha256)" ]] || fail 'installed broker hash drifted'
[[ "$(sha256sum "$RELEASE_DIR/install_loom_kernel_principal_broker.sh" | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" installer_sha256)" ]] ||
  fail 'installed installer hash drifted'
[[ "$(sha256sum "/etc/systemd/system/$SOCKET_UNIT" | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" socket_unit_sha256)" ]] ||
  fail 'installed socket unit hash drifted'
[[ "$(sha256sum "/etc/systemd/system/$SERVICE_UNIT" | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" service_unit_sha256)" ]] ||
  fail 'installed service unit hash drifted'
[[ "$(sha256sum /usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" bootstrap_doc_sha256)" ]] ||
  fail 'installed bootstrap contract hash drifted'
[[ "$(sha256sum /usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" install_doc_sha256)" ]] ||
  fail 'installed installation contract hash drifted'
[[ "$(config_value LOOM_PRINCIPAL_MANIFEST)" == "$MANIFEST" ]] || fail 'configuration manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_AUTHORITY)" == "$AUTHORITY" ]] || fail 'configuration authority target drifted'
[[ "$(config_value LOOM_PRINCIPAL_JOURNAL)" == /var/lib/sounio/loom-principal-broker/leases.v1 ]] ||
  fail 'configuration journal target drifted'

systemctl is-enabled --quiet "$SOCKET_UNIT" || fail 'broker socket is not enabled'
systemctl is-active --quiet "$SOCKET_UNIT" || fail 'broker socket is not active'
[[ -S "$SOCKET_PATH" ]] || fail 'broker socket path is absent'
[[ "$(stat -c '%u:%g:%a' "$SOCKET_PATH")" == 0:0:600 ]] || fail 'broker socket metadata drifted'

PROBE="$($BROKER_LINK --probe-live --socket-path "$SOCKET_PATH")"
[[ "$PROBE" == 'LOOM_KERNEL_PRINCIPAL_BROKER_STATUS state=READY '* ]] || fail 'live broker did not report READY'
[[ "$PROBE" == *" manifest_sha256=$MANIFEST_SHA256 "* ]] || fail 'live broker reported a different manifest'
[[ "$PROBE" == *" authority_sha256=$AUTHORITY_SHA256 "* ]] || fail 'live broker reported a different authority'
[[ "$PROBE" == *' live_probe=PASS launch=closed recycle=closed unknown=denied' ]] ||
  fail 'live broker negative protocol controls failed'

systemctl is-active --quiet "$SERVICE_UNIT" || fail 'broker service is not active after socket probe'
MAIN_PID="$(systemctl show "$SERVICE_UNIT" --property MainPID --value)"
[[ "$MAIN_PID" =~ ^[0-9]+$ && "$MAIN_PID" -gt 1 ]] || fail 'broker service has no live MainPID'
[[ -r "/proc/$MAIN_PID/cgroup" ]] || fail 'broker service process disappeared'
grep -Fq 'sounio-loom-principal-broker.service' "/proc/$MAIN_PID/cgroup" ||
  fail 'broker process is outside the expected service cgroup'
[[ "$(systemctl show "$SERVICE_UNIT" --property User --value)" == root ]] || fail 'broker service User is not root'
[[ "$(systemctl show "$SERVICE_UNIT" --property Group --value)" == root ]] || fail 'broker service Group is not root'
[[ "$(systemctl show "$SERVICE_UNIT" --property Delegate --value)" == yes ]] || fail 'broker service lacks cgroup delegation'
[[ "$(systemctl show "$SERVICE_UNIT" --property NoNewPrivileges --value)" == yes ]] ||
  fail 'broker service lost NoNewPrivileges'

printf 'sounio-loom-kernel-principal-broker-host-gate: HOST_ACTIVATION_PASS release=%s main_pid=%s manifest_sha256=%s authority_sha256=%s broker_sha256=%s socket_activation=verified service_cgroup=verified root_peer=verified launch=closed recycle=closed material_broker=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n' \
  "$(receipt_value "$RECEIPT" release_id)" "$MAIN_PID" "$MANIFEST_SHA256" "$AUTHORITY_SHA256" "$BROKER_SHA256"
