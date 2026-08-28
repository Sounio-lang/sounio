#!/usr/bin/env bash

set -euo pipefail
umask 077

SOCKET_UNIT=sounio-loom-principal-broker.socket
SERVICE_UNIT=sounio-loom-principal-broker.service
SOCKET_PATH=/run/sounio/loom-principal-broker.sock
BROKER_LINK=/usr/libexec/sounio/loom-kernel-principal-broker
CONFIG=/etc/sounio/loom-principal-broker.conf

unavailable() {
  printf 'sounio-loom-kernel-principal-broker-host-gate: HOST_GATE_UNAVAILABLE reason=%s material_broker=false material_capsule=false\n' "$*" >&2
  exit 77
}

fail() {
  printf 'sounio-loom-kernel-principal-broker-host-gate: FAIL reason=%s material_broker=false material_capsule=false\n' "$*" >&2
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

line_value() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  fail "live status omitted field: $key"
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
command -v timeout >/dev/null 2>&1 || unavailable 'timeout is absent'

[[ -L "$BROKER_LINK" ]] || fail 'stable broker path is not a symlink'
[[ "$(stat -c '%u:%g' "$BROKER_LINK")" == 0:0 ]] || fail 'stable broker symlink is not root-owned'
BROKER_TARGET="$(readlink "$BROKER_LINK")"
[[ "$BROKER_TARGET" =~ ^/usr/lib/sounio/loom/releases/9030-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}/loom-kernel-principal-broker$ ]] ||
  fail 'stable broker target is outside an action 9027+9028+9029+9030 release'
RELEASE_DIR="${BROKER_TARGET%/loom-kernel-principal-broker}"
RECEIPT="$RELEASE_DIR/install.receipt.v1"
MANIFEST="$RELEASE_DIR/kernel_principal_lease_authority.freeze.v1"
AUTHORITY="$RELEASE_DIR/sounio-loom-kernel-principal-lease-authority-runtime"
CAPSULE_MANIFEST="$RELEASE_DIR/kernel_principal_capsule_authority.freeze.v1"
CAPSULE_AUTHORITY="$RELEASE_DIR/sounio-loom-kernel-principal-capsule-authority-runtime"
INVOCATION_MANIFEST="$RELEASE_DIR/kernel_invocation_cell_authority.freeze.v1"
INVOCATION_AUTHORITY="$RELEASE_DIR/sounio-loom-kernel-invocation-cell-authority-runtime"
EXEC_GRANT_MANIFEST="$RELEASE_DIR/kernel_exec_grant_cell_authority.freeze.v1"
RESIDENT_MANIFEST="$RELEASE_DIR/resident_membrane.runtime.v4"
RESIDENT_RUNTIME="$RELEASE_DIR/sounio-loom-resident-membrane-runtime-v4"

root_file_mode "$BROKER_TARGET" 555
root_file_mode "$MANIFEST" 444
root_file_mode "$AUTHORITY" 555
root_file_mode "$CAPSULE_MANIFEST" 444
root_file_mode "$CAPSULE_AUTHORITY" 555
root_file_mode "$INVOCATION_MANIFEST" 444
root_file_mode "$INVOCATION_AUTHORITY" 555
root_file_mode "$EXEC_GRANT_MANIFEST" 444
root_file_mode "$RESIDENT_MANIFEST" 444
root_file_mode "$RESIDENT_RUNTIME" 555
root_file_mode "$RECEIPT" 444
root_file_mode "$RELEASE_DIR/install_loom_kernel_principal_broker.sh" 444
root_file_mode "$CONFIG" 600
root_file_mode "/etc/systemd/system/$SOCKET_UNIT" 644
root_file_mode "/etc/systemd/system/$SERVICE_UNIT" 644
root_file_mode "/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md" 444
root_file_mode "/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md" 444
root_file_mode "/usr/share/doc/sounio/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md" 444
root_file_mode "/usr/share/doc/sounio/loom/HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md" 444
[[ "$(stat -c '%u:%g:%a' "$RELEASE_DIR")" == 0:0:555 ]] || fail 'release directory metadata drifted'

[[ "$(receipt_value "$RECEIPT" semantic_producer)" == Sounio ]] || fail 'receipt semantic producer drifted'
[[ "$(receipt_value "$RECEIPT" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'receipt semantic role drifted'
[[ "$(receipt_value "$RECEIPT" semantic_actions)" == 9027+9028+9029+9030 ]] || fail 'receipt semantic actions drifted'
[[ "$(receipt_value "$RECEIPT" material_producer)" == C++20 ]] || fail 'receipt material producer drifted'
[[ "$(receipt_value "$RECEIPT" material_role)" == MATERIAL_PARITY ]] || fail 'receipt material role drifted'
[[ "$(receipt_value "$RECEIPT" material_transitory)" == true ]] || fail 'receipt lost transitory marker'
[[ "$(receipt_value "$RECEIPT" launch_open)" == false ]] || fail 'receipt opened launch'
[[ "$(receipt_value "$RECEIPT" recycle_open)" == false ]] || fail 'receipt opened recycle'
[[ "$(receipt_value "$RECEIPT" admission_open)" == true ]] || fail 'receipt closed decision admission'
[[ "$(receipt_value "$RECEIPT" material_broker)" == false ]] || fail 'receipt promoted material broker'
[[ "$(receipt_value "$RECEIPT" material_capsule)" == false ]] || fail 'receipt promoted material capsule'
[[ "$(receipt_value "$RECEIPT" material_invocation)" == false ]] || fail 'receipt promoted material invocation'
[[ "$(receipt_value "$RECEIPT" resident_action_9030_attached)" == true ]] ||
  fail 'receipt omitted resident action 9030'
[[ "$(receipt_value "$RECEIPT" decision_transport_material)" == true ]] ||
  fail 'receipt omitted material decision transport'
[[ "$(receipt_value "$RECEIPT" material_grant)" == false ]] || fail 'receipt promoted material grant'
[[ "$(receipt_value "$RECEIPT" material_execution)" == false ]] || fail 'receipt promoted material execution'
[[ "$(receipt_value "$RECEIPT" barrier_release)" == false ]] || fail 'receipt released the host barrier'

MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
AUTHORITY_SHA256="$(sha256sum "$AUTHORITY" | cut -d ' ' -f 1)"
CAPSULE_MANIFEST_SHA256="$(sha256sum "$CAPSULE_MANIFEST" | cut -d ' ' -f 1)"
CAPSULE_AUTHORITY_SHA256="$(sha256sum "$CAPSULE_AUTHORITY" | cut -d ' ' -f 1)"
INVOCATION_MANIFEST_SHA256="$(sha256sum "$INVOCATION_MANIFEST" | cut -d ' ' -f 1)"
INVOCATION_AUTHORITY_SHA256="$(sha256sum "$INVOCATION_AUTHORITY" | cut -d ' ' -f 1)"
EXEC_GRANT_MANIFEST_SHA256="$(sha256sum "$EXEC_GRANT_MANIFEST" | cut -d ' ' -f 1)"
RESIDENT_MANIFEST_SHA256="$(sha256sum "$RESIDENT_MANIFEST" | cut -d ' ' -f 1)"
RESIDENT_RUNTIME_SHA256="$(sha256sum "$RESIDENT_RUNTIME" | cut -d ' ' -f 1)"
BROKER_SHA256="$(sha256sum "$BROKER_TARGET" | cut -d ' ' -f 1)"
[[ "$MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" lease_manifest_sha256)" ]] || fail 'installed manifest hash drifted'
[[ "$AUTHORITY_SHA256" == "$(receipt_value "$RECEIPT" lease_authority_sha256)" ]] || fail 'installed authority hash drifted'
[[ "$CAPSULE_MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" capsule_manifest_sha256)" ]] ||
  fail 'installed capsule manifest hash drifted'
[[ "$CAPSULE_AUTHORITY_SHA256" == "$(receipt_value "$RECEIPT" capsule_authority_sha256)" ]] ||
  fail 'installed capsule authority hash drifted'
[[ "$INVOCATION_MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" invocation_manifest_sha256)" ]] ||
  fail 'installed InvocationCell manifest hash drifted'
[[ "$INVOCATION_AUTHORITY_SHA256" == "$(receipt_value "$RECEIPT" invocation_authority_sha256)" ]] ||
  fail 'installed InvocationCell authority hash drifted'
[[ "$EXEC_GRANT_MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" exec_grant_manifest_sha256)" ]] ||
  fail 'installed ExecGrantCell manifest hash drifted'
[[ "$RESIDENT_MANIFEST_SHA256" == "$(receipt_value "$RECEIPT" resident_manifest_sha256)" ]] ||
  fail 'installed resident manifest hash drifted'
[[ "$RESIDENT_RUNTIME_SHA256" == "$(receipt_value "$RECEIPT" resident_runtime_sha256)" ]] ||
  fail 'installed resident runtime hash drifted'
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
[[ "$(sha256sum /usr/share/doc/sounio/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" invocation_doc_sha256)" ]] ||
  fail 'installed InvocationCell admission contract hash drifted'
[[ "$(sha256sum /usr/share/doc/sounio/loom/HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md | cut -d ' ' -f 1)" == "$(receipt_value "$RECEIPT" resident_attachment_doc_sha256)" ]] ||
  fail 'installed resident attachment contract hash drifted'
[[ "$(config_value LOOM_PRINCIPAL_MANIFEST)" == "$MANIFEST" ]] || fail 'configuration manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_AUTHORITY)" == "$AUTHORITY" ]] || fail 'configuration authority target drifted'
[[ "$(config_value LOOM_PRINCIPAL_CAPSULE_MANIFEST)" == "$CAPSULE_MANIFEST" ]] ||
  fail 'configuration capsule manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_CAPSULE_AUTHORITY)" == "$CAPSULE_AUTHORITY" ]] ||
  fail 'configuration capsule authority target drifted'
[[ "$(config_value LOOM_PRINCIPAL_INVOCATION_MANIFEST)" == "$INVOCATION_MANIFEST" ]] ||
  fail 'configuration InvocationCell manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_INVOCATION_AUTHORITY)" == "$INVOCATION_AUTHORITY" ]] ||
  fail 'configuration InvocationCell authority target drifted'
[[ "$(config_value LOOM_PRINCIPAL_EXEC_GRANT_MANIFEST)" == "$EXEC_GRANT_MANIFEST" ]] ||
  fail 'configuration ExecGrantCell manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_RESIDENT_MANIFEST)" == "$RESIDENT_MANIFEST" ]] ||
  fail 'configuration resident manifest target drifted'
[[ "$(config_value LOOM_PRINCIPAL_RESIDENT_RUNTIME)" == "$RESIDENT_RUNTIME" ]] ||
  fail 'configuration resident runtime target drifted'
[[ "$(config_value LOOM_PRINCIPAL_JOURNAL)" == /var/lib/sounio/loom-principal-broker/leases.v1 ]] ||
  fail 'configuration journal target drifted'

systemctl is-enabled --quiet "$SOCKET_UNIT" || fail 'broker socket is not enabled'
systemctl is-active --quiet "$SOCKET_UNIT" || fail 'broker socket is not active'
[[ -S "$SOCKET_PATH" ]] || fail 'broker socket path is absent'
[[ "$(stat -c '%u:%g:%a' "$SOCKET_PATH")" == 0:0:600 ]] || fail 'broker socket metadata drifted'

set +e
PROBE="$(timeout --signal=TERM --kill-after=2s 15s \
  "$BROKER_LINK" --probe-live --socket-path "$SOCKET_PATH" 2>&1)"
PROBE_STATUS=$?
set -e
[[ $PROBE_STATUS -eq 0 ]] ||
  fail "live broker probe failed or timed out status=$PROBE_STATUS output=$PROBE"
[[ "$PROBE" == 'LOOM_KERNEL_PRINCIPAL_BROKER_STATUS state=READY '* ]] || fail 'live broker did not report READY'
[[ "$PROBE" == *" lease_manifest_sha256=$MANIFEST_SHA256 "* ]] ||
  fail 'live broker reported a different lease manifest'
[[ "$PROBE" == *" lease_authority_sha256=$AUTHORITY_SHA256 "* ]] ||
  fail 'live broker reported a different lease authority'
[[ "$PROBE" == *" capsule_manifest_sha256=$CAPSULE_MANIFEST_SHA256 "* ]] ||
  fail 'live broker reported a different capsule manifest'
[[ "$PROBE" == *" capsule_authority_sha256=$CAPSULE_AUTHORITY_SHA256 "* ]] ||
  fail 'live broker reported a different capsule authority'
[[ "$PROBE" == *" invocation_manifest_sha256=$INVOCATION_MANIFEST_SHA256 "* ]] ||
  fail 'live broker reported a different InvocationCell manifest'
[[ "$PROBE" == *" invocation_authority_sha256=$INVOCATION_AUTHORITY_SHA256 "* ]] ||
  fail 'live broker reported a different InvocationCell authority'
[[ "$PROBE" == *" exec_grant_manifest_sha256=$EXEC_GRANT_MANIFEST_SHA256 "* ]] ||
  fail 'live broker reported a different ExecGrantCell manifest'
[[ "$PROBE" == *" resident_manifest_sha256=$RESIDENT_MANIFEST_SHA256 "* ]] ||
  fail 'live broker reported a different resident manifest'
[[ "$PROBE" == *" resident_runtime_sha256=$RESIDENT_RUNTIME_SHA256 "* ]] ||
  fail 'live broker reported a different resident runtime'
[[ "$PROBE" == *' resident_poisoned=false '* ]] ||
  fail 'live resident generation is poisoned'
[[ "$PROBE" == *' live_probe=PASS admission=DENY424 grant_admission=DENY424 resident_action_9030=verified barrier_release=false launch=closed recycle=closed unknown=denied' ]] ||
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

RESIDENT_PID="$(line_value "$PROBE" resident_pid)"
RESIDENT_START_TICK="$(line_value "$PROBE" resident_start_tick)"
RESIDENT_GENERATION="$(line_value "$PROBE" resident_generation_sha256)"
RESIDENT_SEQUENCE_ONE="$(line_value "$PROBE" resident_sequence)"
[[ "$RESIDENT_PID" =~ ^[0-9]+$ && "$RESIDENT_PID" -gt 1 ]] ||
  fail 'live resident PID is invalid'
[[ "$RESIDENT_START_TICK" =~ ^[0-9]+$ && "$RESIDENT_START_TICK" -gt 0 ]] ||
  fail 'live resident start tick is invalid'
[[ "$RESIDENT_GENERATION" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'live resident generation digest is invalid'
[[ "$RESIDENT_SEQUENCE_ONE" =~ ^[0-9]+$ ]] ||
  fail 'live resident sequence is invalid'
[[ -r "/proc/$RESIDENT_PID/stat" && -r "/proc/$RESIDENT_PID/cgroup" ]] ||
  fail 'live resident process disappeared'
RESIDENT_PPID="$(sed -n 's/^[^)]*) //p' "/proc/$RESIDENT_PID/stat" | awk '{print $2}')"
RESIDENT_START_TICK_OBSERVED="$(sed -n 's/^[^)]*) //p' "/proc/$RESIDENT_PID/stat" | awk '{print $20}')"
[[ "$RESIDENT_PPID" == "$MAIN_PID" ]] || fail 'resident is not a direct broker child'
[[ "$RESIDENT_START_TICK_OBSERVED" == "$RESIDENT_START_TICK" ]] ||
  fail 'resident start tick differs from broker status'
[[ "$(readlink -f "/proc/$RESIDENT_PID/exe")" == "$RESIDENT_RUNTIME" ]] ||
  fail 'resident executable differs from immutable runtime'
grep -Fq 'sounio-loom-principal-broker.service' "/proc/$RESIDENT_PID/cgroup" ||
  fail 'resident process is outside the broker service cgroup'

set +e
PROBE_TWO="$(timeout --signal=TERM --kill-after=2s 15s \
  "$BROKER_LINK" --probe-live --socket-path "$SOCKET_PATH" 2>&1)"
PROBE_TWO_STATUS=$?
set -e
[[ $PROBE_TWO_STATUS -eq 0 ]] ||
  fail "second live broker probe failed status=$PROBE_TWO_STATUS output=$PROBE_TWO"
[[ "$(line_value "$PROBE_TWO" resident_pid)" == "$RESIDENT_PID" ]] ||
  fail 'resident PID changed between live requests'
[[ "$(line_value "$PROBE_TWO" resident_start_tick)" == "$RESIDENT_START_TICK" ]] ||
  fail 'resident start tick changed between live requests'
[[ "$(line_value "$PROBE_TWO" resident_generation_sha256)" == "$RESIDENT_GENERATION" ]] ||
  fail 'resident generation changed between live requests'
RESIDENT_SEQUENCE_TWO="$(line_value "$PROBE_TWO" resident_sequence)"
[[ "$RESIDENT_SEQUENCE_TWO" =~ ^[0-9]+$ && \
  "$RESIDENT_SEQUENCE_TWO" -eq $((RESIDENT_SEQUENCE_ONE + 1)) ]] ||
  fail 'resident sequence did not advance monotonically'
[[ "$PROBE_TWO" == *' resident_poisoned=false '* ]] ||
  fail 'resident became poisoned between live requests'

printf 'sounio-loom-kernel-principal-broker-host-gate: HOST_ACTIVATION_PASS release=%s main_pid=%s resident_pid=%s resident_start_tick=%s resident_generation_sha256=%s resident_sequence=%s,%s lease_manifest_sha256=%s lease_authority_sha256=%s capsule_manifest_sha256=%s capsule_authority_sha256=%s invocation_manifest_sha256=%s invocation_authority_sha256=%s exec_grant_manifest_sha256=%s resident_manifest_sha256=%s resident_runtime_sha256=%s broker_sha256=%s socket_activation=verified service_cgroup=verified root_peer=verified resident_process=stable admission=DENY424 grant_admission=DENY424 resident_action_9030_attached=true decision_transport_material=true barrier_release=false launch=closed recycle=closed material_broker=false material_capsule=false material_invocation=false material_grant=false material_execution=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n' \
  "$(receipt_value "$RECEIPT" release_id)" "$MAIN_PID" "$RESIDENT_PID" "$RESIDENT_START_TICK" \
  "$RESIDENT_GENERATION" "$RESIDENT_SEQUENCE_ONE" "$RESIDENT_SEQUENCE_TWO" \
  "$MANIFEST_SHA256" "$AUTHORITY_SHA256" \
  "$CAPSULE_MANIFEST_SHA256" "$CAPSULE_AUTHORITY_SHA256" "$INVOCATION_MANIFEST_SHA256" \
  "$INVOCATION_AUTHORITY_SHA256" "$EXEC_GRANT_MANIFEST_SHA256" "$RESIDENT_MANIFEST_SHA256" \
  "$RESIDENT_RUNTIME_SHA256" "$BROKER_SHA256"
