#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal-broker.XXXXXX")"
AUTHORITY="$TEST_ROOT/kernel-principal-lease-authority"
CAPSULE_AUTHORITY="$TEST_ROOT/kernel-principal-capsule-authority"
INVOCATION_AUTHORITY="$TEST_ROOT/kernel-invocation-cell-authority"
RESIDENT_RUNTIME="$TEST_ROOT/resident-v4"
BROKER_ONE="$TEST_ROOT/kernel-principal-broker-one"
BROKER_TWO="$TEST_ROOT/kernel-principal-broker-two"
JOURNAL="$TEST_ROOT/leases.v1"
RECOVERY_JOURNAL="$TEST_ROOT/recovery-leases.v1"
COLLISION_JOURNAL="$TEST_ROOT/collision-leases.v1"
TAMPERED_JOURNAL="$TEST_ROOT/leases-tampered.v1"
TAMPERED_MANIFEST="$TEST_ROOT/manifest-tampered.v1"
TAMPERED_AUTHORITY="$TEST_ROOT/authority-tampered"
TAMPERED_CAPSULE_MANIFEST="$TEST_ROOT/capsule-manifest-tampered.v1"
TAMPERED_CAPSULE_AUTHORITY="$TEST_ROOT/capsule-authority-tampered"
TAMPERED_INVOCATION_MANIFEST="$TEST_ROOT/invocation-manifest-tampered.v1"
TAMPERED_INVOCATION_AUTHORITY="$TEST_ROOT/invocation-authority-tampered"
DIRECT_JOURNAL="$TEST_ROOT/direct-serve-must-not-exist.v1"
MISSING_CAPSULE_JOURNAL="$TEST_ROOT/missing-capsule-serve-must-not-exist.v1"
MISSING_INVOCATION_JOURNAL="$TEST_ROOT/missing-invocation-serve-must-not-exist.v1"
MISSING_RESIDENT_JOURNAL="$TEST_ROOT/missing-resident-serve-must-not-exist.v1"
SUDO_JOURNAL="$TEST_ROOT/sudo-serve-must-not-exist.v1"
MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_lease_authority.freeze.v1"
MANIFEST_SHA256='7bb5bbf30106d269644b0f9e6d80ee09f43eecf0e4a840bc3f429cfb6eca7cb5'
CAPSULE_MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_capsule_authority.freeze.v1"
CAPSULE_MANIFEST_SHA256='76ac860306c8cc00517f81f3fe2a4a2742a1cd4b9c4b4bb34b144b25fbcdf26f'
INVOCATION_MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
INVOCATION_MANIFEST_SHA256='61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e'
EXEC_GRANT_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
EXEC_GRANT_MANIFEST_SHA256='8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051'
RESIDENT_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v4"
RESIDENT_MANIFEST_SHA256='f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473'

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-broker-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

line_hash() {
  local sum
  sum="$(printf '%s\n' "$1" | sha256sum)"
  printf '%s' "${sum%% *}"
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "receipt omitted $key"
}

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ "$status" == 70 ]] || fail "$label exited $status: $output"
  [[ "$output" == 'loom-kernel-principal-broker: REFUSE reason='* ]] ||
    fail "$label omitted fail-closed refusal: $output"
  printf '%s' "$output"
}

[[ "$(sha256sum "$MANIFEST" | cut -d' ' -f1)" == "$MANIFEST_SHA256" ]] ||
  fail 'frozen action 9027 manifest drifted'
[[ "$(sha256sum "$CAPSULE_MANIFEST" | cut -d' ' -f1)" == "$CAPSULE_MANIFEST_SHA256" ]] ||
  fail 'frozen action 9028 manifest drifted'
[[ "$(sha256sum "$INVOCATION_MANIFEST" | cut -d' ' -f1)" == "$INVOCATION_MANIFEST_SHA256" ]] ||
  fail 'frozen action 9029 manifest drifted'
[[ "$(sha256sum "$EXEC_GRANT_MANIFEST" | cut -d' ' -f1)" == "$EXEC_GRANT_MANIFEST_SHA256" ]] ||
  fail 'frozen action 9030 manifest drifted'
[[ "$(sha256sum "$RESIDENT_MANIFEST" | cut -d' ' -f1)" == "$RESIDENT_MANIFEST_SHA256" ]] ||
  fail 'frozen resident v4 manifest drifted'
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_lease_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_capsule_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_exec_grant_cell_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_resident_transport_v4_freeze_selftest.sh" >/dev/null

SOUNIO_LOOM_KERNEL_PRINCIPAL_LEASE_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_lease_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_OUTPUT="$CAPSULE_AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_capsule_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$INVOCATION_AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$RESIDENT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
RESIDENT_SERVE_ARGS=(
  --exec-grant-manifest "$EXEC_GRANT_MANIFEST"
  --resident-manifest "$RESIDENT_MANIFEST"
  --resident-runtime "$RESIDENT_RUNTIME"
)
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_ONE" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_TWO" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two C++ broker-bootstrap builds differ'
[[ "$(stat -c '%a' "$BROKER_ONE")" == 755 ]] || fail 'broker binary mode is not 0755'
[[ ! -u "$BROKER_ONE" && ! -g "$BROKER_ONE" ]] || fail 'broker binary acquired set-id privilege'
dependencies="$(ldd "$BROKER_ONE")"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'broker has a forbidden Python or Rust runtime dependency'
fi

diagnostic="$($BROKER_ONE --diagnose --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --socket-path /run/sounio/loom-principal-broker.sock)"
receipt="$(printf '%s\n' "$diagnostic" | sed -n '1p')"
receipt_digest="$(printf '%s\n' "$diagnostic" | sed -n '2s/^LOOM_KERNEL_PRINCIPAL_BROKER_RECEIPT_SHA256 //p')"
decision="$(printf '%s\n' "$diagnostic" | sed -n '3p')"
[[ "$receipt" == 'LOOM_KERNEL_PRINCIPAL_BROKER_DIAGNOSTIC '* ]] ||
  fail 'diagnostic receipt is missing'
[[ "$receipt_digest" =~ ^[0-9a-f]{64}$ ]] || fail 'receipt digest is malformed'
[[ "$(line_hash "$receipt")" == "$receipt_digest" ]] || fail 'receipt digest differs'
[[ "$(field "$receipt" manifest_sha256)" == "$MANIFEST_SHA256" ]] ||
  fail 'diagnostic omitted frozen manifest binding'
[[ "$(field "$receipt" root_identity)" == 0 && \
   "$(field "$receipt" pid1_systemd)" == 0 && \
   "$(field "$receipt" parent_is_pid1)" == 0 && \
   "$(field "$receipt" service_cgroup)" == 0 && \
   "$(field "$receipt" listen_environment)" == 0 && \
   "$(field "$receipt" inherited_root_socket)" == 0 && \
   "$(field "$receipt" activation_complete)" == 0 && \
   "$(field "$receipt" material_broker)" == 0 ]] ||
  fail 'current pod was laundered as a host service boundary'
[[ "$decision" == \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=463 reason=host-broker-boundary-incomplete stage=SEMANTICS_FROZEN' ]] ||
  fail "Sounio did not refuse current broker material: $decision"

spoofed_diagnostic="$(SOUNIO_LOOM_BROKER_ASSUME_ROOT=1 \
  SOUNIO_LOOM_BROKER_ASSUME_SYSTEMD=1 LISTEN_PID=1 LISTEN_FDS=1 \
  "$BROKER_ONE" --diagnose --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --socket-path /run/sounio/loom-principal-broker.sock)"
spoofed_decision="$(printf '%s\n' "$spoofed_diagnostic" | sed -n '3p')"
[[ "$spoofed_decision" == "$decision" ]] || fail 'environment spoof changed Sounio decision'

protocol="$($BROKER_ONE --selftest-protocol)"
[[ "$protocol" == \
  'LOOM_KERNEL_PRINCIPAL_BROKER_PROTOCOL_SELFTEST PASS admission_without_context=denied malformed_admission=denied grant_admission_without_context=denied malformed_grant_admission=denied launch=closed recycle=closed unknown=denied partial_status=denied' ]] ||
  fail "bootstrap protocol opened unexpectedly: $protocol"

journal_result="$($BROKER_ONE --selftest-journal --journal "$JOURNAL")"
[[ "$journal_result" == 'LOOM_KERNEL_PRINCIPAL_BROKER_JOURNAL_SELFTEST PASS records=6 '* \
   && "$journal_result" == *' final_state=FREE fsync=per-record' ]] ||
  fail "journal selftest failed: $journal_result"
verify_result="$($BROKER_ONE --verify-journal --journal "$JOURNAL")"
[[ "$verify_result" == 'LOOM_KERNEL_PRINCIPAL_BROKER_JOURNAL_VERIFY PASS records=6 '* ]] ||
  fail "journal replay failed: $verify_result"
recovery_result="$($BROKER_ONE --selftest-recovery --journal "$RECOVERY_JOURNAL")"
[[ "$recovery_result" == 'LOOM_KERNEL_PRINCIPAL_BROKER_RECOVERY_SELFTEST PASS records=4 quarantined=1 final_state=QUARANTINED recovery_epoch=2 '* ]] ||
  fail "crash recovery did not force quarantine: $recovery_result"
collision_result="$($BROKER_ONE --selftest-collision --journal "$COLLISION_JOURNAL")"
[[ "$collision_result" == \
  'LOOM_KERNEL_PRINCIPAL_BROKER_COLLISION_SELFTEST PASS first=RESERVED second=REFUSED records=1' ]] ||
  fail "overlapping active range was admitted: $collision_result"

cp "$JOURNAL" "$TAMPERED_JOURNAL"
printf 'X' | dd of="$TAMPERED_JOURNAL" bs=1 seek=20 conv=notrunc status=none
journal_tamper_refusal="$(run_refusal journal-tamper "$BROKER_ONE" \
  --verify-journal --journal "$TAMPERED_JOURNAL")"
[[ "$journal_tamper_refusal" == *'lease-journal'* ]] ||
  fail 'journal mutation was not classified as journal refusal'
duplicate_refusal="$(run_refusal duplicate-journal "$BROKER_ONE" \
  --selftest-journal --journal "$JOURNAL")"
[[ "$duplicate_refusal" == *'cannot open lease journal'* ]] ||
  fail 'selftest reopened an existing journal'

cp "$MANIFEST" "$TAMPERED_MANIFEST"
printf '\n' >> "$TAMPERED_MANIFEST"
manifest_refusal="$(run_refusal manifest-tamper "$BROKER_ONE" --diagnose \
  --manifest "$TAMPERED_MANIFEST" --authority "$AUTHORITY")"
[[ "$manifest_refusal" == *'manifest hash mismatch'* ]] ||
  fail 'manifest mutation reached Sounio execution'

cp "$AUTHORITY" "$TAMPERED_AUTHORITY"
printf 'X' >> "$TAMPERED_AUTHORITY"
authority_refusal="$(run_refusal authority-tamper "$BROKER_ONE" --diagnose \
  --manifest "$MANIFEST" --authority "$TAMPERED_AUTHORITY")"
[[ "$authority_refusal" == *'authority executable hash mismatch'* ]] ||
  fail 'authority mutation reached Sounio execution'

cp "$CAPSULE_MANIFEST" "$TAMPERED_CAPSULE_MANIFEST"
printf '\n' >> "$TAMPERED_CAPSULE_MANIFEST"
capsule_manifest_refusal="$(run_refusal capsule-manifest-tamper "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$TAMPERED_CAPSULE_MANIFEST" \
  --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" --invocation-authority "$INVOCATION_AUTHORITY" \
  "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$DIRECT_JOURNAL")"
[[ "$capsule_manifest_refusal" == *'action 9028 manifest hash mismatch'* ]] ||
  fail 'capsule manifest mutation reached activation measurement'

cp "$CAPSULE_AUTHORITY" "$TAMPERED_CAPSULE_AUTHORITY"
printf 'X' >> "$TAMPERED_CAPSULE_AUTHORITY"
capsule_authority_refusal="$(run_refusal capsule-authority-tamper "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" \
  --capsule-authority "$TAMPERED_CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" --invocation-authority "$INVOCATION_AUTHORITY" \
  "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$DIRECT_JOURNAL")"
[[ "$capsule_authority_refusal" == *'action 9028 authority executable hash mismatch'* ]] ||
  fail 'capsule authority mutation reached activation measurement'

missing_capsule_refusal="$(run_refusal capsule-absent "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" --journal "$MISSING_CAPSULE_JOURNAL")"
[[ "$missing_capsule_refusal" == *'capsule manifest and authority are required'* ]] ||
  fail 'serve without capsule authority did not trigger the dual-authority rule'
[[ ! -e "$MISSING_CAPSULE_JOURNAL" ]] ||
  fail 'serve without capsule authority created a journal'

missing_invocation_refusal="$(run_refusal invocation-absent "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --journal "$MISSING_INVOCATION_JOURNAL")"
[[ "$missing_invocation_refusal" == *'InvocationCell manifest and authority are required'* ]] ||
  fail 'serve without InvocationCell authority did not trigger the triple-authority rule'
[[ ! -e "$MISSING_INVOCATION_JOURNAL" ]] ||
  fail 'serve without InvocationCell authority created a journal'

missing_resident_refusal="$(run_refusal resident-absent "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" --invocation-authority "$INVOCATION_AUTHORITY" \
  --journal "$MISSING_RESIDENT_JOURNAL")"
[[ "$missing_resident_refusal" == *'ExecGrantCell manifest, resident manifest, and resident runtime are required'* ]] ||
  fail 'serve without resident action 9030 did not fail closed'
[[ ! -e "$MISSING_RESIDENT_JOURNAL" ]] ||
  fail 'serve without resident action 9030 created a journal'

cp "$INVOCATION_MANIFEST" "$TAMPERED_INVOCATION_MANIFEST"
printf '\n' >> "$TAMPERED_INVOCATION_MANIFEST"
invocation_manifest_refusal="$(run_refusal invocation-manifest-tamper "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$TAMPERED_INVOCATION_MANIFEST" \
  --invocation-authority "$INVOCATION_AUTHORITY" "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$DIRECT_JOURNAL")"
[[ "$invocation_manifest_refusal" == *'action 9029 manifest hash mismatch'* ]] ||
  fail 'InvocationCell manifest mutation reached activation measurement'

cp "$INVOCATION_AUTHORITY" "$TAMPERED_INVOCATION_AUTHORITY"
printf 'X' >> "$TAMPERED_INVOCATION_AUTHORITY"
invocation_authority_refusal="$(run_refusal invocation-authority-tamper "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" \
  --invocation-authority "$TAMPERED_INVOCATION_AUTHORITY" "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$DIRECT_JOURNAL")"
[[ "$invocation_authority_refusal" == *'action 9029 authority executable hash mismatch'* ]] ||
  fail 'InvocationCell authority mutation reached activation measurement'

direct_refusal="$(run_refusal direct-serve "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" --invocation-authority "$INVOCATION_AUTHORITY" \
  "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$DIRECT_JOURNAL")"
[[ "$direct_refusal" == *'service-manager activation boundary incomplete'* ]] ||
  fail 'direct non-root serve reached journal access'
[[ ! -e "$DIRECT_JOURNAL" ]] || fail 'direct non-root serve created a journal'

command -v sudo >/dev/null 2>&1 || fail 'sudo is missing from the current blocker witness'
sudo -n /usr/bin/true >/dev/null 2>&1 || fail 'outer privilege-regain blocker disappeared'
sudo_refusal="$(run_refusal sudo-serve sudo -n "$BROKER_ONE" --serve \
  --manifest "$MANIFEST" --authority "$AUTHORITY" \
  --capsule-manifest "$CAPSULE_MANIFEST" --capsule-authority "$CAPSULE_AUTHORITY" \
  --invocation-manifest "$INVOCATION_MANIFEST" --invocation-authority "$INVOCATION_AUTHORITY" \
  "${RESIDENT_SERVE_ARGS[@]}" \
  --journal "$SUDO_JOURNAL")"
[[ "$sudo_refusal" == *'service-manager activation boundary incomplete'* ]] ||
  fail 'sudo-launched root broker bypassed service-manager activation'
[[ ! -e "$SUDO_JOURNAL" ]] || fail 'sudo-launched broker created a journal'

SOCKET_UNIT="$ROOT_DIR/tools/loom/systemd/sounio-loom-principal-broker.socket"
SERVICE_UNIT="$ROOT_DIR/tools/loom/systemd/sounio-loom-principal-broker.service"
CONFIG_EXAMPLE="$ROOT_DIR/tools/loom/systemd/sounio-loom-principal-broker.conf.example"
grep -Fqx 'SocketUser=root' "$SOCKET_UNIT" || fail 'socket unit is not root-owned'
grep -Fqx 'SocketGroup=root' "$SOCKET_UNIT" || fail 'socket unit group is not root'
grep -Fqx 'SocketMode=0600' "$SOCKET_UNIT" || fail 'socket mode is not 0600'
grep -Fqx 'Accept=no' "$SOCKET_UNIT" || fail 'socket activation does not pass one listener'
grep -Fqx 'User=root' "$SERVICE_UNIT" || fail 'service does not run in host root boundary'
grep -Fqx 'Group=root' "$SERVICE_UNIT" || fail 'service group is not root'
grep -Fqx 'NoNewPrivileges=yes' "$SERVICE_UNIT" || fail 'service permits privilege escalation'
grep -Fq 'ExecStart=/usr/libexec/sounio/loom-kernel-principal-broker --serve ' "$SERVICE_UNIT" ||
  fail 'service does not execute the fixed broker directly'
if grep -Eq 'ExecStart=.*/(sh|bash|zsh|python|node|ruby)( |$)' "$SERVICE_UNIT"; then
  fail 'service uses a disposable-language launcher'
fi
grep -Fqx 'LOOM_PRINCIPAL_MANIFEST=/usr/lib/sounio/loom/kernel_principal_lease_authority.freeze.v1' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen manifest path'
grep -Fqx 'LOOM_PRINCIPAL_CAPSULE_MANIFEST=/usr/lib/sounio/loom/kernel_principal_capsule_authority.freeze.v1' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen capsule manifest path'
grep -Fqx 'LOOM_PRINCIPAL_INVOCATION_MANIFEST=/usr/lib/sounio/loom/kernel_invocation_cell_authority.freeze.v1' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen InvocationCell manifest path'
grep -Fqx 'LOOM_PRINCIPAL_EXEC_GRANT_MANIFEST=/usr/lib/sounio/loom/kernel_exec_grant_cell_authority.freeze.v1' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen ExecGrantCell manifest path'
grep -Fqx 'LOOM_PRINCIPAL_RESIDENT_MANIFEST=/usr/lib/sounio/loom/resident_membrane.runtime.v4' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen resident v4 manifest path'
grep -Fqx 'LOOM_PRINCIPAL_RESIDENT_RUNTIME=/usr/libexec/sounio/sounio-loom-resident-membrane-runtime-v4' "$CONFIG_EXAMPLE" ||
  fail 'config example omits frozen resident v4 runtime path'
grep -Fq -- '--invocation-manifest ${LOOM_PRINCIPAL_INVOCATION_MANIFEST} --invocation-authority ${LOOM_PRINCIPAL_INVOCATION_AUTHORITY}' "$SERVICE_UNIT" ||
  fail 'service does not pin the InvocationCell authority'
grep -Fq -- '--exec-grant-manifest ${LOOM_PRINCIPAL_EXEC_GRANT_MANIFEST} --resident-manifest ${LOOM_PRINCIPAL_RESIDENT_MANIFEST} --resident-runtime ${LOOM_PRINCIPAL_RESIDENT_RUNTIME}' "$SERVICE_UNIT" ||
  fail 'service does not pin resident action 9030'

command -v systemd-analyze >/dev/null 2>&1 || fail 'systemd-analyze is required for unit verification'
UNIT_ROOT="$TEST_ROOT/units"
mkdir -p "$UNIT_ROOT"
cp "$SOCKET_UNIT" "$UNIT_ROOT/sounio-loom-principal-broker.socket"
sed "s#/usr/libexec/sounio/loom-kernel-principal-broker#$BROKER_ONE#" \
  "$SERVICE_UNIT" > "$UNIT_ROOT/sounio-loom-principal-broker.service"
systemd-analyze verify "$UNIT_ROOT/sounio-loom-principal-broker.socket" \
  "$UNIT_ROOT/sounio-loom-principal-broker.service" >/dev/null

broker_source_sha256="$(sha256sum "$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp" | cut -d' ' -f1)"
broker_binary_sha256="$(sha256sum "$BROKER_ONE" | cut -d' ' -f1)"
printf '%s\n' \
  "sounio-loom-kernel-principal-broker-selftest: PASS semantic_authority=Sounio operational_realization=C++20+Linux+systemd-bootstrap role=MATERIAL_PARITY transitory=true actions=9027+9028+9029+9030 lease_manifest_sha256=$MANIFEST_SHA256 capsule_manifest_sha256=$CAPSULE_MANIFEST_SHA256 invocation_manifest_sha256=$INVOCATION_MANIFEST_SHA256 exec_grant_manifest_sha256=$EXEC_GRANT_MANIFEST_SHA256 resident_manifest_sha256=$RESIDENT_MANIFEST_SHA256 current_material=DENY463 direct_nonroot=refused sudo_root=refused environment_spoof=DENY463 manifest_tamper=refused authority_tamper=refused capsule_manifest_tamper=refused capsule_authority_tamper=refused invocation_manifest_tamper=refused invocation_authority_tamper=refused capsule_absent=refused invocation_absent=refused resident_absent=refused admission=decision-only grant_admission=resident-decision-only partial_status=denied journal_records=6 journal_tamper=refused journal_fsync=per-record crash_recovery=QUARANTINED range_collision=refused launch=closed recycle=closed systemd_unit=verified source_sha256=$broker_source_sha256 binary_sha256=$broker_binary_sha256 material_broker=false material_capsule=false material_invocation=false material_grant=false material_execution=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false"
