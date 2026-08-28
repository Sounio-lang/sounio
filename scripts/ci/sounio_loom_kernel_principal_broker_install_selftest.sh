#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
INSTALLER="$ROOT_DIR/scripts/dev/install_loom_kernel_principal_broker.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_broker_host_gate.sh"

fail() {
  printf 'sounio-loom-kernel-principal-broker-install-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local line="$1"
  local key="$2"
  local token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  return 1
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

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 70 ]] || fail "$label exited $status instead of fail-closed 70: $output"
  printf '%s\n' "$output"
}

run_unavailable() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 77 ]] || fail "$label returned $status instead of HOST_GATE_UNAVAILABLE: $output"
  [[ "$output" == *'HOST_GATE_UNAVAILABLE'* && "$output" == *'material_broker=false'* && \
     "$output" == *'material_capsule=false'* ]] ||
    fail "$label omitted the unavailable material boundary"
  printf '%s\n' "$output"
}

[[ -x "$INSTALLER" ]] || fail 'installer is missing or not executable'
[[ -x "$HOST_GATE" ]] || fail 'host gate is missing or not executable'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-principal-install-selftest.XXXXXX")"
cleanup() {
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
STAGE="$WORK/stage"
mkdir "$STAGE"

FIRST="$($INSTALLER --staging-root "$STAGE")"
[[ "$FIRST" == 'LOOM_KERNEL_PRINCIPAL_BROKER_INSTALL PASS '* ]] || fail "first staging install failed: $FIRST"
[[ "$(field "$FIRST" mode)" == STAGING_ONLY ]] || fail 'staging install claimed host mode'
[[ "$(field "$FIRST" activated)" == false ]] || fail 'staging install claimed activation'
[[ "$(field "$FIRST" material_broker)" == false ]] || fail 'staging install claimed a material broker'
[[ "$(field "$FIRST" material_capsule)" == false ]] || fail 'staging install claimed a material capsule'
[[ "$(field "$FIRST" material_invocation)" == false ]] || fail 'staging install claimed material invocation'
[[ "$(field "$FIRST" admission)" == decision-only ]] || fail 'staging install omitted decision-only admission'
[[ "$(field "$FIRST" launch)" == closed && "$(field "$FIRST" recycle)" == closed ]] ||
  fail 'staging install opened a material operation'

RELEASE_ID="$(field "$FIRST" release)"
MANIFEST_SHA256="$(field "$FIRST" lease_manifest_sha256)"
AUTHORITY_SHA256="$(field "$FIRST" lease_authority_sha256)"
CAPSULE_MANIFEST_SHA256="$(field "$FIRST" capsule_manifest_sha256)"
CAPSULE_AUTHORITY_SHA256="$(field "$FIRST" capsule_authority_sha256)"
INVOCATION_MANIFEST_SHA256="$(field "$FIRST" invocation_manifest_sha256)"
INVOCATION_AUTHORITY_SHA256="$(field "$FIRST" invocation_authority_sha256)"
BROKER_SHA256="$(field "$FIRST" broker_sha256)"
BUNDLE_SHA256="$(field "$FIRST" bundle_sha256)"
RELEASE="$STAGE/usr/lib/sounio/loom/releases/$RELEASE_ID"
MANIFEST="$RELEASE/kernel_principal_lease_authority.freeze.v1"
AUTHORITY="$RELEASE/sounio-loom-kernel-principal-lease-authority-runtime"
CAPSULE_MANIFEST="$RELEASE/kernel_principal_capsule_authority.freeze.v1"
CAPSULE_AUTHORITY="$RELEASE/sounio-loom-kernel-principal-capsule-authority-runtime"
INVOCATION_MANIFEST="$RELEASE/kernel_invocation_cell_authority.freeze.v1"
INVOCATION_AUTHORITY="$RELEASE/sounio-loom-kernel-invocation-cell-authority-runtime"
BROKER="$RELEASE/loom-kernel-principal-broker"
RECEIPT="$RELEASE/install.receipt.v1"
BROKER_LINK="$STAGE/usr/libexec/sounio/loom-kernel-principal-broker"

[[ -d "$RELEASE" && ! -L "$RELEASE" && "$(stat -c '%a' "$RELEASE")" == 555 ]] ||
  fail 'immutable release directory layout or mode is wrong'
[[ -f "$MANIFEST" && "$(stat -c '%a' "$MANIFEST")" == 444 ]] || fail 'installed manifest mode is wrong'
[[ -x "$AUTHORITY" && "$(stat -c '%a' "$AUTHORITY")" == 555 ]] || fail 'installed authority mode is wrong'
[[ -f "$CAPSULE_MANIFEST" && "$(stat -c '%a' "$CAPSULE_MANIFEST")" == 444 ]] ||
  fail 'installed capsule manifest mode is wrong'
[[ -x "$CAPSULE_AUTHORITY" && "$(stat -c '%a' "$CAPSULE_AUTHORITY")" == 555 ]] ||
  fail 'installed capsule authority mode is wrong'
[[ -f "$INVOCATION_MANIFEST" && "$(stat -c '%a' "$INVOCATION_MANIFEST")" == 444 ]] ||
  fail 'installed InvocationCell manifest mode is wrong'
[[ -x "$INVOCATION_AUTHORITY" && "$(stat -c '%a' "$INVOCATION_AUTHORITY")" == 555 ]] ||
  fail 'installed InvocationCell authority mode is wrong'
[[ -x "$BROKER" && "$(stat -c '%a' "$BROKER")" == 555 ]] || fail 'installed broker mode is wrong'
[[ -f "$RECEIPT" && "$(stat -c '%a' "$RECEIPT")" == 444 ]] || fail 'install receipt mode is wrong'
[[ -f "$RELEASE/install_loom_kernel_principal_broker.sh" && \
   "$(stat -c '%a' "$RELEASE/install_loom_kernel_principal_broker.sh")" == 444 ]] ||
  fail 'installed installer snapshot mode is wrong'
[[ -L "$BROKER_LINK" ]] || fail 'stable broker link is absent'
[[ "$(readlink "$BROKER_LINK")" == "/usr/lib/sounio/loom/releases/$RELEASE_ID/loom-kernel-principal-broker" ]] ||
  fail 'stable broker link escaped the immutable release'
[[ "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" == "$MANIFEST_SHA256" ]] || fail 'installed manifest hash differs'
[[ "$(sha256sum "$AUTHORITY" | cut -d ' ' -f 1)" == "$AUTHORITY_SHA256" ]] || fail 'installed authority hash differs'
[[ "$(sha256sum "$CAPSULE_MANIFEST" | cut -d ' ' -f 1)" == "$CAPSULE_MANIFEST_SHA256" ]] ||
  fail 'installed capsule manifest hash differs'
[[ "$(sha256sum "$CAPSULE_AUTHORITY" | cut -d ' ' -f 1)" == "$CAPSULE_AUTHORITY_SHA256" ]] ||
  fail 'installed capsule authority hash differs'
[[ "$(sha256sum "$INVOCATION_MANIFEST" | cut -d ' ' -f 1)" == "$INVOCATION_MANIFEST_SHA256" ]] ||
  fail 'installed InvocationCell manifest hash differs'
[[ "$(sha256sum "$INVOCATION_AUTHORITY" | cut -d ' ' -f 1)" == "$INVOCATION_AUTHORITY_SHA256" ]] ||
  fail 'installed InvocationCell authority hash differs'
[[ "$(sha256sum "$BROKER" | cut -d ' ' -f 1)" == "$BROKER_SHA256" ]] || fail 'installed broker hash differs'
[[ "$(receipt_value "$RECEIPT" semantic_producer)" == Sounio ]] || fail 'semantic producer receipt drifted'
[[ "$(receipt_value "$RECEIPT" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'semantic role receipt drifted'
[[ "$(receipt_value "$RECEIPT" semantic_actions)" == 9027+9028+9029 ]] || fail 'semantic action receipt drifted'
[[ "$(receipt_value "$RECEIPT" lease_manifest_sha256)" == "$MANIFEST_SHA256" ]] ||
  fail 'lease manifest receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" lease_authority_sha256)" == "$AUTHORITY_SHA256" ]] ||
  fail 'lease authority receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" capsule_manifest_sha256)" == "$CAPSULE_MANIFEST_SHA256" ]] ||
  fail 'capsule manifest receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" capsule_authority_sha256)" == "$CAPSULE_AUTHORITY_SHA256" ]] ||
  fail 'capsule authority receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" invocation_manifest_sha256)" == "$INVOCATION_MANIFEST_SHA256" ]] ||
  fail 'InvocationCell manifest receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" invocation_authority_sha256)" == "$INVOCATION_AUTHORITY_SHA256" ]] ||
  fail 'InvocationCell authority receipt hash drifted'
[[ "$(receipt_value "$RECEIPT" material_producer)" == C++20 ]] || fail 'material producer receipt drifted'
[[ "$(receipt_value "$RECEIPT" material_role)" == MATERIAL_PARITY ]] || fail 'material role receipt drifted'
[[ "$(receipt_value "$RECEIPT" material_transitory)" == true ]] || fail 'transitory receipt marker is absent'
[[ "$(receipt_value "$RECEIPT" bundle_sha256)" == "$BUNDLE_SHA256" ]] || fail 'bundle receipt hash drifted'
[[ "$(sha256sum "$RELEASE/install_loom_kernel_principal_broker.sh" | cut -d ' ' -f 1)" == \
   "$(receipt_value "$RECEIPT" installer_sha256)" ]] || fail 'installer snapshot hash drifted'
[[ "$(receipt_value "$RECEIPT" material_broker)" == false ]] || fail 'receipt promoted material broker'
[[ "$(receipt_value "$RECEIPT" material_capsule)" == false ]] || fail 'receipt promoted material capsule'
[[ "$(receipt_value "$RECEIPT" material_invocation)" == false ]] || fail 'receipt promoted material invocation'
[[ "$(receipt_value "$RECEIPT" admission_open)" == true ]] || fail 'receipt closed decision admission'

SECOND="$($INSTALLER --staging-root "$STAGE")"
[[ "$(field "$SECOND" release)" == "$RELEASE_ID" ]] || fail 'source-fresh reinstall changed release identity'
[[ "$(field "$SECOND" lease_authority_sha256)" == "$AUTHORITY_SHA256" ]] ||
  fail 'source-fresh authority rebuild was not deterministic'
[[ "$(field "$SECOND" capsule_authority_sha256)" == "$CAPSULE_AUTHORITY_SHA256" ]] ||
  fail 'source-fresh capsule authority rebuild was not deterministic'
[[ "$(field "$SECOND" invocation_authority_sha256)" == "$INVOCATION_AUTHORITY_SHA256" ]] ||
  fail 'source-fresh InvocationCell authority rebuild was not deterministic'
[[ "$(field "$SECOND" broker_sha256)" == "$BROKER_SHA256" ]] || fail 'source-fresh broker rebuild was not deterministic'

TAMPER_BYTES="$WORK/tamper-bytes"
cp -a "$STAGE" "$TAMPER_BYTES"
TAMPER_MANIFEST="$TAMPER_BYTES/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_principal_lease_authority.freeze.v1"
chmod 0644 "$TAMPER_MANIFEST"
printf X | dd of="$TAMPER_MANIFEST" bs=1 seek=0 conv=notrunc status=none
bytes_refusal="$(run_refusal release-byte-tamper "$INSTALLER" --staging-root "$TAMPER_BYTES")"
[[ "$bytes_refusal" == *'existing immutable release manifest drifted'* ]] ||
  fail 'one-byte release sabotage did not trigger the manifest rule'

TAMPER_MODE="$WORK/tamper-mode"
cp -a "$STAGE" "$TAMPER_MODE"
chmod 0755 "$TAMPER_MODE/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-principal-lease-authority-runtime"
mode_refusal="$(run_refusal release-mode-tamper "$INSTALLER" --staging-root "$TAMPER_MODE")"
[[ "$mode_refusal" == *'existing immutable release authority mode drifted'* ]] ||
  fail 'release-mode sabotage did not trigger the authority-mode rule'

TAMPER_CAPSULE_BYTES="$WORK/tamper-capsule-bytes"
cp -a "$STAGE" "$TAMPER_CAPSULE_BYTES"
TAMPER_CAPSULE_MANIFEST="$TAMPER_CAPSULE_BYTES/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_principal_capsule_authority.freeze.v1"
chmod 0644 "$TAMPER_CAPSULE_MANIFEST"
printf X | dd of="$TAMPER_CAPSULE_MANIFEST" bs=1 seek=0 conv=notrunc status=none
capsule_bytes_refusal="$(run_refusal capsule-release-byte-tamper "$INSTALLER" --staging-root "$TAMPER_CAPSULE_BYTES")"
[[ "$capsule_bytes_refusal" == *'existing immutable release capsule manifest drifted'* ]] ||
  fail 'one-byte capsule release sabotage did not trigger the capsule-manifest rule'

TAMPER_CAPSULE_MODE="$WORK/tamper-capsule-mode"
cp -a "$STAGE" "$TAMPER_CAPSULE_MODE"
chmod 0755 "$TAMPER_CAPSULE_MODE/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-principal-capsule-authority-runtime"
capsule_mode_refusal="$(run_refusal capsule-release-mode-tamper "$INSTALLER" --staging-root "$TAMPER_CAPSULE_MODE")"
[[ "$capsule_mode_refusal" == *'existing immutable release capsule authority mode drifted'* ]] ||
  fail 'capsule release-mode sabotage did not trigger the capsule-authority-mode rule'

TAMPER_INVOCATION_BYTES="$WORK/tamper-invocation-bytes"
cp -a "$STAGE" "$TAMPER_INVOCATION_BYTES"
TAMPER_INVOCATION_MANIFEST="$TAMPER_INVOCATION_BYTES/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_invocation_cell_authority.freeze.v1"
chmod 0644 "$TAMPER_INVOCATION_MANIFEST"
printf X | dd of="$TAMPER_INVOCATION_MANIFEST" bs=1 seek=0 conv=notrunc status=none
invocation_bytes_refusal="$(run_refusal invocation-release-byte-tamper "$INSTALLER" --staging-root "$TAMPER_INVOCATION_BYTES")"
[[ "$invocation_bytes_refusal" == *'existing immutable release InvocationCell manifest drifted'* ]] ||
  fail 'one-byte InvocationCell release sabotage did not trigger the manifest rule'

TAMPER_INVOCATION_MODE="$WORK/tamper-invocation-mode"
cp -a "$STAGE" "$TAMPER_INVOCATION_MODE"
chmod 0755 "$TAMPER_INVOCATION_MODE/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-invocation-cell-authority-runtime"
invocation_mode_refusal="$(run_refusal invocation-release-mode-tamper "$INSTALLER" --staging-root "$TAMPER_INVOCATION_MODE")"
[[ "$invocation_mode_refusal" == *'existing immutable release InvocationCell authority mode drifted'* ]] ||
  fail 'InvocationCell release-mode sabotage did not trigger the authority-mode rule'

protocol="$($BROKER --selftest-protocol)"
[[ "$protocol" == 'LOOM_KERNEL_PRINCIPAL_BROKER_PROTOCOL_SELFTEST PASS admission_without_context=denied malformed_admission=denied launch=closed recycle=closed unknown=denied partial_status=denied' ]] ||
  fail 'offline bootstrap protocol selftest failed'
probe_refusal="$(run_refusal nonroot-live-probe "$BROKER" --probe-live --socket-path "$WORK/absent.sock")"
[[ "$probe_refusal" == *'live broker probe requires root identity'* ]] ||
  fail 'non-root live probe did not trigger the identity rule'

host_refusal="$(run_refusal direct-host-install "$INSTALLER" --host-install)"
[[ "$host_refusal" == *'host install requires root identity'* ]] ||
  fail 'direct host install did not trigger the root rule'
run_unavailable direct-host-gate "$HOST_GATE" >/dev/null

sudo_install=not-available
sudo_gate=not-available
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  sudo_refusal="$(run_refusal sudo-host-install sudo -n "$INSTALLER" --host-install)"
  [[ "$sudo_refusal" == *'host install requires PID 1 systemd'* ]] ||
    fail 'sudo host install did not trigger the service-manager rule'
  run_unavailable sudo-host-gate sudo -n "$HOST_GATE" >/dev/null
  sudo_install=refused
  sudo_gate=unavailable
fi

ldd "$BROKER" > "$WORK/broker.ldd"
! grep -Eiq 'python|rust|cargo' "$WORK/broker.ldd" || fail 'broker gained a Python or Rust runtime dependency'

printf 'sounio-loom-kernel-principal-broker-install-selftest: PASS semantic_authority=Sounio operational_realization=C++20+Linux+systemd-bootstrap role=MATERIAL_PARITY transitory=true actions=9027+9028+9029 release=%s lease_manifest_sha256=%s lease_authority_sha256=%s capsule_manifest_sha256=%s capsule_authority_sha256=%s invocation_manifest_sha256=%s invocation_authority_sha256=%s broker_sha256=%s bundle_sha256=%s staging_reinstall=deterministic lease_release_tamper=refused lease_mode_tamper=refused capsule_release_tamper=refused capsule_mode_tamper=refused invocation_release_tamper=refused invocation_mode_tamper=refused nonroot_probe=refused direct_host_install=refused direct_host_gate=unavailable sudo_host_install=%s sudo_host_gate=%s admission=decision-only launch=closed recycle=closed host_activation=unavailable material_broker=false material_capsule=false material_invocation=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n' \
  "$RELEASE_ID" "$MANIFEST_SHA256" "$AUTHORITY_SHA256" "$CAPSULE_MANIFEST_SHA256" \
  "$CAPSULE_AUTHORITY_SHA256" "$INVOCATION_MANIFEST_SHA256" "$INVOCATION_AUTHORITY_SHA256" \
  "$BROKER_SHA256" "$BUNDLE_SHA256" "$sudo_install" "$sudo_gate"
