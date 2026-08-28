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
  [[ "$output" == *'HOST_GATE_UNAVAILABLE'* && "$output" == *'material_broker=false'* ]] ||
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
[[ "$(field "$FIRST" launch)" == closed && "$(field "$FIRST" recycle)" == closed ]] ||
  fail 'staging install opened a material operation'

RELEASE_ID="$(field "$FIRST" release)"
MANIFEST_SHA256="$(field "$FIRST" manifest_sha256)"
AUTHORITY_SHA256="$(field "$FIRST" authority_sha256)"
BROKER_SHA256="$(field "$FIRST" broker_sha256)"
BUNDLE_SHA256="$(field "$FIRST" bundle_sha256)"
RELEASE="$STAGE/usr/lib/sounio/loom/releases/$RELEASE_ID"
MANIFEST="$RELEASE/kernel_principal_lease_authority.freeze.v1"
AUTHORITY="$RELEASE/sounio-loom-kernel-principal-lease-authority-runtime"
BROKER="$RELEASE/loom-kernel-principal-broker"
RECEIPT="$RELEASE/install.receipt.v1"
BROKER_LINK="$STAGE/usr/libexec/sounio/loom-kernel-principal-broker"

[[ -d "$RELEASE" && ! -L "$RELEASE" && "$(stat -c '%a' "$RELEASE")" == 555 ]] ||
  fail 'immutable release directory layout or mode is wrong'
[[ -f "$MANIFEST" && "$(stat -c '%a' "$MANIFEST")" == 444 ]] || fail 'installed manifest mode is wrong'
[[ -x "$AUTHORITY" && "$(stat -c '%a' "$AUTHORITY")" == 555 ]] || fail 'installed authority mode is wrong'
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
[[ "$(sha256sum "$BROKER" | cut -d ' ' -f 1)" == "$BROKER_SHA256" ]] || fail 'installed broker hash differs'
[[ "$(receipt_value "$RECEIPT" semantic_producer)" == Sounio ]] || fail 'semantic producer receipt drifted'
[[ "$(receipt_value "$RECEIPT" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'semantic role receipt drifted'
[[ "$(receipt_value "$RECEIPT" material_producer)" == C++20 ]] || fail 'material producer receipt drifted'
[[ "$(receipt_value "$RECEIPT" material_role)" == MATERIAL_PARITY ]] || fail 'material role receipt drifted'
[[ "$(receipt_value "$RECEIPT" material_transitory)" == true ]] || fail 'transitory receipt marker is absent'
[[ "$(receipt_value "$RECEIPT" bundle_sha256)" == "$BUNDLE_SHA256" ]] || fail 'bundle receipt hash drifted'
[[ "$(sha256sum "$RELEASE/install_loom_kernel_principal_broker.sh" | cut -d ' ' -f 1)" == \
   "$(receipt_value "$RECEIPT" installer_sha256)" ]] || fail 'installer snapshot hash drifted'
[[ "$(receipt_value "$RECEIPT" material_broker)" == false ]] || fail 'receipt promoted material broker'

SECOND="$($INSTALLER --staging-root "$STAGE")"
[[ "$(field "$SECOND" release)" == "$RELEASE_ID" ]] || fail 'source-fresh reinstall changed release identity'
[[ "$(field "$SECOND" authority_sha256)" == "$AUTHORITY_SHA256" ]] || fail 'source-fresh authority rebuild was not deterministic'
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

protocol="$($BROKER --selftest-protocol)"
[[ "$protocol" == 'LOOM_KERNEL_PRINCIPAL_BROKER_PROTOCOL_SELFTEST PASS launch=closed recycle=closed unknown=denied' ]] ||
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

printf 'sounio-loom-kernel-principal-broker-install-selftest: PASS semantic_authority=Sounio operational_realization=C++20+Linux+systemd-bootstrap role=MATERIAL_PARITY transitory=true action=9027 release=%s manifest_sha256=%s authority_sha256=%s broker_sha256=%s bundle_sha256=%s staging_reinstall=deterministic release_tamper=refused mode_tamper=refused nonroot_probe=refused direct_host_install=refused direct_host_gate=unavailable sudo_host_install=%s sudo_host_gate=%s launch=closed recycle=closed host_activation=unavailable material_broker=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false\n' \
  "$RELEASE_ID" "$MANIFEST_SHA256" "$AUTHORITY_SHA256" "$BROKER_SHA256" "$BUNDLE_SHA256" "$sudo_install" "$sudo_gate"
