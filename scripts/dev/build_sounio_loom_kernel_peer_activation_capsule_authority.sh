#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_MAIN:-$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-kernel-peer-activation-capsule-authority-runtime}"

fail() {
  printf 'build-sounio-loom-kernel-peer-activation-capsule-authority: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail "action 9031 module is missing or linked: $MODULE"
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail "action 9031 entrypoint is missing or linked: $ENTRYPOINT"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-activation-capsule.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_kernel_peer_activation_capsule_authority_runtime.sio"
compiled="$work/sounio-loom-kernel-peer-activation-capsule-authority-runtime"

# Mechanical assembly only. Sounio owns every decision and expected result.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the action 9031 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

selftest="$(printf '0\n' | "$OUTPUT")"
[[ "$selftest" == 'SOUNIO_KERNEL_PEER_ACTIVATION_CAPSULE_SELFTEST PASS cases=13' ]] ||
  fail "Sounio-owned action 9031 selftest failed: $selftest"
fixtures="$(printf '1\n' | "$OUTPUT")"
metadata="$(printf '%s\n' "$fixtures" | sed -n '1p')"
[[ "$metadata" == 'LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_FIXTURES_V1 producer=Sounio role=SEMANTIC_AUTHORITY action=9031 cases=16' ]] ||
  fail "Sounio-owned fixture metadata drifted: $metadata"
[[ "$(printf '%s\n' "$fixtures" | grep -c '^CASE ')" == 16 ]] ||
  fail 'Sounio-owned fixture count drifted'

printf 'BUILT_KERNEL_PEER_ACTIVATION_CAPSULE_AUTHORITY path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9031 engine=%s selftest_cases=13 fixture_cases=16\n' \
  "$OUTPUT" "$ENGINE"
