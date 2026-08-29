#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
ACTION_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT:-$ROOT_DIR/tools/loom/kernel_peer_activation_capsule.current.v1}"

fail() {
  printf 'build-sounio-loom-kernel-peer-activation-capsule-current-frame: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$ACTION_MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "action manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$ACTION_MANIFEST")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

[[ -f "$ACTION_MANIFEST" ]] || fail 'frozen action 9031 manifest is missing'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'action 9031 is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'action 9031 producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9031 role is not semantic authority'
[[ "$(field action)" == 9031 ]] || fail 'wrong semantic action'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-activation-current.XXXXXX")"
trap 'rm -rf "$work"' EXIT
reference="$work/action-9031"
bundle="$work/action-9031-fixture-bundle"
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$reference" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_authority.sh" >/dev/null
printf '1\n' | "$reference" > "$bundle"

[[ "$(file_hash "$bundle")" == "$(field fixture_bundle_sha256)" ]] ||
  fail 'Sounio-produced fixture bundle hash differs from the semantic freeze'
[[ "$(grep -c '^CASE label=current_material ' "$bundle" || true)" == 1 ]] ||
  fail 'current_material projection is not unique'
[[ "$(grep -c '^CASE label=seal ' "$bundle" || true)" == 1 ]] ||
  fail 'positive seal sabotage projection is not unique'

mkdir -p "$(dirname "$OUTPUT")"
install -m 0644 "$bundle" "$OUTPUT"

printf 'BUILT_KERNEL_PEER_ACTIVATION_CURRENT_PROJECTION path=%s producer=Sounio action=9031 bundle_sha256=%s current_labels=1 authorizing=false\n' \
  "$OUTPUT" "$(file_hash "$OUTPUT")"
