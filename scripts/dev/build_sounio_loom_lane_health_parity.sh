#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_LANE_HEALTH_PARITY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_LANE_HEALTH_PARITY_ENGINE:-lean_single}"
FREEZE="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
MODULE="$ROOT_DIR/stdlib/coordination/loom_lane_health.sio"
SHA256_MODULE="$ROOT_DIR/stdlib/crypto/sha256.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/lane_health_parity_main.sio"
OUTPUT="${SOUNIO_LOOM_LANE_HEALTH_PARITY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-lane-health-parity-runtime}"

fail() {
  printf 'build-sounio-loom-lane-health-parity: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$FREEZE" || true)"
  [[ "$count" == 1 ]] || fail "freeze field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$FREEZE")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$FREEZE" ]] || fail "lane-health freeze manifest is missing: $FREEZE"
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'lane-health semantics are not frozen'
[[ "$(field parity_open)" == false ]] || fail 'freeze manifest was already promoted'
[[ "$(file_hash "$MODULE")" == "$(field source_sha256)" ]] || fail 'frozen lane-health source drifted'
[[ -f "$SHA256_MODULE" ]] || fail "Sounio SHA-256 module is missing: $SHA256_MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "parity entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-lane-health-parity-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_lane_health_parity_runtime.sio"
compiled="$work/sounio-loom-lane-health-parity-runtime"

# This adapter measures the already-frozen Sounio function. It does not add
# classification rules or expected results.
sed -n '1,$p' "$MODULE" "$SHA256_MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native parity executable'
install -m 0755 "$compiled" "$OUTPUT"

printf 'BUILT_LANE_HEALTH_PARITY path=%s language=Sounio engine=%s domain=8388608 parent=%s\n' \
  "$OUTPUT" "$ENGINE" "$(field semantics_sha256)"
