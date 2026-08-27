#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_LANE_HEALTH_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_LANE_HEALTH_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_LANE_HEALTH_MODULE:-$ROOT_DIR/stdlib/coordination/loom_lane_health.sio}"
ENTRYPOINT="${SOUNIO_LOOM_LANE_HEALTH_MAIN:-$ROOT_DIR/tools/loom/lane_health_main.sio}"
OUTPUT="${SOUNIO_LOOM_LANE_HEALTH_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-lane-health-runtime}"

fail() {
  printf 'build-sounio-loom-lane-health: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "lane-health module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "lane-health entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-lane-health-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_lane_health_runtime.sio"
compiled="$work/sounio-loom-lane-health-runtime"

# Mechanical source assembly only. Classification rules and expected cases
# originate in the Sounio module.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native lane-health executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_LANE_HEALTH_SELFTEST PASS cases=28' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_LANE_HEALTH path=%s language=Sounio engine=%s cases=28\n' \
  "$OUTPUT" "$ENGINE"
