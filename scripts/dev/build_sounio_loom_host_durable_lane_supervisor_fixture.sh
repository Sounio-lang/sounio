#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_HOST_DURABLE_LANE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_HOST_DURABLE_LANE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_HOST_DURABLE_LANE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_host_durable_lane_supervisor.sio}"
ENTRYPOINT="${SOUNIO_LOOM_HOST_DURABLE_LANE_MAIN:-$ROOT_DIR/tools/loom/host_durable_lane_supervisor_fixture_main.sio}"
OUTPUT="${SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-durable-lane-supervisor}"

fail() {
  printf 'build-sounio-loom-host-durable-lane-supervisor: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-durable-lane.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/host_durable_lane_supervisor.sio"
compiled="$work/sounio-loom-host-durable-lane-supervisor"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_HOST_DURABLE_LANE_SELFTEST PASS cases=10' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_HOST_DURABLE_LANE_SUPERVISOR path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9032 engine=%s cases=10\n' \
  "$OUTPUT" "$ENGINE"
