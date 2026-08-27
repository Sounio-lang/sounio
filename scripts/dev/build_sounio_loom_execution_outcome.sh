#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EXECUTION_OUTCOME_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EXECUTION_OUTCOME_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EXECUTION_OUTCOME_MODULE:-$ROOT_DIR/stdlib/coordination/loom_execution_outcome_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EXECUTION_OUTCOME_MAIN:-$ROOT_DIR/tools/loom/execution_outcome_main.sio}"
OUTPUT="${SOUNIO_LOOM_EXECUTION_OUTCOME_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-execution-outcome-runtime}"

fail() {
  printf 'build-sounio-loom-execution-outcome: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" ]] || fail "execution-outcome module is missing: $MODULE"
[[ -f "$ENTRYPOINT" ]] || fail "execution-outcome entrypoint is missing: $ENTRYPOINT"
mkdir -p "$(dirname "$OUTPUT")"

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-execution-outcome-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/loom_execution_outcome_runtime.sio"
compiled="$work/sounio-loom-execution-outcome-runtime"

# Mechanical source assembly only. Decisions and expected results live in the
# Sounio module, never in this launcher.
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native execution-outcome executable'
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EXECUTION_OUTCOME_SELFTEST PASS cases=28' ]] ||
  fail "Sounio-owned expected-result suite failed: $probe"

printf 'BUILT_EXECUTION_OUTCOME path=%s language=Sounio engine=%s cases=28\n' \
  "$OUTPUT" "$ENGINE"
