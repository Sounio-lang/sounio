#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EXEC_RESULT_HANDLE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EXEC_RESULT_HANDLE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EXEC_RESULT_HANDLE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_exec_result_handle_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EXEC_RESULT_HANDLE_MAIN:-$ROOT_DIR/tools/loom/exec_result_handle_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-result-handle}"

fail() {
  printf 'build-sounio-loom-exec-result-handle: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-handle.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/exec_result_handle.sio"
compiled="$work/sounio-loom-exec-result-handle"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EXEC_RESULT_HANDLE_SELFTEST PASS cases=16' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_EXEC_RESULT_HANDLE path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9033 engine=%s cases=16\n' \
  "$OUTPUT" "$ENGINE"
