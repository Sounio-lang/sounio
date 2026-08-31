#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MODULE:-$ROOT_DIR/stdlib/coordination/loom_native_hook_cutover_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MAIN:-$ROOT_DIR/tools/loom/native_hook_cutover_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-native-hook-cutover}"

fail() {
  printf 'build-sounio-loom-native-hook-cutover: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-cutover.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/native-hook-cutover.sio"
compiled="$work/sounio-loom-native-hook-cutover"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" >"$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_NATIVE_HOOK_CUTOVER_SELFTEST PASS cases=12' ]] ||
  fail "selftest diverged: $probe"
printf 'BUILT_NATIVE_HOOK_CUTOVER path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9045 engine=%s cases=12\n' \
  "$OUTPUT" "$ENGINE"
