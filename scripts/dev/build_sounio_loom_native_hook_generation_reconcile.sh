#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_MODULE:-$ROOT_DIR/stdlib/coordination/loom_native_hook_generation_reconcile_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_MAIN:-$ROOT_DIR/tools/loom/native_hook_generation_reconcile_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-native-hook-generation-reconcile}"

fail() {
  printf 'build-sounio-loom-native-hook-generation-reconcile: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-generation-reconcile.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/native-hook-generation-reconcile.sio"
compiled="$work/sounio-loom-native-hook-generation-reconcile"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" >"$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_NATIVE_HOOK_GENERATION_RECONCILE_SELFTEST PASS cases=13' ]] ||
  fail "selftest diverged: $probe"
printf 'BUILT_NATIVE_HOOK_GENERATION_RECONCILE path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9047 engine=%s cases=13\n' \
  "$OUTPUT" "$ENGINE"
