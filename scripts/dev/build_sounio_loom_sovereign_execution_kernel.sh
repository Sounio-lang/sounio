#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_SOVEREIGN_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_SOVEREIGN_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_SOVEREIGN_MODULE:-$ROOT_DIR/stdlib/coordination/loom_sovereign_execution_kernel_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_SOVEREIGN_MAIN:-$ROOT_DIR/tools/loom/sovereign_execution_kernel_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_SOVEREIGN_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-sovereign-execution-kernel}"

fail() {
  printf 'build-sounio-loom-sovereign-execution-kernel: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-sovereign.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/sovereign.sio"
compiled="$work/sounio-loom-sovereign-execution-kernel"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_SOVEREIGN_EXECUTION_KERNEL_SELFTEST PASS cases=14' ]] ||
  fail "selftest diverged: $probe"
printf 'BUILT_SOVEREIGN_EXECUTION_KERNEL path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9042 engine=%s cases=14\n' \
  "$OUTPUT" "$ENGINE"
