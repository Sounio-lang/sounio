#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_EXEC_OPERATION_CATALOG_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_EXEC_OPERATION_CATALOG_ENGINE:-lean_single}"
MODULE="${SOUNIO_LOOM_EXEC_OPERATION_CATALOG_MODULE:-$ROOT_DIR/stdlib/coordination/loom_exec_operation_catalog_authority.sio}"
ENTRYPOINT="${SOUNIO_LOOM_EXEC_OPERATION_CATALOG_MAIN:-$ROOT_DIR/tools/loom/exec_operation_catalog_authority_main.sio}"
OUTPUT="${SOUNIO_LOOM_EXEC_OPERATION_CATALOG_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-operation-catalog}"

fail() {
  printf 'build-sounio-loom-exec-operation-catalog: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'Sounio authority module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'Sounio entrypoint is absent or linked'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-catalog.XXXXXX")"
trap 'rm -rf "$work"' EXIT
combined="$work/exec_operation_catalog.sio"
compiled="$work/sounio-loom-exec-operation-catalog"
sed -n '1,$p' "$MODULE" "$ENTRYPOINT" > "$combined"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$combined" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the native Sounio executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

probe="$(printf '0\n' | "$OUTPUT")"
[[ "$probe" == 'SOUNIO_EXEC_OPERATION_CATALOG_SELFTEST PASS cases=11' ]] ||
  fail "Sounio selftest diverged: $probe"
printf 'BUILT_EXEC_OPERATION_CATALOG path=%s language=Sounio role=SEMANTIC_AUTHORITY action=9035 engine=%s cases=11\n' \
  "$OUTPUT" "$ENGINE"
