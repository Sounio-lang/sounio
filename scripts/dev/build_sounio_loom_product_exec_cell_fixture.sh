#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/product_exec_cell_fixture_main.sio"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-product-exec-cell-fixture}"

fail() {
  printf 'build-sounio-loom-product-exec-cell-fixture: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$AUTHORITY_MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "authority field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$AUTHORITY_MANIFEST")"
  printf '%s' "${line#*=}"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'fixture source is absent or linked'
[[ -f "$AUTHORITY_MANIFEST" && ! -L "$AUTHORITY_MANIFEST" ]] ||
  fail 'frozen action 9030 manifest is absent or linked'
[[ "$(sha256sum "$AUTHORITY_MANIFEST" | cut -d ' ' -f 1)" == \
  8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 ]] ||
  fail 'frozen action 9030 manifest hash drifted'
[[ "$(manifest_value producing_language)" == Sounio &&
   "$(manifest_value language_role)" == SEMANTIC_AUTHORITY &&
   "$(manifest_value action)" == 9030 &&
   "$(manifest_value stage)" == SEMANTICS_FROZEN ]] ||
  fail 'action 9030 authority contract drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-product-exec-cell-fixture.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-product-exec-cell-fixture"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio fixture executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PRODUCT_EXEC_CELL_FIXTURE_V1 semantic_authority=Sounio action=9030 fixtures=4' ]] ||
  fail "fixture metadata diverged: $metadata"

printf 'BUILT_PRODUCT_EXEC_CELL_FIXTURE path=%s language=Sounio role=SEMANTIC_FIXTURE_PRODUCER authority_manifest_sha256=8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 engine=%s fixtures=4\n' \
  "$OUTPUT" "$ENGINE"
