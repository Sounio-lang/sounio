#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/product_exec_cell_fixture.freeze.v1"

fail() {
  printf 'sounio-loom-product-exec-cell-fixture-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

expect_value() {
  local key="$1" expected="$2" actual
  actual="$(manifest_value "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key expected $expected but found $actual"
}

expect_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "$path is absent or linked"
  actual="$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] || fail "$path hash drifted"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
expect_value schema loom-product-exec-cell-fixture-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_FIXTURE_PRODUCER
expect_value semantic_authority Sounio
expect_value action 9030
expect_value source_precedes_material true
expect_value fixture_count 4
expect_value command_mismatch_result DENY492
expect_value causal_sabotage PASS
expect_value material_grant false
expect_value material_execution false
expect_value exec_cell_attached false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in source garden authority_manifest payload_manifest product_lane_cell_manifest build_script selftest freeze_selftest evidence; do
  expect_hash "$(manifest_value "${key}_path")" "$(manifest_value "${key}_sha256")"
done
expect_hash "$(manifest_value toolchain_wrapper_path)" "$(manifest_value toolchain_wrapper_sha256)"
expect_hash "$(manifest_value toolchain_compiler_path)" "$(manifest_value toolchain_compiler_sha256)"

source_commit="$(manifest_value source_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" ||
  fail 'source commit is unavailable'
source_at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$(manifest_value source_path)" | sha256sum | cut -d ' ' -f 1)"
[[ "$source_at_commit" == "$(manifest_value source_sha256)" ]] ||
  fail 'source commit does not bind the frozen Sounio source'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-product-exec-cell-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_OUTPUT="$work/fixture-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_product_exec_cell_fixture.sh" \
      >/dev/null
  "$work/fixture-$ordinal" > "$work/bundle-$ordinal"
done
cmp "$work/fixture-one" "$work/fixture-two" || fail 'fixture rebuild is nondeterministic'
cmp "$work/bundle-one" "$work/bundle-two" || fail 'fixture output is nondeterministic'
[[ "$(sha256sum "$work/fixture-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'fixture executable hash drifted'
[[ "$(sha256sum "$work/bundle-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value bundle_sha256)" ]] || fail 'fixture bundle hash drifted'
[[ "$(wc -l < "$work/bundle-one")" == "$(manifest_value bundle_lines)" ]] ||
  fail 'fixture bundle line count drifted'

command="$(sed -n 's/^COMMAND //p' "$work/bundle-one")"
[[ "$command" == "$(manifest_value command)" ]] || fail 'Sounio command intent drifted'
[[ "$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value command_sha256)" ]] || fail 'Sounio command hash drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_product_exec_cell_fixture_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'fixture selftest result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'fixture selftest result hash drifted'

printf 'sounio-loom-product-exec-cell-fixture-freeze-selftest: PASS semantic_authority=Sounio action=9030 stage=SEMANTICS_FROZEN source_sha256=%s executable_sha256=%s bundle_sha256=%s command_sha256=%s intent_sha256=%s command_mismatch=DENY492 causal_sabotage=PASS source_precedes_material=true python_executed=false rust_executed=false material_grant=false material_execution=false exec_cell_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value source_sha256)" "$(manifest_value executable_sha256)" \
  "$(manifest_value bundle_sha256)" "$(manifest_value command_sha256)" \
  "$(manifest_value intent_sha256)"
