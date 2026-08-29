#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-product-exec-cell-fixture.XXXXXX")"
FIXTURE_ONE="$TEST_ROOT/fixture-one"
FIXTURE_TWO="$TEST_ROOT/fixture-two"
AUTHORITY="$TEST_ROOT/action-9030-authority"
BUNDLE="$TEST_ROOT/fixtures.v1"
MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-product-exec-cell-fixture-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "authority field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

frame_for() {
  local label="$1" prefix="FRAME $1 " count line parents key digest
  count="$(grep -c "^FRAME ${label} " "$BUNDLE" || true)"
  [[ "$count" == 1 ]] || fail "fixture $label occurs $count times"
  line="$(grep -m1 "^FRAME ${label} " "$BUNDLE")"
  parents="$(grep -m1 '^PARENT_BINDINGS ' "$BUNDLE")"
  printf '%s %s' "${line#"$prefix"}" "${parents#PARENT_BINDINGS }"
  for key in grant_identity command_environment peer_vector transition_journal source_semantics_toolchain result_receipt; do
    digest="$(sed -n "s/^BINDING ${key} //p" "$BUNDLE")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "binding hash $key is malformed"
    hex_u32 "$digest"
  done
}

hex_u32() {
  local digest="$1" offset
  for ((offset = 0; offset < 64; offset += 8)); do
    printf ' %u' "$((16#${digest:offset:8}))"
  done
}

for output in "$FIXTURE_ONE" "$FIXTURE_TWO"; do
  SOUNIO_LOOM_PRODUCT_EXEC_CELL_FIXTURE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_product_exec_cell_fixture.sh" \
      >/dev/null
done
cmp "$FIXTURE_ONE" "$FIXTURE_TWO" || fail 'two Sounio fixture builds differ'
[[ "$(stat -c '%a' "$FIXTURE_ONE")" == 755 && ! -u "$FIXTURE_ONE" &&
   ! -g "$FIXTURE_ONE" ]] || fail 'fixture executable mode is unsafe'

SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" \
    >/dev/null

"$FIXTURE_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 14 ]] || fail 'fixture bundle line count diverged'
[[ "$(sed -n '1p' "$BUNDLE")" == \
  'LOOM_PRODUCT_EXEC_CELL_FIXTURE_V1 semantic_authority=Sounio action=9030 fixtures=4' ]] ||
  fail 'fixture bundle metadata diverged'

intent="$(sed -n '2p' "$BUNDLE")"
command="$(sed -n 's/^COMMAND //p' "$BUNDLE")"
command_sha256="$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)"
[[ "$intent" == "INTENT command_sha256=$command_sha256 "* &&
   "$intent" == *' intent_sha256=9f809fbd089f8aa0cf309e9f3ca166033f84bafbf9c87196fe9b00d3c8bcfc74' &&
   "$command" == 'loom-exec-cell-v1 calibration-9030 payload=7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d' ]] ||
  fail 'Sounio execution intent binding diverged'

check_fixture() {
  local label="$1" expected_field="$2" frame actual expected
  frame="$(frame_for "$label")"
  [[ "$frame" == '9030 3 1 '* && "$frame" != *$'\n'* &&
     ${#frame} -le 65535 ]] || fail "fixture $label has an invalid transport shape"
  actual="$(printf '%s\n' "$frame" | "$AUTHORITY" || true)"
  expected="$(manifest_value "$expected_field")"
  [[ "$actual" == "$expected" ]] ||
    fail "Sounio authority disagreed with fixture $label: $actual"
}

check_fixture issue issue_decision
check_fixture consume consume_decision
check_fixture close close_decision

identity_case="$(manifest_value result)"
[[ "$identity_case" == *' identity=DENY492 '* ]] ||
  fail 'frozen Sounio authority omitted the identity control'
mismatch_actual="$(printf '%s\n' "$(frame_for command_mismatch)" | "$AUTHORITY" || true)"
[[ "$mismatch_actual" == 'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=492 '* ]] ||
  fail "Sounio command mismatch control diverged: $mismatch_actual"

python_sentinel="$TEST_ROOT/python3"
rust_sentinel="$TEST_ROOT/rustc"
oracle_executed="$TEST_ROOT/oracle-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$python_sentinel"
cp "$python_sentinel" "$rust_sentinel"
chmod 0755 "$python_sentinel" "$rust_sentinel"
[[ ! -e "$oracle_executed" ]] || fail 'prohibited oracle executed'

dependencies="$(ldd "$FIXTURE_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'fixture executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-product-exec-cell-fixture-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER action=9030 fixtures=4 treatment=issue+consume+close command_mismatch=DENY492 causal_sabotage=PASS source_precedes_material=true command_sha256=%s intent_sha256=9f809fbd089f8aa0cf309e9f3ca166033f84bafbf9c87196fe9b00d3c8bcfc74 source_sha256=%s executable_sha256=%s bundle_sha256=%s deterministic=true shell_expected_results=false python_executed=false rust_executed=false runtime_dependencies=clean material_grant=false material_execution=false exec_cell_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$command_sha256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/product_exec_cell_fixture_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$FIXTURE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
