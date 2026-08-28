#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-exec-quorum-fixture.XXXXXX")"
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
  printf 'sounio-loom-host-exec-quorum-fixture-selftest: FAIL: %s\n' "$*" >&2
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
  local label="$1" prefix="FIXTURE $1 " count line
  count="$(grep -c "^FIXTURE ${label} " "$BUNDLE" || true)"
  [[ "$count" == 1 ]] || fail "fixture $label occurs $count times"
  line="$(grep -m1 "^FIXTURE ${label} " "$BUNDLE")"
  printf '%s' "${line#"$prefix"}"
}

for output in "$FIXTURE_ONE" "$FIXTURE_TWO"; do
  SOUNIO_LOOM_HOST_EXEC_QUORUM_FIXTURE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_exec_quorum_fixture.sh" \
      >/dev/null
done
cmp "$FIXTURE_ONE" "$FIXTURE_TWO" || fail 'two Sounio fixture builds differ'
[[ "$(stat -c '%a' "$FIXTURE_ONE")" == 755 && ! -u "$FIXTURE_ONE" &&
   ! -g "$FIXTURE_ONE" ]] || fail 'fixture executable mode is unsafe'

SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" \
    >/dev/null

"$FIXTURE_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 6 ]] || fail 'fixture bundle line count diverged'
[[ "$(sed -n '1p' "$BUNDLE")" == \
  'LOOM_HOST_EXEC_QUORUM_FIXTURE_V1 semantic_authority=Sounio action=9030 fixtures=5' ]] ||
  fail 'fixture bundle metadata diverged'

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
check_fixture current current_decision
check_fixture python python_decision

python_sentinel="$TEST_ROOT/python3"
python_executed="$TEST_ROOT/python-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$python_executed" > "$python_sentinel"
chmod 0755 "$python_sentinel"
python_actual="$(printf '%s\n' "$(frame_for python)" | "$AUTHORITY" || true)"
if [[ "$python_actual" == SOUNIO_KERNEL_EXEC_GRANT_CELL_ALLOW* ]]; then
  "$python_sentinel"
fi
[[ ! -e "$python_executed" ]] || fail 'Python fixture crossed the Sounio refusal'

dependencies="$(ldd "$FIXTURE_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'fixture executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-host-exec-quorum-fixture-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER action=9030 fixtures=5 positive=issue+consume+close treatment=current python_control=refused python_executed=false source_sha256=%s executable_sha256=%s bundle_sha256=%s deterministic=true shell_expected_results=false runtime_dependencies=clean material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/host_exec_quorum_fixture_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$FIXTURE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
