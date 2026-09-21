#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-material-change-selftest.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-sovereign-material-change-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_MATERIAL_CHANGE_OUTPUT="$TEST_ROOT/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_material_change.sh" >/dev/null
done
cmp "$TEST_ROOT/runtime-one" "$TEST_ROOT/runtime-two" || fail 'two Sounio builds differ'
RUNTIME="$TEST_ROOT/runtime-one"

expect_allow() {
  local mode="$1" word="$2" expected="$3" observed
  observed="$(printf '9044 %s 4 %s 1 2 3 4 9 9\n' "$mode" "$word" | "$RUNTIME")"
  [[ "$observed" == "SOUNIO_SOVEREIGN_MATERIAL_CHANGE $expected semantic_authority=Sounio action=9044" ]] ||
    fail "$expected diverged: $observed"
}

expect_allow 1 65535 MATERIAL_PREPARED
expect_allow 2 1048575 MATERIAL_CONSUMED
expect_allow 3 16777215 COMMIT_ADMIT
expect_allow 4 268435455 CI_ADMIT
expect_allow 5 4294967295 CLAIM_READY

set +e
negative="$(printf '9044 1 4 65531 1 2 3 4 9 9\n' | "$RUNTIME")"
negative_code=$?
zero_descriptor="$(printf '9044 1 4 65535 0 0 0 0 9 9\n' | "$RUNTIME")"
zero_code=$?
sabotage="$(printf '9044 1 4 65535 1 2 3 4 8 9\n' | "$RUNTIME")"
sabotage_code=$?
set -e
[[ $negative_code -eq 42 && "$negative" == *' DENY632 '* ]] || fail 'provider-readonly control diverged'
[[ $zero_code -eq 42 && "$zero_descriptor" == *' DENY635 '* ]] || fail 'descriptor control diverged'
[[ $sabotage_code -eq 42 && "$sabotage" == *' DENY636 '* ]] || fail 'sabotage-count control diverged'

MUTANT="$TEST_ROOT/mutant.sio"
sed 's/o.provider_root_readonly != 1 ||/false ||/' \
  "$ROOT_DIR/stdlib/coordination/loom_sovereign_material_change_authority.sio" >"$MUTANT"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_sovereign_material_change_authority.sio" "$MUTANT" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'provider-readonly mutation did not change Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_MATERIAL_CHANGE_MODULE="$MUTANT" \
SOUNIO_LOOM_MATERIAL_CHANGE_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_material_change.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'provider-readonly mutant did not build'
mutant="$(printf '9044 1 4 65531 1 2 3 4 9 9\n' | "$MUTANT_RUNTIME")"
[[ "$mutant" == *' MATERIAL_PREPARED '* ]] ||
  fail "load-bearing provider-readonly rule did not promote the unchanged witness: $mutant"

ORACLE_MARKER="$TEST_ROOT/prohibited-oracle"
mkdir "$TEST_ROOT/no-oracle"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf invoked >%q\nexit 97\n' "$ORACLE_MARKER" >"$TEST_ROOT/no-oracle/$name"
  chmod 0555 "$TEST_ROOT/no-oracle/$name"
done
printf '9044 1 4 65535 1 2 3 4 9 9\n' | \
  env PATH="$TEST_ROOT/no-oracle:$PATH" "$RUNTIME" >/dev/null
[[ ! -e "$ORACLE_MARKER" ]] || fail 'a prohibited Python or Rust oracle executed'

printf 'sounio-loom-sovereign-material-change-selftest: PASS semantic_authority=Sounio action=9044 stage=SOUNIO_EXECUTABLE cases=8 prepare=MATERIAL_PREPARED consume=MATERIAL_CONSUMED commit=COMMIT_ADMIT ci=CI_ADMIT claim=CLAIM_READY provider_readonly=LOAD_BEARING descriptor=BOUND causal_sabotage=PASS python_executed=false rust_executed=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_sovereign_material_change_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
