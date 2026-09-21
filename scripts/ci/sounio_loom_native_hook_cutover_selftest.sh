#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-cutover-selftest.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-native-hook-cutover-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$TEST_ROOT/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_cutover.sh" >/dev/null
done
cmp "$TEST_ROOT/runtime-one" "$TEST_ROOT/runtime-two" || fail 'two Sounio builds differ'
RUNTIME="$TEST_ROOT/runtime-one"

expect_decision() {
  local frame="$1" expected="$2" observed
  observed="$(printf '9045 %s\n' "$frame" | "$RUNTIME")"
  [[ "$observed" == "SOUNIO_NATIVE_HOOK_CUTOVER $expected semantic_authority=Sounio action=9045" ]] ||
    fail "$expected diverged: $observed"
}

expect_decision '1 3 1 1 3 8388607 0 1 2 3 4 4 4' HOOK_EVENT_ADMIT
expect_decision '1 3 2 1 1 8359935 0 1 2 3 4 4 4' HOOK_EVENT_ADMIT
expect_decision '1 3 3 2 1 8359935 0 1 2 3 4 4 4' HOOK_EVENT_ADMIT
expect_decision '1 3 4 3 1 8359935 0 1 2 3 4 4 4' HOOK_EVENT_ADMIT
expect_decision '2 3 3 2 1 16748543 4 1 2 3 4 4 4' PROVIDER_CANARY_ADMIT
expect_decision '3 3 0 0 0 268406783 15 1 2 3 4 4 4' CLAIM_READY

expect_deny() {
  local frame="$1" expected="$2" observed code
  set +e
  observed="$(printf '9045 %s\n' "$frame" | "$RUNTIME")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == *" $expected "* ]] ||
    fail "$expected control diverged: code=$code result=$observed"
}

expect_deny '1 3 3 1 1 8359935 0 1 2 3 4 4 4' DENY655
expect_deny '1 3 1 1 3 8359935 0 1 2 3 4 4 4' DENY657
expect_deny '1 3 1 1 1 8228863 0 1 2 3 4 4 4' DENY654
expect_deny '2 3 3 2 1 16748543 8 1 2 3 4 4 4' DENY658
expect_deny '3 3 0 0 0 268406783 14 1 2 3 4 4 4' DENY659
expect_deny '1 3 1 1 1 8359935 0 1 2 3 4 3 4' DENY660

MUTANT="$TEST_ROOT/python-rule-removed.sio"
sed 's/o.python_bridge_absent != 1 ||/false ||/' \
  "$ROOT_DIR/stdlib/coordination/loom_native_hook_cutover_authority.sio" >"$MUTANT"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_native_hook_cutover_authority.sio" "$MUTANT" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'Python-absence mutation did not change Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_MODULE="$MUTANT" \
SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_cutover.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'Python-absence mutant did not build'
mutant="$(printf '9045 1 3 1 1 1 8228863 0 1 2 3 4 4 4\n' | "$MUTANT_RUNTIME")"
[[ "$mutant" == *' HOOK_EVENT_ADMIT '* ]] ||
  fail "load-bearing Python-absence rule did not promote the unchanged witness: $mutant"

ORACLE_MARKER="$TEST_ROOT/prohibited-oracle"
mkdir "$TEST_ROOT/no-oracle"
for name in python python3 rustc cargo node ruby awk bc; do
  printf '#!/bin/sh\nprintf invoked >%q\nexit 97\n' "$ORACLE_MARKER" >"$TEST_ROOT/no-oracle/$name"
  chmod 0555 "$TEST_ROOT/no-oracle/$name"
done
printf '9045 1 3 1 1 3 8388607 0 1 2 3 4 4 4\n' | \
  env PATH="$TEST_ROOT/no-oracle:$PATH" "$RUNTIME" >/dev/null
[[ ! -e "$ORACLE_MARKER" ]] || fail 'a prohibited disposable-language oracle executed'

printf 'sounio-loom-native-hook-cutover-selftest: PASS semantic_authority=Sounio action=9045 stage=SOUNIO_EXECUTABLE cases=12 providers=codex,claude,cursor,grok provider_dialects=LOAD_BEARING pre_execution=BOUND four_provider_mask=BOUND python_absence=LOAD_BEARING causal_sabotage=PASS python_executed=false rust_executed=false disposable_oracle_executed=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_native_hook_cutover_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
