#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-generation-drain-selftest.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-native-hook-generation-drain-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_OUTPUT="$TEST_ROOT/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_drain.sh" \
      >/dev/null
done
cmp "$TEST_ROOT/runtime-one" "$TEST_ROOT/runtime-two" ||
  fail 'two Sounio builds differ'
RUNTIME="$TEST_ROOT/runtime-one"

expect_decision() {
  local frame="$1" expected="$2" observed
  observed="$(printf '9046 %s\n' "$frame" | "$RUNTIME")"
  [[ "$observed" == "SOUNIO_NATIVE_HOOK_GENERATION_DRAIN $expected semantic_authority=Sounio action=9046" ]] ||
    fail "$expected diverged: $observed"
}

expect_deny() {
  local frame="$1" expected="$2" observed code
  set +e
  observed="$(printf '9046 %s\n' "$frame" | "$RUNTIME")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == *" $expected "* ]] ||
    fail "$expected control diverged: code=$code result=$observed"
}

expect_decision '1 3 3932159 5 5 2 2 1 0 15 1 2 3 4 5 6 6' DRAINING
expect_decision '1 3 3932159 3 3 1 0 0 2 15 1 2 3 4 5 6 6' DRAINING
expect_decision '2 3 8388607 4 4 4 0 0 0 15 1 2 3 4 5 6 6' CUTOVER_READY
expect_deny '2 3 8388479 4 4 4 0 0 0 15 1 2 3 4 5 6 6' DENY673
expect_deny '2 3 8388607 5 4 4 0 0 0 15 1 2 3 4 5 6 6' DENY674
expect_deny '2 3 8388095 4 4 4 0 0 0 15 1 2 3 4 5 6 6' DENY675
expect_deny '2 3 8355839 4 4 4 0 0 0 15 1 2 3 4 5 6 6' DENY676
expect_deny '2 3 8388607 4 4 4 0 0 0 15 1 2 3 4 5 5 6' DENY677
expect_deny '2 3 8388607 4 4 4 0 0 0 14 1 2 3 4 5 6 6' DENY678
expect_deny '1 3 4194303 5 5 2 2 1 0 15 1 2 3 4 5 6 6' DENY679
expect_deny '2 3 8388607 4 4 3 1 0 0 15 1 2 3 4 5 6 6' DENY680
expect_deny '2 3 8388607 0 0 0 0 0 0 15 1 2 3 4 5 6 6' DENY680

MUTANT="$TEST_ROOT/inventory-completeness-rule-removed.sio"
sed 's/o.inventory_complete != 1 ||/false ||/' \
  "$ROOT_DIR/stdlib/coordination/loom_native_hook_generation_drain_authority.sio" \
  >"$MUTANT"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_native_hook_generation_drain_authority.sio" \
    "$MUTANT" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'inventory-completeness mutation did not change Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_MODULE="$MUTANT" \
SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_drain.sh" \
    >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'inventory-completeness mutant did not build'
mutant="$(printf '9046 2 3 8388479 4 4 4 0 0 0 15 1 2 3 4 5 6 6\n' | \
  "$MUTANT_RUNTIME")"
[[ "$mutant" == *' CUTOVER_READY '* ]] ||
  fail "load-bearing inventory-completeness rule did not promote the unchanged witness: $mutant"

ORACLE_MARKER="$TEST_ROOT/prohibited-oracle"
mkdir "$TEST_ROOT/no-oracle"
for name in python python3 rustc cargo node ruby awk bc; do
  printf '#!/bin/sh\nprintf invoked >%q\nexit 97\n' "$ORACLE_MARKER" \
    >"$TEST_ROOT/no-oracle/$name"
  chmod 0555 "$TEST_ROOT/no-oracle/$name"
done
printf '9046 2 3 8388607 4 4 4 0 0 0 15 1 2 3 4 5 6 6\n' | \
  env PATH="$TEST_ROOT/no-oracle:$PATH" "$RUNTIME" >/dev/null
[[ ! -e "$ORACLE_MARKER" ]] || fail 'a prohibited disposable-language oracle executed'

printf 'sounio-loom-native-hook-generation-drain-selftest: PASS semantic_authority=Sounio action=9046 stage=SOUNIO_EXECUTABLE cases=12 affirmative_absence=INVENTORY+CLASSIFICATION+ZERO_NONNATIVE causal_sabotage=inventory-completeness-rule-removed python_executed=false rust_executed=false disposable_oracle_executed=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_native_hook_generation_drain_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
