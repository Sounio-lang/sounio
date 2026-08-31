#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-change-selftest.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-sovereign-change-kernel-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_CHANGE_OUTPUT="$TEST_ROOT/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_change_kernel.sh" >/dev/null
done
cmp "$TEST_ROOT/runtime-one" "$TEST_ROOT/runtime-two" || fail 'two Sounio builds differ'

RUNTIME="$TEST_ROOT/runtime-one"
PREPARE=4294967295
CONSUME=137438953471
COMMIT=2251799813685247
CI=72057594037927935
PRODUCTION=4611686018427387903

first_line() { printf '%s' "$1" | sed -n '1p'; }
expect_allow() {
  local mode="$1" word="$2" expected="$3" observed
  observed="$(printf '9043 %s 3 %s 12 12\n' "$mode" "$word" | "$RUNTIME")"
  [[ "$(first_line "$observed")" == \
    "SOUNIO_SOVEREIGN_CHANGE_KERNEL $expected semantic_authority=Sounio action=9043" ]] ||
    fail "$expected diverged: $observed"
}

expect_allow 1 "$PREPARE" CHANGE_PREPARED
expect_allow 2 "$CONSUME" CHANGE_CONSUMED
expect_allow 3 "$COMMIT" COMMIT_ADMIT
expect_allow 4 "$CI" CI_ADMIT
expect_allow 5 "$PRODUCTION" PRODUCTION_GATE_READY

declare -A CONTROLS=(
  [stage]="9043 1 2 $PREPARE 12 12|DENY613"
  [parent]="9043 1 3 $((PREPARE ^ (1 << 1))) 12 12|DENY614"
  [authority]="9043 1 3 $((PREPARE ^ (1 << 4))) 12 12|DENY615"
  [intent]="9043 1 3 $((PREPARE ^ (1 << 8))) 12 12|DENY616"
  [prestate]="9043 1 3 $((PREPARE ^ (1 << 14))) 12 12|DENY617"
  [peer]="9043 1 3 $((PREPARE ^ (1 << 20))) 12 12|DENY618"
  [grant]="9043 1 3 $((PREPARE ^ (1 << 26))) 12 12|DENY619"
  [lifecycle]="9043 1 3 $((PREPARE ^ (1 << 32))) 12 12|DENY620"
  [post]="9043 2 3 $((CONSUME ^ (1 << 32))) 12 12|DENY621"
  [path]="9043 2 3 $((CONSUME ^ (1 << 35))) 12 12|DENY622"
  [commit]="9043 3 3 $((COMMIT ^ (1 << 37))) 12 12|DENY623"
  [receipt]="9043 3 3 $((COMMIT ^ (1 << 42))) 12 12|DENY624"
  [ci]="9043 4 3 $((CI ^ (1 << 55))) 12 12|DENY625"
  [attachment]="9043 5 3 $((PRODUCTION ^ (1 << 56))) 12 12|DENY626"
  [claim]="9043 5 3 $((PRODUCTION ^ (1 << 61))) 12 12|DENY627"
  [sabotage]="9043 1 3 $PREPARE 11 12|DENY628"
)
for label in stage parent authority intent prestate peer grant lifecycle post path commit receipt ci attachment claim sabotage; do
  IFS='|' read -r frame expected <<< "${CONTROLS[$label]}"
  set +e
  observed="$(printf '%s\n' "$frame" | "$RUNTIME")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == \
    "SOUNIO_SOVEREIGN_CHANGE_KERNEL $expected semantic_authority=Sounio action=9043" ]] ||
    fail "$label control diverged: $observed"
done

set +e
malformed="$(printf '9043 1 3\n' | "$RUNTIME")"
code=$?
set -e
[[ $code -eq 42 && "$malformed" == \
  'SOUNIO_SOVEREIGN_CHANGE_KERNEL DENY424 reason=malformed-frame semantic_authority=Sounio action=9043' ]] ||
  fail "malformed control diverged: $malformed"

MUTANT="$TEST_ROOT/mutant.sio"
sed 's/observation.ci_decision_consumed_not_reinterpreted != 1 {/false {/' \
  "$ROOT_DIR/stdlib/coordination/loom_sovereign_change_kernel_authority.sio" > "$MUTANT"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_sovereign_change_kernel_authority.sio" "$MUTANT" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'CI reinterpretation mutation did not change Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_CHANGE_MODULE="$MUTANT" SOUNIO_LOOM_CHANGE_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_change_kernel.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'CI reinterpretation mutant did not build'
negative_ci="9043 4 3 $((CI ^ (1 << 55))) 12 12"
mutant="$(printf '%s\n' "$negative_ci" | "$MUTANT_RUNTIME")"
[[ "$(first_line "$mutant")" == \
  'SOUNIO_SOVEREIGN_CHANGE_KERNEL CI_ADMIT semantic_authority=Sounio action=9043' ]] ||
  fail "load-bearing CI rule mutation did not promote unchanged witness: $mutant"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
printf '9043 1 3 %s 12 12\n' "$PREPARE" | env PATH="$TEST_ROOT:$PATH" "$RUNTIME" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'

printf 'sounio-loom-sovereign-change-kernel-selftest: PASS semantic_authority=Sounio action=9043 stage=SOUNIO_EXECUTABLE cases=21 prepare=CHANGE_PREPARED consume=CHANGE_CONSUMED commit=COMMIT_ADMIT ci=CI_ADMIT production=PRODUCTION_GATE_READY parent=9042-frozen+production grant=resident-memory+non-bearer+single-use+atomic intent=event+patch+worktree+HEAD+index+file-set peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation ci_policy=consume-not-reinterpret write_attached=false commit_attached=false ci_attached=false claim_ready=false causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_sovereign_change_kernel_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
