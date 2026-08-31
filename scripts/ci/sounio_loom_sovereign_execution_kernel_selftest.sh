#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-sovereign-selftest.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for ordinal in one two; do
  SOUNIO_LOOM_SOVEREIGN_OUTPUT="$TEST_ROOT/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh" >/dev/null
done
cmp "$TEST_ROOT/runtime-one" "$TEST_ROOT/runtime-two" || fail 'two Sounio builds differ'

RUNTIME="$TEST_ROOT/runtime-one"
TREATMENT=543674140196863
GUARDIAN_DEATH=298961936056319
PRODUCTION=561266326241279

first_line() { printf '%s' "$1" | sed -n '1p'; }

treatment="$(printf '9042 1 3 %s 9 9\n' "$TREATMENT" | "$RUNTIME")"
[[ "$(first_line "$treatment")" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL EXEC_ADMIT semantic_authority=Sounio action=9042' ]] ||
  fail "treatment diverged: $treatment"
death="$(printf '9042 2 3 %s 9 9\n' "$GUARDIAN_DEATH" | "$RUNTIME")"
[[ "$(first_line "$death")" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL GUARDIAN_REVOKE semantic_authority=Sounio action=9042' ]] ||
  fail "guardian death diverged: $death"
production="$(printf '9042 3 3 %s 9 9\n' "$PRODUCTION" | "$RUNTIME")"
[[ "$(first_line "$production")" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL PRODUCTION_GATE_READY semantic_authority=Sounio action=9042' ]] ||
  fail "production gate diverged: $production"

declare -A CONTROLS=(
  [stage]="9042 1 2 $TREATMENT 9 9|DENY602"
  [parent]="9042 1 3 $((TREATMENT ^ (1 << 1))) 9 9|DENY603"
  [authority]="9042 1 3 $((TREATMENT ^ (1 << 5))) 9 9|DENY604"
  [grant]="9042 1 3 $((TREATMENT ^ (1 << 12))) 9 9|DENY605"
  [peer]="9042 1 3 $((TREATMENT ^ (1 << 16))) 9 9|DENY606"
  [release]="9042 1 3 $((TREATMENT ^ (1 << 23))) 9 9|DENY607"
  [continuity]="9042 1 3 $((TREATMENT ^ (1 << 29))) 9 9|DENY608"
  [spoof]="9042 1 3 $((TREATMENT ^ (1 << 34))) 9 9|DENY609"
  [guardian]="9042 2 3 $((GUARDIAN_DEATH ^ (1 << 39))) 9 9|DENY610"
  [material]="9042 1 3 $((TREATMENT ^ (1 << 42))) 9 9|DENY611"
  [production]="9042 3 3 $((PRODUCTION ^ (1 << 43))) 9 9|DENY612"
)
for label in stage parent authority grant peer release continuity spoof guardian material production; do
  IFS='|' read -r frame expected <<< "${CONTROLS[$label]}"
  set +e
  observed="$(printf '%s\n' "$frame" | "$RUNTIME")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == \
    "SOUNIO_SOVEREIGN_EXECUTION_KERNEL $expected semantic_authority=Sounio action=9042" ]] ||
    fail "$label control diverged: $observed"
done

set +e
malformed="$(printf '9042 1 3\n' | "$RUNTIME")"
code=$?
set -e
[[ $code -eq 42 && "$malformed" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL DENY424 reason=malformed-frame semantic_authority=Sounio action=9042' ]] ||
  fail "malformed control diverged: $malformed"

MUTANT="$TEST_ROOT/mutant.sio"
sed 's/observation.same_uid_peer_isolation != 1 {/false {/' \
  "$ROOT_DIR/stdlib/coordination/loom_sovereign_execution_kernel_authority.sio" > "$MUTANT"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_sovereign_execution_kernel_authority.sio" "$MUTANT" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'same-UID mutation did not change Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_SOVEREIGN_MODULE="$MUTANT" \
SOUNIO_LOOM_SOVEREIGN_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'same-UID mutant did not build'
same_uid_negative="9042 3 3 $((PRODUCTION ^ (1 << 43))) 9 9"
mutant="$(printf '%s\n' "$same_uid_negative" | "$MUTANT_RUNTIME")"
[[ "$(first_line "$mutant")" == \
  'SOUNIO_SOVEREIGN_EXECUTION_KERNEL PRODUCTION_GATE_READY semantic_authority=Sounio action=9042' ]] ||
  fail "load-bearing same-UID mutation did not promote unchanged witness: $mutant"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
printf '9042 1 3 %s 9 9\n' "$TREATMENT" | env PATH="$TEST_ROOT:$PATH" "$RUNTIME" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'

printf 'sounio-loom-sovereign-execution-kernel-selftest: PASS semantic_authority=Sounio action=9042 stage=SOUNIO_EXECUTABLE cases=14 treatment=EXEC_ADMIT guardian_death=GUARDIAN_REVOKE production=PRODUCTION_GATE_READY parents=9025+9030+9031 grant=resident-memory+non-bearer+single-use+atomic peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation release_authority=HostGuardian-only interface_release_authority=zero same_uid_spoof=DENY609-before-exec guardian_loss=DENY610-fail-closed production_without_same_uid=DENY612 causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false material_execution=false production_activation=false parity_open=false claim_ready=false source_sha256=%s executable_sha256=%s\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_sovereign_execution_kernel_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
