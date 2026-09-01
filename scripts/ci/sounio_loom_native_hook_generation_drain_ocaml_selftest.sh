#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/native-hook-generation-drain-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
AUTHORITY="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-native-hook-generation-drain"
FORBIDDEN_BIN="$TEST_ROOT/forbidden-bin"
FORBIDDEN_LOG="$TEST_ROOT/forbidden-exec.log"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-generation-drain-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_drain.sh" >/dev/null
[[ -x "$LOOM" && -x "$AUTHORITY" ]] || fail 'OCaml or Sounio runtime is absent'

mkdir -p "$FORBIDDEN_BIN"
for executable in python python3 rustc cargo; do
  printf '%s\n' '#!/bin/sh' \
    'printf "%s\n" "$0" >> "$SOUNIO_FORBIDDEN_EXEC_LOG"' \
    'exit 97' > "$FORBIDDEN_BIN/$executable"
  chmod 700 "$FORBIDDEN_BIN/$executable"
done
export PATH="$FORBIDDEN_BIN:$PATH"
export SOUNIO_FORBIDDEN_EXEC_LOG="$FORBIDDEN_LOG"

fixture_defaults() {
  inventory_fresh=true
  inventory_complete=true
  classification_complete=true
  process_generation_bound=true
  hook_capability_bound=true
  old_runtime_bound=true
  candidate_runtime_bound=true
  candidate_config_bound=true
  final_config_bound=true
  canary_mask=15
  rollback_pair_tested=true
  native_entry_open=true
  bridge_free_candidate=true
  current_legacy_bridge=true
  activation_requested=false
  zero_legacy_claimed=false
  total=4
  classified=4
  native=2
  legacy=1
  unknown=1
  unresponsive=0
}

write_fixture() {
  local path="$1"
  printf '%s\n' \
    'schema=loom-native-hook-generation-drain-fixture-v1' \
    'snapshot_utc=2026-08-31T21:00:00Z' \
    "inventory_fresh=$inventory_fresh" \
    "inventory_complete=$inventory_complete" \
    "classification_complete=$classification_complete" \
    "process_generation_bound=$process_generation_bound" \
    "hook_capability_bound=$hook_capability_bound" \
    "old_runtime_bound=$old_runtime_bound" \
    "candidate_runtime_bound=$candidate_runtime_bound" \
    "candidate_config_bound=$candidate_config_bound" \
    "final_config_bound=$final_config_bound" \
    "canary_mask=$canary_mask" \
    "rollback_pair_tested=$rollback_pair_tested" \
    "native_entry_open=$native_entry_open" \
    "bridge_free_candidate=$bridge_free_candidate" \
    "current_legacy_bridge=$current_legacy_bridge" \
    "activation_requested=$activation_requested" \
    "zero_legacy_claimed=$zero_legacy_claimed" \
    "total=$total" \
    "classified=$classified" \
    "native=$native" \
    "legacy=$legacy" \
    "unknown=$unknown" \
    "unresponsive=$unresponsive" \
    'inventory_sha256=1111111111111111111111111111111111111111111111111111111111111111' \
    'old_runtime_sha256=2222222222222222222222222222222222222222222222222222222222222222' \
    'candidate_runtime_sha256=3333333333333333333333333333333333333333333333333333333333333333' \
    'config_pair_sha256=4444444444444444444444444444444444444444444444444444444444444444' \
    'current_runtime_id=legacy-v1' \
    'candidate_runtime_id=native-v2' > "$path"
}

expect_decision() {
  local name="$1" expected="$2" expected_rc="$3"
  local fixture="$TEST_ROOT/$name.fixture"
  write_fixture "$fixture"
  set +e
  local output
  output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" \
    hook-generation-drain-snapshot --cwd "$ROOT_DIR" --fixture "$fixture" 2>&1)"
  local rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" && "$output" == *"\"decision\":\"$expected\""* ]] ||
    fail "$name expected $expected/$expected_rc: rc=$rc output=$output"
  [[ "$output" == *'"semantic_authority":"Sounio"'* && \
    "$output" == *'"operational_realization":"OCaml"'* && \
    "$output" == *'"authority_observed":true'* ]] ||
    fail "$name lost Sounio authority binding: $output"
}

fixture_defaults
expect_decision draining DRAINING 0

fixture_defaults
activation_requested=true
zero_legacy_claimed=true
native=4
legacy=0
unknown=0
expect_decision cutover-ready CUTOVER_READY 0

set +e
cutover_ready_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" \
  hook-generation-cutover-admit --cwd "$ROOT_DIR" \
  --fixture "$TEST_ROOT/cutover-ready.fixture" 2>&1)"
cutover_ready_rc=$?
cutover_draining_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" \
  hook-generation-cutover-admit --cwd "$ROOT_DIR" \
  --fixture "$TEST_ROOT/draining.fixture" 2>&1)"
cutover_draining_rc=$?
set -e
[[ "$cutover_ready_rc" -eq 0 && \
  "$cutover_ready_output" == *'"decision":"CUTOVER_READY"'* && \
  "$cutover_ready_output" == *'"cutover_ready":true'* ]] ||
  fail "native cutover admission rejected the ready Sounio receipt: $cutover_ready_output"
[[ "$cutover_draining_rc" -eq 42 && \
  "$cutover_draining_output" != *'"decision":"CUTOVER_READY"'* && \
  "$cutover_draining_output" == *'"cutover_ready":false'* ]] ||
  fail "native cutover admission accepted a non-absent generation: $cutover_draining_output"

fixture_defaults
activation_requested=true
zero_legacy_claimed=true
native=4
legacy=0
unknown=0
inventory_complete=false
expect_decision incomplete-inventory DENY673 42

fixture_defaults
activation_requested=true
zero_legacy_claimed=true
native=4
legacy=0
unknown=0
inventory_fresh=false
expect_decision stale-inventory DENY673 42

fixture_defaults
activation_requested=true
zero_legacy_claimed=true
native=3
legacy=1
unknown=0
expect_decision false-zero-legacy DENY680 42

fixture_defaults
process_generation_bound=false
expect_decision generation-unbound DENY675 42

fixture_defaults
hook_capability_bound=false
expect_decision capability-unbound DENY675 42

fixture_defaults
canary_mask=14
expect_decision canary-incomplete DENY678 42

fixture_defaults
rollback_pair_tested=false
expect_decision rollback-unproven DENY678 42

fixture_defaults
candidate_config_bound=false
expect_decision candidate-config-unbound DENY672 42

fixture_defaults
classified=3
expect_decision count-arithmetic-invalid DENY674 42

set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_RUNTIME=/usr/bin/true \
  "$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR" \
  --fixture "$TEST_ROOT/draining.fixture" 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 42 && "$runtime_output" == *'authority-runtime-hash-mismatch'* && \
  "$runtime_output" == *'"authority_observed":false'* ]] ||
  fail "runtime tamper was not refused before authority execution: $runtime_output"

tampered_manifest="$TEST_ROOT/native_hook_generation_drain.freeze.v1"
cp "$ROOT_DIR/tools/loom/native_hook_generation_drain.freeze.v1" "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_MANIFEST="$tampered_manifest" \
  "$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR" \
  --fixture "$TEST_ROOT/draining.fixture" 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 42 && "$manifest_output" == *'freeze-manifest-hash-mismatch'* && \
  "$manifest_output" == *'"authority_observed":false'* ]] ||
  fail "manifest tamper was not refused: $manifest_output"

set +e
missing_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_DRAIN_MANIFEST="$TEST_ROOT/missing.freeze.v1" \
  "$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR" \
  --fixture "$TEST_ROOT/draining.fixture" 2>&1)"
missing_rc=$?
set -e
[[ "$missing_rc" -eq 42 && "$missing_output" == *'freeze-manifest-missing'* && \
  "$missing_output" == *'"authority_observed":false'* ]] ||
  fail "missing manifest did not fail closed: $missing_output"

live_output=''
for _attempt in 1 2 3 4 5; do
  set +e
  live_output="$("$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR" 2>&1)"
  live_rc=$?
  set -e
  if [[ "$live_rc" -eq 42 && "$live_output" == *'"cutover_ready":false'* ]]; then
    break
  fi
done
[[ "$live_rc" -eq 42 && "$live_output" == *'"cutover_ready":false'* && \
  "$live_output" == *'"cutover_command_exposed":false'* ]] ||
  fail "live observation did not fail closed: $live_output"
if [[ "$live_output" == *'"authority_observed":true'* ]]; then
  [[ "$live_output" == *'"current_legacy_bridge":true'* && \
    "$live_output" == *'"bridge_free_candidate":true'* ]] ||
    fail "live Sounio refusal lost its generation topology: $live_output"
else
  [[ "$live_output" == *'"decision":"FAIL_CLOSED"'* ]] ||
    fail "live operational drift was not classified fail closed: $live_output"
fi

! grep -Fq 'cockpit-snapshot' "$ROOT_DIR/tools/loom/src/loom_hook_generation_drain.ml" ||
  fail 'OCaml observer still calls the shell cockpit snapshot'
! grep -Fq 'Filename.concat root "bin/sounio-coord"' \
  "$ROOT_DIR/tools/loom/src/loom_hook_generation_drain.ml" ||
  fail 'OCaml observer still calls the legacy coordination launcher'
grep -Fq '/api/hook-generation-drain' "$ROOT_DIR/tools/loom/src/loom.ml" ||
  fail 'HTTP route is not wired'
grep -Fq 'NATIVE HOOK GENERATION DRAIN' "$ROOT_DIR/tools/loom/src/loom_ui.ml" ||
  fail 'UI drain inspector is not wired'
grep -Fq 'refreshDrain' "$ROOT_DIR/tools/loom/src/loom_ui.ml" ||
  fail 'UI drain polling is not wired'
[[ ! -e "$FORBIDDEN_LOG" ]] ||
  fail "forbidden Python or Rust executable ran: $(tr '\n' ' ' < "$FORBIDDEN_LOG")"

printf '%s\n' \
  'sounio-loom-native-hook-generation-drain-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml direct_state_inventory=true kernel_process_binding=true stable_double_snapshot=true incomplete_inventory=DENY673 false_zero=DENY680 generation_or_capability_unbound=DENY675 canary_or_rollback_incomplete=DENY678 config_unbound=DENY672 arithmetic_invalid=DENY674 runtime_tamper=fail_closed manifest_tamper=fail_closed manifest_missing=fail_closed live_drift=fail_closed forbidden_python_rust_exec=absent ui_route=wired cutover_command=native+hidden_until_ready'
