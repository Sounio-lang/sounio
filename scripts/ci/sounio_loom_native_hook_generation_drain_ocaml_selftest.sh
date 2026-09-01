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
LIVE_PID=''

cleanup() {
  if [[ -n "$LIVE_PID" ]] && kill -0 "$LIVE_PID" 2>/dev/null; then
    kill "$LIVE_PID" 2>/dev/null || true
    wait "$LIVE_PID" 2>/dev/null || true
  fi
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

LIVE_COMMON="$TEST_ROOT/live-common"
LIVE_STATE="$LIVE_COMMON/sounio-coord-state"
LIVE_RUNTIME_ROOT="$LIVE_COMMON/sounio-coord-runtime"
LIVE_CURRENT="$LIVE_RUNTIME_ROOT/versions/legacy-test"
LIVE_CANDIDATE="$LIVE_RUNTIME_ROOT/versions/native-test"
LIVE_MARKERS="$LIVE_RUNTIME_ROOT/native-hook-drain"
mkdir -p "$LIVE_STATE/process-presences" "$LIVE_STATE/hook-capabilities" \
  "$LIVE_CURRENT/hooks" "$LIVE_CANDIDATE/bin" "$LIVE_MARKERS"
printf '%s\n' 'runtime_id=legacy-test' > "$LIVE_CURRENT/manifest"
printf '%s\n' '# legacy bridge fixture; never executed' \
  > "$LIVE_CURRENT/hooks/sounio_coord_agent_hook_runtime.py"
cp "$LOOM" "$LIVE_CANDIDATE/bin/sounio-loom-runtime"
cp /bin/true "$LIVE_CANDIDATE/bin/sounio-coord-runtime"
chmod 0555 "$LIVE_CANDIDATE/bin/sounio-loom-runtime" \
  "$LIVE_CANDIDATE/bin/sounio-coord-runtime"
ln -s "versions/legacy-test" "$LIVE_RUNTIME_ROOT/current"
ln -s "versions/native-test" "$LIVE_RUNTIME_ROOT/native-next"

LIVE_LOOM_SHA="$(sha256sum "$LIVE_CANDIDATE/bin/sounio-loom-runtime" | awk '{print $1}')"
LIVE_COORD_SHA="$(sha256sum "$LIVE_CANDIDATE/bin/sounio-coord-runtime" | awk '{print $1}')"
CODEX_CONFIG_SHA="$(sha256sum "$ROOT_DIR/.codex/hooks.json" | awk '{print $1}')"
CLAUDE_CONFIG_SHA="$(sha256sum "$ROOT_DIR/.claude/settings.json" | awk '{print $1}')"
CURSOR_CONFIG_SHA="$(sha256sum "$ROOT_DIR/.cursor/hooks.json" | awk '{print $1}')"
GROK_CONFIG_SHA="$(sha256sum "$ROOT_DIR/.grok/hooks/loom-native.json" | awk '{print $1}')"
printf '%s\n' \
  'runtime_id=native-test' \
  'source_sha=native-test-source' \
  'loom_native_hook_cutover_python_bridge_absent=true' \
  "loom_runtime_sha256=$LIVE_LOOM_SHA" \
  "coord_runtime_sha256=$LIVE_COORD_SHA" \
  "loom_native_hook_cutover_codex_config_sha256=$CODEX_CONFIG_SHA" \
  "loom_native_hook_cutover_claude_config_sha256=$CLAUDE_CONFIG_SHA" \
  "loom_native_hook_cutover_cursor_config_sha256=$CURSOR_CONFIG_SHA" \
  "loom_native_hook_cutover_grok_config_sha256=$GROK_CONFIG_SHA" \
  > "$LIVE_CANDIDATE/manifest"

CONFIG_COMPONENTS="$TEST_ROOT/config-components.bin"
: > "$CONFIG_COMPONENTS"
for relative in .codex/hooks.json .claude/settings.json .cursor/hooks.json \
    .grok/hooks/loom-native.json; do
  printf '%s\0%s' "$relative" "$(sha256sum "$ROOT_DIR/$relative" | awk '{print $1}')" \
    >> "$CONFIG_COMPONENTS"
  [[ "$relative" == .grok/hooks/loom-native.json ]] || printf '\0' >> "$CONFIG_COMPONENTS"
done
CONFIG_BUNDLE_SHA="$({ printf 'loom-hook-config-bundle-v1\0'; cat "$CONFIG_COMPONENTS"; } | sha256sum | awk '{print $1}')"
CANDIDATE_MANIFEST_SHA="$(sha256sum "$LIVE_CANDIDATE/manifest" | awk '{print $1}')"
printf '%s\n' 'fixture guardian public key' > "$LIVE_MARKERS/guardian-ed25519-public.pem"
GUARDIAN_PUBLIC_SHA="$(sha256sum "$LIVE_MARKERS/guardian-ed25519-public.pem" | awk '{print $1}')"
printf '%s\n' \
  'schema=loom-native-hook-final-config-v1' \
  'state=FINAL_CONFIG_BOUND' \
  'runtime_id=native-test' \
  "runtime_manifest_sha256=$CANDIDATE_MANIFEST_SHA" \
  "config_bundle_sha256=$CONFIG_BUNDLE_SHA" \
  "guardian_public_key_sha256=$GUARDIAN_PUBLIC_SHA" \
  'semantic_authority=Sounio' \
  'action=9046' > "$LIVE_MARKERS/final-config.v1"

LIVE_CALLER_FIXTURE="$TEST_ROOT/large-caller-sleep"
cp /bin/sleep "$LIVE_CALLER_FIXTURE"
truncate -s 9437184 "$LIVE_CALLER_FIXTURE"
chmod 0555 "$LIVE_CALLER_FIXTURE"
[[ "$(stat -c '%s' "$LIVE_CALLER_FIXTURE")" -gt 8388608 ]] ||
  fail 'large caller fixture did not cross the bounded metadata-read threshold'
"$LIVE_CALLER_FIXTURE" 120 &
LIVE_PID=$!
LIVE_PID_START="$(sed 's/^[^)]*) //' "/proc/$LIVE_PID/stat" | awk '{print $20}')"
LIVE_BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
LIVE_PID_NAMESPACE="$(readlink "/proc/$LIVE_PID/ns/pid")"
LIVE_CALLER="$(readlink -f "/proc/$LIVE_PID/exe")"
LIVE_CALLER_SHA="$(sha256sum "$LIVE_CALLER" | awk '{print $1}')"
LIVE_NOW="$(date +%s)"

write_presence() {
  local lane="$1" session_id="$2" key
  key="grok--$lane"
  printf '%s\n' \
    "presence_id=$key" 'agent=grok' "lane=$lane" "worktree=$ROOT_DIR" \
    'harness=grok' "session_id=$session_id" 'host=selftest' \
    "boot_id=$LIVE_BOOT_ID" "pid_namespace=$LIVE_PID_NAMESPACE" \
    "pid=$LIVE_PID" "pid_start=$LIVE_PID_START" 'generation=1' \
    'created_utc=2026-09-01T00:00:00Z' \
    'last_seen_utc=2026-09-01T00:00:00Z' "last_seen_epoch=$LIVE_NOW" \
    'ttl_seconds=1800' > "$LIVE_STATE/process-presences/$key.presence"
}

write_capability() {
  local lane="$1" session_id="$2" generation="$3" key
  key="grok--$lane"
  printf '%s\n' \
    'schema=loom-native-hook-capability-v1' 'state=NATIVE_HOOK_ATTESTED' \
    'agent=grok' "lane=$lane" "session_id=$session_id" \
    "generation=$generation" "worktree=$ROOT_DIR" 'harness=grok' \
    "presence_pid=$LIVE_PID" "presence_pid_start=$LIVE_PID_START" \
    "presence_boot_id=$LIVE_BOOT_ID" \
    "presence_pid_namespace=$LIVE_PID_NAMESPACE" \
    "producer_executable=$LIVE_CANDIDATE/bin/sounio-loom-runtime" \
    "producer_sha256=$LIVE_LOOM_SHA" \
    "coord_executable=$LIVE_CANDIDATE/bin/sounio-coord-runtime" \
    "coord_sha256=$LIVE_COORD_SHA" "caller_pid=$LIVE_PID" \
    "caller_pid_start=$LIVE_PID_START" "caller_boot_id=$LIVE_BOOT_ID" \
    "caller_pid_namespace=$LIVE_PID_NAMESPACE" \
    "caller_executable=$LIVE_CALLER" "caller_sha256=$LIVE_CALLER_SHA" \
    'wake_eligible=1' 'runtime_id=native-test' \
    'source_sha=native-test-source' 'created_utc=2026-09-01T00:00:00Z' \
    "created_epoch=$LIVE_NOW" "expires_epoch=$((LIVE_NOW + 1800))" \
    > "$LIVE_STATE/hook-capabilities/$key.capability"
}

FLEET_LANE='fleet-positive'
FLEET_SESSION='fleet-positive-session'
NATIVE_LANE='session-positive'
NATIVE_SESSION='native-positive-session'
write_presence "$FLEET_LANE" "$FLEET_SESSION"
write_presence "$NATIVE_LANE" "$NATIVE_SESSION"
NATIVE_GENERATION="process-$NATIVE_SESSION-g1-$LIVE_PID-$LIVE_PID_START"
write_capability "$NATIVE_LANE" "$NATIVE_SESSION" "$NATIVE_GENERATION"

observe_isolated_live() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_NATIVE_HOOK_DRAIN_COMMON_DIR="$LIVE_COMMON" \
  SOUNIO_LOOM_NATIVE_HOOK_DRAIN_SKIP_UI_ATTESTATION=1 \
    "$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR" 2>&1
}

set +e
live_positive_output="$(observe_isolated_live)"
live_positive_rc=$?
set -e
[[ "$live_positive_rc" -eq 42 && \
  "$live_positive_output" == *'"authority_observed":true'* && \
  "$live_positive_output" == *'"process_generation_bound":true'* && \
  "$live_positive_output" == *'"hook_capability_bound":true'* && \
  "$live_positive_output" == *'"records":2,"processes":1,"collapsed_aliases":1,"total":1,"classified":1,"native":1,"legacy":0,"unknown":0,"unresponsive":0'* && \
  "$live_positive_output" == *'"lane":"session-positive"'* && \
  "$live_positive_output" != *'"lane":"fleet-positive"'* ]] ||
  fail "canonical live capability or process alias collapse failed: $live_positive_output"

write_capability "$NATIVE_LANE" "$NATIVE_SESSION" '1'
set +e
generation_drift_output="$(observe_isolated_live)"
generation_drift_rc=$?
set -e
[[ "$generation_drift_rc" -eq 42 && \
  "$generation_drift_output" == *'"decision":"DENY675"'* && \
  "$generation_drift_output" == *'"hook_capability_bound":false'* && \
  "$generation_drift_output" == *'"native":0,"legacy":0,"unknown":1,"unresponsive":0'* ]] ||
  fail "non-canonical capability generation did not fail closed: $generation_drift_output"

write_capability "$NATIVE_LANE" "$NATIVE_SESSION" "$NATIVE_GENERATION"
FLEET_GENERATION="process-$FLEET_SESSION-g1-$LIVE_PID-$LIVE_PID_START"
write_capability "$FLEET_LANE" "$FLEET_SESSION" "$FLEET_GENERATION"
set +e
alias_conflict_output="$(observe_isolated_live)"
alias_conflict_rc=$?
set -e
[[ "$alias_conflict_rc" -eq 42 && \
  "$alias_conflict_output" == *'"decision":"DENY675"'* && \
  "$alias_conflict_output" == *'"capability_reason":"process-alias-capability-conflict"'* && \
  "$alias_conflict_output" == *'"native":0,"legacy":0,"unknown":1,"unresponsive":0'* ]] ||
  fail "conflicting capabilities for one kernel process were not refused: $alias_conflict_output"
rm -f "$LIVE_STATE/hook-capabilities/grok--$FLEET_LANE.capability"

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
  'sounio-loom-native-hook-generation-drain-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml direct_state_inventory=true kernel_process_binding=true canonical_process_generation=true fleet_session_alias_collapse=true duplicate_capability_conflict=DENY675 large_caller_hashing=streamed stable_double_snapshot=true incomplete_inventory=DENY673 false_zero=DENY680 generation_or_capability_unbound=DENY675 canary_or_rollback_incomplete=DENY678 config_unbound=DENY672 arithmetic_invalid=DENY674 runtime_tamper=fail_closed manifest_tamper=fail_closed manifest_missing=fail_closed live_drift=fail_closed forbidden_python_rust_exec=absent ui_route=wired cutover_command=native+hidden_until_ready'
