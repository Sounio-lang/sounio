#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
AUTHORITY="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-native-hook-generation-reconcile"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-reconcile-ocaml.XXXXXX")"
FORBIDDEN_BIN="$WORK/forbidden-bin"
FORBIDDEN_LOG="$WORK/forbidden.log"
CHILDREN=()

cleanup() {
  local pid
  for pid in "${CHILDREN[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-generation-reconcile-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_text() {
  local text="$1" expected="$2" label="$3"
  [[ "$text" == *"$expected"* ]] || fail "$label missing $expected: $text"
}

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_reconcile.sh" >/dev/null
[[ -x "$LOOM" && -x "$AUTHORITY" ]] || fail 'OCaml or Sounio runtime is absent'

mkdir -p "$FORBIDDEN_BIN"
: >"$FORBIDDEN_LOG"
for executable in python python3 rustc cargo; do
  printf '%s\n' '#!/bin/sh' \
    'printf "%s\n" "$0" >> "$SOUNIO_FORBIDDEN_EXEC_LOG"' \
    'exit 97' >"$FORBIDDEN_BIN/$executable"
  chmod 700 "$FORBIDDEN_BIN/$executable"
done
export PATH="$FORBIDDEN_BIN:$PATH"
export SOUNIO_FORBIDDEN_EXEC_LOG="$FORBIDDEN_LOG"
export SOUNIO_LOOM_HOOK_TEST_MODE=1

BOOT_ID="$(tr -d '\n' </proc/sys/kernel/random/boot_id)"
PID_NAMESPACE="$(readlink /proc/self/ns/pid)"

new_state() {
  local name="$1"
  local state="$WORK/$name"
  mkdir -p "$state/process-presences" "$state/delivery-endpoints" \
    "$state/hook-capabilities" "$state/claims"
  printf '%s\n' "$state"
}

write_presence() {
  local state="$1" agent="$2" lane="$3" pid="$4" pid_start="$5"
  local last_seen="$6" ttl="$7" worktree="$8"
  local key="${agent}--${lane}"
  printf '%s\n' \
    "presence_id=$key" "agent=$agent" "lane=$lane" "worktree=$worktree" \
    'harness=codex' 'session_id=reconcile-test-session' 'host=test-host' \
    "boot_id=$BOOT_ID" "pid_namespace=$PID_NAMESPACE" "pid=$pid" \
    "pid_start=$pid_start" 'generation=7' 'created_utc=2026-09-01T00:00:00Z' \
    'last_seen_utc=2026-09-01T00:00:00Z' "last_seen_epoch=$last_seen" \
    "ttl_seconds=$ttl" >"$state/process-presences/$key.presence"
}

write_related() {
  local state="$1" agent="$2" lane="$3" worktree="$4"
  local key="${agent}--${lane}"
  printf '%s\n' "endpoint_id=$key" "agent=$agent" "lane=$lane" \
    "worktree=$worktree" 'harness=codex' \
    'session_id=reconcile-test-session' "harness_pid=$RELATED_PID" \
    "harness_pid_start=$RELATED_PID_START" \
    >"$state/delivery-endpoints/$key.endpoint"
  printf '%s\n' "claim_id=$key" "agent=$agent" "lane=$lane" \
    "worktree=$worktree" 'branch=test' 'sha=0123456789ab' \
    >"$state/claims/$key.claim"
  printf '%s\n' 'schema=loom-native-hook-capability-v1' \
    'state=NATIVE_HOOK_ATTESTED' "agent=$agent" "lane=$lane" \
    "worktree=$worktree" 'session_id=reconcile-test-session' 'generation=7' \
    'harness=codex' "presence_pid=$RELATED_PID" \
    "presence_pid_start=$RELATED_PID_START" "presence_boot_id=$BOOT_ID" \
    "presence_pid_namespace=$PID_NAMESPACE" \
    >"$state/hook-capabilities/$key.capability"
}

run_reconcile() {
  local state="$1"
  shift
  SOUNIO_LOOM_NATIVE_HOOK_RECONCILE_STATE_DIR="$state" \
    "$LOOM" hook-generation-reconcile --cwd "$ROOT_DIR" "$@"
}

RELATED_PID=0
RELATED_PID_START=0

sleep 300 &
LIVE_PID=$!
CHILDREN+=("$LIVE_PID")
STAT_TEXT="$(<"/proc/$LIVE_PID/stat")"
STAT_TAIL="${STAT_TEXT##*) }"
read -r -a STAT_FIELDS <<<"$STAT_TAIL"
LIVE_START="${STAT_FIELDS[19]}"
NOW="$(date +%s)"

LIVE_STATE="$(new_state live)"
write_presence "$LIVE_STATE" test live "$LIVE_PID" "$LIVE_START" "$NOW" 300 "$WORK"
RELATED_PID="$LIVE_PID"
RELATED_PID_START="$LIVE_START"
write_related "$LIVE_STATE" test live "$WORK"
LIVE_OUTPUT="$(run_reconcile "$LIVE_STATE" --agent test --lane live)"
expect_text "$LIVE_OUTPUT" '"decision":"KEEP"' live-control
expect_text "$LIVE_OUTPUT" '"absence_reason":"none"' live-cause
expect_text "$LIVE_OUTPUT" '"mutation_applied":false' live-mutation

HEARTBEAT_STATE="$(new_state heartbeat)"
write_presence "$HEARTBEAT_STATE" test heartbeat "$LIVE_PID" "$LIVE_START" 1 1 "$WORK"
RELATED_PID="$LIVE_PID"
RELATED_PID_START="$LIVE_START"
write_related "$HEARTBEAT_STATE" test heartbeat "$WORK"
HEARTBEAT_OUTPUT="$(run_reconcile "$HEARTBEAT_STATE" --agent test --lane heartbeat)"
expect_text "$HEARTBEAT_OUTPUT" '"decision":"KEEP"' heartbeat-control
expect_text "$HEARTBEAT_OUTPUT" '"absence_reason":"none"' heartbeat-not-absence

CAUSAL_STATE="$(new_state causal)"
write_presence "$CAUSAL_STATE" test causal "$LIVE_PID" "$LIVE_START" "$NOW" 300 "$WORK"
RELATED_PID="$LIVE_PID"
RELATED_PID_START="$LIVE_START"
write_related "$CAUSAL_STATE" test causal "$WORK"
CAUSAL_LIVE="$(run_reconcile "$CAUSAL_STATE" --agent test --lane causal)"
expect_text "$CAUSAL_LIVE" '"decision":"KEEP"' causal-live-control
kill "$LIVE_PID"
wait "$LIVE_PID" || true
CHILDREN=()
CAUSAL_ABSENT="$(run_reconcile "$CAUSAL_STATE" --agent test --lane causal)"
expect_text "$CAUSAL_ABSENT" '"decision":"QUARANTINE_ELIGIBLE"' causal-treatment
expect_text "$CAUSAL_ABSENT" '"absence_reason":"process-missing"' causal-pid-rule

APPLY_OUTPUT="$(run_reconcile "$CAUSAL_STATE" --agent test --lane causal --apply)"
expect_text "$APPLY_OUTPUT" '"decision":"QUARANTINE_READY"' apply-decision
expect_text "$APPLY_OUTPUT" '"mutation_applied":true' apply-mutation
expect_text "$APPLY_OUTPUT" '"moved_artifacts":4' apply-coverage
[[ ! -e "$CAUSAL_STATE/process-presences/test--causal.presence" ]] || \
  fail 'presence source survived committed quarantine'
[[ ! -e "$CAUSAL_STATE/delivery-endpoints/test--causal.endpoint" ]] || \
  fail 'endpoint source survived committed quarantine'
[[ ! -e "$CAUSAL_STATE/claims/test--causal.claim" ]] || \
  fail 'claim source survived committed quarantine'
[[ ! -e "$CAUSAL_STATE/hook-capabilities/test--causal.capability" ]] || \
  fail 'capability source survived committed quarantine'
RECEIPTS=("$CAUSAL_STATE"/generation-quarantine-receipts/*.receipt)
WALS=("$CAUSAL_STATE"/generation-quarantine-wal/*.wal)
[[ -f "${RECEIPTS[0]}" && -f "${WALS[0]}" ]] || fail 'receipt or WAL missing'
grep -q '^state=COMMITTED$' "${WALS[0]}" || fail 'WAL did not commit'
grep -q '^semantic_authority=Sounio$' "${RECEIPTS[0]}" || fail 'receipt lost authority'
grep -q '^same_uid_peer_isolation=false$' "${RECEIPTS[0]}" || \
  fail 'receipt overclaimed same-UID isolation'
grep -q 'verdict=ALLOW decision=QUARANTINE_READY reason=process-missing' \
  "$CAUSAL_STATE/generation-reconcile-decisions.log" || \
  fail 'committed allow decision was not audited'

DRIFT_STATE="$(new_state identity-drift)"
write_presence "$DRIFT_STATE" test drift 2147483000 1 1 1 "$WORK"
RELATED_PID=2147483000
RELATED_PID_START=1
write_related "$DRIFT_STATE" test drift "$WORK"
printf '%s\n' 'endpoint_id=test--drift' 'agent=test' 'lane=wrong' \
  "worktree=$WORK" 'harness=codex' 'session_id=reconcile-test-session' \
  'harness_pid=2147483000' 'harness_pid_start=1' \
  >"$DRIFT_STATE/delivery-endpoints/test--drift.endpoint"
set +e
DRIFT_OUTPUT="$(run_reconcile "$DRIFT_STATE" --agent test --lane drift 2>&1)"
DRIFT_RC=$?
set -e
[[ "$DRIFT_RC" -eq 42 ]] || fail "identity drift returned $DRIFT_RC"
expect_text "$DRIFT_OUTPUT" '"decision":"FAIL_CLOSED"' identity-drift-decision
[[ -f "$DRIFT_STATE/process-presences/test--drift.presence" ]] || \
  fail 'identity drift mutated presence'
grep -q 'verdict=DENY decision=FAIL_CLOSED' \
  "$DRIFT_STATE/generation-reconcile-decisions.log" || \
  fail 'identity drift denial was not audited'

LOCK_STATE="$(new_state lock)"
write_presence "$LOCK_STATE" test lock 2147483000 1 1 1 "$WORK"
RELATED_PID=2147483000
RELATED_PID_START=1
write_related "$LOCK_STATE" test lock "$WORK"
LOCK_READY="$WORK/lock-ready"
(
  flock -x 9
  : >"$LOCK_READY"
  sleep 4
) 9>"$LOCK_STATE/.claims.lock" &
LOCK_PID=$!
CHILDREN+=("$LOCK_PID")
for _ in {1..100}; do [[ -f "$LOCK_READY" ]] && break; sleep 0.02; done
[[ -f "$LOCK_READY" ]] || fail 'shell flock holder did not start'
set +e
LOCK_OUTPUT="$(run_reconcile "$LOCK_STATE" --agent test --lane lock 2>&1)"
LOCK_RC=$?
set -e
[[ "$LOCK_RC" -eq 42 ]] || fail "shared lock timeout returned $LOCK_RC"
expect_text "$LOCK_OUTPUT" 'state-lock-timeout' shared-flock-control
wait "$LOCK_PID"
CHILDREN=()

PYTHON_STATE="$(new_state python-attempt)"
write_presence "$PYTHON_STATE" test python 2147483000 1 1 1 "$WORK"
RELATED_PID=2147483000
RELATED_PID_START=1
write_related "$PYTHON_STATE" test python "$WORK"
set +e
PYTHON_OUTPUT="$(SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_RUNTIME="$FORBIDDEN_BIN/python" \
  run_reconcile "$PYTHON_STATE" --agent test --lane python 2>&1)"
PYTHON_RC=$?
set -e
[[ "$PYTHON_RC" -eq 42 ]] || fail "Python oracle attempt returned $PYTHON_RC"
expect_text "$PYTHON_OUTPUT" 'authority-runtime-hash-mismatch' python-pre-exec-refusal
[[ ! -s "$FORBIDDEN_LOG" ]] || fail 'forbidden Python or Rust executable ran'

printf '%s\n' \
  'sounio-loom-native-hook-generation-reconcile-ocaml-selftest: PASS semantic_authority=Sounio action=9047 stage=PARITY_OPEN operational_realization=OCaml live=KEEP heartbeat_only=KEEP pid_absent_plan=QUARANTINE_ELIGIBLE pid_absent_apply=QUARANTINE_READY related_artifacts=4 wal=COMMITTED audit=ALLOW+DENY identity_drift=FAIL_CLOSED shared_lock=flock2 timeout=FAIL_CLOSED python_oracle_attempt=PRE_EXEC_REFUSED causal_control=LIVE_THEN_PID_ABSENT python_executed=false rust_executed=false disposable_oracle_executed=false same_uid_peer_isolation=false quarantine_committed=true native_entry_open=false cutover_ready=false bridge_free_current=false'
