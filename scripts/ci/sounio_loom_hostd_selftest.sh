#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/bin/sounio-loom"
AGENT=loom-hostd-test
LANE=durable-lane
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-hostd.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
export SOUNIO_COORD_RUNTIME_MODE=local

fail() {
  printf 'sounio-loom-hostd-selftest: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

cleanup() {
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

wait_guardian_bridge_zero() {
  local output='' attempt
  for attempt in $(seq 1 160); do
    output="$($LOOM guardian-status --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
      --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    [[ "$output" == *'bridge_clients=0'* ]] && return 0
    sleep 0.05
  done
  fail "Guardian did not release the dead kernel bridge: $output"
}

wait_process_absent() {
  local pid="$1" label="$2" attempt
  for attempt in $(seq 1 160); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.05
  done
  fail "$label process remained live: $pid"
}

bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh" >/dev/null
runtime="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
authority="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-boot-reconciler"
[[ -x "$runtime" && -x "$authority" ]] || fail 'Loom or Sounio authority runtime is absent'
[[ "$(sha256sum "$authority" | cut -d ' ' -f 1)" == \
   '99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b' ]] ||
  fail 'Sounio authority runtime hash drifted'

SOUNIO_LOOM_DURABLE_LANE_CANARY=1 "$LOOM" start --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$LANE" --session-id loom-hostd-selftest \
  --cwd "$TEST_ROOT" -- "$runtime" _durable-lane-canary >/dev/null

initial="$($LOOM status --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE")"
instance="$(field instance_id "$initial")"
kernel_before="$(field daemon_pid "$initial")"
guardian="$(field guardian_pid "$initial")"
harness="$(field harness_pid "$initial")"
[[ -n "$instance" && -n "$kernel_before" && -n "$guardian" && -n "$harness" ]] ||
  fail 'initial physical identity is incomplete'

enroll="$($LOOM host-enroll --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE")"
[[ "$enroll" == *'authority=Sounio action=9041 service_enabled=false'* ]] ||
  fail "host enrollment omitted authority boundary: $enroll"

active="$($LOOM host-reconcile --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled)"
[[ "$active" == *'decision=NOOP_ACTIVE action=noop'* ]] ||
  fail "active reconciliation diverged: $active"

python_path="$(command -v python3 || true)"
[[ -n "$python_path" ]] || fail 'Python negative-control executable is unavailable'
if SOUNIO_LOOM_HOST_BOOT_AUTHORITY="$python_path" "$LOOM" host-reconcile \
  --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" --agent "$AGENT" --lane "$LANE" \
  --service-enabled >"$TEST_ROOT/python.out" 2>"$TEST_ROOT/python.err"; then
  fail 'Python oracle was admitted as host reconciliation authority'
fi
grep -q 'host-boot-authority-digest-mismatch' "$TEST_ROOT/python.err" ||
  fail 'Python oracle was refused by the wrong boundary'

supervise_one="$($LOOM host-supervise --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled --once)"
[[ "$supervise_one" == *'decision=NOOP_ACTIVE'* ]] ||
  fail "first supervisor process diverged: $supervise_one"
supervisor_state="$STATE_DIR/hostd/supervisor.state"
first_supervisor_pid="$(sed -n 's/^pid=//p' "$supervisor_state")"
[[ "$(sed -n 's/^state=//p' "$supervisor_state")" == active ]] ||
  fail 'first supervisor state was not active'
supervise_two="$($LOOM host-supervise --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled --once)"
second_supervisor_pid="$(sed -n 's/^pid=//p' "$supervisor_state")"
[[ "$supervise_two" == *'decision=NOOP_ACTIVE'* && \
   "$first_supervisor_pid" != "$second_supervisor_pid" ]] ||
  fail 'supervisor restart did not preserve reconciliation'

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
wait_guardian_bridge_zero
kill -0 "$guardian" 2>/dev/null || fail 'Guardian died with disposable kernel'
kill -0 "$harness" 2>/dev/null || fail 'harness died with disposable kernel'

plan="$($LOOM host-reconcile --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled)"
[[ "$plan" == *'decision=RECOVER_SAME_PHYSICAL action=plan'* ]] ||
  fail "same-physical recovery plan diverged: $plan"
kill -0 "$kernel_before" 2>/dev/null && fail 'plan mode restarted the dead kernel'

descriptor="$STATE_DIR/sessions/$AGENT--$LANE/session.state"
cp "$descriptor" "$TEST_ROOT/descriptor.clean"
sed -i 's/^guardian_pid_start=.*/guardian_pid_start=1/' "$descriptor"
if "$LOOM" host-reconcile --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled --apply \
  >"$TEST_ROOT/sabotage.out" 2>"$TEST_ROOT/sabotage.err"; then
  fail 'Guardian start-tick sabotage authorized recovery'
fi
grep -q 'DENY545' "$TEST_ROOT/sabotage.err" ||
  fail 'Guardian start-tick sabotage was refused by the wrong Sounio rule'
kill -0 "$guardian" 2>/dev/null || fail 'refused sabotage changed Guardian custody'
kill -0 "$harness" 2>/dev/null || fail 'refused sabotage changed harness custody'
cp "$TEST_ROOT/descriptor.clean" "$descriptor"

recovered="$($LOOM host-reconcile --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled --apply)"
[[ "$recovered" == *'decision=RECOVER_SAME_PHYSICAL action=applied'* ]] ||
  fail "same-physical recovery was not applied: $recovered"
after="$($LOOM status --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE")"
kernel_after="$(field daemon_pid "$after")"
[[ "$(field instance_id "$after")" == "$instance" && \
   "$(field guardian_pid "$after")" == "$guardian" && \
   "$(field harness_pid "$after")" == "$harness" && \
   "$kernel_after" != "$kernel_before" ]] ||
  fail 'applied recovery changed physical lane identity or reused the kernel'

"$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
wait_process_absent "$kernel_after" kernel
wait_process_absent "$guardian" Guardian
wait_process_absent "$harness" harness

hold="$($LOOM host-reconcile --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE" --service-enabled --apply)"
[[ "$hold" == *'decision=HOLD_LINEAGE_REQUIRED action=hold'* && \
   "$hold" == *'same_pty_claim=false'* ]] ||
  fail "Guardian loss was not held at the lineage boundary: $hold"
kill -0 "$guardian" 2>/dev/null && fail 'lineage hold created a Guardian'
kill -0 "$harness" 2>/dev/null && fail 'lineage hold created a harness'

verified="$($LOOM host-verify --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --agent "$AGENT" --lane "$LANE")"
receipt_count="$(field receipts "$verified")"
[[ "$verified" == *'hash_chain=PASS semantic_authority=Sounio action=9041'* && \
   "$receipt_count" -ge 8 ]] || fail "receipt chain did not verify: $verified"

printf 'sounio-loom-hostd-selftest: PASS semantic_authority=Sounio action=9041 language=OCaml role=EFFECT_PARITY active=NOOP_ACTIVE supervisor_restart=PASS python_oracle=DENIED_PRE_EXEC recover_plan=PASS guardian_start_sabotage=DENY545 same_physical_recovery=PASS guardian_loss=HOLD_LINEAGE_REQUIRED same_pty_claim=false receipts=%s hash_chain=PASS python_executed=false rust_executed=false production_activation=false\n' \
  "$receipt_count"
