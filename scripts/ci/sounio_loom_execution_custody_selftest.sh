#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-custody.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
COORD_DIR="$TEST_ROOT/coord"
LEAKED_CAPABILITY_DIR="$TEST_ROOT/leaked-v1-capabilities"
FOREIGN_AGENTD_SOCKET="$TEST_ROOT/foreign-agentd.socket"
FOREIGN_AGENTD_TOKEN="$TEST_ROOT/foreign-agentd.token"
SESSION_ID="loom-custody-v2-$$"
AGENT=codex
LANE="session-$SESSION_ID"
HOOK_LANE="$LANE"
ACTIVE=0

fail() {
  echo "sounio-loom-execution-custody-selftest: FAIL: $* test_root=$TEST_ROOT" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

status() {
  "$LOOM" status --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
    --agent "$AGENT" --lane "$LANE"
}

wait_status() {
  local expected="$1" output='' attempt
  for attempt in $(seq 1 160); do
    output="$(status 2>/dev/null || true)"
    [[ "$output" == *"$expected"* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "status did not reach $expected; last=$output"
}

snapshot_from() {
  local cursor="$1"
  "$LOOM" snapshot --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
    --agent "$AGENT" --lane "$LANE" --cursor "$cursor" 2>/dev/null || true
}

send_harness() {
  local command="$1"
  printf '%s\n' "$command" | "$LOOM" attach --state-dir "$STATE_DIR" \
    --cwd "$ROOT_DIR" --agent "$AGENT" --lane "$LANE" --cursor end \
    --no-raw >/dev/null
}

wait_output() {
  local cursor="$1" expected="$2" output='' attempt
  for attempt in $(seq 1 160); do
    output="$(snapshot_from "$cursor")"
    [[ "$output" == *"$expected"* ]] && {
      printf '%s' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "output did not contain $expected; last=$output"
}

cleanup() {
  if [[ "$ACTIVE" == 1 ]]; then
    "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
      --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  fi
  "$ROOT_DIR/bin/sounio-coord" release --agent codex --lane "$HOOK_LANE" \
    --reason "Loom custody selftest finished" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

cat > "$TEST_ROOT/harness.sh" <<'HARNESS'
#!/usr/bin/env bash
set -u
stty -echo

issue() {
  local mode="$1" event output command='/usr/bin/printf CUSTODY_OK'
  [[ "$mode" == nonzero ]] && command='/usr/bin/false'
  if [[ "$mode" == expiry ]]; then
    export SOUNIO_LOOM_HOOK_TEST_MODE=1
    export SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS=1
    export SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR="$SOUNIO_LOOM_TEST_CAPABILITY_DIR"
    export SOUNIO_AGENTD_SOCKET="$SOUNIO_LOOM_TEST_FOREIGN_AGENTD_SOCKET"
    export SOUNIO_AGENTD_TOKEN_FILE="$SOUNIO_LOOM_TEST_FOREIGN_AGENTD_TOKEN"
    export SOUNIO_AGENTD_AGENT=cursor
    export SOUNIO_AGENTD_LANE=foreign-lane
    export SOUNIO_AGENTD_SESSION_ID=foreign-session
    export SOUNIO_AGENTD_WORKTREE="$SOUNIO_LOOM_TEST_ROOT"
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
  elif [[ "$mode" == outcome-recovery ]]; then
    export SOUNIO_LOOM_HOOK_TEST_MODE=1
    export SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS=3
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
  elif [[ "$mode" == outcome-replay ]]; then
    export SOUNIO_LOOM_HOOK_TEST_MODE=1
    export SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY=1
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
  elif [[ "$mode" == outcome-signal ]]; then
    export SOUNIO_LOOM_HOOK_TEST_MODE=1
    export SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL=15
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_MANIFEST
  elif [[ "$mode" == outcome-missing-policy ]]; then
    export SOUNIO_LOOM_HOOK_TEST_MODE=1
    export SOUNIO_LOOM_EXECUTION_OUTCOME_MANIFEST="$SOUNIO_LOOM_TEST_ROOT/tools/loom/missing-execution-outcome.freeze.v1"
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
  else
    unset SOUNIO_LOOM_HOOK_TEST_MODE SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_MANIFEST
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
  fi
  event="$(jq -cn --arg session "$SOUNIO_LOOM_TEST_SESSION" \
    --arg root "$SOUNIO_LOOM_TEST_ROOT" \
    --arg command "$command" \
    '{hook_event_name:"PreToolUse",session_id:$session,cwd:$root,
      tool_name:"exec_command",tool_input:{cmd:$command,workdir:$root}}')"
  if ! output="$(printf '%s\n' "$event" | \
      "$SOUNIO_LOOM_TEST_BINARY" agent-hook --agent codex 2>&1)"; then
    printf 'ISSUE_FAILED:%s\n' "$output"
    return
  fi
  HELD_REPLACEMENT="$(printf '%s\n' "$output" | \
    jq -er '.hookSpecificOutput.updatedInput.cmd')" || {
      printf 'ISSUE_PARSE_FAILED:%s\n' "$output"
      return
    }
  eval "set -- $HELD_REPLACEMENT"
  HELD_INSTANCE="$4"
  HELD_GENERATION="$6"
  HELD_HANDLE="$8"
  printf 'ISSUED mode=%s instance=%s generation=%s handle=%s\n' \
    "$mode" "$HELD_INSTANCE" "$HELD_GENERATION" "$HELD_HANDLE"
}

consume() {
  local mode="$1" rc output cwd="$SOUNIO_LOOM_TEST_ROOT"
  [[ "$mode" == wrong-cwd ]] && cwd="$SOUNIO_LOOM_TEST_ROOT/tools/loom"
  output="$(cd "$cwd" && /bin/sh -c "$HELD_REPLACEMENT" 2>&1)"
  rc=$?
  printf 'CONSUMED mode=%s rc=%d output=%s\n' "$mode" "$rc" "$output"
  if [[ "$mode" == expiry || "$mode" == outcome-recovery || \
    "$mode" == outcome-replay || "$mode" == outcome-signal || \
    "$mode" == outcome-missing-policy ]]; then
    unset SOUNIO_LOOM_HOOK_TEST_MODE SOUNIO_LOOM_EXECUTION_CAPABILITY_TTL_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_PAUSE_SECONDS
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_REPLAY
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_CHILD_SIGNAL
    unset SOUNIO_LOOM_EXECUTION_OUTCOME_MANIFEST
    unset SOUNIO_LOOM_EXECUTION_CAPABILITY_DIR SOUNIO_AGENTD_SOCKET
    unset SOUNIO_AGENTD_TOKEN_FILE SOUNIO_AGENTD_AGENT SOUNIO_AGENTD_LANE
    unset SOUNIO_AGENTD_SESSION_ID SOUNIO_AGENTD_WORKTREE
  fi
  return 0
}

printf 'CUSTODY_HARNESS_READY\n'
HELD_REPLACEMENT=''
while IFS= read -r command; do
  case "$command" in
    ISSUE) issue normal ;;
    ISSUE_EXPIRY) issue expiry ;;
    ISSUE_RECOVERY) issue recovery ;;
    ISSUE_OUTCOME_RECOVERY) issue outcome-recovery ;;
    ISSUE_OUTCOME_REPLAY) issue outcome-replay ;;
    ISSUE_OUTCOME_NONZERO) issue nonzero ;;
    ISSUE_OUTCOME_SIGNAL) issue outcome-signal ;;
    ISSUE_OUTCOME_MISSING_POLICY) issue outcome-missing-policy ;;
    CONSUME) consume normal ;;
    CONSUME_WRONG_CWD) consume wrong-cwd ;;
    CONSUME_EXPIRY) consume expiry ;;
    CONSUME_RECOVERY) consume recovery ;;
    CONSUME_OUTCOME_RECOVERY) consume outcome-recovery ;;
    CONSUME_OUTCOME_REPLAY) consume outcome-replay ;;
    CONSUME_OUTCOME_NONZERO) consume nonzero ;;
    CONSUME_OUTCOME_SIGNAL) consume outcome-signal ;;
    CONSUME_OUTCOME_MISSING_POLICY) consume outcome-missing-policy ;;
    *) printf 'UNKNOWN:%s\n' "$command" ;;
  esac
done
HARNESS
chmod +x "$TEST_ROOT/harness.sh"
mkdir -p "$LEAKED_CAPABILITY_DIR"
: >"$FOREIGN_AGENTD_SOCKET"
: >"$FOREIGN_AGENTD_TOKEN"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
export SOUNIO_LOOM_TEST_BINARY="$LOOM"
export SOUNIO_LOOM_TEST_ROOT="$ROOT_DIR"
export SOUNIO_LOOM_TEST_SESSION="$SESSION_ID"
export SOUNIO_LOOM_TEST_CAPABILITY_DIR="$LEAKED_CAPABILITY_DIR"
export SOUNIO_LOOM_TEST_FOREIGN_AGENTD_SOCKET="$FOREIGN_AGENTD_SOCKET"
export SOUNIO_LOOM_TEST_FOREIGN_AGENTD_TOKEN="$FOREIGN_AGENTD_TOKEN"
export SOUNIO_LOOM_COORD_AUTO=0
export SOUNIO_COORD_DIR="$COORD_DIR"
export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1

set +e
fixture_flag_output="$("$LOOM" agent-hook --agent codex \
  --test-file-capability-fixture </dev/null 2>&1)"
fixture_flag_rc=$?
set -e
[[ "$fixture_flag_rc" -eq 2 && \
   "$fixture_flag_output" == *'file-capability-fixture-requires-test-mode'* ]] || \
  fail "file fixture flag was not test-mode gated: rc=$fixture_flag_rc output=$fixture_flag_output"

"$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --session-id "$SESSION_ID" --cwd "$ROOT_DIR" -- \
  /bin/bash "$TEST_ROOT/harness.sh" >/dev/null
ACTIVE=1

initial="$(wait_status 'pending_exec_grants=0')"
[[ "$initial" == *'pending_exec_outcomes=0'* ]] || fail 'initial outcome obligations are not empty'
descriptor="$STATE_DIR/sessions/$AGENT--$LANE/session.state"
socket="$(sed -n 's/^socket=//p' "$descriptor")"
token_file="$(sed -n 's/^token_file=//p' "$descriptor")"
instance="$(field instance_id "$initial")"
generation="$(field kernel_generation "$initial")"
journal="$(field journal "$initial")"
[[ -S "$socket" && -f "$token_file" && -n "$instance" && \
   ${#generation} -eq 64 && -f "$journal" ]] || fail 'kernel identity incomplete'

boot="$(wait_output 0 CUSTODY_HARNESS_READY)"
cursor="$(field output_cursor "$(status)")"
send_harness ISSUE
issued="$(wait_output "$cursor" 'ISSUED mode=normal')"
issued_line="$(grep 'ISSUED mode=normal' <<< "$issued" | tail -1)"
handle="$(sed -n 's/.* handle=\([0-9a-f]*\).*/\1/p' <<< "$issued_line")"
issued_generation="$(sed -n 's/.* generation=\([0-9a-f]*\) handle=.*/\1/p' <<< "$issued_line")"
[[ ${#handle} -eq 64 && "$issued_generation" == "$generation" ]] || \
  fail "issued grant identity malformed: $issued_line"
wait_status 'pending_exec_grants=1' >/dev/null

common_dir="$(git rev-parse --git-common-dir)"
[[ "$common_dir" = /* ]] || common_dir="$ROOT_DIR/$common_dir"
outcome_dir="$common_dir/sounio-loom-execution-authority/outcomes"
[[ ! -e "$common_dir/sounio-loom-execution-capabilities/$handle.cap" ]] || \
  fail 'production route wrote an authority capability file'

set +e
attacker_output="$(SOUNIO_LOOM_SOCKET="$socket" \
  SOUNIO_LOOM_TOKEN_FILE="$token_file" SOUNIO_LOOM_INSTANCE_ID="$instance" \
  "$LOOM" exec-capability --instance "$instance" --generation "$generation" \
  --handle "$handle" 2>&1)"
attacker_rc=$?
set -e
[[ "$attacker_rc" -eq 126 && "$attacker_output" == *'outside-harness-ancestry'* ]] || \
  fail "same-UID peer was not refused by ancestry: rc=$attacker_rc output=$attacker_output"
wait_status 'pending_exec_grants=1' >/dev/null

cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_WRONG_CWD
wrong_cwd="$(wait_output "$cursor" 'CONSUMED mode=wrong-cwd')"
[[ "$wrong_cwd" == *'rc=126'* && "$wrong_cwd" == *'exec-grant-cwd-mismatch'* ]] || \
  fail "cwd drift was not refused: $wrong_cwd"
wait_status 'pending_exec_grants=1' >/dev/null

cursor="$(field output_cursor "$(status)")"
receipt_marker="$TEST_ROOT/receipt-marker"
touch "$receipt_marker"
send_harness CONSUME
consumed="$(wait_output "$cursor" 'CONSUMED mode=normal')"
[[ "$consumed" == *'rc=0 output=CUSTODY_OK'* ]] || \
  fail "rightful descendant did not execute once: $consumed"
wait_status 'pending_exec_grants=0' >/dev/null
wait_status 'pending_exec_outcomes=0' >/dev/null

incomplete_before_missing_policy="$(grep -c $'\tEXEC_OUTCOME_INCOMPLETE\t' "$journal" || true)"
cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_OUTCOME_MISSING_POLICY
wait_output "$cursor" 'ISSUED mode=outcome-missing-policy' >/dev/null
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_OUTCOME_MISSING_POLICY
missing_policy_outcome="$(wait_output "$cursor" 'CONSUMED mode=outcome-missing-policy')"
[[ "$missing_policy_outcome" == *'rc=126'* && \
   "$missing_policy_outcome" == *'execution-outcome-policy-missing'* && \
   "$missing_policy_outcome" != *'CUSTODY_OK'* ]] || \
  fail "missing outcome policy did not refuse before child execution: $missing_policy_outcome"
wait_status 'pending_exec_outcomes=0' >/dev/null
incomplete_after_missing_policy="$(grep -c $'\tEXEC_OUTCOME_INCOMPLETE\t' "$journal" || true)"
[[ "$incomplete_after_missing_policy" -gt "$incomplete_before_missing_policy" ]] || \
  fail 'broker exit after missing outcome policy did not materialize INCOMPLETE'

cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_OUTCOME_NONZERO
wait_output "$cursor" 'ISSUED mode=nonzero' >/dev/null
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_OUTCOME_NONZERO
nonzero_outcome="$(wait_output "$cursor" 'CONSUMED mode=nonzero')"
[[ "$nonzero_outcome" == *'rc=1 output='* ]] || \
  fail "nonzero exit was not preserved: $nonzero_outcome"
wait_status 'pending_exec_outcomes=0' >/dev/null

signal_marker="$TEST_ROOT/signal-receipt-marker"
touch "$signal_marker"
cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_OUTCOME_SIGNAL
wait_output "$cursor" 'ISSUED mode=outcome-signal' >/dev/null
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_OUTCOME_SIGNAL
signaled_outcome="$(wait_output "$cursor" 'CONSUMED mode=outcome-signal')"
[[ "$signaled_outcome" == *'rc=143 output='* ]] || \
  fail "signaled exit was not preserved: $signaled_outcome"
wait_status 'pending_exec_outcomes=0' >/dev/null
signal_receipt="$(find "$outcome_dir" -type f -name 'outcome-*.receipt' \
  -newer "$signal_marker" -print -quit 2>/dev/null || true)"
[[ -n "$signal_receipt" ]] || fail 'signaled execution receipt is absent'
grep -q '^outcome_kind=2$' "$signal_receipt" || fail 'signaled receipt omitted outcome kind'
grep -q '^signal=15$' "$signal_receipt" || fail 'signaled receipt omitted signal number'
outcome_receipt="$(find "$outcome_dir" -type f -name 'outcome-*.receipt' \
  -newer "$receipt_marker" -print -quit 2>/dev/null || true)"
[[ -n "$outcome_receipt" ]] || fail 'durable execution outcome receipt is absent'
for required in schema=loom-execution-observation-v1 preexec_source_sha256= \
  outcome_source_sha256= grant_sha256= kernel_generation= \
  issue_decision_sha256= consume_decision_sha256= result_sha256= \
  outcome_authority_decision_sha256= record_sha256=; do
  grep -q "^$required" "$outcome_receipt" || \
    fail "outcome receipt omitted $required"
done
receipt_record_sha="$(sed -n 's/^record_sha256=//p' "$outcome_receipt")"
receipt_body_sha="$(sed '$d' "$outcome_receipt" | sha256sum | cut -d' ' -f1)"
[[ "$receipt_record_sha" == "$receipt_body_sha" ]] || \
  fail 'outcome receipt body hash is invalid'

cursor="$(field output_cursor "$(status)")"
send_harness CONSUME
replayed="$(wait_output "$cursor" 'CONSUMED mode=normal')"
[[ "$replayed" == *'rc=126'* && "$replayed" == *'missing-or-replayed'* ]] || \
  fail "handle replay was not refused: $replayed"

outcome_refused_before="$(grep -c $'\tEXEC_OUTCOME_REFUSED\t' "$journal" || true)"
cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_OUTCOME_REPLAY
wait_output "$cursor" 'ISSUED mode=outcome-replay' >/dev/null
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_OUTCOME_REPLAY
outcome_replayed="$(wait_output "$cursor" 'CONSUMED mode=outcome-replay')"
[[ "$outcome_replayed" == *'rc=0 output=CUSTODY_OK'* ]] || \
  fail "outcome replay control disturbed the first commit: $outcome_replayed"
outcome_refused_after="$(grep -c $'\tEXEC_OUTCOME_REFUSED\t' "$journal" || true)"
[[ "$outcome_refused_after" -gt "$outcome_refused_before" ]] || \
  fail 'duplicate outcome did not produce a kernel refusal'
wait_status 'pending_exec_outcomes=0' >/dev/null

# Crash after the child has ended and the full receipt is durable, but before
# the broker can commit EXEC_OUTCOME. Recovery must materialize INCOMPLETE and
# reject the old generation instead of promoting the observed exit.
cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_OUTCOME_RECOVERY
wait_output "$cursor" 'ISSUED mode=outcome-recovery' >/dev/null
consumed_before="$(grep -c $'\tEXEC_GRANT_CONSUMED\t' "$journal" || true)"
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_OUTCOME_RECOVERY
for _ in $(seq 1 160); do
  consumed_now="$(grep -c $'\tEXEC_GRANT_CONSUMED\t' "$journal" || true)"
  [[ "$consumed_now" -gt "$consumed_before" ]] && break
  sleep 0.025
done
wait_status 'pending_exec_outcomes=1' >/dev/null
generation_before_outcome_crash="$(field kernel_generation "$(status)")"
"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
for _ in $(seq 1 160); do
  [[ "$(sed -n 's/^state=//p' "$descriptor" 2>/dev/null || true)" == recoverable ]] && break
  sleep 0.025
done
"$LOOM" recover --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
outcome_recovered="$(wait_status 'pending_exec_outcomes=0')"
generation_after_outcome_crash="$(field kernel_generation "$outcome_recovered")"
[[ "$generation_after_outcome_crash" != "$generation_before_outcome_crash" ]] || \
  fail 'outcome crash recovery retained the kernel generation'
outcome_recovery_result="$(wait_output "$cursor" 'CONSUMED mode=outcome-recovery')"
[[ "$outcome_recovery_result" == *'rc=126'* && \
   "$outcome_recovery_result" == *'generation-mismatch'* ]] || \
  fail "pre-crash outcome was not refused: $outcome_recovery_result"
grep -q $'\tEXEC_OUTCOME_INCOMPLETE\t' "$journal" || \
  fail 'recovery omitted explicit incomplete outcome'

cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_EXPIRY
wait_output "$cursor" 'ISSUED mode=expiry' >/dev/null
wait_status 'pending_exec_grants=1' >/dev/null
[[ -z "$(find "$LEAKED_CAPABILITY_DIR" -type f -name '*.cap' -print -quit)" ]] || \
  fail 'leaked test environment re-enabled the V1 file issuer'
sleep 2
cursor="$(field output_cursor "$(status)")"
send_harness CONSUME_EXPIRY
expired="$(wait_output "$cursor" 'CONSUMED mode=expiry')"
[[ "$expired" == *'rc=126'* && "$expired" == *'missing-or-replayed'* ]] || \
  fail "expired in-memory grant was not refused: $expired"

cursor="$(field output_cursor "$(status)")"
send_harness ISSUE_RECOVERY
recovery_issue="$(wait_output "$cursor" 'ISSUED mode=recovery')"
old_generation="$(sed -n 's/.* generation=\([0-9a-f]*\) handle=.*/\1/p' \
  <<< "$(grep 'ISSUED mode=recovery' <<< "$recovery_issue" | tail -1)")"
wait_status 'pending_exec_grants=1' >/dev/null

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
for _ in $(seq 1 160); do
  [[ "$(sed -n 's/^state=//p' "$descriptor" 2>/dev/null || true)" == recoverable ]] && break
  sleep 0.05
done
"$LOOM" recover --state-dir "$STATE_DIR" --cwd "$ROOT_DIR" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
recovered="$(wait_status 'pending_exec_grants=0')"
new_generation="$(field kernel_generation "$recovered")"
[[ ${#new_generation} -eq 64 && "$new_generation" != "$old_generation" ]] || \
  fail 'kernel recovery did not revoke the generation'

cursor="$(field output_cursor "$recovered")"
send_harness CONSUME_RECOVERY
recovery_consume="$(wait_output "$cursor" 'CONSUMED mode=recovery')"
[[ "$recovery_consume" == *'rc=126'* && \
   "$recovery_consume" == *'generation-mismatch'* ]] || \
  fail "pre-crash handle survived recovery: $recovery_consume"

grep -q $'\tEXEC_GRANT_ISSUED\t' "$journal" || fail 'journal omitted grant issuance'
grep -q $'\tEXEC_GRANT_CONSUMED\t' "$journal" || fail 'journal omitted grant consumption'
grep -q $'\tEXEC_OUTCOME_RECORDED\t' "$journal" || fail 'journal omitted outcome commit'
grep -q $'\tEXEC_CONSUME_REFUSED\t' "$journal" || fail 'journal omitted refusal'
grep -q $'\tKERNEL_RECOVERED\t' "$journal" || fail 'journal omitted kernel recovery'
if grep -q -e 'CUSTODY_OK' -e '/usr/bin/printf' "$journal"; then
  fail 'kernel journal leaked capability payload or command'
fi

echo "sounio-loom-execution-custody-selftest: PASS custody=kernel-memory peer=SO_PEERCRED pidfd=bound ancestry=exact-harness same_uid_outside_ancestry=refused wrong_cwd=refused+not-burned execute_once=PASS outcome_receipt=durable+hash-bound outcome_replay=refused outcome_policy_missing=refused+incomplete outcome_crash=incomplete+generation-refused replay=refused expiry=refused recovery=revoked no_capability_file=true leaked_test_env=kernel-only foreign_agentd=ignored fixture_flag=explicit journal=digest-only"
