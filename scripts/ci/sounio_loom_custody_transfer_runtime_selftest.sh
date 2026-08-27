#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-custody-runtime.XXXXXX")"
WORKTREE="$TEST_ROOT/worktree"
HOME_DIR="$TEST_ROOT/home"
COORD_DIR="$TEST_ROOT/coord"
LEGACY_STATE="$TEST_ROOT/legacy"
LOOM="$TEST_ROOT/bin/sounio-loom-runtime"
POLICY="$TEST_ROOT/bin/sounio-loom-custody-transfer-runtime"
COORD="$TEST_ROOT/bin/sounio-coord-runtime"
FLEET_AGENT="$TEST_ROOT/bin/sounio-fleet-agent-runtime"
CLAUDE="$TEST_ROOT/bin/claude"
PROVIDER_SESSION='99999999-9999-4999-8999-999999999999'

fail() {
  printf 'sounio-loom-custody-transfer-runtime-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  for session in "$TEST_ROOT"/sessions/*; do
    [[ -f "$session" ]] || continue
    IFS=$'\t' read -r state_dir slot loom_session < "$session"
    SOUNIO_LOOM_PROVIDER_CLAUDE="$CLAUDE" SOUNIO_COORD_DIR="$COORD_DIR" \
      "$LOOM" stop --state-dir "$state_dir" --agent claude --lane "$slot" \
      --session-id "$loom_session" --cwd "$WORKTREE" >/dev/null 2>&1 || true
  done
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$WORKTREE" "$HOME_DIR/.claude" "$COORD_DIR" "$LEGACY_STATE" \
  "$TEST_ROOT/bin" "$TEST_ROOT/sessions"
git -C "$WORKTREE" init -q
git -C "$WORKTREE" config user.name 'Loom Custody Transfer Selftest'
git -C "$WORKTREE" config user.email 'loom-custody-transfer@sounio.local'
printf 'seed\n' > "$WORKTREE/README"
git -C "$WORKTREE" add README
git -C "$WORKTREE" commit -qm seed

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
cp "$ROOT_DIR/tools/loom/_build/default/src/loom.exe" "$LOOM"
cp "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-custody-transfer-runtime" \
  "$POLICY"
cp "$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" "$COORD"
chmod 0755 "$LOOM" "$POLICY" "$COORD"

cat > "$CLAUDE" <<'FAKE_CLAUDE'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}:${2:-}" in
  --version:)
    printf 'Claude Code custody-transfer-test\n'
    ;;
  auth:status)
    printf '{"loggedIn":true,"authMethod":"subscription"}\n'
    ;;
  --resume:*)
    if [[ "${SOUNIO_FAKE_TARGET_FAIL:-0}" == 1 ]]; then
      printf 'FAKE_CLAUDE_TARGET_FAILURE\n' >&2
      exit 55
    fi
    if [[ "${SOUNIO_FAKE_DUAL_AUTHORITY:-0}" == 1 ]]; then
      : > "$SOUNIO_FAKE_LEGACY_STATE/$SOUNIO_FAKE_TRANSFER_SLOT.active"
    fi
    printf 'FAKE_CLAUDE_READY:%s\n' "$*"
    while IFS= read -r wake; do
      printf 'FAKE_CLAUDE_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
    ;;
  *)
    printf 'unexpected fake Claude invocation: %s\n' "$*" >&2
    exit 42
    ;;
esac
FAKE_CLAUDE
chmod 0755 "$CLAUDE"

cat > "$FLEET_AGENT" <<'FAKE_FLEET'
#!/usr/bin/env bash
set -euo pipefail
command_name="${1:-}"
shift || true
slot=''
while (($#)); do
  case "$1" in
    --slot) slot="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$slot" ]] || exit 2
lane="source-$slot"
session='99999999-9999-4999-8999-999999999999'
case "$command_name" in
  plan-kind)
    printf 'agent=claude\nlane=%s\nsession_id=%s\nidentity=exact\n' \
      "$lane" "$session"
    ;;
  status)
    if [[ -f "$SOUNIO_FAKE_LEGACY_STATE/$slot.active" ]]; then
      observed_session="$session"
      [[ "${SOUNIO_FAKE_SOURCE_DRIFT:-0}" == 1 ]] && \
        observed_session='aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
      printf 'FLEET_SLOT_STATUS state=active slot=%s agent=claude lane=%s session_id=%s identity=exact instance_id=source-instance argv_digest=source-digest harness_pid=123 worktree=%s\n' \
        "$slot" "$lane" "$observed_session" "$SOUNIO_FAKE_WORKTREE"
      printf 'fleet_slots=1 unhealthy=0\n'
    else
      printf 'fleet_slots=0 unhealthy=0\n'
    fi
    ;;
  stop)
    previous=absent
    [[ -f "$SOUNIO_FAKE_LEGACY_STATE/$slot.active" ]] && previous=active
    rm -f "$SOUNIO_FAKE_LEGACY_STATE/$slot.active"
    [[ "${SOUNIO_FAKE_STOP_FORGED:-0}" == 1 ]] && previous=forged
    [[ "${SOUNIO_FAKE_STOP_SILENT:-0}" == 1 ]] || \
      printf 'FLEET_SLOT_STOPPED slot=%s state=%s\n' "$slot" "$previous"
    ;;
  launch|launch-kind)
    [[ "${SOUNIO_FAKE_ROLLBACK_FAIL:-0}" == 1 ]] && exit 57
    : > "$SOUNIO_FAKE_LEGACY_STATE/$slot.active"
    printf 'FLEET_SLOT action=started slot=%s agent=claude lane=%s session_id=%s identity=exact\n' \
      "$slot" "$lane" "$session"
    ;;
  *) exit 2 ;;
esac
FAKE_FLEET
chmod 0755 "$FLEET_AGENT"

loom() {
  SOUNIO_LOOM_PROVIDER_CLAUDE="$CLAUDE" \
  SOUNIO_LOOM_FLEET_AGENT_COMMAND="$FLEET_AGENT" \
  SOUNIO_LOOM_CUSTODY_TRANSFER_COMMAND="${SOUNIO_LOOM_CUSTODY_TRANSFER_COMMAND_OVERRIDE:-$POLICY}" \
  SOUNIO_COORD_DIR="$COORD_DIR" \
  SOUNIO_FAKE_LEGACY_STATE="$LEGACY_STATE" \
  SOUNIO_FAKE_WORKTREE="$WORKTREE" \
    "$LOOM" "$@"
}

session_for() {
  local ordinal="$1"
  printf '%08d-1111-4111-8111-%012d' "$ordinal" "$ordinal"
}

prepare_source() {
  local state_dir="$1" slot="$2"
  mkdir -p "$state_dir"
  loom fleet-enroll --state-dir "$state_dir" --slot "$slot" \
    --kind claude --custody agentd --agent claude --home "$HOME_DIR" \
    --cwd "$WORKTREE" >/dev/null
  : > "$LEGACY_STATE/$slot.active"
}

record_session() {
  local state_dir="$1" slot="$2" loom_session="$3"
  printf '%s\t%s\t%s\n' "$state_dir" "$slot" "$loom_session" > \
    "$TEST_ROOT/sessions/$slot"
}

transfer_args() {
  local state_dir="$1" slot="$2" loom_session="$3"
  printf '%s\n' \
    --state-dir "$state_dir" --slot "$slot" --session-id "$loom_session" \
    --provider-session "$PROVIDER_SESSION" --source-agent claude \
    --source-lane "source-$slot" --source-session "$PROVIDER_SESSION" \
    --coord-dir "$COORD_DIR" --prompt "TRANSFER_BOOTSTRAP_$slot" \
    --deadline-seconds 5 --cwd "$WORKTREE"
}

run_transfer() {
  local state_dir="$1" slot="$2" loom_session="$3"
  mapfile -t args < <(transfer_args "$state_dir" "$slot" "$loom_session")
  SOUNIO_FAKE_TRANSFER_SLOT="$slot" loom fleet-transfer "${args[@]}"
}

recover_transfer() {
  local state_dir="$1" slot="$2"
  SOUNIO_FAKE_TRANSFER_SLOT="$slot" loom fleet-transfer-recover \
    --state-dir "$state_dir" --slot "$slot" --deadline-seconds 5 \
    --cwd "$WORKTREE"
}

assert_agentd() {
  local state_dir="$1" slot="$2"
  grep -q '^custody=agentd$' "$state_dir/fleet/$slot.state" || \
    fail "$slot lost agentd catalog authority"
  [[ -f "$LEGACY_STATE/$slot.active" ]] || fail "$slot source was not active"
}

assert_loom() {
  local state_dir="$1" slot="$2"
  grep -q '^custody=loom$' "$state_dir/fleet/$slot.state" || \
    fail "$slot did not commit Loom catalog authority"
  grep -q '^provider_mode=resume$' "$state_dir/fleet/$slot.state" || \
    fail "$slot lost resume mode"
  grep -q "^provider_session=$PROVIDER_SESSION$" \
    "$state_dir/fleet/$slot.state" || fail "$slot lost provider session"
  [[ ! -f "$LEGACY_STATE/$slot.active" ]] || \
    fail "$slot retained dual source authority"
}

happy_state="$TEST_ROOT/happy-state"
happy_slot='transfer-happy'
happy_session="$(session_for 1)"
prepare_source "$happy_state" "$happy_slot"
record_session "$happy_state" "$happy_slot" "$happy_session"
happy_output="$(run_transfer "$happy_state" "$happy_slot" "$happy_session")"
[[ "$happy_output" == *'state=COMPLETE'* && "$happy_output" == *'authority=Sounio'* ]] || \
  fail 'happy transfer did not complete under Sounio authority'
assert_loom "$happy_state" "$happy_slot"

failure_state="$TEST_ROOT/failure-state"
failure_slot='transfer-target-failure'
failure_session="$(session_for 2)"
prepare_source "$failure_state" "$failure_slot"
record_session "$failure_state" "$failure_slot" "$failure_session"
mapfile -t failure_args < <(transfer_args "$failure_state" "$failure_slot" "$failure_session")
if SOUNIO_FAKE_TARGET_FAIL=1 SOUNIO_FAKE_TRANSFER_SLOT="$failure_slot" \
  loom fleet-transfer "${failure_args[@]}" > "$TEST_ROOT/target-failure.out" 2>&1; then
  fail 'target failure unexpectedly committed'
fi
grep -q 'state=ROLLED_BACK' "$TEST_ROOT/target-failure.out" || \
  fail 'target failure omitted rollback receipt'
assert_agentd "$failure_state" "$failure_slot"

dual_state="$TEST_ROOT/dual-state"
dual_slot='transfer-dual-authority'
dual_session="$(session_for 3)"
prepare_source "$dual_state" "$dual_slot"
record_session "$dual_state" "$dual_slot" "$dual_session"
mapfile -t dual_args < <(transfer_args "$dual_state" "$dual_slot" "$dual_session")
dual_output="$(SOUNIO_FAKE_DUAL_AUTHORITY=1 \
  SOUNIO_FAKE_TRANSFER_SLOT="$dual_slot" loom fleet-transfer "${dual_args[@]}")"
[[ "$dual_output" == *'state=ROLLED_BACK'* ]] || \
  fail 'dual authority did not abort the provisional target'
assert_agentd "$dual_state" "$dual_slot"

drift_state="$TEST_ROOT/drift-state"
drift_slot='transfer-source-drift'
drift_session="$(session_for 4)"
prepare_source "$drift_state" "$drift_slot"
mapfile -t drift_args < <(transfer_args "$drift_state" "$drift_slot" "$drift_session")
if SOUNIO_FAKE_SOURCE_DRIFT=1 SOUNIO_FAKE_TRANSFER_SLOT="$drift_slot" \
  loom fleet-transfer "${drift_args[@]}" > "$TEST_ROOT/source-drift.out" 2>&1; then
  fail 'source identity drift was accepted'
fi
grep -q 'source-identity-drift' "$TEST_ROOT/source-drift.out" || \
  fail 'source identity drift was refused by the wrong rule'
assert_agentd "$drift_state" "$drift_slot"

silent_state="$TEST_ROOT/silent-state"
silent_slot='transfer-silent-stop'
silent_session="$(session_for 5)"
prepare_source "$silent_state" "$silent_slot"
mapfile -t silent_args < <(transfer_args "$silent_state" "$silent_slot" "$silent_session")
if SOUNIO_FAKE_STOP_SILENT=1 SOUNIO_FAKE_TRANSFER_SLOT="$silent_slot" \
  loom fleet-transfer "${silent_args[@]}" > "$TEST_ROOT/silent-stop.out" 2>&1; then
  fail 'silent stop was accepted as positive quiescence'
fi
grep -q 'source-stop-unproved' "$TEST_ROOT/silent-stop.out" || \
  fail 'silent stop was refused by the wrong rule'
grep -q '^phase=1$' \
  "$silent_state/fleet/transfers/$silent_slot/transfer.state" || \
  fail 'silent stop advanced the durable transfer phase'
: > "$LEGACY_STATE/$silent_slot.active"

forged_state="$TEST_ROOT/forged-state"
forged_slot='transfer-forged-stop'
forged_session="$(session_for 8)"
prepare_source "$forged_state" "$forged_slot"
mapfile -t forged_args < <(
  transfer_args "$forged_state" "$forged_slot" "$forged_session"
)
if SOUNIO_FAKE_STOP_FORGED=1 SOUNIO_FAKE_TRANSFER_SLOT="$forged_slot" \
  loom fleet-transfer "${forged_args[@]}" \
  > "$TEST_ROOT/forged-stop.out" 2>&1; then
  fail 'forged stop state was accepted as a quiescence receipt'
fi
grep -q 'source-stop-unproved' "$TEST_ROOT/forged-stop.out" || \
  fail 'forged stop state was refused by the wrong rule'
grep -q '^phase=1$' \
  "$forged_state/fleet/transfers/$forged_slot/transfer.state" || \
  fail 'forged stop state advanced the durable transfer phase'
: > "$LEGACY_STATE/$forged_slot.active"

python_state="$TEST_ROOT/python-state"
python_slot='transfer-python-oracle'
python_session="$(session_for 6)"
prepare_source "$python_state" "$python_slot"
mapfile -t python_args < <(transfer_args "$python_state" "$python_slot" "$python_session")
if SOUNIO_LOOM_CUSTODY_TRANSFER_COMMAND_OVERRIDE=/usr/bin/python3 \
  SOUNIO_FAKE_TRANSFER_SLOT="$python_slot" loom fleet-transfer "${python_args[@]}" \
  > "$TEST_ROOT/python-oracle.out" 2>&1; then
  fail 'Python oracle was accepted as Sounio semantic authority'
fi
grep -q 'custody-transfer-policy-digest-mismatch' "$TEST_ROOT/python-oracle.out" || \
  fail 'Python oracle was refused by the wrong rule'
assert_agentd "$python_state" "$python_slot"

reappear_state="$TEST_ROOT/reappear-state"
reappear_slot='transfer-source-reappears'
reappear_session="$(session_for 7)"
prepare_source "$reappear_state" "$reappear_slot"
mapfile -t reappear_args < <(
  transfer_args "$reappear_state" "$reappear_slot" "$reappear_session"
)
if SOUNIO_LOOM_TRANSFER_CRASH_AT=after-quiesce \
  SOUNIO_FAKE_TRANSFER_SLOT="$reappear_slot" \
  loom fleet-transfer "${reappear_args[@]}" \
  > "$TEST_ROOT/source-reappears-stage.out" 2>&1; then
  fail 'source-reappearance fixture did not stop after quiescence'
fi
: > "$LEGACY_STATE/$reappear_slot.active"
reappear_output="$(recover_transfer "$reappear_state" "$reappear_slot")"
[[ "$reappear_output" == *'state=ROLLED_BACK'* ]] || \
  fail 'source reappearance before target did not execute Sounio rollback'
assert_agentd "$reappear_state" "$reappear_slot"

ordinal=10
for crash_point in after-stage after-quiesce after-target after-commit; do
  slot="transfer-crash-$crash_point"
  state_dir="$TEST_ROOT/$slot-state"
  loom_session="$(session_for "$ordinal")"
  ordinal=$((ordinal + 1))
  prepare_source "$state_dir" "$slot"
  record_session "$state_dir" "$slot" "$loom_session"
  mapfile -t crash_args < <(transfer_args "$state_dir" "$slot" "$loom_session")
  if SOUNIO_LOOM_TRANSFER_CRASH_AT="$crash_point" \
    SOUNIO_FAKE_TRANSFER_SLOT="$slot" loom fleet-transfer "${crash_args[@]}" \
    > "$TEST_ROOT/$slot.out" 2>&1; then
    fail "$crash_point injection unexpectedly completed"
  fi
  grep -q "crash-injected:$crash_point" "$TEST_ROOT/$slot.out" || \
    fail "$crash_point did not stop at the requested boundary"
  recovered="$(recover_transfer "$state_dir" "$slot")"
  [[ "$recovered" == *'state=COMPLETE'* ]] || \
    fail "$crash_point did not recover to committed Loom custody"
  assert_loom "$state_dir" "$slot"
done

printf '%s\n' \
  'sounio-loom-custody-transfer-runtime-selftest: PASS authority=Sounio realization=OCaml happy=committed target_failure=rolled-back dual_authority=abort-target source_drift=refused silent_stop=refused forged_stop=refused python_oracle=refused source_reappears=rolled-back crash_points=after-stage,after-quiesce,after-target,after-commit'
