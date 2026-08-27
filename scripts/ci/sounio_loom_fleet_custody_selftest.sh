#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-fleet-custody.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
ADOPT_STATE_DIR="$TEST_ROOT/adopt-state"
KIMI_STATE_DIR="$TEST_ROOT/kimi-state"
WORKTREE="$TEST_ROOT/worktree"
HOME_DIR="$TEST_ROOT/home"
LEGACY_STATE="$TEST_ROOT/legacy"
COORD_DIR="$TEST_ROOT/coord"
ADOPT_COORD_DIR="$TEST_ROOT/adopt-coord"
FAKE_CODEX="$TEST_ROOT/fake-codex"
FAKE_KIMI="$TEST_ROOT/fake-kimi"
FAKE_FLEET_AGENT="$TEST_ROOT/fake-fleet-agent"
LOOM="${SOUNIO_LOOM_BIN:-$ROOT_DIR/tools/loom/_build/default/src/loom.exe}"
AGENT=codex
LANE=catalog-codex
SESSION_ID=44444444-4444-4444-8444-444444444444
ADOPT_LANE=adopted-codex
ADOPT_SESSION=55555555-5555-4555-8555-555555555555
KIMI_AGENT=kimi
KIMI_LANE=catalog-kimi
KIMI_SESSION=77777777-7777-4777-8777-777777777777

fail() {
  printf 'sounio-loom-fleet-custody-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

stop_lane() {
  local state_dir="$1" lane="$2" agent="${3:-$AGENT}"
  loom stop --state-dir "$state_dir" --agent "$agent" --lane "$lane" \
    --cwd "$WORKTREE" >/dev/null 2>&1 || true
}

cleanup() {
  stop_lane "$STATE_DIR" "$LANE"
  stop_lane "$ADOPT_STATE_DIR" "$ADOPT_LANE"
  stop_lane "$KIMI_STATE_DIR" "$KIMI_LANE" "$KIMI_AGENT"
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

[[ -x "$LOOM" ]] || "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
mkdir -p "$WORKTREE" "$HOME_DIR/.codex" "$LEGACY_STATE"
git -C "$WORKTREE" init -q
git -C "$WORKTREE" config user.name 'Loom Fleet Custody Selftest'
git -C "$WORKTREE" config user.email 'loom-fleet-custody@sounio.local'
printf 'seed\n' > "$WORKTREE/README"
git -C "$WORKTREE" add README
git -C "$WORKTREE" commit -qm seed

cat > "$FAKE_CODEX" <<'FAKE_CODEX'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}:${2:-}" in
  --version:)
    printf 'codex-cli loom-fleet-custody-test\n'
    ;;
  login:status)
    printf 'Logged in using loom-fleet-custody-test\n'
    ;;
  --no-alt-screen:*)
    prompt="${!#}"
    printf 'FLEET_CODEX_READY:%s:HOME=%s:COORD=%s:PID=%s\n' \
      "$prompt" "$HOME" "${SOUNIO_COORD_DIR:-missing}" "$$"
    while IFS= read -r wake; do
      printf 'FLEET_CODEX_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
    ;;
  *)
    printf 'unexpected fake Codex invocation: %s\n' "$*" >&2
    exit 42
    ;;
esac
FAKE_CODEX
chmod +x "$FAKE_CODEX"

cat > "$FAKE_KIMI" <<'FAKE_KIMI'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}:${2:-}" in
  --version:)
    printf '0.38.0-loom-fleet-custody-test\n'
    ;;
  :)
    printf 'FLEET_KIMI_READY:HOME=%s:COORD=%s:PID=%s\n' \
      "$HOME" "${SOUNIO_COORD_DIR:-missing}" "$$"
    while IFS= read -r wake; do
      printf 'FLEET_KIMI_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
    ;;
  *)
    printf 'unexpected fake Kimi invocation: %s\n' "$*" >&2
    exit 42
    ;;
esac
FAKE_KIMI
chmod +x "$FAKE_KIMI"

cat > "$FAKE_FLEET_AGENT" <<'FAKE_FLEET'
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
case "$command_name" in
  status)
    if [[ -f "$SOUNIO_FAKE_LEGACY_STATE/$slot.active" ]]; then
      printf 'FLEET_SLOT_STATUS state=active slot=%s\n' "$slot"
    else
      printf 'FLEET_SLOT_STATUS state=absent slot=%s\n' "$slot"
    fi
    printf 'fleet_slots=1 unhealthy=0\n'
    ;;
  launch-kind)
    : > "$SOUNIO_FAKE_LEGACY_STATE/$slot.active"
    printf 'FLEET_SLOT action=started slot=%s\n' "$slot"
    ;;
  *) exit 2 ;;
esac
FAKE_FLEET
chmod +x "$FAKE_FLEET_AGENT"

loom() {
  SOUNIO_LOOM_PROVIDER_CODEX="$FAKE_CODEX" \
    SOUNIO_LOOM_PROVIDER_KIMI="$FAKE_KIMI" \
    SOUNIO_LOOM_FLEET_AGENT_COMMAND="$FAKE_FLEET_AGENT" \
    SOUNIO_FAKE_LEGACY_STATE="$LEGACY_STATE" "$LOOM" "$@"
}

bootstrap_prompt='CATALOG_BOOTSTRAP_PROMPT'
enrolled="$(loom fleet-enroll --state-dir "$STATE_DIR" --slot "$LANE" \
  --kind codex --custody loom --agent "$AGENT" --home "$HOME_DIR" \
  --session-id "$SESSION_ID" --coord-dir "$COORD_DIR" \
  --prompt "$bootstrap_prompt" --cwd "$WORKTREE")"
[[ "$enrolled" == *'custody=loom'* && "$enrolled" == *'adopted=no'* ]] || \
  fail 'catalog did not persist explicit Loom custody'

descriptor="$STATE_DIR/fleet/$LANE.state"
prompt_file="$STATE_DIR/fleet/prompts/$LANE.txt"
grep -q '^version=2$' "$descriptor" || fail 'catalog did not write schema v2'
grep -q '^custody=loom$' "$descriptor" || fail 'catalog omitted Loom custody'
grep -q "^session_id=$SESSION_ID$" "$descriptor" || fail 'catalog omitted stable session identity'
grep -q "^coord_dir=$COORD_DIR$" "$descriptor" || \
  fail 'catalog omitted the shared coordination authority'
[[ "$(stat -c '%a' "$prompt_file")" == 600 ]] || fail 'sealed prompt permissions are not private'
grep -Fq "$bootstrap_prompt" "$prompt_file" || fail 'sealed prompt content changed'
if grep -Fq "$bootstrap_prompt" "$descriptor"; then
  fail 'catalog descriptor leaked the raw bootstrap prompt'
fi

plan="$(loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE")"
[[ "$plan" == *'custody=loom state=absent action=provider-open mode=plan'* ]] || \
  fail 'reconciler did not plan persistent provider custody'

opened="$(loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" --apply)"
[[ "$opened" == *'custody=loom state=active action=opened'* ]] || \
  fail 'reconciler did not open the persistent provider'
status="$(loom status --machine --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$LANE" --cwd "$WORKTREE")"
before_instance="$(sed -n 's/^instance_id=//p' <<< "$status")"
before_kernel="$(sed -n 's/^daemon_pid=//p' <<< "$status")"
before_guardian="$(sed -n 's/^guardian_pid=//p' <<< "$status")"
before_provider="$(sed -n 's/^harness_pid=//p' <<< "$status")"
[[ -n "$before_instance" && -n "$before_kernel" && -n "$before_guardian" && \
  -n "$before_provider" ]] || fail 'opened custody omitted process identities'

repeat="$(loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" --apply)"
[[ "$repeat" == *'custody=loom state=active action=noop'* ]] || \
  fail 'idempotent reconcile opened a duplicate provider'

: > "$LEGACY_STATE/$LANE.active"
if loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" \
  > "$TEST_ROOT/dual-authority.out" 2>&1; then
  fail 'reconciler accepted simultaneous agentd and Loom authority'
fi
grep -q 'fleet-authority-conflict.*desired=loom.*agentd:active' \
  "$TEST_ROOT/dual-authority.out" || \
  fail 'dual authority sabotage was refused by the wrong rule'
rm "$LEGACY_STATE/$LANE.active"

cp "$descriptor" "$TEST_ROOT/catalog.backup"
sed 's/^custody=.*/custody=forged/' "$descriptor" > "$descriptor.tmp"
mv "$descriptor.tmp" "$descriptor"
if loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" \
  > "$TEST_ROOT/forged-custody.out" 2>&1; then
  fail 'catalog accepted forged custody'
fi
grep -q 'unsupported fleet custody forged' "$TEST_ROOT/forged-custody.out" || \
  fail 'forged custody was refused by the wrong rule'
mv "$TEST_ROOT/catalog.backup" "$descriptor"

printf 'tampered prompt\n' > "$prompt_file"
if loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" \
  > "$TEST_ROOT/tampered-prompt.out" 2>&1; then
  fail 'catalog accepted a tampered bootstrap prompt'
fi
grep -q 'fleet prompt digest mismatch' "$TEST_ROOT/tampered-prompt.out" || \
  fail 'prompt sabotage was refused by the wrong rule'
printf '%s' "$bootstrap_prompt" > "$prompt_file"

loom crash-kernel --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --cwd "$WORKTREE" --at now >/dev/null
for _ in $(seq 1 100); do
  if ! loom status --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
    --cwd "$WORKTREE" >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
recover_plan="$(loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE")"
[[ "$recover_plan" == *'custody=loom state=recoverable action=recover mode=plan'* ]] || \
  fail 'catalog did not distinguish recoverable custody from absence'
recovered="$(loom fleet-reconcile --state-dir "$STATE_DIR" --cwd "$WORKTREE" --apply)"
[[ "$recovered" == *'custody=loom state=active action=recovered'* ]] || \
  fail 'catalog did not recover the disposable kernel'
after="$(loom status --machine --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$LANE" --cwd "$WORKTREE")"
[[ "$(sed -n 's/^instance_id=//p' <<< "$after")" == "$before_instance" ]] || \
  fail 'catalog recovery replaced the Loom instance'
[[ "$(sed -n 's/^guardian_pid=//p' <<< "$after")" == "$before_guardian" ]] || \
  fail 'catalog recovery replaced the Guardian'
[[ "$(sed -n 's/^harness_pid=//p' <<< "$after")" == "$before_provider" ]] || \
  fail 'catalog recovery replaced the provider'
[[ "$(sed -n 's/^daemon_pid=//p' <<< "$after")" != "$before_kernel" ]] || \
  fail 'kernel sabotage did not produce a new kernel'

post_recovery='CATALOG_RECOVERY_WITNESS'
loom wake --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --session-id "$SESSION_ID" --message-id fleet-catalog-recovery \
  --prompt "$post_recovery" --cwd "$WORKTREE" >/dev/null
for _ in $(seq 1 100); do
  replay="$(loom snapshot --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
    --cwd "$WORKTREE" --cursor 0 2>/dev/null || true)"
  grep -q "FLEET_CODEX_WAKE:$post_recovery" <<< "$replay" && break
  sleep 0.05
done
grep -q "FLEET_CODEX_READY:$bootstrap_prompt:HOME=$HOME_DIR:COORD=$COORD_DIR" \
  <<< "$replay" || \
  fail 'provider did not inherit enrolled credential and coordination authorities'
grep -q "FLEET_CODEX_WAKE:$post_recovery" <<< "$replay" || \
  fail 'recovered catalog lane did not accept a second input'

manual_prompt='ACTIVE_ADOPTION_BOOTSTRAP'
loom provider-open --provider codex --state-dir "$ADOPT_STATE_DIR" \
  --agent "$AGENT" --lane "$ADOPT_LANE" --session-id "$ADOPT_SESSION" \
  --cwd "$WORKTREE" --prompt "$manual_prompt" >/dev/null
if loom fleet-enroll --state-dir "$ADOPT_STATE_DIR" --slot "$ADOPT_LANE" \
  --kind codex --custody agentd --agent "$AGENT" --home "$HOME_DIR" \
  --cwd "$WORKTREE" > "$TEST_ROOT/reverse-authority.out" 2>&1; then
  fail 'agentd desired state concealed an active Loom authority'
fi
grep -q 'fleet-authority-conflict.*desired=agentd.*loom:active' \
  "$TEST_ROOT/reverse-authority.out" || \
  fail 'reverse dual authority sabotage was refused by the wrong rule'
if loom fleet-enroll --state-dir "$ADOPT_STATE_DIR" --slot "$ADOPT_LANE" \
  --kind codex --custody loom --agent "$AGENT" --home "$HOME_DIR" \
  --session-id "$ADOPT_SESSION" --coord-dir "$ADOPT_COORD_DIR" \
  --prompt "$manual_prompt" --cwd "$WORKTREE" \
  > "$TEST_ROOT/unapproved-adoption.out" 2>&1; then
  fail 'catalog silently adopted an already-active Loom lane'
fi
grep -q 'active Loom lane requires --adopt-active' \
  "$TEST_ROOT/unapproved-adoption.out" || \
  fail 'unapproved adoption was refused by the wrong rule'
if loom fleet-enroll --state-dir "$ADOPT_STATE_DIR" --slot "$ADOPT_LANE" \
  --kind codex --custody loom --agent "$AGENT" --home "$HOME_DIR" \
  --session-id 66666666-6666-4666-8666-666666666666 \
  --coord-dir "$ADOPT_COORD_DIR" --prompt "$manual_prompt" \
  --cwd "$WORKTREE" --adopt-active \
  > "$TEST_ROOT/identity-drift.out" 2>&1; then
  fail 'active adoption accepted a forged session identity'
fi
grep -q 'fleet Loom identity drift.*field=session_id' \
  "$TEST_ROOT/identity-drift.out" || \
  fail 'forged adoption identity was refused by the wrong rule'
adopted="$(loom fleet-enroll --state-dir "$ADOPT_STATE_DIR" --slot "$ADOPT_LANE" \
  --kind codex --custody loom --agent "$AGENT" --home "$HOME_DIR" \
  --session-id "$ADOPT_SESSION" --coord-dir "$ADOPT_COORD_DIR" \
  --prompt "$manual_prompt" --cwd "$WORKTREE" \
  --adopt-active)"
[[ "$adopted" == *'adopted=active'* ]] || \
  fail 'explicit active adoption did not publish its receipt'
adopt_plan="$(loom fleet-reconcile --state-dir "$ADOPT_STATE_DIR" --cwd "$WORKTREE")"
[[ "$adopt_plan" == *'custody=loom state=active action=noop'* ]] || \
  fail 'adopted lane did not reconcile idempotently'

kimi_prompt='KIMI_CATALOG_BOOTSTRAP_PROMPT'
kimi_enrolled="$(loom fleet-enroll --state-dir "$KIMI_STATE_DIR" \
  --slot "$KIMI_LANE" --kind kimi --custody loom --agent "$KIMI_AGENT" \
  --home "$HOME_DIR" --session-id "$KIMI_SESSION" --coord-dir "$COORD_DIR" \
  --prompt "$kimi_prompt" --cwd "$WORKTREE")"
[[ "$kimi_enrolled" == *'kind=kimi custody=loom'* ]] || \
  fail 'catalog did not admit verified persistent Kimi custody'
kimi_descriptor="$KIMI_STATE_DIR/fleet/$KIMI_LANE.state"
if loom fleet-enroll --state-dir "$KIMI_STATE_DIR" \
  --slot kimi-native-store-alias --kind kimi --custody loom \
  --agent kimi-native-store-alias --home "$HOME_DIR" \
  --session-id 99999999-9999-4999-8999-999999999999 \
  --coord-dir "$COORD_DIR" --prompt "$kimi_prompt" --cwd "$WORKTREE" \
  > "$TEST_ROOT/kimi-home-alias.out" 2>&1; then
  fail 'catalog admitted two native-store Kimi lanes with one HOME'
fi
grep -q 'fleet-native-store-home-conflict provider=kimi .*existing_slot=catalog-kimi requested_slot=kimi-native-store-alias' \
  "$TEST_ROOT/kimi-home-alias.out" || \
  fail 'same-HOME native-store alias was refused by an unrelated rule'

kimi_plan="$(loom fleet-reconcile --state-dir "$KIMI_STATE_DIR" --cwd "$WORKTREE")"
[[ "$kimi_plan" == *'custody=loom state=absent action=provider-open mode=plan'* ]] || \
  fail 'catalog did not plan persistent Kimi custody'
kimi_opened="$(loom fleet-reconcile --state-dir "$KIMI_STATE_DIR" \
  --cwd "$WORKTREE" --apply)"
[[ "$kimi_opened" == *'custody=loom state=active action=opened'* ]] || \
  fail 'catalog did not open persistent Kimi custody'

kimi_status="$(loom status --machine --state-dir "$KIMI_STATE_DIR" \
  --agent "$KIMI_AGENT" --lane "$KIMI_LANE" --cwd "$WORKTREE")"
kimi_before_instance="$(sed -n 's/^instance_id=//p' <<< "$kimi_status")"
kimi_before_kernel="$(sed -n 's/^daemon_pid=//p' <<< "$kimi_status")"
kimi_before_guardian="$(sed -n 's/^guardian_pid=//p' <<< "$kimi_status")"
kimi_before_provider="$(sed -n 's/^harness_pid=//p' <<< "$kimi_status")"
grep -q '^command=fake-kimi$' \
  "$KIMI_STATE_DIR/sessions/$KIMI_AGENT--$KIMI_LANE/session.state" || \
  fail 'catalog obscured the native Kimi process identity'

kimi_replay=''
for _ in $(seq 1 100); do
  kimi_replay="$(loom snapshot --state-dir "$KIMI_STATE_DIR" \
    --agent "$KIMI_AGENT" --lane "$KIMI_LANE" --cwd "$WORKTREE" \
    --cursor 0 2>/dev/null || true)"
  grep -q "FLEET_KIMI_WAKE:$kimi_prompt" <<< "$kimi_replay" && break
  sleep 0.05
done
grep -q "FLEET_KIMI_READY:HOME=$HOME_DIR:COORD=$COORD_DIR" \
  <<< "$kimi_replay" || fail 'Kimi did not inherit enrolled authorities'
grep -q "FLEET_KIMI_WAKE:$kimi_prompt" <<< "$kimi_replay" || \
  fail 'catalog bootstrap did not traverse the Kimi input lease'

loom crash-kernel --state-dir "$KIMI_STATE_DIR" --agent "$KIMI_AGENT" \
  --lane "$KIMI_LANE" --cwd "$WORKTREE" --at now >/dev/null
for _ in $(seq 1 100); do
  if ! loom status --state-dir "$KIMI_STATE_DIR" --agent "$KIMI_AGENT" \
    --lane "$KIMI_LANE" --cwd "$WORKTREE" >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
kimi_recover_plan="$(loom fleet-reconcile --state-dir "$KIMI_STATE_DIR" \
  --cwd "$WORKTREE")"
[[ "$kimi_recover_plan" == *'custody=loom state=recoverable action=recover mode=plan'* ]] || \
  fail 'catalog did not classify Kimi kernel loss as recoverable'
kimi_recovered="$(loom fleet-reconcile --state-dir "$KIMI_STATE_DIR" \
  --cwd "$WORKTREE" --apply)"
[[ "$kimi_recovered" == *'custody=loom state=active action=recovered'* ]] || \
  fail 'catalog did not recover Kimi custody'
kimi_after="$(loom status --machine --state-dir "$KIMI_STATE_DIR" \
  --agent "$KIMI_AGENT" --lane "$KIMI_LANE" --cwd "$WORKTREE")"
[[ "$(sed -n 's/^instance_id=//p' <<< "$kimi_after")" == "$kimi_before_instance" ]] || \
  fail 'Kimi catalog recovery replaced the Loom instance'
[[ "$(sed -n 's/^guardian_pid=//p' <<< "$kimi_after")" == "$kimi_before_guardian" ]] || \
  fail 'Kimi catalog recovery replaced the Guardian'
[[ "$(sed -n 's/^harness_pid=//p' <<< "$kimi_after")" == "$kimi_before_provider" ]] || \
  fail 'Kimi catalog recovery replaced the provider process'
[[ "$(sed -n 's/^daemon_pid=//p' <<< "$kimi_after")" != "$kimi_before_kernel" ]] || \
  fail 'Kimi kernel sabotage did not produce a new kernel'

cp "$kimi_descriptor" "$TEST_ROOT/kimi-catalog.backup"
sed 's/^kind=.*/kind=cursor/' "$kimi_descriptor" > "$kimi_descriptor.tmp"
mv "$kimi_descriptor.tmp" "$kimi_descriptor"
if loom fleet-reconcile --state-dir "$KIMI_STATE_DIR" --cwd "$WORKTREE" \
  > "$TEST_ROOT/unverified-persistent-provider.out" 2>&1; then
  fail 'catalog accepted a provider without a verified persistent adapter'
fi
grep -q 'persistent fleet provider unavailable for kind cursor' \
  "$TEST_ROOT/unverified-persistent-provider.out" || \
  fail 'persistent-provider sabotage was refused by the wrong rule'
mv "$TEST_ROOT/kimi-catalog.backup" "$kimi_descriptor"

stop_lane "$STATE_DIR" "$LANE"
stop_lane "$ADOPT_STATE_DIR" "$ADOPT_LANE"
stop_lane "$KIMI_STATE_DIR" "$KIMI_LANE" "$KIMI_AGENT"
printf 'sounio-loom-fleet-custody-selftest: PASS catalog=v2 custody=typed providers=codex,kimi prompt=sealed prompt_transport=loom-wake native_store_home=isolated dual_authority=refused unsupported_persistent=refused adoption=explicit kernel_recovery=stable-provider\n'
