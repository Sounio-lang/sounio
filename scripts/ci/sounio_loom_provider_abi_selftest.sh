#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/loom"
export SOUNIO_COORD_RUNTIME_MODE=local
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-provider-abi.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
SESSION_ID='11111111-1111-4111-8111-111111111111'
AGENT='provider-abi-test'
LANE='codex-headless'
PERSISTENT_LANE='codex-persistent'

fail() {
  printf 'sounio-loom-provider-abi-selftest: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$PERSISTENT_LANE" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

for command in jq sha256sum; do
  command -v "$command" >/dev/null 2>&1 || fail "required command is missing: $command"
done

cat > "$TEST_ROOT/fake-provider" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail

name="$(basename "$0")"
case "$name:${1:-}:${2:-}" in
  fake-codex:--version:)
    printf 'codex-cli provider-abi-test\n'
    ;;
  fake-codex:login:status)
    printf 'Logged in using provider-abi-test\n'
    ;;
  fake-codex:login:)
    printf 'FAKE_LOGIN provider=codex\n'
    ;;
  fake-codex:exec:*)
    if [[ -n "${CODEX_SESSION_ID+x}${CODEX_THREAD_ID+x}${CODEX_CI+x}${CLAUDECODE+x}${CLAUDE_CODE_ENTRYPOINT+x}${CLAUDE_CODE_SESSION_ID+x}${TMUX+x}${TMUX_PANE+x}${TMUX_TMPDIR+x}" ]]; then
      printf 'parent harness identity leaked into provider process\n' >&2
      exit 40
    fi
    prompt="${!#}"
    if IFS= read -r -t 0.1 _unexpected; then
      printf 'stdin was not closed\n' >&2
      exit 41
    fi
    printf '{"type":"thread.started","thread_id":"provider-abi-thread"}\n'
    printf 'FAKE_CODEX_OUTPUT:%s\n' "$prompt"
    sleep 1
    ;;
  fake-codex:--no-alt-screen:*)
    if [[ -n "${CODEX_SESSION_ID+x}${CODEX_THREAD_ID+x}${CODEX_CI+x}${CLAUDECODE+x}${CLAUDE_CODE_ENTRYPOINT+x}${CLAUDE_CODE_SESSION_ID+x}${TMUX+x}${TMUX_PANE+x}${TMUX_TMPDIR+x}" ]]; then
      printf 'parent harness identity leaked into persistent provider process\n' >&2
      exit 43
    fi
    prompt="${!#}"
    printf 'FAKE_CODEX_TUI_READY:%s\n' "$prompt"
    while IFS= read -r wake; do
      printf 'FAKE_CODEX_TUI_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
    ;;
  fake-claude:--version:)
    printf 'Claude Code provider-abi-test\n'
    ;;
  fake-claude:auth:status)
    printf '{"loggedIn":false,"authMethod":"none"}\n'
    exit 1
    ;;
  fake-claude:auth:login)
    printf 'FAKE_LOGIN provider=claude\n'
    ;;
  fake-grok:--version:)
    printf 'grok provider-abi-test\n'
    ;;
  fake-grok:login:)
    printf 'FAKE_LOGIN provider=grok\n'
    ;;
  fake-opencode:--version:)
    printf 'opencode provider-abi-test\n'
    ;;
  fake-opencode:providers:list)
    printf 'Credentials delegated to provider store\n'
    ;;
  fake-opencode:providers:login)
    printf 'FAKE_LOGIN provider=opencode\n'
    ;;
  *)
    printf 'unexpected fake provider invocation: %s\n' "$*" >&2
    exit 42
    ;;
esac
FAKE
chmod +x "$TEST_ROOT/fake-provider"
for provider in codex claude grok opencode; do
  cp "$TEST_ROOT/fake-provider" "$TEST_ROOT/fake-$provider"
done

export SOUNIO_LOOM_PROVIDER_CODEX="$TEST_ROOT/fake-codex"
export SOUNIO_LOOM_PROVIDER_CLAUDE="$TEST_ROOT/fake-claude"
export SOUNIO_LOOM_PROVIDER_GROK="$TEST_ROOT/fake-grok"
export SOUNIO_LOOM_PROVIDER_OPENCODE="$TEST_ROOT/fake-opencode"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
version="$($LOOM runtime-version)"
grep -q '^runtime_version=2026.08.26.15$' <<< "$version" || \
  fail 'public loom launcher selected the wrong runtime'

providers="$($LOOM provider-list --json)"
jq -e '.schema == "loom-provider-abi-v1" and (.providers | length == 4)' \
  <<< "$providers" >/dev/null || fail 'provider catalog schema or cardinality changed'
jq -e '.providers[] | select(.provider == "codex") |
  .installed == true and .auth == "authenticated" and
  .credential_authority == "native" and .session_binding == "stream-observed"' \
  <<< "$providers" >/dev/null || fail 'Codex provider status was not normalized'
jq -e '.providers[] | select(.provider == "claude") |
  .auth == "unauthenticated" and .session_binding == "caller"' \
  <<< "$providers" >/dev/null || \
  fail 'Claude nonzero auth-status JSON was not classified as unauthenticated'
jq -e '.providers[] | select(.provider == "grok") |
  .auth == "unknown" and .auth_reason == "native-cli-has-no-offline-auth-status"' \
  <<< "$providers" >/dev/null || fail 'Grok auth uncertainty was laundered'
jq -e '.providers[] | select(.provider == "opencode") |
  .auth == "delegated" and .session_binding == "stream-observed"' \
  <<< "$providers" >/dev/null || fail 'OpenCode delegated auth was not preserved'

secret='PROVIDER_ABI_SECRET_PROMPT'
secret_sha="$(printf '%s' "$secret" | sha256sum | awk '{print $1}')"
plan="$($LOOM provider-plan --provider codex --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --model provider-test --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$plan"; then
  fail 'provider plan disclosed the raw prompt'
fi
jq -e --arg digest "$secret_sha" '
  .schema == "loom-provider-abi-v1" and .provider == "codex" and
  .lifecycle == "turn" and .stdin_authority == "closed" and
  .prompt_sha256 == $digest and .prompt_bytes == 26 and
  .unsafe_auto == false and .context_isolation == false and
  (.argv | index("--dangerously-bypass-approvals-and-sandbox") == null) and
  (.argv | index("--ephemeral") == null)' \
  <<< "$plan" >/dev/null || fail 'safe Codex plan has the wrong custody fields'

persistent_plan="$($LOOM provider-plan --provider codex --lifecycle persistent \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$persistent_plan"; then
  fail 'persistent provider plan disclosed the raw prompt'
fi
jq -e '.lifecycle == "persistent" and .stdin_authority == "loom-lease" and
  .context_isolation == false and (.argv | index("--no-alt-screen") != null) and
  (.argv | index("exec") == null)' <<< "$persistent_plan" >/dev/null || \
  fail 'persistent Codex plan has the wrong input authority'

if "$LOOM" provider-plan --provider codex --lifecycle persistent \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt test --isolate-context \
  > "$TEST_ROOT/persistent-isolation.out" 2> "$TEST_ROOT/persistent-isolation.err"; then
  fail 'persistent provider silently accepted headless context isolation'
fi
grep -q 'persistent-context-isolation-unavailable:codex' \
  "$TEST_ROOT/persistent-isolation.err" || \
  fail 'persistent context isolation was refused by the wrong rule'

if "$LOOM" provider-plan --provider claude --lifecycle persistent \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt test \
  > "$TEST_ROOT/persistent-claude.out" 2> "$TEST_ROOT/persistent-claude.err"; then
  fail 'persistent provider accepted an unimplemented Claude adapter'
fi
grep -q 'persistent-provider-unavailable:claude:new' \
  "$TEST_ROOT/persistent-claude.err" || \
  fail 'unimplemented persistent provider was refused by the wrong rule'

if "$LOOM" provider-start --provider codex --lifecycle persistent \
  --state-dir "$STATE_DIR" --agent "$AGENT" --lane refused-start \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt test \
  > "$TEST_ROOT/persistent-start.out" 2> "$TEST_ROOT/persistent-start.err"; then
  fail 'provider-start crossed into persistent lifecycle'
fi
grep -q 'provider-start-requires-turn-lifecycle' "$TEST_ROOT/persistent-start.err" || \
  fail 'provider-start lifecycle sabotage was refused by the wrong rule'

if "$LOOM" provider-open --provider codex --lifecycle turn \
  --state-dir "$STATE_DIR" --agent "$AGENT" --lane refused-open \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt test \
  > "$TEST_ROOT/turn-open.out" 2> "$TEST_ROOT/turn-open.err"; then
  fail 'provider-open crossed into turn lifecycle'
fi
grep -q 'provider-open-requires-persistent-lifecycle' "$TEST_ROOT/turn-open.err" || \
  fail 'provider-open lifecycle sabotage was refused by the wrong rule'

unsafe_plan="$($LOOM provider-plan --provider codex --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$secret" --unsafe-auto --json)"
jq -e '.unsafe_auto == true and
  (.argv | index("--dangerously-bypass-approvals-and-sandbox") != null)' \
  <<< "$unsafe_plan" >/dev/null || fail 'unsafe permission elevation was not explicit'
if grep -Fq "$secret" <<< "$unsafe_plan"; then
  fail 'unsafe provider plan disclosed the raw prompt'
fi

isolated_codex_plan="$($LOOM provider-plan --provider codex \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt "$secret" \
  --isolate-context --json)"
jq -e '.context_isolation == true and
  (.argv | index("--ephemeral") != null) and
  (.argv | index("--ignore-rules") != null)' \
  <<< "$isolated_codex_plan" >/dev/null || \
  fail 'Codex context isolation was not normalized'

isolated_claude_plan="$($LOOM provider-plan --provider claude \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt "$secret" \
  --isolate-context --json)"
jq -e '.context_isolation == true and (.argv | index("--safe-mode") != null)' \
  <<< "$isolated_claude_plan" >/dev/null || \
  fail 'Claude context isolation was not normalized'

isolated_grok_plan="$($LOOM provider-plan --provider grok \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt "$secret" \
  --isolate-context --json)"
jq -e '.context_isolation == true and
  (.argv | index("--no-memory") != null) and
  (.argv | index("--no-subagents") != null) and
  (.argv | index("--disable-web-search") != null)' \
  <<< "$isolated_grok_plan" >/dev/null || \
  fail 'Grok context isolation was not normalized'

isolated_opencode_plan="$($LOOM provider-plan --provider opencode \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt "$secret" \
  --isolate-context --json)"
jq -e '.context_isolation == true and (.argv | index("--pure") != null)' \
  <<< "$isolated_opencode_plan" >/dev/null || \
  fail 'OpenCode context isolation was not normalized'

resume_plan="$($LOOM provider-plan --provider claude --mode resume \
  --provider-session 22222222-2222-4222-8222-222222222222 \
  --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$secret" --json)"
jq -e '.mode == "resume" and
  .provider_session == "22222222-2222-4222-8222-222222222222" and
  .session_binding == "caller" and (.argv | index("--resume") != null)' \
  <<< "$resume_plan" >/dev/null || fail 'resume plan lost native provider identity'

if "$LOOM" provider-plan --provider claude --mode resume \
  --provider-session not-a-uuid --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt test \
  > "$TEST_ROOT/resume-uuid.out" 2> "$TEST_ROOT/resume-uuid.err"; then
  fail 'caller-bound resume accepted a non-UUID provider identity'
fi
grep -q 'provider-session-must-be-uuid:claude' "$TEST_ROOT/resume-uuid.err" || \
  fail 'invalid resume identity was refused by the wrong rule'

if "$LOOM" provider-plan --provider grok --session-id not-a-uuid \
  --cwd "$TEST_ROOT" --prompt test > "$TEST_ROOT/uuid.out" 2> "$TEST_ROOT/uuid.err"; then
  fail 'caller-bound provider accepted a non-UUID session identity'
fi
grep -q 'provider-session-id-must-be-uuid:grok' "$TEST_ROOT/uuid.err" || \
  fail 'invalid Grok session identity was refused by the wrong rule'

if "$LOOM" provider-status --provider no-such-provider \
  > "$TEST_ROOT/unknown.out" 2> "$TEST_ROOT/unknown.err"; then
  fail 'unknown provider was accepted'
fi
grep -q 'unsupported-provider:no-such-provider' "$TEST_ROOT/unknown.err" || \
  fail 'unknown provider was refused by the wrong rule'

if SOUNIO_LOOM_PROVIDER_CODEX=relative-codex \
  "$LOOM" provider-status --provider codex \
  > "$TEST_ROOT/relative.out" 2> "$TEST_ROOT/relative.err"; then
  fail 'relative provider override was accepted'
fi
grep -q 'provider-override-must-be-absolute:codex' "$TEST_ROOT/relative.err" || \
  fail 'relative provider override was refused by the wrong rule'

login="$($LOOM provider-auth-login --provider codex)"
grep -q 'LOOM_PROVIDER_AUTH_DELEGATE .*credential_authority=native' <<< "$login" || \
  fail 'Loom did not identify native credential authority before login'
grep -q '^FAKE_LOGIN provider=codex$' <<< "$login" || \
  fail 'provider login was not delegated to the native CLI'

run_prompt='PROVIDER_ABI_RUN_WITNESS'
CODEX_SESSION_ID=parent-session CODEX_THREAD_ID=parent-thread CODEX_CI=1 \
CLAUDECODE=1 CLAUDE_CODE_ENTRYPOINT=parent CLAUDE_CODE_SESSION_ID=parent-claude \
TMUX=parent-tmux TMUX_PANE=parent-pane TMUX_TMPDIR=parent-tmp \
"$LOOM" provider-start --provider codex --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$LANE" --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$run_prompt" --isolate-context \
  > "$TEST_ROOT/start.out"
grep -q 'LOOM_PROVIDER_STARTED schema=loom-provider-abi-v1 provider=codex' \
  "$TEST_ROOT/start.out" || fail 'provider-start omitted its ABI receipt'
grep -q 'unsafe_auto=false context_isolation=true' "$TEST_ROOT/start.out" || \
  fail 'provider-start omitted its explicit policy state'

descriptor="$STATE_DIR/sessions/$AGENT--$LANE/session.state"
state=''
for _ in $(seq 1 100); do
  state="$(sed -n 's/^state=//p' "$descriptor" 2>/dev/null || true)"
  [[ "$state" == exited ]] && break
  sleep 0.05
done
[[ "$state" == exited ]] || fail 'fake provider did not reach terminal state'
grep -q '^command=fake-codex$' "$descriptor" || \
  fail 'internal provider executor obscured the native CLI identity'

replay="$($LOOM snapshot --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --cwd "$TEST_ROOT" --cursor 0 --meta 2> "$TEST_ROOT/snapshot.meta")"
grep -q "FAKE_CODEX_OUTPUT:$run_prompt" <<< "$replay" || \
  fail 'provider output was not durably replayable'
grep -q 'source=offline' "$TEST_ROOT/snapshot.meta" || \
  fail 'terminal provider replay did not use verified offline custody'

persistent_prompt='PERSISTENT_INITIAL_WITNESS'
CODEX_SESSION_ID=parent-session CODEX_THREAD_ID=parent-thread CODEX_CI=1 \
CLAUDECODE=1 CLAUDE_CODE_ENTRYPOINT=parent CLAUDE_CODE_SESSION_ID=parent-claude \
TMUX=parent-tmux TMUX_PANE=parent-pane TMUX_TMPDIR=parent-tmp \
"$LOOM" provider-open --provider codex --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$PERSISTENT_LANE" --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$persistent_prompt" \
  > "$TEST_ROOT/open.out"
grep -q 'LOOM_PROVIDER_OPENED schema=loom-provider-abi-v1 provider=codex lifecycle=persistent stdin_authority=loom-lease' \
  "$TEST_ROOT/open.out" || fail 'provider-open omitted its persistent custody receipt'

persistent_descriptor="$STATE_DIR/sessions/$AGENT--$PERSISTENT_LANE/session.state"
before_status="$($LOOM status --machine --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT")"
before_instance="$(sed -n 's/^instance_id=//p' <<< "$before_status")"
before_kernel="$(sed -n 's/^daemon_pid=//p' <<< "$before_status")"
before_guardian="$(sed -n 's/^guardian_pid=//p' <<< "$before_status")"
before_harness="$(sed -n 's/^harness_pid=//p' <<< "$before_status")"
[[ -n "$before_instance" && -n "$before_kernel" && -n "$before_guardian" && \
  -n "$before_harness" ]] || fail 'provider-open omitted its process authority lattice'

persistent_replay=''
for _ in $(seq 1 100); do
  persistent_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" --cursor 0 \
    2>/dev/null || true)"
  grep -q "FAKE_CODEX_TUI_READY:$persistent_prompt" <<< "$persistent_replay" && break
  sleep 0.05
done
grep -q "FAKE_CODEX_TUI_READY:$persistent_prompt" <<< "$persistent_replay" || \
  fail 'persistent provider initial output was not durably visible'

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" --at now >/dev/null
for _ in $(seq 1 100); do
  if ! "$LOOM" status --state-dir "$STATE_DIR" --agent "$AGENT" \
    --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
guardian_during="$($LOOM guardian-status --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT")"
grep -q "instance_id=$before_instance" <<< "$guardian_during" || \
  fail 'Guardian lost the persistent provider generation after kernel death'
grep -q "guardian_pid=$before_guardian" <<< "$guardian_during" || \
  fail 'Guardian PID changed after kernel death'
grep -q "harness_pid=$before_harness" <<< "$guardian_during" || \
  fail 'provider PID changed after kernel death'

"$LOOM" recover --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" > "$TEST_ROOT/recover.out"
after_status="$($LOOM status --machine --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT")"
after_instance="$(sed -n 's/^instance_id=//p' <<< "$after_status")"
after_kernel="$(sed -n 's/^daemon_pid=//p' <<< "$after_status")"
after_guardian="$(sed -n 's/^guardian_pid=//p' <<< "$after_status")"
after_harness="$(sed -n 's/^harness_pid=//p' <<< "$after_status")"
[[ "$after_instance" == "$before_instance" ]] || fail 'recovery changed provider generation'
[[ "$after_guardian" == "$before_guardian" ]] || fail 'recovery replaced the Guardian'
[[ "$after_harness" == "$before_harness" ]] || fail 'recovery replaced the provider process'
[[ "$after_kernel" != "$before_kernel" ]] || fail 'kernel sabotage did not replace the kernel'

post_recovery_prompt='PERSISTENT_AFTER_RECOVERY_WITNESS'
"$LOOM" wake --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$PERSISTENT_LANE" --session-id "$SESSION_ID" \
  --message-id provider-open-recovery --prompt "$post_recovery_prompt" \
  --cwd "$TEST_ROOT" >/dev/null
for _ in $(seq 1 100); do
  persistent_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" --cursor 0 \
    2>/dev/null || true)"
  grep -q "FAKE_CODEX_TUI_WAKE:$post_recovery_prompt" <<< "$persistent_replay" && break
  sleep 0.05
done
grep -q "FAKE_CODEX_TUI_WAKE:$post_recovery_prompt" <<< "$persistent_replay" || \
  fail 'recovered kernel could not deliver a second provider input'

"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null
for _ in $(seq 1 100); do
  [[ "$(sed -n 's/^state=//p' "$persistent_descriptor" 2>/dev/null || true)" == exited ]] && break
  sleep 0.05
done
journal="$(sed -n 's/^journal_file=//p' "$persistent_descriptor")"
guardian_journal="$(sed -n 's/^guardian_journal_file=//p' "$persistent_descriptor")"
"$LOOM" verify-journal --journal "$journal" | grep -q '^JOURNAL_OK ' || \
  fail 'persistent provider semantic journal did not verify'
"$LOOM" verify-guardian-journal --journal "$guardian_journal" | \
  grep -q '^GUARDIAN_JOURNAL_OK ' || \
  fail 'persistent provider Guardian journal did not verify'

printf 'sounio-loom-provider-abi-selftest: PASS providers=4 credentials=native prompt=redacted unsafe=explicit context_isolation=normalized harness_identity=clean stdin=closed persistent_stdin=loom-lease kernel_recovery=stable-provider session_binding=typed replay=verified\n'
