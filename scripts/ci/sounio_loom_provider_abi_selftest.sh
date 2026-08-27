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
CLAUDE_PERSISTENT_LANE='claude-persistent'
CLAUDE_RESUME_LANE='claude-resume'
KIMI_LANE='kimi-headless'
KIMI_PERSISTENT_LANE='kimi-persistent'

fail() {
  printf 'sounio-loom-provider-abi-selftest: FAIL: %s test_root=%s\n' "$*" "$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$PERSISTENT_LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$CLAUDE_PERSISTENT_LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$CLAUDE_RESUME_LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$KIMI_LANE" >/dev/null 2>&1 || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" >/dev/null 2>&1 || true
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
  fake-claude:--session-id:*|fake-claude:--resume:*)
    if [[ -n "${CODEX_SESSION_ID+x}${CODEX_THREAD_ID+x}${CODEX_CI+x}${CLAUDECODE+x}${CLAUDE_CODE_ENTRYPOINT+x}${CLAUDE_CODE_SESSION_ID+x}${TMUX+x}${TMUX_PANE+x}${TMUX_TMPDIR+x}" ]]; then
      printf 'parent harness identity leaked into persistent Claude process\n' >&2
      exit 47
    fi
    printf 'FAKE_CLAUDE_TUI_READY:%s\n' "$*"
    while IFS= read -r wake; do
      printf 'FAKE_CLAUDE_TUI_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
    ;;
  fake-kimi:--version:)
    printf '0.38.0-provider-abi-test\n'
    ;;
  fake-kimi:login:)
    printf 'FAKE_LOGIN provider=kimi\n'
    ;;
  fake-kimi:--output-format:stream-json)
    if [[ -n "${CODEX_SESSION_ID+x}${CODEX_THREAD_ID+x}${CODEX_CI+x}${CLAUDECODE+x}${CLAUDE_CODE_ENTRYPOINT+x}${CLAUDE_CODE_SESSION_ID+x}${KIMI_SESSION_ID+x}${KIMI_CLI_SESSION_ID+x}${CURSOR_SESSION_ID+x}${CURSOR_AGENT_SESSION_ID+x}${GROK_SESSION_ID+x}${SOUNIO_AGENT_ID+x}${SOUNIO_LANE_ID+x}${SOUNIO_AGENTD_SOCKET+x}${SOUNIO_AGENTD_TOKEN_FILE+x}${TMUX+x}${TMUX_PANE+x}${TMUX_TMPDIR+x}" ]]; then
      printf 'parent harness identity leaked into Kimi provider process\n' >&2
      exit 44
    fi
    prompt="${!#}"
    if IFS= read -r -t 0.1 _unexpected; then
      printf 'Kimi stdin was not closed\n' >&2
      exit 45
    fi
    printf '{"role":"assistant","content":"FAKE_KIMI_OUTPUT:%s"}\n' "$prompt"
    sleep 1
    ;;
  fake-kimi::)
    if [[ -n "${CODEX_SESSION_ID+x}${CODEX_THREAD_ID+x}${CODEX_CI+x}${CLAUDECODE+x}${CLAUDE_CODE_ENTRYPOINT+x}${CLAUDE_CODE_SESSION_ID+x}${KIMI_SESSION_ID+x}${KIMI_CLI_SESSION_ID+x}${CURSOR_SESSION_ID+x}${CURSOR_AGENT_SESSION_ID+x}${GROK_SESSION_ID+x}${SOUNIO_AGENT_ID+x}${SOUNIO_LANE_ID+x}${SOUNIO_AGENTD_SOCKET+x}${SOUNIO_AGENTD_TOKEN_FILE+x}${TMUX+x}${TMUX_PANE+x}${TMUX_TMPDIR+x}" ]]; then
      printf 'parent harness identity leaked into persistent Kimi process\n' >&2
      exit 46
    fi
    printf 'FAKE_KIMI_TUI_READY\n'
    while IFS= read -r wake; do
      printf 'FAKE_KIMI_TUI_WAKE:%s\n' "$wake"
      [[ "$wake" == /exit ]] && break
    done
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
for provider in codex claude kimi grok opencode; do
  cp "$TEST_ROOT/fake-provider" "$TEST_ROOT/fake-$provider"
done

export SOUNIO_LOOM_PROVIDER_CODEX="$TEST_ROOT/fake-codex"
export SOUNIO_LOOM_PROVIDER_CLAUDE="$TEST_ROOT/fake-claude"
export SOUNIO_LOOM_PROVIDER_KIMI="$TEST_ROOT/fake-kimi"
export SOUNIO_LOOM_PROVIDER_GROK="$TEST_ROOT/fake-grok"
export SOUNIO_LOOM_PROVIDER_OPENCODE="$TEST_ROOT/fake-opencode"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
version="$($LOOM runtime-version)"
grep -q '^runtime_version=2026.08.27.37$' <<< "$version" || \
  fail 'public loom launcher selected the wrong runtime'

providers="$($LOOM provider-list --json)"
jq -e '.schema == "loom-provider-abi-v1" and (.providers | length == 5)' \
  <<< "$providers" >/dev/null || fail 'provider catalog schema or cardinality changed'
jq -e '.providers[] | select(.provider == "codex") |
  .installed == true and .auth == "authenticated" and
  .credential_authority == "native" and .session_binding == "stream-observed"' \
  <<< "$providers" >/dev/null || fail 'Codex provider status was not normalized'
jq -e '.providers[] | select(.provider == "claude") |
  .auth == "unauthenticated" and .session_binding == "caller"' \
  <<< "$providers" >/dev/null || \
  fail 'Claude nonzero auth-status JSON was not classified as unauthenticated'
jq -e '.providers[] | select(.provider == "kimi") |
  .installed == true and .auth == "unknown" and
  .auth_reason == "native-cli-has-no-offline-auth-status" and
  .session_binding == "native-store" and
  (.capabilities | index("persistent-input") != null)' \
  <<< "$providers" >/dev/null || fail 'Kimi provider status was not normalized'
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
  .prompt_transport == "argv" and .context_isolation == false and
  (.argv | index("--no-alt-screen") != null) and
  (.argv | index("exec") == null)' <<< "$persistent_plan" >/dev/null || \
  fail 'persistent Codex plan has the wrong input authority'

kimi_plan="$($LOOM provider-plan --provider kimi --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --model kimi-test --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$kimi_plan"; then
  fail 'headless Kimi plan disclosed the raw prompt'
fi
jq -e --arg digest "$secret_sha" '
  .provider == "kimi" and .lifecycle == "turn" and
  .stdin_authority == "closed" and .prompt_transport == "argv" and
  .session_binding == "native-store" and .prompt_sha256 == $digest and
  (.argv | index("--output-format") != null) and
  (.argv | index("stream-json") != null) and
  (.argv | index("--prompt") != null) and
  (.argv | index("-m") != null) and (.argv | index("kimi-test") != null)' \
  <<< "$kimi_plan" >/dev/null || fail 'headless Kimi plan has the wrong native argv'

kimi_persistent_plan="$($LOOM provider-plan --provider kimi \
  --lifecycle persistent --session-id "$SESSION_ID" --cwd "$TEST_ROOT" \
  --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$kimi_persistent_plan"; then
  fail 'persistent Kimi plan disclosed the lease-delivered prompt'
fi
jq -e '.provider == "kimi" and .lifecycle == "persistent" and
  .stdin_authority == "loom-lease" and .prompt_transport == "loom-wake" and
  (.argv | length == 1) and (.argv | index("--prompt") == null)' \
  <<< "$kimi_persistent_plan" >/dev/null || \
  fail 'persistent Kimi plan did not separate process argv from input custody'

if "$LOOM" provider-plan --provider codex --lifecycle persistent \
  --session-id "$SESSION_ID" --cwd "$TEST_ROOT" --prompt test --isolate-context \
  > "$TEST_ROOT/persistent-isolation.out" 2> "$TEST_ROOT/persistent-isolation.err"; then
  fail 'persistent provider silently accepted headless context isolation'
fi
grep -q 'persistent-context-isolation-unavailable:codex' \
  "$TEST_ROOT/persistent-isolation.err" || \
  fail 'persistent context isolation was refused by the wrong rule'

if "$LOOM" provider-plan --provider kimi --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt test --isolate-context \
  > "$TEST_ROOT/kimi-isolation.out" 2> "$TEST_ROOT/kimi-isolation.err"; then
  fail 'Kimi plan silently claimed unavailable context isolation'
fi
grep -q 'provider-context-isolation-unavailable:kimi' \
  "$TEST_ROOT/kimi-isolation.err" || \
  fail 'Kimi context isolation was refused by the wrong rule'

claude_persistent_plan="$($LOOM provider-plan --provider claude \
  --lifecycle persistent --session-id "$SESSION_ID" --cwd "$TEST_ROOT" \
  --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$claude_persistent_plan"; then
  fail 'persistent Claude plan disclosed the lease-delivered prompt'
fi
jq -e --arg session "$SESSION_ID" '
  .provider == "claude" and .lifecycle == "persistent" and
  .mode == "new" and .prompt_transport == "loom-wake" and
  .stdin_authority == "loom-lease" and
  (.argv | index("--session-id") != null) and
  (.argv | index($session) != null) and
  (.argv | index("--setting-sources") != null) and
  (.argv | index("user,local") != null) and
  (.argv | index("--continue") == null) and
  (.argv | index("--fork-session") == null) and
  (.argv | index("--resume") == null)' \
  <<< "$claude_persistent_plan" >/dev/null || \
  fail 'persistent Claude new-session plan lost exact identity binding'

CLAUDE_PROVIDER_SESSION='22222222-2222-4222-8222-222222222222'
claude_persistent_resume_plan="$($LOOM provider-plan --provider claude \
  --lifecycle persistent --mode resume \
  --provider-session "$CLAUDE_PROVIDER_SESSION" --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$secret" --json)"
if grep -Fq "$secret" <<< "$claude_persistent_resume_plan"; then
  fail 'persistent Claude resume plan disclosed the lease-delivered prompt'
fi
jq -e --arg provider_session "$CLAUDE_PROVIDER_SESSION" '
  .provider == "claude" and .lifecycle == "persistent" and
  .mode == "resume" and .prompt_transport == "loom-wake" and
  .provider_session == $provider_session and
  (.argv | index("--resume") != null) and
  (.argv | index($provider_session) != null) and
  (.argv | index("--session-id") == null) and
  (.argv | index("--continue") == null) and
  (.argv | index("--fork-session") == null)' \
  <<< "$claude_persistent_resume_plan" >/dev/null || \
  fail 'persistent Claude resume plan lost exact provider identity'

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

unsafe_kimi_plan="$($LOOM provider-plan --provider kimi --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$secret" --unsafe-auto --json)"
jq -e '.unsafe_auto == true and (.argv | index("--auto") != null)' \
  <<< "$unsafe_kimi_plan" >/dev/null || \
  fail 'Kimi unsafe permission elevation was not explicit'

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

kimi_resume_plan="$($LOOM provider-plan --provider kimi --mode resume \
  --provider-session ses_provider_abi --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$secret" --json)"
jq -e '.mode == "resume" and .provider_session == "ses_provider_abi" and
  .session_binding == "native-store" and (.argv | index("--session") != null)' \
  <<< "$kimi_resume_plan" >/dev/null || \
  fail 'Kimi resume plan lost its native session identity'

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

kimi_run_prompt='KIMI_ABI_RUN_WITNESS'
CODEX_SESSION_ID=parent-session CODEX_THREAD_ID=parent-thread CODEX_CI=1 \
CLAUDECODE=1 CLAUDE_CODE_ENTRYPOINT=parent CLAUDE_CODE_SESSION_ID=parent-claude \
TMUX=parent-tmux TMUX_PANE=parent-pane TMUX_TMPDIR=parent-tmp \
"$LOOM" provider-start --provider kimi --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$KIMI_LANE" --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$kimi_run_prompt" > "$TEST_ROOT/kimi-start.out"
grep -q 'LOOM_PROVIDER_STARTED schema=loom-provider-abi-v1 provider=kimi' \
  "$TEST_ROOT/kimi-start.out" || fail 'Kimi provider-start omitted its ABI receipt'
kimi_descriptor="$STATE_DIR/sessions/$AGENT--$KIMI_LANE/session.state"
kimi_state=''
for _ in $(seq 1 100); do
  kimi_state="$(sed -n 's/^state=//p' "$kimi_descriptor" 2>/dev/null || true)"
  [[ "$kimi_state" == exited ]] && break
  sleep 0.05
done
[[ "$kimi_state" == exited ]] || fail 'fake Kimi provider did not reach terminal state'
grep -q '^command=fake-kimi$' "$kimi_descriptor" || \
  fail 'internal provider executor obscured the native Kimi identity'
kimi_replay="$($LOOM snapshot --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$KIMI_LANE" --cwd "$TEST_ROOT" --cursor 0)"
grep -q "FAKE_KIMI_OUTPUT:$kimi_run_prompt" <<< "$kimi_replay" || \
  fail 'headless Kimi output was not durably replayable'

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

CLAUDE_NEW_SESSION='33333333-3333-4333-8333-333333333333'
claude_new_prompt='CLAUDE_PERSISTENT_NEW_WITNESS'
CODEX_SESSION_ID=parent-session CODEX_THREAD_ID=parent-thread CODEX_CI=1 \
CLAUDECODE=1 CLAUDE_CODE_ENTRYPOINT=parent CLAUDE_CODE_SESSION_ID=parent-claude \
TMUX=parent-tmux TMUX_PANE=parent-pane TMUX_TMPDIR=parent-tmp \
"$LOOM" provider-open --provider claude --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$CLAUDE_PERSISTENT_LANE" \
  --session-id "$CLAUDE_NEW_SESSION" --cwd "$TEST_ROOT" \
  --prompt "$claude_new_prompt" > "$TEST_ROOT/claude-new-open.out"
grep -q 'LOOM_PROVIDER_OPENED schema=loom-provider-abi-v1 provider=claude lifecycle=persistent stdin_authority=loom-lease prompt_transport=loom-wake' \
  "$TEST_ROOT/claude-new-open.out" || \
  fail 'persistent Claude new session omitted its custody receipt'
claude_new_replay=''
for _ in $(seq 1 100); do
  claude_new_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$CLAUDE_PERSISTENT_LANE" --cwd "$TEST_ROOT" \
    --cursor 0 2>/dev/null || true)"
  grep -q "FAKE_CLAUDE_TUI_WAKE:$claude_new_prompt" \
    <<< "$claude_new_replay" && break
  sleep 0.05
done
grep -q "FAKE_CLAUDE_TUI_READY:--session-id $CLAUDE_NEW_SESSION --setting-sources user,local" \
  <<< "$claude_new_replay" || fail 'persistent Claude new argv was not exact'
grep -q "FAKE_CLAUDE_TUI_WAKE:$claude_new_prompt" <<< "$claude_new_replay" || \
  fail 'persistent Claude new bootstrap did not cross the Loom lease'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$CLAUDE_PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null

CLAUDE_RESUME_SESSION='44444444-4444-4444-8444-444444444444'
claude_resume_prompt='CLAUDE_PERSISTENT_RESUME_WITNESS'
"$LOOM" provider-open --provider claude --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$CLAUDE_RESUME_LANE" \
  --session-id "$CLAUDE_RESUME_SESSION" --mode resume \
  --provider-session "$CLAUDE_PROVIDER_SESSION" --cwd "$TEST_ROOT" \
  --prompt "$claude_resume_prompt" > "$TEST_ROOT/claude-resume-open.out"
claude_resume_replay=''
for _ in $(seq 1 100); do
  claude_resume_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$CLAUDE_RESUME_LANE" --cwd "$TEST_ROOT" \
    --cursor 0 2>/dev/null || true)"
  grep -q "FAKE_CLAUDE_TUI_WAKE:$claude_resume_prompt" \
    <<< "$claude_resume_replay" && break
  sleep 0.05
done
grep -q "FAKE_CLAUDE_TUI_READY:--resume $CLAUDE_PROVIDER_SESSION --setting-sources user,local" \
  <<< "$claude_resume_replay" || fail 'persistent Claude resume argv was not exact'
grep -q "FAKE_CLAUDE_TUI_WAKE:$claude_resume_prompt" \
  <<< "$claude_resume_replay" || \
  fail 'persistent Claude resume bootstrap did not cross the Loom lease'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$CLAUDE_RESUME_LANE" --cwd "$TEST_ROOT" >/dev/null

kimi_persistent_prompt='KIMI_PERSISTENT_INITIAL_WITNESS'
CODEX_SESSION_ID=parent-session CODEX_THREAD_ID=parent-thread CODEX_CI=1 \
CLAUDECODE=1 CLAUDE_CODE_ENTRYPOINT=parent CLAUDE_CODE_SESSION_ID=parent-claude \
KIMI_SESSION_ID=parent-kimi KIMI_CLI_SESSION_ID=parent-kimi-cli \
CURSOR_SESSION_ID=parent-cursor CURSOR_AGENT_SESSION_ID=parent-cursor-agent \
GROK_SESSION_ID=parent-grok SOUNIO_AGENT_ID=parent-agent \
SOUNIO_LANE_ID=parent-lane SOUNIO_AGENTD_SOCKET=parent-agentd-socket \
SOUNIO_AGENTD_TOKEN_FILE=parent-agentd-token \
TMUX=parent-tmux TMUX_PANE=parent-pane TMUX_TMPDIR=parent-tmp \
"$LOOM" provider-open --provider kimi --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --session-id "$SESSION_ID" \
  --cwd "$TEST_ROOT" --prompt "$kimi_persistent_prompt" \
  > "$TEST_ROOT/kimi-open.out"
grep -q 'LOOM_WAKE state=delivered .*message_id=provider-bootstrap-' \
  "$TEST_ROOT/kimi-open.out" || \
  fail 'persistent Kimi bootstrap did not traverse the authenticated Loom wake path'
bootstrap_id="$(sed -n 's/.*message_id=\(provider-bootstrap-[^ ]*\).*/\1/p' \
  "$TEST_ROOT/kimi-open.out" | head -n 1)"
bootstrap_digest="$(printf '%s\0%s\0%s' kimi "$SESSION_ID" \
  "$kimi_persistent_prompt" | sha256sum | cut -d' ' -f1)"
expected_bootstrap_id="provider-bootstrap-${bootstrap_digest:0:16}"
[[ "$bootstrap_id" == "$expected_bootstrap_id" ]] || \
  fail 'persistent bootstrap identity was not bound to provider, session, and prompt'
grep -q 'LOOM_PROVIDER_OPENED schema=loom-provider-abi-v1 provider=kimi lifecycle=persistent stdin_authority=loom-lease prompt_transport=loom-wake' \
  "$TEST_ROOT/kimi-open.out" || \
  fail 'persistent Kimi receipt omitted its prompt transport'

kimi_persistent_status="$($LOOM status --machine --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT")"
kimi_before_instance="$(sed -n 's/^instance_id=//p' <<< "$kimi_persistent_status")"
kimi_before_kernel="$(sed -n 's/^daemon_pid=//p' <<< "$kimi_persistent_status")"
kimi_before_guardian="$(sed -n 's/^guardian_pid=//p' <<< "$kimi_persistent_status")"
kimi_before_harness="$(sed -n 's/^harness_pid=//p' <<< "$kimi_persistent_status")"
kimi_persistent_replay=''
for _ in $(seq 1 100); do
  kimi_persistent_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" \
    --cursor 0 2>/dev/null || true)"
  grep -q "FAKE_KIMI_TUI_WAKE:$kimi_persistent_prompt" \
    <<< "$kimi_persistent_replay" && break
  sleep 0.05
done
grep -q 'FAKE_KIMI_TUI_READY' <<< "$kimi_persistent_replay" || \
  fail 'persistent Kimi TUI did not become durably visible'
grep -q "FAKE_KIMI_TUI_WAKE:$kimi_persistent_prompt" \
  <<< "$kimi_persistent_replay" || \
  fail 'persistent Kimi bootstrap prompt was not durably visible'

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" --at now >/dev/null
for _ in $(seq 1 100); do
  if ! "$LOOM" status --state-dir "$STATE_DIR" --agent "$AGENT" \
    --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
kimi_guardian_during="$($LOOM guardian-status --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT")"
grep -q "instance_id=$kimi_before_instance" <<< "$kimi_guardian_during" || \
  fail 'Guardian lost the Kimi generation after kernel death'
grep -q "guardian_pid=$kimi_before_guardian" <<< "$kimi_guardian_during" || \
  fail 'Kimi Guardian PID changed after kernel death'
grep -q "harness_pid=$kimi_before_harness" <<< "$kimi_guardian_during" || \
  fail 'Kimi provider PID changed after kernel death'

"$LOOM" recover --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null
kimi_after_status="$($LOOM status --machine --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT")"
[[ "$(sed -n 's/^instance_id=//p' <<< "$kimi_after_status")" == "$kimi_before_instance" ]] || \
  fail 'Kimi kernel recovery changed the provider generation'
[[ "$(sed -n 's/^guardian_pid=//p' <<< "$kimi_after_status")" == "$kimi_before_guardian" ]] || \
  fail 'Kimi kernel recovery replaced the Guardian'
[[ "$(sed -n 's/^harness_pid=//p' <<< "$kimi_after_status")" == "$kimi_before_harness" ]] || \
  fail 'Kimi kernel recovery replaced the provider process'
[[ "$(sed -n 's/^daemon_pid=//p' <<< "$kimi_after_status")" != "$kimi_before_kernel" ]] || \
  fail 'Kimi kernel sabotage did not replace the kernel'

kimi_post_recovery='KIMI_PERSISTENT_AFTER_RECOVERY_WITNESS'
"$LOOM" wake --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$KIMI_PERSISTENT_LANE" --session-id "$SESSION_ID" \
  --message-id kimi-provider-recovery --prompt "$kimi_post_recovery" \
  --cwd "$TEST_ROOT" >/dev/null
for _ in $(seq 1 100); do
  kimi_persistent_replay="$($LOOM snapshot --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" \
    --cursor 0 2>/dev/null || true)"
  grep -q "FAKE_KIMI_TUI_WAKE:$kimi_post_recovery" \
    <<< "$kimi_persistent_replay" && break
  sleep 0.05
done
grep -q "FAKE_KIMI_TUI_WAKE:$kimi_post_recovery" \
  <<< "$kimi_persistent_replay" || \
  fail 'recovered kernel could not deliver a second Kimi input'

"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane "$KIMI_PERSISTENT_LANE" --cwd "$TEST_ROOT" >/dev/null

printf 'sounio-loom-provider-abi-selftest: PASS providers=5 credentials=native prompt=redacted prompt_transport=typed wake_argv=prompt-free bootstrap_identity=provider+session+prompt unsafe=explicit context_isolation=normalized harness_identity=clean stdin=closed persistent_stdin=loom-lease kernel_recovery=stable-provider session_binding=typed replay=verified\n'
