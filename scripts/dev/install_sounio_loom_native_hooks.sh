#!/usr/bin/env bash

set -euo pipefail
umask 077

usage() {
  cat <<'USAGE'
Usage: scripts/dev/install_sounio_loom_native_hooks.sh \
  --target-root PATH [--source-root PATH] --activate

Atomically promote the repository's native LOOM hook configurations into a
serialized control checkout. The command verifies the active immutable runtime,
holds both a repository-wide promotion lock and the target worktree index lock,
keeps an audit backup, runs production-mode policyless canaries, and rolls all
four provider configuration files back on any failure.

Options:
  --target-root PATH  control checkout receiving the hook configurations
  --source-root PATH  source checkout (default: repository containing script)
  --activate          required acknowledgement that files may be replaced
  -h, --help          show this help
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local manifest="$1" key="$2"
  sed -n "s/^${key}=//p" "$manifest" | head -1
}

verify_manifest_file() {
  local manifest="$1" key="$2" path="$3" expected actual
  expected="$(manifest_value "$manifest" "$key")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] ||
    die "runtime manifest has invalid $key"
  [[ -f "$path" ]] || die "runtime manifest target is missing: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] ||
    die "runtime manifest hash mismatch: $key expected=$expected actual=$actual"
}

validate_candidate_config() {
  local codex="$1" claude="$2" cursor="$3" grok="$4"
  [[ -f "$codex" && -f "$claude" && -f "$cursor" && -f "$grok" ]] ||
    die 'source checkout omits a hook configuration'
  ! grep -Eiq 'python|pypy|rustc|cargo|sounio_coord_agent_hook\.py' \
      "$codex" "$claude" "$cursor" "$grok" ||
    die 'candidate hook configuration retains a prohibited Python/Rust bridge'
  [[ "$(grep -Fc 'bin/sounio-loom-runtime\" agent-hook --agent codex' "$codex")" -eq 5 ]] ||
    die 'candidate Codex configuration does not contain five native hook calls'
  [[ "$(grep -Fc 'bin/sounio-loom-runtime\" agent-hook --agent claude' "$claude")" -eq 6 ]] ||
    die 'candidate Claude configuration does not contain six native hook calls'
  [[ "$(grep -Fc 'bin/sounio-loom-runtime\" agent-hook --agent cursor' "$cursor")" -eq 6 ]] ||
    die 'candidate Cursor configuration does not contain six native hook calls'
  [[ "$(grep -Fc 'bin/sounio-loom-runtime\" agent-hook --agent grok' "$grok")" -eq 6 ]] ||
    die 'candidate Grok configuration does not contain six native hook calls'
  [[ "$(grep -Fc 'readlink -f' "$codex")" -eq 5 ]] ||
    die 'candidate Codex hooks do not pin one physical runtime generation'
  [[ "$(grep -Fc 'readlink -f' "$claude")" -eq 6 ]] ||
    die 'candidate Claude hooks do not pin one physical runtime generation'
  [[ "$(grep -Fc 'readlink -f' "$cursor")" -eq 6 ]] ||
    die 'candidate Cursor hooks do not pin one physical runtime generation'
  [[ "$(grep -Fc 'readlink -f' "$grok")" -eq 6 ]] ||
    die 'candidate Grok hooks do not pin one physical runtime generation'
  grep -Fq '"failClosed": true' "$cursor" ||
    die 'candidate Cursor hooks are not fail closed'
  ! grep -Eq '"matcher": "[^"]*(Bash|Exec)' "$codex" "$claude" ||
    die 'candidate attaches Bash/Exec before Sounio freezes its authority contract'
}

atomic_copy() {
  local source="$1" target="$2" tmp
  tmp="$(mktemp "$(dirname "$target")/.loom-hook-promote.XXXXXX")"
  cp "$source" "$tmp"
  chmod --reference="$source" "$tmp"
  sync -f "$tmp"
  mv -f "$tmp" "$target"
}

rollback_owned_file() {
  local candidate="$1" original="$2" target="$3" mode="$4" existed="$5"
  local current_sha candidate_sha original_sha
  current_sha="$(sha256sum "$target" 2>/dev/null | awk '{print $1}')"
  candidate_sha="$(sha256sum "$candidate" | awk '{print $1}')"
  original_sha=''
  [[ "$existed" -eq 0 ]] || original_sha="$(sha256sum "$original" | awk '{print $1}')"
  if [[ "$current_sha" == "$candidate_sha" ]]; then
    if [[ "$existed" -eq 1 ]]; then
      atomic_copy "$original" "$target"
      chmod "$mode" "$target"
    else
      rm -f "$target"
    fi
  elif [[ "$current_sha" != "$original_sha" ]]; then
    printf 'ROLLBACK_CONFLICT target=%s reason=unexpected-concurrent-edit\n' \
      "$target" >&2
    return 1
  fi
}

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TARGET_ROOT=''
ACTIVATE=0
while (($#)); do
  case "$1" in
    --source-root) [[ $# -ge 2 ]] || die "$1 requires a path"; SOURCE_ROOT="$2"; shift 2 ;;
    --target-root) [[ $# -ge 2 ]] || die "$1 requires a path"; TARGET_ROOT="$2"; shift 2 ;;
    --activate) ACTIVATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "$ACTIVATE" -eq 1 ]] || die 'promotion requires --activate'
[[ -n "$TARGET_ROOT" ]] || die 'promotion requires --target-root'
SOURCE_ROOT="$(cd "$SOURCE_ROOT" && pwd -P)"
TARGET_ROOT="$(cd "$TARGET_ROOT" && pwd -P)"
[[ "$(git -C "$SOURCE_ROOT" rev-parse --show-toplevel 2>/dev/null)" == "$SOURCE_ROOT" ]] ||
  die "source root is not a Git worktree: $SOURCE_ROOT"
[[ "$(git -C "$TARGET_ROOT" rev-parse --show-toplevel 2>/dev/null)" == "$TARGET_ROOT" ]] ||
  die "target root is not a Git worktree: $TARGET_ROOT"
[[ "$SOURCE_ROOT" != "$TARGET_ROOT" ]] || die 'source and target worktrees must differ'

SOURCE_CODEX="$SOURCE_ROOT/.codex/hooks.json"
SOURCE_CLAUDE="$SOURCE_ROOT/.claude/settings.json"
SOURCE_CURSOR="$SOURCE_ROOT/.cursor/hooks.json"
SOURCE_GROK="$SOURCE_ROOT/.grok/hooks/loom-native.json"
TARGET_CODEX="$TARGET_ROOT/.codex/hooks.json"
TARGET_CLAUDE="$TARGET_ROOT/.claude/settings.json"
TARGET_CURSOR="$TARGET_ROOT/.cursor/hooks.json"
TARGET_GROK="$TARGET_ROOT/.grok/hooks/loom-native.json"
SOURCE_HEAD="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
validate_candidate_config "$SOURCE_CODEX" "$SOURCE_CLAUDE" \
  "$SOURCE_CURSOR" "$SOURCE_GROK"
[[ -f "$TARGET_CODEX" && -f "$TARGET_CLAUDE" ]] ||
  die 'target checkout omits the existing hook configuration boundary'

TARGET_GIT_DIR="$(git -C "$TARGET_ROOT" rev-parse --path-format=absolute --git-dir)"
TARGET_COMMON_DIR="$(git -C "$TARGET_ROOT" rev-parse --path-format=absolute --git-common-dir)"
RUNTIME_ROOT="${SOUNIO_COORD_RUNTIME_DIR:-$TARGET_COMMON_DIR/sounio-coord-runtime}"
STATE_ROOT="${SOUNIO_COORD_DIR:-$TARGET_COMMON_DIR/sounio-coord-state}"
RUNTIME_BUNDLE="$(readlink -f "$RUNTIME_ROOT/current" 2>/dev/null || true)"
[[ -n "$RUNTIME_BUNDLE" ]] || die "active shared runtime is missing: $RUNTIME_ROOT/current"
RUNTIME_MANIFEST="$RUNTIME_BUNDLE/manifest"
[[ -f "$RUNTIME_MANIFEST" && -x "$RUNTIME_BUNDLE/bin/sounio-loom-runtime" ]] ||
  die 'active shared runtime is incomplete'
for capability in loom-native-agent-hook-v1 loom-native-hook-binary-attestation-v1 \
    loom-runtime-authority-capsule-v1 loom-native-hook-cutover-v1; do
  grep -q "^capability=$capability$" "$RUNTIME_MANIFEST" ||
    die "active shared runtime omits capability=$capability"
done
verify_manifest_file "$RUNTIME_MANIFEST" loom_runtime_sha256 \
  "$RUNTIME_BUNDLE/bin/sounio-loom-runtime"
verify_manifest_file "$RUNTIME_MANIFEST" coord_runtime_sha256 \
  "$RUNTIME_BUNDLE/bin/sounio-coord-runtime"
AUTHORITY_ROOT="$RUNTIME_BUNDLE/policy/language-authority"
verify_manifest_file "$RUNTIME_MANIFEST" loom_language_authority_policy_manifest_sha256 \
  "$AUTHORITY_ROOT/tools/loom/language_authority.freeze.v1"
verify_manifest_file "$RUNTIME_MANIFEST" loom_language_authority_policy_source_sha256 \
  "$AUTHORITY_ROOT/stdlib/coordination/loom_language_authority.sio"
verify_manifest_file "$RUNTIME_MANIFEST" loom_language_authority_policy_entrypoint_sha256 \
  "$AUTHORITY_ROOT/tools/loom/language_authority_main.sio"
CUTOVER_ROOT="$RUNTIME_BUNDLE/policy/native-hook-cutover"
verify_manifest_file "$RUNTIME_MANIFEST" loom_native_hook_cutover_manifest_sha256 \
  "$CUTOVER_ROOT/tools/loom/native_hook_cutover.freeze.v1"
verify_manifest_file "$RUNTIME_MANIFEST" loom_native_hook_cutover_source_sha256 \
  "$CUTOVER_ROOT/stdlib/coordination/loom_native_hook_cutover_authority.sio"
verify_manifest_file "$RUNTIME_MANIFEST" loom_native_hook_cutover_entrypoint_sha256 \
  "$CUTOVER_ROOT/tools/loom/native_hook_cutover_authority_main.sio"
verify_manifest_file "$RUNTIME_MANIFEST" loom_native_hook_cutover_runtime_sha256 \
  "$RUNTIME_BUNDLE/bin/sounio-loom-native-hook-cutover"
for provider in codex claude cursor grok; do
  verify_manifest_file "$RUNTIME_MANIFEST" \
    "loom_native_hook_cutover_${provider}_config_sha256" \
    "$CUTOVER_ROOT/configs/${provider}.json"
done
[[ "$(manifest_value "$RUNTIME_MANIFEST" loom_native_hook_cutover_semantics_sha256)" == \
  27c5fd758d161026c5c41d0cd0be0f1aa90bd4e3f4287da3c60fb748d1334882 ]] ||
  die 'active runtime is not bound to frozen Sounio action 9045 semantics'
[[ ! -e "$RUNTIME_BUNDLE/hooks/sounio_coord_agent_hook.py" && \
  ! -e "$RUNTIME_BUNDLE/hooks/sounio_coord_agent_hook_runtime.py" ]] ||
  die 'active runtime still contains the Python hook bridge'

mkdir -p "$TARGET_COMMON_DIR/sounio-loom-native-hook-promotions"
exec 9>"$TARGET_COMMON_DIR/sounio-loom-native-hook-promotions/.promotion.lock"
flock -n 9 || die 'another native hook promotion is active'

TARGET_HEAD="$(git -C "$TARGET_ROOT" rev-parse HEAD)"
TARGET_BRANCH="$(git -C "$TARGET_ROOT" symbolic-ref --short -q HEAD || printf detached)"
git -C "$TARGET_ROOT" diff --quiet -- \
  .codex/hooks.json .claude/settings.json .cursor/hooks.json \
  .grok/hooks/loom-native.json ||
  die 'target hook configurations have unstaged changes'
git -C "$TARGET_ROOT" diff --cached --quiet -- \
  .codex/hooks.json .claude/settings.json .cursor/hooks.json \
  .grok/hooks/loom-native.json ||
  die 'target hook configurations have staged changes'

TXN_ID="hook-promotion-$(date -u +%Y%m%dT%H%M%SZ)-$$"
TXN_DIR="$TARGET_COMMON_DIR/sounio-loom-native-hook-promotions/$TXN_ID"
mkdir -p "$TXN_DIR/original" "$TXN_DIR/candidate" "$TXN_DIR/canary"
TARGET_CODEX_EXISTED=1
TARGET_CLAUDE_EXISTED=1
TARGET_CURSOR_EXISTED=0
TARGET_GROK_EXISTED=0
cp "$TARGET_CODEX" "$TXN_DIR/original/codex-hooks.json"
cp "$TARGET_CLAUDE" "$TXN_DIR/original/claude-settings.json"
if [[ -f "$TARGET_CURSOR" ]]; then
  TARGET_CURSOR_EXISTED=1
  cp "$TARGET_CURSOR" "$TXN_DIR/original/cursor-hooks.json"
fi
if [[ -f "$TARGET_GROK" ]]; then
  TARGET_GROK_EXISTED=1
  cp "$TARGET_GROK" "$TXN_DIR/original/grok-hooks.json"
fi
cp "$SOURCE_CODEX" "$TXN_DIR/candidate/codex-hooks.json"
cp "$SOURCE_CLAUDE" "$TXN_DIR/candidate/claude-settings.json"
cp "$SOURCE_CURSOR" "$TXN_DIR/candidate/cursor-hooks.json"
cp "$SOURCE_GROK" "$TXN_DIR/candidate/grok-hooks.json"
CANDIDATE_CODEX="$TXN_DIR/candidate/codex-hooks.json"
CANDIDATE_CLAUDE="$TXN_DIR/candidate/claude-settings.json"
CANDIDATE_CURSOR="$TXN_DIR/candidate/cursor-hooks.json"
CANDIDATE_GROK="$TXN_DIR/candidate/grok-hooks.json"
validate_candidate_config "$CANDIDATE_CODEX" "$CANDIDATE_CLAUDE" \
  "$CANDIDATE_CURSOR" "$CANDIDATE_GROK"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$SOURCE_HEAD" && \
  "$(sha256sum "$SOURCE_CODEX" | awk '{print $1}')" == \
    "$(sha256sum "$CANDIDATE_CODEX" | awk '{print $1}')" && \
  "$(sha256sum "$SOURCE_CLAUDE" | awk '{print $1}')" == \
    "$(sha256sum "$CANDIDATE_CLAUDE" | awk '{print $1}')" && \
  "$(sha256sum "$SOURCE_CURSOR" | awk '{print $1}')" == \
    "$(sha256sum "$CANDIDATE_CURSOR" | awk '{print $1}')" && \
  "$(sha256sum "$SOURCE_GROK" | awk '{print $1}')" == \
    "$(sha256sum "$CANDIDATE_GROK" | awk '{print $1}')" ]] ||
  die 'source hook configuration changed while the candidate was frozen'
TARGET_CODEX_MODE="$(stat -c %a "$TARGET_CODEX")"
TARGET_CLAUDE_MODE="$(stat -c %a "$TARGET_CLAUDE")"
TARGET_CURSOR_MODE="$(stat -c %a "$SOURCE_CURSOR")"
TARGET_GROK_MODE="$(stat -c %a "$SOURCE_GROK")"
[[ "$TARGET_CURSOR_EXISTED" -eq 0 ]] || TARGET_CURSOR_MODE="$(stat -c %a "$TARGET_CURSOR")"
[[ "$TARGET_GROK_EXISTED" -eq 0 ]] || TARGET_GROK_MODE="$(stat -c %a "$TARGET_GROK")"

INDEX_LOCK="$TARGET_GIT_DIR/index.lock"
INDEX_LOCK_TOKEN="loom-native-hook-promotion:$TXN_ID"
if ! (set -C; printf '%s\n' "$INDEX_LOCK_TOKEN" > "$INDEX_LOCK") 2>/dev/null; then
  die "target Git index is already locked: $INDEX_LOCK"
fi
INDEX_LOCK_OWNED=1
TRANSACTION_OPEN=0
CANARY_PID=''

cleanup() {
  local status=$?
  set +e
  if [[ -n "${CANARY_PID:-}" ]] && kill -0 "$CANARY_PID" 2>/dev/null; then
    : > "${CANARY_ROOT:-$TXN_DIR/canary}/continue"
    wait "$CANARY_PID" 2>/dev/null || true
  fi
  if [[ "${TRANSACTION_OPEN:-0}" -eq 1 ]]; then
    rollback_owned_file "$CANDIDATE_CODEX" \
      "$TXN_DIR/original/codex-hooks.json" "$TARGET_CODEX" \
      "$TARGET_CODEX_MODE" "$TARGET_CODEX_EXISTED" || true
    rollback_owned_file "$CANDIDATE_CLAUDE" \
      "$TXN_DIR/original/claude-settings.json" "$TARGET_CLAUDE" \
      "$TARGET_CLAUDE_MODE" "$TARGET_CLAUDE_EXISTED" || true
    rollback_owned_file "$CANDIDATE_CURSOR" \
      "$TXN_DIR/original/cursor-hooks.json" "$TARGET_CURSOR" \
      "$TARGET_CURSOR_MODE" "$TARGET_CURSOR_EXISTED" || true
    rollback_owned_file "$CANDIDATE_GROK" \
      "$TXN_DIR/original/grok-hooks.json" "$TARGET_GROK" \
      "$TARGET_GROK_MODE" "$TARGET_GROK_EXISTED" || true
    printf 'ROLLED_BACK transaction=%s\n' "$TXN_ID" >&2
  fi
  if [[ "${INDEX_LOCK_OWNED:-0}" -eq 1 && -f "$INDEX_LOCK" && \
    "$(cat "$INDEX_LOCK" 2>/dev/null || true)" == "$INDEX_LOCK_TOKEN" ]]; then
    rm -f "$INDEX_LOCK"
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

[[ "$(git -C "$TARGET_ROOT" rev-parse HEAD)" == "$TARGET_HEAD" ]] ||
  die 'target HEAD changed before promotion lock acquisition'
git -C "$TARGET_ROOT" diff --quiet -- \
  .codex/hooks.json .claude/settings.json .cursor/hooks.json \
  .grok/hooks/loom-native.json ||
  die 'target hooks changed before promotion lock acquisition'

TRANSACTION_OPEN=1
mkdir -p "$(dirname "$TARGET_CURSOR")" "$(dirname "$TARGET_GROK")"
atomic_copy "$CANDIDATE_CODEX" "$TARGET_CODEX"
chmod "$TARGET_CODEX_MODE" "$TARGET_CODEX"
atomic_copy "$CANDIDATE_CLAUDE" "$TARGET_CLAUDE"
chmod "$TARGET_CLAUDE_MODE" "$TARGET_CLAUDE"
atomic_copy "$CANDIDATE_CURSOR" "$TARGET_CURSOR"
chmod "$TARGET_CURSOR_MODE" "$TARGET_CURSOR"
atomic_copy "$CANDIDATE_GROK" "$TARGET_GROK"
chmod "$TARGET_GROK_MODE" "$TARGET_GROK"
validate_candidate_config "$TARGET_CODEX" "$TARGET_CLAUDE" \
  "$TARGET_CURSOR" "$TARGET_GROK"
[[ "$(sha256sum "$TARGET_CODEX" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CODEX" | awk '{print $1}')" ]] ||
  die 'promoted Codex configuration differs from the candidate'
[[ "$(sha256sum "$TARGET_CLAUDE" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CLAUDE" | awk '{print $1}')" ]] ||
  die 'promoted Claude configuration differs from the candidate'
[[ "$(sha256sum "$TARGET_CURSOR" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CURSOR" | awk '{print $1}')" ]] ||
  die 'promoted Cursor configuration differs from the candidate'
[[ "$(sha256sum "$TARGET_GROK" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_GROK" | awk '{print $1}')" ]] ||
  die 'promoted Grok configuration differs from the candidate'

if [[ "${SOUNIO_LOOM_NATIVE_HOOK_PROMOTION_SABOTAGE_AFTER_SWAP:-0}" == 1 ]]; then
  SELFTEST_TMP="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
  case "$TARGET_ROOT" in
    "$SELFTEST_TMP"/sounio-loom-native-hook-promotion.*) die 'sabotage-after-swap' ;;
    *) die 'promotion sabotage is restricted to its temporary selftest root' ;;
  esac
fi

CANARY_ROOT="$TXN_DIR/canary"
CANARY_REPO="$CANARY_ROOT/policyless"
CANARY_RECEIPT="$CANARY_REPO/.git/sounio-loom-language-authority/agent-hook.tsv"
mkdir -p "$CANARY_REPO/bin"
git init -q "$CANARY_REPO"
cp "$TARGET_ROOT/bin/sounio-coord" "$CANARY_REPO/bin/"

run_provider_canary() {
  local provider="$1" harness_name="$2" dialect="$3"
  local provider_root="$CANARY_ROOT/$provider"
  local harness="$provider_root/$harness_name"
  local session="promotion-$provider-$RANDOM-$RANDOM-$$"
  local lane="session-${session:0:24}"
  local hook_command start_event prompt_event end_event capability ready=0

  mkdir -p "$provider_root"
  cp "$(command -v bash)" "$harness"
  chmod 0755 "$harness"
  printf '%s\n' "$lane" > "$provider_root/lane"
  hook_command='runtime_dir="$(readlink -f "$SOUNIO_COORD_RUNTIME_DIR/current")" && test -n "$runtime_dir" && exec env SOUNIO_LOOM_LANGUAGE_AUTHORITY_ROOT="$runtime_dir/policy/language-authority" SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_ROOT="$runtime_dir/policy/native-hook-cutover" "$runtime_dir/bin/sounio-loom-runtime" agent-hook --agent '"$provider"
  case "$provider" in
    codex)
      printf -v start_event '{"hook_event_name":"SessionStart","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      printf -v prompt_event '{"hook_event_name":"UserPromptSubmit","session_id":"%s","cwd":"%s","prompt":"native hook promotion canary"}' \
        "$session" "$CANARY_REPO"
      printf -v end_event '{"hook_event_name":"Stop","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      ;;
    claude)
      printf -v start_event '{"hook_event_name":"SessionStart","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      printf -v prompt_event '{"hook_event_name":"UserPromptSubmit","session_id":"%s","cwd":"%s","prompt":"native hook promotion canary"}' \
        "$session" "$CANARY_REPO"
      printf -v end_event '{"hook_event_name":"SessionEnd","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      ;;
    cursor)
      printf -v start_event '{"hook_event_name":"sessionStart","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      printf -v prompt_event '{"hook_event_name":"beforeSubmitPrompt","session_id":"%s","cwd":"%s","prompt":"native hook promotion canary"}' \
        "$session" "$CANARY_REPO"
      printf -v end_event '{"hook_event_name":"sessionEnd","session_id":"%s","cwd":"%s"}' \
        "$session" "$CANARY_REPO"
      ;;
    grok)
      printf -v start_event '{"hookEventName":"session_start","sessionId":"%s","cwd":"%s","workspaceRoot":"%s"}' \
        "$session" "$CANARY_REPO" "$CANARY_REPO"
      printf -v prompt_event '{"hookEventName":"user_prompt_submit","sessionId":"%s","cwd":"%s","workspaceRoot":"%s","prompt":"native hook promotion canary"}' \
        "$session" "$CANARY_REPO" "$CANARY_REPO"
      printf -v end_event '{"hookEventName":"session_end","sessionId":"%s","cwd":"%s","workspaceRoot":"%s"}' \
        "$session" "$CANARY_REPO" "$CANARY_REPO"
      ;;
    *) die "unsupported promotion canary provider: $provider" ;;
  esac

  SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
  TMUX='' TMUX_PANE='' SOUNIO_AGENTD_SOCKET='' SOUNIO_AGENTD_TOKEN_FILE='' \
  "$harness" -c '
    root="$1"; hook="$2"; start="$3"; prompt="$4"; finish="$5"
    printf "%s\n" "$start" | /bin/bash -c "$hook" >"$root/start.out" 2>"$root/start.err"
    printf "%s\n" "$?" >"$root/start.rc"
    : >"$root/ready"
    while [[ ! -e "$root/continue" ]]; do sleep 0.05; done
    printf "%s\n" "$prompt" | /bin/bash -c "$hook" >"$root/prompt.out" 2>"$root/prompt.err"
    printf "%s\n" "$?" >"$root/prompt.rc"
    printf "%s\n" "$finish" | /bin/bash -c "$hook" >"$root/end.out" 2>"$root/end.err"
    printf "%s\n" "$?" >"$root/end.rc"
  ' _ "$provider_root" "$hook_command" "$start_event" "$prompt_event" "$end_event" &
  CANARY_PID=$!
  for _ in $(seq 1 400); do
    if [[ -e "$provider_root/ready" ]]; then ready=1; break; fi
    if ! kill -0 "$CANARY_PID" 2>/dev/null; then break; fi
    sleep 0.05
  done
  if [[ "$ready" -ne 1 || "$(cat "$provider_root/start.rc" 2>/dev/null || true)" != 0 ]] ||
     [[ -s "$provider_root/start.err" ]] ||
     ! grep -Fq 'Sounio coordination joined:' "$provider_root/start.out"; then
    : > "$provider_root/continue"
    wait "$CANARY_PID" || true
    CANARY_PID=''
    die "policyless $provider promotion canary failed at SessionStart"
  fi
  capability="$(
    SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
    "$CANARY_REPO/bin/sounio-coord" hook-capability-status \
      --agent "$provider" --lane "$lane"
  )"
  grep -Fq 'state=NATIVE_HOOK_ATTESTED' <<< "$capability" ||
    die "policyless $provider promotion canary omitted native attestation"
  grep -Fq 'wake_eligible=1' <<< "$capability" ||
    die "policyless $provider promotion canary is not wake eligible"
  : > "$provider_root/continue"
  wait "$CANARY_PID"
  CANARY_PID=''
  [[ "$(cat "$provider_root/prompt.rc")" == 0 && \
    "$(cat "$provider_root/end.rc")" == 0 ]] ||
    die "policyless $provider promotion canary failed after SessionStart"
  [[ ! -s "$provider_root/prompt.err" && ! -s "$provider_root/end.err" ]] ||
    die "policyless $provider promotion canary emitted a refusal"

  if [[ "$provider" == codex ]]; then
    SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
      "$CANARY_REPO/bin/sounio-coord" hook-capability-unregister \
        --agent "$provider" --lane "$lane" --session-id "$session" >/dev/null
    SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
      "$CANARY_REPO/bin/sounio-coord" presence-unregister \
        --agent "$provider" --lane "$lane" >/dev/null 2>&1 || true
    SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
      "$CANARY_REPO/bin/sounio-coord" release \
        --agent "$provider" --lane "$lane" --reason 'promotion canary complete' \
        >/dev/null 2>&1 || true
  fi
  if SOUNIO_COORD_RUNTIME_DIR="$RUNTIME_ROOT" SOUNIO_COORD_DIR="$STATE_ROOT" \
    "$CANARY_REPO/bin/sounio-coord" hook-capability-status \
      --agent "$provider" --lane "$lane" >/dev/null 2>&1; then
    die "policyless $provider promotion canary left its capability active"
  fi
  [[ "$(grep -Fc $'provider='"$provider"$'\tdialect='"$dialect" "$CANARY_RECEIPT")" -eq 3 ]] ||
    die "policyless $provider promotion canary did not emit three provider-bound receipts"
}

run_provider_canary codex codex snake
run_provider_canary claude claude snake
run_provider_canary cursor cursor-agent cursor-camel
run_provider_canary grok grok grok-camel

[[ "$(grep -c 'decision=ALLOW' "$CANARY_RECEIPT")" -eq 12 ]] ||
  die 'four-provider policyless canary did not emit twelve ALLOW receipts'
[[ "$(grep -c 'semantic_authority_origin=runtime-capsule' "$CANARY_RECEIPT")" -eq 12 ]] ||
  die 'four-provider policyless canary did not use the runtime authority capsule'
[[ "$(grep -Fc $'result=SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045' "$CANARY_RECEIPT")" -eq 12 ]] ||
  die 'four-provider policyless canary did not bind every event to Sounio action 9045'
[[ "$(git -C "$TARGET_ROOT" rev-parse HEAD)" == "$TARGET_HEAD" ]] ||
  die 'target HEAD changed while the promotion lock was active'
[[ "$(sha256sum "$TARGET_CODEX" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CODEX" | awk '{print $1}')" && \
  "$(sha256sum "$TARGET_CLAUDE" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CLAUDE" | awk '{print $1}')" && \
  "$(sha256sum "$TARGET_CURSOR" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_CURSOR" | awk '{print $1}')" && \
  "$(sha256sum "$TARGET_GROK" | awk '{print $1}')" == \
  "$(sha256sum "$CANDIDATE_GROK" | awk '{print $1}')" ]] ||
  die 'promoted hook configuration changed during the policyless canary'

{
  printf 'schema=loom-native-hook-promotion-receipt-v1\n'
  printf 'transaction=%s\n' "$TXN_ID"
  printf 'source_root=%s\n' "$SOURCE_ROOT"
  printf 'source_commit=%s\n' "$SOURCE_HEAD"
  printf 'target_root=%s\n' "$TARGET_ROOT"
  printf 'target_branch=%s\n' "$TARGET_BRANCH"
  printf 'target_head=%s\n' "$TARGET_HEAD"
  printf 'runtime_id=%s\n' "$(manifest_value "$RUNTIME_MANIFEST" runtime_id)"
  printf 'runtime_bundle_sha256=%s\n' "$(manifest_value "$RUNTIME_MANIFEST" bundle_sha256)"
  printf 'codex_config_sha256=%s\n' "$(sha256sum "$TARGET_CODEX" | awk '{print $1}')"
  printf 'claude_config_sha256=%s\n' "$(sha256sum "$TARGET_CLAUDE" | awk '{print $1}')"
  printf 'cursor_config_sha256=%s\n' "$(sha256sum "$TARGET_CURSOR" | awk '{print $1}')"
  printf 'grok_config_sha256=%s\n' "$(sha256sum "$TARGET_GROK" | awk '{print $1}')"
  for provider in codex claude cursor grok; do
    printf 'canary_%s_lane=%s\n' "$provider" "$(cat "$CANARY_ROOT/$provider/lane")"
  done
  printf 'canary_receipt_sha256=%s\n' "$(sha256sum "$CANARY_RECEIPT" | awk '{print $1}')"
  printf 'canary_allow_receipts=12\n'
  printf 'canary_runtime_capsule_receipts=12\n'
  printf 'canary_action_9045_receipts=12\n'
  printf 'canary_providers=codex+claude+cursor+grok\n'
  printf 'result=ACTIVATED\n'
} > "$TXN_DIR/receipt.v1"
sync -f "$TXN_DIR/receipt.v1"

TRANSACTION_OPEN=0
printf 'LOOM_NATIVE_HOOKS_ACTIVATED transaction=%s target=%s runtime_id=%s receipt=%s\n' \
  "$TXN_ID" "$TARGET_ROOT" "$(manifest_value "$RUNTIME_MANIFEST" runtime_id)" \
  "$TXN_DIR/receipt.v1"
