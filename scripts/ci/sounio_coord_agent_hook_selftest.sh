#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-hook-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/state"

cleanup() {
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-agent-hook-selftest: FAIL: $*" >&2
  exit 1
}

mkdir -p "$REPO/bin" "$REPO/scripts/dev" "$REPO/self-hosted/parser"
cp "$ROOT_DIR/bin/sounio-coord" "$REPO/bin/sounio-coord"
cp "$ROOT_DIR/scripts/dev/sounio_coord_agent_hook.py" "$REPO/scripts/dev/"
chmod +x "$REPO/bin/sounio-coord" "$REPO/scripts/dev/sounio_coord_agent_hook.py"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Hook Selftest'
git -C "$REPO" config user.email 'coord-hook-selftest@sounio.local'
printf 'seed\n' > "$REPO/README.md"
printf 'parser\n' > "$REPO/self-hosted/parser/ast.sio"
printf 'items\n' > "$REPO/self-hosted/parser/items.sio"
git -C "$REPO" add .
git -C "$REPO" commit -qm seed
git -C "$REPO" worktree add -q -b second-lane "$SECOND"

run_hook() {
  local agent="$1" cwd="$2" payload="$3"
  printf '%s\n' "$payload" | SOUNIO_COORD_DIR="$STATE" \
    python3 "$cwd/scripts/dev/sounio_coord_agent_hook.py" --agent "$agent"
}

run_coord() {
  local cwd="$1"
  shift
  (cd "$cwd" && SOUNIO_COORD_DIR="$STATE" bin/sounio-coord "$@")
}

output="$(run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"SessionStart\"}")"
# Lanes are scoped to (session, worktree) -- see sounio_coord_agent_hook.py and
# issue #1477. The suffix is a hash of the worktree path, so the selftest must
# READ the lane the hook actually registered instead of hardcoding it; two
# worktrees of the same session deliberately get different lanes.
grep -q 'agent=codex lane=session-codex-a' <<< "$output" || \
  fail 'Codex session identity was not injected'
CODEX_LANE="$(sed -n 's/.*agent=codex lane=\(session-codex-a[A-Za-z0-9_-]*\).*/\1/p' <<< "$output" | head -1)"
[[ -n "$CODEX_LANE" ]] || fail 'could not read the codex lane from hook output'
output="$(run_coord "$REPO" brief --max-rows 4)"
grep -q 'ACTIVE claim_id=codex--session-codex-a' <<< "$output" || \
  fail 'Codex session presence was not registered'

run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"apply_patch\",\"tool_input\":{\"patch\":\"*** Update File: self-hosted/parser/ast.sio\"}}"
run_hook codex "$REPO" \
  "{\"session_id\":\"codex-a\",\"cwd\":\"$REPO\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"apply_patch\",\"tool_input\":{\"patch\":\"*** Update File: self-hosted/parser/items.sio\"}}"
output="$(run_coord "$REPO" brief --max-rows 4)"
grep -q 'files=self-hosted/parser/ast.sio,self-hosted/parser/items.sio' <<< "$output" || \
  fail 'automatic write scope did not accumulate files'

output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionStart\"}")"
grep -q 'agent=claude lane=session-claude-b' <<< "$output" || \
  fail 'Claude session identity was not injected'
CLAUDE_LANE="$(sed -n 's/.*agent=claude lane=\(session-claude-b[A-Za-z0-9_-]*\).*/\1/p' <<< "$output" | head -1)"
[[ -n "$CLAUDE_LANE" ]] || fail 'could not read the claude lane from hook output'
# The two lanes must differ: SECOND is a different worktree than REPO.
[[ "$CLAUDE_LANE" != "$CODEX_LANE" ]] || fail 'lanes in different worktrees collided'
send_output="$(run_coord "$REPO" send --agent codex --lane "$CODEX_LANE" \
  --to-agent claude --to-lane "$CLAUDE_LANE" --kind request \
  --message 'Please review the parser ownership boundary')"
message_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$send_output")"
[[ -n "$message_id" ]] || fail 'message id was not returned'

output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"PostToolUse\",\"tool_name\":\"Read\",\"tool_input\":{}}")"
grep -q "MESSAGE id=$message_id" <<< "$output" || fail 'message was not delivered to Claude'
run_coord "$SECOND" ack --agent claude --lane "$CLAUDE_LANE" --message "$message_id" >/dev/null
output="$(run_coord "$SECOND" inbox --agent claude --lane "$CLAUDE_LANE")"
grep -q '^inbox_messages=0$' <<< "$output" || fail 'acknowledged message remained unread'

set +e
conflict_output="$(run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"PreToolUse\",\"tool_name\":\"Write\",\"tool_input\":{\"file_path\":\"self-hosted/parser/ast.sio\"}}" 2>&1)"
conflict_rc=$?
set -e
[[ "$conflict_rc" -eq 2 ]] || fail "conflicting write returned $conflict_rc instead of 2"
grep -q 'requested file set overlaps an active claim' <<< "$conflict_output" || \
  fail 'conflicting write did not explain the ownership collision'
output="$(run_coord "$REPO" inbox --agent codex --lane "$CODEX_LANE")"
grep -q "kind=request text=Write conflict requested by claude/$CLAUDE_LANE" <<< "$output" || \
  fail 'conflict owner did not receive an automatic message'

run_hook claude "$SECOND" \
  "{\"session_id\":\"claude-b\",\"cwd\":\"$SECOND\",\"hook_event_name\":\"SessionEnd\"}"
output="$(run_coord "$SECOND" brief --max-rows 4)"
grep -q 'ACTIVE claim_id=claude--session-claude-b' <<< "$output" && \
  fail 'Claude session claim survived SessionEnd'

run_coord "$REPO" send --agent codex --lane "$CODEX_LANE" --kind info \
  --ttl-seconds 1 --message 'ephemeral selftest message' >/dev/null
sleep 2
output="$(run_coord "$REPO" prune)"
grep -q 'pruned_messages=1$' <<< "$output" || fail 'expired message was not pruned'

echo 'sounio-coord-agent-hook-selftest: PASS'
