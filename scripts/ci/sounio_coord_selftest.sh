#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$ROOT_DIR/bin/sounio-coord"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
STATE="$TEST_ROOT/state"

cleanup() {
  git -C "$REPO" worktree remove --force "$TEST_ROOT/second-worktree" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-selftest: FAIL: $*" >&2
  exit 1
}

bash -n "$TOOL"
mkdir -p "$REPO/self-hosted/parser" "$REPO/self-hosted/codegen"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Coordination Selftest'
git -C "$REPO" config user.email 'coord-selftest@sounio.local'
printf 'seed\n' > "$REPO/README.md"
git -C "$REPO" add README.md
git -C "$REPO" commit -qm 'seed'
git -C "$REPO" worktree add -q -b second-lane "$TEST_ROOT/second-worktree"

run_coord() {
  SOUNIO_COORD_DIR="$STATE" "$TOOL" "$@"
}

output="$(
  cd "$REPO"
  run_coord claim --agent agent-a --lane parser --ttl-seconds 600 \
    --intent 'parser ownership' --files 'self-hosted/parser/**'
)"
grep -qE '^CLAIMED claim_id=agent-a--parser$' <<< "$output" || fail 'first claim was not created'

output="$(
  cd "$REPO"
  run_coord authorize --agent agent-a --files self-hosted/parser/example.sio
)"
grep -qE '^AUTHORIZED claim_id=agent-a--parser ' <<< "$output" || \
  fail 'owning worktree did not authorize a covered child path'

if (
  cd "$TEST_ROOT/second-worktree"
  run_coord authorize --agent agent-a --files self-hosted/parser/example.sio
) >/dev/null 2>&1; then
  fail 'claim authorized a write from the wrong worktree'
fi

if (
  cd "$TEST_ROOT/second-worktree"
  run_coord claim --agent agent-b --lane parser-child --ttl-seconds 600 \
    --intent 'must conflict' --files self-hosted/parser/example.sio
) >/dev/null 2>&1; then
  fail 'overlapping claim was accepted'
fi

output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord claim --agent agent-b --lane codegen --ttl-seconds 600 \
    --intent 'disjoint ownership' --files 'self-hosted/codegen/**'
)"
grep -qE '^CLAIMED claim_id=agent-b--codegen$' <<< "$output" || fail 'disjoint claim was rejected'

output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord authorize --agent agent-b --lane codegen --files self-hosted/codegen/example.sio
)"
grep -qE '^AUTHORIZED claim_id=agent-b--codegen ' <<< "$output" || \
  fail 'explicit lane authorization rejected a covered path'

if (
  cd "$TEST_ROOT/second-worktree"
  run_coord authorize --agent agent-b --lane codegen --files self-hosted
) >/dev/null 2>&1; then
  fail 'a child claim authorized its parent directory'
fi

output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord brief --max-rows 4
)"
grep -qE 'ACTIVE claim_id=agent-a--parser' <<< "$output" || fail 'claim was not visible across worktrees'

if (
  cd "$TEST_ROOT/second-worktree"
  run_coord release --agent agent-a --lane parser --reason 'wrong worktree'
) >/dev/null 2>&1; then
  fail 'claim was released from a non-owning worktree'
fi

output="$(
  cd "$REPO"
  run_coord heartbeat --agent agent-a --lane parser
)"
grep -qE '^HEARTBEAT claim_id=agent-a--parser' <<< "$output" || fail 'heartbeat failed'

output="$(
  cd "$REPO"
  run_coord check --brief --max-rows 4
)"
grep -qE '^COORDINATION_CHECK=PASS$' <<< "$output" || fail 'coordination check failed'

output="$(
  cd "$REPO"
  run_coord release --agent agent-a --lane parser --reason 'selftest complete'
)"
grep -qE '^RELEASED claim_id=agent-a--parser' <<< "$output" || fail 'first release failed'

output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord release --agent agent-b --lane codegen --reason 'selftest complete'
)"
grep -qE '^RELEASED claim_id=agent-b--codegen' <<< "$output" || fail 'second release failed'

output="$(
  cd "$REPO"
  run_coord brief --max-rows 4
)"
grep -qE '^summary=active_claims:0 stale_claims:0 conflicts:0$' <<< "$output" || fail 'claims remained active'

broadcast_output="$(
  cd "$REPO"
  run_coord send --agent observer --lane announcements --kind info \
    --message 'broadcast must not bury directed work'
)"
broadcast_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$broadcast_output")"
first_output="$(
  cd "$REPO"
  run_coord send --agent agent-a --lane parser --to-agent agent-b --to-lane codegen \
    --kind request --message 'first directed request'
)"
first_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$first_output")"
second_output="$(
  cd "$REPO"
  run_coord send --agent agent-a --lane parser --to-agent agent-b --to-lane codegen \
    --kind request --message 'newest directed request'
)"
second_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$second_output")"
[[ -n "$broadcast_id" && -n "$first_id" && -n "$second_id" ]] || fail 'message ids were not returned'

output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord inbox --agent agent-b --lane codegen --directed-only --newest-first --limit 1
)"
grep -q "MESSAGE id=$second_id .*text=newest directed request" <<< "$output" || \
  fail 'limited directed inbox did not return the newest message'
grep -q "$first_id" <<< "$output" && fail 'limited directed inbox returned an older message body'
grep -q "$broadcast_id" <<< "$output" && fail 'directed inbox included a broadcast'
grep -q '^inbox_matching=2$' <<< "$output" || fail 'directed inbox matching count is wrong'
grep -q '^inbox_omitted=1$' <<< "$output" || fail 'directed inbox omitted count is wrong'

output="$(
  cd "$REPO"
  run_coord message-status --agent agent-a --lane parser --message "$second_id"
)"
grep -q 'request_state=open injected=0 acknowledged=0 responses=0' <<< "$output" || \
  fail 'unanswered request was not reported open'

set +e
output="$(
  cd "$REPO"
  run_coord wait --agent agent-a --lane parser --message "$first_id" --timeout-seconds 0
)"
wait_rc=$?
set -e
[[ "$wait_rc" -eq 3 ]] || fail "reply wait timeout returned $wait_rc instead of 3"
grep -q "^WAIT_TIMEOUT message_id=$first_id timeout_seconds=0$" <<< "$output" || \
  fail 'reply wait did not report its timeout'

reply_output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord send --agent agent-b --lane codegen --kind reply \
    --reply-to "$second_id" --message 'threaded reply'
)"
reply_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$reply_output")"
thread_id="$(sed -n 's/.* thread_id=\([^ ]*\).*/\1/p' <<< "$reply_output")"
[[ "$thread_id" == "$second_id" ]] || fail 'reply did not inherit the request thread'
grep -q 'to_agent=agent-a to_lane=parser' <<< "$reply_output" || \
  fail 'reply did not inherit the request sender as its destination'
output="$(
  cd "$REPO"
  run_coord inbox --agent agent-a --lane parser --thread "$thread_id"
)"
grep -q "MESSAGE id=$reply_id .*kind=reply text=threaded reply thread=$thread_id reply_to=$second_id" <<< "$output" || \
  fail 'threaded reply metadata was not delivered'
output="$(
  cd "$REPO"
  run_coord wait --agent agent-a --lane parser --message "$second_id" --timeout-seconds 0
)"
grep -q "^WAIT_RESPONSE request_id=$second_id request_state=answered$" <<< "$output" || \
  fail 'reply wait did not return the existing response'
output="$(
  cd "$REPO"
  run_coord message-status --agent agent-a --lane parser --message "$second_id"
)"
grep -q "request_state=answered .*responses=1 latest_response=$reply_id" <<< "$output" || \
  fail 'replied request was not reported answered'

blocked_output="$(
  cd "$REPO"
  run_coord send --agent agent-a --lane parser --to-agent agent-b --to-lane codegen \
    --kind request --message 'request that will block'
)"
blocked_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$blocked_output")"
blocker_output="$(
  cd "$TEST_ROOT/second-worktree"
  run_coord send --agent agent-b --lane codegen --kind blocker \
    --reply-to "$blocked_id" --message 'blocked by acceptance gate'
)"
blocker_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$blocker_output")"
output="$(
  cd "$REPO"
  run_coord message-status --agent agent-a --lane parser --message "$blocked_id"
)"
grep -q "request_state=blocked .*responses=1 latest_response=$blocker_id" <<< "$output" || \
  fail 'blocked request was not reported blocked'

echo 'sounio-coord-selftest: PASS'
