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

# A busy state lock must create bounded backpressure rather than a retry storm.
mkdir -p "$STATE"
exec 9>"$STATE/.claims.lock"
flock -x 9
(
  cd "$REPO"
  run_coord scope --agent lock-waiter --lane contention --intent 'wait for lock'
) >"$TEST_ROOT/lock-wait.out" 2>"$TEST_ROOT/lock-wait.err" &
lock_wait_pid=$!
sleep 0.2
kill -0 "$lock_wait_pid" 2>/dev/null || fail 'contended client did not wait for the state lock'
flock -u 9
wait "$lock_wait_pid" || fail "contended client failed after lock release: $(cat "$TEST_ROOT/lock-wait.err")"
grep -qE '^SCOPED claim_id=lock-waiter--contention' "$TEST_ROOT/lock-wait.out" || \
  fail 'contended client did not progress after lock release'
(
  cd "$REPO"
  run_coord release --agent lock-waiter --lane contention --reason 'lock selftest complete'
) >/dev/null

flock -x 9
set +e
timeout_output="$(
  cd "$REPO"
  SOUNIO_COORD_LOCK_WAIT_SECONDS=1 run_coord scope \
    --agent lock-timeout --lane contention --intent 'must time out' 2>&1
)"
timeout_status=$?
set -e
flock -u 9
[[ $timeout_status -ne 0 ]] || fail 'contended client ignored its state-lock timeout'
grep -q 'state lock timed out after 1s' <<<"$timeout_output" || \
  fail "state-lock timeout was not classified: $timeout_output"

output="$(
  cd "$REPO"
  run_coord claim --agent agent-a --lane parser --ttl-seconds 600 \
    --intent 'parser ownership' --files 'self-hosted/parser/**'
)"
grep -qE '^CLAIMED claim_id=agent-a--parser$' <<< "$output" || fail 'first claim was not created'

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

# A lease taken from a worktree that is later removed must stay reachable to its
# owner. Before the orphan handling existed, both release and scope died with
# "claim belongs to worktree <gone>" and only TTL expiry could clear the lease.
git -C "$REPO" worktree add -q --detach "$TEST_ROOT/throwaway" HEAD
(
  cd "$TEST_ROOT/throwaway"
  run_coord scope --agent agent-c --lane orphan --intent 'orphan probe'
) >/dev/null
git -C "$REPO" worktree remove --force "$TEST_ROOT/throwaway"

output="$(
  cd "$REPO"
  run_coord release --agent agent-c --lane orphan --reason 'worktree removed'
)"
grep -qE '^RELEASED claim_id=agent-c--orphan' <<< "$output" || fail 'orphaned claim could not be released'

git -C "$REPO" worktree add -q --detach "$TEST_ROOT/throwaway" HEAD
(
  cd "$TEST_ROOT/throwaway"
  run_coord scope --agent agent-c --lane orphan2 --intent 'orphan probe'
) >/dev/null
git -C "$REPO" worktree remove --force "$TEST_ROOT/throwaway"

(
  cd "$REPO"
  run_coord scope --agent agent-c --lane orphan2 --intent 'retaken here'
) >/dev/null || fail 'orphaned claim could not be re-scoped'
grep -q 'event=ORPHANED' "$STATE/events.log" || fail 'orphan adoption was not recorded'

# The guard itself must survive: a claim whose worktree still exists stays
# protected, and another agent still cannot take over an orphan.
git -C "$REPO" worktree add -q --detach "$TEST_ROOT/throwaway" HEAD
(
  cd "$TEST_ROOT/throwaway"
  run_coord scope --agent agent-c --lane live --intent 'still alive'
) >/dev/null
if (
  cd "$REPO"
  run_coord release --agent agent-c --lane live --reason 'should be refused'
) >/dev/null 2>&1; then
  fail 'claim was released while its worktree still existed'
fi
git -C "$REPO" worktree remove --force "$TEST_ROOT/throwaway"
if (
  cd "$REPO"
  run_coord release --agent agent-d --lane live --reason 'hijack attempt'
) >/dev/null 2>&1; then
  fail 'a different agent released an orphaned claim'
fi

(
  cd "$REPO"
  run_coord release --agent agent-c --lane orphan2 --reason 'selftest complete'
  run_coord release --agent agent-c --lane live --reason 'selftest complete'
) >/dev/null

echo 'sounio-coord-selftest: PASS'
