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

echo 'sounio-coord-selftest: PASS'
