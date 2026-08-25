#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$ROOT_DIR/bin/sounio-coord"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-semantic-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
SECOND="$TEST_ROOT/second-worktree"
STATE="$TEST_ROOT/state"

cleanup() {
  git -C "$REPO" worktree remove --force "$SECOND" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  echo "sounio-coord-semantic-selftest: FAIL: $*" >&2
  exit 1
}

run_coord() {
  local cwd="$1"
  shift
  (cd "$cwd" && SOUNIO_COORD_DIR="$STATE" SOUNIO_COORD_RUNTIME_MODE=local \
    SOUNIO_COORD_DURABLE_OBLIGATIONS=0 "$TOOL" "$@")
}

mkdir -p "$REPO/semantic"
git -C "$REPO" init -q
git -C "$REPO" config user.name 'Sounio Semantic Coordination Selftest'
git -C "$REPO" config user.email 'coord-semantic-selftest@sounio.local'
printf 'semantic a\n' > "$REPO/semantic/a.txt"
printf 'semantic b\n' > "$REPO/semantic/b.txt"
git -C "$REPO" add semantic/a.txt semantic/b.txt
git -C "$REPO" commit -qm 'seed semantic surfaces'
printf 'gate receipt\n' > "$REPO/semantic/gate.receipt"
git -C "$REPO" add semantic/gate.receipt
git -C "$REPO" commit -qm 'add gate receipt'
git -C "$REPO" worktree add -q -b semantic-consumer "$SECOND"

output="$(run_coord "$REPO" claim --agent agent-a --lane semantic --ttl-seconds 600 \
  --intent 'own epistemic meaning and diagnostic' \
  --resources 'concept:epistemic/**' diagnostic:E230 --files semantic/a.txt)"
grep -q '^CLAIMED claim_id=agent-a--semantic$' <<< "$output" || fail 'semantic claim was not created'
grep -Fq 'resources=concept:epistemic/**,diagnostic:E230' <<< "$output" || \
  fail 'claim did not report its typed resources'

output="$(run_coord "$REPO" authorize --agent agent-a --lane semantic \
  --resources concept:epistemic/noise diagnostic:E230 --files semantic/a.txt)"
grep -q '^AUTHORIZED claim_id=agent-a--semantic ' <<< "$output" || \
  fail 'wildcard semantic owner was not authorized for its child concept'

if run_coord "$SECOND" authorize --agent agent-a --lane semantic \
  --resources concept:epistemic/noise --files semantic/a.txt >/dev/null 2>&1; then
  fail 'semantic ownership authorized the wrong worktree'
fi

set +e
conflict_output="$(run_coord "$SECOND" claim --agent agent-b --lane same-concept \
  --ttl-seconds 600 --intent 'must conflict despite disjoint file' \
  --resources concept:epistemic/noise --files semantic/b.txt 2>&1)"
conflict_rc=$?
set -e
[[ "$conflict_rc" -ne 0 ]] || fail 'child concept claim bypassed its wildcard owner'
grep -Fq 'resource=concept:epistemic/** requested_resource=concept:epistemic/noise' \
  <<< "$conflict_output" || fail 'concept conflict did not identify the enforcing resource'
grep -q 'requested semantic resource set overlaps an active claim' <<< "$conflict_output" || \
  fail 'concept conflict did not identify the semantic rule'

set +e
conflict_output="$(run_coord "$SECOND" claim --agent agent-b --lane same-diagnostic \
  --ttl-seconds 600 --intent 'must conflict on diagnostic id' \
  --resources diagnostic:E230 --files semantic/b.txt 2>&1)"
conflict_rc=$?
set -e
[[ "$conflict_rc" -ne 0 ]] || fail 'duplicate diagnostic claim was accepted'
grep -Fq 'resource=diagnostic:E230 requested_resource=diagnostic:E230' \
  <<< "$conflict_output" || fail 'diagnostic conflict did not identify E230'

output="$(run_coord "$SECOND" claim --agent agent-b --lane independent \
  --ttl-seconds 600 --intent 'disjoint semantic ownership' \
  --resources concept:codegen --files semantic/b.txt)"
grep -q '^CLAIMED claim_id=agent-b--independent$' <<< "$output" || \
  fail 'unrelated semantic claim was rejected'
output="$(run_coord "$SECOND" authorize --agent agent-b --lane independent \
  --resources concept:codegen --files semantic/b.txt)"
grep -q '^AUTHORIZED claim_id=agent-b--independent ' <<< "$output" || \
  fail 'unrelated semantic owner was not authorized'
run_coord "$SECOND" release --agent agent-b --lane independent \
  --reason 'semantic independence established' >/dev/null

output="$(run_coord "$SECOND" claim --agent agent-b --lane gate-only \
  --ttl-seconds 600 --intent 'resource-only gate ownership' \
  --resources gate:semantic-gate)"
grep -q '^files=$' <<< "$output" || fail 'resource-only claim unexpectedly required a file'
grep -q '^resources=gate:semantic-gate$' <<< "$output" || \
  fail 'resource-only claim did not preserve its gate resource'
output="$(run_coord "$SECOND" authorize --agent agent-b --lane gate-only \
  --resources gate:semantic-gate)"
grep -q '^AUTHORIZED claim_id=agent-b--gate-only ' <<< "$output" || \
  fail 'resource-only claim did not authorize its owner'
run_coord "$SECOND" release --agent agent-b --lane gate-only \
  --reason 'resource-only ownership established' >/dev/null

request_output="$(run_coord "$SECOND" send --agent agent-b --lane consumer \
  --to-agent agent-a --to-lane semantic --kind request \
  --message 'Deliver the accepted epistemic change')"
request_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$request_output")"
[[ -n "$request_id" ]] || fail 'handoff request did not return a message id'

if run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'missing gate sabotage' \
  --commit HEAD --evidence semantic/gate.receipt --reply-to "$request_id" >/dev/null 2>&1; then
  fail 'handoff without a gate was accepted'
fi
if run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'missing evidence sabotage' \
  --commit HEAD --gate semantic-gate=PASS --reply-to "$request_id" >/dev/null 2>&1; then
  fail 'handoff without evidence was accepted'
fi
if run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'failed gate sabotage' \
  --commit HEAD --gate semantic-gate=FAIL --evidence semantic/gate.receipt \
  --reply-to "$request_id" >/dev/null 2>&1; then
  fail 'handoff with a non-passing gate was accepted'
fi
if run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'stale commit sabotage' \
  --commit HEAD^ --gate semantic-gate=PASS --evidence semantic/gate.receipt \
  --reply-to "$request_id" >/dev/null 2>&1; then
  fail 'handoff for a non-HEAD commit was accepted'
fi
printf 'dirty semantic state\n' > "$REPO/semantic/a.txt"
if run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'dirty claim sabotage' \
  --commit HEAD --gate semantic-gate=PASS --evidence semantic/gate.receipt \
  --reply-to "$request_id" >/dev/null 2>&1; then
  fail 'handoff with uncommitted claimed-file changes was accepted'
fi
git -C "$REPO" show HEAD:semantic/a.txt > "$REPO/semantic/a.txt"

output="$(run_coord "$REPO" authorize --agent agent-a --lane semantic \
  --resources concept:epistemic/noise diagnostic:E230 --files semantic/a.txt)"
grep -q '^AUTHORIZED claim_id=agent-a--semantic ' <<< "$output" || \
  fail 'a refused handoff released the semantic claim'
output="$(run_coord "$SECOND" inbox --agent agent-b --lane consumer --kind handoff)"
grep -q '^inbox_messages=0$' <<< "$output" || fail 'a refused handoff published a message'

head_sha="$(git -C "$REPO" rev-parse HEAD)"
output="$(run_coord "$REPO" handoff --agent agent-a --lane semantic \
  --to-agent agent-b --to-lane consumer --message 'epistemic change accepted' \
  --commit HEAD --gate semantic-gate=PASS --evidence semantic/gate.receipt \
  --reply-to "$request_id")"
handoff_id="$(sed -n 's/^HANDED_OFF .* message_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$handoff_id" ]] || fail 'valid handoff did not return a message id'
grep -q "commit=$head_sha" <<< "$output" || fail 'handoff did not report the exact commit'

if run_coord "$REPO" authorize --agent agent-a --lane semantic \
  --resources concept:epistemic/noise --files semantic/a.txt >/dev/null 2>&1; then
  fail 'valid handoff left the source claim active'
fi
output="$(run_coord "$SECOND" inbox --agent agent-b --lane consumer --kind handoff)"
grep -q "MESSAGE id=$handoff_id .*kind=handoff text=epistemic change accepted" <<< "$output" || \
  fail 'recipient did not receive the handoff'
grep -q "thread=$request_id reply_to=$request_id commit=$head_sha" <<< "$output" || \
  fail 'handoff did not preserve request correlation and commit identity'
grep -Fq 'gates=semantic-gate=PASS evidence=semantic/gate.receipt files=semantic/a.txt resources=concept:epistemic/**,diagnostic:E230' \
  <<< "$output" || fail 'handoff did not carry its proof and ownership snapshot'
output="$(run_coord "$SECOND" message-status --agent agent-b --lane consumer \
  --message "$request_id")"
grep -q "request_state=answered .*responses=1 latest_response=$handoff_id" <<< "$output" || \
  fail 'proof-carrying handoff did not close the request lifecycle'

echo 'sounio-coord-semantic-selftest: PASS'
