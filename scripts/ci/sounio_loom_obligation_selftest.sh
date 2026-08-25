#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="${SOUNIO_LOOM_BIN:-$ROOT_DIR/tools/loom/_build/default/src/loom.exe}"
ADAPTER="${SOUNIO_LOOM_OBLIGATION_ADAPTER:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-obligation-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-obligation-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$LOOM" ]] || fail "Loom binary is missing: $LOOM"
[[ -x "$ADAPTER" ]] || fail "Sounio obligation adapter is missing: $ADAPTER"
export SOUNIO_LOOM_OBLIGATION_ADAPTER="$ADAPTER"

state="$WORK/state"
message_digest="$(printf 'durable request body' | sha256sum | awk '{print $1}')"
printf 'bounded outcome\n' > "$WORK/outcome.txt"
printf 'independent evidence\n' > "$WORK/evidence.txt"

run() { "$LOOM" "$@" --state-dir "$state"; }

run obligation-open --message msg-obligation-1 --message-digest "$message_digest" \
  --from-agent sender --from-lane sender-lane --to-agent worker --to-lane work-lane \
  > "$WORK/open.log"
grep -q 'state=durable .*unclosed=yes' "$WORK/open.log" || fail 'open did not persist durable state'

run obligation-open --message msg-obligation-1 --message-digest "$message_digest" \
  --from-agent sender --from-lane sender-lane --to-agent worker --to-lane work-lane \
  > "$WORK/open-idempotent.log"
grep -q '^LOOM_OBLIGATION_OPEN idempotent=yes ' "$WORK/open-idempotent.log" || \
  fail 'repeated open was not idempotent'

run obligation-consume --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-1 > "$WORK/consume.log"
run obligation-claim --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-1 --claim claim-1 --ttl-seconds 120 > "$WORK/claim.log"
grep -q 'state=claimed .*generation=generation-1 .*claim=claim-1 .*lease=active' \
  "$WORK/claim.log" || fail 'first generation did not acquire the claim'

set +e
run obligation-claim --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-2 --claim claim-rival --ttl-seconds 120 \
  > "$WORK/rival.log" 2>&1
rival_rc=$?
set -e
[[ "$rival_rc" -eq 1 ]] || fail "second live generation acquired the claim rc=$rival_rc"
grep -q 'obligation-claim-invalid-state:claimed' "$WORK/rival.log" || \
  fail 'second live generation refused for the wrong reason'

set +e
run obligation-complete --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-1 --claim claim-1 --outcome "$WORK/outcome.txt" \
  --evidence "$WORK/missing-evidence.txt" > "$WORK/missing.log" 2>&1
missing_rc=$?
set -e
[[ "$missing_rc" -eq 1 ]] || fail 'completion without evidence succeeded'
grep -q 'obligation-evidence-missing' "$WORK/missing.log" || \
  fail 'completion without evidence refused for the wrong reason'

run obligation-interrupt --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-1 --claim claim-1 --reason simulated-total-loss \
  > "$WORK/interrupt.log"
run obligation-recover --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-2 > "$WORK/recover.log"
grep -q 'state=recoverable .*generation=generation-2 .*predecessor_claim=claim-1' \
  "$WORK/recover.log" || fail 'new generation did not recover the predecessor claim'

run obligation-claim --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-2 --claim claim-2 --ttl-seconds 120 > "$WORK/reclaim.log"

set +e
run obligation-complete --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-1 --claim claim-1 --outcome "$WORK/outcome.txt" \
  --evidence "$WORK/evidence.txt" > "$WORK/stale-complete.log" 2>&1
stale_rc=$?
set -e
[[ "$stale_rc" -eq 1 ]] || fail 'interrupted generation completed after recovery'
grep -q 'obligation-complete-current-claim-required' "$WORK/stale-complete.log" || \
  fail 'stale completion refused for the wrong reason'

run obligation-complete --message msg-obligation-1 --actor worker --lane work-lane \
  --generation generation-2 --claim claim-2 --outcome "$WORK/outcome.txt" \
  --evidence "$WORK/evidence.txt" > "$WORK/complete.log"
grep -q 'state=completed .*unclosed=no .*generation=generation-2 .*claim=claim-2 ' \
  "$WORK/complete.log" || fail 'recovered generation did not complete'

run obligation-verify --message msg-obligation-1 > "$WORK/verify.log"
grep -q 'events=7 hash_chain=PASS semantics=PASS' "$WORK/verify.log" || \
  fail 'journal verification omitted the exact seven-event chain'
run obligation-list --json > "$WORK/list.json"
grep -q '"count":1,"unclosed":0' "$WORK/list.json" || \
  fail 'completed obligation remained in the unclosed projection'

printf 'loom-obligation-selftest: PASS durable=1 idempotent_open=1 exclusive_generation=1 evidence_required=1 interrupted_owner_fenced=1 recovered=1 events=7\n'
