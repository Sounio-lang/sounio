#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="${SOUNIO_LOOM_BIN:-$ROOT_DIR/tools/loom/_build/default/src/loom.exe}"
ADAPTER="${SOUNIO_LOOM_OBLIGATION_ADAPTER:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-obligation-runtime}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-obligation-crash.XXXXXX")"
SUPERVISOR_PID=''
GUI_PID=''

cleanup() {
  if [[ -n "$SUPERVISOR_PID" ]]; then kill -9 "$SUPERVISOR_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "$GUI_PID" ]]; then kill -9 "$GUI_PID" >/dev/null 2>&1 || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-obligation-crash-matrix: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -x "$LOOM" ]] || fail "Loom binary is missing: $LOOM"
[[ -x "$ADAPTER" ]] || fail "Sounio obligation adapter is missing: $ADAPTER"
export SOUNIO_LOOM_OBLIGATION_ADAPTER="$ADAPTER"
state="$WORK/state"
message_digest="$(printf 'crash-matrix-request' | sha256sum | awk '{print $1}')"
printf 'outcome\n' > "$WORK/outcome.txt"
printf 'evidence\n' > "$WORK/evidence.txt"

run() { "$LOOM" "$@" --state-dir "$state"; }
open_message() {
  run obligation-open --message "$1" --message-digest "$message_digest" \
    --from-agent sender --from-lane source --to-agent worker --to-lane target >/dev/null
}

# Matrix A: two serialized claim attempts cannot produce two owners.
open_message msg-exclusive
run obligation-consume --message msg-exclusive --actor worker --lane target \
  --generation generation-a >/dev/null
set +e
run obligation-claim --message msg-exclusive --actor worker --lane target \
  --generation generation-a --claim claim-a --ttl-seconds 120 > "$WORK/claim-a.log" 2>&1 &
pid_a=$!
run obligation-claim --message msg-exclusive --actor worker --lane target \
  --generation generation-a --claim claim-b --ttl-seconds 120 > "$WORK/claim-b.log" 2>&1 &
pid_b=$!
wait "$pid_a"; rc_a=$?
wait "$pid_b"; rc_b=$?
set -e
successes=0
[[ "$rc_a" -ne 0 ]] || successes=$((successes + 1))
[[ "$rc_b" -ne 0 ]] || successes=$((successes + 1))
[[ "$successes" -eq 1 ]] || fail "concurrent claims produced $successes winners"
run obligation-status --message msg-exclusive > "$WORK/exclusive-status.log"
grep -Eq 'state=claimed .*claim=claim-(a|b) .*lease=active' \
  "$WORK/exclusive-status.log" || fail 'winning exclusive claim is not replayable'

# Matrix B: a different generation can fence only an expired predecessor.
open_message msg-expiry
run obligation-consume --message msg-expiry --actor worker --lane target \
  --generation generation-1 >/dev/null
run obligation-claim --message msg-expiry --actor worker --lane target \
  --generation generation-1 --claim expiring-claim --ttl-seconds 1 >/dev/null
set +e
run obligation-interrupt --message msg-expiry --actor worker --lane target \
  --generation generation-2 --claim expiring-claim --reason premature-takeover \
  > "$WORK/premature.log" 2>&1
premature_rc=$?
set -e
[[ "$premature_rc" -eq 1 ]] || fail 'live predecessor was interrupted by another generation'
sleep 2
set +e
run obligation-renew --message msg-expiry --actor worker --lane target \
  --generation generation-1 --claim expiring-claim --ttl-seconds 120 \
  > "$WORK/expired-renew.log" 2>&1
expired_renew_rc=$?
set -e
[[ "$expired_renew_rc" -eq 1 ]] || fail 'expired predecessor renewed its claim'
run obligation-interrupt --message msg-expiry --actor worker --lane target \
  --generation generation-2 --claim expiring-claim --reason expired-takeover >/dev/null
run obligation-recover --message msg-expiry --actor worker --lane target \
  --generation generation-2 > "$WORK/expiry-recover.log"
grep -q 'state=recoverable .*generation=generation-2 .*predecessor_claim=expiring-claim' \
  "$WORK/expiry-recover.log" || fail 'expired predecessor did not become recoverable'

# Matrix C: death between consume and claim remains recoverable after the
# consumer lease expires; a live consumer remains fenced against takeover.
open_message msg-consumed-crash
run obligation-consume --message msg-consumed-crash --actor worker --lane target \
  --generation consumed-generation-1 --ttl-seconds 1 >/dev/null
set +e
run obligation-interrupt --message msg-consumed-crash --actor worker --lane target \
  --generation consumed-generation-2 --reason premature-consumer-takeover \
  > "$WORK/premature-consumer.log" 2>&1
premature_consumer_rc=$?
set -e
[[ "$premature_consumer_rc" -eq 1 ]] || fail 'live consumer was interrupted by another generation'
sleep 2
run obligation-interrupt --message msg-consumed-crash --actor worker --lane target \
  --generation consumed-generation-2 --reason expired-consumer-takeover >/dev/null
run obligation-recover --message msg-consumed-crash --actor worker --lane target \
  --generation consumed-generation-2 >/dev/null
run obligation-claim --message msg-consumed-crash --actor worker --lane target \
  --generation consumed-generation-2 --claim recovered-consumer-claim \
  --ttl-seconds 120 > "$WORK/consumed-recovered.log"
grep -q 'state=claimed .*generation=consumed-generation-2 .*claim=recovered-consumer-claim' \
  "$WORK/consumed-recovered.log" || fail 'abandoned consumer did not recover into a new claim'

# Matrix D: no evidence path means no completed event.
set +e
run obligation-complete --message msg-exclusive --actor worker --lane target \
  --generation generation-a --claim claim-a --outcome "$WORK/outcome.txt" \
  --evidence "$WORK/does-not-exist" > "$WORK/no-evidence.log" 2>&1
no_evidence_rc=$?
set -e
[[ "$no_evidence_rc" -eq 1 ]] || fail 'missing evidence created a completion'
run obligation-status --message msg-exclusive > "$WORK/not-completed.log"
grep -q 'state=claimed .*unclosed=yes' "$WORK/not-completed.log" || \
  fail 'evidence-less completion changed the obligation state'

# Matrix E: kill every Loom process, then replay the unclosed journals.
"$LOOM" obligation-supervise --interval-seconds 1 --state-dir "$state" \
  > "$WORK/supervisor.log" 2>&1 &
SUPERVISOR_PID=$!
for _ in 1 2 3 4 5; do
  [[ -s "$state/obligation-supervisor.state" ]] && break
  sleep 1
done
[[ -s "$state/obligation-supervisor.state" ]] || fail 'supervisor did not publish replay state'
run obligation-supervisor-status > "$WORK/supervisor-live.log"
grep -q 'state=live ' "$WORK/supervisor-live.log" || fail 'supervisor was not independently live'
kill -9 "$SUPERVISOR_PID"
wait "$SUPERVISOR_PID" >/dev/null 2>&1 || true
SUPERVISOR_PID=''
run obligation-supervisor-status > "$WORK/supervisor-dead.log"
grep -q 'state=stopped ' "$WORK/supervisor-dead.log" || fail 'killed supervisor remained live'

# At this point no Loom process exists. A fresh process must rediscover all
# unfinished objects solely from the retained journals.
run obligation-supervise --once > "$WORK/supervisor-restarted.log"
grep -q 'replayed=3 unclosed=3' "$WORK/supervisor-restarted.log" || \
  fail 'fresh supervisor did not rediscover every unclosed obligation'
run obligation-list --json > "$WORK/replayed-list.json"
grep -q '"count":3,"unclosed":3' "$WORK/replayed-list.json" || \
  fail 'unclosed projection was lost across total process death'

# Matrix F: TUI and GUI read the reducer; neither owns a mutable ledger.
run obligation-tui > "$WORK/tui.log"
grep -q '^LOOM_OBLIGATION_LIST count=3 unclosed=3$' "$WORK/tui.log" || \
  fail 'non-interactive TUI projection diverged from replay'
port=$((20000 + $$ % 20000))
"$LOOM" obligation-serve --state-dir "$state" --bind 127.0.0.1 --port "$port" \
  > "$WORK/gui.log" 2>&1 &
GUI_PID=$!
for _ in 1 2 3 4 5; do
  grep -q LOOM_OBLIGATION_GUI "$WORK/gui.log" && break
  sleep 1
done
grep -q LOOM_OBLIGATION_GUI "$WORK/gui.log" || fail 'obligation GUI did not start'
curl -fsS "http://127.0.0.1:$port/api/obligations" > "$WORK/api.json"
grep -q '"count":3,"unclosed":3' "$WORK/api.json" || \
  fail 'GUI API projection diverged from replay'
kill "$GUI_PID"
wait "$GUI_PID" >/dev/null 2>&1 || true
GUI_PID=''

# Matrix G: corruption is detected before state is projected.
open_message msg-corrupt
corrupt_journal="$(find "$state/loom-obligations" -path '*/journal.tsv' -type f \
  -exec grep -l '6d73672d636f7272757074' {} \; | head -1)"
[[ -n "$corrupt_journal" ]] || fail 'could not locate corruption fixture journal'
printf 'corrupt\n' >> "$corrupt_journal"
set +e
run obligation-verify --message msg-corrupt > "$WORK/corrupt.log" 2>&1
corrupt_rc=$?
set -e
[[ "$corrupt_rc" -eq 1 ]] || fail 'corrupt journal was accepted'
grep -q 'journal record does not have' "$WORK/corrupt.log" || \
  fail 'corrupt journal refused for the wrong reason'

printf 'loom-obligation-crash-matrix: PASS exclusive_winner=1 live_takeover=refused expired_renew=refused expired_takeover=recovered consumed_crash=recovered no_evidence=refused total_process_loss=replayed unclosed=3 corruption=refused tui=replayed gui_api=replayed\n'
