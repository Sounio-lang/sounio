#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-fleet-crash.XXXXXX")"
REPO="$TEST_ROOT/repo"
RUNTIME="$TEST_ROOT/runtime"
STATE="$TEST_ROOT/agentd-state"
DB="$TEST_ROOT/fleet.db"
CONFIG="$TEST_ROOT/fleet.toml"
RECEIVER="$TEST_ROOT/receiver.py"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
SUMMARY="$TEST_ROOT/summary.txt"
EVIDENCE="$TEST_ROOT/evidence.txt"
PRIVATE_KEY="$TEST_ROOT/private.pem"
PUBLIC_KEY="$TEST_ROOT/public.pem"
ANCHORS="$TEST_ROOT/anchors"
CERTIFICATE="$TEST_ROOT/crash-certificate.json"
SLOT='crash-lane'

cleanup() {
  if [[ -x "$RUNTIME/sounio-fleet-agent-runtime" && -d "$REPO" ]]; then
    SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot "$SLOT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-fleet-crash-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

fleetd() {
  SOUNIO_AGENTD_DIR="$STATE" \
  SOUNIO_FLEET_AGENT_COMMAND="$RUNTIME/sounio-fleet-agent-runtime" \
    "$RUNTIME/sounio-fleet-runtime" --db "$DB" "$@"
}

expect_crash() {
  local point="$1" log="$2" rc=0
  shift 2
  set +e
  SOUNIO_FLEET_FAILPOINT="$point" fleetd "$@" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" == 197 ]] || {
    cat "$log" >&2
    fail "$point returned $rc instead of the crash-lab exit 197"
  }
  grep -q "FLEET_FAILPOINT name=$point exit=197" "$log" || \
    fail "$point did not emit its exact crash witness"
}

wait_for_starts() {
  local expected="$1" attempt
  for attempt in $(seq 1 100); do
    if [[ -f "$RECEIVER_LOG" ]] && \
      [[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == "$expected" ]]; then
      return 0
    fi
    sleep 0.1
  done
  fail "receiver did not reach exactly $expected starts"
}

stop_slot() {
  SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
    stop --cwd "$REPO" --slot "$SLOT" >/dev/null
}

new_checkpoint() {
  local output checkpoint_id
  output="$(fleetd checkpoint-create --config "$CONFIG" --slot "$SLOT" \
    --kind cognitive --summary-file "$SUMMARY" --evidence "$EVIDENCE")"
  checkpoint_id="$(sed -n 's/.*checkpoint_id=\([^ ]*\).*/\1/p' <<< "$output")"
  [[ -n "$checkpoint_id" ]] || fail 'checkpoint creation omitted its identity'
  fleetd checkpoint-verify --checkpoint-id "$checkpoint_id" >/dev/null
  printf '%s\n' "$checkpoint_id"
}

mkdir -p "$REPO" "$RUNTIME"
git -C "$REPO" init -q
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" \
  "$RUNTIME/sounio-agentd-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" \
  "$RUNTIME/sounio-fleet-agent-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" \
  "$RUNTIME/sounio-fleet-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_fleet_trace_verify.py" \
  "$RUNTIME/sounio-fleet-trace-verify"

cat > "$RECEIVER" <<'PY'
#!/usr/bin/env python3
import os
import sys

with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(f"START pid={os.getpid()}\n")
    handle.flush()
for _ in sys.stdin:
    pass
PY
chmod +x "$RECEIVER"
printf 'crash laboratory checkpoint\n' > "$SUMMARY"
printf 'crash laboratory evidence\n' > "$EVIDENCE"

cat > "$CONFIG" <<EOF
version = 1

[[lane]]
slot = "$SLOT"
enabled = true
restart = "always"
cwd = "$REPO"
agent = "crash-agent"
lane = "crash-runtime"
session_id = "11111111-2222-4333-8444-555555555555"
identity = "exact"
command = ["$RECEIVER", "$RECEIVER_LOG"]
EOF

fleetd init --config "$CONFIG" >/dev/null

cap1="$TEST_ROOT/start-issued.json"
expect_crash 'start-capability:issued' "$TEST_ROOT/start-cap-issued.log" \
  authorize --config "$CONFIG" --slot "$SLOT" --out "$cap1"
[[ ! -e "$cap1" ]] || fail 'issued-only crash unexpectedly published a secret file'
fleetd authorize --config "$CONFIG" --slot "$SLOT" --out "$cap1" >/dev/null
fleetd reconcile --config "$CONFIG" --apply --capability "$cap1" >/dev/null
wait_for_starts 1
stop_slot

cap2="$TEST_ROOT/start-file.json"
expect_crash 'start-capability:file-written' "$TEST_ROOT/start-cap-file.log" \
  authorize --config "$CONFIG" --slot "$SLOT" --out "$cap2"
[[ -s "$cap2" ]] || fail 'file-written crash lost the atomic capability file'
fleetd reconcile --config "$CONFIG" --apply --capability "$cap2" >/dev/null
wait_for_starts 2
stop_slot

cap3="$TEST_ROOT/start-consumed.json"
fleetd authorize --config "$CONFIG" --slot "$SLOT" --out "$cap3" >/dev/null
expect_crash 'start-action:authority-consumed' "$TEST_ROOT/start-consumed.log" \
  reconcile --config "$CONFIG" --apply --capability "$cap3"
[[ ! -e "$STATE/fleet-slots/$SLOT.json" ]] || \
  fail 'authority-consumed crash launched before the requested boundary'
fleetd reconcile --config "$CONFIG" --apply >/dev/null
wait_for_starts 3
stop_slot

cap4="$TEST_ROOT/start-requested.json"
fleetd authorize --config "$CONFIG" --slot "$SLOT" --out "$cap4" >/dev/null
expect_crash 'start-action:requested' "$TEST_ROOT/start-requested.log" \
  reconcile --config "$CONFIG" --apply --capability "$cap4"
[[ ! -e "$STATE/fleet-slots/$SLOT.json" ]] || \
  fail 'requested-only crash launched a process'
fleetd reconcile --config "$CONFIG" --apply >/dev/null
wait_for_starts 4
stop_slot

cap5="$TEST_ROOT/start-launched.json"
fleetd authorize --config "$CONFIG" --slot "$SLOT" --out "$cap5" >/dev/null
expect_crash 'start-action:launched' "$TEST_ROOT/start-launched.log" \
  reconcile --config "$CONFIG" --apply --capability "$cap5"
wait_for_starts 5
fleetd reconcile --config "$CONFIG" --apply >/dev/null
wait_for_starts 5

fleetd keygen --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" >/dev/null

index=0
for point in \
  'handoff-prepare:prepared' \
  'handoff-prepare:authority-issued' \
  'handoff-prepare:file-written'; do
  index=$((index + 1))
  checkpoint_id="$(new_checkpoint)"
  handoff_cap="$TEST_ROOT/handoff-recover-$index.json"
  expect_crash "$point" "$TEST_ROOT/handoff-prepare-$index.log" \
    handoff-prepare --checkpoint-id "$checkpoint_id" \
    --to-agent reviewer --to-lane crash-review \
    --capability-out "$handoff_cap"
  output="$(fleetd handoff-prepare --checkpoint-id "$checkpoint_id" \
    --to-agent reviewer --to-lane crash-review \
    --capability-out "$handoff_cap")"
  grep -q 'FLEET_HANDOFF_RECOVERED ' <<< "$output" || \
    fail "$point retry did not recover the prepared transaction"
  [[ -s "$handoff_cap" ]] || fail "$point recovery omitted its capability"
done

checkpoint_request="$(new_checkpoint)"
handoff_request_cap="$TEST_ROOT/handoff-request.json"
output="$(fleetd handoff-prepare --checkpoint-id "$checkpoint_request" \
  --to-agent reviewer --to-lane crash-review \
  --capability-out "$handoff_request_cap")"
handoff_request="$(sed -n 's/.*handoff_id=\([^ ]*\).*/\1/p' <<< "$output")"
fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
expect_crash 'handoff-accept:requested' "$TEST_ROOT/handoff-requested.log" \
  handoff-accept --handoff-id "$handoff_request" \
  --agent reviewer --lane crash-review --capability "$handoff_request_cap" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS"
fleetd handoff-accept --handoff-id "$handoff_request" \
  --agent reviewer --lane crash-review --capability "$handoff_request_cap" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" >/dev/null

checkpoint_consumed="$(new_checkpoint)"
handoff_consumed_cap="$TEST_ROOT/handoff-consumed.json"
output="$(fleetd handoff-prepare --checkpoint-id "$checkpoint_consumed" \
  --to-agent reviewer --to-lane crash-review \
  --capability-out "$handoff_consumed_cap")"
handoff_consumed="$(sed -n 's/.*handoff_id=\([^ ]*\).*/\1/p' <<< "$output")"
fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
expect_crash 'handoff-accept:authority-consumed' "$TEST_ROOT/handoff-consumed.log" \
  handoff-accept --handoff-id "$handoff_consumed" \
  --agent reviewer --lane crash-review --capability "$handoff_consumed_cap" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS"
fleetd handoff-accept --handoff-id "$handoff_consumed" \
  --agent reviewer --lane crash-review --capability "$handoff_consumed_cap" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" >/dev/null
fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null

output="$("$RUNTIME/sounio-fleet-trace-verify" --db "$DB" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  --certificate "$CERTIFICATE")"
grep -q 'FLEET_TRACE_CONFORMS .*accepted=2 .*invariants=8 ' <<< "$output" || \
  fail 'crash-recovered Event Log does not refine the abstract fleet machine'

python3 - "$DB" <<'PY'
import json
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as connection:
    rows = connection.execute(
        "SELECT event_type, payload FROM events "
        "WHERE event_type IN ('CAPABILITY_CONSUMED', 'HANDOFF_ACCEPTED')"
    ).fetchall()
consumed = [json.loads(payload)["capability_id"] for kind, payload in rows if kind == "CAPABILITY_CONSUMED"]
accepted = [json.loads(payload)["handoff_id"] for kind, payload in rows if kind == "HANDOFF_ACCEPTED"]
assert len(consumed) == len(set(consumed)), consumed
assert len(accepted) == len(set(accepted)) == 2, accepted
PY

echo 'sounio-fleet-crash-selftest: PASS failpoints=10 starts=5 duplicate_starts=0 duplicate_capability_consumption=0 prepared_recovery=3 accepted_recovery=2 trace_refinement=PASS'
