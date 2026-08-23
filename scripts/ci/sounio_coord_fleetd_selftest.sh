#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-coord-fleetd-selftest.XXXXXX")"
REPO="$TEST_ROOT/repo"
RUNTIME="$TEST_ROOT/runtime"
STATE="$TEST_ROOT/agentd-state"
DB="$TEST_ROOT/fleet.db"
CONFIG="$TEST_ROOT/fleet.toml"
OMIT_CONFIG="$TEST_ROOT/fleet-omitted.toml"
RECEIVER="$TEST_ROOT/receiver.py"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
SLOT='proof-lane'
SESSION_ID='11111111-2222-4333-8444-555555555555'

cleanup() {
  if [[ -x "$RUNTIME/sounio-fleet-agent-runtime" && -d "$REPO" ]]; then
    SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot "$SLOT" >/dev/null 2>&1 || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-coord-fleetd-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

wait_for() {
  local description="$1" command="$2" attempt
  for attempt in $(seq 1 100); do
    if eval "$command"; then return 0; fi
    sleep 0.1
  done
  fail "$description"
}

event_count() {
  python3 - "$DB" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as connection:
    print(connection.execute("SELECT count(*) FROM events").fetchone()[0])
PY
}

mkdir -p "$REPO" "$RUNTIME"
git -C "$REPO" init -q
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_agentd.py" \
  "$RUNTIME/sounio-agentd-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_fleet.py" \
  "$RUNTIME/sounio-fleet-agent-runtime"
install -m 0755 "$ROOT_DIR/scripts/dev/sounio_coord_fleetd.py" \
  "$RUNTIME/sounio-fleet-runtime"

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

cat > "$CONFIG" <<EOF
version = 1

[[lane]]
slot = "$SLOT"
enabled = true
restart = "always"
cwd = "$REPO"
agent = "proof-agent"
lane = "proof-runtime"
session_id = "$SESSION_ID"
identity = "exact"
command = ["$RECEIVER", "$RECEIVER_LOG"]

[[lane]]
slot = "retained-disabled-lane"
enabled = false
restart = "never"
cwd = "$REPO"
agent = "retained-agent"
session_id = "99999999-8888-4777-8666-555555555555"
identity = "exact"
command = ["$RECEIVER", "$TEST_ROOT/disabled.log"]
EOF

cat > "$OMIT_CONFIG" <<EOF
version = 1

[[lane]]
slot = "$SLOT"
enabled = true
restart = "always"
cwd = "$REPO"
agent = "proof-agent"
lane = "proof-runtime"
session_id = "$SESSION_ID"
identity = "exact"
command = ["$RECEIVER", "$RECEIVER_LOG"]
EOF

fleetd() {
  SOUNIO_AGENTD_DIR="$STATE" \
  SOUNIO_FLEET_AGENT_COMMAND="$RUNTIME/sounio-fleet-agent-runtime" \
    "$RUNTIME/sounio-fleet-runtime" --db "$DB" "$@"
}

output="$(fleetd init --config "$CONFIG")"
grep -q 'FLEET_INITIALIZED .*lanes=2' <<< "$output" || \
  fail 'init did not validate and persist desired state'

if fleetd observe --config "$OMIT_CONFIG" >"$TEST_ROOT/omission" 2>&1; then
  fail 'reconciler accepted silent removal of a tracked slot'
fi
grep -q 'retain them with enabled = false: retained-disabled-lane' \
  "$TEST_ROOT/omission" || \
  fail 'tracked-slot omission was not attributed to the retention rule'

output="$(fleetd observe --config "$CONFIG")"
grep -q 'state=absent reason=slot-mapping-absent' <<< "$output" || \
  fail 'observe did not record the missing slot without applying a policy'

output="$(fleetd reconcile --config "$CONFIG")"
grep -q 'observed=absent decision=start' <<< "$output" || \
  fail 'dry reconciliation did not plan the missing desired slot'
[[ ! -e "$STATE/fleet-slots/$SLOT.json" ]] || \
  fail 'dry reconciliation changed the live fleet'
first_count="$(event_count)"
fleetd reconcile --config "$CONFIG" >/dev/null
[[ "$(event_count)" == "$first_count" ]] || \
  fail 'identical reconciliation was not causally deduplicated'

output="$(fleetd reconcile --config "$CONFIG" --apply)"
grep -q 'FLEET_ACTION .*status=committed' <<< "$output" || \
  fail 'authorized reconciliation did not commit the start action'
wait_for 'reconciler did not start the supervised harness' \
  "grep -q '^START pid=' '$RECEIVER_LOG' 2>/dev/null"
output="$(fleetd status --slot "$SLOT")"
grep -q 'observed=active .*decision=noop' <<< "$output" || \
  fail 'materialized view did not converge to active/noop'

fleetd reconcile --config "$CONFIG" --apply >/dev/null
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'reconciliation launched a duplicate harness'
steady_count="$(event_count)"
fleetd watch --config "$CONFIG" --cycles 2 --interval 0.01 >/dev/null
[[ "$(event_count)" == "$steady_count" ]] || \
  fail 'steady-state watch appended duplicate transitions'

output="$(fleetd verify-log)"
grep -q 'FLEET_LOG_VERIFIED events=' <<< "$output" || \
  fail 'hash chain did not verify'
output="$(fleetd explain --slot "$SLOT")"
grep -q '^decision=noop$' <<< "$output" || fail 'explain omitted the decision'
grep -q '^reason=desired-state-satisfied$' <<< "$output" || \
  fail 'explain omitted the causal reason'

mapping="$STATE/fleet-slots/$SLOT.json"
cp "$mapping" "$mapping.good"
python3 - "$mapping" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["instance_id"] = "sabotaged-generation"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True)
    handle.write("\n")
PY
if fleetd reconcile --config "$CONFIG" --apply >"$TEST_ROOT/generation-drift" 2>&1; then
  fail 'reconciler accepted a sabotaged supervisor generation'
fi
grep -q 'decision=blocked reason=supervisor-generation-drift' \
  "$TEST_ROOT/generation-drift" || \
  fail 'generation sabotage was not attributed to the generation rule'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'generation sabotage caused a replacement harness'
mv "$mapping.good" "$mapping"

cp "$CONFIG" "$CONFIG.good"
sed -i "s/$SESSION_ID/aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee/" "$CONFIG"
if fleetd reconcile --config "$CONFIG" --apply >"$TEST_ROOT/identity-drift" 2>&1; then
  fail 'reconciler accepted desired identity drift'
fi
grep -q 'decision=blocked reason=desired-session_id-mismatch' \
  "$TEST_ROOT/identity-drift" || \
  fail 'desired identity sabotage was not attributed to the identity rule'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'identity sabotage caused a replacement harness'
mv "$CONFIG.good" "$CONFIG"
fleetd reconcile --config "$CONFIG" >/dev/null

python3 - "$DB" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as connection:
    connection.execute("DELETE FROM slot_view")
    connection.commit()
PY
if fleetd status >"$TEST_ROOT/view-sabotage" 2>&1; then
  fail 'status trusted a sabotaged materialized view'
fi
grep -q 'materialized fleet view mismatch' "$TEST_ROOT/view-sabotage" || \
  fail 'view sabotage was not attributed to replay inconsistency'
fleetd rebuild-views >/dev/null
grep -q 'observed=active' <(fleetd status --slot "$SLOT") || \
  fail 'replay did not reconstruct the materialized view'

python3 - "$DB" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as connection:
    connection.execute(
        "UPDATE events SET payload = ? WHERE seq = 1",
        ('{"sabotaged":true}',),
    )
    connection.commit()
PY
if fleetd verify-log >"$TEST_ROOT/log-sabotage" 2>&1; then
  fail 'hash-chain verifier accepted a sabotaged event payload'
fi
grep -q 'event 1 hash mismatch' "$TEST_ROOT/log-sabotage" || \
  fail 'log sabotage was not attributed to the hash-chain rule'

echo 'sounio-coord-fleetd-selftest: PASS dry_run=no-mutation duplicate_start=refused omission=blocked generation_sabotage=blocked identity_sabotage=blocked replay=reconstructed hash_sabotage=refused'
