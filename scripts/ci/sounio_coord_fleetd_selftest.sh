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
DISABLED_CONFIG="$TEST_ROOT/fleet-disabled.toml"
RETAINED_ENABLED_CONFIG="$TEST_ROOT/fleet-retained-enabled.toml"
RECEIVER="$TEST_ROOT/receiver.py"
RECEIVER_LOG="$TEST_ROOT/receiver.log"
LANE_HOME="$TEST_ROOT/lane-home"
SLOT='proof-lane'
SESSION_ID='11111111-2222-4333-8444-555555555555'
CAPABILITY="$TEST_ROOT/start.capability.json"
BAD_CAPABILITY="$TEST_ROOT/start.capability.bad.json"
SECOND_CAPABILITY="$TEST_ROOT/start-2.capability.json"
THIRD_CAPABILITY="$TEST_ROOT/start-3.capability.json"
STOP_CAPABILITY="$TEST_ROOT/stop.capability.json"
BAD_STOP_CAPABILITY="$TEST_ROOT/stop.capability.bad.json"
RETAINED_START_CAPABILITY="$TEST_ROOT/retained-start.capability.json"
RETAINED_STOP_CAPABILITY="$TEST_ROOT/retained-stop.capability.json"
RECOVERY_BUDGET="$TEST_ROOT/recovery-budget.json"
BAD_RECOVERY_BUDGET="$TEST_ROOT/recovery-budget.bad.json"
RENEWED_RECOVERY_BUDGET="$TEST_ROOT/recovery-budget-renewed.json"
PRIVATE_KEY="$TEST_ROOT/anchor-private.pem"
PUBLIC_KEY="$TEST_ROOT/anchor-public.pem"
WRONG_PRIVATE_KEY="$TEST_ROOT/wrong-private.pem"
WRONG_PUBLIC_KEY="$TEST_ROOT/wrong-public.pem"
ANCHORS="$TEST_ROOT/anchors"
CHECKPOINT_SUMMARY="$TEST_ROOT/checkpoint-summary.txt"
CHECKPOINT_EVIDENCE="$TEST_ROOT/checkpoint-evidence.txt"
HANDOFF_CAPABILITY="$TEST_ROOT/handoff.capability.json"
TRACE_CERTIFICATE="$TEST_ROOT/trace-certificate.json"
TRACE_SABOTAGE_DB="$TEST_ROOT/trace-sabotage.db"
STOP_TRACE_SABOTAGE_DB="$TEST_ROOT/stop-trace-sabotage.db"
BUDGET_TRACE_SABOTAGE_DB="$TEST_ROOT/budget-trace-sabotage.db"
BACKOFF_TRACE_SABOTAGE_DB="$TEST_ROOT/backoff-trace-sabotage.db"
ISOLATION_DB="$TEST_ROOT/isolation.db"
ISOLATION_STATE="$TEST_ROOT/isolation-agentd-state"
ISOLATION_CONFIG="$TEST_ROOT/isolation-fleet.toml"
ISOLATION_BUDGETS="$TEST_ROOT/isolation-budgets"
ISOLATION_LATCHES="$TEST_ROOT/isolation-latches"
ISOLATION_WRAPPER="$TEST_ROOT/isolation-wrapper.sh"
CURSOR_FAIL_MARKER="$TEST_ROOT/cursor.fail"
ISOLATION_TRACE_SABOTAGE_DB="$TEST_ROOT/isolation-trace-sabotage.db"

cleanup() {
  if [[ -x "$RUNTIME/sounio-fleet-agent-runtime" && -d "$REPO" ]]; then
    SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot "$SLOT" >/dev/null 2>&1 || true
    SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot retained-disabled-lane >/dev/null 2>&1 || true
    SOUNIO_AGENTD_DIR="$ISOLATION_STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot cursor-hard >/dev/null 2>&1 || true
    SOUNIO_AGENTD_DIR="$ISOLATION_STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
      stop --cwd "$REPO" --slot grok-hard >/dev/null 2>&1 || true
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

mkdir -p "$REPO" "$RUNTIME" "$LANE_HOME"
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
printf 'bounded cognitive state transition\n' > "$CHECKPOINT_SUMMARY"
printf 'evidence receipt v1\n' > "$CHECKPOINT_EVIDENCE"

cat > "$CONFIG" <<EOF
version = 1

[[lane]]
slot = "$SLOT"
enabled = true
restart = "always"
cwd = "$REPO"
home = "$LANE_HOME"
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
home = "$LANE_HOME"
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
home = "$LANE_HOME"
agent = "proof-agent"
lane = "proof-runtime"
session_id = "$SESSION_ID"
identity = "exact"
command = ["$RECEIVER", "$RECEIVER_LOG"]
EOF

sed '0,/enabled = true/s//enabled = false/' "$CONFIG" > "$DISABLED_CONFIG"
sed -e '0,/enabled = false/s//enabled = true/' \
  -e '0,/restart = "never"/s//restart = "always"/' \
  "$CONFIG" > "$RETAINED_ENABLED_CONFIG"

fleetd() {
  SOUNIO_AGENTD_DIR="$STATE" \
  SOUNIO_FLEET_AGENT_COMMAND="$RUNTIME/sounio-fleet-agent-runtime" \
    "$RUNTIME/sounio-fleet-runtime" --db "$DB" "$@"
}

fleetd_isolation() {
  SOUNIO_AGENTD_DIR="$ISOLATION_STATE" \
  SOUNIO_FLEET_AGENT_COMMAND="$RUNTIME/sounio-fleet-agent-runtime" \
    "$RUNTIME/sounio-fleet-runtime" --db "$ISOLATION_DB" "$@"
}

python3 - "$RUNTIME/sounio-fleet-runtime" "$TEST_ROOT/policy.db" <<'PY'
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

loader = SourceFileLoader("fleetd_policy", sys.argv[1])
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
loader.exec_module(module)
lane = module.LaneSpec(
    slot="policy-test",
    enabled=True,
    restart="on-failure",
    cwd=Path.cwd(),
    kind=None,
    home=None,
    agent=None,
    lane=None,
    session_id=None,
    identity=None,
    command=("true",),
)
with module.connect_db(Path(sys.argv[2])) as connection:
    initial = module.decide(
        connection, lane, {"state": "absent", "reason": "test"}, 1
    )
    unreachable = module.decide(
        connection, lane, {"state": "unreachable", "reason": "test"}, 1
    )
assert initial[0] == "start", initial
assert unreachable == ("blocked", "probe-unreachable-start-not-authorized"), unreachable
PY

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

# MODEL_CONTROL:capability_required
if fleetd reconcile --config "$CONFIG" --apply >"$TEST_ROOT/no-capability" 2>&1; then
  fail 'reconcile --apply mutated state without a linear capability'
fi
grep -q 'status=refused reason=linear-capability-required' \
  "$TEST_ROOT/no-capability" || \
  fail 'missing mutation authority was not attributed to the capability rule'
[[ ! -e "$STATE/fleet-slots/$SLOT.json" ]] || \
  fail 'capability refusal changed the live fleet'

output="$(fleetd authorize --config "$CONFIG" --slot "$SLOT" --out "$CAPABILITY")"
grep -q 'FLEET_CAPABILITY_ISSUED .*action=start' <<< "$output" || \
  fail 'authorize did not issue a start capability'
[[ "$(stat -c %a "$CAPABILITY")" == 600 ]] || \
  fail 'start capability file mode is not 600'
cp "$CAPABILITY" "$BAD_CAPABILITY"
python3 - "$BAD_CAPABILITY" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["token"] = "sabotaged-secret"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
chmod 600 "$BAD_CAPABILITY"
if fleetd reconcile --config "$CONFIG" --apply --capability "$BAD_CAPABILITY" \
  >"$TEST_ROOT/bad-capability" 2>&1; then
  fail 'reconciler accepted a capability with a sabotaged secret'
fi
grep -q 'status=refused reason=.*secret-does-not-match' \
  "$TEST_ROOT/bad-capability" || \
  fail 'capability secret sabotage was not attributed to token verification'

output="$(fleetd reconcile --config "$CONFIG" --apply --capability "$CAPABILITY")"
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
output="$(fleetd keygen --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY")"
grep -q 'FLEET_ANCHOR_KEY_GENERATED ' <<< "$output" || \
  fail 'Ed25519 anchor key generation did not return a receipt'
[[ "$(stat -c %a "$PRIVATE_KEY")" == 600 ]] || \
  fail 'Ed25519 private key mode is not 600'
output="$(fleetd anchor-log --private-key "$PRIVATE_KEY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS")"
grep -q 'FLEET_LOG_ANCHORED events=' <<< "$output" || \
  fail 'event log was not signed'
fleetd verify-anchors --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
[[ "$(find "$ANCHORS" -type f -name 'anchor-*.json' | wc -l)" == 1 ]] || \
  fail 'anchoring an unchanged prefix created a duplicate anchor'
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

cp "$mapping" "$mapping.good"
python3 - "$mapping" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["start_capability_id"] = "cap-sabotaged-generation-authority"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True)
    handle.write("\n")
PY
if fleetd reconcile --config "$CONFIG" --apply \
  >"$TEST_ROOT/generation-authority-drift" 2>&1; then
  fail 'reconciler accepted a generation relabeled with another capability'
fi
grep -q 'decision=blocked reason=desired-argv_digest-mismatch' \
  "$TEST_ROOT/generation-authority-drift" || \
  fail 'generation authority sabotage was not tied back to supervised argv'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'generation authority sabotage caused a replacement harness'
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

# MODEL_CONTROL:stop_capability_required
if fleetd reconcile --config "$DISABLED_CONFIG" --apply \
  >"$TEST_ROOT/no-stop-capability" 2>&1; then
  fail 'reconcile --apply stopped an active slot without linear stop authority'
fi
grep -q 'action=stop status=refused reason=linear-stop-capability-required' \
  "$TEST_ROOT/no-stop-capability" || \
  fail 'missing stop authority was not attributed to the stop-capability rule'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'missing stop authority changed the active generation'

output="$(fleetd authorize --config "$DISABLED_CONFIG" --slot "$SLOT" \
  --action stop --out "$STOP_CAPABILITY")"
grep -q 'FLEET_CAPABILITY_ISSUED .*action=stop .*generation=' <<< "$output" || \
  fail 'authorize did not bind stop authority to the active generation'
cp "$STOP_CAPABILITY" "$BAD_STOP_CAPABILITY"
python3 - "$BAD_STOP_CAPABILITY" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["generation"] = "sabotaged-stop-generation"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
chmod 600 "$BAD_STOP_CAPABILITY"
if fleetd reconcile --config "$DISABLED_CONFIG" --apply \
  --capability "$BAD_STOP_CAPABILITY" >"$TEST_ROOT/bad-stop-generation" 2>&1; then
  fail 'stop accepted authority for a different generation'
fi
grep -q 'action=stop status=refused reason=.*binding-was-altered' \
  "$TEST_ROOT/bad-stop-generation" || \
  fail 'stop generation sabotage was not attributed to capability binding'
output="$(fleetd reconcile --config "$DISABLED_CONFIG" --apply \
  --capability "$STOP_CAPABILITY")"
grep -q 'FLEET_ACTION .*action=stop status=committed' <<< "$output" || \
  fail 'authorized stop did not commit'
[[ ! -e "$STATE/fleet-slots/$SLOT.json" ]] || \
  fail 'authorized stop retained the fleet slot mapping'

# MODEL_CONTROL:capability_reuse
if fleetd reconcile --config "$CONFIG" --apply --capability "$CAPABILITY" \
  >"$TEST_ROOT/reused-capability" 2>&1; then
  fail 'reconciler accepted a consumed capability after the slot stopped'
fi
grep -q 'status=refused reason=.*already-consumed' "$TEST_ROOT/reused-capability" || \
  fail 'capability reuse was not attributed to linear consumption'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 1 ]] || \
  fail 'consumed capability restarted the harness'
fleetd authorize --config "$CONFIG" --slot "$SLOT" \
  --out "$SECOND_CAPABILITY" >/dev/null
fleetd reconcile --config "$CONFIG" --apply \
  --capability "$SECOND_CAPABILITY" >/dev/null
wait_for 'second linear capability did not restore the stopped slot' \
  "test \"\$(grep -c '^START pid=' '$RECEIVER_LOG')\" = 2"

# Keep a legacy lane active while the primary slot exercises bounded recovery.
# Recovery mode must observe its stop decision without acquiring destructive
# authority or degrading the recovery loop.
fleetd authorize --config "$RETAINED_ENABLED_CONFIG" \
  --slot retained-disabled-lane --out "$RETAINED_START_CAPABILITY" >/dev/null
fleetd reconcile --config "$RETAINED_ENABLED_CONFIG" --apply \
  --capability "$RETAINED_START_CAPABILITY" >/dev/null
wait_for 'retained legacy lane did not start for the mixed-catalog control' \
  "test \"\$(grep -c '^START pid=' '$TEST_ROOT/disabled.log')\" = 1"

# MODEL_CONTROL:stop_capability_reuse
if fleetd reconcile --config "$DISABLED_CONFIG" --apply \
  --capability "$STOP_CAPABILITY" >"$TEST_ROOT/reused-stop-capability" 2>&1; then
  fail 'reconciler accepted a consumed stop capability for a later generation'
fi
grep -q 'action=stop status=refused reason=.*already-consumed' \
  "$TEST_ROOT/reused-stop-capability" || \
  fail 'stop capability reuse was not attributed to linear consumption'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 2 ]] || \
  fail 'reused stop capability changed the later generation'

output="$(fleetd authorize-recovery --config "$CONFIG" --slot "$SLOT" \
  --out "$RECOVERY_BUDGET" --max-starts 2 --backoff-seconds 5 --ttl 600)"
grep -q 'FLEET_RECOVERY_BUDGET_ISSUED .*max_starts=2 .*backoff_seconds=5' \
  <<< "$output" || fail 'bounded recovery authorization omitted its budget'
cp "$RECOVERY_BUDGET" "$BAD_RECOVERY_BUDGET"
python3 - "$BAD_RECOVERY_BUDGET" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["token"] = "sabotaged-recovery-secret"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
chmod 600 "$BAD_RECOVERY_BUDGET"

SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
  stop --cwd "$REPO" --slot "$SLOT" >/dev/null
if fleetd watch --config "$CONFIG" --cycles 1 --interval 0.01 \
  --apply-recovery --recovery-budget "$BAD_RECOVERY_BUDGET" \
  >"$TEST_ROOT/bad-recovery-budget" 2>&1; then
  fail 'watch accepted a recovery budget with a sabotaged secret'
fi
grep -q 'action=start status=refused reason=.*secret-does-not-match' \
  "$TEST_ROOT/bad-recovery-budget" || \
  fail 'recovery secret sabotage was not attributed to budget verification'
grep -q 'slot=retained-disabled-lane action=stop status=held reason=recovery-mode-start-only' \
  "$TEST_ROOT/bad-recovery-budget" || \
  fail 'recovery mode did not hold a destructive legacy-lane stop decision'

output="$(fleetd watch --config "$CONFIG" --cycles 1 --interval 0.01 \
  --apply-recovery --recovery-budget "$RECOVERY_BUDGET")"
grep -q 'slot=retained-disabled-lane action=stop status=held reason=recovery-mode-start-only' \
  <<< "$output" || fail 'bounded recovery attempted to stop the retained legacy lane'
wait_for 'first recovery unit did not restart the stopped slot' \
  "test \"\$(grep -c '^START pid=' '$RECEIVER_LOG')\" = 3"
[[ "$(grep -c '^START pid=' "$TEST_ROOT/disabled.log")" == 1 ]] || \
  fail 'bounded recovery changed the retained legacy lane generation'
[[ -e "$STATE/fleet-slots/retained-disabled-lane.json" ]] || \
  fail 'bounded recovery removed the retained legacy lane mapping'
fleetd authorize --config "$CONFIG" --slot retained-disabled-lane \
  --action stop --out "$RETAINED_STOP_CAPABILITY" >/dev/null
fleetd reconcile --config "$CONFIG" --apply \
  --capability "$RETAINED_STOP_CAPABILITY" >/dev/null
[[ ! -e "$STATE/fleet-slots/retained-disabled-lane.json" ]] || \
  fail 'manual stop authority did not clean up the retained legacy control'
SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
  stop --cwd "$REPO" --slot "$SLOT" >/dev/null
output="$(fleetd watch --config "$CONFIG" --cycles 1 --interval 0.01 \
  --apply-recovery --recovery-budget "$RECOVERY_BUDGET")"
grep -q 'status=deferred reason=recovery-backoff-active' <<< "$output" || \
  fail 'recovery budget ignored its temporal backoff'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 3 ]] || \
  fail 'backoff created a premature restart'
sleep 5.1
fleetd watch --config "$CONFIG" --cycles 1 --interval 0.01 \
  --apply-recovery --recovery-budget "$RECOVERY_BUDGET" >/dev/null
wait_for 'second recovery unit did not restart the stopped slot' \
  "test \"\$(grep -c '^START pid=' '$RECEIVER_LOG')\" = 4"
SOUNIO_AGENTD_DIR="$STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
  stop --cwd "$REPO" --slot "$SLOT" >/dev/null
sleep 5.1
if fleetd watch --config "$CONFIG" --cycles 1 --interval 0.01 \
  --apply-recovery --recovery-budget "$RECOVERY_BUDGET" \
  >"$TEST_ROOT/exhausted-recovery-budget" 2>&1; then
  fail 'watch exceeded the bounded restart budget'
fi
grep -q 'status=refused reason=recovery-budget-exhausted' \
  "$TEST_ROOT/exhausted-recovery-budget" || \
  fail 'budget exhaustion was not attributed to the temporal authority'
[[ "$(grep -c '^START pid=' "$RECEIVER_LOG")" == 4 ]] || \
  fail 'exhausted recovery budget launched an extra generation'
fleetd authorize-recovery --config "$CONFIG" --slot "$SLOT" \
  --out "$RENEWED_RECOVERY_BUDGET" --max-starts 1 --backoff-seconds 0 \
  --ttl 600 >/dev/null || fail 'exhausted recovery budget could not be renewed'

fleetd authorize --config "$CONFIG" --slot "$SLOT" \
  --out "$THIRD_CAPABILITY" >/dev/null
fleetd reconcile --config "$CONFIG" --apply \
  --capability "$THIRD_CAPABILITY" >/dev/null
wait_for 'manual authority did not restore the slot after budget exhaustion' \
  "test \"\$(grep -c '^START pid=' '$RECEIVER_LOG')\" = 5"

output="$(fleetd checkpoint-create --config "$CONFIG" --slot "$SLOT" \
  --kind cognitive --summary-file "$CHECKPOINT_SUMMARY" \
  --evidence "$CHECKPOINT_EVIDENCE")"
checkpoint_id="$(sed -n 's/.*checkpoint_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$checkpoint_id" ]] || fail 'checkpoint draft omitted its typed identity'

# MODEL_CONTROL:wrong_checkpoint_state
if fleetd handoff-prepare --checkpoint-id "$checkpoint_id" \
  --to-agent reviewer --to-lane review-lane \
  --capability-out "$HANDOFF_CAPABILITY" >"$TEST_ROOT/draft-handoff" 2>&1; then
  fail 'handoff preparation accepted a draft checkpoint'
fi
grep -q 'requires a uniquely verified checkpoint' "$TEST_ROOT/draft-handoff" || \
  fail 'draft handoff refusal omitted the checkpoint typestate rule'

cp "$CHECKPOINT_EVIDENCE" "$CHECKPOINT_EVIDENCE.good"
printf 'sabotaged after draft\n' >> "$CHECKPOINT_EVIDENCE"
if fleetd checkpoint-verify --checkpoint-id "$checkpoint_id" \
  >"$TEST_ROOT/evidence-drift" 2>&1; then
  fail 'checkpoint verification accepted drifted evidence'
fi
grep -q 'checkpoint evidence drifted' "$TEST_ROOT/evidence-drift" || \
  fail 'checkpoint evidence sabotage omitted its refusal reason'
mv "$CHECKPOINT_EVIDENCE.good" "$CHECKPOINT_EVIDENCE"
fleetd checkpoint-verify --checkpoint-id "$checkpoint_id" >/dev/null

output="$(fleetd handoff-prepare --checkpoint-id "$checkpoint_id" \
  --to-agent reviewer --to-lane review-lane \
  --capability-out "$HANDOFF_CAPABILITY")"
handoff_id="$(sed -n 's/.*handoff_id=\([^ ]*\).*/\1/p' <<< "$output")"
[[ -n "$handoff_id" ]] || fail 'prepared handoff omitted its typed identity'
[[ "$(stat -c %a "$HANDOFF_CAPABILITY")" == 600 ]] || \
  fail 'handoff capability file mode is not 600'

# MODEL_CONTROL:unanchored_handoff
if fleetd handoff-accept --handoff-id "$handoff_id" \
  --agent reviewer --lane review-lane --capability "$HANDOFF_CAPABILITY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/unanchored-handoff" 2>&1; then
  fail 'handoff acceptance used a log anchor older than its prepared state'
fi
grep -q 'not covered by a signed anchor' "$TEST_ROOT/unanchored-handoff" || \
  fail 'unanchored handoff refusal omitted the signed-prefix rule'

fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
if fleetd handoff-accept --handoff-id "$handoff_id" \
  --agent intruder --lane review-lane --capability "$HANDOFF_CAPABILITY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/wrong-recipient" 2>&1; then
  fail 'handoff acceptance allowed a different recipient'
fi
grep -q 'recipient does not match' "$TEST_ROOT/wrong-recipient" || \
  fail 'wrong recipient refusal omitted the prepared-recipient rule'
cp "$CHECKPOINT_EVIDENCE" "$CHECKPOINT_EVIDENCE.prepared-good"
printf 'sabotaged after handoff preparation\n' >> "$CHECKPOINT_EVIDENCE"
if fleetd handoff-accept --handoff-id "$handoff_id" \
  --agent reviewer --lane review-lane --capability "$HANDOFF_CAPABILITY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/prepared-evidence-drift" 2>&1; then
  fail 'handoff acceptance used evidence changed after preparation'
fi
grep -q 'evidence drifted before handoff acceptance' \
  "$TEST_ROOT/prepared-evidence-drift" || \
  fail 'prepared evidence drift omitted its acceptance refusal reason'
mv "$CHECKPOINT_EVIDENCE.prepared-good" "$CHECKPOINT_EVIDENCE"
output="$(fleetd handoff-accept --handoff-id "$handoff_id" \
  --agent reviewer --lane review-lane --capability "$HANDOFF_CAPABILITY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS")"
grep -q 'FLEET_HANDOFF_ACCEPTED ' <<< "$output" || \
  fail 'anchored handoff did not reach Accepted typestate'
if fleetd handoff-accept --handoff-id "$handoff_id" \
  --agent reviewer --lane review-lane --capability "$HANDOFF_CAPABILITY" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/reused-handoff" 2>&1; then
  fail 'accepted handoff capability was reusable'
fi
grep -q 'handoff was already accepted' "$TEST_ROOT/reused-handoff" || \
  fail 'handoff reuse refusal omitted its terminal typestate'
fleetd anchor-log --private-key "$PRIVATE_KEY" --public-key "$PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >/dev/null
output="$(fleetd verify-anchors --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS")"
grep -q 'FLEET_ANCHORS_VERIFIED anchors=3 ' <<< "$output" || \
  fail 'signed anchor chain did not retain all three log prefixes'
# MODEL_CONTROL:anchor_removal
mapfile -t anchor_files < <(find "$ANCHORS" -type f -name 'anchor-*.json' | sort)
[[ "${#anchor_files[@]}" == 3 ]] || fail 'anchor chain file count is wrong'
mv "${anchor_files[0]}" "$TEST_ROOT/removed-anchor.json"
if fleetd verify-anchors --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/anchor-removal" 2>&1; then
  fail 'anchor verifier accepted an omitted predecessor'
fi
grep -q 'anchor chain predecessor mismatch' "$TEST_ROOT/anchor-removal" || \
  fail 'anchor removal was not attributed to predecessor continuity'
mv "$TEST_ROOT/removed-anchor.json" "${anchor_files[0]}"

cp "${anchor_files[1]}" "$TEST_ROOT/anchor.good"
# MODEL_CONTROL:signature_sabotage
python3 - "${anchor_files[1]}" <<'PY'
import json
import sys
path = sys.argv[1]
value = json.load(open(path, encoding="utf-8"))
value["signature_base64"] = "AAAA"
with open(path, "w", encoding="utf-8") as handle:
    json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\n")
PY
if fleetd verify-anchors --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  >"$TEST_ROOT/signature-sabotage" 2>&1; then
  fail 'anchor verifier accepted a sabotaged Ed25519 signature'
fi
grep -q 'OpenSSL refused Ed25519 operation' "$TEST_ROOT/signature-sabotage" || \
  fail 'signature sabotage was not attributed to Ed25519 verification'
mv "$TEST_ROOT/anchor.good" "${anchor_files[1]}"

fleetd keygen --private-key "$WRONG_PRIVATE_KEY" \
  --public-key "$WRONG_PUBLIC_KEY" >/dev/null
if fleetd verify-anchors --public-key "$WRONG_PUBLIC_KEY" \
  --anchor-dir "$ANCHORS" >"$TEST_ROOT/key-substitution" 2>&1; then
  fail 'anchor verifier accepted a substituted public key'
fi
grep -q 'anchor public-key identity mismatch' "$TEST_ROOT/key-substitution" || \
  fail 'key substitution was not attributed to key identity'

output="$("$RUNTIME/sounio-fleet-trace-verify" --db "$DB" \
  --public-key "$PUBLIC_KEY" --anchor-dir "$ANCHORS" \
  --certificate "$TRACE_CERTIFICATE")"
grep -q 'FLEET_TRACE_CONFORMS .*accepted=1 .*invariants=12 ' <<< "$output" || \
  fail 'independent trace verifier did not certify the accepted handoff'
[[ -s "$TRACE_CERTIFICATE" ]] || \
  fail 'independent trace verifier omitted its refinement certificate'

python3 - "$DB" "$TRACE_SABOTAGE_DB" <<'PY'
import hashlib
import json
import sqlite3
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

source = sqlite3.connect(sys.argv[1])
target = sqlite3.connect(sys.argv[2])
source.backup(target)
source.close()
target.row_factory = sqlite3.Row
accepted = target.execute(
    "SELECT * FROM events WHERE event_type = 'HANDOFF_ACCEPTED'"
).fetchone()
start_cap = target.execute(
    "SELECT payload FROM events WHERE event_type = 'CAPABILITY_ISSUED' ORDER BY seq"
).fetchone()
payload = json.loads(accepted["payload"])
payload["capability_id"] = json.loads(start_cap["payload"])["capability_id"]
target.execute(
    "UPDATE events SET payload = ? WHERE seq = ?",
    (canonical(payload), accepted["seq"]),
)
previous = target.execute(
    "SELECT event_hash FROM events WHERE seq = ?", (accepted["seq"] - 1,)
).fetchone()[0]
for row in target.execute(
    "SELECT * FROM events WHERE seq >= ? ORDER BY seq", (accepted["seq"],)
).fetchall():
    material = {
        "causal_key": row["causal_key"],
        "event_type": row["event_type"],
        "occurred_utc": row["occurred_utc"],
        "payload": row["payload"],
        "prev_hash": previous,
        "seq": row["seq"],
        "slot": row["slot"],
    }
    event_hash = digest(material)
    target.execute(
        "UPDATE events SET prev_hash = ?, event_hash = ?, event_id = ? WHERE seq = ?",
        (previous, event_hash, f"evt-{event_hash[:24]}", row["seq"]),
    )
    previous = event_hash
target.commit()
target.close()
PY
if "$RUNTIME/sounio-fleet-trace-verify" --db "$TRACE_SABOTAGE_DB" \
  >"$TEST_ROOT/trace-semantic-sabotage" 2>&1; then
  fail 'independent trace verifier accepted a rehashed capability substitution'
fi
grep -q 'accepted handoff capability mismatch' \
  "$TEST_ROOT/trace-semantic-sabotage" || \
  fail 'semantic sabotage was not attributed to the independent handoff rule'

python3 - "$DB" "$STOP_TRACE_SABOTAGE_DB" "$BUDGET_TRACE_SABOTAGE_DB" \
  "$BACKOFF_TRACE_SABOTAGE_DB" <<'PY'
import hashlib
import json
import sqlite3
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def rewrite(source_path, target_path, event_type, predicate, mutate):
    source = sqlite3.connect(source_path)
    target = sqlite3.connect(target_path)
    source.backup(target)
    source.close()
    target.row_factory = sqlite3.Row
    selected = None
    for row in target.execute(
        "SELECT * FROM events WHERE event_type = ? ORDER BY seq", (event_type,)
    ).fetchall():
        payload = json.loads(row["payload"])
        if predicate(payload):
            selected = (row, payload)
            break
    assert selected is not None, event_type
    row, payload = selected
    mutate(payload, target)
    target.execute(
        "UPDATE events SET payload = ? WHERE seq = ?", (canonical(payload), row["seq"])
    )
    previous = target.execute(
        "SELECT event_hash FROM events WHERE seq = ?", (row["seq"] - 1,)
    ).fetchone()[0]
    for current in target.execute(
        "SELECT * FROM events WHERE seq >= ? ORDER BY seq", (row["seq"],)
    ).fetchall():
        material = {
            "causal_key": current["causal_key"],
            "event_type": current["event_type"],
            "occurred_utc": current["occurred_utc"],
            "payload": current["payload"],
            "prev_hash": previous,
            "seq": current["seq"],
            "slot": current["slot"],
        }
        event_hash = digest(material)
        target.execute(
            "UPDATE events SET prev_hash = ?, event_hash = ?, event_id = ? WHERE seq = ?",
            (previous, event_hash, f"evt-{event_hash[:24]}", current["seq"]),
        )
        previous = event_hash
    target.commit()
    target.close()

rewrite(
    sys.argv[1],
    sys.argv[2],
    "ACTION_COMMITTED",
    lambda payload: payload.get("action") == "stop",
    lambda payload, _target: payload.__setitem__(
        "generation", "rehashed-wrong-stop-generation"
    ),
)
rewrite(
    sys.argv[1],
    sys.argv[3],
    "RECOVERY_BUDGET_SPENT",
    lambda payload: payload.get("ordinal") == 2,
    lambda payload, _target: payload.__setitem__("ordinal", 3),
)

def reuse_first_spend_time(payload, target):
    first = target.execute(
        "SELECT payload FROM events WHERE event_type = 'RECOVERY_BUDGET_SPENT' "
        "ORDER BY seq LIMIT 1"
    ).fetchone()
    assert first is not None
    payload["spent_unix"] = json.loads(first[0])["spent_unix"]

rewrite(
    sys.argv[1],
    sys.argv[4],
    "RECOVERY_BUDGET_SPENT",
    lambda payload: payload.get("ordinal") == 2,
    reuse_first_spend_time,
)
PY
if "$RUNTIME/sounio-fleet-trace-verify" --db "$STOP_TRACE_SABOTAGE_DB" \
  >"$TEST_ROOT/stop-trace-sabotage" 2>&1; then
  fail 'independent trace verifier accepted a rehashed stop-generation substitution'
fi
grep -q 'committed stop generation mismatch' "$TEST_ROOT/stop-trace-sabotage" || \
  fail 'stop trace sabotage was not attributed to exact-generation binding'
if "$RUNTIME/sounio-fleet-trace-verify" --db "$BUDGET_TRACE_SABOTAGE_DB" \
  >"$TEST_ROOT/budget-trace-sabotage" 2>&1; then
  fail 'independent trace verifier accepted a rehashed recovery-budget overflow'
fi
grep -q 'recovery budget ordinal is not contiguous' \
  "$TEST_ROOT/budget-trace-sabotage" || \
  fail 'budget trace sabotage was not attributed to bounded temporal authority'
if "$RUNTIME/sounio-fleet-trace-verify" --db "$BACKOFF_TRACE_SABOTAGE_DB" \
  >"$TEST_ROOT/backoff-trace-sabotage" 2>&1; then
  fail 'independent trace verifier accepted a rehashed recovery-budget backoff bypass'
fi
grep -q 'recovery budget backoff was violated' \
  "$TEST_ROOT/backoff-trace-sabotage" || \
  fail 'backoff sabotage was not attributed to temporal budget authority'

if fleetd watch --config "$CONFIG" --cycles 1 --apply \
  >"$TEST_ROOT/watch-apply" 2>&1; then
  fail 'watch accepted reusable mutation authority'
fi
grep -q 'watch cannot hold reusable mutation authority' "$TEST_ROOT/watch-apply" || \
  fail 'watch mutation refusal omitted the linear-authority reason'

# Two hostile harness shapes share one recovery cycle. A deterministic Cursor
# failure must spend and latch only Cursor while Grok still converges.
cat > "$ISOLATION_WRAPPER" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
marker="$1"
log="$2"
receiver="$3"
if [[ -e "$marker" ]]; then
  exit 73
fi
exec "$receiver" "$log"
SH
chmod +x "$ISOLATION_WRAPPER"
cat > "$ISOLATION_CONFIG" <<EOF
version = 1

[[lane]]
slot = "cursor-hard"
enabled = true
restart = "always"
cwd = "$REPO"
agent = "cursor"
lane = "fleet-cursor-hard"
session_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
identity = "standalone"
command = ["$ISOLATION_WRAPPER", "$CURSOR_FAIL_MARKER", "$TEST_ROOT/cursor.log", "$RECEIVER"]

[[lane]]
slot = "grok-hard"
enabled = true
restart = "always"
cwd = "$REPO"
agent = "grok"
lane = "fleet-grok-hard"
session_id = "ffffffff-eeee-4ddd-8ccc-bbbbbbbbbbbb"
identity = "standalone"
command = ["$ISOLATION_WRAPPER", "$TEST_ROOT/grok.never-fails", "$TEST_ROOT/grok.log", "$RECEIVER"]
EOF
mkdir -m 700 "$ISOLATION_BUDGETS" "$ISOLATION_LATCHES"
fleetd_isolation init --config "$ISOLATION_CONFIG" >/dev/null
fleetd_isolation authorize-recovery --config "$ISOLATION_CONFIG" \
  --slot cursor-hard --out "$ISOLATION_BUDGETS/cursor-hard.json" \
  --max-starts 2 --backoff-seconds 0 --ttl 600 >/dev/null
fleetd_isolation authorize-recovery --config "$ISOLATION_CONFIG" \
  --slot grok-hard --out "$ISOLATION_BUDGETS/grok-hard.json" \
  --max-starts 2 --backoff-seconds 0 --ttl 600 >/dev/null
touch "$CURSOR_FAIL_MARKER"
if fleetd_isolation watch --config "$ISOLATION_CONFIG" --cycles 1 \
  --interval 0.01 --apply-recovery \
  --recovery-budget-dir "$ISOLATION_BUDGETS" \
  --recovery-latch-dir "$ISOLATION_LATCHES" \
  >"$TEST_ROOT/isolation-first" 2>&1; then
  fail 'isolated recovery did not report the sabotaged Cursor launch'
fi
grep -q 'slot=cursor-hard action=start status=failed' \
  "$TEST_ROOT/isolation-first" || \
  fail 'Cursor sabotage was not attributed to its start action'
grep -q 'slot=cursor-hard status=set .*reason=start-action-failed' \
  "$TEST_ROOT/isolation-first" || \
  fail 'Cursor sabotage did not close its persistent recovery latch'
grep -q 'slot=grok-hard action=start status=committed' \
  "$TEST_ROOT/isolation-first" || \
  fail 'Cursor sabotage prevented independent Grok convergence'
[[ -f "$ISOLATION_LATCHES/cursor-hard.halted.json" ]] || \
  fail 'Cursor recovery latch was not retained on disk'
[[ "$(grep -c '^START pid=' "$TEST_ROOT/grok.log")" == 1 ]] || \
  fail 'Grok did not start exactly once beside the Cursor failure'
grok_generation="$(python3 - "$ISOLATION_STATE/fleet-slots/grok-hard.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["instance_id"])
PY
)"
SOUNIO_AGENTD_DIR="$ISOLATION_STATE" "$RUNTIME/sounio-fleet-agent-runtime" \
  stop --cwd "$REPO" --slot cursor-hard >/dev/null
output="$(fleetd_isolation watch --config "$ISOLATION_CONFIG" --cycles 1 \
  --interval 0.01 --apply-recovery \
  --recovery-budget-dir "$ISOLATION_BUDGETS" \
  --recovery-latch-dir "$ISOLATION_LATCHES")"
grep -q 'slot=cursor-hard action=start status=held reason=recovery-latch-present' \
  <<< "$output" || fail 'Cursor retried through its closed recovery latch'
cursor_spends="$(python3 - "$ISOLATION_DB" <<'PY'
import sqlite3
import sys
with sqlite3.connect(sys.argv[1]) as connection:
    print(connection.execute(
        "SELECT count(*) FROM events WHERE slot = 'cursor-hard' "
        "AND event_type = 'RECOVERY_BUDGET_SPENT'"
    ).fetchone()[0])
PY
)"
[[ "$cursor_spends" == 1 ]] || \
  fail 'closed Cursor latch spent an additional recovery ordinal'
rm "$CURSOR_FAIL_MARKER"
fleetd_isolation recovery-latch-clear --config "$ISOLATION_CONFIG" \
  --slot cursor-hard --recovery-latch-dir "$ISOLATION_LATCHES" \
  >"$TEST_ROOT/isolation-clear"
grep -q 'slot=cursor-hard status=cleared' "$TEST_ROOT/isolation-clear" || \
  fail 'explicit Cursor recovery-latch clear was not audited'
fleetd_isolation watch --config "$ISOLATION_CONFIG" --cycles 1 \
  --interval 0.01 --apply-recovery \
  --recovery-budget-dir "$ISOLATION_BUDGETS" \
  --recovery-latch-dir "$ISOLATION_LATCHES" >/dev/null
wait_for 'Cursor did not converge after an explicit latch clear' \
  "test \"\$(grep -c '^START pid=' '$TEST_ROOT/cursor.log')\" = 1"
[[ "$(python3 - "$ISOLATION_STATE/fleet-slots/grok-hard.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["instance_id"])
PY
)" == "$grok_generation" ]] || \
  fail 'Cursor recovery replaced the independent Grok generation'
chmod 755 "$ISOLATION_BUDGETS"
if fleetd_isolation watch --config "$ISOLATION_CONFIG" --cycles 1 \
  --interval 0.01 --apply-recovery \
  --recovery-budget-dir "$ISOLATION_BUDGETS" \
  --recovery-latch-dir "$ISOLATION_LATCHES" \
  >"$TEST_ROOT/isolation-permissions" 2>&1; then
  fail 'fleet recovery accepted a non-private budget directory'
fi
grep -q 'recovery budget directory permissions are not private' \
  "$TEST_ROOT/isolation-permissions" || \
  fail 'budget-directory sabotage was not attributed to directory authority'
chmod 700 "$ISOLATION_BUDGETS"
"$RUNTIME/sounio-fleet-trace-verify" --db "$ISOLATION_DB" \
  --certificate "$TEST_ROOT/isolation-trace-certificate.json" \
  >"$TEST_ROOT/isolation-trace"
python3 - "$TEST_ROOT/isolation-trace-certificate.json" <<'PY' || \
  fail 'independent trace verifier omitted the recovery-latch invariant'
import json
import sys
certificate = json.load(open(sys.argv[1], encoding="utf-8"))
assert certificate["invariants"]["recovery_latch_clear_is_identity_bound"] is True
assert certificate["invariants"]["recovery_latch_prevents_new_start_authority"] is True
PY
python3 - "$ISOLATION_DB" "$ISOLATION_TRACE_SABOTAGE_DB" <<'PY'
import hashlib
import json
import sqlite3
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

source = sqlite3.connect(sys.argv[1])
target = sqlite3.connect(sys.argv[2])
source.backup(target)
source.close()
target.row_factory = sqlite3.Row
row = target.execute(
    "SELECT * FROM events WHERE event_type = 'RECOVERY_LATCH_CLEAR_REQUESTED' "
    "ORDER BY seq LIMIT 1"
).fetchone()
assert row is not None
payload = json.loads(row["payload"])
payload["latch_id"] = "recovery-latch-rehashed-substitution"
target.execute(
    "UPDATE events SET payload = ? WHERE seq = ?", (canonical(payload), row["seq"])
)
previous = target.execute(
    "SELECT event_hash FROM events WHERE seq = ?", (row["seq"] - 1,)
).fetchone()[0]
for current in target.execute(
    "SELECT * FROM events WHERE seq >= ? ORDER BY seq", (row["seq"],)
).fetchall():
    material = {
        "causal_key": current["causal_key"],
        "event_type": current["event_type"],
        "occurred_utc": current["occurred_utc"],
        "payload": current["payload"],
        "prev_hash": previous,
        "seq": current["seq"],
        "slot": current["slot"],
    }
    event_hash = digest(material)
    target.execute(
        "UPDATE events SET prev_hash = ?, event_hash = ?, event_id = ? WHERE seq = ?",
        (previous, event_hash, f"evt-{event_hash[:24]}", current["seq"]),
    )
    previous = event_hash
target.commit()
target.close()
PY
if "$RUNTIME/sounio-fleet-trace-verify" --db "$ISOLATION_TRACE_SABOTAGE_DB" \
  >"$TEST_ROOT/isolation-trace-sabotage" 2>&1; then
  fail 'independent trace verifier accepted a rehashed recovery-latch clear'
fi
grep -q 'recovery latch clear latch_id mismatch' \
  "$TEST_ROOT/isolation-trace-sabotage" || \
  fail 'latch-clear sabotage was not attributed to exact latch identity'

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

echo 'sounio-coord-fleetd-selftest: PASS dry_run=no-mutation capability_required=1 capability_secret_sabotage=refused capability_reuse=refused stop_capability_required=1 stop_capability_reuse=refused stop_generation_sabotage=refused stop_semantic_rehash=refused recovery_budget=2 recovery_start_only=held recovery_backoff=enforced recovery_exhaustion=refused recovery_directory=private recovery_latch=per-slot latch_trace=verified latch_clear_sabotage=refused cursor_sabotage=isolated grok_convergence=preserved budget_semantic_rehash=refused backoff_semantic_rehash=refused duplicate_start=refused unreachable_start=blocked initial_on_failure=start omission=blocked generation_sabotage=blocked generation_authority_sabotage=blocked identity_sabotage=blocked checkpoint=draft-verified evidence_drift=refused prepared_evidence_drift=refused handoff=prepared-anchored-accepted handoff_reuse=refused ed25519_anchor=verified anchor_removal=refused signature_sabotage=refused key_substitution=refused trace_refinement=verified semantic_rehash_sabotage=refused replay=reconstructed hash_sabotage=refused'
