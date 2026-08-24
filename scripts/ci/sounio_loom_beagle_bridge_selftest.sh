#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
export SOUNIO_COORD_RUNTIME_MODE=local
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-beagle.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
BRIDGE_LOG="$TEST_ROOT/bridge.log"
BRIDGE_PID=''
BASE_URL=''
PANE_PATH='session-alpha%3Aterminal'

fail() {
  echo "sounio-loom-beagle-bridge-selftest: FAIL: $* test_root=$TEST_ROOT" >&2
  exit 1
}

json_value() {
  local file="$1" expression="$2"
  node --input-type=module -e \
    'import fs from "node:fs"; const value=JSON.parse(fs.readFileSync(process.argv[1], "utf8")); const result=Function("value", `return (${process.argv[2]})`)(value); if (result === undefined || result === null) process.exit(3); process.stdout.write(String(result));' \
    "$file" "$expression"
}

post_json() {
  local path="$1" body="$2" output="$3"
  curl --fail --silent --show-error --request POST \
    --header 'content-type: application/json' --data "$body" \
    "$BASE_URL$path" > "$output"
}

wait_snapshot() {
  local witness="$1" output="$2" attempt
  for attempt in $(seq 1 120); do
    curl --fail --silent --show-error \
      "$BASE_URL/v1/panes/$PANE_PATH/snapshot" > "$output" 2>/dev/null || true
    if [[ -s "$output" ]] && grep -Fq "$witness" "$output"; then
      return 0
    fi
    sleep 0.05
  done
  fail "snapshot did not contain $witness"
}

wait_state() {
  local expected="$1" output="$2" attempt state=''
  for attempt in $(seq 1 120); do
    curl --fail --silent --show-error \
      "$BASE_URL/v1/panes/$PANE_PATH/snapshot" > "$output" 2>/dev/null || true
    if [[ -s "$output" ]]; then
      state="$(json_value "$output" 'value.pane.loomState' 2>/dev/null || true)"
      [[ "$state" == "$expected" ]] && return 0
    fi
    sleep 0.05
  done
  fail "pane did not reach state=$expected; last=$state"
}

cleanup() {
  if [[ -n "$BASE_URL" ]]; then
    curl --silent --request POST --header 'content-type: application/json' \
      --data '{}' "$BASE_URL/v1/panes/$PANE_PATH/terminate" >/dev/null 2>&1 || true
  fi
  [[ -z "$BRIDGE_PID" ]] || kill "$BRIDGE_PID" >/dev/null 2>&1 || true
  [[ -z "$BRIDGE_PID" ]] || wait "$BRIDGE_PID" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

command -v curl >/dev/null || fail 'curl is required'
command -v node >/dev/null || fail 'node is required for the Beagle WebSocket compatibility probe'

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

if "$LOOM" beagle-serve --state-dir "$STATE_DIR" --bind 0.0.0.0 --port 0 \
  > "$TEST_ROOT/remote-bind.out" 2>&1; then
  fail 'non-loopback bridge bind succeeded without --allow-remote'
fi
grep -q 'remote Beagle bridge bind requires --allow-remote' \
  "$TEST_ROOT/remote-bind.out" || fail 'remote bind refusal omitted its reason'

"$LOOM" beagle-serve --state-dir "$STATE_DIR" --bind 127.0.0.1 --port 0 \
  > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

for _ in $(seq 1 100); do
  grep -q '^LOOM_BEAGLE_BRIDGE ' "$BRIDGE_LOG" 2>/dev/null && break
  kill -0 "$BRIDGE_PID" 2>/dev/null || fail "bridge exited: $(cat "$BRIDGE_LOG")"
  sleep 0.05
done
port="$(sed -n 's#.*url=http://127\.0\.0\.1:\([0-9][0-9]*\).*#\1#p' "$BRIDGE_LOG" | head -1)"
[[ -n "$port" ]] || fail 'bridge did not report its selected port'
BASE_URL="http://127.0.0.1:$port"

curl --fail --silent --show-error "$BASE_URL/v1/health" > "$TEST_ROOT/health.json"
[[ "$(json_value "$TEST_ROOT/health.json" 'value.authority')" == loom ]] || \
  fail 'health did not assign runtime authority to Loom'
[[ "$(json_value "$TEST_ROOT/health.json" 'value.supervisorProtocol')" == beagle-pty-supervisor-v1 ]] || \
  fail 'health did not preserve the Beagle supervisor protocol'

spawn_body="{\"sessionId\":\"session-alpha\",\"paneId\":\"session-alpha:terminal\",\"cwd\":\"$TEST_ROOT\",\"shell\":\"/bin/bash\",\"cols\":101,\"rows\":37}"
post_json /v1/spawn "$spawn_body" "$TEST_ROOT/spawn-one.json"

instance_one="$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.loomInstanceId')"
kernel_one="$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.loomKernelPid')"
guardian_one="$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.loomGuardianPid')"
harness_one="$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.pid')"
fingerprint_one="$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.generationFingerprint')"
[[ "$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.status')" == running ]] || \
  fail 'spawned pane is not running'
[[ "$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.cols')" == 101 ]] || \
  fail 'initial terminal width was not applied'
[[ "$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.authorityStatus.journalVerified')" == true ]] || \
  fail 'spawned pane journals did not verify'
[[ "$(json_value "$TEST_ROOT/spawn-one.json" 'value.pane.authorityStatus.kernelRecoveryCount')" == 0 ]] || \
  fail 'fresh generation reported a recovery'

post_json /v1/spawn "$spawn_body" "$TEST_ROOT/spawn-idempotent.json"
[[ "$(json_value "$TEST_ROOT/spawn-idempotent.json" 'value.pane.loomInstanceId')" == "$instance_one" ]] || \
  fail 'idempotent spawn replaced the physical generation'
[[ "$(json_value "$TEST_ROOT/spawn-idempotent.json" 'value.pane.pid')" == "$harness_one" ]] || \
  fail 'idempotent spawn replaced the harness'

conflict_code="$(curl --silent --output "$TEST_ROOT/conflict.json" --write-out '%{http_code}' \
  --request POST --header 'content-type: application/json' \
  --data "${spawn_body/session-alpha/session-beta}" "$BASE_URL/v1/spawn")"
[[ "$conflict_code" == 409 ]] || fail "identity conflict returned HTTP $conflict_code instead of 409"
grep -q 'pane-identity-conflict' "$TEST_ROOT/conflict.json" || \
  fail 'identity conflict omitted its refusal reason'

BASE_URL="$BASE_URL" PANE_PATH="$PANE_PATH" EXPECTED_INSTANCE="$instance_one" \
  node --input-type=module <<'NODE'
const base = process.env.BASE_URL;
const panePath = process.env.PANE_PATH;
const expected = process.env.EXPECTED_INSTANCE;
const ws = new WebSocket(`${base.replace(/^http/, "ws")}/v1/panes/${panePath}/stream`);
const timer = setTimeout(() => {
  console.error("websocket compatibility probe timed out");
  process.exit(2);
}, 8000);
let readyCursor = -1;
ws.onmessage = async (event) => {
  const message = JSON.parse(String(event.data));
  if (message.type === "ready") {
    if (message.supervisorProtocol !== "beagle-pty-supervisor-v1") process.exit(3);
    if (message.loomInstanceId !== expected) process.exit(4);
    readyCursor = Number(message.loomCursor);
    const response = await fetch(`${base}/v1/panes/${panePath}/input`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ data: "printf 'BEAGLE_WS_STREAM_OK\\n'\n" }),
    });
    if (!response.ok) process.exit(5);
  }
  if (message.type === "raw_output" && message.data.includes("BEAGLE_WS_STREAM_OK")) {
    if (Number(message.loomCursor) <= readyCursor) process.exit(6);
    if (message.loomInstanceId !== expected) process.exit(7);
    clearTimeout(timer);
    ws.close();
    process.exit(0);
  }
};
ws.onerror = (error) => {
  console.error(error);
  process.exit(8);
};
NODE

wait_snapshot BEAGLE_WS_STREAM_OK "$TEST_ROOT/stream-snapshot.json"

post_json "/v1/panes/$PANE_PATH/resize" '{"cols":132,"rows":44}' \
  "$TEST_ROOT/resize.json"
[[ "$(json_value "$TEST_ROOT/resize.json" 'value.pane.cols')" == 132 ]] || \
  fail 'resize width was not retained'
[[ "$(json_value "$TEST_ROOT/resize.json" 'value.pane.rows')" == 44 ]] || \
  fail 'resize height was not retained'

post_json "/v1/panes/$PANE_PATH/signal" '{"signal":"SIGINT"}' \
  "$TEST_ROOT/signal.json"
unsupported_signal_code="$(curl --silent --output "$TEST_ROOT/unsupported-signal.json" \
  --write-out '%{http_code}' --request POST --header 'content-type: application/json' \
  --data '{"signal":"SIGKILL"}' "$BASE_URL/v1/panes/$PANE_PATH/signal")"
[[ "$unsupported_signal_code" == 400 ]] || \
  fail "unsupported signal returned HTTP $unsupported_signal_code instead of 400"
grep -q 'unsupported-signal' "$TEST_ROOT/unsupported-signal.json" || \
  fail 'unsupported signal omitted its refusal reason'

semantic_journal="$(find "$STATE_DIR" -path '*/generations/*/journal.tsv' -print -quit)"
guardian_journal="$(find "$STATE_DIR" -path '*/generations/*/guardian.tsv' -print -quit)"
[[ -f "$semantic_journal" && -f "$guardian_journal" ]] || \
  fail 'generation journals are missing'
"$LOOM" verify-journal --journal "$semantic_journal" > "$TEST_ROOT/semantic-verify.out"
"$LOOM" verify-guardian-journal --journal "$guardian_journal" > "$TEST_ROOT/guardian-verify.out"
grep -q $'\tRESIZE\t' "$semantic_journal" || fail 'semantic journal omitted resize'
grep -q $'\tRESIZE\t' "$guardian_journal" || fail 'guardian journal omitted resize'
grep -q $'\tSIGNAL\t' "$semantic_journal" || fail 'semantic journal omitted signal'
grep -q $'\tSIGNAL\t' "$guardian_journal" || fail 'guardian journal omitted signal'

kill -9 "$kernel_one"
post_json /v1/spawn "$spawn_body" "$TEST_ROOT/spawn-recovered.json"
instance_recovered="$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.loomInstanceId')"
kernel_recovered="$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.loomKernelPid')"
[[ "$instance_recovered" == "$instance_one" ]] || fail 'kernel recovery replaced the Loom generation'
[[ "$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.pid')" == "$harness_one" ]] || \
  fail 'kernel recovery replaced the harness'
[[ "$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.loomGuardianPid')" == "$guardian_one" ]] || \
  fail 'kernel recovery replaced the Guardian'
[[ "$kernel_recovered" != "$kernel_one" ]] || fail 'kernel recovery retained the dead kernel pid'
[[ "$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.authorityStatus.kernelRecoveryCount')" == 1 ]] || \
  fail 'recovery evidence was not exposed to Beagle'
[[ "$(json_value "$TEST_ROOT/spawn-recovered.json" 'value.pane.authorityStatus.journalVerified')" == true ]] || \
  fail 'recovered journals did not verify'

post_json "/v1/panes/$PANE_PATH/input" '{"data":"printf '\''AFTER_RECOVERY_OK\\n'\''\n"}' \
  "$TEST_ROOT/input-after-recovery.json"
wait_snapshot AFTER_RECOVERY_OK "$TEST_ROOT/recovery-snapshot.json"

post_json "/v1/panes/$PANE_PATH/terminate" '{}' "$TEST_ROOT/terminate.json"
wait_state exited "$TEST_ROOT/exited.json"

post_json /v1/spawn "$spawn_body" "$TEST_ROOT/spawn-two.json"
instance_two="$(json_value "$TEST_ROOT/spawn-two.json" 'value.pane.loomInstanceId')"
fingerprint_two="$(json_value "$TEST_ROOT/spawn-two.json" 'value.pane.generationFingerprint')"
[[ "$instance_two" != "$instance_one" ]] || fail 'a terminated pane was laundered as its old generation'
[[ "$fingerprint_two" != "$fingerprint_one" ]] || fail 'new generation retained the old fingerprint'

post_json "/v1/panes/$PANE_PATH/terminate" '{}' "$TEST_ROOT/terminate-two.json"
wait_state exited "$TEST_ROOT/exited-two.json"

echo "sounio-loom-beagle-bridge-selftest: PASS protocol=beagle-pty-supervisor-v1 authority=loom kernel_recovery=same-generation respawn=new-generation websocket=pass journals=verified"
