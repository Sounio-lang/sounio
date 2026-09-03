#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-message-bridge.XXXXXX")"
TOKEN_FILE="$TEST_ROOT/token.cap"
BRIDGE_LOG="$TEST_ROOT/bridge.log"
BRIDGE_PID=''
BASE_URL=''
PROVIDER_FIXTURE_LOG="$TEST_ROOT/provider-fixture.log"
CODEX_SESSIONS="$TEST_ROOT/codex-sessions"
SECRET='4bdf72c971154463b928f6d64c8cb5cab245107657fc93dd9d9bcc50e7e7b186'
MESSAGE='message bridge isolated acceptance canary'

export SOUNIO_COORD_DIR="$TEST_ROOT/coord-state"
export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_COORD_DURABLE_OBLIGATIONS=0

fail() {
  echo "sounio-loom-message-bridge-selftest: FAIL: $* test_root=$TEST_ROOT" >&2
  exit 1
}

cleanup() {
  [[ -z "$BRIDGE_PID" ]] || kill "$BRIDGE_PID" >/dev/null 2>&1 || true
  [[ -z "$BRIDGE_PID" ]] || wait "$BRIDGE_PID" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

http_status() {
  local output="$1"
  shift
  curl --silent --show-error --max-time 15 --output "$output" \
    --write-out '%{http_code}' "$@"
}

route_http_status() {
  local output="$1"
  shift
  curl --silent --show-error --max-time 75 --output "$output" \
    --write-out '%{http_code}' "$@"
}

command -v curl >/dev/null || fail 'curl is required'
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

printf '%s\n' "$SECRET" > "$TOKEN_FILE"
chmod 0644 "$TOKEN_FILE"
if "$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TOKEN_FILE" \
  --bind 127.0.0.1 --port 0 >"$TEST_ROOT/insecure-token.out" 2>&1; then
  fail 'bridge accepted a group/world-readable capability'
fi
grep -q 'message-bridge-token-permissions' "$TEST_ROOT/insecure-token.out" ||
  fail 'insecure token refusal omitted its reason'

if "$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TEST_ROOT/missing.cap" \
  --bind 127.0.0.1 --port 0 >"$TEST_ROOT/missing-token.out" 2>&1; then
  fail 'bridge accepted a missing capability file'
fi
grep -q 'message-bridge-token-missing' "$TEST_ROOT/missing-token.out" ||
  fail 'missing token refusal omitted its reason'

chmod 0600 "$TOKEN_FILE"
ln -s "$TOKEN_FILE" "$TEST_ROOT/token-link.cap"
if "$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TEST_ROOT/token-link.cap" \
  --bind 127.0.0.1 --port 0 >"$TEST_ROOT/symlink-token.out" 2>&1; then
  fail 'bridge accepted a symlinked capability file'
fi
grep -q 'message-bridge-token-not-regular' "$TEST_ROOT/symlink-token.out" ||
  fail 'symlinked token refusal omitted its reason'

printf '%s\n' '4bdf72c971154463b928f6d64c8cb5ca 245107657fc93dd9d9bcc50e7e7b186' \
  > "$TEST_ROOT/whitespace-token.cap"
chmod 0600 "$TEST_ROOT/whitespace-token.cap"
if "$LOOM" message-serve --cwd "$ROOT_DIR" \
  --token-file "$TEST_ROOT/whitespace-token.cap" --bind 127.0.0.1 --port 0 \
  >"$TEST_ROOT/whitespace-token.out" 2>&1; then
  fail 'bridge accepted whitespace inside a bearer capability'
fi
grep -q 'message-bridge-token-invalid' "$TEST_ROOT/whitespace-token.out" ||
  fail 'whitespace token refusal omitted its reason'

if "$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TOKEN_FILE" \
  --bind 0.0.0.0 --port 0 >"$TEST_ROOT/remote-bind.out" 2>&1; then
  fail 'bridge accepted a non-loopback bind without --allow-remote'
fi
grep -q 'remote message bridge bind requires --allow-remote' \
  "$TEST_ROOT/remote-bind.out" || fail 'remote bind refusal omitted its reason'

mkdir -p "$CODEX_SESSIONS/2026/09/03"
printf '%s\n' \
  '{"timestamp":"2026-09-03T00:00:00Z","type":"event_msg","payload":{"type":"token_count","rate_limits":{"primary":{"used_percent":12.5,"window_minutes":300,"resets_at":1788479999}}}}' \
  >"$CODEX_SESSIONS/2026/09/03/route.jsonl"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'case "${1:-}" in' \
  '  provider-status) printf '\''%s\n'\'' '\''{"schema":"loom-provider-abi-v1","status":{"provider":"codex","installed":true,"auth":"authenticated"}}'\'' ;;' \
  '  provider-plan) printf '\''%s\n'\'' '\''{"schema":"loom-provider-plan-fixture-v1","provider":"codex","role":"REVIEW_ONLY"}'\'' ;;' \
  '  provider-start) printf '\''provider-start\n'\'' >>"$SOUNIO_LOOM_PROVIDER_FIXTURE_LOG" ;;' \
  '  *) exit 64 ;;' \
  'esac' >"$TEST_ROOT/provider-fixture"
chmod 0700 "$TEST_ROOT/provider-fixture"

SOUNIO_LOOM_COMMAND="$TEST_ROOT/provider-fixture" \
SOUNIO_LOOM_CODEX_SESSIONS_DIR="$CODEX_SESSIONS" \
SOUNIO_LOOM_PROVIDER_FIXTURE_LOG="$PROVIDER_FIXTURE_LOG" \
"$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TOKEN_FILE" \
  --routing-state-dir "$TEST_ROOT/routing-state" \
  --agent loom-ui-test --lane apple-client-test --bind 127.0.0.1 --port 0 \
  >"$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

for _ in $(seq 1 100); do
  grep -q '^LOOM_MESSAGE_BRIDGE ' "$BRIDGE_LOG" 2>/dev/null && break
  kill -0 "$BRIDGE_PID" 2>/dev/null || fail "bridge exited: $(cat "$BRIDGE_LOG")"
  sleep 0.05
done
port="$(sed -n 's#.*url=http://127\.0\.0\.1:\([0-9][0-9]*\).*#\1#p' \
  "$BRIDGE_LOG" | head -1)"
[[ -n "$port" ]] || fail 'bridge did not report its selected port'
BASE_URL="http://127.0.0.1:$port"

status="$(http_status "$TEST_ROOT/health.json" "$BASE_URL/health")"
[[ "$status" == 200 ]] || fail "health returned HTTP $status"
grep -q '"schema":"loom-message-bridge-v1"' "$TEST_ROOT/health.json" ||
  fail 'health omitted the bridge schema'

status="$(http_status "$TEST_ROOT/no-token.json" --request POST \
  --header 'content-type: application/json' --data '{}' "$BASE_URL/v1/messages")"
[[ "$status" == 401 ]] || fail "missing capability returned HTTP $status"

status="$(http_status "$TEST_ROOT/no-token-threads.json" "$BASE_URL/v1/threads")"
[[ "$status" == 401 ]] || fail "thread list without capability returned HTTP $status"

status="$(http_status "$TEST_ROOT/no-token-routing.json" "$BASE_URL/v1/routing/config")"
[[ "$status" == 401 ]] || fail "routing config without capability returned HTTP $status"

status="$(http_status "$TEST_ROOT/wrong-token.json" --request POST \
  --header 'content-type: application/json' --header 'authorization: Bearer wrong' \
  --data '{}' "$BASE_URL/v1/messages")"
[[ "$status" == 401 ]] || fail "wrong capability returned HTTP $status"

status="$(http_status "$TEST_ROOT/bad-kind.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data '{"toAgent":"target","toLane":"lane","kind":"handoff","message":"no"}' \
  "$BASE_URL/v1/messages")"
[[ "$status" == 400 ]] || fail "forbidden kind returned HTTP $status"

status="$(http_status "$TEST_ROOT/bad-target.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data '{"toAgent":"bad\nagent","toLane":"lane","kind":"info","message":"no"}' \
  "$BASE_URL/v1/messages")"
[[ "$status" == 400 ]] || fail "newline-bearing target returned HTTP $status"

status="$(http_status "$TEST_ROOT/accepted.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data "{\"toAgent\":\"bridge-target\",\"toLane\":\"isolated-lane\",\"kind\":\"request\",\"message\":\"$MESSAGE\"}" \
  "$BASE_URL/v1/messages")"
[[ "$status" == 202 ]] || fail "valid message returned HTTP $status: $(cat "$TEST_ROOT/accepted.json")"
grep -q '"schema":"loom-message-receipt-v1"' "$TEST_ROOT/accepted.json" ||
  fail 'accepted response omitted the receipt schema'
grep -q '"status":"accepted"' "$TEST_ROOT/accepted.json" ||
  fail 'accepted response omitted accepted status'
message_id="$(sed -n 's/.*"messageId":"\([^"]*\)".*/\1/p' "$TEST_ROOT/accepted.json")"
[[ -n "$message_id" ]] || fail 'accepted response omitted messageId'
message_file="$SOUNIO_COORD_DIR/messages/$message_id.message"
[[ -f "$message_file" ]] || fail 'receipt did not identify a durable bus record'
grep -q '^from_agent=loom-ui-test$' "$message_file" || fail 'durable sender agent drifted'
grep -q '^from_lane=apple-client-test$' "$message_file" || fail 'durable sender lane drifted'
grep -q '^to_agent=bridge-target$' "$message_file" || fail 'durable destination drifted'
grep -q '^kind=request$' "$message_file" || fail 'durable message kind drifted'
grep -Fq "text=$MESSAGE" "$message_file" || fail 'durable message body drifted'

status="$(http_status "$TEST_ROOT/thread-list.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/threads?limit=5")"
[[ "$status" == 200 ]] || fail "thread list returned HTTP $status"
grep -q '"schema":"loom-message-thread-list-v1"' "$TEST_ROOT/thread-list.json" ||
  fail 'thread list omitted its schema'
grep -q "\"id\":\"$message_id\"" "$TEST_ROOT/thread-list.json" ||
  fail 'thread list omitted the durable request'

status="$(http_status "$TEST_ROOT/thread-timeout.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/threads/$message_id?timeoutSeconds=0")"
[[ "$status" == 200 ]] || fail "thread detail returned HTTP $status"
grep -q '"schema":"loom-message-thread-v1"' "$TEST_ROOT/thread-timeout.json" ||
  fail 'thread detail omitted its schema'
grep -q '"kind":"durable_only"' "$TEST_ROOT/thread-timeout.json" ||
  fail 'thread detail omitted the durable-only event'
grep -q '"kind":"timeout"' "$TEST_ROOT/thread-timeout.json" ||
  fail 'thread detail omitted the bus-clock timeout event'

reply_output="$("$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh" reply \
  --agent bridge-target --lane isolated-lane --reply-to "$message_id" \
  --message 'Thread Truth reply')"
reply_id="$(sed -n 's/^SENT message_id=\([^ ]*\).*/\1/p' <<< "$reply_output")"
[[ -n "$reply_id" ]] || fail 'thread fixture did not create a correlated reply'

status="$(http_status "$TEST_ROOT/thread-answered.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/threads/$message_id")"
[[ "$status" == 200 ]] || fail "answered thread detail returned HTTP $status"
grep -q '"kind":"response"' "$TEST_ROOT/thread-answered.json" ||
  fail 'thread detail omitted the correlated response event'
grep -q 'Thread Truth reply' "$TEST_ROOT/thread-answered.json" ||
  fail 'thread detail omitted the correlated response body'

status="$(http_status "$TEST_ROOT/thread-ack.json" --request POST \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/messages/$reply_id/ack")"
[[ "$status" == 200 ]] || fail "thread acknowledgement returned HTTP $status"
grep -q '"schema":"loom-message-ack-v1"' "$TEST_ROOT/thread-ack.json" ||
  fail 'thread acknowledgement omitted its schema'

status="$(http_status "$TEST_ROOT/thread-acked.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/threads/$message_id")"
[[ "$status" == 200 ]] || fail "acknowledged thread detail returned HTTP $status"
grep -q '"kind":"ack"' "$TEST_ROOT/thread-acked.json" ||
  fail 'thread detail omitted the acknowledgement event'

status="$(http_status "$TEST_ROOT/routing-default.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/routing/config")"
[[ "$status" == 200 ]] || fail "default routing config returned HTTP $status"
grep -q '"schema":"loom-routing-config-v1"' "$TEST_ROOT/routing-default.json" ||
  fail 'default routing config omitted its schema'
grep -q '"revision":0' "$TEST_ROOT/routing-default.json" ||
  fail 'default routing config did not start at revision zero'

routing_update='{"schema":"loom-routing-config-v1","policy":"authority-first","model":"gpt-5.6-sol","effort":"high","poolOrder":["pool-openai-team"],"adapterOrder":["adapter-codex"]}'
status="$(http_status "$TEST_ROOT/routing-stored.json" --request PUT \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data "$routing_update" "$BASE_URL/v1/routing/config")"
[[ "$status" == 200 ]] || fail "valid routing config returned HTTP $status: $(cat "$TEST_ROOT/routing-stored.json")"
grep -q '"schema":"loom-routing-config-receipt-v1"' "$TEST_ROOT/routing-stored.json" ||
  fail 'routing config receipt omitted its schema'
grep -q '"revision":1' "$TEST_ROOT/routing-stored.json" ||
  fail 'routing config receipt did not advance revision'
grep -q '"status":"stored"' "$TEST_ROOT/routing-stored.json" ||
  fail 'routing config receipt did not confirm persistence'
[[ -f "$TEST_ROOT/routing-state/routing-config-v1.json" ]] ||
  fail 'routing config did not persist in the configured private state directory'
[[ "$(stat -c '%a' "$TEST_ROOT/routing-state/routing-config-v1.json")" == 600 ]] ||
  fail 'routing config state file permissions are not private'

status="$(http_status "$TEST_ROOT/routing-readback.json" \
  --header "authorization: Bearer $SECRET" "$BASE_URL/v1/routing/config")"
[[ "$status" == 200 ]] || fail "stored routing config readback returned HTTP $status"
grep -q '"model":"gpt-5.6-sol"' "$TEST_ROOT/routing-readback.json" ||
  fail 'stored routing config did not preserve the model'
grep -q '"revision":1' "$TEST_ROOT/routing-readback.json" ||
  fail 'stored routing config revision drifted'

status="$(http_status "$TEST_ROOT/routing-duplicate.json" --request PUT \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data '{"schema":"loom-routing-config-v1","policy":"authority-first","model":"gpt-5.6-sol","effort":"high","poolOrder":["pool-openai-team","pool-openai-team"],"adapterOrder":["adapter-codex"]}' \
  "$BASE_URL/v1/routing/config")"
[[ "$status" == 400 ]] || fail "duplicate routing pool returned HTTP $status"
grep -q 'message-bridge-routing-pool-order-duplicate' "$TEST_ROOT/routing-duplicate.json" ||
  fail 'duplicate routing pool refusal omitted its reason'

route_task='{"schema":"loom-route-task-v1","taskId":"route-positive","kind":"review","title":"Review evidence","prompt":"Report risks without changing files."}'
status="$(route_http_status "$TEST_ROOT/route-positive.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data "$route_task" "$BASE_URL/v1/routing/tasks")"
[[ "$status" == 200 ]] || fail "positive route returned HTTP $status"
grep -q '"schema":"loom-route-operation-v1"' "$TEST_ROOT/route-positive.json" ||
  fail 'positive route omitted operation schema'
grep -q '"status":"running"' "$TEST_ROOT/route-positive.json" ||
  fail 'positive route did not confirm provider custody'
grep -q '"producingLanguage":"Sounio"' "$TEST_ROOT/route-positive.json" ||
  fail 'positive route did not preserve Sounio semantic authority'
grep -q '"providerRole":"REVIEW_ONLY"' "$TEST_ROOT/route-positive.json" ||
  fail 'positive route promoted the provider role'
[[ "$(wc -l <"$PROVIDER_FIXTURE_LOG")" == 1 ]] ||
  fail 'positive route did not launch exactly one provider fixture'

rm "$CODEX_SESSIONS/2026/09/03/route.jsonl"
unknown_task='{"schema":"loom-route-task-v1","taskId":"route-unknown","kind":"review","title":"Unknown quota","prompt":"This route must remain closed."}'
status="$(route_http_status "$TEST_ROOT/route-unknown.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data "$unknown_task" "$BASE_URL/v1/routing/tasks")"
[[ "$status" == 200 ]] || fail "unknown quota route returned HTTP $status"
grep -q '"reason":"quota-unknown"' "$TEST_ROOT/route-unknown.json" ||
  fail 'unknown quota did not produce Sounio DENY608'
grep -q '"status":"refused"' "$TEST_ROOT/route-unknown.json" ||
  fail 'unknown quota did not fail closed'
[[ "$(wc -l <"$PROVIDER_FIXTURE_LOG")" == 1 ]] ||
  fail 'unknown quota launched a provider'

printf '%s\n' \
  '{"timestamp":"2026-09-03T00:00:00Z","type":"event_msg","payload":{"type":"token_count","rate_limits":{"primary":{"used_percent":12.5,"window_minutes":300,"resets_at":1788479999}}}}' \
  >"$CODEX_SESSIONS/2026/09/03/route.jsonl"
ownership_task='{"schema":"loom-route-task-v1","taskId":"route-ownership","kind":"write","title":"Forbidden write","prompt":"This must never launch."}'
status="$(route_http_status "$TEST_ROOT/route-ownership.json" --request POST \
  --header 'content-type: application/json' --header "authorization: Bearer $SECRET" \
  --data "$ownership_task" "$BASE_URL/v1/routing/tasks")"
[[ "$status" == 200 ]] || fail "ownership route returned HTTP $status"
grep -q '"reason":"ownership-block"' "$TEST_ROOT/route-ownership.json" ||
  fail 'write-shaped route did not produce Sounio DENY607'
[[ "$(wc -l <"$PROVIDER_FIXTURE_LOG")" == 1 ]] ||
  fail 'ownership refusal launched a provider'

python_frame='9032 3 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 7 8 6 6 0 0 0 0 0 0 0'
printf '%s\n' "$python_frame" | \
  "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-routing-authority-runtime" \
  >"$TEST_ROOT/python-oracle-deny.out"
grep -q 'DENY code=617 reason=language-role-forbidden' \
  "$TEST_ROOT/python-oracle-deny.out" ||
  fail 'Sounio did not deny the deliberate Python oracle frame'
[[ "$(wc -l <"$PROVIDER_FIXTURE_LOG")" == 1 ]] ||
  fail 'Python oracle denial launched a provider'

grep -q 'LOOM_MESSAGE_DECISION decision=ALLOW' "$BRIDGE_LOG" ||
  fail 'bridge did not audit its ALLOW decision'
[[ "$(grep -c 'LOOM_MESSAGE_DECISION decision=DENY' "$BRIDGE_LOG")" -ge 4 ]] ||
  fail 'bridge did not audit every DENY decision'
if grep -Fq "$SECRET" "$BRIDGE_LOG"; then
  fail 'bridge leaked its bearer capability into the audit log'
fi
if grep -Fq "$MESSAGE" "$BRIDGE_LOG"; then
  fail 'bridge leaked message content into the audit log'
fi

kill "$BRIDGE_PID" >/dev/null 2>&1 || true
wait "$BRIDGE_PID" >/dev/null 2>&1 || true
BRIDGE_PID=''
printf '%s\n' '#!/usr/bin/env bash' 'sleep 30' >"$TEST_ROOT/stalled-coord"
chmod 0700 "$TEST_ROOT/stalled-coord"
SOUNIO_COORD_COMMAND="$TEST_ROOT/stalled-coord" \
  "$LOOM" message-serve --cwd "$ROOT_DIR" --token-file "$TOKEN_FILE" \
  --agent loom-ui-test --lane timeout-test --bind 127.0.0.1 --port 0 \
  >"$TEST_ROOT/timeout-bridge.log" 2>&1 &
BRIDGE_PID=$!
for _ in $(seq 1 100); do
  grep -q '^LOOM_MESSAGE_BRIDGE ' "$TEST_ROOT/timeout-bridge.log" 2>/dev/null && break
  kill -0 "$BRIDGE_PID" 2>/dev/null || fail 'timeout bridge exited before serving'
  sleep 0.05
done
timeout_port="$(sed -n 's#.*url=http://127\.0\.0\.1:\([0-9][0-9]*\).*#\1#p' \
  "$TEST_ROOT/timeout-bridge.log" | head -1)"
[[ -n "$timeout_port" ]] || fail 'timeout bridge did not report its selected port'
BASE_URL="http://127.0.0.1:$timeout_port"
status="$(curl --silent --show-error --max-time 12 --output "$TEST_ROOT/timeout.json" \
  --write-out '%{http_code}' --request POST --header 'content-type: application/json' \
  --header "authorization: Bearer $SECRET" \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"timeout"}' \
  "$BASE_URL/v1/messages")"
[[ "$status" == 503 ]] || fail "stalled runtime returned HTTP $status"
grep -q 'message-bridge-runtime-timeout' "$TEST_ROOT/timeout.json" ||
  fail 'timeout refusal omitted its reason'
grep -q 'decision=DENY reason=message-bridge-runtime-timeout' \
  "$TEST_ROOT/timeout-bridge.log" || fail 'timeout refusal was not audited'

echo "sounio-loom-message-bridge-selftest: PASS receipt=$message_id authority=durable-message-bus"
