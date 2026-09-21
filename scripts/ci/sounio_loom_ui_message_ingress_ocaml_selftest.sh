#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/ui-message-ingress.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SERVER_PID=''

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-ui-message-ingress-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
[[ -x "$LOOM" ]] || fail 'OCaml runtime is absent'

COORD_FIXTURE="$TEST_ROOT/sounio-coord-runtime"
COORD_LOG="$TEST_ROOT/coord.argv"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'printf "%s\n" "$@" > "$SOUNIO_UI_COORD_FIXTURE_LOG"' \
  '[[ "$1" == send ]]' \
  'printf "%s\n" "SENT message_id=msg-ui-selftest to_agent=target to_lane=lane kind=request thread_id=msg-ui-selftest reply_to=-"' \
  'printf "%s\n" "WAKE_UNAVAILABLE message_id=msg-ui-selftest status=unavailable"' \
  > "$COORD_FIXTURE"
chmod 700 "$COORD_FIXTURE"

SERVER_LOG="$TEST_ROOT/server.log"
SOUNIO_COORD_COMMAND="$COORD_FIXTURE" \
SOUNIO_UI_COORD_FIXTURE_LOG="$COORD_LOG" \
  "$LOOM" serve --cwd "$ROOT_DIR" --bind 127.0.0.1 --port 0 \
    --write-agent founder --write-lane loom-ui >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

url=''
for _ in $(seq 1 100); do
  url="$(sed -n 's/^LOOM_GUI url=\([^ ]*\).*/\1/p' "$SERVER_LOG" | tail -n 1)"
  [[ -n "$url" ]] && break
  kill -0 "$SERVER_PID" 2>/dev/null || fail "server exited: $(cat "$SERVER_LOG")"
  sleep 0.05
done
[[ -n "$url" ]] || fail 'server URL was not published'

state="$(curl -fsS "$url/api/write-state")"
token="$(sed -n 's/.*"token":"\([^"]*\)".*/\1/p' <<< "$state")"
[[ "$state" == *'"enabled":true'* && ${#token} -eq 64 ]] ||
  fail "write state is not capability-bound: $state"

receipt="$(curl -fsS \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H "Origin: $url" \
  --data '{"toAgent":"target","toLane":"lane","kind":"request","message":"review the frozen artifact"}' \
  "$url/api/send")"
[[ "$receipt" == *'"durable":true'* && \
  "$receipt" == *'"messageId":"msg-ui-selftest"'* && \
  "$receipt" == *'"wake":"unavailable"'* ]] ||
  fail "valid send lost its canonical receipt: $receipt"

mapfile -t argv < "$COORD_LOG"
expected=(send --agent founder --lane loom-ui --to-agent target --to-lane lane --kind request --message 'review the frozen artifact')
[[ "${argv[*]}" == "${expected[*]}" ]] ||
  fail "canonical coordination argv mismatch: ${argv[*]}"

expect_refused() {
  local name="$1" expected="$2"
  shift 2
  local body="$TEST_ROOT/$name.json" status
  status="$(curl -sS -o "$body" -w '%{http_code}' "$@")"
  [[ "$status" == "$expected" ]] ||
    fail "$name expected HTTP $expected, got $status: $(cat "$body")"
}

expect_refused missing-token 403 \
  -H 'Content-Type: application/json' \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"deny"}' \
  "$url/api/send"
expect_refused cross-site 403 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: cross-site' \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"deny"}' \
  "$url/api/send"
expect_refused no-browser-origin 403 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: none' \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"deny"}' \
  "$url/api/send"
expect_refused mismatched-origin 403 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H 'Origin: http://attacker.invalid' \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"deny"}' \
  "$url/api/send"
expect_refused unsupported-kind 400 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H "Origin: $url" \
  --data '{"toAgent":"target","toLane":"lane","kind":"handoff","message":"deny"}' \
  "$url/api/send"
expect_refused empty-message 400 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H "Origin: $url" \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":""}' \
  "$url/api/send"
expect_refused invalid-target 400 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H "Origin: $url" \
  --data '{"toAgent":"target --agent forged","toLane":"lane","kind":"info","message":"deny"}' \
  "$url/api/send"
expect_refused control-message 400 \
  -H 'Content-Type: application/json' \
  -H "X-Loom-Write-Token: $token" \
  -H 'Sec-Fetch-Site: same-origin' \
  -H "Origin: $url" \
  --data '{"toAgent":"target","toLane":"lane","kind":"info","message":"deny\nforged"}' \
  "$url/api/send"

set +e
remote_output="$(SOUNIO_COORD_COMMAND="$COORD_FIXTURE" \
  SOUNIO_UI_COORD_FIXTURE_LOG="$COORD_LOG" \
  "$LOOM" serve --cwd "$ROOT_DIR" --bind 0.0.0.0 --allow-remote --port 0 \
    --write-agent founder --write-lane loom-ui 2>&1)"
remote_rc=$?
set -e
[[ "$remote_rc" -ne 0 && "$remote_output" == *'requires a loopback bind'* ]] ||
  fail "remote write mode did not fail closed: rc=$remote_rc output=$remote_output"

printf 'sounio-loom-ui-message-ingress-ocaml-selftest: PASS operational_realization=OCaml canonical_bus=argv-direct durable_receipt=true loopback_write_only=true csrf_token=true missing_token=REFUSED cross_site=REFUSED no_browser_origin=REFUSED mismatched_origin=REFUSED unsupported_kind=REFUSED empty_message=REFUSED invalid_target=REFUSED control_message=REFUSED remote_write=REFUSED python_executed=false rust_executed=false disposable_oracle_executed=false\n'
