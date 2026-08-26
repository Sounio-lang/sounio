#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
LOOM="$ROOT_DIR/bin/sounio-loom"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-arrow.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
AGENT=loom-arrow-test
LANE=spectral-plane
SERVER_PID=''

export SOUNIO_COORD_RUNTIME_MODE=local
export SOUNIO_LOOM_COORD_AUTO=0

fail() {
  echo "sounio-loom-arrow-selftest: FAIL: $* test_root=$TEST_ROOT" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

cleanup() {
  [[ -z "$SERVER_PID" ]] || kill "$SERVER_PID" 2>/dev/null || true
  [[ -z "$SERVER_PID" ]] || wait "$SERVER_PID" 2>/dev/null || true
  "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

cat > "$TEST_ROOT/harness.sh" <<'HARNESS'
#!/bin/sh
stty -echo
printf 'SPECTRAL_READY\n'
while IFS= read -r line; do
  printf 'SPECTRAL_ECHO:%s\n' "$line"
done
HARNESS
chmod +x "$TEST_ROOT/harness.sh"

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

"$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
  --session-id loom-arrow-selftest --cwd "$TEST_ROOT" -- \
  /bin/sh "$TEST_ROOT/harness.sh" >/dev/null

status=''
for _ in $(seq 1 100); do
  status="$($LOOM status --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
    --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
  [[ "$status" == *'state=active'* && "$status" == *'output_cursor='* ]] && break
  sleep 0.05
done
[[ "$status" == *'state=active'* ]] || fail "session did not become active: $status"
journal="$(field journal "$status")"
[[ -s "$journal" ]] || fail 'semantic journal missing'

"$LOOM" export-events-arrow --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --out "$TEST_ROOT/events.arrow" > "$TEST_ROOT/export.out"
grep -q 'authority=verified-derived' "$TEST_ROOT/export.out" || \
  fail 'export did not declare derived authority'
inspect="$($LOOM verify-events-arrow --file "$TEST_ROOT/events.arrow")"
grep -q 'schema=loom-spectral-events-v1' <<< "$inspect" || \
  fail 'native IPC reader did not recover the Loom schema'
rows="$(sed -n 's/.* rows=\([0-9][0-9]*\).*/\1/p' <<< "$inspect")"
[[ -n "$rows" && "$rows" -gt 0 ]] || fail "expected at least one event row: $inspect"

"$LOOM" serve --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --bind 127.0.0.1 --port 0 > "$TEST_ROOT/server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 100); do
  grep -q 'LOOM_GUI url=' "$TEST_ROOT/server.log" && break
  sleep 0.05
done
port="$(sed -n 's/.*url=http:\/\/127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' \
  "$TEST_ROOT/server.log" | head -n 1)"
[[ -n "$port" ]] || fail 'read-only server did not publish its port'
curl -fsS -D "$TEST_ROOT/http.headers" \
  "http://127.0.0.1:$port/api/events.arrow" -o "$TEST_ROOT/http.arrow"
grep -qi '^Content-Type: application/vnd.apache.arrow.stream' \
  "$TEST_ROOT/http.headers" || fail 'HTTP projection used the wrong content type'
grep -qi '^X-Loom-Authority: verified-derived' "$TEST_ROOT/http.headers" || \
  fail 'HTTP projection omitted its authority boundary'
cmp "$TEST_ROOT/events.arrow" "$TEST_ROOT/http.arrow" >/dev/null || \
  fail 'CLI and HTTP projections diverged over the same journal snapshot'
"$LOOM" verify-events-arrow --file "$TEST_ROOT/http.arrow" >/dev/null

cp "$TEST_ROOT/events.arrow" "$TEST_ROOT/corrupt.arrow"
printf '\x00' | dd of="$TEST_ROOT/corrupt.arrow" bs=1 seek=0 count=1 \
  conv=notrunc status=none
if "$LOOM" verify-events-arrow --file "$TEST_ROOT/corrupt.arrow" \
  > "$TEST_ROOT/corrupt.out" 2> "$TEST_ROOT/corrupt.err"; then
  fail 'corrupted Arrow continuation marker passed native verification'
fi
grep -q 'Expected 0xFFFFFFFF' "$TEST_ROOT/corrupt.err" || \
  fail 'corrupted IPC stream failed for an unrelated reason'

awk -F '\t' 'BEGIN { OFS="\t" }
  NR == 1 { $3 = (substr($3, 1, 1) == "0" ? "1" : "0") substr($3, 2) }
  { print }' "$journal" > "$journal.tampered"
mv "$journal.tampered" "$journal"
if "$LOOM" export-events-arrow --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --out "$TEST_ROOT/laundered.arrow" \
  > "$TEST_ROOT/laundered.out" 2> "$TEST_ROOT/laundered.err"; then
  fail 'tampered authority journal was laundered into Arrow'
fi
grep -q 'hash:event-digest-mismatch seq=1' "$TEST_ROOT/laundered.err" || \
  fail 'journal sabotage was refused by an unrelated rule'

curl -sS -D "$TEST_ROOT/refused.headers" \
  "http://127.0.0.1:$port/api/events.arrow" -o "$TEST_ROOT/refused.json"
grep -q '^HTTP/1.1 409 Conflict' "$TEST_ROOT/refused.headers" || \
  fail 'HTTP projection did not fail closed after journal sabotage'
grep -q 'spectral_projection_refused' "$TEST_ROOT/refused.json" || \
  fail 'HTTP refusal omitted the spectral authority boundary'
grep -q 'hash:event-digest-mismatch seq=1' "$TEST_ROOT/refused.json" || \
  fail 'HTTP sabotage refusal did not preserve the causal reason'

printf 'SOUNIO_LOOM_ARROW_GATE_PASS=true schema=loom-spectral-events-v1 rows=%s sabotage=PASS runtime=OCaml+C\n' \
  "$rows"
