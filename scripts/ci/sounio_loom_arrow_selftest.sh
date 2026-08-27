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

descriptor="$STATE_DIR/sessions/${AGENT}--${LANE}/session.state"
[[ -f "$descriptor" ]] || fail 'session descriptor missing'
cp "$descriptor" "$TEST_ROOT/current-session.state"
legacy_instance=a42a648cdb2e2dc5c9736a6dd1dfd4ec
legacy_dir="$STATE_DIR/sessions/legacy-arrow--pre-guardian"
legacy_generation="$legacy_dir/generations/$legacy_instance"
legacy_journal="$legacy_generation/journal.tsv"
mkdir -p "$legacy_generation"
cat > "$legacy_journal" <<'LEGACY_JOURNAL'
1	0000000000000000000000000000000000000000000000000000000000000000	f724187be5efcdc5b3acb73110efcbc5a6c868c055cdac8f916b2a5e98c13d8b	2026-08-24T02:54:31.822423Z	SESSION_STARTED	61343261363438636462326532646335633937333661366464316466643465633a323332343632
2	f724187be5efcdc5b3acb73110efcbc5a6c868c055cdac8f916b2a5e98c13d8b	c0bccc7d64c0a1249c35cd1ffcabc73db34a4805fdd47a9c07ae56b30c233780	2026-08-24T02:54:31.860836Z	OUTPUT	303a33363a33653035336231363133333339656464303134373332646565343036613530623732663132646563653063363132613731396134653734363736623834393965
3	c0bccc7d64c0a1249c35cd1ffcabc73db34a4805fdd47a9c07ae56b30c233780	35591495e6b73ab7c4e1a380a538aa0edb4b2adc63667ddadb09bedc190bfbb4	2026-08-24T08:58:57.321789Z	SESSION_EXITED	313137
LEGACY_JOURNAL
cat > "$legacy_generation/session.state" <<LEGACY_GENERATION
protocol=1
runtime_version=2026.08.24.0
state=exited
agent=legacy-arrow
lane=pre-guardian
instance_id=$legacy_instance
journal_file=$legacy_journal
started_utc=2026-08-24T02:54:31.808998Z
LEGACY_GENERATION
cp "$legacy_generation/session.state" "$legacy_dir/session.state"

"$LOOM" export-events-arrow --state-dir "$STATE_DIR" --cwd "$TEST_ROOT" \
  --out "$TEST_ROOT/legacy.arrow" > "$TEST_ROOT/legacy.out"
grep -q 'legacy_semantic_only_sessions=1' "$TEST_ROOT/legacy.out" || \
  fail 'known pre-Guardian runtime did not declare semantic-only projection'
curl -fsS -D "$TEST_ROOT/legacy.headers" \
  "http://127.0.0.1:$port/api/events.arrow" -o "$TEST_ROOT/legacy-http.arrow"
grep -qi '^X-Loom-Legacy-Semantic-Only-Sessions: 1' \
  "$TEST_ROOT/legacy.headers" || \
  fail 'HTTP projection hid its pre-Guardian session count'
curl -fsS "http://127.0.0.1:$port/api/events" \
  -o "$TEST_ROOT/legacy-events.json"
grep -q '"journal_profile":"semantic-only-legacy"' \
  "$TEST_ROOT/legacy-events.json" || \
  fail 'JSON projection disagreed with the Arrow evolution profile'

awk '
  /^runtime_version=/ { print "runtime_version=2026.08.24.0"; next }
  /^guardian_/ { next }
  { print }
' \
  "$TEST_ROOT/current-session.state" > "$descriptor.current-missing"
mv "$descriptor.current-missing" "$descriptor"
curl -sS -D "$TEST_ROOT/current-missing.headers" \
  "http://127.0.0.1:$port/api/events.arrow" \
  -o "$TEST_ROOT/current-missing.json"
grep -q '^HTTP/1.1 409 Conflict' "$TEST_ROOT/current-missing.headers" || \
  fail 'current runtime without Guardian journal did not fail closed'
grep -q 'guardianless-generation-runtime-mismatch:descriptor=2026.08.24.0:generation=2026.08.26.28' \
  "$TEST_ROOT/current-missing.json" || \
  fail 'descriptor-downgrade sabotage was refused by an unrelated rule'

awk '/^guardian_journal_file=/ { next } { print }' \
  "$TEST_ROOT/current-session.state" > "$descriptor.current-required"
mv "$descriptor.current-required" "$descriptor"
curl -sS -D "$TEST_ROOT/current-required.headers" \
  "http://127.0.0.1:$port/api/events.arrow" \
  -o "$TEST_ROOT/current-required.json"
grep -q '^HTTP/1.1 409 Conflict' "$TEST_ROOT/current-required.headers" || \
  fail 'current runtime without Guardian journal did not fail closed'
grep -q 'guardian-journal-required:runtime-version=2026.08.26.28' \
  "$TEST_ROOT/current-required.json" || \
  fail 'current-runtime omission was refused by an unrelated rule'

awk -v missing="$TEST_ROOT/missing-guardian.tsv" '
  /^guardian_journal_file=/ { print "guardian_journal_file=" missing; next }
  { print }
' "$TEST_ROOT/current-session.state" > "$descriptor.file-missing"
mv "$descriptor.file-missing" "$descriptor"
curl -sS -D "$TEST_ROOT/file-missing.headers" \
  "http://127.0.0.1:$port/api/events.arrow" \
  -o "$TEST_ROOT/file-missing.json"
grep -q '^HTTP/1.1 409 Conflict' "$TEST_ROOT/file-missing.headers" || \
  fail 'declared but missing Guardian journal did not fail closed'
grep -q "guardian-journal-missing:path=$TEST_ROOT/missing-guardian.tsv" \
  "$TEST_ROOT/file-missing.json" || \
  fail 'missing-file sabotage was refused by an unrelated rule'
cp "$TEST_ROOT/current-session.state" "$descriptor"

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

printf 'SOUNIO_LOOM_ARROW_GATE_PASS=true schema=loom-spectral-events-v1 rows=%s legacy_profile=PASS downgrade=REFUSED current_missing=REFUSED missing_file=REFUSED sabotage=PASS runtime=OCaml+C\n' \
  "$rows"
