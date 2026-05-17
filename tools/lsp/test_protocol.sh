#!/usr/bin/env bash
# tools/lsp/test_protocol.sh — integration smoke for the pure-Sounio LSP.
#
# Spawns `./bin/souc lsp --stdio` (which auto-builds bin/sounio-lsp from
# self-hosted/lsp/server.sio when stale) and round-trips a sequence of
# framed JSON-RPC requests against it. Asserts the response stream.
#
# Exits 0 on PASS, 1 on FAIL. Prints a one-line per-check tally.

set -uo pipefail
# Intentionally NOT `set -e`: each check below uses grep -q whose exit
# status is interpreted as test result, not as a fatal error.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LSP_TEST_LOG_DIR:-/tmp/lsp_proto_test}"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
FAIL_REASONS=()

note_pass() { PASS=$((PASS + 1)); printf '  [pass] %s\n' "$1"; }
note_fail() { FAIL=$((FAIL + 1)); FAIL_REASONS+=("$1"); printf '  [FAIL] %s\n' "$1"; }

# Build one LSP frame envelope on stdout
frame() {
  local body="$1"
  printf 'Content-Length: %d\r\n\r\n%s' "${#body}" "$body"
}

# Run a session that pipes a sequence of framed requests into souc lsp
# and writes the response stream to $1.
run_session() {
  local out="$1"
  shift
  local payloads=()
  while [ $# -gt 0 ]; do
    payloads+=("$(frame "$1")")
    shift
  done
  printf '%s' "${payloads[@]}" \
    | timeout 30 ./bin/souc lsp --stdio >"$out" 2>"$out.err"
}

echo "[lsp-proto] building lsp binary"
./bin/souc compile self-hosted/lsp/server.sio -o "$LOG_DIR/sounio-lsp.bin" \
  >"$LOG_DIR/build.log" 2>&1 || {
  echo "FAIL: build of self-hosted/lsp/server.sio"
  tail -10 "$LOG_DIR/build.log"
  exit 1
}
chmod +x "$LOG_DIR/sounio-lsp.bin"
echo "  built ($(wc -c <"$LOG_DIR/sounio-lsp.bin") bytes)"

# ----- T1: initialize → capabilities envelope ---------------------------
T1_OUT="$LOG_DIR/t1.out"
run_session "$T1_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"hoverProvider":true' "$T1_OUT" && note_pass "T1.hoverProvider" || note_fail "T1.hoverProvider"
grep -q '"definitionProvider":true' "$T1_OUT" && note_pass "T1.definitionProvider" || note_fail "T1.definitionProvider"
grep -q '"referencesProvider":true' "$T1_OUT" && note_pass "T1.referencesProvider" || note_fail "T1.referencesProvider"
grep -qE '"renameProvider":(true|\{"prepareProvider":true\})' "$T1_OUT" && note_pass "T1.renameProvider" || note_fail "T1.renameProvider"
grep -q '"completionProvider"' "$T1_OUT" && note_pass "T1.completionProvider" || note_fail "T1.completionProvider"
grep -q '"sounio-lsp"' "$T1_OUT" && note_pass "T1.serverInfo.name" || note_fail "T1.serverInfo.name"

# ----- T2: didOpen broken source → publishDiagnostics --------------------
T2_OUT="$LOG_DIR/t2.out"
run_session "$T2_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    let x: i64 = \"hello\"\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q 'textDocument/publishDiagnostics' "$T2_OUT" && note_pass "T2.publishDiagnostics emitted" || note_fail "T2.publishDiagnostics emitted"
grep -q '"sounio:type"' "$T2_OUT" && note_pass "T2.source=sounio:type" || note_fail "T2.source=sounio:type"
grep -q 'initializer type does not match declaration' "$T2_OUT" && \
  note_pass "T2.real type-checker diagnostic" || note_fail "T2.real type-checker diagnostic"

# ----- T3: hover on a keyword (static table fast path) ------------------
T3_OUT="$LOG_DIR/t3.out"
run_session "$T3_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 { 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":10,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":0}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":10' "$T3_OUT" && grep -q '"kind":"markdown"' "$T3_OUT" && \
  note_pass "T3.hover markdown" || note_fail "T3.hover markdown"
grep -q '\*\*keyword\*\* `fn`' "$T3_OUT" && \
  note_pass "T3.fn keyword content" || note_fail "T3.fn keyword content"

# ----- T4: completion with `re` prefix → re*, ret*, read_*  -------------
T4_OUT="$LOG_DIR/t4.out"
run_session "$T4_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 { re }\n"}}}' \
  '{"jsonrpc":"2.0","id":20,"method":"textDocument/completion","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":22}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":20' "$T4_OUT" && grep -q '"items":\[' "$T4_OUT" && \
  note_pass "T4.completion items" || note_fail "T4.completion items"
grep -q '"label":"return"' "$T4_OUT" && note_pass "T4.return suggested" || note_fail "T4.return suggested"
grep -q '"label":"read_line"' "$T4_OUT" && note_pass "T4.read_line suggested" || note_fail "T4.read_line suggested"

# ----- T5: in-file definition jump --------------------------------------
T5_OUT="$LOG_DIR/t5.out"
run_session "$T5_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { let a = helper(1); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":30,"method":"textDocument/definition","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":28}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":30' "$T5_OUT" && grep -q '"line":0,"character":3' "$T5_OUT" && \
  note_pass "T5.def jumps to fn helper" || note_fail "T5.def jumps to fn helper"

# ----- T6: references with includeDeclaration ---------------------------
T6_OUT="$LOG_DIR/t6.out"
run_session "$T6_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { helper(1); helper(2); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":40,"method":"textDocument/references","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":21},"context":{"includeDeclaration":true}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# Expect 3 occurrences in the Location array
n_refs=$(grep -oE '"uri":"file:///tmp/t\.sio"' "$T6_OUT" | wc -l)
if [ "$n_refs" -ge 3 ]; then
  note_pass "T6.refs ≥3 (decl + 2 uses)"
else
  note_fail "T6.refs ≥3 (got $n_refs)"
fi

# ----- T7: rename → WorkspaceEdit ---------------------------------------
T7_OUT="$LOG_DIR/t7.out"
run_session "$T7_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { helper(1) }\n"}}}' \
  '{"jsonrpc":"2.0","id":50,"method":"textDocument/rename","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":5},"newName":"do_thing"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":50' "$T7_OUT" && grep -q '"changes":' "$T7_OUT" && grep -q '"newText":"do_thing"' "$T7_OUT" && \
  note_pass "T7.rename WorkspaceEdit" || note_fail "T7.rename WorkspaceEdit"
grep -q '"id":50,"result":null' "$T7_OUT" && note_fail "T7.rename returned null" || true

# ----- T7b: documentSymbol outline --------------------------------------
T7B_OUT="$LOG_DIR/t7b.out"
run_session "$T7B_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"struct Point { x: i64 }\nfn helper() -> i32 { 0 }\nlet GLOBAL: i64 = 0\npub fn shared() -> i32 { 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":55,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"name":"Point","kind":22' "$T7B_OUT" && note_pass "T7b.Point struct" || note_fail "T7b.Point struct"
grep -q '"name":"helper","kind":12' "$T7B_OUT" && note_pass "T7b.helper function" || note_fail "T7b.helper function"
grep -q '"name":"GLOBAL","kind":13' "$T7B_OUT" && note_pass "T7b.GLOBAL variable" || note_fail "T7b.GLOBAL variable"
grep -q '"name":"shared","kind":12' "$T7B_OUT" && note_pass "T7b.shared (pub fn)" || note_fail "T7b.shared (pub fn)"

# ----- T7c: semanticTokens/full -----------------------------------------
T7C_OUT="$LOG_DIR/t7c.out"
run_session "$T7C_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"// hello\nfn helper(x: i64) -> i64 { return 42 }\n"}}}' \
  '{"jsonrpc":"2.0","id":65,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# Capabilities legend should advertise the token-type vocabulary
grep -q '"tokenTypes":\["keyword","type","variable","function","string","number","comment","operator"\]' "$T7C_OUT" \
  && note_pass "T7c.legend" || note_fail "T7c.legend"
# Encoded uint[] data array present + non-empty
grep -q '"id":65,"result":{"data":\[[0-9]' "$T7C_OUT" \
  && note_pass "T7c.data non-empty" || note_fail "T7c.data non-empty"
# `// hello` on line 0 → first 5 ints are 0,0,8,6,0
grep -q '"data":\[0,0,8,6,0,' "$T7C_OUT" \
  && note_pass "T7c.first token = line-0 comment" || note_fail "T7c.first token = line-0 comment"
# `helper` follows `fn` with type=3 (function) → look for ,3,6,3,0
grep -q ',3,6,3,0' "$T7C_OUT" \
  && note_pass "T7c.helper classified as function" || note_fail "T7c.helper classified as function"

# ----- T7d: documentHighlight + prepareRename ---------------------------
T7D_OUT="$LOG_DIR/t7d.out"
run_session "$T7D_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { helper(1); helper(2); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":75,"method":"textDocument/documentHighlight","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":5}}}' \
  '{"jsonrpc":"2.0","id":76,"method":"textDocument/prepareRename","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":5}}}' \
  '{"jsonrpc":"2.0","id":77,"method":"textDocument/prepareRename","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":0}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# documentHighlight should return 3 ranges with kind:1 (text)
n_dh=$(grep -oE '"id":75,"result":\[[^]]*\]' "$T7D_OUT" | grep -oE '"line":[0-9]+' | wc -l)
if [ "$n_dh" -ge 6 ]; then
  note_pass "T7d.documentHighlight ≥3 ranges (got $((n_dh/2)))"
else
  note_fail "T7d.documentHighlight expected ≥3 ranges (got $((n_dh/2)))"
fi
grep -q '"id":75,"result":\[.*"kind":1' "$T7D_OUT" && note_pass "T7d.kind=1 (text)" || note_fail "T7d.kind=1 (text)"

# prepareRename on identifier should return a range
grep -q '"id":76,"result":{"start":' "$T7D_OUT" && note_pass "T7d.prepareRename valid range" || note_fail "T7d.prepareRename valid range"
# prepareRename on `fn` keyword should return null
grep -q '"id":77,"result":null' "$T7D_OUT" && note_pass "T7d.prepareRename rejects keyword" || note_fail "T7d.prepareRename rejects keyword"

# ----- T8: shutdown + exit clean ----------------------------------------
T8_OUT="$LOG_DIR/t8.out"
run_session "$T8_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"shutdown"}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":2,"result":null' "$T8_OUT" && \
  note_pass "T8.shutdown ack" || note_fail "T8.shutdown ack"

echo
echo "[lsp-proto] tally: $PASS pass, $FAIL fail"
if [ "$FAIL" -gt 0 ]; then
  printf '[lsp-proto] failures:\n'
  for r in "${FAIL_REASONS[@]}"; do printf '  - %s\n' "$r"; done
  exit 1
fi
echo "[lsp-proto] LSP_PROTOCOL_PASS"
