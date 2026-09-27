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
    | timeout 30 ${LSP_SERVER_CMD:-./bin/souc lsp --stdio} >"$out" 2>"$out.err"
}

if [ -n "${LSP_SERVER_CMD:-}" ]; then
  echo "[lsp-proto] using external server: $LSP_SERVER_CMD (skipping build)"
else
  echo "[lsp-proto] building lsp binary"
  ./bin/souc compile self-hosted/lsp/server.sio -o "$LOG_DIR/sounio-lsp.bin" \
    >"$LOG_DIR/build.log" 2>&1 || {
    echo "FAIL: build of self-hosted/lsp/server.sio"
    tail -10 "$LOG_DIR/build.log"
    exit 1
  }
  chmod +x "$LOG_DIR/sounio-lsp.bin"
  echo "  built ($(wc -c <"$LOG_DIR/sounio-lsp.bin") bytes)"
fi

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

# ----- T7e: signatureHelp ------------------------------------------------
T7E_OUT="$LOG_DIR/t7e.out"
# Doc:
#   line 0: fn add(a: i64, b: i64) -> i64 { a + b }
#   line 1: fn main() -> i32 { add(10, 20); 0 }
# Cursor positions exercise activeParameter detection.
run_session "$T7E_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i32 { add(10, 20); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":80,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":23}}}' \
  '{"jsonrpc":"2.0","id":81,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":28}}}' \
  '{"jsonrpc":"2.0","id":82,"method":"textDocument/signatureHelp","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":2}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# id 80: cursor inside add(10|, 20) → activeParameter == 0
grep -q '"id":80,"result":{"signatures":\[{"label":"fn add(a: i64, b: i64)' "$T7E_OUT" \
  && note_pass "T7e.signatureHelp label" || note_fail "T7e.signatureHelp label"
grep -q '"id":80,.*"activeParameter":0}],' "$T7E_OUT" \
  && note_pass "T7e.activeParameter=0" || note_fail "T7e.activeParameter=0"
# id 81: cursor after the comma → activeParameter == 1
grep -q '"id":81,.*"activeParameter":1}],' "$T7E_OUT" \
  && note_pass "T7e.activeParameter=1 after comma" || note_fail "T7e.activeParameter=1 after comma"
# id 81: parameter ranges advertised (two `{"label":[N,M]}` entries)
n_par=$(grep -oE '"id":81,"result":\{.*"activeParameter":1\}\],' "$T7E_OUT" \
        | grep -oE '"label":\[[0-9]+,[0-9]+\]' | wc -l)
if [ "$n_par" -eq 2 ]; then
  note_pass "T7e.two parameter ranges"
else
  note_fail "T7e.two parameter ranges (got $n_par)"
fi
# id 82: outside any call → empty signatures
grep -q '"id":82,"result":{"signatures":\[\]' "$T7E_OUT" \
  && note_pass "T7e.outside call → empty" || note_fail "T7e.outside call → empty"

# ----- T7f: foldingRange -------------------------------------------------
T7F_OUT="$LOG_DIR/t7f.out"
# Doc spans 6 lines:
#   0: fn helper(x: i64) -> i64 {
#   1:     if x > 0 {
#   2:         x + 1
#   3:     } else {
#   4:         0
#   5:     }
#   6: }
# Expect at least 3 folding ranges (outer fn body + 2 if/else arms).
run_session "$T7F_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 {\n    if x > 0 {\n        x + 1\n    } else {\n        0\n    }\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":90,"method":"textDocument/foldingRange","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

n_fold=$(grep -oE '"id":90,"result":\[[^]]*\]' "$T7F_OUT" | grep -oE '"startLine":[0-9]+' | wc -l)
if [ "$n_fold" -ge 3 ]; then
  note_pass "T7f.foldingRange ≥3 regions (got $n_fold)"
else
  note_fail "T7f.foldingRange expected ≥3 (got $n_fold)"
fi
grep -q '"id":90,"result":\[.*"kind":"region"' "$T7F_OUT" \
  && note_pass "T7f.kind=region" || note_fail "T7f.kind=region"
# Outer fn body must span lines 0..6
grep -q '"startLine":0,"endLine":6' "$T7F_OUT" \
  && note_pass "T7f.outer fn fold 0..6" || note_fail "T7f.outer fn fold 0..6"

# ----- T7g: foldingRange comment + string awareness ----------------------
T7G_OUT="$LOG_DIR/t7g.out"
# 4 consecutive // lines (0..3) then a code block. Brace inside a string
# must NOT open a fold.
run_session "$T7G_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"// header line one\n// header line two\n// header line three\n// header line four\nfn f() -> i32 {\n    let s = \"open { brace\"\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":91,"method":"textDocument/foldingRange","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"startLine":0,"endLine":3,"kind":"comment"' "$T7G_OUT" \
  && note_pass "T7g.comment run 0..3" || note_fail "T7g.comment run 0..3"
# Exactly one region fold (fn body) — the `{` in the string must not open a second.
n_region=$(grep -oE '"id":91,"result":\[[^]]*\]' "$T7G_OUT" | grep -oE '"kind":"region"' | wc -l)
if [ "$n_region" -eq 1 ]; then
  note_pass "T7g.string brace ignored (1 region)"
else
  note_fail "T7g.string brace ignored expected 1 region (got $n_region)"
fi

# ----- T7h: workspace/symbol --------------------------------------------
T7H_OUT="$LOG_DIR/t7h.out"
# Two open documents. Workspace query should hit symbols across both.
run_session "$T7H_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/a.sio","languageId":"sounio","version":1,"text":"fn alpha() -> i32 { 0 }\nstruct AlphaCfg { x: i64 }\n"}}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/b.sio","languageId":"sounio","version":1,"text":"fn beta() -> i32 { 0 }\nfn alphabet() -> i32 { 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":100,"method":"workspace/symbol","params":{"query":"alpha"}}' \
  '{"jsonrpc":"2.0","id":101,"method":"workspace/symbol","params":{"query":"beta"}}' \
  '{"jsonrpc":"2.0","id":102,"method":"workspace/symbol","params":{"query":"ZZZNOMATCH"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# id 100: should hit fn alpha + struct AlphaCfg + fn alphabet (3 results)
n_alpha=$(grep -oE '"id":100,"result":\[.*\]\}' "$T7H_OUT" | grep -oE '"name":"[A-Za-z]+"' | wc -l)
if [ "$n_alpha" -ge 3 ]; then
  note_pass "T7h.alpha matches ≥3 (got $n_alpha)"
else
  note_fail "T7h.alpha matches expected ≥3 (got $n_alpha)"
fi
# Cross-file: result for query alpha must include URIs for both a.sio and b.sio
grep -q '"id":100,.*file:///tmp/a.sio' "$T7H_OUT" \
  && note_pass "T7h.alpha cross-file a.sio" || note_fail "T7h.alpha cross-file a.sio"
grep -q '"id":100,.*file:///tmp/b.sio' "$T7H_OUT" \
  && note_pass "T7h.alpha cross-file b.sio" || note_fail "T7h.alpha cross-file b.sio"
# id 101: should hit fn beta exactly (case-insensitive substring; alphabet has no "beta")
grep -q '"id":101,"result":\[{"name":"beta"' "$T7H_OUT" \
  && note_pass "T7h.beta exact match" || note_fail "T7h.beta exact match"
# id 102: no match → empty array
grep -q '"id":102,"result":\[\]' "$T7H_OUT" \
  && note_pass "T7h.no match empty" || note_fail "T7h.no match empty"

# ----- T7i: selectionRange ----------------------------------------------
T7I_OUT="$LOG_DIR/t7i.out"
# Doc: fn add(a: i64, b: i64) -> i64 { a + b }
# Position at character 7 = `a` (the param identifier).
# Expected chain: `a` → param list inside `()` → `(a: i64, b: i64)` →
#                  fn signature region → whole file
run_session "$T7I_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn add(a: i64, b: i64) -> i64 { a + b }\n"}}}' \
  '{"jsonrpc":"2.0","id":110,"method":"textDocument/selectionRange","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"positions":[{"line":0,"character":7}]}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# Must return at least one SelectionRange with nested parent chain.
grep -q '"id":110,"result":\[{"range"' "$T7I_OUT" \
  && note_pass "T7i.selectionRange shape" || note_fail "T7i.selectionRange shape"
# Innermost should be the identifier `a` at char 7..8.
grep -q '"start":{"line":0,"character":7},"end":{"line":0,"character":8}' "$T7I_OUT" \
  && note_pass "T7i.innermost = ident a" || note_fail "T7i.innermost = ident a"
# Chain must include a `"parent":` link.
grep -q '"id":110,.*"parent":{"range"' "$T7I_OUT" \
  && note_pass "T7i.parent chain present" || note_fail "T7i.parent chain present"

# ----- T7j: inlayHint ---------------------------------------------------
T7J_OUT="$LOG_DIR/t7j.out"
# Doc:
#   line 0: fn add(a: i64, b: i64) -> i64 { a + b }
#   line 1: fn main() -> i32 { add(10, 20); 0 }
# Expect inlay hints "a:" before 10 and "b:" before 20.
run_session "$T7J_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn add(a: i64, b: i64) -> i64 { a + b }\nfn main() -> i32 { add(10, 20); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":120,"method":"textDocument/inlayHint","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":0,"character":0},"end":{"line":2,"character":0}}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"label":"a:","kind":2' "$T7J_OUT" \
  && note_pass "T7j.hint a: kind 2" || note_fail "T7j.hint a: kind 2"
grep -q '"label":"b:","kind":2' "$T7J_OUT" \
  && note_pass "T7j.hint b: kind 2" || note_fail "T7j.hint b: kind 2"
# Both hints must land on line 1 (call site)
n_l1=$(grep -oE '"position":\{"line":1,"character":[0-9]+\}' "$T7J_OUT" | wc -l)
if [ "$n_l1" -ge 2 ]; then
  note_pass "T7j.both hints on line 1 (got $n_l1)"
else
  note_fail "T7j.both hints on line 1 (got $n_l1)"
fi
# paddingRight true
grep -q '"paddingRight":true' "$T7J_OUT" \
  && note_pass "T7j.paddingRight" || note_fail "T7j.paddingRight"

# ----- T7k: declaration / typeDefinition / implementation ---------------
T7K_OUT="$LOG_DIR/t7k.out"
run_session "$T7K_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { helper(1); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":130,"method":"textDocument/declaration","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":20}}}' \
  '{"jsonrpc":"2.0","id":131,"method":"textDocument/typeDefinition","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":20}}}' \
  '{"jsonrpc":"2.0","id":132,"method":"textDocument/implementation","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":1,"character":20}}}' \
  '{"jsonrpc":"2.0","id":133,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# All three should return a Location pointing at the fn helper decl
grep -q '"id":130,"result":{"uri":"file:///tmp/t.sio"' "$T7K_OUT" \
  && note_pass "T7k.declaration → helper" || note_fail "T7k.declaration → helper"
grep -q '"id":131,"result":{"uri":"file:///tmp/t.sio"' "$T7K_OUT" \
  && note_pass "T7k.typeDefinition → helper" || note_fail "T7k.typeDefinition → helper"
grep -q '"id":132,"result":{"uri":"file:///tmp/t.sio"' "$T7K_OUT" \
  && note_pass "T7k.implementation → helper" || note_fail "T7k.implementation → helper"
# initialize must now advertise the three new capabilities.
grep -q '"declarationProvider":true' "$T7K_OUT" \
  && note_pass "T7k.cap.declaration" || note_fail "T7k.cap.declaration"
grep -q '"typeDefinitionProvider":true' "$T7K_OUT" \
  && note_pass "T7k.cap.typeDefinition" || note_fail "T7k.cap.typeDefinition"
grep -q '"implementationProvider":true' "$T7K_OUT" \
  && note_pass "T7k.cap.implementation" || note_fail "T7k.cap.implementation"

# ----- T7l: codeAction (loop → while true) ------------------------------
T7L_OUT="$LOG_DIR/t7l.out"
# Doc with a `loop {`. Request codeAction over the whole file.
run_session "$T7L_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    loop {\n        break\n    }\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":140,"method":"textDocument/codeAction","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":0,"character":0},"end":{"line":5,"character":0}},"context":{"diagnostics":[]}}}' \
  '{"jsonrpc":"2.0","id":141,"method":"textDocument/codeAction","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":3,"character":0},"end":{"line":4,"character":0}},"context":{"diagnostics":[]}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# id 140: whole-file range → one quickfix for the `loop` on line 1
grep -q '"id":140,"result":\[{"title":"Replace `loop` with `while true`"' "$T7L_OUT" \
  && note_pass "T7l.title" || note_fail "T7l.title"
grep -q '"kind":"quickfix"' "$T7L_OUT" \
  && note_pass "T7l.kind=quickfix" || note_fail "T7l.kind=quickfix"
grep -q '"newText":"while true"' "$T7L_OUT" \
  && note_pass "T7l.newText=while true" || note_fail "T7l.newText=while true"
grep -q '"start":{"line":1,"character":4},"end":{"line":1,"character":8}' "$T7L_OUT" \
  && note_pass "T7l.edit range = loop keyword" || note_fail "T7l.edit range = loop keyword"
# id 141: range that excludes the `loop` → empty result
grep -q '"id":141,"result":\[\]' "$T7L_OUT" \
  && note_pass "T7l.range outside loop → empty" || note_fail "T7l.range outside loop → empty"

# ----- T7m: codeAction `let mut` → `var` --------------------------------
T7M_OUT="$LOG_DIR/t7m.out"
# Doc with `let mut x = 1`. Request codeAction over the whole file.
run_session "$T7M_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    let mut x = 1\n    x\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":150,"method":"textDocument/codeAction","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":0,"character":0},"end":{"line":3,"character":0}},"context":{"diagnostics":[]}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"title":"Replace `let mut` with `var`"' "$T7M_OUT" \
  && note_pass "T7m.title" || note_fail "T7m.title"
grep -q '"newText":"var"' "$T7M_OUT" \
  && note_pass "T7m.newText=var" || note_fail "T7m.newText=var"
# Edit range: `let mut` on line 1, char 4..11
grep -q '"start":{"line":1,"character":4},"end":{"line":1,"character":11}' "$T7M_OUT" \
  && note_pass "T7m.edit range = let-mut span" || note_fail "T7m.edit range = let-mut span"

# ----- T7n: call hierarchy ----------------------------------------------
T7N_OUT="$LOG_DIR/t7n.out"
# Doc:
#   line 0: fn helper(x: i64) -> i64 { x + 1 }
#   line 1: fn main() -> i32 { helper(1); helper(2); 0 }
#   line 2: fn other() -> i32 { helper(3); 0 }
# Cursor on `helper` decl @ line 0, char 5.
run_session "$T7N_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { helper(1); helper(2); 0 }\nfn other() -> i32 { helper(3); 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":160,"method":"textDocument/prepareCallHierarchy","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":0,"character":5}}}' \
  '{"jsonrpc":"2.0","id":161,"method":"callHierarchy/incomingCalls","params":{"item":{"name":"helper","kind":12,"uri":"file:///tmp/t.sio","range":{"start":{"line":0,"character":3},"end":{"line":0,"character":9}},"selectionRange":{"start":{"line":0,"character":3},"end":{"line":0,"character":9}}}}}' \
  '{"jsonrpc":"2.0","id":162,"method":"callHierarchy/outgoingCalls","params":{"item":{"name":"main","kind":12,"uri":"file:///tmp/t.sio","range":{"start":{"line":1,"character":3},"end":{"line":1,"character":7}},"selectionRange":{"start":{"line":1,"character":3},"end":{"line":1,"character":7}}}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

# prepareCallHierarchy: returns [{name:"helper",kind:12,uri:...}]
grep -q '"id":160,"result":\[{"name":"helper","kind":12' "$T7N_OUT" \
  && note_pass "T7n.prepare returns helper item" || note_fail "T7n.prepare returns helper item"
# incomingCalls: should find callers main + other (2 results)
# Count unique caller names — v0 emits one IncomingCall per call site
# (not per caller fn), so 2 calls in main + 1 in other = 3 entries but
# the unique caller set is {main, other}.
n_uniq=$(grep -oE '"id":161,"result":\[.*\]\}' "$T7N_OUT" | grep -oE '"from":\{"name":"[a-z]+"' | sort -u | wc -l)
if [ "$n_uniq" -eq 2 ]; then
  note_pass "T7n.incoming = 2 unique callers"
else
  note_fail "T7n.incoming expected 2 unique callers (got $n_uniq)"
fi
grep -q '"from":{"name":"main"' "$T7N_OUT" \
  && note_pass "T7n.incoming.main" || note_fail "T7n.incoming.main"
grep -q '"from":{"name":"other"' "$T7N_OUT" \
  && note_pass "T7n.incoming.other" || note_fail "T7n.incoming.other"
# outgoingCalls for main: should find helper called twice (2 entries OR 1 entry with 2 fromRanges)
grep -q '"id":162,.*"to":{"name":"helper"' "$T7N_OUT" \
  && note_pass "T7n.outgoing.helper" || note_fail "T7n.outgoing.helper"

# ----- T7o: documentLink ------------------------------------------------
T7O_OUT="$LOG_DIR/t7o.out"
run_session "$T7O_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///workspace/sounio/self-hosted/lsp/x.sio","languageId":"sounio","version":1,"text":"use compiler::lean_single\nuse check::check::Checker\nfn main() -> i32 { 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":600,"method":"textDocument/documentLink","params":{"textDocument":{"uri":"file:///workspace/sounio/self-hosted/lsp/x.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"target":"file:///workspace/sounio/self-hosted/compiler/lean_single.sio"' "$T7O_OUT" \
  && note_pass "T7o.documentLink → lean_single.sio" || note_fail "T7o.documentLink → lean_single.sio"
grep -q '"target":"file:///workspace/sounio/self-hosted/check/check/Checker.sio"' "$T7O_OUT" \
  && note_pass "T7o.documentLink → Checker.sio" || note_fail "T7o.documentLink → Checker.sio"

# ----- T7p: onTypeFormatting (newline auto-indent) ----------------------
T7P_OUT="$LOG_DIR/t7p.out"
run_session "$T7P_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    if true {\n\n    }\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":700,"method":"textDocument/onTypeFormatting","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":2,"character":0},"ch":"\n","options":{"tabSize":4,"insertSpaces":true}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":700,"result":\[{"range":{"start":{"line":2,"character":0},"end":{"line":2,"character":0}},"newText":"        "}\]' "$T7P_OUT" \
  && note_pass "T7p.onType \\n → 8 spaces (depth 2)" || note_fail "T7p.onType \\n → 8 spaces (depth 2)"

# ----- T7q: formatting (whitespace trim) --------------------------------
T7Q_OUT="$LOG_DIR/t7q.out"
run_session "$T7Q_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"a   \nb\nc\t \n"}}}' \
  '{"jsonrpc":"2.0","id":800,"method":"textDocument/formatting","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"options":{"tabSize":4,"insertSpaces":true}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"start":{"line":0,"character":1},"end":{"line":0,"character":4}' "$T7Q_OUT" \
  && note_pass "T7q.fmt trims line 0" || note_fail "T7q.fmt trims line 0"
grep -q '"start":{"line":2,"character":1},"end":{"line":2,"character":3}' "$T7Q_OUT" \
  && note_pass "T7q.fmt trims line 2" || note_fail "T7q.fmt trims line 2"
n_fmt_edits=$(grep -oE '"id":800,"result":\[[^]]*\]' "$T7Q_OUT" | grep -oE '"newText":""' | wc -l)
if [ "$n_fmt_edits" -eq 2 ]; then
  note_pass "T7q.fmt exactly 2 edits"
else
  note_fail "T7q.fmt expected 2 edits (got $n_fmt_edits)"
fi

# ----- T7r: rangeFormatting --------------------------------------------
T7R_OUT="$LOG_DIR/t7r.out"
run_session "$T7R_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"a   \nb \nc\t \n"}}}' \
  '{"jsonrpc":"2.0","id":900,"method":"textDocument/rangeFormatting","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":1,"character":0},"end":{"line":2,"character":0}},"options":{"tabSize":4,"insertSpaces":true}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"start":{"line":1,"character":1},"end":{"line":1,"character":2}' "$T7R_OUT" \
  && note_pass "T7r.range trims line 1" || note_fail "T7r.range trims line 1"
grep -q '"id":900,"result":\[[^]]*"line":0' "$T7R_OUT" \
  && note_fail "T7r.range leaks to line 0" || note_pass "T7r.range leaves line 0"
grep -q '"id":900,"result":\[[^]]*"line":2' "$T7R_OUT" \
  && note_fail "T7r.range leaks to line 2" || note_pass "T7r.range leaves line 2"

# ----- T7s: codeAction trailing-; quickfix ------------------------------
T7S_OUT="$LOG_DIR/t7s.out"
run_session "$T7S_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    let x = 1;\n    let s = \";\"\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","id":1000,"method":"textDocument/codeAction","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"range":{"start":{"line":0,"character":0},"end":{"line":5,"character":0}},"context":{"diagnostics":[]}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"title":"Remove trailing `;` (Sounio uses newlines)"' "$T7S_OUT" \
  && note_pass "T7s.title" || note_fail "T7s.title"
grep -q '"start":{"line":1,"character":13},"end":{"line":1,"character":14}' "$T7S_OUT" \
  && note_pass "T7s.edit at stray ;" || note_fail "T7s.edit at stray ;"
grep -q '"start":{"line":2,"character":13},"end":{"line":2,"character":14}' "$T7S_OUT" \
  && note_fail "T7s.in-string ; not skipped" || note_pass "T7s.in-string ; skipped"

# ----- T7t: codeLens + $/cancelRequest no-op + watchedFiles no-op -------
T7T_OUT="$LOG_DIR/t7t.out"
run_session "$T7T_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"pub fn helper(x: i64) -> i64 { x + 1 }\nfn main() -> i32 { 0 }\n"}}}' \
  '{"jsonrpc":"2.0","id":2000,"method":"textDocument/codeLens","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","id":2001,"method":"$/cancelRequest","params":{"id":2000}}' \
  '{"jsonrpc":"2.0","method":"workspace/didChangeWatchedFiles","params":{"changes":[]}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"title":"✓ Check","command":"sounio.check"' "$T7T_OUT" \
  && note_pass "T7t.codeLens ✓ Check" || note_fail "T7t.codeLens ✓ Check"
grep -q '"range":{"start":{"line":1,"character":0},"end":{"line":1,"character":0}},"command":{"title":"▶ Run","command":"sounio.run"' "$T7T_OUT" \
  && note_pass "T7t.codeLens ▶ Run main" || note_fail "T7t.codeLens ▶ Run main"
grep -q '"codeLensProvider"' "$T7T_OUT" \
  && note_pass "T7t.cap.codeLens" || note_fail "T7t.cap.codeLens"
grep -q '"workspaceFolders"' "$T7T_OUT" \
  && note_pass "T7t.cap.workspaceFolders" || note_fail "T7t.cap.workspaceFolders"
grep -q '"id":2001' "$T7T_OUT" \
  && note_fail "T7t.cancelRequest leaked a response" || note_pass "T7t.cancelRequest silent"

# ----- T8: shutdown + exit clean ----------------------------------------
T8_OUT="$LOG_DIR/t8.out"
run_session "$T8_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","id":2,"method":"shutdown"}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":2,"result":null' "$T8_OUT" && \
  note_pass "T8.shutdown ack" || note_fail "T8.shutdown ack"

# ----- T9: Sprint-2 #5 — semanticTokens/full resultId + full/delta ------
# /full must now carry a resultId and advertise delta support.
T9A_OUT="$LOG_DIR/t9a.out"
run_session "$T9A_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"// hello\nfn helper(x: i64) -> i64 { return 42 }\n"}}}' \
  '{"jsonrpc":"2.0","id":65,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"full":{"delta":true}' "$T9A_OUT" && \
  note_pass "T9a.cap.full.delta" || note_fail "T9a.cap.full.delta"
grep -q '"id":65,"result":{"data":\[0,0,8,6,0,' "$T9A_OUT" && \
  note_pass "T9a.full data unchanged" || note_fail "T9a.full data unchanged"
grep -q '\],"resultId":"st1"}' "$T9A_OUT" && \
  note_pass "T9a.full resultId st1" || note_fail "T9a.full resultId st1"

# Delta with the matching previousResultId and no edit → empty edits.
T9B_OUT="$LOG_DIR/t9b.out"
run_session "$T9B_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"// hello\nfn helper(x: i64) -> i64 { return 42 }\n"}}}' \
  '{"jsonrpc":"2.0","id":65,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","id":66,"method":"textDocument/semanticTokens/full/delta","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"previousResultId":"st1"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":66,"result":{"resultId":"st2","edits":\[\]}' "$T9B_OUT" && \
  note_pass "T9b.no-change delta → edits:[]" || note_fail "T9b.no-change delta → edits:[]"

# Delta after a trailing-line append → one edit with deleteCount 0 and
# the new tokens as data.
T9C_OUT="$LOG_DIR/t9c.out"
run_session "$T9C_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { return 42 }\n"}}}' \
  '{"jsonrpc":"2.0","id":65,"method":"textDocument/semanticTokens/full","params":{"textDocument":{"uri":"file:///tmp/t.sio"}}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///tmp/t.sio","version":2},"contentChanges":[{"text":"fn helper(x: i64) -> i64 { return 42 }\n// tail\n"}]}}' \
  '{"jsonrpc":"2.0","id":66,"method":"textDocument/semanticTokens/full/delta","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"previousResultId":"st1"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":66,"result":{"resultId":"st2","edits":\[{"start":[0-9]*,"deleteCount":0,"data":\[1,0,7,6,0\]}' "$T9C_OUT" && \
  note_pass "T9c.append delta → single insert edit" || note_fail "T9c.append delta → single insert edit"

# Delta with an unknown previousResultId → full-result fallback.
T9D_OUT="$LOG_DIR/t9d.out"
run_session "$T9D_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn helper(x: i64) -> i64 { return 42 }\n"}}}' \
  '{"jsonrpc":"2.0","id":66,"method":"textDocument/semanticTokens/full/delta","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"previousResultId":"st99"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":66,"result":{"data":\[[0-9]' "$T9D_OUT" && \
  note_pass "T9d.stale resultId → full fallback" || note_fail "T9d.stale resultId → full fallback"

# ----- T10: Sprint-2 #7 — workspace/symbol cross-file scan ---------------
# Fixture: a workspace root with two *.sio files on disk, neither opened
# via didOpen. initialize carries rootUri; workspace/symbol must find
# symbols from both files.
WS_TEST_DIR="$LOG_DIR/ws_test"
rm -rf "$WS_TEST_DIR"
mkdir -p "$WS_TEST_DIR"
printf 'fn alpha_fn(x: i64) -> i64 { x }\n' > "$WS_TEST_DIR/a.sio"
printf 'struct BetaPoint {\n    x: i64\n}\n' > "$WS_TEST_DIR/b.sio"
printf 'not sounio, must be ignored\n' > "$WS_TEST_DIR/c.txt"

T10A_OUT="$LOG_DIR/t10a.out"
run_session "$T10A_OUT" \
  "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"rootUri\":\"file://$WS_TEST_DIR\"}}" \
  '{"jsonrpc":"2.0","id":70,"method":"workspace/symbol","params":{"query":""}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":70,"result":\[' "$T10A_OUT" && \
  note_pass "T10a.symbol result array" || note_fail "T10a.symbol result array"
grep -q '"name":"alpha_fn"' "$T10A_OUT" && \
  note_pass "T10a.finds a.sio alpha_fn" || note_fail "T10a.finds a.sio alpha_fn"
grep -q '"name":"BetaPoint"' "$T10A_OUT" && \
  note_pass "T10a.finds b.sio BetaPoint" || note_fail "T10a.finds b.sio BetaPoint"
grep -q "\"uri\":\"file://$WS_TEST_DIR/a.sio\"" "$T10A_OUT" && \
  note_pass "T10a.uri has file:// prefix" || note_fail "T10a.uri has file:// prefix"

# Filtered query: "alph" matches alpha_fn but not BetaPoint.
T10B_OUT="$LOG_DIR/t10b.out"
run_session "$T10B_OUT" \
  "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"rootUri\":\"file://$WS_TEST_DIR\"}}" \
  '{"jsonrpc":"2.0","id":71,"method":"workspace/symbol","params":{"query":"alph"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"name":"alpha_fn"' "$T10B_OUT" && \
  note_pass "T10b.query filter hit" || note_fail "T10b.query filter hit"
grep -q '"name":"BetaPoint"' "$T10B_OUT" && \
  note_fail "T10b.query filter leaks BetaPoint" || note_pass "T10b.query filter excludes BetaPoint"

# An open slot must not be duplicated by the on-disk scan: didOpen a.sio,
# then query "alpha" → exactly one alpha_fn hit.
T10C_OUT="$LOG_DIR/t10c.out"
run_session "$T10C_OUT" \
  "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"rootUri\":\"file://$WS_TEST_DIR\"}}" \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file://$WS_TEST_DIR/a.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"fn alpha_fn(x: i64) -> i64 { x }\\n\"}}}" \
  '{"jsonrpc":"2.0","id":72,"method":"workspace/symbol","params":{"query":"alpha"}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

[ "$(grep -o '"name":"alpha_fn"' "$T10C_OUT" | wc -l)" -eq 1 ] && \
  note_pass "T10c.open slot deduped vs disk" || note_fail "T10c.open slot deduped vs disk"

# ----- T11: Sprint-2 #8 — incremental textDocumentSync (kind=2) ----------
T11A_OUT="$LOG_DIR/t11a.out"
run_session "$T11A_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"textDocumentSync":2' "$T11A_OUT" && \
  note_pass "T11a.cap sync kind=2" || note_fail "T11a.cap sync kind=2"

# One ranged edit: replace the `1` initializer with "hi" → the stored
# document must update and the diagnostic pipeline must flag it.
T11B_OUT="$LOG_DIR/t11b.out"
run_session "$T11B_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"fn main() -> i32 {\n    let x: i64 = 1\n    0\n}\n"}}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///tmp/t.sio","version":2},"contentChanges":[{"range":{"start":{"line":1,"character":17},"end":{"line":1,"character":18}},"text":"\"hi\""}]}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q 'Type mismatch' "$T11B_OUT" && \
  note_pass "T11b.ranged edit re-diagnosed" || note_fail "T11b.ranged edit re-diagnosed"

# Two ranged edits in one notification, applied in order: aaa→ccc and
# bbb→ddd. workspace/symbol must see the post-edit names only.
T11C_OUT="$LOG_DIR/t11c.out"
run_session "$T11C_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"let aaa = 1\nlet bbb = 2\n"}}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///tmp/t.sio","version":2},"contentChanges":[{"range":{"start":{"line":0,"character":4},"end":{"line":0,"character":7}},"text":"ccc"},{"range":{"start":{"line":1,"character":4},"end":{"line":1,"character":7}},"text":"ddd"}]}}' \
  '{"jsonrpc":"2.0","id":73,"method":"workspace/symbol","params":{"query":""}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"name":"ccc"' "$T11C_OUT" && \
  note_pass "T11c.first edit applied" || note_fail "T11c.first edit applied"
grep -q '"name":"ddd"' "$T11C_OUT" && \
  note_pass "T11c.second edit applied in order" || note_fail "T11c.second edit applied in order"
grep -q '"name":"aaa"' "$T11C_OUT" && \
  note_fail "T11c.stale aaa leaks" || note_pass "T11c.no stale aaa"
grep -q '"name":"bbb"' "$T11C_OUT" && \
  note_fail "T11c.stale bbb leaks" || note_pass "T11c.no stale bbb"

# A full-replacement change object (no "range") still works under kind=2.
T11D_OUT="$LOG_DIR/t11d.out"
run_session "$T11D_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/t.sio","languageId":"sounio","version":1,"text":"let aaa = 1\n"}}}' \
  '{"jsonrpc":"2.0","method":"textDocument/didChange","params":{"textDocument":{"uri":"file:///tmp/t.sio","version":2},"contentChanges":[{"text":"let zzz = 3\n"}]}}' \
  '{"jsonrpc":"2.0","id":74,"method":"workspace/symbol","params":{"query":""}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"name":"zzz"' "$T11D_OUT" && \
  note_pass "T11d.full-replacement change works" || note_fail "T11d.full-replacement change works"
grep -q '"name":"aaa"' "$T11D_OUT" && \
  note_fail "T11d.stale aaa leaks" || note_pass "T11d.no stale aaa"

# ----- T12: Sprint-2 #6 — typeHierarchy (prepare + super/sub) ------------
# Fixture doc declares trait Greet, struct Point, and `impl Greet for
# Point`. Hierarchy semantics: supertypes(S) = traits implemented by S,
# subtypes(T) = implementors of trait T (single-document scope).
T12_DOC='trait Greet {\n    fn greet(self) -> i64\n}\nstruct Point {\n    x: i64\n}\nimpl Greet for Point {\n    fn greet(self) -> i64 { 1 }\n}\n'

T12A_OUT="$LOG_DIR/t12a.out"
run_session "$T12A_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"typeHierarchyProvider":true' "$T12A_OUT" && \
  note_pass "T12a.cap.typeHierarchy" || note_fail "T12a.cap.typeHierarchy"

# prepare on `Point` (line 3, char 8) → one item, kind 23 (Struct).
T12B_OUT="$LOG_DIR/t12b.out"
run_session "$T12B_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file:///tmp/t.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"$T12_DOC\"}}}" \
  '{"jsonrpc":"2.0","id":80,"method":"textDocument/prepareTypeHierarchy","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":3,"character":8}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":80,"result":\[{"name":"Point","kind":23' "$T12B_OUT" && \
  note_pass "T12b.prepare struct item" || note_fail "T12b.prepare struct item"
grep -q '"uri":"file:///tmp/t.sio"' "$T12B_OUT" && \
  note_pass "T12b.item carries uri" || note_fail "T12b.item carries uri"

# prepare on a non-type identifier (`x`, line 4 char 4) → result null.
T12C_OUT="$LOG_DIR/t12c.out"
run_session "$T12C_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file:///tmp/t.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"$T12_DOC\"}}}" \
  '{"jsonrpc":"2.0","id":81,"method":"textDocument/prepareTypeHierarchy","params":{"textDocument":{"uri":"file:///tmp/t.sio"},"position":{"line":4,"character":4}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":81,"result":null' "$T12C_OUT" && \
  note_pass "T12c.prepare non-type → null" || note_fail "T12c.prepare non-type → null"

# supertypes of Point → trait Greet (kind 11 = Interface).
T12D_OUT="$LOG_DIR/t12d.out"
run_session "$T12D_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file:///tmp/t.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"$T12_DOC\"}}}" \
  '{"jsonrpc":"2.0","id":82,"method":"typeHierarchy/supertypes","params":{"item":{"name":"Point","kind":23,"uri":"file:///tmp/t.sio","range":{"start":{"line":3,"character":0},"end":{"line":5,"character":1}},"selectionRange":{"start":{"line":3,"character":7},"end":{"line":3,"character":12}}}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":82,"result":\[{"name":"Greet","kind":11' "$T12D_OUT" && \
  note_pass "T12d.supertypes → trait Greet" || note_fail "T12d.supertypes → trait Greet"

# subtypes of Greet → implementor Point (kind 23).
T12E_OUT="$LOG_DIR/t12e.out"
run_session "$T12E_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file:///tmp/t.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"$T12_DOC\"}}}" \
  '{"jsonrpc":"2.0","id":83,"method":"typeHierarchy/subtypes","params":{"item":{"name":"Greet","kind":11,"uri":"file:///tmp/t.sio","range":{"start":{"line":0,"character":0},"end":{"line":2,"character":1}},"selectionRange":{"start":{"line":0,"character":6},"end":{"line":0,"character":11}}}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":83,"result":\[{"name":"Point","kind":23' "$T12E_OUT" && \
  note_pass "T12e.subtypes → implementor Point" || note_fail "T12e.subtypes → implementor Point"

# subtypes of Point → [] (no struct inheritance in Sounio).
T12F_OUT="$LOG_DIR/t12f.out"
run_session "$T12F_OUT" \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"file:///tmp/t.sio\",\"languageId\":\"sounio\",\"version\":1,\"text\":\"$T12_DOC\"}}}" \
  '{"jsonrpc":"2.0","id":84,"method":"typeHierarchy/subtypes","params":{"item":{"name":"Point","kind":23,"uri":"file:///tmp/t.sio","range":{"start":{"line":3,"character":0},"end":{"line":5,"character":1}},"selectionRange":{"start":{"line":3,"character":7},"end":{"line":3,"character":12}}}}}' \
  '{"jsonrpc":"2.0","method":"exit"}'

grep -q '"id":84,"result":\[\]' "$T12F_OUT" && \
  note_pass "T12f.subtypes of struct → []" || note_fail "T12f.subtypes of struct → []"

echo
echo "[lsp-proto] tally: $PASS pass, $FAIL fail"
if [ "$FAIL" -gt 0 ]; then
  printf '[lsp-proto] failures:\n'
  for r in "${FAIL_REASONS[@]}"; do printf '  - %s\n' "$r"; done
  exit 1
fi
echo "[lsp-proto] LSP_PROTOCOL_PASS"
