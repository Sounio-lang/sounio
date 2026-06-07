#!/usr/bin/env bash
# Self-test for diag_bridge.sh: feed source files with errors at known lines
# and assert the emitted LSP diagnostic carries the correct 0-based line.
#
# Requires a souc binary. Defaults to ./bin/souc-linux-x86_64 (the mini_native
# build the in-tree LSP server execs); override with SOUC_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUC_BIN:-$ROOT/bin/souc-linux-x86_64}"
BRIDGE="$ROOT/tools/lsp/diag_bridge.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fails=0

check() {
    local name="$1" file="$2" want_line="$3"
    local out; out="$("$BRIDGE" "$file" "$SOUC")"
    if printf '%s' "$out" | grep -q "\"line\":$want_line"; then
        echo "[PASS] $name (line=$want_line)"
    else
        echo "[FAIL] $name: expected line=$want_line, got: $out"
        fails=$((fails+1))
    fi
}
check_empty() {
    local name="$1" file="$2"
    local out; out="$("$BRIDGE" "$file" "$SOUC")"
    if printf '%s' "$out" | grep -q '"diagnostics":\[\]'; then
        echo "[PASS] $name (no diagnostics)"
    else
        echo "[FAIL] $name: expected empty diagnostics, got: $out"
        fails=$((fails+1))
    fi
}

# Type error on line 5 → 0-based line 4.
cat >"$TMP/a.sio" <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn helper(x: i64) -> i64 { x * 2 }
fn main() -> i64 {
    let a = helper(10)
    let b = add(a, "wrong")
    b
}
EOF
check "type-error-line5" "$TMP/a.sio" 4

# Type error on line 3 → 0-based line 2.
cat >"$TMP/b.sio" <<'EOF'
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 {
    let r = add("hello", 5)
    r
}
EOF
check "type-error-line3" "$TMP/b.sio" 2

# Clean file → no diagnostics.
printf 'fn main() -> i64 { 0 }\n' >"$TMP/ok.sio"
check_empty "clean-file" "$TMP/ok.sio"

if (( fails > 0 )); then
    echo "diag_bridge self-test: $fails failure(s)"
    exit 1
fi
echo "diag_bridge self-test: all passed"
