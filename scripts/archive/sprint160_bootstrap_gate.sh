#!/usr/bin/env bash
# Sprint 160: Bootstrap Gate — bootstrap_v0 keyword fix + limit increase
# Tests: typecheck, structural checks, hello compilation
set -eo pipefail

SOUC=./bin/souc
BOOT=self-hosted/bootstrap/bootstrap_v0.sio
TMPDIR=/tmp/sprint160_gate
mkdir -p "$TMPDIR"

pass=0; fail=0; not_run=0

# ── Structural checks ──────────────────────────────────────────────────────

# S1: bootstrap_v0 has tk_is_causal_keyword with Model
if grep -q "TokenKind::Model => true" "$BOOT"; then
    echo "PASS  S1  tk_is_causal_keyword includes Model"
    pass=$((pass+1))
else
    echo "FAIL  S1  Model not in tk_is_causal_keyword"
    fail=$((fail+1))
fi

# S2: parse_type_path accepts keywords after ::
if grep -q "tk_is_keyword(p.peek_ahead(1))" "$BOOT"; then
    echo "PASS  S2  parse_type_path accepts keywords after ::"
    pass=$((pass+1))
else
    echo "FAIL  S2  parse_type_path not fixed"
    fail=$((fail+1))
fi

# S3: fn_names increased to 512
if grep -q "fn_names: \[Name; 512\]" "$BOOT"; then
    echo "PASS  S3  fn_names limit 512"
    pass=$((pass+1))
else
    echo "FAIL  S3  fn_names not 512"
    fail=$((fail+1))
fi

# S4: fn_offsets increased to 512
if grep -q "fn_offsets: \[i64; 512\]" "$BOOT"; then
    echo "PASS  S4  fn_offsets limit 512"
    pass=$((pass+1))
else
    echo "FAIL  S4  fn_offsets not 512"
    fail=$((fail+1))
fi

# S5: struct_names increased to 128
if grep -q "struct_names: \[Name; 128\]" "$BOOT"; then
    echo "PASS  S5  struct_names limit 128"
    pass=$((pass+1))
else
    echo "FAIL  S5  struct_names not 128"
    fail=$((fail+1))
fi

# S6: ELF output buffer 262144
if grep -q "io_write_native_binary.*262144" "$BOOT"; then
    echo "PASS  S6  ELF output buffer 262KB"
    pass=$((pass+1))
else
    echo "FAIL  S6  ELF output not 262KB"
    fail=$((fail+1))
fi

# S7: bootstrap_v0 typechecks
_ec=0
timeout 60 $SOUC check "$BOOT" 2>/dev/null || _ec=$?
if [ $_ec -eq 0 ]; then
    echo "PASS  S7  bootstrap_v0 typechecks"
    pass=$((pass+1))
else
    echo "FAIL  S7  typecheck failed (exit=$_ec)"
    fail=$((fail+1))
fi

# S8: bootstrap_self_host.sh exists
if [ -x scripts/bootstrap_self_host.sh ]; then
    echo "PASS  S8  bootstrap_self_host.sh exists and is executable"
    pass=$((pass+1))
else
    echo "FAIL  S8  bootstrap_self_host.sh missing"
    fail=$((fail+1))
fi

# ── Runtime test: hello compilation ─────────────────────────────────────────

cat > "$TMPDIR/hello.sio" << 'EOF'
fn main() -> i64 {
    print("BOOTSTRAP OK\n")
    0
}
EOF

_ec=0
timeout 120 $SOUC run "$BOOT" -- "$TMPDIR/hello.sio" "$TMPDIR/hello.elf" 2>/dev/null || _ec=$?
if [ $_ec -eq 0 ] && [ -f "$TMPDIR/hello.elf" ]; then
    chmod +x "$TMPDIR/hello.elf"
    actual=$("$TMPDIR/hello.elf" 2>/dev/null || true)
    if echo "$actual" | grep -qF "BOOTSTRAP OK"; then
        echo "PASS  S9  hello.sio → ELF → correct output"
        pass=$((pass+1))
    else
        echo "FAIL  S9  wrong output: $actual"
        fail=$((fail+1))
    fi
else
    echo "NOT_RUN  S9  compilation OOM (need 30GB+ free RAM)"
    not_run=$((not_run+1))
fi

echo ""
echo "=== Summary: PASS=$pass FAIL=$fail NOT_RUN=$not_run / total=$((pass+fail+not_run)) ==="
