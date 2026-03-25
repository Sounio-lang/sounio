#!/usr/bin/env bash
# Sprint 16 gate: H6.1 — user-defined effect declarations parsed and enforced
#
# Gate cases:
#   1.  'effect' keyword wired in parse_item() in items.sio
#   2.  parse_effect_item() function exists in items.sio
#   3.  ItemEffect variant in ItemKind enum in ast.sio
#   4.  EffectDef / EffectOpDef structs in ast.sio
#   5.  SymEffect / SymEffectOp in SymbolKind enum in symbol.sio
#   6.  EffectTable / EffectOpTable in defs.sio
#   7.  effects: EffectTable field in Checker struct in check.sio
#   8.  handler_depth field in Checker struct
#   9.  collect_item_effect method wired in collect_item dispatch
#  10.  extract_effects_with_checker method exists
#  11.  JIT probe: effects_unhandled.sio → error=type_check_failed
#  12.  JIT probe: effect_handler_missing_arm.sio → error=type_check_failed
#  13.  JIT probe: effect_resume_wrong_type.sio → error=type_check_failed
#  14.  JIT probe: effect_handler_basic.sio → ok
#  15.  SELFHOST_PASS count ≥ 24 (was 23 before sprint 16)
#  16.  Sprint 15 regression: BLOCKED tests ≤ 2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SELFHOST_DIR="$ROOT_DIR/self-hosted"
CF_DIR="$ROOT_DIR/tests/compile-fail"
JIT="${SOUNIO_JIT_BIN:-$ROOT_DIR/bin/souc}"

pass=0
fail=0
total=0

result() {
    total=$((total+1))
    if [ "$1" = "pass" ]; then
        pass=$((pass+1))
        echo "  [PASS] $2"
    else
        fail=$((fail+1))
        echo "  [FAIL] $2"
    fi
}

echo "Sprint 16: H6.1 — Effect Handler Semantics Gate"
echo "================================================"

# --- Structural checks ---

# 1. 'effect' keyword wired in parse_item dispatch
if grep -q "TokenKind::Effect" "$SELFHOST_DIR/parser/items.sio"; then
    result pass "TokenKind::Effect wired in parse_item"
else
    result fail "TokenKind::Effect NOT found in items.sio"
fi

# 2. parse_effect_item function exists
if grep -q "fn parse_effect_item" "$SELFHOST_DIR/parser/items.sio"; then
    result pass "parse_effect_item() exists in items.sio"
else
    result fail "parse_effect_item() missing from items.sio"
fi

# 3. ItemEffect in ItemKind enum
if grep -q "ItemEffect" "$SELFHOST_DIR/parser/ast.sio"; then
    result pass "ItemEffect in ast.sio"
else
    result fail "ItemEffect missing from ast.sio"
fi

# 4. EffectDef / EffectOpDef structs
if grep -q "struct EffectDef" "$SELFHOST_DIR/parser/ast.sio" && \
   grep -q "struct EffectOpDef" "$SELFHOST_DIR/parser/ast.sio"; then
    result pass "EffectDef + EffectOpDef structs in ast.sio"
else
    result fail "EffectDef/EffectOpDef missing from ast.sio"
fi

# 5. SymEffect / SymEffectOp in symbol.sio
if grep -q "SymEffect" "$SELFHOST_DIR/resolve/symbol.sio" && \
   grep -q "SymEffectOp" "$SELFHOST_DIR/resolve/symbol.sio"; then
    result pass "SymEffect + SymEffectOp in symbol.sio"
else
    result fail "SymEffect/SymEffectOp missing from symbol.sio"
fi

# 6. EffectTable / EffectOpTable in defs.sio
if grep -q "struct EffectTable" "$SELFHOST_DIR/check/defs.sio" && \
   grep -q "struct EffectOpTable" "$SELFHOST_DIR/check/defs.sio"; then
    result pass "EffectTable + EffectOpTable in defs.sio"
else
    result fail "EffectTable/EffectOpTable missing from defs.sio"
fi

# 7. effects: EffectTable field in Checker
if grep -q "effects: EffectTable" "$SELFHOST_DIR/check/check.sio"; then
    result pass "effects: EffectTable field in Checker struct"
else
    result fail "effects: EffectTable field missing from Checker"
fi

# 8. handler_depth field in Checker
if grep -q "handler_depth: i64" "$SELFHOST_DIR/check/check.sio"; then
    result pass "handler_depth: i64 field in Checker struct"
else
    result fail "handler_depth: i64 field missing from Checker"
fi

# 9. collect_item_effect wired in collect_item dispatch
if grep -q "collect_item_effect" "$SELFHOST_DIR/check/check.sio"; then
    result pass "collect_item_effect wired in check.sio"
else
    result fail "collect_item_effect missing from check.sio"
fi

# 10. extract_effects_with_checker method
if grep -q "fn extract_effects_with_checker" "$SELFHOST_DIR/check/check.sio"; then
    result pass "extract_effects_with_checker method in check.sio"
else
    result fail "extract_effects_with_checker missing from check.sio"
fi

# --- JIT probes ---

probe_expect_fail() {
    local label="$1"
    local file="$2"
    local out
    out=$(timeout 30 "$JIT" run self-hosted/compiler/main.sio -- --probe-frontend "$file" 2>&1 || true)
    if echo "$out" | grep -q "probe_frontend: error=type_check_failed"; then
        result pass "JIT probe: $label → error=type_check_failed"
    else
        result fail "JIT probe: $label → expected type_check_failed, got: $(echo "$out" | grep probe_frontend | tail -1)"
    fi
}

probe_expect_ok() {
    local label="$1"
    local file="$2"
    local out
    out=$(timeout 30 "$JIT" run self-hosted/compiler/main.sio -- --probe-frontend "$file" 2>&1 || true)
    if echo "$out" | grep -q "probe_frontend: ok"; then
        result pass "JIT probe: $label → ok"
    else
        result fail "JIT probe: $label → expected ok, got: $(echo "$out" | grep probe_frontend | tail -1)"
    fi
}

# 11. effects_unhandled.sio
probe_expect_fail "effects_unhandled.sio" "tests/compile-fail/effects_unhandled.sio"

# 12. effect_handler_missing_arm.sio
probe_expect_fail "effect_handler_missing_arm.sio" "tests/compile-fail/effect_handler_missing_arm.sio"

# 13. effect_resume_wrong_type.sio
probe_expect_fail "effect_resume_wrong_type.sio" "tests/compile-fail/effect_resume_wrong_type.sio"

# 14. effect_handler_basic.sio
probe_expect_ok "effect_handler_basic.sio" "tests/run-pass/effect_handler_basic.sio"

# --- Regression checks ---

# 15. SELFHOST_PASS count ≥ 24
sp_count=$(grep -rl 'SELFHOST_PASS' "$CF_DIR" 2>/dev/null | wc -l || true)
if [ "$sp_count" -ge 24 ]; then
    result pass "SELFHOST_PASS count = $sp_count (≥ 24)"
else
    result fail "SELFHOST_PASS count = $sp_count (< 24)"
fi

# 16. BLOCKED remaining ≤ 2
blocked_count=$(grep -rl 'BLOCKED:' "$CF_DIR" 2>/dev/null | wc -l || true)
if [ "$blocked_count" -le 2 ]; then
    result pass "BLOCKED count = $blocked_count (≤ 2)"
else
    result fail "BLOCKED count = $blocked_count (> 2)"
fi

# --- Summary ---
echo ""
echo "Results: $pass/$total passed"
echo ""

if [ "$fail" -eq 0 ]; then
    echo "status=pass reason=effect_handlers_h6_1 metrics: {total:$total, passed:$pass}"
    exit 0
else
    echo "status=fail reason=effect_handlers_h6_1 metrics: {total:$total, passed:$pass, failed:$fail}"
    exit 1
fi
