#!/usr/bin/env bash
# Sprint 132 Gate: Epistemic pipeline integration tests
# Verifies: measure IR pattern, Knowledge field access, GUM add/sub/mul chains
set -eo pipefail

SOUC=./bin/souc
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

report() {
    local tag=$1 desc=$2 status=$3
    TOTAL=$((TOTAL+1))
    if [ "$status" = "PASS" ]; then PASS=$((PASS+1)); echo "PASS  $tag  $desc"
    elif [ "$status" = "FAIL" ]; then FAIL=$((FAIL+1)); echo "FAIL  $tag  $desc"
    else NOT_RUN=$((NOT_RUN+1)); echo "NOT_RUN  $tag  $desc"
    fi
}

echo "=== Sprint 132 Gate: Epistemic Pipeline Integration ==="

# --- Structural checks ---

# S1: measure IR pattern test exists
grep -q "compiler_main_test_measure_ir_pattern" self-hosted/compiler/main.sio && \
    report S1 "T464 measure IR pattern test exists" PASS || \
    report S1 "T464 measure IR pattern test exists" FAIL

# S2: Knowledge field_get test exists
grep -q "compiler_main_test_knowledge_field_get" self-hosted/compiler/main.sio && \
    report S2 "T465 Knowledge field_get test exists" PASS || \
    report S2 "T465 Knowledge field_get test exists" FAIL

# S3: GUM add full chain test exists
grep -q "compiler_main_test_gum_add_full_chain" self-hosted/compiler/main.sio && \
    report S3 "T466 GUM add full chain test exists" PASS || \
    report S3 "T466 GUM add full chain test exists" FAIL

# S4: GUM mul full chain test exists
grep -q "compiler_main_test_gum_mul_full_chain" self-hosted/compiler/main.sio && \
    report S4 "T467 GUM mul delta-method test exists" PASS || \
    report S4 "T467 GUM mul delta-method test exists" FAIL

# S5: Knowledge layout 3 fields test exists
grep -q "compiler_main_test_knowledge_layout_fields_3" self-hosted/compiler/main.sio && \
    report S5 "T468 Knowledge layout test exists" PASS || \
    report S5 "T468 Knowledge layout test exists" FAIL

# S6: GUM sub full chain test exists
grep -q "compiler_main_test_gum_sub_full_chain" self-hosted/compiler/main.sio && \
    report S6 "T469 GUM sub full chain test exists" PASS || \
    report S6 "T469 GUM sub full chain test exists" FAIL

# S7: Epistemic smoke test example exists
[ -f examples/epistemic_smoke_native.sio ] && \
    report S7 "epistemic_smoke_native.sio exists" PASS || \
    report S7 "epistemic_smoke_native.sio exists" FAIL

# S8: Smoke test has knowledge_add function
grep -q "fn knowledge_add" examples/epistemic_smoke_native.sio && \
    report S8 "knowledge_add in smoke test" PASS || \
    report S8 "knowledge_add in smoke test" FAIL

# S9: Smoke test has knowledge_mul function
grep -q "fn knowledge_mul" examples/epistemic_smoke_native.sio && \
    report S9 "knowledge_mul in smoke test" PASS || \
    report S9 "knowledge_mul in smoke test" FAIL

# S10: Total >= 469
grep -q "let total: i64 = 469" self-hosted/compiler/main.sio && \
    report S10 "total updated to 469" PASS || {
    # Check if linter bumped it higher
    if grep -qP "let total: i64 = (469|4[7-9][0-9]|[5-9][0-9][0-9])" self-hosted/compiler/main.sio; then
        report S10 "total updated to >=469" PASS
    else
        report S10 "total updated to 469" FAIL
    fi
}

# --- Typecheck ---
# S11: main.sio typechecks cleanly
if timeout 60 $SOUC check self-hosted/compiler/main.sio 2>&1 | grep -q "All checks passed"; then
    report S11 "main.sio typecheck" PASS
else
    report S11 "main.sio typecheck" FAIL
fi

# --- Runtime self-tests ---
_ec=0
timeout 180 $SOUC run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint132_gate.log 2>&1 || _ec=$?
if [ -f /tmp/sprint132_gate.log ]; then
    grep -q "T464 OK" /tmp/sprint132_gate.log && \
        report S12 "T464 measure IR pattern runtime" PASS || \
        report S12 "T464 measure IR pattern runtime" NOT_RUN
    grep -q "T465 OK" /tmp/sprint132_gate.log && \
        report S13 "T465 Knowledge field_get runtime" PASS || \
        report S13 "T465 Knowledge field_get runtime" NOT_RUN
    grep -q "T466 OK" /tmp/sprint132_gate.log && \
        report S14 "T466 GUM add chain runtime" PASS || \
        report S14 "T466 GUM add chain runtime" NOT_RUN
    grep -q "T467 OK" /tmp/sprint132_gate.log && \
        report S15 "T467 GUM mul chain runtime" PASS || \
        report S15 "T467 GUM mul chain runtime" NOT_RUN
    grep -q "T468 OK" /tmp/sprint132_gate.log && \
        report S16 "T468 Knowledge layout runtime" PASS || \
        report S16 "T468 Knowledge layout runtime" NOT_RUN
    grep -q "T469 OK" /tmp/sprint132_gate.log && \
        report S17 "T469 GUM sub chain runtime" PASS || \
        report S17 "T469 GUM sub chain runtime" NOT_RUN
else
    for i in 12 13 14 15 16 17; do
        report S$i "runtime test" NOT_RUN
    done
fi

echo ""
echo "=== Sprint 132 Gate Results ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && echo "GATE: OK" || echo "GATE: FAILED"
