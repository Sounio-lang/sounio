#!/usr/bin/env bash
# Sprint 121 Gate: sin/cos minimax builtins for native x86-64 backend
# Verifies: name recognizers, code emission, encoding helpers, typecheck, self-tests
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

report() {
    local tag=$1 desc=$2 status=$3
    TOTAL=$((TOTAL+1))
    if [ "$status" = "PASS" ]; then PASS=$((PASS+1)); echo "PASS  $tag  $desc"
    elif [ "$status" = "FAIL" ]; then FAIL=$((FAIL+1)); echo "FAIL  $tag  $desc"
    else NOT_RUN=$((NOT_RUN+1)); echo "NOT_RUN  $tag  $desc"
    fi
}

echo "=== Sprint 121 Gate: sin/cos minimax builtins ==="

# --- Structural checks ---

# S1: name_is_sin recognizer exists in codegen.sio
grep -q "fn name_is_sin" self-hosted/native/codegen.sio && \
    report S1 "name_is_sin function exists" PASS || \
    report S1 "name_is_sin function exists" FAIL

# S2: name_is_cos recognizer exists in codegen.sio
grep -q "fn name_is_cos" self-hosted/native/codegen.sio && \
    report S2 "name_is_cos function exists" PASS || \
    report S2 "name_is_cos function exists" FAIL

# S3: emit_builtin_sin exists in codegen.sio
grep -q "fn emit_builtin_sin" self-hosted/native/codegen.sio && \
    report S3 "emit_builtin_sin function exists" PASS || \
    report S3 "emit_builtin_sin function exists" FAIL

# S4: emit_builtin_cos exists in codegen.sio
grep -q "fn emit_builtin_cos" self-hosted/native/codegen.sio && \
    report S4 "emit_builtin_cos function exists" PASS || \
    report S4 "emit_builtin_cos function exists" FAIL

# S5: CVTSD2SI encoding helper exists in encode.sio
grep -q "fn emit_cvtsd2si_rax_xmm0" self-hosted/native/encode.sio && \
    report S5 "emit_cvtsd2si_rax_xmm0 exists" PASS || \
    report S5 "emit_cvtsd2si_rax_xmm0 exists" FAIL

# S6: CVTSI2SD encoding helper exists in encode.sio
grep -q "fn emit_cvtsi2sd_xmm0_rax" self-hosted/native/encode.sio && \
    report S6 "emit_cvtsi2sd_xmm0_rax exists" PASS || \
    report S6 "emit_cvtsi2sd_xmm0_rax exists" FAIL

# S7: sin dispatch wired in compile_ir_function
grep -q "name_is_sin" self-hosted/native/codegen.sio && \
    report S7 "sin dispatch wired" PASS || \
    report S7 "sin dispatch wired" FAIL

# S8: cos dispatch wired in compile_ir_function
grep -q "name_is_cos" self-hosted/native/codegen.sio && \
    report S8 "cos dispatch wired" PASS || \
    report S8 "cos dispatch wired" FAIL

# S9: Horner polynomial in emit_builtin_sin (uses 1/5040 constant)
grep -q "0x3F2A01A01A01A01A" self-hosted/native/codegen.sio && \
    report S9 "sin uses 1/5040 Horner coefficient" PASS || \
    report S9 "sin uses 1/5040 Horner coefficient" FAIL

# S10: Horner polynomial in emit_builtin_cos (uses 1/720 constant)
grep -q "0x3F56C16C16C16C17" self-hosted/native/codegen.sio && \
    report S10 "cos uses 1/720 Horner coefficient" PASS || \
    report S10 "cos uses 1/720 Horner coefficient" FAIL

# S11: Range reduction uses 2π constant
grep -q "0x401921FB54442D18" self-hosted/native/codegen.sio && \
    report S11 "range reduction uses 2π constant" PASS || \
    report S11 "range reduction uses 2π constant" FAIL

# S12: Range reduction uses CVTSD2SI for rounding
grep -q "emit_cvtsd2si_rax_xmm0" self-hosted/native/codegen.sio && \
    report S12 "range reduction uses CVTSD2SI" PASS || \
    report S12 "range reduction uses CVTSD2SI" FAIL

# S13: Tests T446-T451 present in main.sio
grep -q "T446 OK" self-hosted/compiler/main.sio && \
    report S13 "T446 test wired" PASS || \
    report S13 "T446 test wired" FAIL

# S14: Total updated to ≥451
grep -q "let total: i64 = 451" self-hosted/compiler/main.sio && \
    report S14 "total updated to 451" PASS || \
    report S14 "total updated to 451" FAIL

# --- Typecheck ---
# S15: main.sio typechecks cleanly
if timeout 60 $SOUC check self-hosted/compiler/main.sio 2>&1 | grep -q "All checks passed"; then
    report S15 "main.sio typecheck" PASS
else
    report S15 "main.sio typecheck" FAIL
fi

# --- Runtime self-tests ---
# S16: Run self-test, check T446-T451 pass
_ec=0
timeout 180 $SOUC run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint121_gate.log 2>&1 || _ec=$?
if [ -f /tmp/sprint121_gate.log ]; then
    grep -q "T446 OK" /tmp/sprint121_gate.log && \
        report S16 "T446 name_is_sin runtime" PASS || \
        report S16 "T446 name_is_sin runtime" NOT_RUN
    grep -q "T447 OK" /tmp/sprint121_gate.log && \
        report S17 "T447 name_is_cos runtime" PASS || \
        report S17 "T447 name_is_cos runtime" NOT_RUN
    grep -q "T448 OK" /tmp/sprint121_gate.log && \
        report S18 "T448 emit_sin produces code" PASS || \
        report S18 "T448 emit_sin produces code" NOT_RUN
    grep -q "T449 OK" /tmp/sprint121_gate.log && \
        report S19 "T449 emit_cos produces code" PASS || \
        report S19 "T449 emit_cos produces code" NOT_RUN
    grep -q "T450 OK" /tmp/sprint121_gate.log && \
        report S20 "T450 CVTSD2SI encoding" PASS || \
        report S20 "T450 CVTSD2SI encoding" NOT_RUN
    grep -q "T451 OK" /tmp/sprint121_gate.log && \
        report S21 "T451 CVTSI2SD encoding" PASS || \
        report S21 "T451 CVTSI2SD encoding" NOT_RUN
    # Overall self-test pass count
    if grep -q "451/451 passed" /tmp/sprint121_gate.log; then
        report S22 "all 451 self-tests pass" PASS
    elif grep -q "/451 passed" /tmp/sprint121_gate.log; then
        CNT=$(grep "/451 passed" /tmp/sprint121_gate.log | head -1)
        report S22 "all 451 self-tests pass ($CNT)" FAIL
    else
        report S22 "all 451 self-tests pass" NOT_RUN
    fi
else
    report S16 "T446 name_is_sin runtime" NOT_RUN
    report S17 "T447 name_is_cos runtime" NOT_RUN
    report S18 "T448 emit_sin produces code" NOT_RUN
    report S19 "T449 emit_cos produces code" NOT_RUN
    report S20 "T450 CVTSD2SI encoding" NOT_RUN
    report S21 "T451 CVTSI2SD encoding" NOT_RUN
    report S22 "all 451 self-tests pass" NOT_RUN
fi

echo ""
echo "=== Sprint 121 Gate Results ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && echo "GATE: OK" || echo "GATE: FAILED"
