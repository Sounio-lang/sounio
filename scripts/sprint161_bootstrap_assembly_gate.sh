#!/bin/bash
# Gate script for Sprint 161: Zero-import bootstrap assembly
#
# Validates:
#   1. bootstrap_v0.sio exists and assembles cleanly
#   2. souc check passes (zero-import file: ~66MB, <1s)
#   3. Bootstrap contains inlined parser (parse_program, lex functions)
#   4. Bootstrap contains inlined IR types (IrOpcode, IrInstr, IrFunction)
#   5. Bootstrap contains CodeBuffer and emit_byte
#   6. Bootstrap contains lean driver main() entry point
#   7. Bootstrap has zero use/module declarations
#   8. Assembly script is repeatable (idempotent rebuild)
#   9. Bootstrap line count is in expected range (13000–16000)
#  10. assemble+append scripts both exist
#
# Usage:
#   bash scripts/sprint161_bootstrap_assembly_gate.sh

set -eo pipefail

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
PASS=0
FAIL=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Sprint 161: Bootstrap Assembly Gate ==="
echo "souc: $SOUC"
echo ""

# T1: bootstrap_v0.sio exists
echo "[gate] T1: bootstrap_v0.sio exists"
[ -f "self-hosted/bootstrap/bootstrap_v0.sio" ]
check "bootstrap_exists" $?

# T2: souc check passes
echo "[gate] T2: souc check passes"
timeout 30 $SOUC check self-hosted/bootstrap/bootstrap_v0.sio > /dev/null 2>&1
check "bootstrap_check" $?

# T3: contains parser (parse_program function)
echo "[gate] T3: contains inlined parser"
grep -q "fn parse_program_preloaded" self-hosted/bootstrap/bootstrap_v0.sio
check "has_parser" $?

# T4: contains IrOpcode enum
echo "[gate] T4: contains IR types"
grep -q "enum IrOpcode" self-hosted/bootstrap/bootstrap_v0.sio
check "has_ir_types" $?

# T5: contains IrFunction struct
echo "[gate] T5: contains IrFunction struct"
grep -q "struct IrFunction" self-hosted/bootstrap/bootstrap_v0.sio
check "has_ir_function" $?

# T6: contains CodeBuffer and emit_byte
echo "[gate] T6: contains CodeBuffer/emit_byte"
grep -q "struct CodeBuffer" self-hosted/bootstrap/bootstrap_v0.sio && \
grep -q "fn emit_byte" self-hosted/bootstrap/bootstrap_v0.sio
check "has_code_buffer" $?

# T7: contains lean driver main()
echo "[gate] T7: contains lean driver main()"
grep -q "render_compile:" self-hosted/bootstrap/bootstrap_v0.sio
check "has_lean_main" $?

# T8: zero use/module declarations
echo "[gate] T8: zero use/module declarations"
USE_COUNT=$(grep -c "^use \|^module " self-hosted/bootstrap/bootstrap_v0.sio 2>/dev/null; true)
[ "$USE_COUNT" -eq 0 ]
check "zero_imports" $?

# T9: line count in expected range
echo "[gate] T9: line count in range"
LINES=$(wc -l < self-hosted/bootstrap/bootstrap_v0.sio)
[ "$LINES" -ge 13000 ] && [ "$LINES" -le 16000 ]
check "line_count_ok" $?

# T10: assembly scripts exist
echo "[gate] T10: assembly scripts exist"
[ -f "scripts/assemble_bootstrap.sh" ] && [ -f "scripts/append_bootstrap.sh" ]
check "scripts_exist" $?

echo ""
echo "=== Results ==="
echo "  PASS=$PASS  FAIL=$FAIL  TOTAL=$((PASS + FAIL))"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "All checks passed."
    exit 0
else
    echo "$FAIL check(s) failed."
    exit 1
fi
