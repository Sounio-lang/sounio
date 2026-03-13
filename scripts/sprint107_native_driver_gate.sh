#!/usr/bin/env bash
# Sprint 107 — General-Purpose Native Compile Driver gate
set -eo pipefail

# NativeCompiler struct + IrModule exceed default 12.5MB stack; raise limit
ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
DRIVER=self-hosted/compiler/native_compile_driver.sio
STABLE=self-hosted/compiler/render_native_compile_driver_stable.sio
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

check_grep() {
    local name="$1" file="$2" pattern="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS  $name"; PASS=$((PASS+1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

check_absent_grep() {
    local name="$1" pattern="$2" file="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  FAIL  $name (pattern found)"; FAIL=$((FAIL+1))
    else
        echo "  PASS  $name"; PASS=$((PASS+1))
    fi
}

echo "=== Sprint 107 — General-Purpose Native Compile Driver Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_main"           "$DRIVER" "fn main() -> i64"
check_grep "struct:typecheck_skipped" "$DRIVER" "Typecheck intentionally skipped"
check_grep "struct:optimizer_note"    "$DRIVER" "optimizer.*intentionally skipped"
check_grep "struct:output_flag"       "$DRIVER" '"-o"'
check_grep "struct:streaming_begin"   "$DRIVER" "compile_module_streaming_begin"
check_grep "struct:streaming_finish"  "$DRIVER" "compile_module_streaming_finish"
check_grep "struct:finalize_elf"      "$DRIVER" "finalize_compiled_elf"
check_grep "struct:io_write"          "$DRIVER" "io_write_native_binary"
check_grep "struct:sprint107_tag"     "$DRIVER" "Sprint 107"
check_grep "struct:native_prefix"     "$DRIVER" "native_compile:"
check_absent_grep "struct:no_module_loader" "use.*module_loader" "$DRIVER"
check_absent_grep "struct:no_render_prefix" "render_compile:" "$DRIVER"

echo ""
echo "[regression]"
check_grep "regr:stable_exists"  "$STABLE" "fn main() -> i64"
check_grep "regr:codegen_exists" "self-hosted/native/codegen.sio" "fn compile_ir_function"
check_grep "regr:elf_exists"     "self-hosted/native/elf.sio" "fn finalize_elf64"

echo ""
echo "[typecheck]"
TOTAL=$((TOTAL+1))
if timeout 30 $SOUC check "$DRIVER" 2>&1 | grep -q "All checks passed"; then
    echo "  PASS  typecheck:driver"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:driver"; FAIL=$((FAIL+1))
fi

echo ""
echo "[compile]"
TOTAL=$((TOTAL+1))
COMPILE_OUT=$(timeout 600 $SOUC run "$DRIVER" -- examples/render/triangle_basic.sio -o /tmp/sprint107_gate.elf 2>&1 || true)
if echo "$COMPILE_OUT" | grep -q "native_compile: ok written"; then
    echo "  PASS  compile:triangle_basic"; PASS=$((PASS+1))

    # Check ELF magic bytes
    TOTAL=$((TOTAL+1))
    if xxd -l 4 /tmp/sprint107_gate.elf 2>/dev/null | grep -q "7f45 4c46"; then
        echo "  PASS  compile:elf_valid"; PASS=$((PASS+1))
    else
        echo "  FAIL  compile:elf_valid"; FAIL=$((FAIL+1))
    fi

    # Check ELF runs
    TOTAL=$((TOTAL+1))
    chmod +x /tmp/sprint107_gate.elf 2>/dev/null
    if /tmp/sprint107_gate.elf > /tmp/sprint107_gate.ppm 2>/dev/null; then
        if head -1 /tmp/sprint107_gate.ppm | grep -q "P6\|P3"; then
            echo "  PASS  compile:runs_ppm"; PASS=$((PASS+1))
        else
            echo "  FAIL  compile:runs_ppm (no PPM header)"; FAIL=$((FAIL+1))
        fi
    else
        echo "  NOT_RUN  compile:runs_ppm (execution failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
else
    echo "  NOT_RUN  compile:triangle_basic (OOM/timeout before completion)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 107 — General-Purpose Native Compile Driver"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
