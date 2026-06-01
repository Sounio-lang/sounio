#!/usr/bin/env bash
# native_v2_e2e_codegen_suite_gate.sh
#
# Broadened end-to-end codegen-CORRECTNESS gate for the modular self-hosted
# compiler's native-v2 x86-64 backend. Where native_v2_e2e_exit_code_gate.sh
# proves a single scalar witness (main(){return 13} -> exit 13), this gate emits
# a SUITE of executables exercising distinct codegen shapes and RUNS each,
# asserting the process exit code matches the expected value:
#
#   scalar(N)  -> exit N           value fidelity across the byte range
#   call       -> exit 6           params + single call + arithmetic (3*2)
#   multicall  -> exit 84          chained calls + 2-param fns (add(10,32);mul(_,2))
#   control    -> exit 1           conditional branch TAKEN + label
#   control-ft -> exit 7           conditional branch NOT taken (fall-through)
#   arith      -> exit 42          OpSub + OpDiv ((100-16)/2)
#
# SCOPE: this hardens the IR -> machine code -> loaded ELF -> exit status path
# (the back-half, proven correct). It does NOT exercise parse -> check -> lower
# (the front-half), which remains blocked on the G1 *mut large-struct-move
# miscompile and is tracked as a separate workstream. "Suite PASS" means the
# backend codegen is correct across these shapes, NOT that source compiles.
#
# Exit code wraps mod 256, so scalar witnesses stay in 0..255.
#
# The modular compiler binary is taken from $SOUNIO_MODULAR_SOUC if set (a prebuilt
# main.sio ELF), otherwise built from self-hosted/compiler/main.sio using the
# bootstrap compiler $SOUNIO_BOOTSTRAP_SOUC (default ./bin/souc).
#
# Exit 0 = all witnesses verified; non-zero = a mismatch (codegen regression).
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ulimit -s 1048576 2>/dev/null || true

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
MODULAR_SOUC="${SOUNIO_MODULAR_SOUC:-}"

if [[ -z "$MODULAR_SOUC" ]]; then
    echo "[gate] building modular compiler from self-hosted/compiler/main.sio (bootstrap=$BOOTSTRAP_SOUC)"
    MODULAR_SOUC="$WORK/mc.elf"
    if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/main.sio "$MODULAR_SOUC" > "$WORK/build.log" 2>&1; then
        echo "[gate] FAIL: could not build modular compiler"
        tail -5 "$WORK/build.log"
        exit 2
    fi
    chmod +x "$MODULAR_SOUC"
fi

if [[ ! -x "$MODULAR_SOUC" ]]; then
    echo "[gate] FAIL: modular compiler not executable: $MODULAR_SOUC"
    exit 2
fi

FAILED=0

# emit_and_check <expected_exit> <emit-args...>
# Emits to a fresh ELF in $WORK, verifies ELF magic, runs it, asserts exit code.
emit_and_check() {
    local expected="$1"; shift
    local label="$1"; shift
    local elf="$WORK/witness_${label}_$$.elf"
    rm -f "$elf"

    # Build the emit invocation: the LAST arg is always the out_path we inject.
    "$MODULAR_SOUC" "$@" "$elf" > "$WORK/emit_${label}.log" 2>&1 || true

    if [[ ! -f "$elf" ]]; then
        echo "[gate] FAIL[$label]: compiler did not emit $elf"
        grep -a 'native_v2_emit_exit_code' "$WORK/emit_${label}.log" | tr -cd '[:print:]\n' || true
        FAILED=1
        return
    fi

    local magic
    magic="$(head -c4 "$elf" | od -An -tx1 | tr -d ' ')"
    if [[ "$magic" != "7f454c46" ]]; then
        echo "[gate] FAIL[$label]: emitted file is not ELF (magic=$magic)"
        FAILED=1
        return
    fi

    chmod +x "$elf"
    "$elf"
    local got=$?
    if [[ "$got" -ne "$expected" ]]; then
        echo "[gate] FAIL[$label]: exit code=$got, expected $expected"
        FAILED=1
        return
    fi
    echo "[gate] PASS[$label]: exit $got (expected $expected)"
}

echo "[gate] native-v2 codegen suite (IR -> ELF -> exit); front-half/G1 out of scope"

# Scalar value fidelity across the byte range.
emit_and_check 1   "scalar1"   --native-v2-emit-scalar 1
emit_and_check 42  "scalar42"  --native-v2-emit-scalar 42
emit_and_check 200 "scalar200" --native-v2-emit-scalar 200
emit_and_check 255 "scalar255" --native-v2-emit-scalar 255

# Shapes.
emit_and_check 6   "call"       --native-v2-emit-call
emit_and_check 84  "multicall"  --native-v2-emit-multicall
emit_and_check 1   "control"    --native-v2-emit-control
emit_and_check 7   "control-ft" --native-v2-emit-control-ft
emit_and_check 42  "arith"      --native-v2-emit-arith

if [[ "$FAILED" -ne 0 ]]; then
    echo "[gate] FAIL: one or more native-v2 codegen witnesses regressed"
    exit 5
fi

echo "[gate] PASS: modular native-v2 backend emits correct executables across scalar/call/multicall/control/control-ft/arith (IR->ELF->exit)"
exit 0
